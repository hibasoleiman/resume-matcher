"""
Final stage of the pipeline: take the in-memory results and write them to
disk as machine-readable JSON and a human-readable markdown report.

No LLM calls here — this is a pure transform from Pydantic models to text.
That's why test_reporter.py can run in milliseconds with no API key.

Two outputs:
  - results.json: {"candidates": [...sorted by overall_fit desc...],
                   "errors":     [...resumes that failed at any stage...]}
  - report.md:    a ranked table plus per-candidate breakdowns and gaps.
"""

import json
import csv
from pathlib import Path

from src.bias_auditor import VARIANT_DISPLAY_LABELS
from src.models import BiasAuditReport, ProcessingError, ScoredCandidate

Result = ScoredCandidate | ProcessingError


def _split(results: list[Result]) -> tuple[list[ScoredCandidate], list[ProcessingError]]:
    candidates = [r for r in results if isinstance(r, ScoredCandidate)]
    errors = [r for r in results if isinstance(r, ProcessingError)]
    candidates.sort(key=lambda c: c.overall_fit.score, reverse=True)
    return candidates, errors


def write_json(results: list[Result], path: str | Path) -> None:
    """Write {candidates, errors} to a JSON file. Candidates sorted by overall_fit desc."""

    candidates, errors = _split(results)
    payload = {
        "candidates": [c.model_dump() for c in candidates],
        "errors": [e.model_dump() for e in errors],
    }
    Path(path).write_text(json.dumps(payload, indent=2))
def write_csv(results:list[Result],path:str |Path)-> None:
    fieldnames=["rank","name","overall_fit","skills_match","experience_match","role_relevance","years_experience","gaps","source_file"]
    candidates, errors=_split(results)
    with open(path,"w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rank,c in enumerate(candidates, start=1):
            writer.writerow({
                "rank":rank,
                "name":c.profile.name,
                "overall_fit":c.overall_fit.score,
                "skills_match":c.skills_match.score,
                "experience_match":c.experience_match.score,
                "role_relevance":c.role_relevance.score,
                "years_experience":c.profile.years_experience,
                "gaps":";".join(f"{g.category}:{g.detail}" for g in c.gaps),
                "source_file":c.source_file
                })


def find_candidates(json_path: str | Path, query: str) -> list[dict]:
    """Search an existing results.json for candidates matching `query`.

    Case-insensitive substring match against the candidate's name or their
    source PDF filename. Returns a list so callers can tell 0 / 1 / many
    matches apart — no LLM calls, just reads the file write_json() produced.
    """
    data = json.loads(Path(json_path).read_text())
    q = query.lower()
    return [
        c for c in data["candidates"]
        if q in c["profile"]["name"].lower() or q in c["source_file"].lower()
    ]




def _bias_drift_table(audit: BiasAuditReport) -> list[str]:
    """Render the bias audit as a markdown drift table for one candidate."""
    dims = ("skills_match", "experience_match", "role_relevance", "overall_fit")
    headers = ("Skills", "Experience", "Role", "Overall", "Max Δ")

    lines: list[str] = []
    flag_label = "⚠ FLAGGED" if audit.flagged else "✓ No significant drift"
    lines.append(
        f"**Bias Audit** — largest score drift across demographic signal swaps: "
        f"**{audit.max_score_drift:.0f} pts** · {flag_label}"
    )
    lines.append("")
    lines.append(f"_Threshold: {audit.drift_threshold} pts. "
                 f"Bold cells exceeded the threshold._")
    lines.append("")
    lines.append(f"| Signal swapped | {' | '.join(headers)} |")
    lines.append(f"| --- | {' | '.join(['---'] * len(headers))} |")

    base_scores = {d: getattr(audit.baseline, d).score for d in dims}
    base_cells = " | ".join(str(base_scores[d]) for d in dims)
    lines.append(f"| Baseline (original resume) | {base_cells} | 0 |")

    for v in audit.variants:
        display_label = VARIANT_DISPLAY_LABELS.get(v.label, v.label)
        cells: list[str] = []
        max_delta = 0
        for d in dims:
            base = base_scores[d]
            var = getattr(v.scores, d).score
            delta = var - base
            max_delta = max(max_delta, abs(delta))
            cell = str(var)
            if abs(delta) > audit.drift_threshold:
                cell = f"**{var}**"
            cells.append(cell)
        lines.append(f"| {display_label} | {' | '.join(cells)} | {max_delta} |")

    if audit.flag_reason:
        lines.append("")
        lines.append(f"> **Finding:** {audit.flag_reason}")

    return lines


def write_markdown(
    results: list[Result],
    path: str | Path,
    jd_path: str | Path,
    audits: dict[str, BiasAuditReport] | None = None,
) -> None:
    """Write a markdown report: ranked table + per-candidate breakdowns + errors."""

    candidates, errors = _split(results)

    lines: list[str] = []
    lines.append("# Resume Match Report")
    lines.append("")
    lines.append(f"Job description: `{jd_path}`")
    lines.append(f"Candidates scored: {len(candidates)}  ·  Failed: {len(errors)}")
    lines.append("")

    lines.append("## Ranked candidates")
    lines.append("")
    if candidates:
        lines.append("| Rank | Name | Overall | Skills | Experience | Role | Source |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- |")
        for i, c in enumerate(candidates, start=1):
            lines.append(
                f"| {i} "
                f"| {c.profile.name} "
                f"| {c.overall_fit.score} "
                f"| {c.skills_match.score} "
                f"| {c.experience_match.score} "
                f"| {c.role_relevance.score} "
                f"| {c.source_file} |"
            )
    else:
        lines.append("_No candidates scored._")
    lines.append("")

    for c in candidates:
        lines.append(f"### {c.profile.name} — {c.overall_fit.score}")
        lines.append("")
        lines.append(f"_Source: `{c.source_file}` · {c.profile.years_experience} yrs experience_")
        lines.append("")
        lines.append(c.reasoning)
        lines.append("")
        lines.append("**Score breakdown**")
        lines.append("")
        lines.append(f"- Skills ({c.skills_match.score}): {c.skills_match.reasoning}")
        lines.append(f"- Experience ({c.experience_match.score}): {c.experience_match.reasoning}")
        lines.append(f"- Role relevance ({c.role_relevance.score}): {c.role_relevance.reasoning}")
        lines.append(f"- Overall fit ({c.overall_fit.score}): {c.overall_fit.reasoning}")
        lines.append("")
        if c.gaps:
            lines.append("**Gaps**")
            lines.append("")
            for gap in c.gaps:
                lines.append(f"- _{gap.category}_ — {gap.detail}")
            lines.append("")

        if audits and c.source_file in audits:
            lines.extend(_bias_drift_table(audits[c.source_file]))
            lines.append("")

    if errors:
        lines.append("## Could not process")
        lines.append("")
        lines.append("| Source | Stage | Message |")
        lines.append("| --- | --- | --- |")
        for e in errors:
            # Replace pipes in the message so they don't break the table layout.
            msg = e.message.replace("|", "\\|").replace("\n", " ")
            lines.append(f"| {e.source_file} | {e.stage} | {msg} |")
        lines.append("")

    Path(path).write_text("\n".join(lines))
