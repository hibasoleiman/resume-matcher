import os
import tempfile

import streamlit as st
from dotenv import load_dotenv
from anthropic import Anthropic

from src.bias_auditor import VARIANT_DISPLAY_LABELS, run_bias_audit
from src.extractor import DEFAULT_MODEL, extract_candidate
from src.models import BiasAuditReport, ProcessingError, ScoredCandidate
from src.pdf_parser import parse_pdf
from src.scorer import score_candidate

load_dotenv()

Result = ScoredCandidate | ProcessingError


def process_one(
    client, pdf_path, filename, jd_text, model, bias_audit
) -> tuple[Result, BiasAuditReport | None]:
    """Run one resume through parse -> extract -> score (-> bias audit)."""

    # 1. Parse the PDF
    try:
        resume_text = parse_pdf(pdf_path)
    except Exception as e:
        return ProcessingError(source_file=filename, stage="parse", message=str(e)), None

    # 2. Extract candidate info
    try:
        profile = extract_candidate(client, resume_text, model=model)
    except Exception as e:
        return ProcessingError(source_file=filename, stage="extract", message=str(e)), None

    # 3. Score against the JD
    try:
        scored = score_candidate(client, profile, jd_text, filename, model=model)
    except Exception as e:
        return ProcessingError(source_file=filename, stage="score", message=str(e)), None

    # 4. Optional bias audit
    audit: BiasAuditReport | None = None
    if bias_audit:
        try:
            audit = run_bias_audit(client, profile, jd_text, model=model)
        except Exception:
            pass  # scoring result is still valid if audit fails

    return scored, audit


# ── Page config ──
st.set_page_config(page_title="Resume Matcher", page_icon="🎯", layout="wide")

st.markdown(
    """
    <style>
    div.stButton > button:first-child {
        background: linear-gradient(90deg, #6C5CE7, #00B894);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.6rem 1.4rem;
        font-weight: 600;
        box-shadow: 0 4px 14px rgba(108, 92, 231, 0.35);
        transition: all 0.15s ease-in-out;
    }
    div.stButton > button:first-child:hover {
        box-shadow: 0 6px 20px rgba(108, 92, 231, 0.45);
        transform: translateY(-1px);
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🎯 Resume Matcher")
st.caption("Upload a job description and resumes — Claude parses, extracts, and scores every candidate for fit.")

# ── Sidebar: settings ──
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    model = st.text_input("Claude model", value=DEFAULT_MODEL)
    bias_audit = st.checkbox(
        "Enable bias audit (5 extra LLM calls per resume)",
        help=(
            "Re-scores each candidate with demographic signals swapped "
            "(name, graduation year) and reports score drift per dimension."
        ),
    )
    st.divider()
    with st.expander("ℹ️ How this works"):
        st.markdown(
            "Each resume goes through:\n"
            "1. **Parse** — extract text from the PDF\n"
            "2. **Extract** — Claude pulls structured candidate info\n"
            "3. **Score** — Claude rates fit against the job description\n"
            "4. **Bias audit** *(optional)* — re-scores with demographic details swapped to check for drift"
        )

# ── File uploads ──
col1, col2 = st.columns(2)

with col1:
    with st.container(border=True):
        st.subheader("📋 Job Description")
        jd_file = st.file_uploader("Upload JD (TXT)", type=["txt"])
        st.caption("Plain-text job description to score every resume against.")

with col2:
    with st.container(border=True):
        st.subheader("📄 Resumes")
        resume_files = st.file_uploader(
            "Upload resumes (PDF)", type=["pdf"], accept_multiple_files=True
        )
        st.caption("One or more resume PDFs — each is scored independently.")

# ── Run button ──
if st.button("Match Resumes", type="primary"):
    # Validate inputs
    if not os.getenv("ANTHROPIC_API_KEY"):
        st.error("ANTHROPIC_API_KEY is not set. Add it to your .env file.")
        st.stop()

    if not jd_file:
        st.error("Please upload a job description PDF.")
        st.stop()

    if not resume_files:
        st.error("Please upload at least one resume PDF.")
        st.stop()

    # Read the JD text file
    jd_text = jd_file.read().decode("utf-8")

    # Show the parsed JD
    with st.expander("Parsed Job Description"):
        st.text(jd_text)

    # Process each resume
    client = Anthropic()
    pairs: list[tuple[Result, BiasAuditReport | None]] = []

    with st.status(f"Processing {len(resume_files)} resume(s)...", expanded=True) as status:
        progress = st.progress(0)
        for i, resume_file in enumerate(resume_files):
            st.write(f"🔎 {resume_file.name} — parsing, extracting, scoring...")

            # Save uploaded PDF to a temp file so pdfplumber can read it
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp.write(resume_file.read())
                tmp_path = tmp.name

            result, audit = process_one(
                client, tmp_path, resume_file.name, jd_text, model, bias_audit
            )
            os.unlink(tmp_path)
            pairs.append((result, audit))

            if isinstance(result, ScoredCandidate):
                st.write(f"✅ {resume_file.name} scored **{result.overall_fit.score}/100**")
            else:
                st.write(f"⚠️ {resume_file.name} failed at `{result.stage}`")

            progress.progress((i + 1) / len(resume_files))

        status.update(label="Done!", state="complete", expanded=False)

    # ── Split results into scored candidates and errors ──
    scored_pairs = [(r, a) for r, a in pairs if isinstance(r, ScoredCandidate)]
    errors = [r for r, _ in pairs if isinstance(r, ProcessingError)]

    # Sort by overall score, highest first
    scored_pairs.sort(key=lambda x: x[0].overall_fit.score, reverse=True)
    scored = [r for r, _ in scored_pairs]

    # ── Display ranked candidates ──
    if scored:
        st.divider()
        st.subheader("🏆 Ranked Candidates")

        overall_scores = [c.overall_fit.score for c in scored]
        flagged_count = sum(1 for _, a in scored_pairs if a and a.flagged)

        metric_cols = st.columns(4 if bias_audit else 3)
        metric_cols[0].metric("Resumes scored", len(scored))
        metric_cols[1].metric("Top match", scored[0].profile.name, f"{scored[0].overall_fit.score}/100")
        metric_cols[2].metric("Average score", f"{sum(overall_scores) / len(overall_scores):.0f}/100")
        if bias_audit:
            metric_cols[3].metric("Bias-flagged", flagged_count, delta_color="inverse")

        # Summary table
        _MEDALS = {1: "🥇", 2: "🥈", 3: "🥉"}
        table_data = []
        for rank, c in enumerate(scored, 1):
            table_data.append({
                "Rank": f"{_MEDALS.get(rank, '')} {rank}".strip(),
                "Candidate": c.profile.name,
                "Overall": c.overall_fit.score,
                "Skills": c.skills_match.score,
                "Experience": c.experience_match.score,
                "Role Relevance": c.role_relevance.score,
            })
        st.dataframe(
            table_data,
            hide_index=True,
            use_container_width=True,
            column_config={
                col: st.column_config.ProgressColumn(col, min_value=0, max_value=100, format="%d")
                for col in ("Overall", "Skills", "Experience", "Role Relevance")
            },
        )

        # Detailed view for each candidate
        for rank, (c, audit) in enumerate(scored_pairs, 1):
            medal = _MEDALS.get(rank, "")
            with st.expander(f"{medal} {c.profile.name} — {c.overall_fit.score}/100".strip()):
                st.caption(f"Source: {c.source_file}  ·  {c.profile.years_experience} yrs experience")
                st.write(f"**Skills:** {', '.join(c.profile.skills)}")
                st.info(c.reasoning)

                score_cols = st.columns(4)
                score_cols[0].metric("Skills", c.skills_match.score)
                score_cols[1].metric("Experience", c.experience_match.score)
                score_cols[2].metric("Role fit", c.role_relevance.score)
                score_cols[3].metric("Overall", c.overall_fit.score)

                with st.expander("Score reasoning"):
                    st.write(f"- **Skills match** — {c.skills_match.reasoning}")
                    st.write(f"- **Experience match** — {c.experience_match.reasoning}")
                    st.write(f"- **Role relevance** — {c.role_relevance.reasoning}")
                    st.write(f"- **Overall fit** — {c.overall_fit.reasoning}")

                if c.gaps:
                    st.write("**Gaps:**")
                    for gap in c.gaps:
                        st.write(f"- _{gap.category}_: {gap.detail}")

                if audit:
                    st.write("---")
                    st.write("**Bias Audit**")
                    st.caption(
                        f"Each row re-scores the same candidate with one demographic "
                        f"signal swapped. Threshold: {audit.drift_threshold} pts."
                    )

                    if audit.flagged:
                        st.warning(
                            f"Score drift detected — largest shift: "
                            f"{audit.max_score_drift:.0f} pts. "
                            + (audit.flag_reason or "")
                        )
                    else:
                        st.success(
                            f"Stable — max drift {audit.max_score_drift:.0f} pts "
                            f"(within {audit.drift_threshold} pt threshold)"
                        )

                    _DIMS = [
                        ("skills_match",     "Skills"),
                        ("experience_match", "Experience"),
                        ("role_relevance",   "Role"),
                        ("overall_fit",      "Overall"),
                    ]
                    base = {k: getattr(audit.baseline, k).score for k, _ in _DIMS}
                    rows = [
                        {
                            "Signal swapped": "Baseline (original resume)",
                            **{label: base[k] for k, label in _DIMS},
                            "Max Δ": 0,
                        }
                    ]
                    for v in audit.variants:
                        var_scores = {k: getattr(v.scores, k).score for k, _ in _DIMS}
                        max_delta = max(abs(var_scores[k] - base[k]) for k, _ in _DIMS)
                        rows.append({
                            "Signal swapped": VARIANT_DISPLAY_LABELS.get(v.label, v.label),
                            **{label: var_scores[k] for k, label in _DIMS},
                            "Max Δ": max_delta,
                        })
                    st.dataframe(
                        rows,
                        hide_index=True,
                        use_container_width=True,
                        column_config={
                            **{
                                label: st.column_config.ProgressColumn(
                                    label, min_value=0, max_value=100, format="%d"
                                )
                                for _, label in _DIMS
                            },
                            "Max Δ": st.column_config.NumberColumn(
                                help="Largest score change vs baseline across all dimensions"
                            ),
                        },
                    )

    # ── Display errors ──
    if errors:
        st.divider()
        st.subheader("⚠️ Errors")
        for e in errors:
            st.error(f"**{e.source_file}** failed at `{e.stage}`: {e.message}")
