"""Tests for reporter.py — pure transform, no API calls."""

import csv
import json
from pathlib import Path

from src.models import (
    CandidateProfile,
    DimensionScore,
    Gap,
    ProcessingError,
    Role,
    ScoredCandidate,
)
from src.reporter import find_candidates, write_csv, write_json, write_markdown


def _make_scored(name: str, overall: int, source: str) -> ScoredCandidate:
    return ScoredCandidate(
        profile=CandidateProfile(
            name=name,
            years_experience=5.0,
            skills=["Python", "SQL"],
            past_roles=[
                Role(
                    title="Backend Engineer",
                    company="Acme",
                    duration_months=24,
                    description="Worked on APIs.",
                )
            ],
            education=["BS Computer Science"],
            raw_summary="I build backend systems.",
        ),
        skills_match=DimensionScore(score=overall - 5, reasoning="Skills reason"),
        experience_match=DimensionScore(score=overall, reasoning="Experience reason"),
        role_relevance=DimensionScore(score=overall + 2, reasoning="Role reason"),
        overall_fit=DimensionScore(score=overall, reasoning="Overall reason"),
        reasoning="Two-sentence reasoning. Looks strong.",
        gaps=[Gap(category="skill", detail="No Kafka experience")],
        source_file=source,
    )


def test_json_output_sorts_by_overall_fit(tmp_path: Path) -> None:
    results = [
        _make_scored("Bob", 70, "bob.pdf"),
        _make_scored("Alice", 90, "alice.pdf"),
        _make_scored("Carol", 80, "carol.pdf"),
    ]

    out = tmp_path / "results.json"
    write_json(results, out)

    data = json.loads(out.read_text())
    names = [c["profile"]["name"] for c in data["candidates"]]
    assert names == ["Alice", "Carol", "Bob"]
    assert data["errors"] == []


def test_json_separates_errors_from_candidates(tmp_path: Path) -> None:
    results = [
        _make_scored("Alice", 90, "alice.pdf"),
        ProcessingError(
            source_file="broken.pdf", stage="parse", message="PDF corrupt"
        ),
    ]

    out = tmp_path / "results.json"
    write_json(results, out)

    data = json.loads(out.read_text())
    assert len(data["candidates"]) == 1
    assert data["candidates"][0]["profile"]["name"] == "Alice"
    assert len(data["errors"]) == 1
    assert data["errors"][0]["stage"] == "parse"


def test_markdown_contains_ranked_table_and_sections(tmp_path: Path) -> None:
    results = [
        _make_scored("Bob", 70, "bob.pdf"),
        _make_scored("Alice", 90, "alice.pdf"),
    ]

    out = tmp_path / "report.md"
    write_markdown(results, out, Path("jd.txt"))
    content = out.read_text()

    assert "# Resume Match Report" in content
    assert "## Ranked candidates" in content
    assert "Alice" in content
    assert "Bob" in content
    # Alice should appear before Bob in the ranked table
    assert content.index("Alice") < content.index("Bob")


def test_markdown_includes_error_section_when_errors_present(tmp_path: Path) -> None:
    results = [
        _make_scored("Alice", 90, "alice.pdf"),
        ProcessingError(
            source_file="broken.pdf", stage="extract", message="API timeout"
        ),
    ]

    out = tmp_path / "report.md"
    write_markdown(results, out, Path("jd.txt"))
    content = out.read_text()

    assert "## Could not process" in content
    assert "broken.pdf" in content
    assert "extract" in content
    assert "API timeout" in content


def test_markdown_omits_error_section_when_no_errors(tmp_path: Path) -> None:
    results = [_make_scored("Alice", 90, "alice.pdf")]

    out = tmp_path / "report.md"
    write_markdown(results, out, Path("jd.txt"))
    content = out.read_text()

    assert "## Could not process" not in content


def test_csv_output_sorts_by_overall_fit(tmp_path: Path) -> None:
    results = [
        _make_scored("Bob", 70, "bob.pdf"),
        _make_scored("Alice", 90, "alice.pdf"),
        _make_scored("Carol", 80, "carol.pdf"),
    ]

    out = tmp_path / "results.csv"
    write_csv(results, out)

    with out.open(newline="") as f:
        rows = list(csv.DictReader(f))

    assert [r["name"] for r in rows] == ["Alice", "Carol", "Bob"]
    assert [r["rank"] for r in rows] == ["1", "2", "3"]


def test_csv_has_expected_columns(tmp_path: Path) -> None:
    results = [_make_scored("Alice", 90, "alice.pdf")]

    out = tmp_path / "results.csv"
    write_csv(results, out)

    with out.open(newline="") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == [
            "rank",
            "name",
            "overall_fit",
            "skills_match",
            "experience_match",
            "role_relevance",
            "years_experience",
            "gaps",
            "source_file",
        ]


def test_csv_excludes_errors(tmp_path: Path) -> None:
    results = [
        _make_scored("Alice", 90, "alice.pdf"),
        ProcessingError(source_file="broken.pdf", stage="parse", message="PDF corrupt"),
    ]

    out = tmp_path / "results.csv"
    write_csv(results, out)

    with out.open(newline="") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 1
    assert rows[0]["name"] == "Alice"


def test_csv_gaps_joined_as_string(tmp_path: Path) -> None:
    candidate = _make_scored("Alice", 90, "alice.pdf")
    candidate.gaps = [
        Gap(category="skill", detail="No Kafka experience"),
        Gap(category="experience", detail="Never led a team"),
    ]

    out = tmp_path / "results.csv"
    write_csv([candidate], out)

    with out.open(newline="") as f:
        rows = list(csv.DictReader(f))

    assert rows[0]["gaps"] == "skill:No Kafka experience;experience:Never led a team"


def test_find_candidates_matches_by_name(tmp_path: Path) -> None:
    results = [
        _make_scored("Alice Chen", 90, "alice.pdf"),
        _make_scored("Bob Smith", 70, "bob.pdf"),
    ]
    out = tmp_path / "results.json"
    write_json(results, out)

    matches = find_candidates(out, "alice")
    assert len(matches) == 1
    assert matches[0]["profile"]["name"] == "Alice Chen"


def test_find_candidates_matches_by_source_file(tmp_path: Path) -> None:
    results = [_make_scored("Alice Chen", 90, "alice_chen_resume.pdf")]
    out = tmp_path / "results.json"
    write_json(results, out)

    matches = find_candidates(out, "alice_chen_resume")
    assert len(matches) == 1
    assert matches[0]["source_file"] == "alice_chen_resume.pdf"


def test_find_candidates_is_case_insensitive(tmp_path: Path) -> None:
    results = [_make_scored("Alice Chen", 90, "alice.pdf")]
    out = tmp_path / "results.json"
    write_json(results, out)

    assert len(find_candidates(out, "ALICE")) == 1
    assert len(find_candidates(out, "AlIcE")) == 1


def test_find_candidates_no_match_returns_empty_list(tmp_path: Path) -> None:
    results = [_make_scored("Alice Chen", 90, "alice.pdf")]
    out = tmp_path / "results.json"
    write_json(results, out)

    assert find_candidates(out, "zzz-nobody") == []


def test_find_candidates_ambiguous_returns_all_matches(tmp_path: Path) -> None:
    results = [
        _make_scored("Jane Doe", 90, "jane_doe.pdf"),
        _make_scored("Jane Smith", 70, "jane_smith.pdf"),
    ]
    out = tmp_path / "results.json"
    write_json(results, out)

    matches = find_candidates(out, "jane")
    assert len(matches) == 2
