"""
Look up one candidate from an existing output/results.json without rereading
the whole report.md. No LLM calls — this just reads the file main.py already
wrote.

Usage:
    python scripts/show_candidate.py "Jane"
    python scripts/show_candidate.py --output out/ jane_doe.pdf
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.reporter import find_candidates


def print_candidate(c: dict) -> None:
    profile = c["profile"]
    print(f"{profile['name']} — {c['overall_fit']['score']} ({c['source_file']})")
    print(f"{profile['years_experience']} yrs experience")
    print()
    print(c["reasoning"])
    print()
    print("Score breakdown")
    for dim in ("skills_match", "experience_match", "role_relevance", "overall_fit"):
        d = c[dim]
        print(f"  {dim}: {d['score']} — {d['reasoning']}")
    if c["gaps"]:
        print()
        print("Gaps")
        for gap in c["gaps"]:
            print(f"  [{gap['category']}] {gap['detail']}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "query",
        help="Candidate name or source filename (or part of it) to search for.",
    )
    parser.add_argument(
        "--output",
        default="output",
        help="Output dir containing results.json (default: ./output).",
    )
    args = parser.parse_args()

    json_path = Path(args.output) / "results.json"
    if not json_path.is_file():
        print(f"No results.json found at {json_path}. Run main.py first.")
        return 1

    matches = find_candidates(json_path, args.query)

    if not matches:
        print(f"No candidate matching '{args.query}'.")
        return 1

    if len(matches) > 1:
        print(f"{len(matches)} candidates match '{args.query}' — narrow your search:")
        for m in matches:
            print(f"  {m['profile']['name']} ({m['source_file']})")
        return 1

    print_candidate(matches[0])
    return 0


if __name__ == "__main__":
    sys.exit(main())
