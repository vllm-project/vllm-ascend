#!/usr/bin/env python3
"""Call the LLM for description completeness checking.

Reads the issue/PR title and body straight from the GitHub event payload,
runs the shared review pipeline (``lib/review.py``), and writes the structured
compliance result to a JSON file.

This is the single production step of both the Issue Review and PR Review Bot
workflows. It uses the exact same ``review()`` code path as the CSV test
harness (``tests/test_csv.py``), so the LLM input is identical in both.
"""

import argparse
import json
from pathlib import Path

from lib.review import review_from_event

# Skip-marker result written when the title is not eligible for review.
SKIPPED_RESULT = {
    "ok": True,
    "score": 0,
    "reasoning": "Title not eligible for review; skipped.",
    "summary": "Skipped",
    "missing_items": [],
    "suggestions": [],
    "executed": False,
    "skipped": True,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Call LLM for description completeness check")
    parser.add_argument("--output", default="review_result.json", help="File to write the review result JSON to")
    parser.add_argument("--kind", default="issue", choices=["issue", "pr"], help="Target kind: issue or pr")
    args = parser.parse_args()

    print("Calling LLM for description completeness check...")
    outcome = review_from_event(args.kind)

    if outcome is None:
        validated = SKIPPED_RESULT
    else:
        validated = outcome["result"]

    output_path = Path(args.output)
    output_path.write_text(json.dumps(validated, ensure_ascii=False, indent=2))
    status = "SKIP" if validated.get("skipped") else ("PASS" if validated["ok"] else "FAIL")
    print(f"Review complete: ok={validated['ok']} score={validated['score']} status={status}")
    print(f"Result written to {output_path}")


if __name__ == "__main__":
    main()
