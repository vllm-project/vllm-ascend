#!/usr/bin/env python3
"""Step 1: Extract issue type from title, write to a file for the next step."""

import argparse
import os
import sys
from pathlib import Path

ISSUE_TITLE = os.environ["ISSUE_TITLE"]

TITLE_TO_TEMPLATE = {
    "[Bug]": "400-bug-report.yml",
    "[Installation]": "200-installation.yml",
    "[Usage]": "300-usage.yml",
    "[Doc]": "100-documentation.yml",
    "[Misc]": "800-others.yml",
}


def extract_issue_type(title: str) -> str | None:
    for prefix in TITLE_TO_TEMPLATE:
        if title.startswith(f"{prefix}:"):
            return prefix
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract issue type from title")
    parser.add_argument("--output", default="issue_type.txt", help="File to write the issue type to")
    args = parser.parse_args()

    issue_type = extract_issue_type(ISSUE_TITLE)
    if issue_type is None:
        print(f"Unrecognized issue title format: {ISSUE_TITLE}")
        sys.exit(0)

    Path(args.output).write_text(issue_type)
    print(f"Issue type: {issue_type}")


if __name__ == "__main__":
    main()
