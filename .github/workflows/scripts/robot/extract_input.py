#!/usr/bin/env python3
"""Extract the issue type prefix from the title and write it to a file.

Used as step 1 of the Issue Review Bot workflow.
"""

import argparse
import os
from pathlib import Path

from lib.prefix_map import extract_issue_type

ISSUE_TITLE = os.environ["ISSUE_TITLE"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract issue type from title")
    parser.add_argument("--output", default="issue_type.txt", help="File to write the issue type to")
    args = parser.parse_args()

    issue_type = extract_issue_type(ISSUE_TITLE)
    if issue_type is None:
        print(f"Unrecognized issue title format: {ISSUE_TITLE}, skipping review")
        return

    Path(args.output).write_text(issue_type)
    print(f"Issue type: {issue_type}")


if __name__ == "__main__":
    main()
