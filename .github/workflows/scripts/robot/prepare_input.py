#!/usr/bin/env python3
"""Prepare all LLM inputs in one step.

1. Extract issue type prefix from title → issue_type.txt
2. Load matching issue template → template.txt
3. Truncate body to MAX_BODY_CHARS → body.txt

Used as step 1 of the Issue Review Bot workflow.
"""

import argparse
import os
from pathlib import Path

from lib.prefix_map import extract_issue_type
from lib.templates import load_issue_template

ISSUE_TITLE = os.environ["ISSUE_TITLE"]
ISSUE_BODY = os.environ.get("ISSUE_BODY", "")

MAX_BODY_CHARS = 4000


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare issue type, template, and body")
    parser.add_argument("--type-output", default="issue_type.txt", help="File to write the issue type to")
    parser.add_argument("--template-output", default="template.txt", help="File to write the template to")
    parser.add_argument("--body-output", default="body.txt", help="File to write the truncated body to")
    args = parser.parse_args()

    issue_type = extract_issue_type(ISSUE_TITLE)
    if issue_type is None:
        print(f"Unrecognized issue title format: {ISSUE_TITLE}, skipping review")
        return

    Path(args.type_output).write_text(issue_type)
    print(f"Issue type: {issue_type}")

    template_text = load_issue_template(issue_type)
    Path(args.template_output).write_text(template_text)
    print(f"Template prepared ({len(template_text)} chars)")

    body = ISSUE_BODY[:MAX_BODY_CHARS]
    if len(ISSUE_BODY) > MAX_BODY_CHARS:
        body += f"\n\n[... truncated, original length: {len(ISSUE_BODY)} chars ...]"
    Path(args.body_output).write_text(body)
    print(f"Body prepared ({len(body)} chars, original: {len(ISSUE_BODY)} chars)")


if __name__ == "__main__":
    main()
