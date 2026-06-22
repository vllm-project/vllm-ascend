#!/usr/bin/env python3
"""Load the matching issue or PR template and write it to a file.

Supports two input formats:

* ``issue_type.txt`` — contains an issue-type prefix (e.g. ``[Bug]``).
* ``pr_info.json`` — contains PR metadata; loads ``PULL_REQUEST_TEMPLATE.md``.

Used as step 2 of both the Issue Review and PR Review Bot workflows.
"""

import argparse
from pathlib import Path

from lib.templates import load_issue_template, load_pr_template


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare issue/PR template")
    parser.add_argument("--input", default="issue_type.txt", help="File containing the issue type or PR info")
    parser.add_argument("--output", default="template.txt", help="File to write the template to")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if input_path.suffix == ".json":
        print("Loading PR template...")
        template_text = load_pr_template()
    else:
        issue_type = input_path.read_text().strip()
        print(f"Loading template for: {issue_type}")
        template_text = load_issue_template(issue_type)

    Path(args.output).write_text(template_text)
    print(f"Template prepared ({len(template_text)} chars)")


if __name__ == "__main__":
    main()
