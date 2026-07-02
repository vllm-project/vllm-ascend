#!/usr/bin/env python3
"""Apply label add/remove actions from a JSON action file.

Reads ``label_actions.json`` (``{"add": [...], "remove": [...]}``) and
applies the changes to the issue/PR via the GitHub API.

Replaces the JS-based ``actions/github-script`` label management that was
previously duplicated in both the PR and Issue review workflows.
"""

import argparse
import json
import os
from pathlib import Path

from lib.github_api import manage_labels

ISSUE_NUMBER = os.environ["ISSUE_NUMBER"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply label actions from JSON")
    parser.add_argument("--input", default="label_actions.json", help="Label actions JSON file")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"No label actions file found: {input_path}")
        return

    actions = json.loads(input_path.read_text())
    add = actions.get("add", [])
    remove = actions.get("remove", [])

    if not add and not remove:
        print("No label actions to apply")
        return

    manage_labels(ISSUE_NUMBER, add=add, remove=remove)


if __name__ == "__main__":
    main()
