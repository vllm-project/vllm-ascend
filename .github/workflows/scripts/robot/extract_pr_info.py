#!/usr/bin/env python3
"""Step 1: Extract PR metadata from event payload and write to JSON."""

import argparse
import json
import os
from pathlib import Path

EVENT_PATH = os.environ["GITHUB_EVENT_PATH"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract PR info from event payload")
    parser.add_argument("--output", default="pr_info.json", help="File to write the PR info JSON to")
    args = parser.parse_args()

    with open(EVENT_PATH) as f:
        event = json.load(f)

    pr = event.get("pull_request", {})
    action = event.get("action", "")
    changes = event.get("changes", {})

    title_changed = bool(changes.get("title"))
    body_changed = bool(changes.get("body"))

    info = {
        "action": action,
        "number": pr.get("number", 0),
        "title": pr.get("title", ""),
        "body": pr.get("body") or "",
        "state": pr.get("state", ""),
        "base_sha": pr.get("base", {}).get("sha", ""),
        "head_sha": pr.get("head", {}).get("sha", ""),
        "title_changed": title_changed,
        "body_changed": body_changed,
    }

    Path(args.output).write_text(json.dumps(info, ensure_ascii=False, indent=2))
    print(f"PR info extracted: action={action} #{info['number']}")
    print(f"  base={info['base_sha'][:7]} head={info['head_sha'][:7]}")


if __name__ == "__main__":
    main()
