#!/usr/bin/env python3
"""Prepare the body for LLM input.

Reads the body from the GitHub event payload (works for both issues and PRs),
truncates to MAX_BODY_CHARS, and writes it to a file.

Used as step 1 of both the Issue Review and PR Review Bot workflows.
"""

import argparse
import json
import os
from pathlib import Path

MAX_BODY_CHARS = 4000


def get_body() -> str:
    event_path = os.environ.get("GITHUB_EVENT_PATH", "")
    if not event_path:
        return ""
    with open(event_path) as f:
        event = json.load(f)
    # Works for both issue and pull_request events
    body = event.get("issue", event.get("pull_request", {})).get("body") or ""
    return body


def main() -> None:
    parser = argparse.ArgumentParser(description="Truncate issue/PR body for LLM input")
    parser.add_argument("--body-output", default="body.txt", help="File to write the truncated body to")
    args = parser.parse_args()

    raw_body = get_body()
    body = raw_body[:MAX_BODY_CHARS]
    if len(raw_body) > MAX_BODY_CHARS:
        body += f"\n\n[... truncated, original length: {len(raw_body)} chars ...]"
    Path(args.body_output).write_text(body)
    print(f"Body prepared ({len(body)} chars, original: {len(raw_body)} chars)")


if __name__ == "__main__":
    main()
