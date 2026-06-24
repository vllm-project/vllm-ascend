#!/usr/bin/env python3
"""Load a system prompt text file for the given variant and write it to a file.

Supported variants: ``issue``, ``pr``.

Used in both the Issue Review and PR Review Bot workflows.
"""

import argparse
from pathlib import Path

from lib.prompts import load_system_prompt


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare system prompt")
    parser.add_argument("--variant", default="issue", choices=["issue", "pr"], help="Prompt variant: issue or pr")
    parser.add_argument("--output", default="system_prompt.txt", help="File to write the system prompt to")
    args = parser.parse_args()

    prompt = load_system_prompt(args.variant)
    Path(args.output).write_text(prompt)
    print(f"System prompt prepared: variant={args.variant} ({len(prompt)} chars)")


if __name__ == "__main__":
    main()
