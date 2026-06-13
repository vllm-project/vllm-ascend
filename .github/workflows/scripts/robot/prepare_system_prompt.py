#!/usr/bin/env python3
"""Step 3: Load the system prompt and write it to a file."""

import argparse
from pathlib import Path

PROMPT_DIR = Path(__file__).resolve().parent / "issue_review_prompts"


def load_system_prompt() -> str:
    path = PROMPT_DIR / "system_prompt.txt"
    if path.exists():
        return path.read_text()
    return "You are an issue triage assistant."


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare system prompt")
    parser.add_argument("--output", default="system_prompt.txt", help="File to write the system prompt to")
    args = parser.parse_args()

    prompt = load_system_prompt()
    Path(args.output).write_text(prompt)
    print(f"System prompt prepared ({len(prompt)} chars)")


if __name__ == "__main__":
    main()
