#!/usr/bin/env python3
"""Step 3: Load the system prompt and write it to a file.

Supports --variant flag for different prompt types:
  - issue: description_check_prompts/system_prompt.txt
  - pr:    description_check_prompts/pr_system_prompt.txt
  - commit: commit_check_prompts/system_prompt.txt
"""

import argparse
from pathlib import Path

PROMPT_DIR = Path(__file__).resolve().parent / "description_check_prompts"
COMMIT_PROMPT_DIR = Path(__file__).resolve().parent / "commit_check_prompts"
DEFAULT_FALLBACK = "You are an issue triage assistant."

PROMPT_NAME_MAP = {
    "issue": ("description_check_prompts", "system_prompt.txt"),
    "pr": ("description_check_prompts", "pr_system_prompt.txt"),
    "commit": ("commit_check_prompts", "system_prompt.txt"),
}


def load_system_prompt(variant: str = "issue") -> str:
    if variant in PROMPT_NAME_MAP:
        subdir, filename = PROMPT_NAME_MAP[variant]
        prompt_dir = COMMIT_PROMPT_DIR if variant == "commit" else PROMPT_DIR
        path = prompt_dir / filename
    else:
        path = PROMPT_DIR / "system_prompt.txt"

    if path.exists():
        return path.read_text()
    return DEFAULT_FALLBACK


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare system prompt")
    parser.add_argument("--variant", default="issue", choices=["issue", "pr", "commit"],
                        help="Prompt variant: issue, pr, or commit")
    parser.add_argument("--output", default="system_prompt.txt", help="File to write the system prompt to")
    args = parser.parse_args()

    prompt = load_system_prompt(args.variant)
    Path(args.output).write_text(prompt)
    print(f"System prompt prepared: variant={args.variant} ({len(prompt)} chars)")


if __name__ == "__main__":
    main()
