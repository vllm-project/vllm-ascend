#!/usr/bin/env python3
"""Step 4: Call LLM with assembled prompt.

Reads system prompt, template, and issue content, calls the vLLM endpoint,
and writes the raw review output to a file for the next workflow step.

Usage:
    python call_llm.py --system-prompt system_prompt.txt --template template.txt --output review_output.md
"""

import argparse
import os
import sys
from pathlib import Path

import requests

VLLM_BASE_URL = os.environ["VLLM_BASE_URL"]
VLLM_API_KEY = os.environ["VLLM_API_KEY"]
ISSUE_TITLE = os.environ["ISSUE_TITLE"]
ISSUE_BODY = os.environ.get("ISSUE_BODY", "")


def call_vllm(system_prompt: str, user_prompt: str) -> str:
    headers = {
        "Authorization": f"Bearer {VLLM_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "default",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.3,
        "max_tokens": 1024,
    }
    resp = requests.post(
        f"{VLLM_BASE_URL}/v1/chat/completions",
        headers=headers,
        json=payload,
        timeout=120,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Call LLM for issue review")
    parser.add_argument("--system-prompt", default="system_prompt.txt", help="File containing the system prompt")
    parser.add_argument("--template", default="template.txt", help="File containing the issue template")
    parser.add_argument("--output", default="review_output.md", help="File to write the LLM review output to")
    args = parser.parse_args()

    system_prompt_path = Path(args.system_prompt)
    template_path = Path(args.template)

    if not system_prompt_path.exists():
        print(f"System prompt file not found: {system_prompt_path}")
        sys.exit(1)
    if not template_path.exists():
        print(f"Template file not found: {template_path}")
        sys.exit(1)

    system_prompt = system_prompt_path.read_text()
    template_text = template_path.read_text()

    user_prompt = f"""## Issue Title
{ISSUE_TITLE}

## Issue Template (what the user was asked to fill in)
{template_text}

## Issue Content (what the user actually submitted)
{ISSUE_BODY}

Please review this issue and provide your feedback in the specified format.
"""

    print("Calling vLLM for review...")
    review = call_vllm(system_prompt, user_prompt)

    output_path = Path(args.output)
    output_path.write_text(review)
    print(f"Review written to {output_path} ({len(review)} chars)")


if __name__ == "__main__":
    main()
