#!/usr/bin/env python3
"""Call the LLM for description completeness checking.

Reads the prepared system prompt, template, type-key, and body files,
calls the LLM, parses the structured compliance result, and writes JSON to a
file. Shared review logic lives in ``lib/review.py``.

Used as step 4 of both the Issue Review and PR Review Bot workflows.
"""

import argparse
import json
import os
from pathlib import Path

from lib.llm import call_llm
from lib.prefix_map import PREFIX_TO_TYPE_KEY
from lib.review import build_review_prompt, parse_error_result, parse_json_output, validate_result

ISSUE_TITLE = os.environ["ISSUE_TITLE"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Call LLM for description completeness check")
    parser.add_argument("--system-prompt", default="system_prompt.txt", help="File containing the system prompt")
    parser.add_argument("--template", default="template.txt", help="File containing the issue/PR template")
    parser.add_argument("--type-key", default="issue_type.txt", help="File containing the issue type prefix")
    parser.add_argument("--body", default="body.txt", help="File containing the (pre-truncated) issue/PR body")
    parser.add_argument("--output", default="review_result.json", help="File to write the review result JSON to")
    parser.add_argument("--kind", default="issue", choices=["issue", "pr"], help="Target kind: issue or pr")
    args = parser.parse_args()

    system_prompt_path = Path(args.system_prompt)
    template_path = Path(args.template)
    type_key_path = Path(args.type_key)
    body_path = Path(args.body)

    if not system_prompt_path.exists():
        raise FileNotFoundError(f"System prompt file not found: {system_prompt_path}")
    if not template_path.exists():
        raise FileNotFoundError(f"Template file not found: {template_path}")

    system_prompt = system_prompt_path.read_text()
    template_text = template_path.read_text()
    body_text = body_path.read_text() if body_path.exists() else os.environ.get("ISSUE_BODY", "")

    type_prefix = type_key_path.read_text().strip() if type_key_path.exists() else ""
    type_key = PREFIX_TO_TYPE_KEY.get(type_prefix, "other")

    user_prompt = build_review_prompt(ISSUE_TITLE, body_text, template_text, type_key, args.kind)

    print("Calling LLM for description completeness check...")
    raw_output = call_llm(system_prompt, user_prompt)

    try:
        validated = validate_result(parse_json_output(raw_output))
    except (json.JSONDecodeError, ValueError) as e:
        print(f"Failed to parse LLM output: {e}")
        print(f"Raw output (first 500 chars): {raw_output[:500]}")
        validated = parse_error_result(e)

    output_path = Path(args.output)
    output_path.write_text(json.dumps(validated, ensure_ascii=False, indent=2))
    status = "PASS" if validated["ok"] else "FAIL"
    print(f"Review complete: ok={validated['ok']} score={validated['score']} status={status}")
    print(f"Result written to {output_path}")


if __name__ == "__main__":
    main()
