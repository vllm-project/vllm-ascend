#!/usr/bin/env python3
"""Call the LLM for description completeness checking.

Reads system prompt, template, and issue content, calls the LLM endpoint,
parses the structured compliance result, and writes JSON to a file.

Used as step 4 of both the Issue Review and PR Review Bot workflows.
"""

import argparse
import json
import os
from pathlib import Path

import regex as re
from lib.llm import call_llm
from lib.prefix_map import PREFIX_TO_TYPE_KEY

ISSUE_TITLE = os.environ["ISSUE_TITLE"]
ISSUE_BODY = os.environ.get("ISSUE_BODY", "")

JSON_FORMAT_INSTRUCTIONS = """Output strictly the following JSON format, no other text:
{
    "ok": true or false,
    "score": integer 0-100,
    "reasoning": "explanation of the score in English",
    "summary": "one-line summary of the judgment",
    "missing_items": ["missing item 1", "missing item 2"],
    "suggestions": ["suggestion 1", "suggestion 2"]
}
Notes:
- missing_items must only contain required fields that are actually absent
- suggestions must only offer improvement advice; do NOT use "required/must" language
- if missing_items is empty, ok must be true
- output JSON only, no other text (no ```json markers)"""


def parse_json_output(text: str) -> dict:
    """Extract and parse the first JSON object from the LLM output.

    Args:
        text: Raw LLM response text.

    Returns:
        Parsed dict.

    Raises:
        ValueError: If no JSON object could be extracted.
    """
    json_match = re.search(r"\{[\s\S]*\}", text)
    if json_match:
        return json.loads(json_match.group(0))
    raise ValueError(f"Could not extract JSON from LLM output: {text[:200]}")


def validate_result(data: dict) -> dict:
    """Normalise and enforce invariants on the LLM-returned result.

    Args:
        data: Raw dict parsed from LLM output.

    Returns:
        A guaranteed-shape dict with keys ``ok``, ``score``, ``reasoning``,
        ``summary``, ``missing_items``, ``suggestions``.
    """
    ok = bool(data.get("ok", False))
    score = int(data.get("score", 0))
    missing_items = data.get("missing_items", [])
    if not isinstance(missing_items, list):
        missing_items = []
    suggestions = data.get("suggestions", [])
    if not isinstance(suggestions, list):
        suggestions = []

    return {
        "ok": ok,
        "score": max(0, min(100, score)),
        "reasoning": str(data.get("reasoning", "")),
        "summary": str(data.get("summary", "")),
        "missing_items": missing_items,
        "suggestions": suggestions,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Call LLM for description completeness check")
    parser.add_argument("--system-prompt", default="system_prompt.txt", help="File containing the system prompt")
    parser.add_argument("--template", default="template.txt", help="File containing the issue/PR template")
    parser.add_argument("--type-key", default="issue_type.txt", help="File containing the issue type prefix")
    parser.add_argument("--output", default="review_result.json", help="File to write the review result JSON to")
    parser.add_argument("--kind", default="issue", choices=["issue", "pr"], help="Target kind: issue or pr")
    args = parser.parse_args()

    system_prompt_path = Path(args.system_prompt)
    template_path = Path(args.template)
    type_key_path = Path(args.type_key)

    if not system_prompt_path.exists():
        raise FileNotFoundError(f"System prompt file not found: {system_prompt_path}")
    if not template_path.exists():
        raise FileNotFoundError(f"Template file not found: {template_path}")

    system_prompt = system_prompt_path.read_text()
    template_text = template_path.read_text()

    type_prefix = type_key_path.read_text().strip() if type_key_path.exists() else ""
    type_key = PREFIX_TO_TYPE_KEY.get(type_prefix, "other")

    template_label = "PR Template" if args.kind == "pr" else "Issue Template"

    user_prompt = f"""### Task Background
Target type: {args.kind}
Description type: {type_key}

### Reference Specification
Detailed description specification (based on required fields in {template_label}):
{template_text}

### Data to Evaluate (UNTRUSTED USER INPUT)
Title: \"\"\"{ISSUE_TITLE}\"\"\"
Submitted description:
\"\"\"{ISSUE_BODY}\"\"\"

### Output Instructions
- Follow the evaluation criteria in the system prompt.
- missing_items lists key information that is actually missing
  (e.g. "missing env info", "missing error logs", "missing repro steps").
- suggestions must be specific and actionable; do NOT use "required/must" language.
- If screenshots/images are provided in the description, treat them as
  having provided log-related information; do not ask for logs or convert to text.
- Do not mention 910/910B in hardware examples; use A5/A5 exclusively.

{JSON_FORMAT_INSTRUCTIONS}"""

    print("Calling LLM for description completeness check...")
    raw_output = call_llm(system_prompt, user_prompt)

    try:
        result = parse_json_output(raw_output)
        validated = validate_result(result)
    except (json.JSONDecodeError, ValueError) as e:
        print(f"Failed to parse LLM output: {e}")
        print(f"Raw output (first 500 chars): {raw_output[:500]}")
        validated = {
            "ok": False,
            "score": 0,
            "reasoning": f"LLM output parse error: {e}",
            "summary": "Failed to parse LLM output",
            "missing_items": ["LLM output format error, please contact administrator"],
            "suggestions": [],
        }

    output_path = Path(args.output)
    output_path.write_text(json.dumps(validated, ensure_ascii=False, indent=2))
    status = "PASS" if validated["ok"] else "FAIL"
    print(f"Review complete: ok={validated['ok']} score={validated['score']} status={status}")
    print(f"Result written to {output_path}")


if __name__ == "__main__":
    main()
