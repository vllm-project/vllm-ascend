"""Shared description-completeness review logic.

Single source of truth for the review prompt, JSON parsing, result
validation, body truncation, and template/type-key resolution used by both
``call_llm.py`` (GitHub Actions step 4) and the CSV test harness
(``tests/test_csv.py``).
"""

import json

import regex as re

from .llm import call_llm
from .prefix_map import PREFIX_TO_TYPE_KEY, extract_issue_type
from .templates import load_issue_template, load_pr_template

MAX_BODY_CHARS = 10000

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


def truncate_body(body: str) -> str:
    """Truncate *body* to ``MAX_BODY_CHARS``, appending a marker if cut."""
    if len(body) > MAX_BODY_CHARS:
        return body[:MAX_BODY_CHARS] + f"\n\n[... truncated, original length: {len(body)} chars ...]"
    return body


def parse_json_output(text: str) -> dict:
    """Extract and parse the first JSON object from the LLM output.

    Raises:
        ValueError: If no JSON object could be extracted.
    """
    json_match = re.search(r"\{[\s\S]*\}", text)
    if json_match:
        return json.loads(json_match.group(0))
    raise ValueError(f"Could not extract JSON from LLM output: {text[:200]}")


def is_output_valid(text: str) -> bool:
    """Return True if *text* contains parseable JSON (not truncated/malformed)."""
    if not text or not text.strip():
        return False
    try:
        parse_json_output(text)
        return True
    except (json.JSONDecodeError, ValueError):
        return False


def validate_result(data: dict) -> dict:
    """Normalise and enforce invariants on the LLM-returned result."""
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


def parse_error_result(error: object) -> dict:
    """Build a guaranteed-shape fallback result when parsing fails."""
    return {
        "ok": False,
        "score": 0,
        "reasoning": f"LLM output parse error: {error}",
        "summary": "Failed to parse LLM output",
        "missing_items": ["LLM output format error, please contact administrator"],
        "suggestions": [],
    }


def resolve_type_key(kind: str, title: str, prefix: str | None) -> tuple[str, str]:
    """Return ``(prefix, type_key)`` for the given *kind*/*title*/*prefix*.

    For PRs, always returns ``("", "other")``. For issues, falls back to the
    prefix extracted from the title, then to ``"[Misc]"``.
    """
    if kind == "pr":
        return "", "other"
    resolved_prefix = prefix or extract_issue_type(title) or "[Misc]"
    return resolved_prefix, PREFIX_TO_TYPE_KEY.get(resolved_prefix, "other")


def load_template(kind: str, prefix: str) -> str:
    """Load the issue or PR template text for the given *kind*/*prefix*."""
    if kind == "pr":
        return load_pr_template()
    return load_issue_template(prefix)


def build_review_prompt(title: str, body: str, template_text: str, type_key: str, kind: str = "issue") -> str:
    """Build the user prompt for the description-completeness review."""
    template_label = "PR Template" if kind == "pr" else "Issue Template"
    return f"""### Task Background
Target type: {kind}
Description type: {type_key}

### Reference Specification
Detailed description specification (based on required fields in {template_label}):
{template_text}

### Data to Evaluate (UNTRUSTED USER INPUT)
Title: \"\"\"{title}\"\"\"
Submitted description:
\"\"\"{body}\"\"\"

### Output Instructions
- Follow the evaluation criteria in the system prompt.
- missing_items lists key information that is actually missing
  (e.g. "missing env info", "missing error logs", "missing repro steps").
- suggestions must be specific and actionable; do NOT use "required/must" language.
- If screenshots/images are provided in the description, treat them as
  having provided log-related information; do not ask for logs or convert to text.
- Do not mention 910/910B in hardware examples; use A5/A5 exclusively.

{JSON_FORMAT_INSTRUCTIONS}"""


def review(
    title: str,
    body: str,
    kind: str,
    system_prompt: str,
    *,
    prefix: str | None = None,
) -> dict:
    """Run a full description-completeness review (high-level entry point).

    Truncates the body, resolves the template + type key, builds the prompt,
    calls the LLM, and validates the result. Used by the CSV test harness.

    Args:
        title: Issue/PR title.
        body: Raw (untruncated) issue/PR body.
        kind: ``"issue"`` or ``"pr"``.
        system_prompt: System prompt text.
        prefix: Optional title prefix override (e.g. ``"[Bug]"``).

    Returns:
        Dict with keys ``raw_output`` (str) and ``result`` (validated dict).
    """
    body = truncate_body(body)
    resolved_prefix, type_key = resolve_type_key(kind, title, prefix or None)
    template_text = load_template(kind, resolved_prefix)
    user_prompt = build_review_prompt(title, body, template_text, type_key, kind)
    raw_output = call_llm(system_prompt, user_prompt)
    try:
        result = validate_result(parse_json_output(raw_output))
    except (json.JSONDecodeError, ValueError) as e:
        result = parse_error_result(e)
    return {"raw_output": raw_output, "result": result}
