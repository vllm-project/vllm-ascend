#!/usr/bin/env python3
"""Test script: read issues/PRs from CSV, call LLM using the same pipeline
as GitHub Actions (call_llm.py), and write the LLM output + expected_ok_dsv4
back to the CSV.

Modes:
    issue       - Evaluate issues against issue templates
    pr          - Evaluate PRs against PR template
    issue_judge - Judge the correctness of existing issue evaluations
    pr_judge    - Judge the correctness of existing PR evaluations

Usage:
    # Evaluate issues
    python test_issues_csv.py --mode issue --input issues.csv

    # Evaluate PRs
    python test_issues_csv.py --mode pr --input prs.csv

    # Judge existing evaluations
    python test_issues_csv.py --mode issue_judge --input issues.csv
    python test_issues_csv.py --mode pr_judge --input prs.csv

Requires VLLM_BASE_URL / LLM_BASE_URL and VLLM_API_KEY / LLM_API_KEY
environment variables (same as production).
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import regex as re

# Add parent to path so we can import the step modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lib.llm import call_llm
from lib.prefix_map import PREFIX_TO_TYPE_KEY, extract_issue_type
from lib.prompts import load_system_prompt
from lib.templates import load_issue_template, load_pr_template

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

JUDGE_SYSTEM_PROMPT = """You are a review quality auditor.
Your task is to audit whether another LLM (Review Bot)'s evaluation
of Issue/PR description completeness is correct.

Audit dimensions:
1. **Is the ok judgment reasonable**: Whether the Review Bot's ok=true/false matches the actual situation. Criteria:
   - If the description is truly high-quality (informative, clear), ok should be true
   - If the description severely lacks key information (empty, no substance), ok should be false
   - The scoring standard should match the description quality

2. **Are missing_items accurate**: Whether the items listed in missing_items are genuinely missing:
   - If the description already contains certain information, it should not be listed as missing
   - Missing items should be required information, not optional items

3. **Are suggestions effective**: Whether the improvement suggestions are:
   - Specific and actionable (not vague generalities)
   - Do not use "required"/"must" type language
   - Constructive guidance for supplementing missing information

4. **Is reasoning self-consistent**: Whether the review reasoning aligns
   with ok judgment and score, and the logic is coherent.

Output format:
{
    "ok_reasonable": true or false,
    "reasoning_valid": true or false,
    "suggestions_valid": true or false,
    "judge_reasoning": "overall audit notes, pointing out specific issues"
}

Notes:
- Output JSON only, no other text (no ```json markers)
- judge_reasoning should be concise, stating the audit conclusion and issues
"""

JUDGE_JSON_FORMAT = """Output strictly the following JSON format:
{
    "ok_reasonable": true or false,
    "reasoning_valid": true or false,
    "suggestions_valid": true or false,
    "judge_reasoning": "overall audit notes"
}
Notes: output JSON only, no other text (no ```json markers)"""


def build_judge_prompt(row: dict, kind: str) -> str:
    title = row.get("title", "")
    body = row.get("body", "")
    raw_output = row.get("deepseek_v4_flash_output", "")
    expected_ok = row.get("expected_ok_dsv4", "")
    kind_label = "PR" if "pr" in kind else "Issue"

    # Try to parse existing evaluation
    try:
        eval_data = parse_json_output(raw_output)
        eval_summary = json.dumps(eval_data, ensure_ascii=False, indent=2)
    except Exception:
        eval_summary = raw_output[:2000]

    return f"""### Original Content to Audit
{kind_label} title: {title}
{kind_label} description:
\"\"\"{body}\"\"\"

### Review Bot Output
Expected ok judgment: {expected_ok}
Detailed review output:
{eval_summary}

### Audit Task
Please audit whether the Review Bot's output is correct:
- Is the ok judgment reasonable?
- Is the reasoning self-consistent?
- Are the suggestions effective and free of "required"/"must" language?
- Are the items in missing_items genuinely missing?

{JUDGE_JSON_FORMAT}"""


def judge_row(row: dict, kind: str) -> dict:
    """Judge an existing evaluation row. Returns dict with judge output."""
    user_prompt = build_judge_prompt(row, kind)
    raw_output = call_llm(JUDGE_SYSTEM_PROMPT, user_prompt)
    try:
        parsed = parse_json_output(raw_output)
        return {
            "judge_raw_output": raw_output,
            "ok_reasonable": str(parsed.get("ok_reasonable", "")).lower(),
            "reasoning_valid": str(parsed.get("reasoning_valid", "")).lower(),
            "suggestions_valid": str(parsed.get("suggestions_valid", "")).lower(),
            "judge_reasoning": parsed.get("judge_reasoning", ""),
        }
    except Exception:
        return {
            "judge_raw_output": raw_output,
            "ok_reasonable": "error",
            "reasoning_valid": "error",
            "suggestions_valid": "error",
            "judge_reasoning": f"parse error: {raw_output[:200]}",
        }


def parse_json_output(text: str) -> dict:
    json_match = re.search(r"\{[\s\S]*\}", text)
    if json_match:
        return json.loads(json_match.group(0))
    raise ValueError(f"Could not extract JSON from LLM output: {text[:200]}")


def is_output_valid(text: str) -> bool:
    """Check if LLM output is parseable JSON (not truncated/malformed)."""
    if not text or not text.strip():
        return False
    try:
        parse_json_output(text)
        return True
    except (json.JSONDecodeError, ValueError):
        return False


def validate_result(data: dict) -> dict:
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


def build_user_prompt(title: str, body: str, template_text: str, type_key: str, kind: str = "issue") -> str:
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


def load_template(kind: str, prefix: str) -> str:
    if kind == "pr":
        return load_pr_template()
    return load_issue_template(prefix)


def process_row(row: dict, kind: str, system_prompt: str) -> dict:
    title = row.get("title", "")
    body = row.get("body", "")
    prefix = row.get("prefix", "") or (extract_issue_type(title) or "[Misc]")
    type_key = PREFIX_TO_TYPE_KEY.get(prefix, "other")
    template_text = load_template(kind, prefix)
    user_prompt = build_user_prompt(title, body, template_text, type_key, kind)
    raw_output = call_llm(system_prompt, user_prompt)
    try:
        parsed = validate_result(parse_json_output(raw_output))
    except (json.JSONDecodeError, ValueError):
        parsed = {"ok": False, "score": 0}
    return {"raw_output": raw_output, "expected_ok": parsed["ok"], "score": parsed["score"]}


def main() -> None:
    parser = argparse.ArgumentParser(description="Test issues/PRs CSV with LLM (same pipeline as GitHub Actions)")
    parser.add_argument(
        "--mode",
        default="issue",
        choices=["issue", "pr", "issue_judge", "pr_judge"],
        help="Test mode: issue, pr, issue_judge, pr_judge",
    )
    parser.add_argument("--input", default=None, help="Input CSV file")
    parser.add_argument("--output", default=None, help="Output CSV file (defaults to same as input)")
    parser.add_argument("--start", type=int, default=0, help="Start index (0-based)")
    parser.add_argument("--limit", type=int, default=0, help="Max rows to process (0 = all)")
    parser.add_argument(
        "--skip-existing", action="store_true", default=True, help="Skip rows that already have results"
    )
    parser.add_argument(
        "--retry-errors",
        action="store_true",
        default=False,
        help="Re-evaluate/judge rows with malformed or error outputs",
    )
    args = parser.parse_args()

    is_judge = args.mode in ("issue_judge", "pr_judge")
    kind = "pr" if "pr" in args.mode else "issue"

    default_input = {
        "issue": "issues.csv",
        "pr": "prs.csv",
        "issue_judge": "issues.csv",
        "pr_judge": "prs.csv",
    }[args.mode]
    input_path = Path(args.input or default_input)
    output_path = Path(args.output or args.input or default_input)

    if not input_path.exists():
        print(f"Input file not found: {input_path}")
        sys.exit(1)

    with open(input_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames)
        rows = list(reader)

    if not is_judge:
        if "expected_ok_dsv4" not in fieldnames:
            fieldnames.append("expected_ok_dsv4")
        if "deepseek_v4_flash_output" not in fieldnames:
            idx = fieldnames.index("expected_ok_dsv4")
            fieldnames.insert(idx, "deepseek_v4_flash_output")
    else:
        for col in ["judge_raw_output", "ok_reasonable", "reasoning_valid", "suggestions_valid", "judge_reasoning"]:
            if col not in fieldnames:
                fieldnames.append(col)

    total = len(rows)
    print(f"Loaded {total} rows from {input_path}")
    if is_judge:
        print(f"Mode: {args.mode} (judging existing evaluations)")
    else:
        system_prompt = load_system_prompt(kind)
        print(f"System prompt: {len(system_prompt)} chars")
        print(f"Mode: {args.mode}")

    end = total if args.limit == 0 else min(args.start + args.limit, total)
    print(f"Processing rows {args.start} to {end - 1} ({end - args.start} total)")

    success = 0
    failed = 0
    skipped = 0

    for i in range(args.start, end):
        row = rows[i]
        num = row.get("issue_number") or row.get("pr_number") or "?"
        title = row.get("title", "")
        print(f"\n[{i + 1}/{total}] #{num}: {title[:80]}...")

        if is_judge:
            existing_judge = (row.get("judge_reasoning") or "").strip()
            existing_output = (row.get("deepseek_v4_flash_output") or "").strip()
            existing_ok_r = (row.get("ok_reasonable") or "").strip()

            if args.skip_existing and existing_judge:
                if args.retry_errors and existing_ok_r == "error":
                    print("  RETRY (prev judge had error, re-judging)")
                    row["judge_reasoning"] = ""
                    row["ok_reasonable"] = ""
                else:
                    print("  SKIP (already judged)")
                    skipped += 1
                    continue

            if not existing_output:
                print("  SKIP (no evaluation to judge)")
                skipped += 1
                continue

            if args.skip_existing and existing_output and not is_output_valid(existing_output):
                print("  SKIP (eval output still malformed, needs re-evaluation first)")
                skipped += 1
                continue

            try:
                result = judge_row(row, kind)
                row["judge_raw_output"] = result["judge_raw_output"]
                row["ok_reasonable"] = result["ok_reasonable"]
                row["reasoning_valid"] = result["reasoning_valid"]
                row["suggestions_valid"] = result["suggestions_valid"]
                row["judge_reasoning"] = result["judge_reasoning"]
                success += 1
                print(
                    f"  JUDGED: ok_reasonable={result['ok_reasonable']} "
                    f"reasoning_valid={result['reasoning_valid']} "
                    f"suggestions_valid={result['suggestions_valid']}"
                )
            except Exception as e:
                row["judge_reasoning"] = f"ERROR: {e}"
                failed += 1
                print(f"  FAILED: {e}")
        else:
            existing_val = (row.get("expected_ok_dsv4") or "").strip()
            existing_output = (row.get("deepseek_v4_flash_output") or "").strip()

            if args.skip_existing and existing_val:
                if args.retry_errors and existing_output and not is_output_valid(existing_output):
                    print("  RETRY (output malformed, re-evaluating)")
                    row["expected_ok_dsv4"] = ""
                    row["judge_reasoning"] = ""
                    row["ok_reasonable"] = ""
                else:
                    print(f"  SKIP (already has expected_ok_dsv4={existing_val})")
                    skipped += 1
                    continue

            if args.skip_existing and existing_output and not existing_val:
                if args.retry_errors and not is_output_valid(existing_output):
                    print("  RETRY (output malformed, re-evaluating)")
                else:
                    try:
                        parsed = validate_result(parse_json_output(existing_output))
                        row["expected_ok_dsv4"] = str(parsed["ok"]).lower()
                        print(f"  PARSED from existing output: ok={parsed['ok']} score={parsed['score']}")
                        success += 1
                    except (json.JSONDecodeError, ValueError):
                        print("  Could not parse existing output, will call LLM")
                    else:
                        continue

            try:
                result = process_row(row, kind, system_prompt)
                row["deepseek_v4_flash_output"] = result["raw_output"]
                row["expected_ok_dsv4"] = str(result["expected_ok"]).lower()
                success += 1
                print(f"  OK: ok={result['expected_ok']} score={result['score']} ({len(result['raw_output'])} chars)")
            except Exception as e:
                row["deepseek_v4_flash_output"] = f"ERROR: {e}"
                row["expected_ok_dsv4"] = ""
                failed += 1
                print(f"  FAILED: {e}")

        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        time.sleep(0.1)

    print(f"\n{'=' * 60}")
    print(f"Done. Success: {success}, Failed: {failed}, Skipped: {skipped}")
    print(f"Results written to {output_path}")


if __name__ == "__main__":
    main()
