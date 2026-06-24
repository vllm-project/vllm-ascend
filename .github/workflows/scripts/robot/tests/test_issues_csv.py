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
import os
import re
import sys
import time
from pathlib import Path

# Add parent to path so we can import the step modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lib.llm import call_llm
from lib.prefix_map import PREFIX_TO_TYPE_KEY, extract_issue_type
from lib.prompts import load_system_prompt
from lib.templates import load_issue_template, load_pr_template

JSON_FORMAT_INSTRUCTIONS = """请严格按照以下 JSON 格式输出，不要包含任何其他文本：
{
    "ok": true或false,
    "score": 0到100的整数,
    "reasoning": "评分理由的中文说明",
    "summary": "一句话总结判断依据",
    "missing_items": ["缺失项1", "缺失项2"],
    "suggestions": ["改进建议1", "改进建议2"]
}
注意：
- missing_items 只能包含必填项中确实缺失的内容
- suggestions 只能给改进建议，禁止使用"必填/必须"等表述
- 如果 missing_items 为空，ok 必须为 true
- 严格输出 JSON，不要输出任何其他文本（不要输出 ```json 标记）"""


JUDGE_SYSTEM_PROMPT = """你是一个评审质量审核员。你的任务是审核另一个 LLM（评审 Bot）对 Issue/PR 描述完整性的评审结果是否正确。

审核维度：
1. **ok 判断是否合理**：评审 Bot 给出的 ok=true/false 是否符合实际情况。判断标准：
   - 如果描述确实是高质量的（信息充分、清晰），ok 应为 true
   - 如果描述严重缺失关键信息（空描述、无实质性内容），ok 应为 false
   - 评分标准应与描述质量一致

2. **missing_items 是否准确**：missing_items 中列出的缺失项是否真的缺失：
   - 如果描述的确定中已经包含了某项信息，不应列为缺失
   - 缺失项应该是必填信息，不应包含可选项目

3. **suggestions 是否有效**：改进建议是否：
   - 具体、可执行（不是泛泛而谈）
   - 没有使用"必填""必须"等强制性表述
   - 是对缺失信息的建设性补充引导

4. **reasoning 是否自洽**：评审理由是否与 ok 判断和 score 一致，逻辑是否连贯。

输出格式：
{
    "ok_reasonable": true或false,
    "reasoning_valid": true或false,
    "suggestions_valid": true或false,
    "judge_reasoning": "整体审核说明，指出具体问题"
}

注意：
- 严格输出 JSON，不要输出任何其他文本（不要输出 ```json 标记）
- judge_reasoning 用中文，简洁指出审核结论和问题
"""

JUDGE_JSON_FORMAT = """请严格按照以下 JSON 格式输出：
{
    "ok_reasonable": true或false,
    "reasoning_valid": true或false,
    "suggestions_valid": true或false,
    "judge_reasoning": "整体审核说明"
}
注意：严格输出 JSON，不要输出任何其他文本（不要输出 ```json 标记）"""


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

    return f"""### 待审核的原始内容
{kind_label} 标题：{title}
{kind_label} 描述：
\"\"\"{body}\"\"\"

### 评审 Bot 的输出
预期 ok 判断：{expected_ok}
评审详细输出：
{eval_summary}

### 审核任务
请审核评审 Bot 的输出是否正确：
- ok 判断是否合理？
- reasoning 是否自洽？
- suggestions 是否有效且不包含"必填"等强制表述？
- missing_items 中列出的缺失项是否真的缺失？

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


def build_user_prompt(title: str, body: str, template_text: str,
                      type_key: str, kind: str = "issue") -> str:
    template_label = "PR 模板" if kind == "pr" else "Issue 模板"
    return f"""### 任务背景
目标类型：{kind}
描述类型：{type_key}

### 规范参考
详细描述规范（根据 {template_label} 中的必填字段判定）：
{template_text}

### 待评估数据 (UNTRUSTED USER INPUT)
标题：\"\"\"{title}\"\"\"
提交的描述：
\"\"\"{body}\"\"\"

### 输出指令
- 遵循系统提示中的评估准则进行判断。
- missing_items 列出实际缺失的关键信息（如"缺失环境信息""缺失错误日志""缺失复现步骤"）。
- suggestions 给出具体、可执行的改进建议，禁止使用"必填/必须"等表述。
- 若描述中已提供截图/图片，视为已提供日志相关信息，不得要求补充日志或要求转成文本。
- 硬件型号示例中不要出现 910/910B，统一使用 A3/A5。

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
    parser = argparse.ArgumentParser(
        description="Test issues/PRs CSV with LLM (same pipeline as GitHub Actions)"
    )
    parser.add_argument("--mode", default="issue",
                        choices=["issue", "pr", "issue_judge", "pr_judge"],
                        help="Test mode: issue, pr, issue_judge, pr_judge")
    parser.add_argument("--input", default=None, help="Input CSV file")
    parser.add_argument("--output", default=None, help="Output CSV file (defaults to same as input)")
    parser.add_argument("--start", type=int, default=0, help="Start index (0-based)")
    parser.add_argument("--limit", type=int, default=0, help="Max rows to process (0 = all)")
    parser.add_argument("--skip-existing", action="store_true", default=True,
                        help="Skip rows that already have results")
    parser.add_argument("--retry-errors", action="store_true", default=False,
                        help="Re-evaluate/judge rows with malformed or error outputs")
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
        for col in ["judge_raw_output", "ok_reasonable", "reasoning_valid",
                     "suggestions_valid", "judge_reasoning"]:
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
                    print(f"  RETRY (prev judge had error, re-judging)")
                    row["judge_reasoning"] = ""
                    row["ok_reasonable"] = ""
                else:
                    print(f"  SKIP (already judged)")
                    skipped += 1
                    continue

            if not existing_output:
                print(f"  SKIP (no evaluation to judge)")
                skipped += 1
                continue

            if args.skip_existing and existing_output and not is_output_valid(existing_output):
                print(f"  SKIP (eval output still malformed, needs re-evaluation first)")
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
                print(f"  JUDGED: ok_reasonable={result['ok_reasonable']} "
                      f"reasoning_valid={result['reasoning_valid']} "
                      f"suggestions_valid={result['suggestions_valid']}")
            except Exception as e:
                row["judge_reasoning"] = f"ERROR: {e}"
                failed += 1
                print(f"  FAILED: {e}")
        else:
            existing_val = (row.get("expected_ok_dsv4") or "").strip()
            existing_output = (row.get("deepseek_v4_flash_output") or "").strip()

            if args.skip_existing and existing_val:
                if args.retry_errors and existing_output and not is_output_valid(existing_output):
                    print(f"  RETRY (output malformed, re-evaluating)")
                    row["expected_ok_dsv4"] = ""
                    row["judge_reasoning"] = ""
                    row["ok_reasonable"] = ""
                else:
                    print(f"  SKIP (already has expected_ok_dsv4={existing_val})")
                    skipped += 1
                    continue

            if args.skip_existing and existing_output and not existing_val:
                if args.retry_errors and not is_output_valid(existing_output):
                    print(f"  RETRY (output malformed, re-evaluating)")
                else:
                    try:
                        parsed = validate_result(parse_json_output(existing_output))
                        row["expected_ok_dsv4"] = str(parsed["ok"]).lower()
                        print(f"  PARSED from existing output: ok={parsed['ok']} score={parsed['score']}")
                        success += 1
                    except (json.JSONDecodeError, ValueError):
                        print(f"  Could not parse existing output, will call LLM")
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