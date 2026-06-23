#!/usr/bin/env python3
"""Call the LLM for description completeness checking.

Reads system prompt, template, and issue content, calls the LLM endpoint,
parses the structured compliance result, and writes JSON to a file.

Used as step 4 of both the Issue Review and PR Review Bot workflows.
"""

import argparse
import json
import os
import re
from pathlib import Path

from lib.llm import call_llm
from lib.prefix_map import PREFIX_TO_TYPE_KEY

ISSUE_TITLE = os.environ["ISSUE_TITLE"]
ISSUE_BODY = os.environ.get("ISSUE_BODY", "")

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
    parser.add_argument("--kind", default="issue", choices=["issue", "pr"],
                        help="Target kind: issue or pr")
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

    template_label = "PR 模板" if args.kind == "pr" else "Issue 模板"

    user_prompt = f"""### 任务背景
目标类型：{args.kind}
描述类型：{type_key}

### 规范参考
详细描述规范（根据 {template_label} 中的必填字段判定）：
{template_text}

### 待评估数据 (UNTRUSTED USER INPUT)
标题：\"\"\"{ISSUE_TITLE}\"\"\"
提交的描述：
\"\"\"{ISSUE_BODY}\"\"\"

### 输出指令
- 根据系统提示中的评估准则进行判断，不要依赖模板中是否有显式"(必填)"标记。
- missing_items 列出实际缺失的关键信息（如"缺失环境信息""缺失错误日志""缺失复现步骤"）。
- suggestions 给出具体、可执行的改进建议，禁止使用"必填/必须"等表述。
- 若描述中已提供截图/图片，视为已提供日志相关信息，不得要求补充日志或要求转成文本。
- 硬件型号示例中不要出现 910/910B，统一使用 A3/A5。

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
            "summary": "解析 LLM 输出失败",
            "missing_items": ["LLM 输出格式异常，请联系管理员检查"],
            "suggestions": [],
        }

    output_path = Path(args.output)
    output_path.write_text(json.dumps(validated, ensure_ascii=False, indent=2))
    status = "PASS" if validated["ok"] else "FAIL"
    print(f"Review complete: ok={validated['ok']} score={validated['score']} status={status}")
    print(f"Result written to {output_path}")


if __name__ == "__main__":
    main()
