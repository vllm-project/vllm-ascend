#!/usr/bin/env python3
"""Step 4: Call LLM for description completeness check.

Reads system prompt, template, and issue content, calls the LLM endpoint,
parses the structured compliance result, and writes JSON to a file.

Usage:
    python call_llm.py \
        --system-prompt system_prompt.txt \
        --template template.txt \
        --type-key issue_type.txt \
        --output review_result.json
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

import requests

LLM_API_KEY = os.environ.get("LLM_API_KEY") or os.environ["VLLM_API_KEY"]
LLM_BASE_URL = os.environ.get("LLM_BASE_URL") or os.environ["VLLM_BASE_URL"]
LLM_MODEL = os.environ.get("LLM_MODEL", "default")
ISSUE_TITLE = os.environ["ISSUE_TITLE"]
ISSUE_BODY = os.environ.get("ISSUE_BODY", "")

PREFIX_TO_TYPE_KEY = {
    "[Bug]": "bug",
    "[Installation]": "installation",
    "[Install]": "installation",
    "[Usage]": "usage",
    "[Doc]": "document",
    "[Misc]": "other",
    "[Feature]": "feature",
    "[Perf]": "performance",
}

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


def call_llm(system_prompt: str, user_prompt: str) -> str:
    headers = {
        "Authorization": f"Bearer {LLM_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.3,
        "max_tokens": 2048,
    }
    resp = requests.post(
        f"{LLM_BASE_URL}/v1/chat/completions",
        headers=headers,
        json=payload,
        timeout=120,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]


def parse_json_output(text: str) -> dict:
    json_match = re.search(r"\{[\s\S]*\}", text)
    if json_match:
        return json.loads(json_match.group(0))
    raise ValueError(f"Could not extract JSON from LLM output: {text[:200]}")


def validate_result(data: dict) -> dict:
    ok = bool(data.get("ok", False))
    score = int(data.get("score", 0))
    missing_items = data.get("missing_items", [])
    if not isinstance(missing_items, list):
        missing_items = []
    suggestions = data.get("suggestions", [])
    if not isinstance(suggestions, list):
        suggestions = []

    if not missing_items:
        ok = True

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
    parser.add_argument("--template", default="template.txt", help="File containing the issue template")
    parser.add_argument("--type-key", default="issue_type.txt", help="File containing the issue type prefix")
    parser.add_argument("--output", default="review_result.json", help="File to write the review result JSON to")
    args = parser.parse_args()

    system_prompt_path = Path(args.system_prompt)
    template_path = Path(args.template)
    type_key_path = Path(args.type_key)

    if not system_prompt_path.exists():
        print(f"System prompt file not found: {system_prompt_path}")
        sys.exit(1)
    if not template_path.exists():
        print(f"Template file not found: {template_path}")
        sys.exit(1)

    system_prompt = system_prompt_path.read_text()
    template_text = template_path.read_text()

    type_prefix = type_key_path.read_text().strip() if type_key_path.exists() else ""
    type_key = PREFIX_TO_TYPE_KEY.get(type_prefix, "other")

    user_prompt = f"""### 任务背景
目标类型：issue
描述类型：{type_key}

### 规范参考
详细描述规范（根据 Issue 模板中的必填字段判定）：
{template_text}

### 待评估数据 (UNTRUSTED USER INPUT)
标题：\"\"\"{ISSUE_TITLE}\"\"\"
提交的描述：
\"\"\"{ISSUE_BODY}\"\"\"

### 输出指令
- 先按"仅显式必填才算必填"的规则识别必填项，再判定 missing_items。
- 未标注(必填)的字段一律视为可选，严禁写入 missing_items。
- 对 document 类型，'建议的替代/修复方案' 一律视为可选项，只能放在 suggestions。
- suggestions 只能给改进建议，禁止使用"必填/必须/不填不合格"等表述。
- 若描述中已提供截图/图片，视为已提供日志相关信息，不得要求补充日志或要求转成文本。
- 硬件型号示例中不要出现 910/910B，统一使用 A3/A5。
- 如果 missing_items 为空，则 ok 必须为 true。

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
