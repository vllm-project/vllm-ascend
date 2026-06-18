#!/usr/bin/env python3
"""Step 5: Post the description check result as a GitHub issue comment.

Reads the review result JSON, posts a structured comment if the description
is incomplete, and writes a label action file for the next step.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import requests

GITHUB_TOKEN = os.environ["GITHUB_TOKEN"]
REPO = os.environ["REPO"]
ISSUE_NUMBER = os.environ["ISSUE_NUMBER"]


def post_comment(body: str) -> None:
    url = f"https://api.github.com/repos/{REPO}/issues/{ISSUE_NUMBER}/comments"
    headers = {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json",
    }
    resp = requests.post(url, headers=headers, json={"body": body}, timeout=30)
    resp.raise_for_status()
    print(f"Comment posted to issue #{ISSUE_NUMBER}")


def build_comment(result: dict) -> str:
    missing_items = result.get("missing_items", [])
    suggestions = result.get("suggestions", [])

    lines = [
        "### 描述完整性检查结果",
        "",
        "您提交的 Issue 描述信息不完整，为了帮助维护者更快定位和解决问题，请补充以下内容：",
        "",
    ]

    if missing_items:
        lines.append("**缺失项：**")
        for item in missing_items:
            lines.append(f"- {item}")
        lines.append("")

    if suggestions:
        lines.append("**改进建议：**")
        for item in suggestions:
            lines.append(f"- {item}")
        lines.append("")

    lines.extend([
        "_内容由 AI 生成，请仔细甄别。_",
        f"_AI 综合评分：{result.get('score', '-')}/100_",
    ])

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Handle description check result")
    parser.add_argument("--input", default="review_result.json", help="File containing the review result JSON")
    parser.add_argument("--label-output", default="label_action.txt", help="File to write label action to")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Input file not found: {input_path}")
        sys.exit(1)

    result = json.loads(input_path.read_text())
    ok = result.get("ok", False)
    score = result.get("score", 0)

    label_action = "remove" if ok else "add"
    Path(args.label_output).write_text(label_action)

    if ok:
        print(f"Issue description check passed: score={score}, no comment needed")
        return

    print(f"Issue description check failed: score={score}, posting comment...")
    comment_body = build_comment(result)

    full_comment = (
        "> 这是由 Issue 描述检查 Bot 自动生成的反馈。"
        "内容由 AI 生成，请仔细甄别。\n\n"
        + comment_body
    )
    post_comment(full_comment)
    print("Done.")


if __name__ == "__main__":
    main()
