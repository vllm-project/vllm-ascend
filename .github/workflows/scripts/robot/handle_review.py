#!/usr/bin/env python3
"""Post the review comment and compute label actions.

Supports two modes via ``--kind``:

* ``issue`` — posts issue-specific comment, manages ``need-detail-desc`` label.
* ``pr`` — posts PR-specific comment, manages ``need-detail-desc`` label.

Outputs ``label_actions.json`` for the downstream label-management step.
"""

import argparse
import json
import os
from pathlib import Path

from lib.github_api import get_labels, post_comment

ISSUE_NUMBER = os.environ["ISSUE_NUMBER"]
NEED_DETAIL_LABEL = os.environ.get("NEED_DETAIL_LABEL", "need-detail-desc")


def build_issue_comment(desc_result: dict) -> str:
    """Build the issue review failure comment body."""
    missing_items = desc_result.get("missing_items", [])
    suggestions = desc_result.get("suggestions", [])

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
        f"_AI 综合评分：{desc_result.get('score', '-')}/100_",
    ])
    return "\n".join(lines)


def build_issue_pass_comment(desc_result: dict) -> str:
    """Build a pass-variant issue review comment for flagged→clean transitions."""
    return (
        "### 描述完整性检查结果 (已通过)\n\n"
        ":white_check_mark: 描述信息已完善，感谢您的更新！\n\n"
        "_内容由 AI 生成，请仔细甄别。_"
    )


def build_pr_comment(desc_result: dict) -> str:
    """Build the PR review failure comment body."""
    lines = [
        "## PR Review 检查结果",
        "",
        "### 描述完整性检查",
        "",
        "描述信息不完整，为了帮助维护者更快定位和解决问题，请补充以下内容：",
        "",
    ]
    missing = desc_result.get("missing_items", [])
    if missing:
        lines.append("**缺失项：**")
        for item in missing:
            lines.append(f"- {item}")
        lines.append("")
    suggestions = desc_result.get("suggestions", [])
    if suggestions:
        lines.append("**改进建议：**")
        for item in suggestions:
            lines.append(f"- {item}")
        lines.append("")
    lines.append(f"_AI 综合评分：{desc_result.get('score', '-')}/100_")
    lines.append("")
    lines.append("_内容由 AI 生成，请仔细甄别。_")
    return "\n".join(lines)


def build_pr_pass_comment(desc_result: dict) -> str:
    """Build a pass-variant PR review comment for flagged→clean transitions."""
    return (
        "## PR Review 检查结果 (已通过)\n\n"
        ":white_check_mark: 描述完整性检查已通过，感谢您的改进！\n\n"
        "_内容由 AI 生成，请仔细甄别。_"
    )


def compute_label_actions(desc_result: dict | None) -> dict:
    """Determine which labels to add and remove."""
    add_labels: list[str] = []
    remove_labels: list[str] = []

    if desc_result:
        desc_executed = desc_result.get("executed", True)
        if desc_executed:
            if desc_result.get("ok", False):
                remove_labels.append(NEED_DETAIL_LABEL)
            else:
                add_labels.append(NEED_DETAIL_LABEL)

    return {"add": add_labels, "remove": remove_labels}


def main() -> None:
    parser = argparse.ArgumentParser(description="Handle review result and post comment")
    parser.add_argument("--input", default="review_result.json", help="Description result JSON")
    parser.add_argument("--label-output", default="label_actions.json", help="File to write label actions JSON to")
    parser.add_argument("--kind", default="issue", choices=["issue", "pr"],
                        help="Target kind: issue or pr")
    args = parser.parse_args()

    input_path = Path(args.input)
    desc_result = json.loads(input_path.read_text()) if input_path.exists() else None

    if desc_result is None:
        print("No description result found, skipping")
        Path(args.label_output).write_text(json.dumps({"add": [], "remove": []}))
        return

    current_labels = get_labels(ISSUE_NUMBER)
    previously_flagged = NEED_DETAIL_LABEL in current_labels

    desc_executed = desc_result.get("executed", True)
    desc_ok = desc_result.get("ok", False)

    if not desc_ok:
        print(f"Posting review comment... (score={desc_result.get('score', 0)})")
        if args.kind == "pr":
            prefix = "> 这是由 PR Review Bot 自动生成的反馈。内容由 AI 生成，请仔细甄别。\n\n"
            body = build_pr_comment(desc_result)
        else:
            prefix = "> 这是由 Issue 描述检查 Bot 自动生成的反馈。内容由 AI 生成，请仔细甄别。\n\n"
            body = build_issue_comment(desc_result)
        post_comment(ISSUE_NUMBER, prefix + body)
    elif desc_executed and previously_flagged:
        print("Posting pass comment...")
        body = build_pr_pass_comment(desc_result) if args.kind == "pr" else build_issue_pass_comment(desc_result)
        post_comment(ISSUE_NUMBER, body)
    else:
        print("Review passed: no comment needed")

    actions = compute_label_actions(desc_result)
    Path(args.label_output).write_text(json.dumps(actions, ensure_ascii=False, indent=2))
    print(f"Label actions: {actions}")


if __name__ == "__main__":
    main()
