#!/usr/bin/env python3
"""Post the review comment and compute label actions.

Supports two modes inferred from the presence of commit results:

* **Issue mode** — reads ``description_result.json`` only, posts an
  issue-specific comment.
* **PR mode** — reads ``description_result.json`` + ``commit_results.json``,
  builds a combined PR review comment.

Outputs ``label_actions.json`` for the downstream label-management step.
"""

import argparse
import json
import os
from pathlib import Path

from lib.github_api import post_comment

ISSUE_NUMBER = os.environ["ISSUE_NUMBER"]

NEED_DETAIL_LABEL = os.environ.get("NEED_DETAIL_LABEL", "need-detail-desc")
NEED_COMMIT_FIX_LABEL = os.environ.get("NEED_COMMIT_FIX_LABEL", "need-commit-fix")


def build_desc_section(desc_result: dict | None) -> str:
    """Build the Markdown section for the description completeness check.

    Args:
        desc_result: The validated description check result, or ``None`` if
            the check was skipped.

    Returns:
        Markdown string.
    """
    lines = ["### 1. 描述完整性检查", ""]
    if desc_result is None:
        lines.append("本次未触发描述检查")
    elif desc_result.get("ok", False):
        lines.append(":white_check_mark: 描述信息完整，check pass")
        lines.append(f"_AI 综合评分：{desc_result.get('score', '-')}/100_")
    else:
        lines.append("描述信息不完整，为了帮助维护者更快定位和解决问题，请补充以下内容：")
        lines.append("")
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
    return "\n".join(lines)


def build_commit_section(commit_result: dict | None) -> str:
    """Build the Markdown section for the commit message compliance check.

    Args:
        commit_result: The commit check result, or ``None`` if skipped.

    Returns:
        Markdown string.
    """
    lines = ["### 2. Commit Message 合规性检查", ""]

    if commit_result is None or not commit_result.get("executed"):
        lines.append("本次未触发 commit log 检查")
        return "\n".join(lines)

    if commit_result.get("overall_ok", True):
        total = commit_result.get("total_commits", 0)
        lines.append(f":white_check_mark: 全部 {total} 个 commit message 合规，check pass")
        return "\n".join(lines)

    failed = commit_result.get("failed_commits", [])
    total = commit_result.get("total_commits", 0)
    lines.append(f"检测到 **{len(failed)} 个 commit**（共 {total} 个）存在格式问题，请修改后重新推送：")
    lines.append("")
    lines.append("#### 不合格 Commit 列表")
    lines.append("")

    for c in failed:
        short_sha = c.get("sha", "")[:7]
        lines.append(f"**commit `{short_sha}`**: {c.get('subject', '')}")
        for issue in c.get("issues", []):
            lines.append(f"- **问题：** {issue}")
        for s in c.get("suggestions", []):
            lines.append(f"- **建议：** {s}")
        example = c.get("rewritten_example", "")
        if example:
            lines.append("- **改写示例：**")
            lines.append("```")
            lines.append(example)
            lines.append("```")
        lines.append("")
        lines.append("---")
        lines.append("")

    lines.append("#### 提交信息规范速查")
    lines.append("")
    lines.append("提交信息必须遵循 Conventional Commits 格式：")
    lines.append("```")
    lines.append("<type>: <summary>")
    lines.append("")
    lines.append("<optional body>")
    lines.append("")
    lines.append("Signed-off-by: Your Name <your.email@example.com>")
    lines.append("```")
    lines.append("")
    lines.append("**允许的 type：** `feat`, `fix`, `perf`, `refactor`, `test`, `docs`, `chore`")
    lines.append("")
    lines.append("**Good：**")
    lines.append("```")
    lines.append("feat(npu): add flash attention support for Ascend CANN")
    lines.append("")
    lines.append("Signed-off-by: Your Name <your.email@example.com>")
    lines.append("```")
    lines.append("")
    lines.append("**Bad：** `fix bug` / `add feature` / `update code`")
    lines.append("")
    lines.append("> 修改 commit 可使用 `git rebase -i` 后 `git push --force-with-lease`。")

    return "\n".join(lines)


def build_issue_comment(desc_result: dict) -> str:
    """Build the issue-only review comment body.

    Args:
        desc_result: Validated description check result.

    Returns:
        Markdown string.
    """
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


def build_pr_combined_comment(desc_result: dict | None, commit_result: dict | None) -> str:
    """Build the combined PR review comment body.

    Args:
        desc_result: Description check result or ``None``.
        commit_result: Commit check result or ``None``.

    Returns:
        Markdown string.
    """
    sections = ["## PR Review 检查结果", "", ""]
    sections.append(build_desc_section(desc_result))
    sections.append("")
    sections.append("---")
    sections.append("")
    sections.append(build_commit_section(commit_result))
    sections.append("")
    sections.append("_内容由 AI 生成，请仔细甄别。_")
    return "\n".join(sections)


def compute_label_actions(desc_result: dict | None, commit_result: dict | None) -> dict:
    """Determine which labels to add and remove based on review results.

    Args:
        desc_result: Description check result or ``None``.
        commit_result: Commit check result or ``None``.

    Returns:
        Dict with keys ``add`` and ``remove`` (lists of label names).
    """
    add_labels: list[str] = []
    remove_labels: list[str] = []

    if desc_result:
        desc_executed = desc_result.get("executed", True)
        if desc_executed:
            if desc_result.get("ok", False):
                remove_labels.append(NEED_DETAIL_LABEL)
            else:
                add_labels.append(NEED_DETAIL_LABEL)

    if commit_result:
        commit_executed = commit_result.get("executed", False)
        if commit_executed:
            if commit_result.get("overall_ok", False):
                remove_labels.append(NEED_COMMIT_FIX_LABEL)
            else:
                add_labels.append(NEED_COMMIT_FIX_LABEL)

    return {"add": add_labels, "remove": remove_labels}


def main() -> None:
    parser = argparse.ArgumentParser(description="Handle review result and post comment")
    parser.add_argument("--input", default="review_result.json", help="Description result JSON")
    parser.add_argument("--commit-result", default=None, help="Commit check result JSON (PR mode)")
    parser.add_argument("--label-output", default="label_actions.json", help="File to write label actions JSON to")
    args = parser.parse_args()

    input_path = Path(args.input)
    desc_result = None
    if input_path.exists():
        desc_result = json.loads(input_path.read_text())

    commit_result = None
    commit_path = Path(args.commit_result) if args.commit_result else None
    if commit_path and commit_path.exists():
        commit_result = json.loads(commit_path.read_text())

    is_pr = commit_result is not None

    if is_pr:
        desc_ok = desc_result.get("ok", True) if desc_result else True
        commit_executed = commit_result.get("executed", False) if commit_result else False
        commit_ok = commit_result.get("overall_ok", True) if commit_result else True

        needs_comment = not desc_ok or (commit_executed and not commit_ok)

        if needs_comment:
            print("Posting PR combined review comment...")
            comment_body = build_pr_combined_comment(desc_result, commit_result)
            full_comment = (
                "> 这是由 PR Review Bot 自动生成的反馈。"
                "内容由 AI 生成，请仔细甄别。\n\n"
                + comment_body
            )
            post_comment(ISSUE_NUMBER, full_comment)
        else:
            print("PR review passed: no comment needed")
    else:
        if desc_result is None:
            print("No description result found, skipping")
            Path(args.label_output).write_text(json.dumps({"add": [], "remove": []}))
            return

        desc_ok = desc_result.get("ok", False)
        if not desc_ok:
            print(f"Posting issue description check comment... (score={desc_result.get('score', 0)})")
            comment_body = build_issue_comment(desc_result)
            full_comment = (
                "> 这是由 Issue 描述检查 Bot 自动生成的反馈。"
                "内容由 AI 生成，请仔细甄别。\n\n"
                + comment_body
            )
            post_comment(ISSUE_NUMBER, full_comment)
        else:
            print("Issue description check passed, no comment needed")

    actions = compute_label_actions(desc_result, commit_result)
    Path(args.label_output).write_text(json.dumps(actions, ensure_ascii=False, indent=2))
    print(f"Label actions: {actions}")


if __name__ == "__main__":
    main()
