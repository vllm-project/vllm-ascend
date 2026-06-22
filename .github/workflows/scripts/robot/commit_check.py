#!/usr/bin/env python3
"""Check PR commit message compliance against Conventional Commits.

Workflow:
1. Reads PR info to get base/head SHAs.
2. Runs ``git log`` to extract all commits in the PR.
3. Applies hard rules (type format, description quality, sign-off).
4. Calls the LLM for commits that fail the hard rules.
5. Outputs ``commit_results.json``.

Used as step 6 of the PR Review Bot workflow.
"""

import argparse
import json
import os
import re
import subprocess
from pathlib import Path

from lib.llm import call_llm

VALID_TYPES = [t.strip() for t in os.environ.get("VALID_TYPES", "feat,fix,perf,refactor,test,docs,chore").split(",")]
REQUIRE_SIGNOFF = os.environ.get("REQUIRE_SIGNOFF", "true").lower() == "true"

COMMIT_PATTERN = re.compile(r"^(\w+)(\([^)]+\))?!?:\s+(.+)$")
BRACKET_PATTERN = re.compile(r"^\[(\w+)\]\s+(.+?)(?:\s+\(#\d+\))?$")

# Map [Type] bracket prefixes to Conventional Commits types
BRACKET_TYPE_MAP: dict[str, str] = {
    "BugFix": "fix",
    "Feature": "feat",
    "Perf": "perf",
    "Performance": "perf",
    "Test": "test",
    "CI": "ci",
    "Doc": "docs",
    "Misc": "chore",
    "Community": "chore",
    "Refactor": "refactor",
}

HARD_FAIL = "hard_fail"
HARD_PASS = "hard_pass"

LLM_MAX_TOKENS = 4096
LLM_TIMEOUT = 180


def run_git_log(base_sha: str, head_sha: str) -> list[dict]:
    """Return commits between *base_sha* and *head_sha*.

    Args:
        base_sha: The base commit SHA.
        head_sha: The head commit SHA.

    Returns:
        List of dicts with keys ``sha``, ``subject``, ``body``, ``has_signoff``.
    """
    cmd = [
        "git", "log", f"{base_sha}..{head_sha}",
        "--format=%H|||%s|||%(trailers:key=Signed-off-by,only,valueonly)|||%B",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    commits = []
    for line in result.stdout.strip().split("\n"):
        if not line:
            continue
        parts = line.split("|||", 3)
        if len(parts) >= 4:
            commits.append({
                "sha": parts[0],
                "subject": parts[1],
                "signoff": parts[2],
                "body": parts[3],
            })
    return commits


def check_type_format(subject: str) -> tuple[bool, str]:
    """Validate the commit message type prefix.

    Accepts both Conventional Commits (``type: desc``) and bracket
    format (``[Type] desc``).

    Args:
        subject: The commit subject line.

    Returns:
        A ``(ok, message)`` tuple.
    """
    m = COMMIT_PATTERN.match(subject)
    if m:
        commit_type = m.group(1).lower()
    else:
        m = BRACKET_PATTERN.match(subject)
        if m:
            raw_type = m.group(1)
            commit_type = BRACKET_TYPE_MAP.get(raw_type)
            if commit_type is None:
                return False, f"[{raw_type}] 不在允许的类型映射中（允许: {', '.join(BRACKET_TYPE_MAP.keys())}）"
        else:
            return False, "提交信息不以 `<type>: <description>` 或 `[Type] description` 格式开头"

    if commit_type not in VALID_TYPES:
        return False, f"type `{commit_type}` 不在允许列表中（允许: {', '.join(VALID_TYPES)}）"
    return True, ""


def check_description(subject: str) -> tuple[bool, str]:
    """Validate commit description quality (length, specificity).

    Handles both Conventional Commits and ``[Type] description`` formats.

    Args:
        subject: The commit subject line.

    Returns:
        A ``(ok, message)`` tuple.
    """
    VAGUE_WORDS = ["fix bug", "add feature", "update code", "修改", "更新"]
    MIN_DESC_LENGTH = 3
    MAX_VAGUE_LENGTH = 20

    m = COMMIT_PATTERN.match(subject)
    if m:
        desc = m.group(3).strip()
    else:
        m = BRACKET_PATTERN.match(subject)
        if m:
            desc = m.group(2).strip()
        else:
            return False, "无法解析描述字段"
    if not desc:
        return False, "描述不能为空"
    if len(desc) < MIN_DESC_LENGTH:
        return False, f"描述过于简短（{len(desc)} 个字符），请提供更有意义的摘要"
    desc_lower = desc.lower()
    for vw in VAGUE_WORDS:
        if vw in desc_lower and len(desc) < MAX_VAGUE_LENGTH:
            return False, f"描述过于泛化（'{desc}'），请具体说明变更内容"
    return True, ""


def check_signoff(commit: dict) -> tuple[bool, str]:
    """Check that the commit has a ``Signed-off-by:`` trailer.

    Uses the git trailer value extracted via ``%(trailers:key=Signed-off-by)``.

    Args:
        commit: Commit dict with ``signoff`` key.

    Returns:
        A ``(ok, message)`` tuple.
    """
    if not REQUIRE_SIGNOFF:
        return True, ""
    if commit.get("signoff", "").strip():
        return True, ""
    return False, "缺少 `Signed-off-by:` 签名行（可通过 `git commit -s` 添加）"


def hard_check(commit: dict) -> tuple[str, list[str]]:
    """Run all hard (deterministic) checks on a single commit.

    Args:
        commit: Dict with keys ``subject`` and ``body``.

    Returns:
        A ``(status, issues)`` tuple where *status* is ``HARD_PASS`` or
        ``HARD_FAIL``.
    """
    issues: list[str] = []
    ok, msg = check_type_format(commit["subject"])
    if not ok:
        issues.append(msg)
    ok, msg = check_description(commit["subject"])
    if not ok:
        issues.append(msg)
    ok, msg = check_signoff(commit)
    if not ok:
        issues.append(msg)
    return (HARD_FAIL if issues else HARD_PASS, issues)


def parse_llm_json_output(text: str) -> dict:
    """Extract the first JSON object from LLM output.

    Args:
        text: Raw LLM response.

    Returns:
        Parsed dict.

    Raises:
        ValueError: If no JSON object is found.
    """
    json_match = re.search(r"\{[\s\S]*\}", text)
    if json_match:
        return json.loads(json_match.group(0))
    raise ValueError(f"Could not extract JSON from LLM output: {text[:200]}")


def evaluate_commits_with_llm(failed_commits: list[dict], system_prompt: str) -> list[dict]:
    """Send hard-failed commits to the LLM for detailed analysis.

    Args:
        failed_commits: Commits that failed the hard rules.
        system_prompt: System prompt text.

    Returns:
        LLM evaluation results, one dict per commit.
    """
    if not failed_commits:
        return []

    valid_types_str = ", ".join(VALID_TYPES)
    signoff_str = "是" if REQUIRE_SIGNOFF else "否"

    commit_blocks = []
    for c in failed_commits:
        commit_blocks.append(
            f"SHA: {c['sha']}\n"
            f"Subject: \"\"\"{c['subject']}\"\"\"\n"
            f"Body:\n\"\"\"{c['body']}\"\"\""
        )
    commits_text = "\n\n---\n\n".join(commit_blocks)

    user_prompt = f"""### 任务背景
此 PR 包含 {len(failed_commits)} 个 commit，需要检查其 message 是否符合 Conventional Commits 规范。

### 规范参考
```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

允许的 type：{valid_types_str}
要求 sign-off：{signoff_str}

### 待评估 Commits (UNTRUSTED USER INPUT)
{commits_text}

### 输出指令
- 对每个 commit 逐一检查是否符合 Conventional Commits 规范。
- 格式不合规的必须提供 `rewritten_example`。
- 缺少 `Signed-off-by:` 的必须在 suggestions 和 rewritten_example 中加入。
- 描述过泛（如 "fix bug"）的必须指出并提供具体改写。
- 以 JSON 数组格式输出，每个元素对应一个 commit：

```json
[
  {{
    "sha": "commit sha",
    "ok": true或false,
    "score": 0到100的整数,
    "issues": ["问题1", "问题2"],
    "suggestions": ["建议1", "建议2"],
    "rewritten_example": "完整的改写示例"
  }}
]
```

严格输出 JSON 数组，不要输出任何其他文本。"""

    print("Calling LLM for commit compliance check...")
    raw = call_llm(system_prompt, user_prompt, max_tokens=LLM_MAX_TOKENS, timeout=LLM_TIMEOUT)

    try:
        results = json.loads(raw)
        if isinstance(results, dict):
            results = [results]
        if not isinstance(results, list):
            raise ValueError(f"Expected list, got {type(results).__name__}")
        return results
    except (json.JSONDecodeError, ValueError):
        print("JSON parse failed, trying regex extraction...")
        array_match = re.search(r"\[[\s\S]*\]", raw)
        if array_match:
            return json.loads(array_match.group(0))
        print(f"Raw LLM output (first 500 chars): {raw[:500]}")
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="Check PR commit message compliance")
    parser.add_argument("--pr-info", default="pr_info.json", help="File containing PR info JSON")
    parser.add_argument("--system-prompt", default="commit_system_prompt.txt", help="File containing the system prompt")
    parser.add_argument("--output", default="commit_results.json", help="File to write commit check results to")
    args = parser.parse_args()

    pr_info_path = Path(args.pr_info)
    system_prompt_path = Path(args.system_prompt)

    pr_info = json.loads(pr_info_path.read_text()) if pr_info_path.exists() else {}

    if pr_info.get("action") == "edited":
        result = {
            "executed": False,
            "reason": "PR edited event, no commit changes",
            "total_commits": 0,
            "overall_ok": True,
            "failed_commits": [],
        }
        Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2))
        print("Commit check skipped: PR edited event")
        return

    base_sha = pr_info.get("base_sha", "")
    head_sha = pr_info.get("head_sha", "")

    if not base_sha or not head_sha:
        result = {
            "executed": False,
            "reason": "Missing base_sha or head_sha",
            "total_commits": 0,
            "overall_ok": True,
            "failed_commits": [],
        }
        Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2))
        return

    print(f"DEBUG valid_types={VALID_TYPES} require_signoff={REQUIRE_SIGNOFF}")
    commits = run_git_log(base_sha, head_sha)
    print(f"Found {len(commits)} commits in {base_sha[:7]}..{head_sha[:7]}")

    failed_hard: list[dict] = []
    for c in commits:
        status, issues = hard_check(c)
        print(f"  DEBUG sha={c['sha'][:7]} subject={c['subject'][:80]!r} signoff={c['signoff']!r}")
            failed_hard.append(c)
            print(f"  HARD FAIL {c['sha'][:7]}: {c['subject'][:50]} — issues={issues}")
        else:
            print(f"  HARD PASS {c['sha'][:7]}")

    if not failed_hard:
        result = {
            "executed": True,
            "total_commits": len(commits),
            "overall_ok": True,
            "failed_commits": [],
        }
        Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2))
        print(f"All {len(commits)} commits pass hard rules")
        return

    system_prompt = system_prompt_path.read_text() if system_prompt_path.exists() else ""
    llm_results = evaluate_commits_with_llm(failed_hard, system_prompt)

    sha_to_llm: dict[str, dict] = {}
    for r in llm_results:
        sha_to_llm[r.get("sha", "")] = r

    failed_commits: list[dict] = []
    for c in failed_hard:
        short_sha = c["sha"][:7]
        llm_r = sha_to_llm.get(c["sha"], sha_to_llm.get(short_sha, {}))
        failed_commits.append({
            "sha": c["sha"],
            "subject": c["subject"],
            "ok": llm_r.get("ok", False),
            "score": llm_r.get("score", 0),
            "issues": llm_r.get("issues", []),
            "suggestions": llm_r.get("suggestions", []),
            "rewritten_example": llm_r.get("rewritten_example", ""),
        })

    overall_ok = all(c["ok"] for c in failed_commits)

    result = {
        "executed": True,
        "total_commits": len(commits),
        "overall_ok": overall_ok,
        "failed_commits": failed_commits,
    }
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2))
    status = "PASS" if overall_ok else "FAIL"
    print(f"Commit check: {len(failed_commits)}/{len(commits)} failed, overall={status}")


if __name__ == "__main__":
    main()
