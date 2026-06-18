#!/usr/bin/env python3
"""Step 5: Check PR commit message compliance.

1. Reads PR info to get base/head SHAs
2. Runs git log to extract all commits
3. Applies hard rules (type format, description, sign-off)
4. Calls LLM for commits that fail hard rules
5. Outputs commit_results.json
"""

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import requests

LLM_API_KEY = os.environ.get("LLM_API_KEY") or os.environ.get("VLLM_API_KEY", "")
LLM_BASE_URL = os.environ.get("LLM_BASE_URL") or os.environ.get("VLLM_BASE_URL", "")
LLM_MODEL = os.environ.get("LLM_MODEL", "default")
VALID_TYPES = os.environ.get("VALID_TYPES", "feat,fix,perf,refactor,test,docs,chore").split(",")
REQUIRE_SIGNOFF = os.environ.get("REQUIRE_SIGNOFF", "true").lower() == "true"

COMMIT_PATTERN = re.compile(r"^(\w+)(\([^)]+\))?!?:\s+(.+)$")

HARD_FAIL = "hard_fail"
HARD_PASS = "hard_pass"


def run_git_log(base_sha: str, head_sha: str) -> list[dict]:
    cmd = ["git", "log", f"{base_sha}..{head_sha}", "--format=%H|||%s|||%b"]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    commits = []
    for line in result.stdout.strip().split("\n"):
        if not line:
            continue
        parts = line.split("|||", 2)
        if len(parts) == 3:
            commits.append({"sha": parts[0], "subject": parts[1], "body": parts[2]})
    return commits


def check_type_format(subject: str) -> tuple[bool, str]:
    m = COMMIT_PATTERN.match(subject)
    if not m:
        return False, "提交信息不以 `<type>: <description>` 格式开头（如 feat: add feature）"
    commit_type = m.group(1).lower()
    if commit_type not in VALID_TYPES:
        return False, f"type `{commit_type}` 不在允许列表中（允许: {', '.join(VALID_TYPES)}）"
    return True, ""


def check_description(subject: str) -> tuple[bool, str]:
    m = COMMIT_PATTERN.match(subject)
    if not m:
        return False, "无法解析描述字段"
    desc = m.group(3).strip()
    if not desc:
        return False, "描述不能为空"
    if len(desc) < 3:
        return False, f"描述过于简短（{len(desc)} 个字符），请提供更有意义的摘要"
    vague_words = ["fix bug", "add feature", "update code", "修改", "更新"]
    desc_lower = desc.lower()
    for vw in vague_words:
        if vw in desc_lower and len(desc) < 20:
            return False, f"描述过于泛化（'{desc}'），请具体说明变更内容"
    return True, ""


def check_signoff(body: str) -> tuple[bool, str]:
    if not REQUIRE_SIGNOFF:
        return True, ""
    if "Signed-off-by:" in body:
        return True, ""
    return False, "缺少 `Signed-off-by:` 签名行（可通过 `git commit -s` 添加）"


def hard_check(commit: dict) -> tuple[str, list[str]]:
    issues = []
    ok, msg = check_type_format(commit["subject"])
    if not ok:
        issues.append(msg)
    ok, msg = check_description(commit["subject"])
    if not ok:
        issues.append(msg)
    ok, msg = check_signoff(commit["body"])
    if not ok:
        issues.append(msg)
    return (HARD_FAIL if issues else HARD_PASS, issues)


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
        "max_tokens": 4096,
    }
    resp = requests.post(
        f"{LLM_BASE_URL}/v1/chat/completions",
        headers=headers,
        json=payload,
        timeout=180,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def parse_llm_json_output(text: str) -> dict:
    json_match = re.search(r"\{[\s\S]*\}", text)
    if json_match:
        return json.loads(json_match.group(0))
    raise ValueError(f"Could not extract JSON from LLM output: {text[:200]}")


def evaluate_commits_with_llm(failed_commits: list[dict], system_prompt: str) -> list[dict]:
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
    raw = call_llm(system_prompt, user_prompt)

    try:
        results = json.loads(raw)
        if isinstance(results, dict):
            results = [results]
        if not isinstance(results, list):
            raise ValueError(f"Expected list, got {type(results)}")
        return results
    except (json.JSONDecodeError, ValueError):
        print(f"JSON parse failed, trying regex extraction...")
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

    commits = run_git_log(base_sha, head_sha)
    print(f"Found {len(commits)} commits in {base_sha[:7]}..{head_sha[:7]}")

    passed = []
    failed_hard = []
    for c in commits:
        status, issues = hard_check(c)
        if status == HARD_PASS:
            passed.append({"sha": c["sha"], "subject": c["subject"], "ok": True, "score": 100})
        else:
            failed_hard.append(c)
            print(f"  HARD FAIL {c['sha'][:7]}: {c['subject'][:50]}")

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

    sha_to_llm = {}
    for r in llm_results:
        sha_to_llm[r.get("sha", "")] = r

    failed_commits = []
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
