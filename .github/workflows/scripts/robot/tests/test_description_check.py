#!/usr/bin/env python3
"""End-to-end test for the Issue/PR Description & Commit Check bots.

Supports three modes:
  1. Default (csv): load test cases from CSV, create issues, verify results
  2. --demo: run built-in issue demo tests
  3. --mode pr: run PR tests using git worktrees

Usage:
    export GITHUB_TOKEN=$(gh auth token)

    # Issue tests from CSV
    python test_description_check.py --repo owner/repo

    # Issue demo tests
    python test_description_check.py --repo owner/repo --demo

    # PR tests via worktrees
    python test_description_check.py --repo owner/repo --mode pr
"""

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import requests

GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")
HEADERS = {
    "Authorization": f"Bearer {GITHUB_TOKEN}",
    "Accept": "application/vnd.github+json",
}

LABEL_NAME = "need-detail-desc"
NEED_COMMIT_FIX_LABEL = "need-commit-fix"
POLL_INTERVAL = 15
MAX_WAIT = 180
DEFAULT_CSV = Path(__file__).resolve().parent / "description_check_cases.csv"

WORKTREE_BASE = Path("/tmp/opencode/pr-tests")


class Result:
    def __init__(self, name: str):
        self.name = name
        self.passed = False
        self.error = ""
        self.details = ""

    def fail(self, msg: str):
        self.error = msg

    def ok(self, msg: str = ""):
        self.passed = True
        self.details = msg


def api_request(method: str, url: str, **kwargs):
    last_error = None
    for attempt in range(3):
        try:
            resp = requests.request(method, url, headers=HEADERS, timeout=30, **kwargs)
            resp.raise_for_status()
            return resp
        except (requests.ConnectionError, requests.Timeout) as e:
            last_error = e
            if attempt < 2:
                time.sleep(5 * (attempt + 1))
        except requests.HTTPError:
            raise
    raise last_error


def api_get(url: str) -> dict:
    return api_request("GET", url).json()


def api_post(url: str, data: dict) -> dict:
    return api_request("POST", url, json=data).json()


def api_patch(url: str, data: dict) -> dict:
    return api_request("PATCH", url, json=data).json()


def api_delete(url: str) -> None:
    api_request("DELETE", url)


def api_close_issue(api_base: str, number: int):
    api_patch(f"{api_base}/issues/{number}", {"state": "closed"})


def create_issue(api_base: str, title: str, body: str) -> int:
    data = api_post(f"{api_base}/issues", {"title": title, "body": body})
    return data["number"]


def edit_issue(api_base: str, number: int, title: str, body: str):
    api_patch(f"{api_base}/issues/{number}", {"title": title, "body": body})


def get_labels(api_base: str, number: int) -> list[str]:
    data = api_get(f"{api_base}/issues/{number}")
    return [lb["name"] for lb in data.get("labels", [])]


def get_bot_comments(api_base: str, number: int) -> list[dict]:
    comments = api_get(f"{api_base}/issues/{number}/comments")
    return [c for c in comments if c.get("user", {}).get("login", "").endswith("[bot]")]


def remove_label(api_base: str, number: int, label: str):
    try:
        api_delete(f"{api_base}/issues/{number}/labels/{label}")
    except requests.HTTPError:
        pass


def wait_for_bot_comment(api_base: str, number: int, max_wait: int = MAX_WAIT) -> tuple[list[str], list[dict]]:
    start = time.time()
    while time.time() - start < max_wait:
        labels = get_labels(api_base, number)
        bot_comments = get_bot_comments(api_base, number)
        if bot_comments:
            return labels, bot_comments
        elapsed = int(time.time() - start)
        print(f"  Waiting for bot comment... ({elapsed}s elapsed)")
        time.sleep(POLL_INTERVAL)
    return get_labels(api_base, number), get_bot_comments(api_base, number)


def wait_then_check(api_base: str, number: int, wait_sec: int = 90) -> tuple[list[str], list[dict]]:
    for remaining in range(wait_sec, 0, -POLL_INTERVAL):
        elapsed = wait_sec - remaining
        print(f"  Waiting for bot to process... ({elapsed}s elapsed)")
        time.sleep(min(POLL_INTERVAL, remaining))
    return get_labels(api_base, number), get_bot_comments(api_base, number)


# ── Issue Tests ──────────────────────────────────────────────

def load_csv_cases(csv_path: str) -> list[dict]:
    cases = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["expect_ok"] = row.get("expect_ok", "").strip().lower() == "true"
            row["expect_label"] = row.get("expect_label", "").strip().lower() == "true"
            row["expect_comment"] = row.get("expect_comment", "").strip().lower() == "true"
            row["body"] = (row.get("body") or "").replace("\\n", "\n")
            cases.append(row)
    return cases


def verify_issue_case(api_base: str, case: dict) -> Result:
    title = case["title"]
    body = case.get("body", "")
    expect_label = case.get("expect_label", False)
    expect_comment = case.get("expect_comment", False)
    notes = case.get("notes", "")

    r = Result(f"{title} ({notes})" if notes else title)
    issue_number = None

    try:
        issue_number = create_issue(api_base, title, body)
        print(f"  Created issue #{issue_number}")

        if expect_comment:
            labels, bot_comments = wait_for_bot_comment(api_base, issue_number)
        else:
            labels, bot_comments = wait_then_check(api_base, issue_number)

        has_label = LABEL_NAME in labels
        has_comment = len(bot_comments) > 0

        checks = []
        if has_label != expect_label:
            checks.append(f"label: expected={expect_label} got={has_label} (labels={labels})")
        if has_comment != expect_comment:
            checks.append(f"comment: expected={expect_comment} got={has_comment}")

        if checks:
            r.fail("; ".join(checks))
        else:
            r.ok(f"#{issue_number} label={has_label} comment={has_comment}")

        return r
    except Exception as e:
        r.fail(str(e))
        return r
    finally:
        if issue_number:
            try:
                api_close_issue(api_base, issue_number)
            except Exception:
                pass


def run_issue_csv_tests(api_base: str, csv_path: str):
    cases = load_csv_cases(csv_path)
    print(f"Loaded {len(cases)} test cases from {csv_path}")
    print()

    results = []
    for i, case in enumerate(cases):
        label = f"[{i + 1}/{len(cases)}]"
        print(f"{label} {case['title']}")
        if case.get("notes"):
            print(f"     Notes: {case['notes']}")
        result = verify_issue_case(api_base, case)
        results.append(result)
        status = "PASS" if result.passed else f"FAIL: {result.error}"
        print(f"     {status}")
        print()
    return results


def run_issue_demo_tests(api_base: str) -> list[Result]:
    def test_incomplete() -> Result:
        r = Result("Demo: Incomplete bug -> label + comment")
        n = None
        try:
            n = create_issue(api_base, "[Bug]: 调用接口报错", "有个错误，帮忙看看")
            print(f"  Created issue #{n}")
            labels, comments = wait_for_bot_comment(api_base, n)
            if LABEL_NAME not in labels:
                r.fail(f"Expected label not found. Labels: {labels}")
            elif not comments:
                r.fail("Expected bot comment but none found")
            else:
                r.ok(f"Issue #{n}: label added, comment posted")
            return r
        except Exception as e:
            r.fail(str(e))
            return r
        finally:
            if n:
                try:
                    api_close_issue(api_base, n)
                except Exception:
                    pass

    def test_complete() -> Result:
        r = Result("Demo: Complete bug -> no label, no comment")
        n = None
        try:
            n = create_issue(
                api_base,
                "[Bug]: A5 CANN 8.2 conv2d 算子报错",
                "### 环境信息\n- Ubuntu 22.04\n- A5\n- CANN 8.2.0\n### 问题描述\nconv2d RuntimeError",
            )
            print(f"  Created issue #{n}")
            labels, comments = wait_then_check(api_base, n)
            if comments:
                r.fail(f"Unexpected bot comment: {comments[0]['body'][:200]}")
            else:
                r.ok(f"Issue #{n}: clean pass")
            return r
        except Exception as e:
            r.fail(str(e))
            return r
        finally:
            if n:
                try:
                    api_close_issue(api_base, n)
                except Exception:
                    pass

    def test_edit() -> Result:
        r = Result("Demo: Edit vague->complete -> label removed")
        n = None
        try:
            n = create_issue(api_base, "[Bug]: 安装失败", "装不上，报错了")
            print(f"  Created issue #{n}")
            labels, _ = wait_for_bot_comment(api_base, n)
            if LABEL_NAME not in labels:
                r.fail(f"Phase 1: Expected label not found. Labels: {labels}")
                return r
            print("  Phase 1: label added")
            time.sleep(5)
            edit_issue(api_base, n, "[Bug]: Ubuntu 22.04 CANN 8.2 安装失败",
                       "### 环境信息\n- Ubuntu 22.04\n- CANN 8.2.0\n### 问题描述\npip install ERROR")
            print(f"  Edited issue #{n}")
            labels2, _ = wait_for_bot_comment(api_base, n, max_wait=120)
            if LABEL_NAME in labels2:
                r.fail(f"Phase 2: Label should be removed. Labels: {labels2}")
            else:
                r.ok(f"Issue #{n}: label added then removed")
            return r
        except Exception as e:
            r.fail(str(e))
            return r
        finally:
            if n:
                try:
                    api_close_issue(api_base, n)
                except Exception:
                    pass

    tests = [test_incomplete, test_complete, test_edit]
    results = []
    for i, fn in enumerate(tests):
        print(f"[{i + 1}/3] {fn.__doc__}")
        r = fn()
        results.append(r)
        print(f"     {'PASS' if r.passed else 'FAIL: ' + r.error}")
        print()
    return results


# ── PR Tests (via git worktree) ──────────────────────────────

def run_cmd(cmd: list[str], cwd: str | None = None) -> str:
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\n{result.stderr}")
    return result.stdout.strip()


def create_pr_via_worktree(api_base: str, repo: str, case_name: str, title: str, body: str,
                           commits: list[tuple[str, str, bool]]) -> int:
    """Create a worktree, make commits, push branch, create PR. Returns PR number."""
    import uuid
    branch_name = f"test/bot-{case_name.replace(' ', '-').replace('_', '-')[:30]}"
    worktree_path = WORKTREE_BASE / branch_name
    tmp_branch = f"test/tmp-{uuid.uuid4().hex[:8]}"

    try:
        run_cmd(["git", "worktree", "remove", str(worktree_path), "--force"])
    except Exception:
        pass
    if worktree_path.exists():
        shutil.rmtree(worktree_path)

    run_cmd(["git", "branch", tmp_branch, "main"])
    run_cmd(["git", "worktree", "add", str(worktree_path), tmp_branch])
    print(f"  Worktree created: {worktree_path}")

    commit_shas = []
    for subject, commit_body, signed_off in commits:
        cmd = ["git", "commit", "--allow-empty", "-m", subject]
        if commit_body:
            cmd.extend(["-m", commit_body])
        if signed_off:
            cmd.append("-s")
        run_cmd(cmd, cwd=str(worktree_path))
        sha = run_cmd(["git", "rev-parse", "HEAD"], cwd=str(worktree_path))[:7]
        sig = "(signed)" if signed_off else "(unsigned)"
        print(f"    Commit {sha} {sig}: {subject}")
        commit_shas.append(sha)

    run_cmd(["git", "push", "origin", f"HEAD:{branch_name}", "--force"], cwd=str(worktree_path))
    print(f"  Push to {branch_name}")

    pr_data = api_post(f"{api_base}/pulls", {
        "title": title,
        "body": body,
        "head": branch_name,
        "base": "main",
    })
    pr_number = pr_data["number"]
    print(f"  PR created: #{pr_number}")
    return pr_number


def cleanup_worktree(branch_name: str, pr_number: int, api_base: str):
    worktree_path = WORKTREE_BASE / branch_name
    if worktree_path.exists():
        shutil.rmtree(worktree_path)
    try:
        run_cmd(["git", "worktree", "prune"])
    except Exception:
        pass
    try:
        api_close_issue(api_base, pr_number)
    except Exception:
        pass
    try:
        run_cmd(["git", "push", "origin", "--delete", branch_name])
    except Exception:
        pass
    try:
        run_cmd(["git", "branch", "-D", branch_name])
    except Exception:
        pass
    try:
        result = subprocess.run(["git", "branch"], capture_output=True, text=True)
        for line in result.stdout.split("\n"):
            line = line.strip().lstrip("* ")
            if line.startswith("test/tmp-"):
                run_cmd(["git", "branch", "-D", line])
    except Exception:
        pass


def verify_pr(api_base: str, repo: str, case_name: str, title: str, body: str,
              commits: list[tuple[str, str, bool]],
              expect_desc_ok: bool, expect_commit_ok: bool) -> Result:
    r = Result(f"PR: {case_name}")
    branch_name = f"test/bot-{case_name.replace(' ', '-').replace('_', '-')[:30]}"
    pr_number = None

    try:
        pr_number = create_pr_via_worktree(api_base, repo, case_name, title, body, commits)

        labels, bot_comments = wait_for_bot_comment(api_base, pr_number, max_wait=240)
        has_comment = len(bot_comments) > 0

        has_desc_label = LABEL_NAME in labels
        has_commit_label = NEED_COMMIT_FIX_LABEL in labels

        checks = []
        if has_commit_label != (not expect_commit_ok):
            checks.append(f"commit label: expected={not expect_commit_ok} got={has_commit_label}")

        if has_comment and bot_comments:
            body_text = bot_comments[0]["body"]
            if "Commit Message" not in body_text:
                checks.append("comment missing commit check section")
            if "描述完整性检查" not in body_text:
                checks.append("comment missing description check section")
        elif not has_comment and (not expect_desc_ok or not expect_commit_ok):
            checks.append("expected comment but none found")

        if checks:
            r.fail("; ".join(checks))
        else:
            detail = f"#{pr_number} labels={labels} comment={has_comment}"
            r.ok(detail)

        return r
    except Exception as e:
        r.fail(str(e))
        return r
    finally:
        if pr_number:
            try:
                cleanup_worktree(branch_name, pr_number, api_base)
            except Exception:
                pass


def run_pr_tests(api_base: str, repo: str) -> list[Result]:
    """Run PR bot test cases."""
    print("Running PR review bot tests (via git worktrees)")
    print(f"Worktree dir: {WORKTREE_BASE}")
    print()

    tests = []

    # Test 1: All good → no labels, no comment, event=opened
    tests.append({
        "case_name": "1-all-good",
        "title": "[Feat] add optimised memory allocator for NPU",
        "body": "## Summary\n\nImplements a new memory allocator that reduces fragmentation by 40%.\n\n## Test plan\n- Unit tests added\n- Tested on A5\n- Benchmark shows 15% throughput improvement",
        "commits": [
            ("feat(npu): add optimised memory allocator", "Reduces fragmentation by 40%\n\nSigned-off-by: Dev <dev@test.com>", True),
            ("test: add unit tests for memory allocator", "Covers allocation and deallocation\n\nSigned-off-by: Dev <dev@test.com>", True),
        ],
        "expect_desc_ok": True,
        "expect_commit_ok": True,
    })

    # Test 2: Bad commits → need-commit-fix label, event=opened
    tests.append({
        "case_name": "2-bad-commits",
        "title": "[BugFix] resolve memory leak in tensor pool manager",
        "body": "## Summary\n\nFixes a memory leak in the tensor pool manager. The pool was not releasing freed blocks properly.\n\n## Test plan\n- Verified with valgrind on A5\n- Ran 1000 iterations without growth",
        "commits": [
            ("fix: resolve memory leak in tensor pool", "The pool was not releasing freed blocks\n\nSigned-off-by: Dev <dev@test.com>", True),
            ("fix bug", "", False),
            ("update code", "", False),
        ],
        "expect_desc_ok": True,
        "expect_commit_ok": False,
    })

    # Test 3: Bad desc + bad commits → both labels, event=opened
    tests.append({
        "case_name": "3-bad-desc-and-commits",
        "title": "[Bug]: 报错了",
        "body": "帮我看看",
        "commits": [
            ("wip", "", False),
        ],
        "expect_desc_ok": False,
        "expect_commit_ok": False,
    })

    # Test 4: Empty PR desc (title self-explanatory) + good commits → passes
    tests.append({
        "case_name": "4-empty-desc-good-title",
        "title": "[CI] Fix GitHub Actions yaml indentation in release workflow",
        "body": "",
        "commits": [
            ("fix(ci): correct yaml indentation in release workflow", "The release workflow had incorrect indentation causing parse errors\n\nSigned-off-by: Dev <dev@test.com>", True),
        ],
        "expect_desc_ok": True,
        "expect_commit_ok": True,
    })

    results = []
    for i, tc in enumerate(tests):
        case_name = tc["case_name"]
        print(f"[{i + 1}/{len(tests)}] {case_name}")
        print(f"  Title: {tc['title']}")
        print(f"  Commits: {len(tc['commits'])}")
        print(f"  Expect: desc_ok={tc['expect_desc_ok']} commit_ok={tc['expect_commit_ok']}")

        result = verify_pr(
            api_base, repo, case_name,
            tc["title"], tc["body"], tc["commits"],
            tc["expect_desc_ok"], tc["expect_commit_ok"],
        )
        results.append(result)
        status = "PASS" if result.passed else f"FAIL: {result.error}"
        print(f"  {status}")
        print()
        time.sleep(5)

    return results


# ── Main ─────────────────────────────────────────────────────

def print_summary(results: list[Result]):
    print("=" * 60)
    passed = sum(1 for r in results if r.passed)
    print(f"Results: {passed}/{len(results)} passed")
    for r in results:
        icon = "PASS" if r.passed else "FAIL"
        print(f"  [{icon}] {r.name}")
        if r.details:
            print(f"         {r.details}")
        if r.error:
            print(f"         Error: {r.error}")
    if passed != len(results):
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Test Issue/PR Description & Commit Check bots")
    parser.add_argument("--repo", default="ai-infra-develop/vllm-ascend", help="GitHub repo (owner/name)")
    parser.add_argument("--csv", default=None, help="Path to CSV test cases (issue mode)")
    parser.add_argument("--demo", action="store_true", help="Run built-in issue demo tests")
    parser.add_argument("--mode", default="issue", choices=["issue", "pr"], help="Test mode: issue or pr")
    args = parser.parse_args()

    if not GITHUB_TOKEN:
        print("ERROR: GITHUB_TOKEN environment variable not set")
        sys.exit(1)

    api_base = f"https://api.github.com/repos/{args.repo}"
    print(f"Repo: {args.repo}")
    print(f"Mode: {args.mode}")
    print()

    if args.mode == "pr":
        WORKTREE_BASE.mkdir(parents=True, exist_ok=True)
        results = run_pr_tests(api_base, args.repo)
    elif args.demo:
        results = run_issue_demo_tests(api_base)
    else:
        csv_path = args.csv or str(DEFAULT_CSV)
        if not Path(csv_path).exists():
            print(f"ERROR: CSV not found: {csv_path}")
            sys.exit(1)
        results = run_issue_csv_tests(api_base, csv_path)

    print_summary(results)


if __name__ == "__main__":
    main()
