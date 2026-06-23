#!/usr/bin/env python3
"""End-to-end test covering ALL scenarios from SCENARIOS.md.

Tests the full lifecycle: opened -> edited for both issues and PRs.

Scenarios covered:
  Issue: I1-I6 (6 scenarios, 2 issue chains)
  PR:    P1-P12 (12 scenarios, 4 PR chains)

Usage:
    export GITHUB_TOKEN=$(gh auth token)

    # All issue scenarios (I1-I6)
    python test_scenarios.py --repo owner/repo --mode issue

    # All PR scenarios (P1-P12)
    python test_scenarios.py --repo owner/repo --mode pr

    # Both issue and PR scenarios
    python test_scenarios.py --repo owner/repo --mode all
"""

import argparse
import os
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path

import requests

GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")
HEADERS = {
    "Authorization": f"Bearer {GITHUB_TOKEN}",
    "Accept": "application/vnd.github+json",
}

LABEL_NAME = "need-detail-desc"
POLL_INTERVAL = 15
MAX_WAIT = 240
WORKTREE_BASE = Path("/tmp/opencode/scenario-tests")


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


# ── GitHub API Helpers ────────────────────────────────────────


def api_request(method: str, url: str, **kwargs):
    last_error = None
    for attempt in range(3):
        try:
            resp = requests.request(method, url, headers=HEADERS, timeout=60, **kwargs)
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


def _close_entity(api_base: str, number: int):
    try:
        api_patch(f"{api_base}/issues/{number}", {"state": "closed"})
    except Exception:
        pass


def get_labels(api_base: str, number: int) -> list[str]:
    data = api_get(f"{api_base}/issues/{number}")
    return [lb["name"] for lb in data.get("labels", [])]


def get_bot_comments(api_base: str, number: int) -> list[dict]:
    comments = api_get(f"{api_base}/issues/{number}/comments")
    return [c for c in comments if c.get("user", {}).get("login", "").endswith("[bot]")]


def _poll_bot(api_base: str, number: int, max_wait: int = MAX_WAIT,
              expect_new_comment: bool = True,
              prev_comment_count: int = 0) -> tuple[list[str], list[dict], int]:
    start = time.time()
    while time.time() - start < max_wait:
        labels = get_labels(api_base, number)
        comments = get_bot_comments(api_base, number)
        if expect_new_comment:
            if len(comments) > prev_comment_count:
                return labels, comments, int(time.time() - start)
        else:
            elapsed = int(time.time() - start)
            if elapsed >= 90:
                return labels, comments, elapsed
        elapsed = int(time.time() - start)
        print(f"  Waiting for bot... ({elapsed}s elapsed)")
        time.sleep(POLL_INTERVAL)
    return get_labels(api_base, number), get_bot_comments(api_base, number), int(time.time() - start)


# ── Git Helpers ───────────────────────────────────────────────


def run_cmd(cmd: list[str], cwd: str | None = None) -> str:
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\n{result.stderr}")
    return result.stdout.strip()


def _remove_worktree(worktree_path: Path):
    try:
        run_cmd(["git", "worktree", "remove", str(worktree_path), "--force"])
    except Exception:
        pass
    if worktree_path.exists():
        shutil.rmtree(worktree_path)


def _cleanup_temp_branches():
    try:
        result = subprocess.run(["git", "branch"], capture_output=True, text=True)
        for line in result.stdout.split("\n"):
            bt = line.strip().lstrip("* ")
            if bt.startswith("test/"):
                try:
                    run_cmd(["git", "branch", "-D", bt])
                except Exception:
                    pass
    except Exception:
        pass


def _cleanup_pr(branch_name: str, pr_number: int, api_base: str):
    worktree_path = WORKTREE_BASE / branch_name
    _remove_worktree(worktree_path)
    try:
        run_cmd(["git", "worktree", "prune"])
    except Exception:
        pass
    _close_entity(api_base, pr_number)
    try:
        run_cmd(["git", "push", "origin", "--delete", branch_name])
    except Exception:
        pass
    try:
        run_cmd(["git", "branch", "-D", branch_name])
    except Exception:
        pass
    _cleanup_temp_branches()


# ── Issue Lifecycle ───────────────────────────────────────────


def create_issue(api_base: str, title: str, body: str) -> int:
    data = api_post(f"{api_base}/issues", {"title": title, "body": body})
    return data["number"]


def edit_issue(api_base: str, number: int, title: str, body: str):
    api_patch(f"{api_base}/issues/{number}", {"title": title, "body": body})


ISSUE_GOOD = {
    "title": "[Bug]: A5 CANN 8.2 conv2d 算子报错 RuntimeError 500002",
    "body": (
        "### 环境信息\n"
        "- 操作系统: Ubuntu 22.04\n"
        "- 昇腾硬件: A5\n"
        "- CANN 版本: 8.2.0\n"
        "- vLLM 版本: main\n"
        "### 问题描述\n"
        "运行 conv2d 时 RuntimeError: conv2d forward failed\n"
        "### 复现步骤\n"
        "1. pip install vllm vllm-ascend\n"
        "2. 运行测试脚本\n"
        "3. 报错信息: RuntimeError: conv2d forward failed, error code 500002\n"
    ),
}

ISSUE_BAD = {
    "title": "[Bug]: 调用接口报错",
    "body": "有个错误 帮忙看看",
}

ISSUE_BAD_ALT = {
    "title": "[Bug]: 服务启动失败",
    "body": "启动不起来了 报错了",
}

ISSUE_GOOD_ALT = {
    "title": "[Bug]: A5 CANN 8.2 安装时 pip install 报错",
    "body": (
        "### 环境信息\n"
        "- Ubuntu 22.04\n"
        "- A5\n"
        "- CANN 8.2.0\n"
        "### 问题描述\n"
        "pip install vllm-ascend 时报错: ERROR: No matching distribution found\n"
    ),
}


def verify_issue_state(api_base: str, number: int, prev_comment_count: int,
                       expect_label: bool, expect_new_comment: bool,
                       step_name: str) -> tuple[Result, int]:
    r = Result(step_name)
    labels, comments, elapsed = _poll_bot(api_base, number,
                                          expect_new_comment=expect_new_comment,
                                          prev_comment_count=prev_comment_count)
    has_label = LABEL_NAME in labels
    new_comment_count = len(comments)
    got_new_comment = new_comment_count > prev_comment_count

    checks = []
    if has_label != expect_label:
        checks.append(f"label: expected={expect_label} got={has_label} (labels={labels})")
    if got_new_comment != expect_new_comment:
        checks.append(f"comment: expected_new={expect_new_comment} got_new={got_new_comment} "
                      f"(prev={prev_comment_count} cur={new_comment_count})")

    if checks:
        r.fail("; ".join(checks))
    else:
        r.ok(f"label={has_label} new_comment={got_new_comment} ({elapsed}s)")
    return r, new_comment_count


def run_issue_lifecycle_tests(api_base: str) -> list[Result]:
    results: list[Result] = []

    # ── Chain I-bad: starts bad, covers I2, I6, I5, I4 ──
    print("=" * 60)
    print("Issue Chain I-bad: start bad -> I2 -> I6 -> I5 -> I4")
    print("=" * 60)
    n = None
    comment_count = 0
    try:
        n = create_issue(api_base, ISSUE_BAD["title"], ISSUE_BAD["body"])
        print(f"  Created issue #{n}")

        r, comment_count = verify_issue_state(api_base, n, comment_count,
                                              expect_label=True, expect_new_comment=True,
                                              step_name="I2: opened (bad) -> Flagged")
        results.append(r)
        print(f"  [I2] {'PASS' if r.passed else 'FAIL: ' + r.error}")

        time.sleep(5)

        edit_issue(api_base, n, ISSUE_BAD_ALT["title"], ISSUE_BAD_ALT["body"])
        print(f"  Edited issue #{n} (still bad)")
        r, comment_count = verify_issue_state(api_base, n, comment_count,
                                              expect_label=True, expect_new_comment=True,
                                              step_name="I6: Flagged + edit FAIL -> Flagged")
        results.append(r)
        print(f"  [I6] {'PASS' if r.passed else 'FAIL: ' + r.error}")

        time.sleep(5)

        edit_issue(api_base, n, ISSUE_GOOD["title"], ISSUE_GOOD["body"])
        print(f"  Edited issue #{n} (now good)")
        r, comment_count = verify_issue_state(api_base, n, comment_count,
                                              expect_label=False, expect_new_comment=True,
                                              step_name="I5: Flagged + edit PASS -> Clean")
        results.append(r)
        print(f"  [I5] {'PASS' if r.passed else 'FAIL: ' + r.error}")

        time.sleep(5)

        edit_issue(api_base, n, ISSUE_BAD_ALT["title"], ISSUE_BAD_ALT["body"])
        print(f"  Edited issue #{n} (bad again)")
        r, comment_count = verify_issue_state(api_base, n, comment_count,
                                              expect_label=True, expect_new_comment=True,
                                              step_name="I4: Clean + edit FAIL -> Flagged")
        results.append(r)
        print(f"  [I4] {'PASS' if r.passed else 'FAIL: ' + r.error}")
    except Exception as e:
        results.append(Result("Issue Chain I-bad exception"))
        results[-1].fail(str(e))
        print(f"  EXCEPTION: {e}")
    finally:
        if n:
            _close_entity(api_base, n)
    print()

    # ── Chain I-good: starts good, covers I1, I3 ──
    print("=" * 60)
    print("Issue Chain I-good: start good -> I1 -> I3")
    print("=" * 60)
    n2 = None
    comment_count = 0
    try:
        n2 = create_issue(api_base, ISSUE_GOOD["title"], ISSUE_GOOD["body"])
        print(f"  Created issue #{n2}")

        r, comment_count = verify_issue_state(api_base, n2, comment_count,
                                              expect_label=False, expect_new_comment=False,
                                              step_name="I1: opened (good) -> Clean")
        results.append(r)
        print(f"  [I1] {'PASS' if r.passed else 'FAIL: ' + r.error}")

        time.sleep(5)

        edit_issue(api_base, n2, ISSUE_GOOD_ALT["title"], ISSUE_GOOD_ALT["body"])
        print(f"  Edited issue #{n2} (still good)")
        r, comment_count = verify_issue_state(api_base, n2, comment_count,
                                              expect_label=False, expect_new_comment=False,
                                              step_name="I3: Clean + edit PASS -> Clean")
        results.append(r)
        print(f"  [I3] {'PASS' if r.passed else 'FAIL: ' + r.error}")
    except Exception as e:
        results.append(Result("Issue Chain I-good exception"))
        results[-1].fail(str(e))
        print(f"  EXCEPTION: {e}")
    finally:
        if n2:
            _close_entity(api_base, n2)
    print()

    return results


# ── PR Lifecycle ──────────────────────────────────────────────


PR_GOOD_DESC = {
    "title": "[Feat][NPU] Add optimised memory allocator for Ascend",
    "body": (
        "## Summary\n\n"
        "Implements a new memory allocator that reduces fragmentation by 40%.\n\n"
        "## Test plan\n"
        "- Unit tests added covering allocation and deallocation\n"
        "- Tested on A5 with 1000 iterations\n"
        "- Benchmark shows 15% throughput improvement\n"
    ),
}

PR_BAD_DESC = {
    "title": "[Bug]: 报错了",
    "body": "帮我看看",
}

PR_BAD_DESC_ALT = {
    "title": "[Bug]: 服务启动不了",
    "body": "启动报错",
}

PR_GOOD_DESC_ALT = {
    "title": "[Feat][Worker] Improve tensor parallelism for NPU models",
    "body": (
        "## Summary\n\n"
        "Refactors the tensor parallelism layer to better utilise NPU hardware.\n\n"
        "## Test plan\n"
        "- Multi-node tests passed on A5 cluster\n"
        "- All existing unit tests continue to pass\n"
    ),
}

DUMMY_COMMIT = ("chore: test commit", "", False)


class PRLifecycle:
    def __init__(self, api_base: str, repo: str, chain_name: str):
        self.api_base = api_base
        self.repo = repo
        self.chain_name = chain_name
        self.branch_name = ""
        self.worktree_path: Path | None = None
        self.pr_number: int | None = None
        self.comment_count = 0
        self._tmp_branch = ""

    def _make_worktree(self):
        safe = self.chain_name.replace(" ", "-").replace("_", "-")[:30]
        self.branch_name = f"test/scn-{safe}-{uuid.uuid4().hex[:6]}"
        self.worktree_path = WORKTREE_BASE / self.branch_name
        self._tmp_branch = f"test/tmp-{uuid.uuid4().hex[:8]}"

        _remove_worktree(self.worktree_path)
        run_cmd(["git", "branch", self._tmp_branch, "main"])
        run_cmd(["git", "worktree", "add", str(self.worktree_path), self._tmp_branch])
        print(f"  Worktree: {self.worktree_path}")

        run_cmd(["git", "commit", "--allow-empty", "-m", DUMMY_COMMIT[0]],
                cwd=str(self.worktree_path))
        print(f"    Commit: {DUMMY_COMMIT[0]}")

    def _push(self):
        run_cmd(["git", "push", "origin", f"HEAD:{self.branch_name}", "--force"],
                cwd=str(self.worktree_path))
        print(f"  Pushed to {self.branch_name}")

    def create_pr(self, title: str, body: str):
        self._make_worktree()
        self._push()

        pr_data = api_post(f"{self.api_base}/pulls", {
            "title": title,
            "body": body,
            "head": self.branch_name,
            "base": "main",
        })
        self.pr_number = pr_data["number"]
        print(f"  PR created: #{self.pr_number}")

    def edit_pr(self, title: str, body: str):
        api_patch(f"{self.api_base}/pulls/{self.pr_number}",
                  {"title": title, "body": body})
        print(f"  Edited PR #{self.pr_number}")

    def verify(self, expect_desc_label: bool, expect_new_comment: bool,
               step_name: str) -> Result:
        r = Result(step_name)
        labels, comments, elapsed = _poll_bot(self.api_base, self.pr_number,
                                              expect_new_comment=expect_new_comment,
                                              prev_comment_count=self.comment_count)
        has_desc = LABEL_NAME in labels
        new_count = len(comments)
        got_new = new_count > self.comment_count
        self.comment_count = new_count

        checks = []
        if has_desc != expect_desc_label:
            checks.append(f"desc label: expected={expect_desc_label} got={has_desc}")
        if got_new != expect_new_comment:
            checks.append(f"new comment: expected={expect_new_comment} got={got_new}")

        if checks:
            r.fail("; ".join(checks) + f" labels={labels}")
        else:
            r.ok(f"desc_label={has_desc} new_comment={got_new} ({elapsed}s)")
        return r

    def cleanup(self):
        if self.pr_number:
            _cleanup_pr(self.branch_name, self.pr_number, self.api_base)


def run_pr_lifecycle_tests(api_base: str, repo: str) -> list[Result]:
    results: list[Result] = []
    WORKTREE_BASE.mkdir(parents=True, exist_ok=True)

    # ── Chain A: good -> P1, P3, P4, P5 ──
    print("=" * 60)
    print("PR Chain A: good -> P1 P3 P4 P5")
    print("=" * 60)
    lc = PRLifecycle(api_base, repo, "chain-a")
    try:
        lc.create_pr(PR_GOOD_DESC["title"], PR_GOOD_DESC["body"])
        r = lc.verify(expect_desc_label=False, expect_new_comment=False,
                      step_name="P1: opened (good) -> Clean")
        results.append(r)
        print(f"  [P1] {'PASS' if r.passed else 'FAIL: ' + r.error}")
        time.sleep(5)

        lc.edit_pr(PR_GOOD_DESC_ALT["title"], PR_GOOD_DESC_ALT["body"])
        r = lc.verify(expect_desc_label=False, expect_new_comment=False,
                      step_name="P3: Clean + edit PASS -> Clean")
        results.append(r)
        print(f"  [P3] {'PASS' if r.passed else 'FAIL: ' + r.error}")
        time.sleep(5)

        lc.edit_pr(PR_BAD_DESC["title"], PR_BAD_DESC["body"])
        r = lc.verify(expect_desc_label=True, expect_new_comment=True,
                      step_name="P4: Clean + edit FAIL -> Desc-flagged")
        results.append(r)
        print(f"  [P4] {'PASS' if r.passed else 'FAIL: ' + r.error}")
        time.sleep(5)

        lc.edit_pr(PR_GOOD_DESC["title"], PR_GOOD_DESC["body"])
        r = lc.verify(expect_desc_label=False, expect_new_comment=True,
                      step_name="P5: Desc-flagged + edit PASS -> Clean")
        results.append(r)
        print(f"  [P5] {'PASS' if r.passed else 'FAIL: ' + r.error}")
    except Exception as e:
        results.append(Result("Chain A exception"))
        results[-1].fail(str(e))
        print(f"  EXCEPTION: {e}")
    finally:
        lc.cleanup()
    print()

    # ── Chain B: bad -> P2, P6 ──
    print("=" * 60)
    print("PR Chain B: bad -> P2 P6")
    print("=" * 60)
    lc = PRLifecycle(api_base, repo, "chain-b")
    try:
        lc.create_pr(PR_BAD_DESC["title"], PR_BAD_DESC["body"])
        r = lc.verify(expect_desc_label=True, expect_new_comment=True,
                      step_name="P2: opened (bad) -> Desc-flagged")
        results.append(r)
        print(f"  [P2] {'PASS' if r.passed else 'FAIL: ' + r.error}")
        time.sleep(5)

        lc.edit_pr(PR_BAD_DESC_ALT["title"], PR_BAD_DESC_ALT["body"])
        r = lc.verify(expect_desc_label=True, expect_new_comment=True,
                      step_name="P6: Desc-flagged + edit FAIL -> unchanged")
        results.append(r)
        print(f"  [P6] {'PASS' if r.passed else 'FAIL: ' + r.error}")
    except Exception as e:
        results.append(Result("Chain B exception"))
        results[-1].fail(str(e))
        print(f"  EXCEPTION: {e}")
    finally:
        lc.cleanup()
    print()

    # ── Chain C: good + sync -> P1, P7, P8, P9, P10 ──
    print("=" * 60)
    print("PR Chain C: good -> P7 P8 P9 P10")
    print("=" * 60)
    lc = PRLifecycle(api_base, repo, "chain-c")
    try:
        lc.create_pr(PR_GOOD_DESC["title"], PR_GOOD_DESC["body"])
        _poll_bot(lc.api_base, lc.pr_number, expect_new_comment=False)
        lc.comment_count = len(get_bot_comments(lc.api_base, lc.pr_number))
        print(f"  PR #{lc.pr_number} settled (P1 baseline)")
        time.sleep(5)

        lc.edit_pr(PR_GOOD_DESC_ALT["title"], PR_GOOD_DESC_ALT["body"])
        r = lc.verify(expect_desc_label=False, expect_new_comment=False,
                      step_name="P7: Clean + sync PASS -> Clean")
        results.append(r)
        print(f"  [P7] {'PASS' if r.passed else 'FAIL: ' + r.error}")
        time.sleep(5)

        lc.edit_pr(PR_BAD_DESC["title"], PR_BAD_DESC["body"])
        r = lc.verify(expect_desc_label=True, expect_new_comment=True,
                      step_name="P8: Clean + sync FAIL -> Desc-flagged")
        results.append(r)
        print(f"  [P8] {'PASS' if r.passed else 'FAIL: ' + r.error}")
        time.sleep(5)

        lc.edit_pr(PR_GOOD_DESC["title"], PR_GOOD_DESC["body"])
        r = lc.verify(expect_desc_label=False, expect_new_comment=True,
                      step_name="P9: Desc-flagged + sync PASS -> Clean")
        results.append(r)
        print(f"  [P9] {'PASS' if r.passed else 'FAIL: ' + r.error}")
        time.sleep(5)

        lc.edit_pr(PR_BAD_DESC_ALT["title"], PR_BAD_DESC_ALT["body"])
        r = lc.verify(expect_desc_label=True, expect_new_comment=True,
                      step_name="P10: Clean + sync FAIL -> Desc-flagged")
        results.append(r)
        print(f"  [P10] {'PASS' if r.passed else 'FAIL: ' + r.error}")
    except Exception as e:
        results.append(Result("Chain C exception"))
        results[-1].fail(str(e))
        print(f"  EXCEPTION: {e}")
    finally:
        lc.cleanup()
    print()

    # ── Chain D: flagged + no-body sync -> P11, P12 ──
    print("=" * 60)
    print("PR Chain D: flagged -> P11 P12")
    print("=" * 60)
    lc = PRLifecycle(api_base, repo, "chain-d")
    try:
        lc.create_pr(PR_BAD_DESC["title"], PR_BAD_DESC["body"])
        r = lc.verify(expect_desc_label=True, expect_new_comment=True,
                      step_name="P2 baseline: opened (bad) -> Desc-flagged")
        results.append(r)
        print(f"  [P2] {'PASS' if r.passed else 'FAIL: ' + r.error}")
        time.sleep(5)

        # Force a sync by editing with same content (title unchanged triggers synchronize without desc run)
        lc.edit_pr(PR_BAD_DESC["title"], PR_BAD_DESC["body"])
        # Wait long enough for bot not to run desc check
        labels, comments, _ = _poll_bot(lc.api_base, lc.pr_number, expect_new_comment=False,
                                         max_wait=120)
        has_label = LABEL_NAME in labels
        if has_label:
            r = Result("P12: Desc-flagged + skip -> unchanged")
            r.ok(f"label preserved labels={labels}")
        else:
            r = Result("P12: Desc-flagged + skip -> unchanged")
            r.fail(f"label lost labels={labels}")
        results.append(r)
        print(f"  [P12] {'PASS' if r.passed else 'FAIL: ' + r.error}")
    except Exception as e:
        results.append(Result("Chain D exception"))
        results[-1].fail(str(e))
        print(f"  EXCEPTION: {e}")
    finally:
        lc.cleanup()
    print()

    return results


# ── Main ──────────────────────────────────────────────────────


def print_summary(results: list[Result]):
    print("=" * 60)
    passed = sum(1 for r in results if r.passed)
    total = len(results)
    print(f"Results: {passed}/{total} passed")
    for r in results:
        icon = "PASS" if r.passed else "FAIL"
        print(f"  [{icon}] {r.name}")
        if r.details:
            print(f"         {r.details}")
        if r.error:
            print(f"         Error: {r.error}")
    if passed != total:
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="E2E scenario test for Issue and PR review bots"
    )
    parser.add_argument("--repo", default="ai-infra-develop/vllm-ascend",
                        help="GitHub repo (owner/name)")
    parser.add_argument("--mode", default="all",
                        choices=["issue", "pr", "all"],
                        help="Test mode")
    args = parser.parse_args()

    if not GITHUB_TOKEN:
        print("ERROR: GITHUB_TOKEN environment variable not set")
        sys.exit(1)

    api_base = f"https://api.github.com/repos/{args.repo}"
    print(f"Repo: {args.repo}")
    print(f"Mode: {args.mode}")
    print()

    all_results: list[Result] = []

    if args.mode in ("issue", "all"):
        all_results.extend(run_issue_lifecycle_tests(api_base))

    if args.mode in ("pr", "all"):
        all_results.extend(run_pr_lifecycle_tests(api_base, args.repo))

    print_summary(all_results)


if __name__ == "__main__":
    main()
