#!/usr/bin/env python3
"""End-to-end test covering ALL scenarios from SCENARIOS.md.

Tests the full lifecycle: opened -> edited for both issues and PRs.

For PRs: polls the GitHub Actions run directly, then verifies labels.
"""

import argparse
import calendar
import contextlib
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
WORKTREE_BASE = Path("/tmp/opencode/scenario-tests")
PR_POLL_INTERVAL = 3
PR_MAX_WAIT = 150


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
    with contextlib.suppress(Exception):
        api_patch(f"{api_base}/issues/{number}", {"state": "closed"})


def get_labels(api_base: str, number: int) -> list[str]:
    data = api_get(f"{api_base}/issues/{number}")
    return [lb["name"] for lb in data.get("labels", [])]


def get_bot_comments(api_base: str, number: int) -> list[dict]:
    comments = api_get(f"{api_base}/issues/{number}/comments")
    return [c for c in comments if c.get("user", {}).get("login", "").endswith("[bot]")]


ISSUE_WORKFLOW_ID = None  # lazy-loaded per repo


def _get_issue_workflow_id(api_base: str) -> int:
    global ISSUE_WORKFLOW_ID
    if ISSUE_WORKFLOW_ID is None:
        workflows = api_get(f"{api_base}/actions/workflows")
        for wf in workflows.get("workflows", []):
            if wf.get("path") == ".github/workflows/bot_issue_review.yaml":
                ISSUE_WORKFLOW_ID = wf["id"]
                break
    if ISSUE_WORKFLOW_ID is None:
        raise RuntimeError("Could not find bot_issue_review.yaml workflow ID")
    return ISSUE_WORKFLOW_ID


PR_WORKFLOW_ID = None  # lazy-loaded per repo


def _get_pr_workflow_id(api_base: str) -> int:
    """Get the workflow ID for bot_pr_review.yaml."""
    global PR_WORKFLOW_ID
    if PR_WORKFLOW_ID is None:
        workflows = api_get(f"{api_base}/actions/workflows")
        for wf in workflows.get("workflows", []):
            if wf.get("path") == ".github/workflows/bot_pr_review.yaml":
                PR_WORKFLOW_ID = wf["id"]
                break
    if PR_WORKFLOW_ID is None:
        raise RuntimeError("Could not find bot_pr_review.yaml workflow ID")
    return PR_WORKFLOW_ID


def wait_for_workflow_run(
    repo: str, branch: str, after: float, skip_run_ids: set[int] | None = None, max_wait: int = PR_MAX_WAIT
) -> dict | None:
    """Wait for the bot_pr_review.yaml workflow run to appear and complete."""
    api_base = f"https://api.github.com/repos/{repo}"
    workflow_id = _get_pr_workflow_id(api_base)
    start = time.time()
    run_id = None
    skip = skip_run_ids or set()

    while time.time() - start < max_wait:
        # Phase 1: wait for a bot_pr_review run newer than `after`, not in skip set
        run_id = None
        while time.time() - start < max_wait:
            url = f"{api_base}/actions/runs?branch={branch}&event=pull_request_target&per_page=10"
            found = api_get(url).get("workflow_runs", [])
            for run in found:
                if run.get("workflow_id") != workflow_id:
                    continue
                if run["id"] in skip:
                    continue
                created = run.get("created_at", "")
                run_time = calendar.timegm(time.strptime(created, "%Y-%m-%dT%H:%M:%SZ")) if created else 0
                if run_time >= after:
                    run_id = run["id"]
                    print(
                        f"  Run {run_id} appeared ({run.get('status', '?')}) "
                        f"[run_time={run_time} after={after} diff={run_time - after:.0f}s]"
                    )
                    break
            if run_id:
                break
            elapsed = int(time.time() - start)
            print(f"  Waiting for bot_pr_review run... ({elapsed}s) found={len(found)} after={after}")
            time.sleep(PR_POLL_INTERVAL)

        if not run_id:
            print(f"  WARNING: no workflow run appeared within {max_wait}s")
            return None

        # Phase 2: wait for that run to complete
        while time.time() - start < max_wait:
            run = api_get(f"{api_base}/actions/runs/{run_id}")
            status = run.get("status", "")
            conclusion = run.get("conclusion", "")
            if status == "completed":
                if conclusion == "cancelled":
                    print(f"  Run {run_id} was cancelled, looking for next run")
                    skip.add(run_id)
                    after = time.time()  # look for runs after now
                    run_id = None  # re-enter Phase 1
                    break
                print(f"  Run {run_id} completed: {conclusion}")
                return run
            elapsed = int(time.time() - start)
            print(f"  Run {run_id} in progress ({status})... ({elapsed}s)")
            time.sleep(PR_POLL_INTERVAL)

    print(f"  WARNING: run {run_id} did not complete within {max_wait}s")
    return api_get(f"{api_base}/actions/runs/{run_id}")


def wait_for_issue_run(
    repo: str, issue_title: str, after: float, skip_run_ids: set[int] | None = None, max_wait: int = PR_MAX_WAIT
) -> dict | None:
    """Wait for the bot_issue_review.yaml workflow run to appear and complete for the given issue."""
    api_base = f"https://api.github.com/repos/{repo}"
    workflow_id = _get_issue_workflow_id(api_base)
    start = time.time()
    run_id = None
    skip = skip_run_ids or set()

    while time.time() - start < max_wait:
        # Phase 1: find a bot_issue_review run for this issue newer than `after`
        run_id = None
        while time.time() - start < max_wait:
            url = f"{api_base}/actions/runs?event=issues&per_page=30"
            found = api_get(url).get("workflow_runs", [])
            for run in found:
                if run.get("workflow_id") != workflow_id:
                    continue
                if run["id"] in skip:
                    continue
                created = run.get("created_at", "")
                run_time = calendar.timegm(time.strptime(created, "%Y-%m-%dT%H:%M:%SZ")) if created else 0
                if run_time >= after and run.get("display_title") == issue_title:
                    run_id = run["id"]
                    print(
                        f"  Issue run {run_id} appeared ({run.get('status', '?')}) "
                        f"[run_time={run_time} after={after} diff={run_time - after:.0f}s]"
                    )
                    break
            if run_id:
                break
            elapsed = int(time.time() - start)
            matching = sum(1 for r in found if r.get("workflow_id") == workflow_id)
            print(
                f"  Waiting for issue run... ({elapsed}s) event_runs={len(found)} matching_wf={matching} after={after}"
            )
            time.sleep(PR_POLL_INTERVAL)

        if not run_id:
            print(f"  WARNING: no issue workflow run appeared within {max_wait}s")
            return None

        # Phase 2: wait for that run to complete
        while time.time() - start < max_wait:
            run = api_get(f"{api_base}/actions/runs/{run_id}")
            status = run.get("status", "")
            conclusion = run.get("conclusion", "")
            if status == "completed":
                if conclusion == "cancelled":
                    print(f"  Issue run {run_id} was cancelled, looking for next run")
                    skip.add(run_id)
                    after = time.time()
                    run_id = None
                    break
                print(f"  Issue run {run_id} completed: {conclusion}")
                return run
            elapsed = int(time.time() - start)
            print(f"  Issue run {run_id} in progress ({status})... ({elapsed}s)")
            time.sleep(PR_POLL_INTERVAL)

    print(f"  WARNING: issue run {run_id} did not complete within {max_wait}s")
    return api_get(f"{api_base}/actions/runs/{run_id}")


def get_run_logs(repo: str, run_id: int) -> str:
    """Get the logs for a workflow run."""
    url = f"https://api.github.com/repos/{repo}/actions/runs/{run_id}/logs"
    try:
        resp = requests.get(url, headers=HEADERS, allow_redirects=True, timeout=30)
        return resp.text[:2000]
    except Exception:
        return ""


# ── Issue Helpers ─────────────────────────────────────────────


def create_issue(api_base: str, title: str, body: str) -> int:
    data = api_post(f"{api_base}/issues", {"title": title, "body": body})
    return data["number"]


def edit_issue(api_base: str, number: int, title: str, body: str):
    api_patch(f"{api_base}/issues/{number}", {"title": title, "body": body})


def verify_issue_result(
    api_base: str,
    number: int,
    expect_label: bool,
    step_name: str,
    run: dict,
) -> Result:
    r = Result(step_name)
    labels = get_labels(api_base, number)
    has_label = LABEL_NAME in labels
    comments = get_bot_comments(api_base, number)
    run_id = run["id"]

    if has_label != expect_label:
        r.fail(f"label mismatch: expected={expect_label} got={has_label} (run={run_id}) labels={labels}")
    else:
        r.ok(f"run={run_id} label={has_label} bot_comments={len(comments)}")
    return r


# ── Git Helpers ───────────────────────────────────────────────


def run_cmd(cmd: list[str], cwd: str | None = None) -> str:
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\n{result.stderr}")
    return result.stdout.strip()


def _remove_worktree(worktree_path: Path):
    with contextlib.suppress(Exception):
        run_cmd(["git", "worktree", "remove", str(worktree_path), "--force"])
    if worktree_path.exists():
        shutil.rmtree(worktree_path)


def _cleanup_temp_branches():
    with contextlib.suppress(Exception):
        result = subprocess.run(["git", "branch"], capture_output=True, text=True)
        for line in result.stdout.split("\n"):
            bt = line.strip().lstrip("* ")
            if bt.startswith("test/"):
                with contextlib.suppress(Exception):
                    run_cmd(["git", "branch", "-D", bt])


def _cleanup_pr(branch_name: str, pr_number: int, api_base: str):
    worktree_path = WORKTREE_BASE / branch_name
    _remove_worktree(worktree_path)
    with contextlib.suppress(Exception):
        run_cmd(["git", "worktree", "prune"])
    _close_entity(api_base, pr_number)
    with contextlib.suppress(Exception):
        run_cmd(["git", "push", "origin", "--delete", branch_name])
    with contextlib.suppress(Exception):
        run_cmd(["git", "branch", "-D", branch_name])
    _cleanup_temp_branches()


# ── Issue Lifecycle ───────────────────────────────────────────

ISSUE_GOOD = {
    "title": "[Bug]: A5 CANN 8.2 conv2d operator RuntimeError 500002",
    "body": (
        "### Environment\n"
        "- OS: Ubuntu 22.04\n"
        "- Ascend hardware: A5\n"
        "- CANN version: 8.2.0\n"
        "- vLLM version: main\n"
        "### Problem Description\n"
        "Running conv2d triggers: RuntimeError: conv2d forward failed\n"
        "### Steps to Reproduce\n"
        "1. pip install vllm vllm-ascend\n"
        "2. Run test script\n"
        "3. Error: RuntimeError: conv2d forward failed, error code 500002\n"
    ),
}

ISSUE_BAD = {
    "title": "[Bug]: API call error",
    "body": "There's an error, help me check",
}

ISSUE_BAD_ALT = {
    "title": "[Bug]: Service startup failed",
    "body": "Won't start, got an error",
}

ISSUE_GOOD_ALT = {
    "title": "[Bug]: A5 CANN 8.2 pip install error",
    "body": (
        "### Environment\n"
        "- Ubuntu 22.04\n"
        "- A5\n"
        "- CANN 8.2.0\n"
        "### Problem Description\n"
        "pip install vllm-ascend fails: ERROR: No matching distribution found\n"
    ),
}


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
    "title": "[Bug]: Got error",
    "body": "Help me check",
}

PR_BAD_DESC_ALT = {
    "title": "[Bug]: Service won't start",
    "body": "Startup error",
}

PR_GOOD_DESC_ALT = {
    "title": "[Feat][Worker] Improve tensor parallelism for NPU models",
    "body": (
        "## Summary\n\n"
        "Refactors the tensor parallelism layer to better utilise NPU hardware, "
        "reducing inter-device communication overhead by 25%.\n\n"
        "## Test plan\n"
        "- Multi-node tests passed on A5 cluster (4 nodes, 32 cards)\n"
        "- Throughput improved by 18% on Llama-70B inference\n"
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
        self.last_event_time: float = 0.0
        self.seen_run_ids: set[int] = set()

    def _make_worktree(self):
        safe = self.chain_name.replace(" ", "-").replace("_", "-")[:30]
        self.branch_name = f"test/scn-{safe}-{uuid.uuid4().hex[:6]}"
        self.worktree_path = WORKTREE_BASE / self.branch_name
        tmp_branch = f"test/tmp-{uuid.uuid4().hex[:8]}"

        _remove_worktree(self.worktree_path)
        run_cmd(["git", "branch", tmp_branch, "main"])
        run_cmd(["git", "worktree", "add", str(self.worktree_path), tmp_branch])
        print(f"  Worktree: {self.worktree_path}")

        run_cmd(["git", "commit", "--allow-empty", "-m", DUMMY_COMMIT[0]], cwd=str(self.worktree_path))
        print(f"    Commit: {DUMMY_COMMIT[0]}")

    def _push(self):
        run_cmd(["git", "push", "origin", f"HEAD:{self.branch_name}", "--force"], cwd=str(self.worktree_path))
        print(f"  Pushed to {self.branch_name}")

    def create_pr(self, title: str, body: str):
        self._make_worktree()
        self._push()
        self.last_event_time = time.time()

        pr_data = api_post(
            f"{self.api_base}/pulls",
            {
                "title": title,
                "body": body,
                "head": self.branch_name,
                "base": "main",
            },
        )
        self.pr_number = pr_data["number"]
        print(f"  PR created: #{self.pr_number}")

    def edit_pr(self, title: str, body: str):
        self.last_event_time = time.time()
        api_patch(f"{self.api_base}/pulls/{self.pr_number}", {"title": title, "body": body})
        print(f"  Edited PR #{self.pr_number}")

    def wait_for_action(self, expect_ok: bool, step_name: str) -> Result:
        """Wait for the workflow run triggered after last_event_time to complete."""
        r = Result(step_name)
        run = wait_for_workflow_run(self.repo, self.branch_name, self.last_event_time, self.seen_run_ids)
        if not run:
            r.fail("no workflow run appeared")
            return r

        run_id = run["id"]
        self.seen_run_ids.add(run_id)
        conclusion = run.get("conclusion", "")

        if conclusion == "success":
            labels = get_labels(self.api_base, self.pr_number)
            has_label = LABEL_NAME in labels
            expect_label = "-> Desc-flagged" in step_name or "unchanged" in step_name

            if has_label != expect_label:
                r.fail(
                    f"label mismatch: expected={expect_label} got={has_label} "
                    f"(run={run_id} conclusion={conclusion}) labels={labels}"
                )
            else:
                r.ok(f"run={run_id} {conclusion} label={has_label}")
        else:
            r.fail(f"run={run_id} conclusion={conclusion} (expected success)")

        return r

    def cleanup(self):
        if self.pr_number:
            _cleanup_pr(self.branch_name, self.pr_number, self.api_base)


# ── Issue Tests ───────────────────────────────────────────────


def run_issue_lifecycle_tests(api_base: str, repo: str) -> list[Result]:
    results: list[Result] = []

    print("=" * 60)
    print("Issue Chain I-bad: start bad -> I2 -> I6 -> I5 -> I4")
    print("=" * 60)
    n = None
    seen: set[int] = set()
    after: float = 0.0
    try:
        after = time.time()
        n = create_issue(api_base, ISSUE_BAD["title"], ISSUE_BAD["body"])
        print(f"  Created issue #{n}")

        run = wait_for_issue_run(repo, ISSUE_BAD["title"], after)
        if not run:
            results.append(Result("I2: opened (bad) -> Flagged"))
            results[-1].fail("no workflow run appeared")
            print("  [I2] FAIL: no workflow run appeared")
            return results
        seen.add(run["id"])
        r = verify_issue_result(api_base, n, expect_label=True, step_name="I2: opened (bad) -> Flagged", run=run)
        results.append(r)
        print(f"  [I2] {'PASS' if r.passed else 'FAIL: ' + r.error}")

        time.sleep(3)

        after = time.time()
        edit_issue(api_base, n, ISSUE_BAD_ALT["title"], ISSUE_BAD_ALT["body"])
        print(f"  Edited issue #{n} (still bad)")

        run = wait_for_issue_run(repo, ISSUE_BAD_ALT["title"], after, seen)
        if not run:
            results.append(Result("I6: Flagged + edit FAIL -> Flagged"))
            results[-1].fail("no workflow run appeared")
            print("  [I6] FAIL: no workflow run appeared")
            return results
        seen.add(run["id"])
        r = verify_issue_result(api_base, n, expect_label=True, step_name="I6: Flagged + edit FAIL -> Flagged", run=run)
        results.append(r)
        print(f"  [I6] {'PASS' if r.passed else 'FAIL: ' + r.error}")

        time.sleep(3)

        after = time.time()
        edit_issue(api_base, n, ISSUE_GOOD["title"], ISSUE_GOOD["body"])
        print(f"  Edited issue #{n} (now good)")

        run = wait_for_issue_run(repo, ISSUE_GOOD["title"], after, seen)
        if not run:
            results.append(Result("I5: Flagged + edit PASS -> Clean"))
            results[-1].fail("no workflow run appeared")
            print("  [I5] FAIL: no workflow run appeared")
            return results
        seen.add(run["id"])
        r = verify_issue_result(api_base, n, expect_label=False, step_name="I5: Flagged + edit PASS -> Clean", run=run)
        results.append(r)
        print(f"  [I5] {'PASS' if r.passed else 'FAIL: ' + r.error}")

        time.sleep(3)

        after = time.time()
        edit_issue(api_base, n, ISSUE_BAD_ALT["title"], ISSUE_BAD_ALT["body"])
        print(f"  Edited issue #{n} (bad again)")

        run = wait_for_issue_run(repo, ISSUE_BAD_ALT["title"], after, seen)
        if not run:
            results.append(Result("I4: Clean + edit FAIL -> Flagged"))
            results[-1].fail("no workflow run appeared")
            print("  [I4] FAIL: no workflow run appeared")
            return results
        seen.add(run["id"])
        r = verify_issue_result(api_base, n, expect_label=True, step_name="I4: Clean + edit FAIL -> Flagged", run=run)
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

    print("=" * 60)
    print("Issue Chain I-good: start good -> I1 -> I3")
    print("=" * 60)
    n2 = None
    seen2: set[int] = set()
    after2: float = 0.0
    try:
        after2 = time.time()
        n2 = create_issue(api_base, ISSUE_GOOD["title"], ISSUE_GOOD["body"])
        print(f"  Created issue #{n2}")

        run = wait_for_issue_run(repo, ISSUE_GOOD["title"], after2)
        if not run:
            results.append(Result("I1: opened (good) -> Clean"))
            results[-1].fail("no workflow run appeared")
            print("  [I1] FAIL: no workflow run appeared")
            return results
        seen2.add(run["id"])
        r = verify_issue_result(api_base, n2, expect_label=False, step_name="I1: opened (good) -> Clean", run=run)
        results.append(r)
        print(f"  [I1] {'PASS' if r.passed else 'FAIL: ' + r.error}")

        time.sleep(3)

        after2 = time.time()
        edit_issue(api_base, n2, ISSUE_GOOD_ALT["title"], ISSUE_GOOD_ALT["body"])
        print(f"  Edited issue #{n2} (still good)")

        run = wait_for_issue_run(repo, ISSUE_GOOD_ALT["title"], after2, seen2)
        if not run:
            results.append(Result("I3: Clean + edit PASS -> Clean"))
            results[-1].fail("no workflow run appeared")
            print("  [I3] FAIL: no workflow run appeared")
            return results
        seen2.add(run["id"])
        r = verify_issue_result(api_base, n2, expect_label=False, step_name="I3: Clean + edit PASS -> Clean", run=run)
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


# ── PR Tests ──────────────────────────────────────────────────


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
        r = lc.wait_for_action(expect_ok=True, step_name="P1: opened (good) -> Clean")
        results.append(r)
        print(f"  [P1] {'PASS' if r.passed else 'FAIL: ' + r.error}")
        time.sleep(3)

        lc.edit_pr(PR_GOOD_DESC_ALT["title"], PR_GOOD_DESC_ALT["body"])
        r = lc.wait_for_action(expect_ok=True, step_name="P3: Clean + edit PASS -> Clean")
        results.append(r)
        print(f"  [P3] {'PASS' if r.passed else 'FAIL: ' + r.error}")
        time.sleep(3)

        lc.edit_pr(PR_BAD_DESC["title"], PR_BAD_DESC["body"])
        r = lc.wait_for_action(expect_ok=False, step_name="P4: Clean + edit FAIL -> Desc-flagged")
        results.append(r)
        print(f"  [P4] {'PASS' if r.passed else 'FAIL: ' + r.error}")
        time.sleep(3)

        lc.edit_pr(PR_GOOD_DESC["title"], PR_GOOD_DESC["body"])
        r = lc.wait_for_action(expect_ok=True, step_name="P5: Desc-flagged + edit PASS -> Clean")
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
        r = lc.wait_for_action(expect_ok=False, step_name="P2: opened (bad) -> Desc-flagged")
        results.append(r)
        print(f"  [P2] {'PASS' if r.passed else 'FAIL: ' + r.error}")
        time.sleep(3)

        lc.edit_pr(PR_BAD_DESC_ALT["title"], PR_BAD_DESC_ALT["body"])
        r = lc.wait_for_action(expect_ok=False, step_name="P6: Desc-flagged + edit FAIL -> unchanged")
        results.append(r)
        print(f"  [P6] {'PASS' if r.passed else 'FAIL: ' + r.error}")
    except Exception as e:
        results.append(Result("Chain B exception"))
        results[-1].fail(str(e))
        print(f"  EXCEPTION: {e}")
    finally:
        lc.cleanup()
    print()

    # ── Chain C: flagged + no-body sync -> P11, P12 ──
    print("=" * 60)
    print("PR Chain C: flagged -> P11 P12")
    print("=" * 60)
    lc = PRLifecycle(api_base, repo, "chain-c")
    try:
        lc.create_pr(PR_BAD_DESC["title"], PR_BAD_DESC["body"])
        r = lc.wait_for_action(expect_ok=False, step_name="P2 baseline: opened (bad) -> Desc-flagged")
        results.append(r)
        print(f"  [P2] {'PASS' if r.passed else 'FAIL: ' + r.error}")
        time.sleep(3)

        lc.edit_pr(PR_BAD_DESC["title"], PR_BAD_DESC["body"])
        labels = get_labels(lc.api_base, lc.pr_number)
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
        results.append(Result("Chain C exception"))
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
    parser = argparse.ArgumentParser(description="E2E scenario test for Issue and PR review bots")
    parser.add_argument("--repo", default="ai-infra-develop/vllm-ascend", help="GitHub repo (owner/name)")
    parser.add_argument("--mode", default="all", choices=["issue", "pr", "all"], help="Test mode")
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
        all_results.extend(run_issue_lifecycle_tests(api_base, args.repo))

    if args.mode in ("pr", "all"):
        all_results.extend(run_pr_lifecycle_tests(api_base, args.repo))

    print_summary(all_results)


if __name__ == "__main__":
    main()
