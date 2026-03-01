#!/usr/bin/env python3
#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#
"""
Extract failed test cases from a GitHub Actions workflow run.

This script queries the GitHub API to find failed jobs in a workflow run,
downloads their logs, and extracts the specific test files that failed.
It outputs a JSON report that can be consumed by the bisect workflow.

Requirements:
  - ``gh`` CLI installed and authenticated (pre-installed in GitHub Actions)

Usage:
  # Auto-find the latest failed schedule_test_vllm_main run:
  python3 tools/extract_failures.py --repo vllm-project/vllm-ascend

  # Specify a run ID:
  python3 tools/extract_failures.py --repo vllm-project/vllm-ascend --run-id 22414687320

  # Only process runs from a specific UTC hour (for dedup in cron):
  python3 tools/extract_failures.py --repo vllm-project/vllm-ascend --only-hour 20

  # Check if a bisect is already running before triggering:
  python3 tools/extract_failures.py --repo vllm-project/vllm-ascend --check-duplicate
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import subprocess
import sys
from datetime import datetime

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Patterns for extracting information from CI logs
# ---------------------------------------------------------------------------

# run_suite.py / ci_utils.py failure markers:
#   "✗ FAILED: tests/e2e/singlecard/test_models.py returned exit code 1"
#   or in the summary block:
#   "✗ FAILED:"
#   "  tests/e2e/singlecard/test_models.py (exit code 1)"
_FAILED_INLINE_RE = re.compile(
    r"FAILED:\s+(tests/\S+\.py(?:::\S+)?)\s+.*exit code",
)
_FAILED_SUMMARY_RE = re.compile(
    r"^\s+(tests/\S+\.py(?:::\S+)?)\s+\(exit code",
    re.MULTILINE,
)

# checkout action output patterns for extracting the commit hash:
#   "HEAD is now at abc1234 some commit message"
#   "Checking out 'abc1234def...'"
#   "abc1234def5678901234567890abcdef12345678"  (standalone 40-char hash line)
_CHECKOUT_HASH_RE = re.compile(
    r"(?:Checking out|HEAD is now at)\s+['\"]?([0-9a-f]{7,40})",
)
# Standalone 40-char hash line, possibly with GitHub Actions timestamp prefix
# e.g. "2024-01-15T10:30:00.1234567Z ed42507f6d6e..." or just "ed42507f6d6e..."
_STANDALONE_HASH_RE = re.compile(
    r"^(?:\d{4}-\d{2}-\d{2}T[\d:.]+Z\s+)?([0-9a-f]{40})\s*$",
)
# "git log -1 --format=%H" followed by the hash on the next line
_GIT_LOG_HASH_RE = re.compile(r"git log -1 --format=%H")


# ---------------------------------------------------------------------------
# gh CLI helpers
# ---------------------------------------------------------------------------


def _gh_api(endpoint: str, **kwargs) -> dict | list:
    """Call ``gh api`` and return parsed JSON.

    Query parameters are appended to the URL (GET request).
    """
    # Build query string from kwargs
    if kwargs:
        params = "&".join(f"{k}={v}" for k, v in kwargs.items())
        url = f"{endpoint}?{params}"
    else:
        url = endpoint
    cmd = ["gh", "api", url]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
        )
    except FileNotFoundError:
        logger.error("'gh' CLI not found. Install it or run inside GitHub Actions.")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        logger.error("gh api %s failed: %s", url, e.stderr.strip())
        sys.exit(1)
    return json.loads(result.stdout)


def _gh_api_raw(endpoint: str) -> str:
    """Call ``gh api`` and return raw text (for log downloads)."""
    cmd = ["gh", "api", endpoint]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        logger.warning("Failed to download log from %s: %s", endpoint, e.stderr.strip())
        return ""
    return result.stdout


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------


def find_latest_failed_run(repo: str, workflow: str) -> dict | None:
    """Find the most recent failed run of the given workflow."""
    data = _gh_api(
        f"/repos/{repo}/actions/workflows/{workflow}/runs",
        status="failure",
        per_page="5",
    )
    runs = data.get("workflow_runs", [])
    if not runs:
        return None
    # Return the most recent one (API returns newest first)
    return runs[0]


def get_run_info(repo: str, run_id: int) -> dict:
    """Get metadata for a specific workflow run."""
    return _gh_api(f"/repos/{repo}/actions/runs/{run_id}")


def get_failed_jobs(repo: str, run_id: int) -> list[dict]:
    """List all failed jobs in a workflow run."""
    data = _gh_api(f"/repos/{repo}/actions/runs/{run_id}/jobs", per_page="100")
    return [j for j in data.get("jobs", []) if j.get("conclusion") == "failure"]


# PLACEHOLDER_EXTRACT


def extract_vllm_commit_from_log(log_text: str) -> str | None:
    """Extract the vllm commit hash from a job log.

    The ``actions/checkout`` action prints lines like:
        ``/usr/bin/git checkout --progress --force ...``
        ``HEAD is now at <hash> <message>``
    after a section referencing the vllm repo.

    We search for "HEAD is now at <hash>" that appears after any line
    mentioning "vllm-project/vllm" (but not "vllm-ascend").
    """
    lines = log_text.splitlines()
    seen_vllm_repo = False
    seen_git_log_cmd = False
    for line in lines:
        # Detect any reference to the vllm repo (not vllm-ascend)
        if "vllm-project/vllm" in line and "vllm-ascend" not in line:
            seen_vllm_repo = True
        # Once we've seen the vllm repo, look for the commit hash
        if seen_vllm_repo:
            m = _CHECKOUT_HASH_RE.search(line)
            if m:
                return m.group(1)
            # Detect "git log -1 --format=%H" — the next line is the hash
            if _GIT_LOG_HASH_RE.search(line):
                seen_git_log_cmd = True
                continue
            # Line right after "git log -1 --format=%H"
            if seen_git_log_cmd:
                m2 = _STANDALONE_HASH_RE.match(line.strip())
                if m2:
                    return m2.group(1)
                seen_git_log_cmd = False
            # Some checkout versions print just the 40-char hash on its own line
            stripped = line.strip()
            m3 = _STANDALONE_HASH_RE.match(stripped)
            if m3:
                return m3.group(1)
        # If we hit the next step's group header after seeing the repo,
        # and it's clearly a different step, stop looking
        if seen_vllm_repo and "##[group]Run " in line:
            # Check if this is a new step (not the checkout step itself)
            if "checkout" not in line.lower():
                seen_vllm_repo = False
    return None


def extract_failed_tests_from_log(log_text: str) -> list[str]:
    """Extract failed test file paths from a job log.

    Matches the output format of ``run_suite.py`` / ``ci_utils.py``:
      - Inline: ``✗ FAILED: <path> returned exit code <N>``
      - Summary block: ``  <path> (exit code <N>)``
    """
    failed = set()
    for m in _FAILED_INLINE_RE.finditer(log_text):
        failed.add(m.group(1))
    for m in _FAILED_SUMMARY_RE.finditer(log_text):
        failed.add(m.group(1))
    return sorted(failed)


def process_failed_jobs(
    repo: str,
    jobs: list[dict],
) -> tuple[str | None, list[dict], list[dict]]:
    """Download logs for failed jobs and extract failure info.

    Returns:
        (vllm_commit, succeeded_extractions, failed_extractions)
    """
    vllm_commit = None
    succeeded = []
    extraction_errors = []

    for job in jobs:
        job_id = job["id"]
        job_name = job["name"]
        logger.info("Downloading log for job %s (%s)...", job_id, job_name)

        log_text = _gh_api_raw(f"/repos/{repo}/actions/jobs/{job_id}/logs")
        if not log_text:
            extraction_errors.append(
                {
                    "job_id": job_id,
                    "job_name": job_name,
                    "error": "Failed to download job log",
                }
            )
            continue

        # Extract vllm commit (only need it once, all jobs share the same)
        if vllm_commit is None:
            vllm_commit = extract_vllm_commit_from_log(log_text)

        # Extract failed tests
        failed_tests = extract_failed_tests_from_log(log_text)
        if not failed_tests:
            extraction_errors.append(
                {
                    "job_id": job_id,
                    "job_name": job_name,
                    "error": "Could not parse failed test names from log",
                }
            )
            continue

        succeeded.append(
            {
                "job_id": job_id,
                "job_name": job_name,
                "failed_tests": failed_tests,
            }
        )

    return vllm_commit, succeeded, extraction_errors


# PLACEHOLDER_DEDUP


def check_hour_filter(run_info: dict, only_hour: int | None) -> str | None:
    """Check if the run was created at the expected UTC hour.

    Returns a skip reason string if it should be skipped, None otherwise.
    """
    if only_hour is None:
        return None
    created_at = run_info.get("created_at", "")
    try:
        dt = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
    except ValueError:
        return f"Cannot parse created_at: {created_at}"
    if dt.hour != only_hour:
        return f"Run created at UTC hour {dt.hour}, expected {only_hour}"
    return None


def check_duplicate_bisect(repo: str, bad_commit: str) -> str | None:
    """Check if there's already a bisect run in progress with the same bad commit.

    Returns a skip reason string if duplicate found, None otherwise.
    """
    try:
        data = _gh_api(
            f"/repos/{repo}/actions/workflows/bisect_vllm.yaml/runs",
            status="in_progress",
            per_page="10",
        )
    except SystemExit:
        # If the bisect workflow doesn't exist yet, no duplicate
        return None

    for run in data.get("workflow_runs", []):
        # Check the run's display_title or inputs for the bad commit
        # The run name typically includes the inputs
        title = run.get("display_title", "")
        if bad_commit[:12] in title:
            return f"Bisect already running: {run.get('html_url', run['id'])}"

    return None


def get_good_commit(repo: str) -> str | None:
    """Extract the good (currently pinned) vllm commit from origin/main.

    Tries multiple strategies:
    1. Read pr_test_light.yaml from disk (works in checkout context)
    2. Read from origin/main via git show (works in full clone)
    3. Fallback to bisect_helper.py
    """
    commit_re = re.compile(r"^[0-9a-f]{7,40}$")
    yaml_rel = ".github/workflows/pr_test_light.yaml"

    def _extract_from_content(content: str) -> str | None:
        m = re.search(r"vllm_version:\s*\[([^\]]+)\]", content)
        if not m:
            return None
        entries = [e.strip().strip("'\"") for e in m.group(1).split(",")]
        for entry in entries:
            if commit_re.match(entry):
                return entry
        return None

    # Strategy 1: read from disk (current checkout)
    try:
        repo_root = subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        disk_path = os.path.join(repo_root, yaml_rel)
        if os.path.exists(disk_path):
            with open(disk_path) as f:
                commit = _extract_from_content(f.read())
            if commit:
                logger.info("Good commit from disk (%s): %s", disk_path, commit)
                return commit
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        pass

    # Strategy 2: read from origin/main via git show
    try:
        result = subprocess.run(
            ["git", "show", f"origin/main:{yaml_rel}"],
            capture_output=True,
            text=True,
            check=True,
        )
        commit = _extract_from_content(result.stdout)
        if commit:
            logger.info("Good commit from origin/main: %s", commit)
            return commit
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    # Strategy 3: fallback to bisect_helper.py
    try:
        result = subprocess.run(
            ["python3", "tools/bisect_helper.py", "get-commit"],
            capture_output=True,
            text=True,
            check=True,
        )
        commit = result.stdout.strip()
        if commit:
            return commit
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    logger.warning("Could not determine good commit from any source")
    return None


# PLACEHOLDER_MAIN


def build_result(
    run_info: dict,
    good_commit: str | None,
    bad_commit: str | None,
    succeeded: list[dict],
    extraction_errors: list[dict],
    skip_reason: str | None = None,
) -> dict:
    """Build the final JSON result."""
    all_failed_tests = []
    for job_info in succeeded:
        all_failed_tests.extend(job_info["failed_tests"])
    # Deduplicate while preserving order
    seen = set()
    unique_tests = []
    for t in all_failed_tests:
        if t not in seen:
            seen.add(t)
            unique_tests.append(t)

    test_cmds = [f"pytest -sv {t}" for t in unique_tests]

    return {
        "run_id": run_info.get("id"),
        "run_url": run_info.get("html_url"),
        "run_created_at": run_info.get("created_at"),
        "vllm_bad_commit": bad_commit,
        "vllm_good_commit": good_commit,
        "failed_jobs": succeeded,
        "extraction_errors": extraction_errors,
        "failed_tests": unique_tests,
        "test_cmds": test_cmds,
        "test_cmds_semicolon": "; ".join(test_cmds),
        "skip_reason": skip_reason,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Extract failed tests from a GitHub Actions workflow run.",
    )
    parser.add_argument(
        "--repo",
        required=True,
        help="GitHub repository (e.g. vllm-project/vllm-ascend)",
    )
    parser.add_argument(
        "--run-id",
        type=int,
        default=None,
        help="Specific workflow run ID. If omitted, finds the latest failed run.",
    )
    parser.add_argument(
        "--workflow",
        default="schedule_test_vllm_main.yaml",
        help="Workflow filename to search for failed runs (default: schedule_test_vllm_main.yaml)",
    )
    parser.add_argument(
        "--only-hour",
        type=int,
        default=None,
        help="Only process runs created at this UTC hour (e.g. 20 for UTC+8 4am). Runs at other hours are skipped.",
    )
    parser.add_argument(
        "--check-duplicate",
        action="store_true",
        help="Skip if a bisect workflow is already running with the same bad commit.",
    )
    args = parser.parse_args()

    # Step 1: Find the run
    if args.run_id:
        logger.info("Using specified run ID: %d", args.run_id)
        run_info = get_run_info(args.repo, args.run_id)
    else:
        logger.info("Searching for latest failed run of %s...", args.workflow)
        run_info = find_latest_failed_run(args.repo, args.workflow)
        if not run_info:
            result = build_result({}, None, None, [], [], skip_reason="No failed runs found")
            print(json.dumps(result, indent=2))
            return

    run_id = run_info["id"]
    logger.info("Analyzing run %d: %s", run_id, run_info.get("html_url", ""))

    # Step 2: Hour filter (for dedup in scheduled triggers)
    skip = check_hour_filter(run_info, args.only_hour)
    if skip:
        logger.info("Skipping: %s", skip)
        result = build_result(run_info, None, None, [], [], skip_reason=skip)
        print(json.dumps(result, indent=2))
        return

    # Step 3: Get failed jobs
    failed_jobs = get_failed_jobs(args.repo, run_id)
    if not failed_jobs:
        logger.info("No failed jobs found in run %d", run_id)
        result = build_result(run_info, None, None, [], [], skip_reason="No failed jobs in run")
        print(json.dumps(result, indent=2))
        return

    logger.info("Found %d failed job(s)", len(failed_jobs))

    # Step 4: Download logs and extract failures
    bad_commit, succeeded, extraction_errors = process_failed_jobs(
        args.repo,
        failed_jobs,
    )

    # Step 4b: Fallback for bad_commit — query vllm repo for HEAD at run time
    if not bad_commit:
        logger.warning("Could not extract vllm commit from logs, trying vllm repo API")
        try:
            # Get the latest commit on vllm main at the time the run was created
            vllm_data = _gh_api("/repos/vllm-project/vllm/commits/main")
            bad_commit = vllm_data.get("sha")
            if bad_commit:
                logger.info("Bad commit from vllm main HEAD: %s", bad_commit[:12])
        except SystemExit:
            logger.warning("Could not query vllm repo for HEAD commit")

    # Step 5: Get good commit
    good_commit = get_good_commit(args.repo)
    if good_commit:
        logger.info("Good commit (from origin/main): %s", good_commit)
    else:
        logger.warning("Could not determine good commit automatically")

    # Step 6: Check duplicate bisect
    if args.check_duplicate and bad_commit:
        dup = check_duplicate_bisect(args.repo, bad_commit)
        if dup:
            logger.info("Skipping: %s", dup)
            result = build_result(run_info, good_commit, bad_commit, succeeded, extraction_errors, skip_reason=dup)
            print(json.dumps(result, indent=2))
            return

    # Step 7: Build and output result
    result = build_result(
        run_info,
        good_commit,
        bad_commit,
        succeeded,
        extraction_errors,
    )

    if not succeeded and extraction_errors:
        logger.warning(
            "All %d failed job(s) had extraction errors. No test commands to bisect.",
            len(extraction_errors),
        )
        result["skip_reason"] = "All log extractions failed"

    print(json.dumps(result, indent=2))

    # Log summary to stderr for CI visibility
    if result["failed_tests"]:
        logger.info("Extracted %d failed test(s):", len(result["failed_tests"]))
        for t in result["failed_tests"]:
            logger.info("  - %s", t)
    if extraction_errors:
        logger.warning("Extraction errors in %d job(s):", len(extraction_errors))
        for e in extraction_errors:
            logger.warning("  - %s: %s", e["job_name"], e["error"])


if __name__ == "__main__":
    main()
