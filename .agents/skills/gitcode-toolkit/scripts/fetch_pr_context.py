#!/usr/bin/env python3
#
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#
"""One-shot PR context fetcher: clone → checkout PR → merge-base → diff stats → commit log.

Usage:
    python fetch_pr_context.py --repo cann/ops-math --pr 123

Output (stdout): JSON with work_dir, base_branch, merge_base, changed_files, commits.
Progress / errors: stderr.

This replaces the manual composition of ~8 git commands across 4 reference files
(clone-and-checkout, diff-and-changes, log-and-show, remote-and-branch).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from _logging import init_output_logger, init_error_logger

ALLOWED_DOMAIN = "gitcode.com"
DEFAULT_DEPTH = 500
TMP_PREFIX = "gitcode_pr_ctx_"


@dataclass
class PRContext:
    work_dir: str = ""
    base_branch: str = ""
    merge_base: str = ""
    changed_files: list[dict] = field(default_factory=list)
    added_lines: int = 0
    deleted_lines: int = 0
    commits: list[dict] = field(default_factory=list)
    pr_title: str = ""
    pr_author: str = ""


def run_git(cmd: list[str], cwd: str | None = None, check: bool = True) -> subprocess.CompletedProcess:
    """Execute a git command and return the result."""
    try:
        result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, check=check)
        return result
    except subprocess.CalledProcessError as e:
        ERROR_LOGGER.error("git command failed: %s", " ".join(cmd))
        ERROR_LOGGER.error("stderr: %s", e.stderr.strip())
        raise


def clone_repo(owner: str, repo: str, work_dir: str) -> None:
    """Shallow-clone the repository."""
    repo_url = f"https://{ALLOWED_DOMAIN}/{owner}/{repo}.git"
    ERROR_LOGGER.info("Cloning %s (depth=%d) → %s", repo_url, DEFAULT_DEPTH, work_dir)
    run_git(["git", "clone", f"--depth={DEFAULT_DEPTH}", repo_url, work_dir])


def fetch_pr_branch(pr_number: int, work_dir: str) -> str:
    """Fetch and checkout the PR branch via merge-requests ref."""
    branch_name = f"pr_{pr_number}"
    ref = f"refs/merge-requests/{pr_number}/head"

    ERROR_LOGGER.info("Fetching PR #%d branch via %s", pr_number, ref)
    run_git(["git", "fetch", "origin", f"+{ref}:{branch_name}"], cwd=work_dir)
    run_git(["git", "checkout", branch_name], cwd=work_dir)
    return branch_name


def get_base_branch(work_dir: str) -> str:
    """Determine the base branch from the remote HEAD."""
    result = run_git(["git", "remote", "show", "origin"], cwd=work_dir)
    for line in result.stdout.splitlines():
        if "HEAD branch:" in line:
            return line.split(":")[-1].strip()
    for candidate in ["master", "main"]:
        r = run_git(
            ["git", "ls-remote", "--heads", "origin", candidate],
            cwd=work_dir, check=False
        )
        if r.returncode == 0 and r.stdout.strip():
            return candidate
    raise RuntimeError("Cannot determine base branch")


def compute_merge_base(pr_branch: str, base_branch: str, work_dir: str) -> str:
    """Compute the merge-base between the PR branch and the base branch."""
    ERROR_LOGGER.info("Computing merge-base: %s ← %s", base_branch, pr_branch)
    run_git(["git", "fetch", "origin", f"{base_branch}:base_branch"], cwd=work_dir)
    result = run_git(["git", "merge-base", "base_branch", pr_branch], cwd=work_dir)
    return result.stdout.strip()


def get_diff_stats(merge_base: str, pr_branch: str, work_dir: str) -> list[dict]:
    """Get changed files with status and line counts."""
    ERROR_LOGGER.info("Getting diff stats...")
    status_result = run_git(
        ["git", "diff", "--name-status", merge_base, pr_branch], cwd=work_dir
    )
    numstat_result = run_git(
        ["git", "diff", "--numstat", merge_base, pr_branch], cwd=work_dir
    )

    files = []
    for line in numstat_result.stdout.strip().splitlines():
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        added = int(parts[0]) if parts[0] != "-" else 0
        deleted = int(parts[1]) if parts[1] != "-" else 0
        path = parts[2]
        files.append({"path": path, "added": added, "deleted": deleted, "status": "?"})

    status_map = {}
    for line in status_result.stdout.strip().splitlines():
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) >= 2:
            status_map[parts[1]] = parts[0]

    for f in files:
        f["status"] = status_map.get(f["path"], "?")

    return files


def get_commits(merge_base: str, pr_branch: str, work_dir: str) -> list[dict]:
    """Get commit history between merge-base and PR head."""
    ERROR_LOGGER.info("Getting commit history...")
    result = run_git(
        ["git", "log", f"{merge_base}..{pr_branch}",
         "--pretty=format:%H|%s|%an|%ae", "--no-merges"],
        cwd=work_dir
    )
    commits = []
    for line in result.stdout.strip().splitlines():
        if not line:
            continue
        parts = line.split("|", 3)
        if len(parts) >= 4:
            commits.append({
                "hash": parts[0][:8],
                "subject": parts[1],
                "author": parts[2],
                "email": parts[3],
            })
    return commits


def get_pr_meta(pr_branch: str, work_dir: str) -> tuple[str, str]:
    """Get PR title and author from the latest commit."""
    title_result = run_git(
        ["git", "log", "-1", "--pretty=format:%s", pr_branch], cwd=work_dir
    )
    author_result = run_git(
        ["git", "log", "-1", "--pretty=format:%an", pr_branch], cwd=work_dir
    )
    return title_result.stdout.strip(), author_result.stdout.strip()


def main(argv: list[str] | None = None) -> int:
    OUTPUT_LOGGER = init_output_logger()
    ERROR_LOGGER = init_error_logger()

    parser = argparse.ArgumentParser(
        description="One-shot PR context fetch (clone + diff + log).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--repo", required=True, help="Repository as owner/repo (e.g., cann/ops-math)")
    parser.add_argument("--pr", required=True, type=int, help="PR number")
    parser.add_argument("--work-dir", help="Custom working directory (default: auto-generated)")
    parser.add_argument("--keep", action="store_true", help="Keep working directory after completion")
    args = parser.parse_args(argv)

    if "/" not in args.repo:
        ERROR_LOGGER.error("ERROR: --repo must be in owner/repo format")
        return 1

    owner, repo = args.repo.split("/", 1)
    ts = int(time.time())
    work_dir = args.work_dir or os.path.join("/tmp", f"{TMP_PREFIX}{owner}_{repo}_{ts}")

    ctx = PRContext(work_dir=work_dir)

    try:
        clone_repo(owner, repo, work_dir)
        pr_branch = fetch_pr_branch(args.pr, work_dir)
        ctx.base_branch = get_base_branch(work_dir)
        ctx.merge_base = compute_merge_base(pr_branch, ctx.base_branch, work_dir)
        ctx.changed_files = get_diff_stats(ctx.merge_base, pr_branch, work_dir)
        ctx.added_lines = sum(f["added"] for f in ctx.changed_files)
        ctx.deleted_lines = sum(f["deleted"] for f in ctx.changed_files)
        ctx.commits = get_commits(ctx.merge_base, pr_branch, work_dir)
        ctx.pr_title, ctx.pr_author = get_pr_meta(pr_branch, work_dir)
    except Exception as e:
        ERROR_LOGGER.error("ERROR: %s", e)
        return 1
    finally:
        if not args.keep and os.path.isdir(work_dir):
            shutil.rmtree(work_dir, ignore_errors=True)
            ERROR_LOGGER.info("Cleaned up: %s", work_dir)

    output = {
        "work_dir": ctx.work_dir,
        "base_branch": ctx.base_branch,
        "merge_base": ctx.merge_base,
        "pr_title": ctx.pr_title,
        "pr_author": ctx.pr_author,
        "total_added": ctx.added_lines,
        "total_deleted": ctx.deleted_lines,
        "changed_files": ctx.changed_files,
        "commits": ctx.commits,
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
