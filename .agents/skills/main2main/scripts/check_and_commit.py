#!/usr/bin/env python3
"""Validate working tree changes and create a signed commit.

This script enforces the main2main guardrails before committing:
  - All changed files must be inside the ascend-path repository.
  - No temporary/intermediate files should be staged (logs, patches, etc.).
  - No untracked temporary files should exist in the repo.

Usage:
    python3 check_and_commit.py \\
      --ascend-path <path> \\
      --step-id <id> \\
      --message "<commit message>"

Output (stdout):
    JSON with commit_sha and files_committed on success.
    Exits with code 1 and an error description on validation failure.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

# File patterns that should never appear in the repo working tree.
FORBIDDEN_PATTERNS = [
    "vllm_changes.md",
    "vllm_error_analyze.md",
    ".log",
    ".patch",
    ".jsonl",
    "round-ledger",
    "main2main-failure-summary",
    "ci-summary",
]


def _run_git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def _get_tracked_changed_files(repo: Path) -> list[str]:
    """Return tracked modified/deleted/staged files in the working tree."""
    output = _run_git(repo, "diff", "--name-only", "HEAD")
    staged = _run_git(repo, "diff", "--name-only", "--cached")
    all_files = set()
    for line in (output + staged).strip().splitlines():
        if line.strip():
            all_files.add(line.strip())
    return sorted(all_files)


def _get_untracked_files(repo: Path) -> list[str]:
    """Return list of untracked files."""
    output = _run_git(repo, "ls-files", "--others", "--exclude-standard")
    return [f for f in output.strip().splitlines() if f.strip()]


def _check_forbidden(files: list[str]) -> list[str]:
    """Return list of files that match forbidden patterns."""
    violations = []
    for f in files:
        basename = Path(f).name
        for pattern in FORBIDDEN_PATTERNS:
            if pattern in basename or basename.endswith(pattern):
                violations.append(f)
                break
    return violations


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate and commit changes for a main2main step.",
    )
    parser.add_argument("--ascend-path", type=Path, required=True,
                        help="Path to the vllm-ascend repository")
    parser.add_argument("--step-id", required=True,
                        help="Step identifier for the commit message context")
    parser.add_argument("--message", required=True,
                        help="Commit message")
    args = parser.parse_args()

    repo = args.ascend_path
    if not (repo / ".git").exists():
        print(f"Error: {repo} is not a git repository", file=sys.stderr)
        sys.exit(1)

    # Check for changed files, including newly-created source files.
    tracked_changed = _get_tracked_changed_files(repo)
    untracked = _get_untracked_files(repo)
    changed = sorted(set(tracked_changed + untracked))

    # Check for forbidden files before the no-op case so stray logs fail loudly.
    forbidden = _check_forbidden(changed)
    if forbidden:
        print(
            f"Error: forbidden files in working tree:\n"
            + "\n".join(f"  - {f}" for f in forbidden),
            file=sys.stderr,
        )
        sys.exit(1)

    if not changed:
        print("Error: no changes to commit", file=sys.stderr)
        sys.exit(1)

    # Stage specific files (not git add .).
    for f in changed:
        _run_git(repo, "add", f)

    # Commit with sign-off
    _run_git(repo, "commit", "-s", "-m", args.message)

    # Get the commit SHA
    sha = _run_git(repo, "rev-parse", "HEAD").strip()

    result = {
        "commit_sha": sha,
        "step_id": args.step_id,
        "files_committed": changed,
    }

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
