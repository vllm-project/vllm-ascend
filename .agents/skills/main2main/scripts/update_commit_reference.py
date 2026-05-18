#!/usr/bin/env python3
"""Replace a pinned vLLM commit reference across tracked vllm-ascend files.

The main2main pipeline updates the vLLM commit reference at every verified
step. This script makes that update deterministic and avoids broad shell
commands such as `grep | xargs sed`.

Usage:
    python3 update_commit_reference.py \\
      --ascend-path <path> \\
      --old-commit <40-char-sha> \\
      --new-commit <40-char-sha>

Output (stdout):
    JSON with files_updated and replacement_count.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

COMMIT_RE = re.compile(r"^[0-9a-fA-F]{40}$")


def _run_git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def _validate_commit(label: str, value: str) -> str:
    if not COMMIT_RE.fullmatch(value):
        print(f"Error: {label} must be a 40-character git SHA", file=sys.stderr)
        sys.exit(1)
    return value.lower()


def _tracked_files(repo: Path) -> list[Path]:
    output = _run_git(repo, "ls-files", "-z")
    files: list[Path] = []
    for item in output.split("\0"):
        if item:
            files.append(repo / item)
    return files


def _replace_in_tracked_files(
    repo: Path,
    old_commit: str,
    new_commit: str,
    dry_run: bool,
) -> list[dict[str, int | str]]:
    old_bytes = old_commit.encode("ascii")
    new_bytes = new_commit.encode("ascii")
    updated: list[dict[str, int | str]] = []

    for path in _tracked_files(repo):
        if not path.exists() or not path.is_file():
            continue
        data = path.read_bytes()
        if b"\0" in data:
            continue
        count = data.count(old_bytes)
        if count == 0:
            continue
        if not dry_run:
            path.write_bytes(data.replace(old_bytes, new_bytes))
        updated.append({
            "path": str(path.relative_to(repo)),
            "replacements": count,
        })

    return updated


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replace old vLLM commit references in tracked vllm-ascend files.",
    )
    parser.add_argument("--ascend-path", type=Path, required=True,
                        help="Path to the vllm-ascend repository")
    parser.add_argument("--old-commit", required=True,
                        help="Existing 40-character vLLM commit SHA")
    parser.add_argument("--new-commit", required=True,
                        help="Replacement 40-character vLLM commit SHA")
    parser.add_argument("--dry-run", action="store_true",
                        help="Report matching files without modifying them")
    args = parser.parse_args()

    repo = args.ascend_path
    if not (repo / ".git").exists():
        print(f"Error: {repo} is not a git repository", file=sys.stderr)
        sys.exit(1)

    old_commit = _validate_commit("old-commit", args.old_commit)
    new_commit = _validate_commit("new-commit", args.new_commit)
    if old_commit == new_commit:
        print("Error: old-commit and new-commit are identical", file=sys.stderr)
        sys.exit(1)

    updated = _replace_in_tracked_files(repo, old_commit, new_commit, args.dry_run)
    if not updated:
        print(f"Error: old commit not found in tracked files: {old_commit}", file=sys.stderr)
        sys.exit(1)

    result = {
        "old_commit": old_commit,
        "new_commit": new_commit,
        "dry_run": args.dry_run,
        "files_updated": [item["path"] for item in updated],
        "replacement_count": sum(int(item["replacements"]) for item in updated),
        "replacements": updated,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
