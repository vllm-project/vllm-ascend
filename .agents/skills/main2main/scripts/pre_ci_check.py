#!/usr/bin/env python3
"""Pre-CI verification checks for main2main steps.

Runs two mechanical checks before CI to catch common multi-step errors:
  1. Version string consistency: newly added vllm_version_is() calls use
     the correct release tag (scoped to current diff, not the whole repo).
  2. Temp file cleanliness: no intermediate files in the repository.

Usage:
    python3 pre_ci_check.py \
      --ascend-path <path> \
      --release-tag <version>

Output (stdout):
    JSON with check results and overall pass/fail.
    Exits 0 if all checks pass, 1 if any check fails.

Design note:
    The version string check only examines lines ADDED in the current diff
    (git diff HEAD), not the entire repo. Previous main2main runs leave
    behind guards like vllm_version_is("0.20.2") that are correct for that
    version boundary. Scanning the full repo would flag all historical guards
    as mismatches whenever the release tag advances.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

TEMP_PATTERNS = [
    ".log",
    ".patch",
    ".jsonl",
    "vllm_changes.md",
    "vllm_error_analyze.md",
    "round-ledger",
    "main2main-failure-summary",
    "ci-summary",
]

VERSION_IS_RE = re.compile(r'vllm_version_is\(\s*["\']([^"\']+)["\']\s*\)')


def _run_git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def _get_added_lines(repo: Path) -> list[dict[str, str]]:
    """Return lines added in the working tree diff (unstaged + staged vs HEAD).

    Each entry has 'file', 'line_no', and 'text'.
    """
    diff_output = _run_git(repo, "diff", "HEAD", "-U0")
    added: list[dict[str, str]] = []
    current_file = None
    current_line = 0

    for line in diff_output.splitlines():
        if line.startswith("+++ b/"):
            current_file = line[6:]
        elif line.startswith("@@ "):
            match = re.search(r"\+(\d+)", line)
            if match:
                current_line = int(match.group(1))
        elif line.startswith("+") and not line.startswith("+++"):
            if current_file:
                added.append(
                    {
                        "file": current_file,
                        "line_no": str(current_line),
                        "text": line[1:],
                    }
                )
            current_line += 1
        elif not line.startswith("-"):
            current_line += 1

    return added


def _check_version_strings(added_lines: list[dict[str, str]], release_tag: str) -> dict:
    """Check that newly added vllm_version_is() calls use the current release tag.

    Only examines lines added in the current diff. Historical guards from
    previous main2main runs are not flagged.
    """
    new_calls: list[dict[str, str]] = []
    mismatched: list[dict[str, str]] = []

    for entry in added_lines:
        text = entry["text"]
        if "import " in text or "def " in text:
            continue
        match = VERSION_IS_RE.search(text)
        if not match:
            continue
        version_used = match.group(1)
        call_info = {
            "file": entry["file"],
            "line": entry["line_no"],
            "version_used": version_used,
            "text": text.strip(),
        }
        new_calls.append(call_info)
        if version_used != release_tag:
            mismatched.append(call_info)

    return {
        "release_tag": release_tag,
        "new_calls_count": len(new_calls),
        "mismatched": mismatched,
    }


def _check_temp_files(repo: Path) -> dict:
    """Check that no temporary files exist in the repository working tree."""
    status_output = _run_git(repo, "status", "--short")
    untracked_output = _run_git(repo, "ls-files", "--others", "--exclude-standard")

    all_files = set()
    for line in (status_output + untracked_output).strip().splitlines():
        filepath = line.strip().lstrip("MADRCU?! ").strip()
        if filepath:
            all_files.add(filepath)

    violations: list[str] = []
    for filepath in sorted(all_files):
        basename = Path(filepath).name
        for pattern in TEMP_PATTERNS:
            if pattern in basename or basename.endswith(pattern):
                violations.append(filepath)
                break

    return {
        "violations": violations,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run pre-CI verification checks for main2main.",
    )
    parser.add_argument("--ascend-path", type=Path, required=True, help="Path to the vllm-ascend repository")
    parser.add_argument(
        "--release-tag", required=True, help="Expected release version string (main_vllm_tag from conf.py)"
    )
    args = parser.parse_args()

    repo = args.ascend_path
    if not (repo / ".git").exists():
        print(f"Error: {repo} is not a git repository", file=sys.stderr)
        sys.exit(1)

    added_lines = _get_added_lines(repo)
    versions = _check_version_strings(added_lines, args.release_tag)
    temps = _check_temp_files(repo)

    all_passed = True
    checks: list[dict] = []

    # Check 1: version strings in new code only
    version_ok = len(versions["mismatched"]) == 0
    checks.append(
        {
            "name": "version_strings",
            "passed": version_ok,
            "detail": (
                f"{versions['new_calls_count']} new vllm_version_is() calls all use {args.release_tag}"
                if version_ok
                else f"{len(versions['mismatched'])} new vllm_version_is() calls use wrong version"
            ),
            "mismatched": versions["mismatched"],
        }
    )
    if not version_ok:
        all_passed = False

    # Check 2: temp files
    temp_ok = len(temps["violations"]) == 0
    checks.append(
        {
            "name": "temp_files",
            "passed": temp_ok,
            "detail": ("no temp files in repo" if temp_ok else f"{len(temps['violations'])} temp files found in repo"),
            "violations": temps["violations"],
        }
    )
    if not temp_ok:
        all_passed = False

    result = {
        "all_passed": all_passed,
        "checks": checks,
    }

    print(json.dumps(result, indent=2))
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
