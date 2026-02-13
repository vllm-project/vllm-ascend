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
Helper script for bisect_vllm.sh.

Subcommands:
  detect-env   - Detect runner and image based on test command path.
  get-commit   - Extract vllm commit hash from a workflow yaml file.
  report       - Generate a markdown bisect report.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

# Runner / image mapping, derived from _e2e_test.yaml and _unit_test.yaml
ENV_RULES = [
    {
        "pattern": r"tests/e2e/310p/multicard/",
        "runner": "linux-aarch64-310p-4",
        "image": "swr.cn-southwest-2.myhuaweicloud.com/base_image/ascend-ci/cann:8.5.0-310p-ubuntu22.04-py3.11",
        "test_type": "e2e",
        "extra_env": {},
    },
    {
        "pattern": r"tests/e2e/310p/",
        "runner": "linux-aarch64-310p-1",
        "image": "swr.cn-southwest-2.myhuaweicloud.com/base_image/ascend-ci/cann:8.5.0-310p-ubuntu22.04-py3.11",
        "test_type": "e2e",
        "extra_env": {},
    },
    {
        "pattern": r"tests/e2e/multicard/4-cards/",
        "runner": "linux-aarch64-a3-4",
        "image": "m.daocloud.io/quay.io/ascend/cann:8.5.0-a3-ubuntu22.04-py3.11",
        "test_type": "e2e",
        "extra_env": {},
    },
    {
        "pattern": r"tests/e2e/multicard/2-cards/",
        "runner": "linux-aarch64-a3-2",
        "image": "swr.cn-southwest-2.myhuaweicloud.com/base_image/ascend-ci/cann:8.5.0-a3-ubuntu22.04-py3.11",
        "test_type": "e2e",
        "extra_env": {"HCCL_BUFFSIZE": "1024"},
    },
    {
        "pattern": r"tests/e2e/singlecard/",
        "runner": "linux-aarch64-a2b3-1",
        "image": "swr.cn-southwest-2.myhuaweicloud.com/base_image/ascend-ci/cann:8.5.0-910b-ubuntu22.04-py3.11",
        "test_type": "e2e",
        "extra_env": {},
    },
    {
        "pattern": r"tests/ut/",
        "runner": "linux-amd64-cpu-8-hk",
        "image": "quay.nju.edu.cn/ascend/cann:8.5.0-910b-ubuntu22.04-py3.11",
        "test_type": "ut",
        "extra_env": {},
    },
]

# Default fallback
DEFAULT_RUNNER = "linux-aarch64-a3-4"
DEFAULT_IMAGE = "m.daocloud.io/quay.io/ascend/cann:8.5.0-a3-ubuntu22.04-py3.11"

# Regex to match a 7+ hex-char commit hash (not a vX.Y.Z tag)
COMMIT_HASH_RE = re.compile(r"^[0-9a-f]{7,40}$")


def detect_env(test_cmd: str) -> dict:
    """Detect runner and image based on the test file path in test_cmd."""
    for rule in ENV_RULES:
        if re.search(rule["pattern"], test_cmd):
            return {
                "runner": rule["runner"],
                "image": rule["image"],
                "test_type": rule["test_type"],
                "extra_env": rule.get("extra_env", {}),
            }
    return {"runner": DEFAULT_RUNNER, "image": DEFAULT_IMAGE, "test_type": "e2e", "extra_env": {}}


def get_commit_from_yaml(yaml_path: str, ref: str | None = None) -> str | None:
    """Extract vllm commit hash from a workflow yaml file.

    Reads the file content either from disk (ref=None) or from a git ref
    (e.g. ref='origin/main') via ``git show ref:path``.

    Looks for the vllm_version matrix pattern like:
        vllm_version: [<commit_hash>, v0.15.0]
    and returns the commit hash entry (the one that is NOT a vX.Y.Z tag).
    """
    if ref:
        # Read from git ref
        try:
            # Compute relative path from repo root
            repo_root = subprocess.check_output(
                ["git", "rev-parse", "--show-toplevel"],
                text=True,
            ).strip()
            rel_path = os.path.relpath(yaml_path, repo_root)
            content = subprocess.check_output(
                ["git", "show", f"{ref}:{rel_path}"],
                text=True,
                stderr=subprocess.DEVNULL,
            )
        except subprocess.CalledProcessError:
            return None
    else:
        try:
            content = Path(yaml_path).read_text()
        except FileNotFoundError:
            return None

    # Match patterns like: vllm_version: [abc123, v0.15.0]
    # or multi-line matrix definitions
    match = re.search(
        r"vllm_version:\s*\[([^\]]+)\]",
        content,
    )
    if not match:
        return None

    entries = [e.strip().strip("'\"") for e in match.group(1).split(",")]
    for entry in entries:
        if COMMIT_HASH_RE.match(entry):
            return entry
    return None


def get_pkg_location(pkg_name: str) -> str | None:
    """Get package install location via pip show.

    For editable installs, prefers ``Editable project location`` which
    points directly to the source tree.  Falls back to ``Location``
    (site-packages directory) for regular installs.
    """
    try:
        output = subprocess.check_output(
            ["pip", "show", pkg_name],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        editable_loc = None
        location = None
        for line in output.splitlines():
            if line.startswith("Editable project location:"):
                editable_loc = line.split(":", 1)[1].strip()
            elif line.startswith("Location:"):
                location = line.split(":", 1)[1].strip()
        # Prefer editable location (source tree) over site-packages
        return editable_loc or location
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    return None


def generate_report(
    bad_commit: str,
    good_commit: str,
    first_bad: str,
    first_bad_info: str,
    test_cmd: str,
    total_steps: int,
    total_commits: int,
    skipped: list[str] | None = None,
    log_entries: list[dict] | None = None,
) -> str:
    """Generate a markdown bisect report."""
    lines = [
        "## Bisect Result",
        "",
        f"| Field | Value |",
        f"|-------|-------|",
        f"| First bad commit | `{first_bad}` |",
        f"| Link | https://github.com/vllm-project/vllm/commit/{first_bad} |",
        f"| Good commit | `{good_commit}` |",
        f"| Bad commit | `{bad_commit}` |",
        f"| Range | {total_commits} commits, {total_steps} bisect steps |",
        f"| Test command | `{test_cmd}` |",
        "",
        "### First Bad Commit Details",
        "```",
        first_bad_info,
        "```",
    ]

    if skipped:
        lines += [
            "",
            "### Skipped Commits",
            "",
        ]
        for s in skipped:
            lines.append(f"- `{s}`")

    if log_entries:
        lines += [
            "",
            "### Bisect Log",
            "",
            "| Step | Commit | Result |",
            "|------|--------|--------|",
        ]
        for i, entry in enumerate(log_entries, 1):
            lines.append(
                f"| {i} | `{entry.get('commit', '?')[:12]}` | {entry.get('result', '?')} |"
            )

    lines += [
        "",
        "---",
        "*Generated by `tools/bisect_vllm.sh`*",
    ]
    return "\n".join(lines)


def build_batch_matrix(test_cmds_str: str) -> dict:
    """Parse semicolon-separated test commands and group by (runner, image, test_type).

    Returns a GitHub Actions matrix JSON object with an "include" array.
    Each element has: group, runner, image, test_type, test_cmds (semicolon-joined),
    and extra_env (merged from all commands in the group, JSON string).
    """
    cmds = [c.strip() for c in test_cmds_str.split(";") if c.strip()]
    if not cmds:
        return {"include": []}

    # Group by environment
    groups: dict[tuple[str, str, str], list[str]] = {}
    group_extra_env: dict[tuple[str, str, str], dict] = {}
    for cmd in cmds:
        env = detect_env(cmd)
        key = (env["runner"], env["image"], env["test_type"])
        groups.setdefault(key, []).append(cmd)
        # Merge extra_env from all commands in the group
        merged = group_extra_env.setdefault(key, {})
        merged.update(env.get("extra_env", {}))

    # Build matrix include array
    include = []
    for (runner, image, test_type), group_cmds in groups.items():
        # Generate a human-readable group name from the runner
        group_name = f"{test_type}-{runner.split('-')[-1]}"
        entry = {
            "group": group_name,
            "runner": runner,
            "image": image,
            "test_type": test_type,
            "test_cmds": ";".join(group_cmds),
        }
        extra = group_extra_env.get((runner, image, test_type), {})
        if extra:
            entry["extra_env"] = json.dumps(extra)
        include.append(entry)

    return {"include": include}


def cmd_detect_env(args):
    env = detect_env(args.test_cmd)
    if args.output_format == "github":
        # Write to GITHUB_OUTPUT if available
        github_output = os.environ.get("GITHUB_OUTPUT")
        if github_output:
            with open(github_output, "a") as f:
                f.write(f"runner={env['runner']}\n")
                f.write(f"image={env['image']}\n")
                f.write(f"test_type={env['test_type']}\n")
        # Also print for human readability
        print(f"runner={env['runner']}")
        print(f"image={env['image']}")
        print(f"test_type={env['test_type']}")
    else:
        print(json.dumps(env))


def cmd_batch_matrix(args):
    matrix = build_batch_matrix(args.test_cmds)
    matrix_json = json.dumps(matrix, separators=(",", ":"))
    if args.output_format == "github":
        github_output = os.environ.get("GITHUB_OUTPUT")
        if github_output:
            with open(github_output, "a") as f:
                f.write(f"matrix={matrix_json}\n")
        print(f"matrix={matrix_json}")
        print(f"Total: {len(matrix['include'])} group(s) from {sum(len(g['test_cmds'].split(';')) for g in matrix['include'])} command(s)")
    else:
        print(json.dumps(matrix, indent=2))


def cmd_get_commit(args):
    yaml_path = args.yaml_path
    if not yaml_path:
        # Default: pr_test_light.yaml relative to this script's repo
        try:
            repo_root = subprocess.check_output(
                ["git", "rev-parse", "--show-toplevel"],
                text=True,
            ).strip()
            yaml_path = os.path.join(
                repo_root, ".github/workflows/pr_test_light.yaml"
            )
        except subprocess.CalledProcessError:
            print("ERROR: Cannot determine repo root", file=sys.stderr)
            sys.exit(1)

    commit = get_commit_from_yaml(yaml_path, ref=args.ref)
    if commit:
        print(commit)
    else:
        print(
            f"ERROR: Could not extract vllm commit from {yaml_path}"
            + (f" at ref {args.ref}" if args.ref else ""),
            file=sys.stderr,
        )
        sys.exit(1)


def cmd_report(args):
    skipped = args.skipped.split(",") if args.skipped else None
    log_entries = None
    if args.log_file:
        try:
            with open(args.log_file) as f:
                log_entries = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            pass

    # Read first_bad_info from file or argument
    first_bad_info = args.first_bad_info or ""
    if args.first_bad_info_file:
        try:
            first_bad_info = Path(args.first_bad_info_file).read_text().strip()
        except FileNotFoundError:
            first_bad_info = "N/A"

    report = generate_report(
        bad_commit=args.bad_commit,
        good_commit=args.good_commit,
        first_bad=args.first_bad,
        first_bad_info=first_bad_info,
        test_cmd=args.test_cmd,
        total_steps=args.total_steps,
        total_commits=args.total_commits,
        skipped=skipped,
        log_entries=log_entries,
    )
    print(report)

    # Write to GITHUB_STEP_SUMMARY if available
    summary_file = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_file:
        with open(summary_file, "a") as f:
            f.write(report + "\n")


def cmd_vllm_location(args):
    loc = get_pkg_location("vllm")
    if loc:
        print(loc)
    else:
        print("ERROR: vllm not installed or pip show failed", file=sys.stderr)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Helper for vllm bisect automation"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # detect-env
    p_env = subparsers.add_parser(
        "detect-env", help="Detect runner and image for a test command"
    )
    p_env.add_argument("--test-cmd", required=True, help="The pytest command")
    p_env.add_argument(
        "--output-format",
        choices=["json", "github"],
        default="github",
        help="Output format (default: github)",
    )
    p_env.set_defaults(func=cmd_detect_env)

    # batch-matrix
    p_batch = subparsers.add_parser(
        "batch-matrix",
        help="Build a GitHub Actions matrix from semicolon-separated test commands",
    )
    p_batch.add_argument(
        "--test-cmds", required=True,
        help="Semicolon-separated test commands",
    )
    p_batch.add_argument(
        "--output-format",
        choices=["json", "github"],
        default="github",
        help="Output format (default: github)",
    )
    p_batch.set_defaults(func=cmd_batch_matrix)

    # get-commit
    p_commit = subparsers.add_parser(
        "get-commit", help="Extract vllm commit from workflow yaml"
    )
    p_commit.add_argument(
        "--yaml-path",
        default="",
        help="Path to workflow yaml (default: pr_test_light.yaml)",
    )
    p_commit.add_argument(
        "--ref",
        default=None,
        help="Git ref to read from (e.g. origin/main). If unset, reads from disk.",
    )
    p_commit.set_defaults(func=cmd_get_commit)

    # report
    p_report = subparsers.add_parser(
        "report", help="Generate bisect result report"
    )
    p_report.add_argument("--good-commit", required=True)
    p_report.add_argument("--bad-commit", required=True)
    p_report.add_argument("--first-bad", required=True)
    p_report.add_argument("--first-bad-info", default=None, help="Commit info string (mutually exclusive with --first-bad-info-file)")
    p_report.add_argument("--first-bad-info-file", default=None, help="File containing commit info")
    p_report.add_argument("--test-cmd", required=True)
    p_report.add_argument("--total-steps", type=int, required=True)
    p_report.add_argument("--total-commits", type=int, required=True)
    p_report.add_argument("--skipped", default=None, help="Comma-separated skipped commits")
    p_report.add_argument("--log-file", default=None, help="Path to bisect log JSON file")
    p_report.set_defaults(func=cmd_report)

    # vllm-location
    p_loc = subparsers.add_parser(
        "vllm-location", help="Get vllm install location via pip show"
    )
    p_loc.set_defaults(func=cmd_vllm_location)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
