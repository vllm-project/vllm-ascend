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

# =============================================================================
# Common install commands shared across e2e environments
# =============================================================================
_E2E_SYS_DEPS = (
    "apt install git -y"
    " && apt-get -y install $(cat packages.txt)"
    " && apt-get -y install gcc g++ cmake libnuma-dev clang-15"
    " && update-alternatives --install /usr/bin/clang clang /usr/bin/clang-15 20"
    " && update-alternatives --install /usr/bin/clang++ clang++ /usr/bin/clang++-15 20"
)
_E2E_VLLM_INSTALL = "VLLM_TARGET_DEVICE=empty pip install -e ."
_E2E_ASCEND_INSTALL = (
    "export PIP_EXTRA_INDEX_URL=https://mirrors.huaweicloud.com/ascend/repos/pypi"
    " && pip install --no-cache-dir -r requirements-dev.txt"
    " && pip install --no-cache-dir -v -e ."
)
_E2E_CONTAINER_ENV = {
    "VLLM_LOGGING_LEVEL": "ERROR",
    "VLLM_USE_MODELSCOPE": "True",
    "HF_HUB_OFFLINE": "1",
}
_E2E_RUNTIME_ENV = {
    "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
    "PYTORCH_NPU_ALLOC_CONF": "max_split_size_mb:256",
}

# =============================================================================
# Environment rules: one entry per (test path pattern → full env config).
#
# To add a new test type or modify an existing one, edit ONLY this list.
# The workflow YAML is a generic template that reads all values from the matrix.
# =============================================================================
ENV_RULES = [
    {
        "pattern": r"tests/e2e/310p/multicard/",
        "runner": "linux-aarch64-310p-4",
        "image": "swr.cn-southwest-2.myhuaweicloud.com/base_image/ascend-ci/cann:8.5.0-310p-ubuntu22.04-py3.11",
        "test_type": "e2e",
        "container_env": {**_E2E_CONTAINER_ENV},
        "sys_deps": (
            "apt install git -y"
            " && apt-get -y install $(cat packages.txt)"
            " && apt-get -y install gcc g++ cmake libnuma-dev"
        ),
        "vllm_install": _E2E_VLLM_INSTALL,
        "ascend_install": _E2E_ASCEND_INSTALL,
        "runtime_env": {**_E2E_RUNTIME_ENV},
    },
    {
        "pattern": r"tests/e2e/310p/",
        "runner": "linux-aarch64-310p-1",
        "image": "swr.cn-southwest-2.myhuaweicloud.com/base_image/ascend-ci/cann:8.5.0-310p-ubuntu22.04-py3.11",
        "test_type": "e2e",
        "container_env": {**_E2E_CONTAINER_ENV},
        "sys_deps": (
            "apt install git -y"
            " && apt-get -y install $(cat packages.txt)"
            " && apt-get -y install gcc g++ cmake libnuma-dev"
        ),
        "vllm_install": _E2E_VLLM_INSTALL,
        "ascend_install": _E2E_ASCEND_INSTALL,
        "runtime_env": {**_E2E_RUNTIME_ENV},
    },
    {
        "pattern": r"tests/e2e/multicard/4-cards/",
        "runner": "linux-aarch64-a3-4",
        "image": "m.daocloud.io/quay.io/ascend/cann:8.5.0-a3-ubuntu22.04-py3.11",
        "test_type": "e2e",
        "container_env": {**_E2E_CONTAINER_ENV},
        "sys_deps": _E2E_SYS_DEPS,
        "vllm_install": _E2E_VLLM_INSTALL,
        "ascend_install": _E2E_ASCEND_INSTALL,
        "runtime_env": {**_E2E_RUNTIME_ENV},
    },
    {
        "pattern": r"tests/e2e/multicard/2-cards/",
        "runner": "linux-aarch64-a3-2",
        "image": "swr.cn-southwest-2.myhuaweicloud.com/base_image/ascend-ci/cann:8.5.0-a3-ubuntu22.04-py3.11",
        "test_type": "e2e",
        "container_env": {**_E2E_CONTAINER_ENV, "HCCL_BUFFSIZE": "1024"},
        "sys_deps": _E2E_SYS_DEPS,
        "vllm_install": _E2E_VLLM_INSTALL,
        "ascend_install": _E2E_ASCEND_INSTALL,
        "runtime_env": {**_E2E_RUNTIME_ENV},
    },
    {
        "pattern": r"tests/e2e/singlecard/",
        "runner": "linux-aarch64-a2b3-1",
        "image": "swr.cn-southwest-2.myhuaweicloud.com/base_image/ascend-ci/cann:8.5.0-910b-ubuntu22.04-py3.11",
        "test_type": "e2e",
        "container_env": {**_E2E_CONTAINER_ENV},
        "sys_deps": _E2E_SYS_DEPS,
        "vllm_install": _E2E_VLLM_INSTALL,
        "ascend_install": _E2E_ASCEND_INSTALL,
        "runtime_env": {**_E2E_RUNTIME_ENV},
    },
    {
        "pattern": r"tests/ut/",
        "runner": "linux-amd64-cpu-8-hk",
        "image": "quay.nju.edu.cn/ascend/cann:8.5.0-910b-ubuntu22.04-py3.11",
        "test_type": "ut",
        "container_env": {
            "VLLM_LOGGING_LEVEL": "ERROR",
            "VLLM_USE_MODELSCOPE": "True",
            "HF_HUB_OFFLINE": "1",
            "SOC_VERSION": "ascend910b1",
            "MAX_JOBS": "4",
            "COMPILE_CUSTOM_KERNELS": "0",
        },
        "sys_deps": ("apt-get install -y python3-pip git vim wget net-tools gcc g++ cmake libnuma-dev curl gnupg2"),
        "vllm_install": (
            "VLLM_TARGET_DEVICE=empty python3 -m pip install ."
            " --extra-index https://download.pytorch.org/whl/cpu/"
            " && python3 -m pip uninstall -y triton"
        ),
        "ascend_install": (
            "export PIP_EXTRA_INDEX_URL=https://mirrors.huaweicloud.com/ascend/repos/pypi"
            " && export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/Ascend/ascend-toolkit/latest/x86_64-linux/devlib"
            " && python3 -m pip install -v . --extra-index https://download.pytorch.org/whl/cpu/"
            " && python3 -m pip install -r requirements-dev.txt --extra-index https://download.pytorch.org/whl/cpu/"
        ),
        "runtime_env": {
            "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
            "PYTORCH_NPU_ALLOC_CONF": "max_split_size_mb:256",
            "TORCH_DEVICE_BACKEND_AUTOLOAD": "0",
            "LD_LIBRARY_PATH": "/usr/local/Ascend/ascend-toolkit/latest/x86_64-linux/devlib",
        },
    },
]

# Default fallback
DEFAULT_RUNNER = "linux-aarch64-a3-4"
DEFAULT_IMAGE = "m.daocloud.io/quay.io/ascend/cann:8.5.0-a3-ubuntu22.04-py3.11"

# All possible container_env keys across all rules (used by workflow YAML)
ALL_CONTAINER_ENV_KEYS = sorted({k for rule in ENV_RULES for k in rule.get("container_env", {})})

# All possible runtime_env keys across all rules
ALL_RUNTIME_ENV_KEYS = sorted({k for rule in ENV_RULES for k in rule.get("runtime_env", {})})

# Regex to match a 7+ hex-char commit hash (not a vX.Y.Z tag)
COMMIT_HASH_RE = re.compile(r"^[0-9a-f]{7,40}$")


def detect_env(test_cmd: str) -> dict:
    """Detect full environment config based on the test file path in test_cmd."""
    for rule in ENV_RULES:
        if re.search(rule["pattern"], test_cmd):
            return {k: v for k, v in rule.items() if k != "pattern"}
    # Default fallback: use singlecard e2e config
    for rule in ENV_RULES:
        if rule.get("test_type") == "e2e" and "singlecard" in rule["pattern"]:
            return {k: v for k, v in rule.items() if k != "pattern"}
    return {"runner": DEFAULT_RUNNER, "image": DEFAULT_IMAGE, "test_type": "e2e"}


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
        "| Field | Value |",
        "|-------|-------|",
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
            lines.append(f"| {i} | `{entry.get('commit', '?')[:12]}` | {entry.get('result', '?')} |")

    lines += [
        "",
        "---",
        "*Generated by `tools/bisect_vllm.sh`*",
    ]
    return "\n".join(lines)


def build_batch_matrix(test_cmds_str: str) -> dict:
    """Parse semicolon-separated test commands and group by (runner, image, test_type).

    Returns a GitHub Actions matrix JSON object with an "include" array.
    Each element contains the full environment config needed by the workflow:
    group, runner, image, test_type, test_cmds, container_env, sys_deps,
    vllm_install, ascend_install, runtime_env.
    """
    cmds = [c.strip() for c in test_cmds_str.split(";") if c.strip()]
    if not cmds:
        return {"include": []}

    # Group by (runner, image, test_type) — commands sharing the same env
    groups: dict[tuple[str, str, str], list[str]] = {}
    group_env: dict[tuple[str, str, str], dict] = {}
    for cmd in cmds:
        env = detect_env(cmd)
        key = (env["runner"], env["image"], env["test_type"])
        groups.setdefault(key, []).append(cmd)
        if key not in group_env:
            group_env[key] = env
        else:
            # Merge container_env and runtime_env from all commands in group
            for field in ("container_env", "runtime_env"):
                existing = group_env[key].get(field, {})
                existing.update(env.get(field, {}))
                group_env[key][field] = existing

    # Build matrix include array
    include = []
    for (runner, image, test_type), group_cmds in groups.items():
        env = group_env[(runner, image, test_type)]
        group_name = f"{test_type}-{runner.split('-')[-1]}"

        # Flatten container_env into individual matrix keys (for YAML static refs)
        # Fill all known keys with empty string if not present in this env
        container_env = env.get("container_env", {})
        entry = {
            "group": group_name,
            "runner": runner,
            "image": image,
            "test_type": test_type,
            "test_cmds": ";".join(group_cmds),
            "sys_deps": env.get("sys_deps", "echo 'no sys_deps configured'"),
            "vllm_install": env.get("vllm_install", "echo 'no vllm_install configured'"),
            "ascend_install": env.get("ascend_install", "echo 'no ascend_install configured'"),
            "runtime_env": json.dumps(env.get("runtime_env", {})),
        }
        # Add each container_env key as a top-level matrix field (cenv_XXX)
        for k in ALL_CONTAINER_ENV_KEYS:
            entry[f"cenv_{k}"] = container_env.get(k, "")

        include.append(entry)

    include.sort(
        key=lambda e: (
            e["test_type"],
            int(re.search(r"-(\d+)$", e["group"]).group(1)) if re.search(r"-(\\d+)$", e["group"]) else 9999,
            e["group"],
        )
    )
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
        total_cmds = sum(len(g["test_cmds"].split(";")) for g in matrix["include"])
        print(f"Total: {len(matrix['include'])} group(s) from {total_cmds} command(s)")
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
            yaml_path = os.path.join(repo_root, ".github/workflows/pr_test_light.yaml")
        except subprocess.CalledProcessError:
            print("ERROR: Cannot determine repo root", file=sys.stderr)
            sys.exit(1)

    commit = get_commit_from_yaml(yaml_path, ref=args.ref)
    if commit:
        print(commit)
    else:
        print(
            f"ERROR: Could not extract vllm commit from {yaml_path}" + (f" at ref {args.ref}" if args.ref else ""),
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
    parser = argparse.ArgumentParser(description="Helper for vllm bisect automation")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # detect-env
    p_env = subparsers.add_parser("detect-env", help="Detect runner and image for a test command")
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
        "--test-cmds",
        required=True,
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
    p_commit = subparsers.add_parser("get-commit", help="Extract vllm commit from workflow yaml")
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
    p_report = subparsers.add_parser("report", help="Generate bisect result report")
    p_report.add_argument("--good-commit", required=True)
    p_report.add_argument("--bad-commit", required=True)
    p_report.add_argument("--first-bad", required=True)
    p_report.add_argument(
        "--first-bad-info", default=None, help="Commit info string (mutually exclusive with --first-bad-info-file)"
    )
    p_report.add_argument("--first-bad-info-file", default=None, help="File containing commit info")
    p_report.add_argument("--test-cmd", required=True)
    p_report.add_argument("--total-steps", type=int, required=True)
    p_report.add_argument("--total-commits", type=int, required=True)
    p_report.add_argument("--skipped", default=None, help="Comma-separated skipped commits")
    p_report.add_argument("--log-file", default=None, help="Path to bisect log JSON file")
    p_report.set_defaults(func=cmd_report)

    # vllm-location
    p_loc = subparsers.add_parser("vllm-location", help="Get vllm install location via pip show")
    p_loc.set_defaults(func=cmd_vllm_location)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
