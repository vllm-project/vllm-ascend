# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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
"""Helpers for running the interface analyzer from a vLLM PR job."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

VLLM_REPOSITORY_URL = "https://github.com/vllm-project/vllm.git"
VLLM_BASE_BRANCH = "main"


def _git(root: Path, *args: str) -> str:
    command = ["git", "-C", str(root), *args]
    try:
        return subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except subprocess.CalledProcessError as error:
        details = [
            f"Git command failed with exit code {error.returncode}:",
            subprocess.list2cmdline(command),
        ]
        stdout = (error.stdout or "").strip()
        stderr = (error.stderr or "").strip()
        if stdout:
            details.extend(("stdout:", stdout))
        if stderr:
            details.extend(("stderr:", stderr))
        if not stdout and not stderr:
            details.append("The Git command produced no output.")
        raise RuntimeError("\n".join(details)) from None


def resolve_vllm_range(
    vllm_root: Path,
    *,
    repository_url: str = VLLM_REPOSITORY_URL,
    base_branch: str = VLLM_BASE_BRANCH,
) -> tuple[str, str]:
    """Resolve the exact merge-base-to-HEAD range for this vLLM PR."""
    if not vllm_root.is_dir():
        raise ValueError(f"vLLM checkout does not exist: {vllm_root}")
    if not (vllm_root / ".git").exists():
        raise ValueError(f"vLLM checkout has no Git metadata: {vllm_root}")

    new_sha = _git(vllm_root, "rev-parse", "HEAD")
    _git(vllm_root, "fetch", "--no-tags", repository_url, base_branch)
    try:
        old_sha = _git(vllm_root, "merge-base", new_sha, "FETCH_HEAD")
    except subprocess.CalledProcessError:
        if _git(vllm_root, "rev-parse", "--is-shallow-repository") != "true":
            raise
        _git(vllm_root, "fetch", "--no-tags", "--unshallow", repository_url, base_branch)
        old_sha = _git(vllm_root, "merge-base", new_sha, "FETCH_HEAD")

    _git(vllm_root, "merge-base", "--is-ancestor", old_sha, new_sha)
    return old_sha, new_sha


def build_analysis_command(
    *,
    vllm_root: Path,
    ascend_root: Path,
    old_sha: str,
    new_sha: str,
    ascend_sha: str,
    analysis_workers: int = 3,
    index_workers: int = 1,
) -> list[str]:
    """Build the repository CLI command used by the vLLM PR pytest entry."""
    command = [
        sys.executable,
        "-m",
        "tests.e2e.vllm_interface.vllm_interface_contracts",
        "analyze-range",
        "--vllm-root",
        str(vllm_root),
        "--ascend-root",
        str(ascend_root),
        "--old",
        old_sha,
        "--new",
        new_sha,
        "--expect-ascend-sha",
        ascend_sha,
        "--fail-on",
        "introduced",
        "--analysis-workers",
        str(analysis_workers),
        "--index-workers",
        str(index_workers),
    ]
    return command


def run_analysis(command: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    """Run the analyzer while retaining output for the pytest job log."""
    return subprocess.run(
        command,
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )
