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
"""Helpers for running the interface analyzer from the upstream vLLM CI."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

VLLM_UPSTREAM_URL = "https://github.com/vllm-project/vllm.git"
VLLM_UPSTREAM_BRANCH = "main"


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def resolve_vllm_range(
    vllm_root: Path,
    *,
    upstream_url: str = VLLM_UPSTREAM_URL,
    upstream_branch: str = VLLM_UPSTREAM_BRANCH,
) -> tuple[str, str]:
    """Resolve the exact merge-base-to-HEAD range for an upstream PR checkout."""
    if not vllm_root.is_dir():
        raise ValueError(f"vLLM checkout does not exist: {vllm_root}")
    if not (vllm_root / ".git").exists():
        raise ValueError(f"vLLM checkout has no Git metadata: {vllm_root}")

    new_sha = _git(vllm_root, "rev-parse", "HEAD")
    _git(vllm_root, "fetch", "--no-tags", upstream_url, upstream_branch)
    try:
        old_sha = _git(vllm_root, "merge-base", new_sha, "FETCH_HEAD")
    except subprocess.CalledProcessError:
        if _git(vllm_root, "rev-parse", "--is-shallow-repository") != "true":
            raise
        _git(vllm_root, "fetch", "--no-tags", "--unshallow", upstream_url, upstream_branch)
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
    downstream_index_cache_dir: Path | None = None,
    upstream_file_index_cache_dir: Path | None = None,
    index_workers: int = 1,
) -> list[str]:
    """Build the repository CLI command used by the upstream pytest entry."""
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
        "--stdout-summary",
        "--analysis-workers",
        str(analysis_workers),
        "--index-workers",
        str(index_workers),
    ]
    if downstream_index_cache_dir is not None:
        command.extend(
            [
                "--downstream-index-cache-dir",
                str(downstream_index_cache_dir),
            ]
        )
    if upstream_file_index_cache_dir is not None:
        command.extend(
            [
                "--upstream-file-index-cache-dir",
                str(upstream_file_index_cache_dir),
            ]
        )
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
