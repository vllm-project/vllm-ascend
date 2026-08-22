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
"""Expose vllm-ascend interface analysis through vLLM's existing NPU job."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.e2e.vllm_interface.vllm_interface_contracts.range_analysis import git_head
from tests.e2e.vllm_interface.vllm_interface_contracts.upstream_ci import (
    build_analysis_command,
    resolve_vllm_range,
    run_analysis,
)

VLLM_UPSTREAM_CHECKOUT = Path("/workspace/vllm")
DOWNSTREAM_INDEX_CACHE_DIR = Path.home() / ".cache" / "vllm-interface" / "repository-index"
UPSTREAM_FILE_INDEX_CACHE_DIR = Path.home() / ".cache" / "vllm-interface" / "file-fragments"


def test_upstream_interface_compatibility() -> None:
    """Report interface breaks introduced by the checked-out upstream PR."""
    if not VLLM_UPSTREAM_CHECKOUT.exists():
        pytest.skip("the upstream vLLM checkout is available only in the vLLM NPU job")
    if not (VLLM_UPSTREAM_CHECKOUT / ".git").exists():
        pytest.fail("the upstream vLLM checkout must retain Git metadata")

    ascend_root = Path(__file__).resolve().parents[3]
    old_sha, new_sha = resolve_vllm_range(VLLM_UPSTREAM_CHECKOUT)
    ascend_sha = git_head(ascend_root)
    command = build_analysis_command(
        vllm_root=VLLM_UPSTREAM_CHECKOUT,
        ascend_root=ascend_root,
        old_sha=old_sha,
        new_sha=new_sha,
        ascend_sha=ascend_sha,
        analysis_workers=3,
        downstream_index_cache_dir=DOWNSTREAM_INDEX_CACHE_DIR,
        upstream_file_index_cache_dir=UPSTREAM_FILE_INDEX_CACHE_DIR,
        index_workers=4,
    )

    print("\n+++ vLLM interface compatibility inputs")
    print(f"vllm_old_sha={old_sha}")
    print(f"vllm_new_sha={new_sha}")
    print(f"vllm_ascend_sha={ascend_sha}")
    result = run_analysis(command, cwd=ascend_root)
    if result.stderr:
        print("\n+++ vLLM interface compatibility timings")
        print(result.stderr.rstrip())

    print("\n+++ vLLM interface compatibility result")
    if result.stdout:
        print(result.stdout.rstrip())

    if result.returncode == 1:
        pytest.fail("the upstream PR introduces a vllm-ascend interface break")
    if result.returncode != 0:
        pytest.fail(f"interface analysis failed with exit code {result.returncode}")
