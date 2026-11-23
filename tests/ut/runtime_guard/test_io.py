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

# mypy: ignore-errors
"""UT: runtime_guard.io (token helpers + I/O snapshot cache)."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from vllm_ascend.observability.runtime_guard.io import (
    RequestIoSnapshotManager,
    normalize_token_ids,
    prompt_token_count_for_request,
)
from vllm_ascend.observability.runtime_guard.state import RequestGuardStore


def test_normalize_token_ids_tensor_and_list():
    t = torch.tensor([3, 4, 5], dtype=torch.int64)
    assert normalize_token_ids(t) == [3, 4, 5]
    assert normalize_token_ids([1, torch.tensor(7)]) == [1, 7]
    assert normalize_token_ids(None) == []


def test_prompt_token_count_reads_store_without_runner():
    RequestGuardStore.reset_for_tests()
    try:
        store = RequestGuardStore.get()
        store.set_prompt_token_ids("r-store", [10, 11, 12])
        assert prompt_token_count_for_request(None, "r-store", None) == 3
    finally:
        RequestGuardStore.reset_for_tests()


def test_include_token_ids_cache_invalidated_after_append_output():
    RequestIoSnapshotManager.reset_for_tests()
    try:
        mgr = RequestIoSnapshotManager.get()
        store = RequestGuardStore.get()
        store.get_or_create("r1")
        runner = SimpleNamespace(input_batch=None, requests={}, req_states=None)
        mgr.append_output("r1", [1])
        snap1 = mgr.snapshot(runner, "r1", 0, include_token_ids=True, use_cache=True)
        assert snap1.output_token_ids == [1]
        mgr.append_output("r1", [2])
        snap2 = mgr.snapshot(runner, "r1", 0, include_token_ids=True, use_cache=True)
        assert snap2.output_token_ids == [1, 2]
        assert snap2 is not snap1
    finally:
        RequestIoSnapshotManager.reset_for_tests()
