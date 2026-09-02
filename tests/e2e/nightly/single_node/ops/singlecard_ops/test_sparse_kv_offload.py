#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
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

import pytest
import torch

from vllm_ascend.utils import bootstrap_custom_op_env

# Import the packaged extension so the CPU-dispatched Sparse KV ops are registered.
bootstrap_custom_op_env(include_vendor_lib=True)
import vllm_ascend.vllm_ascend_C  # type: ignore[import-untyped] # noqa: E402,F401


def _make_lru_state(
    num_reqs: int,
    topk: int,
    capacity: int,
    max_token: int,
    workspace_threads: int,
) -> dict[str, torch.Tensor]:
    return {
        "req_ids": torch.empty(num_reqs, dtype=torch.int64),
        "last_req_ids": torch.full((num_reqs,), -1, dtype=torch.int64),
        "topk_indices": torch.empty((num_reqs, topk), dtype=torch.int32),
        "stable_prefix_lens": torch.full((num_reqs,), max_token, dtype=torch.int32),
        "slot_to_token": torch.full((num_reqs, capacity), -1, dtype=torch.int32),
        "lru_slots": torch.arange(capacity, dtype=torch.int32).repeat(num_reqs, 1),
        "current_slots": torch.empty((num_reqs, topk), dtype=torch.int32),
        "miss_count": torch.empty(num_reqs, dtype=torch.int32),
        "miss_tokens": torch.empty((num_reqs, topk), dtype=torch.int32),
        "miss_slots": torch.empty((num_reqs, topk), dtype=torch.int32),
        "token_mark": torch.zeros((workspace_threads, max_token), dtype=torch.int32),
        "token_pos": torch.full((workspace_threads, max_token), -1, dtype=torch.int32),
        "slot_workspace": torch.empty((workspace_threads, capacity * 3), dtype=torch.int32),
        "miss_position_workspace": torch.empty((workspace_threads, topk), dtype=torch.int32),
        "epochs": torch.zeros(workspace_threads, dtype=torch.int32),
    }


def _run_lru_resident_compact(
    state: dict[str, torch.Tensor],
    num_reqs: int,
    topk: int,
    capacity: int,
    max_token: int,
    workspace_threads: int,
) -> None:
    torch.ops._C_ascend.sparse_kv_lru_resident_compact(
        state["req_ids"].data_ptr(),
        state["last_req_ids"].data_ptr(),
        state["topk_indices"].data_ptr(),
        state["stable_prefix_lens"].data_ptr(),
        state["slot_to_token"].data_ptr(),
        state["lru_slots"].data_ptr(),
        state["current_slots"].data_ptr(),
        state["miss_count"].data_ptr(),
        state["miss_tokens"].data_ptr(),
        state["miss_slots"].data_ptr(),
        state["token_mark"].data_ptr(),
        state["token_pos"].data_ptr(),
        state["slot_workspace"].data_ptr(),
        state["miss_position_workspace"].data_ptr(),
        state["epochs"].data_ptr(),
        num_reqs,
        topk,
        capacity,
        max_token,
        workspace_threads,
        workspace_threads,
    )


def test_lru_resident_compact_assigns_initial_misses_for_multiple_requests():
    num_reqs, topk, capacity, max_token, workspace_threads = 2, 3, 4, 16, 2
    state = _make_lru_state(num_reqs, topk, capacity, max_token, workspace_threads)
    state["req_ids"].copy_(torch.tensor([101, 202], dtype=torch.int64))
    state["topk_indices"].copy_(torch.tensor([[2, 5, 7], [1, 4, 6]], dtype=torch.int32))

    _run_lru_resident_compact(state, num_reqs, topk, capacity, max_token, workspace_threads)

    torch.testing.assert_close(state["last_req_ids"], torch.tensor([101, 202], dtype=torch.int64))
    torch.testing.assert_close(
        state["slot_to_token"],
        torch.tensor([[2, 5, 7, -1], [1, 4, 6, -1]], dtype=torch.int32),
    )
    torch.testing.assert_close(
        state["lru_slots"],
        torch.tensor([[3, 0, 1, 2], [3, 0, 1, 2]], dtype=torch.int32),
    )
    torch.testing.assert_close(
        state["current_slots"],
        torch.tensor([[0, 1, 2], [0, 1, 2]], dtype=torch.int32),
    )
    torch.testing.assert_close(state["miss_count"], torch.tensor([3, 3], dtype=torch.int32))
    torch.testing.assert_close(
        state["miss_tokens"],
        torch.tensor([[2, 5, 7], [1, 4, 6]], dtype=torch.int32),
    )
    torch.testing.assert_close(
        state["miss_slots"],
        torch.tensor([[0, 1, 2], [0, 1, 2]], dtype=torch.int32),
    )


def test_lru_resident_compact_reuses_hits_and_evicts_lru_slot():
    num_reqs, topk, capacity, max_token, workspace_threads = 1, 3, 4, 16, 1
    state = _make_lru_state(num_reqs, topk, capacity, max_token, workspace_threads)
    state["req_ids"].fill_(7)
    state["topk_indices"].copy_(torch.tensor([[2, 5, 7]], dtype=torch.int32))
    _run_lru_resident_compact(state, num_reqs, topk, capacity, max_token, workspace_threads)

    state["topk_indices"].copy_(torch.tensor([[7, 2, 9]], dtype=torch.int32))
    _run_lru_resident_compact(state, num_reqs, topk, capacity, max_token, workspace_threads)

    torch.testing.assert_close(state["current_slots"], torch.tensor([[2, 0, 3]], dtype=torch.int32))
    torch.testing.assert_close(state["miss_count"], torch.tensor([1], dtype=torch.int32))
    torch.testing.assert_close(state["miss_tokens"], torch.tensor([[9, -1, -1]], dtype=torch.int32))
    torch.testing.assert_close(state["miss_slots"], torch.tensor([[3, -1, -1]], dtype=torch.int32))
    torch.testing.assert_close(state["slot_to_token"], torch.tensor([[2, 5, 7, 9]], dtype=torch.int32))
    torch.testing.assert_close(state["lru_slots"], torch.tensor([[1, 3, 0, 2]], dtype=torch.int32))


def test_compute_lru_resident_addrs_builds_kv_copy_descriptors():
    miss_count = torch.tensor([2, 1], dtype=torch.int32)
    miss_tokens = torch.tensor([[1, 6, -1], [4, -1, -1]], dtype=torch.int32)
    miss_slots = torch.tensor([[0, 2, -1], [1, -1, -1]], dtype=torch.int32)
    block_table = torch.tensor([[10, 11, 12], [20, 21, 22]], dtype=torch.int32)
    gvas_buffer = torch.full((12,), -1, dtype=torch.int64)
    addr_buffer = torch.full((12,), -1, dtype=torch.int64)
    size_buffer = torch.full((12,), -1, dtype=torch.int32)
    num_tokens_buffer = torch.zeros(1, dtype=torch.int32)

    num_tokens_to_load = torch.ops._C_ascend.sparse_kv_compute_lru_resident_addrs(
        miss_count,
        miss_tokens,
        miss_slots,
        block_table,
        4,
        8,
        12,
        1000,
        2000,
        3000,
        4000,
        4,
        2,
        gvas_buffer,
        addr_buffer,
        size_buffer,
        num_tokens_buffer,
    )

    assert num_tokens_to_load == 3
    torch.testing.assert_close(
        gvas_buffer,
        torch.tensor([1328, 1368, 1672, 2492, 2552, 3008, -1, -1, -1, -1, -1, -1], dtype=torch.int64),
    )
    torch.testing.assert_close(
        addr_buffer,
        torch.tensor([3000, 3016, 3040, 4000, 4024, 4060, -1, -1, -1, -1, -1, -1], dtype=torch.int64),
    )
    torch.testing.assert_close(
        size_buffer,
        torch.tensor([8, 8, 8, 12, 12, 12, -1, -1, -1, -1, -1, -1], dtype=torch.int32),
    )
    torch.testing.assert_close(num_tokens_buffer, torch.tensor([6], dtype=torch.int32))

    with pytest.raises(RuntimeError, match="miss_count wrong dtype, should be int32"):
        torch.ops._C_ascend.sparse_kv_compute_lru_resident_addrs(
            miss_count.to(torch.int64),
            miss_tokens,
            miss_slots,
            block_table,
            4,
            8,
            12,
            1000,
            2000,
            3000,
            4000,
            4,
            2,
            gvas_buffer,
            addr_buffer,
            size_buffer,
            num_tokens_buffer,
        )
