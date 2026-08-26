# Adapt from https://github.com/vllm-project/vllm/blob/main/vllm/v1/worker/gpu/model_runner.py
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
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
#
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import torch
from vllm.v1.worker.gpu.input_batch import InputBatch

import vllm_ascend.worker.v2.pcp_manager as pcp_manager_module
from vllm_ascend.worker.v2.input_batch import AscendInputBatch, AscendInputBuffers
from vllm_ascend.worker.v2.model_runner import NPUModelRunner
from vllm_ascend.worker.v2.pcp_manager import AscendPCPManager


def _mock_async_copy_to_cpu(value, out=None, device=None):
    """Copy PCP metadata without requiring device hooks in CPU-only UTs."""
    if isinstance(value, np.ndarray):
        value = torch.from_numpy(value)
    elif not isinstance(value, torch.Tensor):
        value = torch.as_tensor(value)

    if out is not None:
        out.copy_(value)
        return out

    return value.to(device="cpu")


def _make_local_pcp_batch() -> AscendInputBatch:
    """Build a local batch in the shape returned by the community PCP manager."""
    input_buffers = AscendInputBuffers(
        max_num_reqs=4,
        max_num_tokens=16,
        device=torch.device("cpu"),
    )
    base_batch = InputBatch.make_dummy(
        num_reqs=2,
        num_tokens=6,
        input_buffers=input_buffers,
    )

    # Local PCP rows: one starts at position 6 and contains two tokens; the
    # other starts at position 13 and contains four tokens.
    base_batch.req_ids = ["req-head", "req-tail"]
    base_batch.idx_mapping = torch.tensor([3, 7], dtype=torch.int32)
    base_batch.idx_mapping_np = np.array([3, 7], dtype=np.int32)
    base_batch.expanded_idx_mapping = base_batch.idx_mapping
    base_batch.num_scheduled_tokens = np.array([2, 4], dtype=np.int32)
    base_batch.query_start_loc_np = np.array([0, 2, 6], dtype=np.int32)
    base_batch.query_start_loc.copy_(torch.tensor([0, 2, 6], dtype=torch.int32))
    base_batch.num_computed_tokens_np = np.array([6, 13], dtype=np.int32)
    base_batch.prefill_len_np = np.array([32, 32], dtype=np.int32)
    base_batch.num_computed_prefill_tokens_np = np.array([6, 13], dtype=np.int32)
    base_batch.is_prefilling_np = np.array([True, True])
    base_batch.seq_lens.copy_(torch.tensor([8, 17], dtype=torch.int32))
    base_batch.seq_lens_cpu_upper_bound = torch.tensor([500, 600], dtype=torch.int32)
    base_batch.input_ids.copy_(torch.tensor([10, 11, 20, 21, 22, 23], dtype=torch.int32))
    base_batch.positions.copy_(torch.tensor([6, 7, 13, 14, 15, 16], dtype=torch.int64))
    base_batch.is_padding.fill_(False)

    return AscendInputBatch(
        **base_batch.__dict__,
        seq_lens_np=np.array([101, 102], dtype=np.int32),
        attn_state="global-attn-state",
    )


def _make_global_pcp_batch():
    """Build the global batch that is passed into PCPManager.partition_batch."""
    input_buffers = AscendInputBuffers(
        max_num_reqs=4,
        max_num_tokens=32,
        device=torch.device("cpu"),
    )
    base_batch = InputBatch.make_dummy(
        num_reqs=1,
        num_tokens=18,
        input_buffers=input_buffers,
    )
    base_batch.req_ids = ["global-req"]
    base_batch.idx_mapping = torch.tensor([3], dtype=torch.int32)
    base_batch.idx_mapping_np = np.array([3], dtype=np.int32)
    base_batch.expanded_idx_mapping = base_batch.idx_mapping
    base_batch.num_scheduled_tokens = np.array([18], dtype=np.int32)
    base_batch.query_start_loc_np = np.array([0, 18], dtype=np.int32)
    base_batch.query_start_loc.copy_(torch.tensor([0, 18], dtype=torch.int32))
    base_batch.num_computed_tokens_np = np.array([0], dtype=np.int32)
    base_batch.prefill_len_np = np.array([18], dtype=np.int32)
    base_batch.num_computed_prefill_tokens_np = np.array([0], dtype=np.int32)
    base_batch.is_prefilling_np = np.array([True])
    base_batch.seq_lens.copy_(torch.tensor([18], dtype=torch.int32))
    base_batch.seq_lens_cpu_upper_bound = torch.tensor([18], dtype=torch.int32)
    base_batch.input_ids.copy_(torch.arange(18, dtype=torch.int32))
    base_batch.positions.copy_(torch.arange(18, dtype=torch.int64))
    base_batch.is_padding.fill_(False)

    return AscendInputBatch(
        **base_batch.__dict__,
        seq_lens_np=np.array([18], dtype=np.int32),
        attn_state="global-attn-state",
    )


def test_partition_batch_refreshes_local_ascend_input_batch_metadata():
    """Refresh Ascend metadata after the real PCP local-batch rewrite."""
    vllm_config = object()
    global_batch = _make_global_pcp_batch()
    req_states = SimpleNamespace(
        last_sampled_tokens=torch.zeros(4, dtype=torch.int64),
        prefill_len=SimpleNamespace(gpu=torch.zeros(4, dtype=torch.int32)),
        draft_tokens=torch.empty((4, 0), dtype=torch.int64),
    )
    manager = AscendPCPManager(
        pcp_world_size=2,
        pcp_rank=0,
        device=torch.device("cpu"),
        vllm_config=vllm_config,
        req_states=req_states,
        max_num_reqs=1,
        max_num_tokens=18,
    )
    attn_state = MagicMock()

    with (
        # This Triton helper is unrelated to PCP partitioning and has no CPU
        # implementation. Stub only it; AscendPCPManager.partition_batch and
        # PCPManager.partition_batch both execute unmocked below.
        patch(
            "vllm.v1.worker.gpu.pcp_manager.prepare_pos_seq_lens",
            return_value=None,
        ),
        patch(
            "vllm.v1.worker.gpu.pcp_manager.combine_sampled_and_draft_tokens",
            return_value=torch.zeros(2, dtype=torch.int64),
        ),
        patch(
            "vllm.v1.worker.gpu.pcp_manager.async_copy_to_gpu",
            side_effect=_mock_async_copy_to_cpu,
        ),
        patch.object(pcp_manager_module, "build_attn_state", return_value=attn_state) as build_attn_state,
    ):
        result = manager.partition_batch(global_batch)

    assert isinstance(result, AscendInputBatch)
    assert result is not global_batch
    assert manager._global_batch is global_batch
    np.testing.assert_array_equal(global_batch.seq_lens_np, np.array([18], dtype=np.int32))
    assert global_batch.attn_state == "global-attn-state"

    # PCP=2 rank 0 owns the tail chunk then the head chunk; the real base
    # implementation produces this local row order and pads to rank 1's size.
    assert result.req_ids == ["global-req", "global-req"]
    np.testing.assert_array_equal(result.idx_mapping_np, np.array([3, 3], dtype=np.int32))
    np.testing.assert_array_equal(result.num_scheduled_tokens, np.array([3, 5], dtype=np.int32))
    np.testing.assert_array_equal(result.query_start_loc_np, np.array([0, 3, 8], dtype=np.int32))
    assert result.num_tokens == 8
    assert result.num_tokens_after_padding == 10
    assert torch.equal(result.input_ids[:8], torch.tensor([15, 16, 17, 0, 1, 2, 3, 4], dtype=torch.int32))

    # dataclasses.replace() retains the global Ascend-only fields by default;
    # the override must refresh them from real PCP-local CPU rows.
    expected_seq_lens = np.array([18, 5], dtype=np.int32)
    np.testing.assert_array_equal(result.seq_lens_np, expected_seq_lens)
    assert result.attn_state is attn_state

    args = build_attn_state.call_args.args
    assert args[0] is vllm_config
    np.testing.assert_array_equal(args[1], expected_seq_lens)
    assert args[2] == 2
    np.testing.assert_array_equal(args[3], np.array([3, 5], dtype=np.int32))
    np.testing.assert_array_equal(args[4], np.array([3, 5], dtype=np.int32))


def test_dummy_attention_context_uses_rank_local_identity_view():
    manager = AscendPCPManager.__new__(AscendPCPManager)
    manager.pcp_world_size = 2
    manager.pcp_rank = 1
    manager.device = torch.device("cpu")
    input_batch = _make_local_pcp_batch()
    input_batch.is_dummy = True
    block_tables = (
        torch.tensor([[1]], dtype=torch.int32),
        torch.tensor([[2]], dtype=torch.int32),
    )
    slot_mappings = torch.arange(
        len(block_tables) * manager.pcp_world_size * input_batch.num_tokens,
        dtype=torch.int64,
    ).view(len(block_tables), -1)

    actual = manager.build_attention_context(
        input_batch,
        block_tables,
        slot_mappings,
    )

    expected_slot_mappings = slot_mappings.view(
        len(block_tables),
        manager.pcp_world_size,
        input_batch.num_tokens,
    )[:, manager.pcp_rank]
    restore_start = manager.pcp_rank * input_batch.num_tokens
    assert actual.global_batch is input_batch
    assert actual.global_block_tables is block_tables
    assert torch.equal(actual.global_slot_mappings, expected_slot_mappings)
    assert torch.equal(
        actual.hidden_restore_idx,
        torch.arange(
            restore_start,
            restore_start + input_batch.num_tokens,
        ),
    )
    assert actual.local_num_tokens_after_padding == input_batch.num_tokens


def test_npu_model_runner_uses_ascend_pcp_manager() -> None:
    runner = NPUModelRunner.__new__(NPUModelRunner)
    assert runner.pcp_manager_cls is AscendPCPManager


def test_initialize_kv_cache_skips_pcp_binding_when_disabled() -> None:
    runner = NPUModelRunner.__new__(NPUModelRunner)
    runner.pcp_manager = None
    runner.model_config = MagicMock(enable_return_routed_experts=False)
    kv_cache_config = MagicMock()

    with (
        patch("vllm_ascend.worker.v2.model_runner.graph_manager_wrapper"),
        patch("vllm.v1.worker.gpu.model_runner.GPUModelRunner.initialize_kv_cache") as initialize_kv_cache,
    ):
        runner.initialize_kv_cache(kv_cache_config)

    initialize_kv_cache.assert_called_once_with(kv_cache_config)
