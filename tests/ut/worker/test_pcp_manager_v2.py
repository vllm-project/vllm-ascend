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
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from vllm.config import CUDAGraphMode
from vllm.v1.attention.backends.utils import PAD_SLOT_ID
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.pcp_manager import PCPManager

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
        # implementation. Stub only it; the Ascend partition override
        # executes unmocked below.
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
        patch.object(
            pcp_manager_module,
            "build_attn_state",
            return_value=attn_state,
        ) as build_attn_state,
    ):
        result = manager.partition_batch(global_batch)

    assert isinstance(result, AscendInputBatch)
    assert result is not global_batch
    assert manager._global_batch is global_batch
    np.testing.assert_array_equal(
        global_batch.seq_lens_np,
        np.array([18], dtype=np.int32),
    )
    assert global_batch.attn_state == "global-attn-state"

    # PCP=2 rank 0 owns the tail chunk then the head chunk; the real base
    # implementation produces this local row order and pads to rank 1's size.
    assert result.req_ids == ["global-req", "global-req"]
    np.testing.assert_array_equal(
        result.idx_mapping_np,
        np.array([3, 3], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        result.num_scheduled_tokens,
        np.array([3, 5], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        result.query_start_loc_np,
        np.array([0, 3, 8], dtype=np.int32),
    )
    assert result.num_tokens == 8
    assert result.num_tokens_after_padding == 10
    assert torch.equal(
        result.input_ids[:8],
        torch.tensor(
            [15, 16, 17, 0, 1, 2, 3, 4],
            dtype=torch.int32,
        ),
    )

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


@pytest.mark.parametrize("method", ["mtp", "eagle3", "ngram"])
def test_partition_batch_preserves_global_speculative_batch(
    method: str,
) -> None:
    global_batch = _make_global_pcp_batch()
    global_batch.num_draft_tokens = 3
    global_batch.num_draft_tokens_per_req = np.array([3], dtype=np.int32)
    req_states = SimpleNamespace(
        last_sampled_tokens=torch.zeros(4, dtype=torch.int64),
        prefill_len=SimpleNamespace(gpu=torch.zeros(4, dtype=torch.int32)),
        draft_tokens=torch.empty((4, 3), dtype=torch.int64),
    )
    manager = AscendPCPManager(
        pcp_world_size=2,
        pcp_rank=0,
        device=torch.device("cpu"),
        req_states=req_states,
        max_num_reqs=1,
        max_num_tokens=18,
        vllm_config=SimpleNamespace(
            speculative_config=SimpleNamespace(method=method),
            num_speculative_tokens=3,
        ),
    )
    upstream_partition_batch = PCPManager.partition_batch
    upstream_batches = []

    def call_upstream(manager, batch):
        upstream_batches.append(batch)
        return upstream_partition_batch(manager, batch)

    with (
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
        patch.object(
            pcp_manager_module,
            "build_attn_state",
            return_value=MagicMock(),
        ),
        patch.object(
            PCPManager,
            "partition_batch",
            new=call_upstream,
        ),
    ):
        local_batch = manager.partition_batch(global_batch)

    assert len(upstream_batches) == 1
    assert upstream_batches[0] is not global_batch
    assert upstream_batches[0].num_draft_tokens == 0
    assert upstream_batches[0].num_draft_tokens_per_req is None
    assert manager._global_batch is global_batch
    assert global_batch.num_draft_tokens == 3
    np.testing.assert_array_equal(
        global_batch.num_draft_tokens_per_req,
        np.array([3], dtype=np.int32),
    )
    assert local_batch.num_draft_tokens == 0
    assert local_batch.num_draft_tokens_per_req is None


def test_partition_batch_restores_global_batch_when_upstream_fails() -> None:
    global_batch = _make_global_pcp_batch()
    global_batch.num_draft_tokens = 3
    global_batch.num_draft_tokens_per_req = np.array([3], dtype=np.int32)
    manager = AscendPCPManager(
        pcp_world_size=2,
        pcp_rank=0,
        device=torch.device("cpu"),
        req_states=MagicMock(),
        max_num_reqs=1,
        max_num_tokens=18,
        vllm_config=object(),
    )

    with (
        patch.object(
            PCPManager,
            "partition_batch",
            side_effect=RuntimeError("upstream failure"),
        ),
        pytest.raises(RuntimeError, match="upstream failure"),
    ):
        manager.partition_batch(global_batch)

    assert manager._global_batch is global_batch
    assert global_batch.num_draft_tokens == 3
    np.testing.assert_array_equal(
        global_batch.num_draft_tokens_per_req,
        np.array([3], dtype=np.int32),
    )


def test_prepare_speculator_attn_rebuilds_global_kv_layout() -> None:
    global_batch = _make_global_pcp_batch()
    manager = AscendPCPManager(
        pcp_world_size=2,
        pcp_rank=0,
        device=torch.device("cpu"),
        req_states=MagicMock(),
        max_num_reqs=1,
        max_num_tokens=20,
        vllm_config=object(),
    )
    block_tables = MagicMock()
    gathered_block_tables = (torch.ones(1, 1),)
    global_batch.num_tokens_after_padding = 20
    slot_mappings = torch.arange(20).unsqueeze(0)
    block_tables.gather_block_tables.return_value = gathered_block_tables
    manager._block_tables = block_tables
    manager._global_batch_slot_mappings = slot_mappings
    manager._gathered_kv_slot_mappings = torch.empty((1, 40), dtype=slot_mappings.dtype)
    manager._global_batch = global_batch

    result_block_tables, result_slot_mappings = manager.prepare_speculator_attn(global_batch)

    assert result_block_tables is gathered_block_tables
    expected_rank_slots = torch.cat(
        (
            torch.arange(18),
            torch.full((2,), PAD_SLOT_ID),
        )
    ).unsqueeze(0)
    torch.testing.assert_close(
        result_slot_mappings,
        expected_rank_slots.repeat(1, 2),
    )
    assert manager._gathered_kv_slot_mappings is not None
    assert result_slot_mappings.data_ptr() == manager._gathered_kv_slot_mappings.data_ptr()
    block_tables.gather_block_tables.assert_called_once_with(
        global_batch.idx_mapping,
        num_reqs_padded=global_batch.num_reqs_after_padding,
    )


def test_cached_prefill_partitions_only_the_scheduled_suffix() -> None:
    manager = AscendPCPManager(
        pcp_world_size=2,
        pcp_rank=0,
        device=torch.device("cpu"),
        req_states=MagicMock(),
        max_num_reqs=4,
        max_num_tokens=8,
        vllm_config=object(),
    )

    def local_starts(num_computed_tokens: int) -> list[int]:
        segments = manager._get_rank_segments(
            rank=0,
            num_scheduled_tokens=np.array([8], dtype=np.int32),
            num_computed_tokens=np.array([num_computed_tokens], dtype=np.int32),
            is_prefilling=np.array([True]),
            query_start_loc_np=np.array([0, 8], dtype=np.int32),
        )
        return [num_computed_tokens + segment.global_batch_slice.start for segment in segments]

    # Two scheduler iterations of one longer suffix advance from the cached
    # prefix without repartitioning or recomputing that prefix.
    assert local_starts(128) == [128, 134]
    assert local_starts(136) == [136, 142]


def test_pcp_layout_orders_cache_hit_miss_and_decode_rows() -> None:
    manager = AscendPCPManager(
        pcp_world_size=2,
        pcp_rank=0,
        device=torch.device("cpu"),
        req_states=MagicMock(),
        max_num_reqs=8,
        max_num_tokens=8,
        vllm_config=object(),
    )
    num_scheduled_tokens = np.array([1, 4, 1], dtype=np.int32)
    num_computed_tokens = np.array([128, 0, 256], dtype=np.int32)
    is_prefilling = np.array([True, True, False])
    query_start_loc = np.array([0, 1, 5, 6], dtype=np.int32)

    rank_zero = manager._get_rank_segments(
        0,
        num_scheduled_tokens,
        num_computed_tokens,
        is_prefilling,
        query_start_loc,
    )
    rank_one = manager._get_rank_segments(
        1,
        num_scheduled_tokens,
        num_computed_tokens,
        is_prefilling,
        query_start_loc,
    )

    def layout(segments):
        return [
            (
                segment.global_batch_req_idx,
                segment.global_batch_slice.start,
                segment.global_batch_slice.stop,
            )
            for segment in segments
        ]

    # Continued prefills and replicated decodes stay before fresh prefills.
    # The one-token cache-hit suffix is owned by rank 0; no cached prefix token
    # appears in either rank's scheduled slices.
    assert layout(rank_zero) == [
        (0, 0, 1),
        (2, 5, 6),
        (1, 1, 2),
        (1, 4, 5),
    ]
    assert layout(rank_one) == [
        (2, 5, 6),
        (1, 2, 3),
        (1, 3, 4),
    ]


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


@pytest.mark.parametrize("method", ["mtp", "eagle3"])
def test_validate_config_accepts_supported_speculative_methods(
    method: str,
) -> None:
    vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            prefill_context_parallel_size=2,
            pipeline_parallel_size=1,
            decode_context_parallel_size=1,
        ),
        model_config=SimpleNamespace(
            use_mla=True,
            is_encoder_decoder=False,
            hf_text_config=SimpleNamespace(),
        ),
        speculative_config=SimpleNamespace(method=method),
        lora_config=None,
        compilation_config=SimpleNamespace(
            cudagraph_mode=CUDAGraphMode.FULL,
        ),
    )

    AscendPCPManager.validate_config(
        vllm_config,
        supports_mm_inputs=False,
    )


def test_validate_config_allows_unadapted_speculative_method() -> None:
    vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            prefill_context_parallel_size=2,
            decode_context_parallel_size=1,
            pipeline_parallel_size=1,
        ),
        model_config=SimpleNamespace(
            use_mla=True,
            is_encoder_decoder=False,
            hf_text_config=SimpleNamespace(),
        ),
        speculative_config=SimpleNamespace(method="ngram"),
        lora_config=None,
        compilation_config=SimpleNamespace(
            cudagraph_mode=CUDAGraphMode.FULL,
        ),
    )

    AscendPCPManager.validate_config(
        vllm_config,
        supports_mm_inputs=False,
    )


def _make_decode_local_batch(
    manager: AscendPCPManager,
    num_reqs: int,
) -> AscendInputBatch:
    input_buffers = manager._input_buffers
    assert input_buffers is not None
    base_batch = InputBatch.make_dummy(
        num_reqs=num_reqs,
        num_tokens=num_reqs,
        input_buffers=input_buffers,
    )
    base_batch.num_computed_tokens_np = np.arange(
        10,
        10 + num_reqs,
        dtype=np.int32,
    )
    base_batch.num_scheduled_tokens = np.ones(num_reqs, dtype=np.int32)
    seq_lens = base_batch.num_computed_tokens_np + base_batch.num_scheduled_tokens
    base_batch.seq_lens.copy_(torch.from_numpy(seq_lens))
    base_batch.seq_lens_cpu_upper_bound = torch.from_numpy(seq_lens)
    base_batch.input_ids.copy_(torch.arange(100, 100 + num_reqs, dtype=torch.int32))
    base_batch.positions.copy_(torch.arange(10, 10 + num_reqs, dtype=torch.int64))
    base_batch.is_padding.fill_(False)
    return AscendInputBatch(
        **base_batch.__dict__,
        seq_lens_np=seq_lens,
        attn_state="decode-attn-state",
    )


def test_pad_decode_batch_supports_single_fia_dummy_request():
    """Allow one FIA dummy request to consume multiple padding tokens."""
    num_reqs = 2
    graph_num_reqs = 3
    graph_num_tokens = 4
    manager = AscendPCPManager(
        pcp_world_size=2,
        pcp_rank=0,
        device=torch.device("cpu"),
        req_states=MagicMock(),
        max_num_reqs=graph_num_reqs,
        max_num_tokens=graph_num_tokens,
        vllm_config=object(),
    )
    local_batch = _make_decode_local_batch(manager, num_reqs=num_reqs)
    input_buffers = manager._input_buffers
    assert input_buffers is not None
    input_buffers.input_ids[num_reqs:graph_num_tokens].fill_(999)
    input_buffers.positions[num_reqs:graph_num_tokens].fill_(999)
    input_buffers.seq_lens[num_reqs:graph_num_reqs].fill_(999)
    input_buffers.is_padding[num_reqs:graph_num_tokens].fill_(False)

    graph_input_ids = torch.full(
        (graph_num_tokens,),
        999,
        dtype=torch.int32,
    )
    graph_input_ids[:num_reqs].copy_(local_batch.input_ids)
    graph_positions = torch.full(
        (graph_num_tokens,),
        999,
        dtype=torch.int64,
    )
    graph_positions[:num_reqs].copy_(local_batch.positions)
    manager._global_batch = replace(
        local_batch,
        num_reqs_after_padding=graph_num_reqs,
        num_tokens_after_padding=graph_num_tokens,
        input_ids=graph_input_ids,
        positions=graph_positions,
        is_padding=torch.ones(graph_num_tokens, dtype=torch.bool),
    )

    result = manager._pad_decode_batch_for_full_graph(
        local_batch,
        graph_num_reqs,
        graph_num_tokens,
        local_batch.seq_lens_np,
    )

    expected_query_start_loc = np.array([0, 1, 2, 4], dtype=np.int32)
    np.testing.assert_array_equal(
        result.query_start_loc_np,
        expected_query_start_loc,
    )
    torch.testing.assert_close(
        result.query_start_loc,
        torch.from_numpy(expected_query_start_loc),
    )
    assert result.input_ids.tolist() == [100, 101, 0, 0]
    assert result.positions.tolist() == [10, 11, 0, 0]
    assert result.seq_lens.tolist() == [11, 12, 0]
    assert result.is_padding.tolist() == [False, False, True, True]


def test_pad_decode_batch_uses_k_plus_one_tokens_per_dummy_request() -> None:
    num_reqs = 2
    query_len = 4
    graph_num_reqs = 3
    graph_num_tokens = graph_num_reqs * query_len
    manager = AscendPCPManager(
        pcp_world_size=2,
        pcp_rank=0,
        device=torch.device("cpu"),
        req_states=MagicMock(),
        max_num_reqs=graph_num_reqs,
        max_num_tokens=graph_num_tokens,
        vllm_config=SimpleNamespace(num_speculative_tokens=3),
    )
    local_batch = _make_decode_local_batch(manager, num_reqs=num_reqs)
    input_buffers = manager._input_buffers
    assert input_buffers is not None

    num_tokens = num_reqs * query_len
    query_start_loc_np = np.array([0, 4, 8], dtype=np.int32)
    input_buffers.query_start_loc[: num_reqs + 1].copy_(torch.from_numpy(query_start_loc_np))
    input_buffers.input_ids[:num_tokens].copy_(torch.arange(100, 100 + num_tokens, dtype=torch.int32))
    input_buffers.positions[:num_tokens].copy_(torch.arange(10, 10 + num_tokens, dtype=torch.int64))
    input_buffers.is_padding[:num_tokens].fill_(False)
    local_seq_lens_np = np.array([14, 15], dtype=np.int32)
    input_buffers.seq_lens[:num_reqs].copy_(torch.from_numpy(local_seq_lens_np))
    local_batch = replace(
        local_batch,
        num_scheduled_tokens=np.full(num_reqs, query_len, dtype=np.int32),
        num_tokens=num_tokens,
        num_tokens_after_padding=num_tokens,
        query_start_loc=input_buffers.query_start_loc[: num_reqs + 1],
        query_start_loc_np=query_start_loc_np,
        seq_lens=input_buffers.seq_lens[:num_reqs],
        seq_lens_cpu_upper_bound=torch.from_numpy(local_seq_lens_np.copy()),
        input_ids=input_buffers.input_ids[:num_tokens],
        positions=input_buffers.positions[:num_tokens],
        is_padding=input_buffers.is_padding[:num_tokens],
        seq_lens_np=local_seq_lens_np,
    )
    manager._global_batch = replace(
        local_batch,
        num_reqs_after_padding=graph_num_reqs,
        num_tokens_after_padding=graph_num_tokens,
        input_ids=torch.full((graph_num_tokens,), 999, dtype=torch.int32),
        positions=torch.full((graph_num_tokens,), 999, dtype=torch.int64),
        is_padding=torch.ones(graph_num_tokens, dtype=torch.bool),
    )

    result = manager._pad_decode_batch_for_full_graph(
        local_batch,
        graph_num_reqs,
        graph_num_tokens,
        local_seq_lens_np,
    )

    expected_query_start_loc = np.array([0, 4, 8, 12], dtype=np.int32)
    np.testing.assert_array_equal(
        result.query_start_loc_np,
        expected_query_start_loc,
    )
    torch.testing.assert_close(
        result.query_start_loc,
        torch.from_numpy(expected_query_start_loc),
    )
    assert result.seq_lens.tolist() == [14, 15, 0]
    assert result.is_padding.tolist() == [False] * 8 + [True] * 4
