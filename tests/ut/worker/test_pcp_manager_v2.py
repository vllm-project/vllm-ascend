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
import pytest
import torch
from vllm.config import CUDAGraphMode
from vllm.v1.worker.gpu.input_batch import InputBatch

import vllm_ascend.worker.v2.pcp_manager as pcp_manager_module
from vllm_ascend.worker.v2.input_batch import AscendInputBatch, AscendInputBuffers
from vllm_ascend.worker.v2.pcp_manager import (
    AscendPCPManager,
    maybe_build_ascend_pcp_manager,
)


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


def _mock_expand_idx_mapping(idx_mapping, total_num_logits, cu_num_logits, **_):
    counts = torch.diff(cu_num_logits).to(torch.int64)
    expanded_idx_mapping = torch.repeat_interleave(idx_mapping, counts)
    expanded_local_pos = torch.cat(
        [torch.arange(int(count), dtype=torch.int32) for count in counts]
    )
    assert expanded_idx_mapping.numel() == total_num_logits
    return expanded_idx_mapping, expanded_local_pos


def _make_local_pcp_batch():
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


def _make_global_pcp_batch(num_computed_tokens: int = 0):
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
    prefill_len = num_computed_tokens + 18
    base_batch.num_computed_tokens_np = np.array(
        [num_computed_tokens], dtype=np.int32
    )
    base_batch.prefill_len_np = np.array([prefill_len], dtype=np.int32)
    base_batch.num_computed_prefill_tokens_np = np.array(
        [num_computed_tokens], dtype=np.int32
    )
    base_batch.is_prefilling_np = np.array([True])
    base_batch.seq_lens.copy_(torch.tensor([prefill_len], dtype=torch.int32))
    base_batch.seq_lens_cpu_upper_bound = torch.tensor(
        [prefill_len], dtype=torch.int32
    )
    base_batch.input_ids.copy_(torch.arange(18, dtype=torch.int32))
    base_batch.positions.copy_(torch.arange(18, dtype=torch.int64))
    base_batch.is_padding.fill_(False)

    return AscendInputBatch(
        **base_batch.__dict__,
        seq_lens_np=np.array([prefill_len], dtype=np.int32),
        attn_state="global-attn-state",
    )


def _make_global_mtp_batch(*, mixed_batch: bool):
    """Build either an MTP decode batch or a prefill+MTP decode batch."""
    num_tokens = 12 if mixed_batch else 4
    input_buffers = AscendInputBuffers(
        max_num_reqs=4,
        max_num_tokens=32,
        device=torch.device("cpu"),
    )
    base_batch = InputBatch.make_dummy(
        num_reqs=2,
        num_tokens=num_tokens,
        input_buffers=input_buffers,
    )

    if mixed_batch:
        req_ids = ["prefill", "decode"]
        scheduled = np.array([10, 2], dtype=np.int32)
        computed = np.array([0, 20], dtype=np.int32)
        prefill_lens = np.array([10, 8], dtype=np.int32)
        computed_prefill = np.array([0, 8], dtype=np.int32)
        is_prefilling = np.array([True, False])
        seq_lens = np.array([10, 22], dtype=np.int32)
        draft_counts = np.array([0, 1], dtype=np.int32)
    else:
        req_ids = ["decode-a", "decode-b"]
        scheduled = np.array([2, 2], dtype=np.int32)
        computed = np.array([20, 30], dtype=np.int32)
        prefill_lens = np.array([8, 8], dtype=np.int32)
        computed_prefill = np.array([8, 8], dtype=np.int32)
        is_prefilling = np.array([False, False])
        seq_lens = np.array([22, 32], dtype=np.int32)
        draft_counts = np.array([1, 1], dtype=np.int32)

    query_start_loc_np = np.empty(3, dtype=np.int32)
    query_start_loc_np[0] = 0
    np.cumsum(scheduled, out=query_start_loc_np[1:])
    cu_num_logits_np = np.empty(3, dtype=np.int32)
    cu_num_logits_np[0] = 0
    np.cumsum(draft_counts + 1, out=cu_num_logits_np[1:])

    base_batch.req_ids = req_ids
    base_batch.idx_mapping = torch.tensor([1, 3], dtype=torch.int32)
    base_batch.idx_mapping_np = np.array([1, 3], dtype=np.int32)
    base_batch.num_scheduled_tokens = scheduled
    base_batch.query_start_loc_np = query_start_loc_np
    base_batch.query_start_loc.copy_(torch.from_numpy(query_start_loc_np))
    base_batch.num_computed_tokens_np = computed
    base_batch.prefill_len_np = prefill_lens
    base_batch.num_computed_prefill_tokens_np = computed_prefill
    base_batch.is_prefilling_np = is_prefilling
    base_batch.seq_lens.copy_(torch.from_numpy(seq_lens))
    base_batch.seq_lens_cpu_upper_bound = torch.from_numpy(seq_lens.copy())
    base_batch.input_ids.copy_(torch.arange(num_tokens, dtype=torch.int32))
    base_batch.positions.copy_(torch.arange(num_tokens, dtype=torch.int64))
    base_batch.is_padding.fill_(False)
    base_batch.num_draft_tokens = int(draft_counts.sum())
    base_batch.num_draft_tokens_per_req = draft_counts
    base_batch.cu_num_logits_np = cu_num_logits_np
    base_batch.cu_num_logits = torch.from_numpy(cu_num_logits_np)
    base_batch.expanded_idx_mapping = torch.repeat_interleave(
        base_batch.idx_mapping, torch.from_numpy(draft_counts + 1).to(torch.int64)
    )
    base_batch.expanded_local_pos = torch.tensor(
        [0, 0, 1] if mixed_batch else [0, 1, 0, 1], dtype=torch.int32
    )

    return AscendInputBatch(
        **base_batch.__dict__,
        seq_lens_np=seq_lens.copy(),
        attn_state="global-attn-state",
    )


def _make_mtp_pcp_manager(global_batch):
    req_states = SimpleNamespace(
        last_sampled_tokens=torch.zeros(8, dtype=torch.int64),
        prefill_len=SimpleNamespace(gpu=torch.zeros(8, dtype=torch.int32)),
        draft_tokens=torch.zeros((8, 1), dtype=torch.int64),
    )
    vllm_config = SimpleNamespace(
        speculative_config=SimpleNamespace(
            method="mtp",
            num_speculative_tokens=1,
        )
    )
    return AscendPCPManager(
        pcp_world_size=2,
        pcp_rank=0,
        device=torch.device("cpu"),
        vllm_config=vllm_config,
        req_states=req_states,
        max_num_reqs=global_batch.num_reqs,
        max_num_tokens=global_batch.num_tokens,
    )


def _mtp_partition_patches(attn_state):
    return (
        patch(
            "vllm.v1.worker.gpu.pcp_manager.prepare_pos_seq_lens",
            return_value=None,
        ),
        patch(
            "vllm.v1.worker.gpu.pcp_manager.combine_sampled_and_draft_tokens",
            return_value=torch.zeros(8, dtype=torch.int64),
        ),
        patch(
            "vllm.v1.worker.gpu.pcp_manager.async_copy_to_gpu",
            side_effect=_mock_async_copy_to_cpu,
        ),
        patch.object(
            pcp_manager_module,
            "async_copy_to_gpu",
            side_effect=_mock_async_copy_to_cpu,
        ),
        patch.object(
            pcp_manager_module,
            "expand_idx_mapping",
            side_effect=_mock_expand_idx_mapping,
        ),
        patch.object(
            pcp_manager_module,
            "combine_sampled_and_draft_tokens",
            return_value=torch.arange(8, dtype=torch.int64),
        ),
        patch.object(
            pcp_manager_module,
            "build_attn_state",
            return_value=attn_state,
        ),
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


def test_partition_batch_preserves_prefix_cache_offset_for_continued_prefill():
    """A cache hit offsets each PCP-local segment into the existing sequence."""
    vllm_config = object()
    global_batch = _make_global_pcp_batch(num_computed_tokens=128)
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
        patch.object(pcp_manager_module, "build_attn_state", return_value=MagicMock()),
    ):
        result = manager.partition_batch(global_batch)

    # With a non-zero computed-token count neither segment is a pure prefill,
    # so rank 0 keeps its head chunk followed by its tail chunk. Both local
    # start positions include the 128-token prefix-cache hit.
    np.testing.assert_array_equal(
        result.num_scheduled_tokens, np.array([5, 3], dtype=np.int32)
    )
    np.testing.assert_array_equal(
        result.num_computed_tokens_np, np.array([128, 143], dtype=np.int32)
    )
    np.testing.assert_array_equal(
        result.seq_lens_np, np.array([133, 146], dtype=np.int32)
    )
    np.testing.assert_array_equal(
        result.num_computed_prefill_tokens_np,
        np.array([128, 143], dtype=np.int32),
    )


def test_partition_batch_rebuilds_mtp1_decode_fields_after_pcp_partition():
    global_batch = _make_global_mtp_batch(mixed_batch=False)
    manager = _make_mtp_pcp_manager(global_batch)
    attn_state = MagicMock()

    patches = _mtp_partition_patches(attn_state)
    with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], patches[6] as build_attn_state:
        result = manager.partition_batch(global_batch)

    assert manager._global_batch is global_batch
    assert result.req_ids == ["decode-a", "decode-b"]
    np.testing.assert_array_equal(
        result.num_scheduled_tokens, np.array([2, 2], dtype=np.int32)
    )
    np.testing.assert_array_equal(
        result.num_draft_tokens_per_req, np.array([1, 1], dtype=np.int32)
    )
    assert result.num_draft_tokens == 2
    np.testing.assert_array_equal(
        result.cu_num_logits_np, np.array([0, 2, 4], dtype=np.int32)
    )
    assert torch.equal(
        result.expanded_idx_mapping, torch.tensor([1, 1, 3, 3], dtype=torch.int32)
    )
    assert torch.equal(
        result.expanded_local_pos, torch.tensor([0, 1, 0, 1], dtype=torch.int32)
    )
    valid_tokens = build_attn_state.call_args.args[4]
    np.testing.assert_array_equal(valid_tokens, np.array([1, 1], dtype=np.int32))


def test_partition_batch_keeps_mtp_drafts_only_on_decode_rows_in_mixed_batch():
    global_batch = _make_global_mtp_batch(mixed_batch=True)
    manager = _make_mtp_pcp_manager(global_batch)

    patches = _mtp_partition_patches(MagicMock())
    with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], patches[6]:
        result = manager.partition_batch(global_batch)

    assert result.req_ids.count("prefill") == 2
    assert result.req_ids.count("decode") == 1
    expected_drafts = np.array(
        [1 if req_id == "decode" else 0 for req_id in result.req_ids],
        dtype=np.int32,
    )
    np.testing.assert_array_equal(result.num_draft_tokens_per_req, expected_drafts)
    assert result.num_draft_tokens == 1
    np.testing.assert_array_equal(
        np.diff(result.cu_num_logits_np), expected_drafts + 1
    )


def _make_validation_config(speculative_config, cudagraph_mode):
    return SimpleNamespace(
        parallel_config=SimpleNamespace(
            prefill_context_parallel_size=2,
            pipeline_parallel_size=1,
        ),
        model_config=SimpleNamespace(
            use_mla=True,
            is_encoder_decoder=False,
            hf_text_config=SimpleNamespace(),
        ),
        lora_config=None,
        speculative_config=speculative_config,
        compilation_config=SimpleNamespace(cudagraph_mode=cudagraph_mode),
    )


def test_validate_config_allows_eager_mtp1_with_pcp():
    config = _make_validation_config(
        SimpleNamespace(method="mtp", num_speculative_tokens=1),
        CUDAGraphMode.NONE,
    )
    AscendPCPManager.validate_config(config, supports_mm_inputs=False)


@pytest.mark.parametrize(
    "speculative_config,cudagraph_mode",
    [
        (SimpleNamespace(method="ngram", num_speculative_tokens=1), CUDAGraphMode.NONE),
        (SimpleNamespace(method="mtp", num_speculative_tokens=2), CUDAGraphMode.NONE),
        (
            SimpleNamespace(method="mtp", num_speculative_tokens=1),
            CUDAGraphMode.FULL_DECODE_ONLY,
        ),
    ],
)
def test_validate_config_rejects_unsupported_pcp_spec_modes(
    speculative_config, cudagraph_mode
):
    config = _make_validation_config(speculative_config, cudagraph_mode)
    with pytest.raises(NotImplementedError):
        AscendPCPManager.validate_config(config, supports_mm_inputs=False)


def test_maybe_build_ascend_pcp_manager_returns_none_when_pcp_is_disabled():
    vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(prefill_context_parallel_size=1),
    )

    assert (
        maybe_build_ascend_pcp_manager(
            vllm_config,
            torch.device("cpu"),
            supports_mm_inputs=False,
            req_states=MagicMock(),
            block_tables=MagicMock(),
        )
        is None
    )


def test_maybe_build_ascend_pcp_manager_uses_ascend_subclass():
    vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            prefill_context_parallel_size=2,
            decode_context_parallel_size=2,
            cp_kv_cache_interleave_size=4,
        ),
        scheduler_config=SimpleNamespace(max_num_seqs=8, max_num_batched_tokens=32),
    )
    pcp_group = SimpleNamespace(rank_in_group=1)
    dcp_group = SimpleNamespace(rank_in_group=0)
    req_states = MagicMock()

    with (
        patch.object(AscendPCPManager, "validate_config") as validate_config,
        patch.object(pcp_manager_module, "get_pcp_group", return_value=pcp_group),
        patch.object(pcp_manager_module, "get_dcp_group", return_value=dcp_group),
    ):
        manager = maybe_build_ascend_pcp_manager(
            vllm_config,
            torch.device("cpu"),
            supports_mm_inputs=False,
            req_states=req_states,
            block_tables=None,
        )

    assert isinstance(manager, AscendPCPManager)
    assert manager.vllm_config is vllm_config
    assert manager.pcp_world_size == 2
    assert manager.pcp_rank == 1
    assert manager.dcp_world_size == 2
    assert manager.dcp_rank == 0
    assert manager.cp_interleave == 4
    validate_config.assert_called_once_with(vllm_config, False)
