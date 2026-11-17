# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import sys
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest
import torch
from vllm.config import CUDAGraphMode

from vllm_ascend.attention.flash_attention_v3 import (
    AscendFlashAttentionBackend,
    AscendFlashAttentionImpl,
)


@pytest.fixture
def ops(monkeypatch):
    operators = NS(
        get_scheduler_metadata=Mock(side_effect=lambda **_: torch.zeros(8, dtype=torch.uint8)),
        flash_attn_with_kvcache=Mock(side_effect=lambda query, *_args, **_kwargs: query + 1),
    )
    monkeypatch.setitem(sys.modules, "flash_attn_npu_3", operators)
    return operators


def make_builder(mode, eager=False, draft_roles=(True, True)):
    impl = AscendFlashAttentionImpl(8, 192, 192**-0.5, num_kv_heads=2)
    config = NS(
        speculative_config=NS(method="dflash", enforce_eager=eager, num_speculative_tokens=7),
        scheduler_config=NS(max_num_seqs=4),
        compilation_config=NS(
            cudagraph_mode=mode,
            cudagraph_capture_sizes=[15],
            static_forward_context={
                f"attn.{i}": NS(impl=impl, is_draft_layer=is_draft) for i, is_draft in enumerate(draft_roles)
            },
        ),
    )
    return AscendFlashAttentionBackend.get_builder_cls()(
        NS(dtype=torch.bfloat16, block_size=128), ["attn.0", "attn.1"], config, torch.device("cpu")
    ), impl


def make_common():
    offsets = torch.arange(5, dtype=torch.int32) * 4
    return NS(
        num_reqs=4,
        query_start_loc_cpu=offsets,
        query_start_loc=offsets.clone(),
        seq_lens=torch.tensor([21, 31, 41, 51], dtype=torch.int32),
        block_table_tensor=torch.zeros(4, 8, dtype=torch.int32),
        num_input_tokens=16,
        num_actual_tokens=16,
        max_query_len=4,
        causal=False,
        slot_mapping=torch.arange(16, dtype=torch.int32),
    )


@pytest.mark.parametrize("eager", [False, True])
def test_dflash_rounded_bucket_preserves_exact_device_lengths_and_tiling_storage(ops, eager):
    builder, _ = make_builder(CUDAGraphMode.FULL_DECODE_ONLY, eager)
    common = make_common()
    captured = builder.build_for_graph_capture(common)
    spec = next(iter(captured.scheduler_metadata))
    for offsets in ([0, 4, 4, 4, 4], [0, 4, 8, 12, 12], [0, 4, 8, 12, 16]):
        common.query_start_loc_cpu.copy_(torch.tensor(offsets))
        common.query_start_loc.copy_(common.query_start_loc_cpu)
        common.seq_lens.sub_(1)
        metadata = builder.build(0, common)
        active = offsets.index(offsets[-1])
        arguments = ops.get_scheduler_metadata.call_args.kwargs
        assert arguments["batch_size"] == active
        assert arguments["max_seqlen_q"] == 4
        assert arguments["cache_seqlens"].data_ptr() == common.seq_lens.data_ptr()
        torch.testing.assert_close(metadata.seq_lens, common.seq_lens[:active])
        torch.testing.assert_close(metadata.query_start_loc, common.query_start_loc[: active + 1])
        # The target's 15-token bucket rounds to a 16-token draft graph.
        assert (metadata.scheduler_metadata[spec] is captured.scheduler_metadata[spec]) is not eager
    assert ops.get_scheduler_metadata.call_count == 4  # Shared by two equivalent layers.
    assert bool(builder.scheduler_buffers) is not eager


@pytest.mark.parametrize("mode", [CUDAGraphMode.NONE, CUDAGraphMode.FULL, CUDAGraphMode.FULL_DECODE_ONLY])
def test_qwen_full_attention_updates_ragged_tnd_and_padded_tables_without_changing_graph_addresses(ops, mode):
    # Draft eager must not disable the target's graph buffers.
    builder, impl = make_builder(mode, eager=True, draft_roles=(False, False))
    common = make_common()
    common.causal = True
    captured = builder.build(0, common)
    query = torch.randn(16, 8, 192, dtype=torch.bfloat16)
    layer = NS(_k_scale_float=1.0, _v_scale_float=1.0)
    for offsets in ([0, 3, 11, 11, 11], [0, 2, 2, 2, 2], [0, 4, 8, 12, 16]):
        common.query_start_loc_cpu.copy_(torch.tensor(offsets))
        common.query_start_loc.copy_(common.query_start_loc_cpu)
        common.max_query_len = max(b - a for a, b in zip(offsets, offsets[1:]))
        common.seq_lens.sub_(1)
        common.block_table_tensor.add_(3)
        metadata = builder.build(0, common)
        active = metadata.seq_lens.numel()
        assert metadata.block_tables.data_ptr() == captured.block_tables.data_ptr()
        torch.testing.assert_close(metadata.block_tables[:, :-1], common.block_table_tensor[:active])
        assert not metadata.block_tables[:, -1].any()
        if mode != CUDAGraphMode.NONE:
            for spec, buffer in metadata.scheduler_metadata.items():
                assert buffer is captured.scheduler_metadata[spec]
        tiling = ops.get_scheduler_metadata.call_args.kwargs
        expected_q = (
            common.max_query_len
            if mode == CUDAGraphMode.NONE
            else (8 if mode == CUDAGraphMode.FULL_DECODE_ONLY else 16)
        )
        assert tiling["max_seqlen_q"] == expected_q
        assert tiling["max_seqlen_k"] == 9 * 128
        assert tiling["cache_seqlens"].data_ptr() == common.seq_lens.data_ptr()
        output = torch.empty_like(query)
        impl.forward(layer, query, None, None, (None, None), metadata, output)
        arguments = ops.flash_attn_with_kvcache.call_args.kwargs
        assert ops.flash_attn_with_kvcache.call_args.args[0] is query
        assert arguments["cu_seqlens_q"] is metadata.query_start_loc
        assert arguments["max_seqlen_q"] == expected_q
        assert arguments["window_size"] == tiling["window_size"] == (1023, 0)
        assert arguments["scheduler_metadata"] is next(iter(metadata.scheduler_metadata.values()))
        torch.testing.assert_close(output, query + 1)


def test_shared_builder_keeps_mixed_target_groups_separate_from_draft_metadata(ops):
    # Main groups can contain both roles; the speculator filters to draft layers.
    target_builder, _ = make_builder(CUDAGraphMode.FULL_DECODE_ONLY, draft_roles=(False, True))
    draft_builder, _ = make_builder(CUDAGraphMode.FULL_DECODE_ONLY)
    common = make_common()
    target = target_builder.build(0, common)
    draft = draft_builder.build(0, common)
    assert type(target_builder) is type(draft_builder)
    assert target.full_attention_window == 1024
    assert draft.full_attention_window is None
    assert target.max_query_len == 8 and draft.max_query_len == 4
    assert target.block_tables.shape[1] == 9 and draft.block_tables.shape[1] == 8
    target_tiling = next(iter(target.scheduler_metadata.values()))
    draft_tiling = next(iter(draft.scheduler_metadata.values()))
    assert target_tiling.data_ptr() != draft_tiling.data_ptr()
