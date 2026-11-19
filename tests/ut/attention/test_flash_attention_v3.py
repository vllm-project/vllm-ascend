# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace as NS

import pytest
import torch
from vllm.config import CUDAGraphMode


def make_builder(fa3_backend, mode, eager=False, target=False, method="dflash"):
    impl = fa3_backend.AscendFlashAttentionImpl(8, 192, 192**-0.5, num_kv_heads=2)
    config = NS(
        speculative_config=NS(method=method, enforce_eager=eager, num_speculative_tokens=7) if method else None,
        scheduler_config=NS(max_num_seqs=4),
        compilation_config=NS(
            cudagraph_mode=mode,
            cudagraph_capture_sizes=[15],
            static_forward_context={f"attn.{i}": NS(impl=impl, is_draft_layer=not target) for i in range(2)},
        ),
    )
    return fa3_backend.AscendFlashAttentionBackend.get_builder_cls()(
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


@pytest.mark.parametrize("method,target", [("dflash", False), ("dflash", True), ("dspark", False), (None, True)])
@pytest.mark.parametrize(
    "offsets,num_actual_tokens,active_reqs",
    [
        ([0, 4, 8, 12, 16], 16, 4),
        ([0, 4, 4, 4, 4], 4, 1),
        ([0, 4, 4, 4, 4], 16, 1),  # Draft counts can include graph padding.
        ([0, 4, 8, 12, 16], 4, 1),  # Target offsets can include dummy requests.
        ([0, 0, 0, 0, 0], 0, 1),
    ],
)
def test_active_requests_exclude_graph_padding(fa3_backend, method, target, offsets, num_actual_tokens, active_reqs):
    builder, _ = make_builder(fa3_backend, CUDAGraphMode.NONE, target=target, method=method)
    common = make_common()
    common.query_start_loc_cpu.copy_(torch.tensor(offsets))
    common.query_start_loc.copy_(common.query_start_loc_cpu)
    common.num_actual_tokens = num_actual_tokens
    common.causal = target
    metadata = builder.build(0, common)
    assert fa3_backend.get_scheduler_metadata.call_args.kwargs["batch_size"] == active_reqs
    assert metadata.query_start_loc.numel() == active_reqs + 1
    assert metadata.seq_lens.numel() == active_reqs
    assert metadata.block_tables.shape[0] == active_reqs


@pytest.mark.parametrize("eager", [False, True])
@pytest.mark.parametrize("method,query_len", [("dflash", 8), ("dspark", 7)])
def test_speculative_bucket_preserves_exact_device_lengths_and_tiling_storage(fa3_backend, eager, method, query_len):
    builder, _ = make_builder(fa3_backend, CUDAGraphMode.FULL_DECODE_ONLY, eager, method=method)
    common = make_common()
    common.query_start_loc_cpu.copy_(torch.arange(5) * query_len)
    common.query_start_loc.copy_(common.query_start_loc_cpu)
    common.num_actual_tokens = common.num_input_tokens = 4 * query_len
    common.max_query_len = query_len
    captured = builder.build_for_graph_capture(common)
    spec = next(iter(captured.scheduler_metadata))
    for active in (1, 3, 4):
        common.query_start_loc_cpu.copy_(torch.arange(5).clamp(max=active) * query_len)
        common.query_start_loc.copy_(common.query_start_loc_cpu)
        common.seq_lens.sub_(1)
        metadata = builder.build(0, common)
        arguments = fa3_backend.get_scheduler_metadata.call_args.kwargs
        assert arguments["batch_size"] == active
        assert arguments["max_seqlen_q"] == query_len
        assert arguments["cache_seqlens"].data_ptr() == common.seq_lens.data_ptr()
        torch.testing.assert_close(metadata.seq_lens, common.seq_lens[:active])
        torch.testing.assert_close(metadata.query_start_loc, common.query_start_loc[: active + 1])
        # Actual draft graph sizes need not appear in the target's size list.
        assert (metadata.scheduler_metadata[spec] is captured.scheduler_metadata[spec]) is not eager
    assert fa3_backend.get_scheduler_metadata.call_count == 4  # Shared by two equivalent layers.
    assert bool(builder.scheduler_buffers) is not eager


@pytest.mark.parametrize("draft_eager", [False, True])
def test_target_full_attention_preserves_pages_and_reuses_tiling_after_rejection(fa3_backend, draft_eager):
    builder, impl = make_builder(fa3_backend, CUDAGraphMode.FULL_DECODE_ONLY, draft_eager, target=True)
    common = make_common()
    common.causal = True
    captured = builder.build_for_graph_capture(common)
    spec = next(iter(captured.scheduler_metadata))
    common.query_start_loc_cpu.copy_(torch.tensor([0, 1, 3, 6, 10]))
    common.query_start_loc.copy_(common.query_start_loc_cpu)
    common.seq_lens.sub_(1)
    common.max_query_len = 4
    metadata = builder.build(0, common)
    arguments = fa3_backend.get_scheduler_metadata.call_args.kwargs
    assert arguments["window_size"] == (-1, -1)
    assert arguments["max_seqlen_k"] == 1024
    assert metadata.block_tables.data_ptr() == common.block_table_tensor.data_ptr()
    assert metadata.block_tables.shape == common.block_table_tensor.shape
    assert metadata.max_query_len == captured.max_query_len == 8
    assert metadata.scheduler_metadata[spec] is captured.scheduler_metadata[spec]
    query = torch.zeros(16, 8, 192, dtype=torch.bfloat16)
    cache = torch.zeros(2, 1, 128, 2, 192, dtype=query.dtype)
    layer = NS(layer_name="target.attn", _k_scale_float=1.0, _v_scale_float=1.0)
    impl.forward(layer, query, None, None, cache, metadata, torch.empty_like(query))
    forward_args = fa3_backend.flash_attn_with_kvcache.call_args.kwargs
    assert forward_args["causal"] is True
    assert forward_args["window_size"] == (-1, -1)
    assert forward_args["page_table"] is metadata.block_tables
