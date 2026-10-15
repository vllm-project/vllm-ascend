# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MTP reuses the compacted LIM output without per-draft metadata rebuilds."""

import math
from types import SimpleNamespace

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.attention.sfa_kv_offload import AscendSFAKVOffloadImpl
from vllm_ascend.utils import enable_custom_op

TOPK = 2048
BLOCK = 128
STRIDE_BLOCKS = TOPK // BLOCK + 2


def make_impl():
    impl = AscendSFAKVOffloadImpl.__new__(AscendSFAKVOffloadImpl)
    device = "npu:0"
    impl.lim_topk_src = torch.empty((8, 1, TOPK), dtype=torch.int32, device=device)
    impl.lim_topk_dst = torch.empty_like(impl.lim_topk_src)
    impl.lim_topk_misses = torch.full((8,), TOPK, dtype=torch.int32, device=device)
    impl.lim_miss_src = torch.zeros((4, 32768), dtype=torch.int32, device=device)
    impl.lim_miss_dst = torch.zeros_like(impl.lim_miss_src)
    impl.lim_misses = torch.full((4,), TOPK, dtype=torch.int32, device=device)
    impl.copy_sfa_reuse_logical_lens = torch.full((4,), TOPK, dtype=torch.int32, device=device)
    impl.copy_sfa_reuse_logical_lens[1] = 0
    impl.copy_sfa_reuse_cache_tokens = torch.full((4,), TOPK, dtype=torch.int32, device=device)
    impl.lim_reuse_topk_misses = torch.zeros(8, dtype=torch.int32, device=device)
    impl.lim_reuse_misses = torch.zeros(4, dtype=torch.int32, device=device)
    impl.lim_reuse_request_count = 2
    impl.lim_indexer_owner = impl
    impl.skip_topk = True
    impl.has_indexer = True
    impl.kv_lora_rank = 512
    impl.qk_rope_head_dim = 64
    impl.scale = 1 / math.sqrt(576)
    return impl


def metadata():
    device = "npu:0"
    pools = torch.tensor([1, 5], dtype=torch.int32, device=device)
    return SimpleNamespace(
        num_decode_tokens=2,
        copy_sfa_pool_entries=pools,
        copy_sfa_query_ends=torch.tensor([1, 2], dtype=torch.int32, device=device),
        copy_sfa_hbm_block_table=pools[:, None] * STRIDE_BLOCKS
        + torch.arange(STRIDE_BLOCKS, dtype=torch.int32, device=device)[None],
        copy_sfa_source_block_table=torch.arange(256, dtype=torch.int32, device=device).reshape(2, 128),
    )


def seed_step0_rows(impl):
    rows = torch.arange(8, dtype=torch.int32, device="npu:0")[:, None, None]
    sources = torch.arange(TOPK, dtype=torch.int32, device="npu:0")[None, None]
    impl.lim_topk_src.copy_(sources + rows * TOPK)
    impl.lim_topk_dst.copy_(sources.expand(8, -1, -1))
    impl.lim_topk_dst[6].fill_(-1)


def test_compaction_preserves_complete_step0_lim_rows():
    impl = make_impl()
    seed_step0_rows(impl)
    source_before = impl.lim_topk_src.clone()
    destination_before = impl.lim_topk_dst.clone()
    indices = torch.tensor([1, 6], dtype=torch.int32, device="npu:0")

    impl.compact_lim_topk_metadata(indices)

    torch.testing.assert_close(impl.lim_topk_src[:2], source_before[indices])
    torch.testing.assert_close(impl.lim_topk_dst[:2], destination_before[indices])
    assert impl.lim_reuse_topk_misses.count_nonzero().item() == 0
    assert impl.lim_reuse_misses.count_nonzero().item() == 0


def test_compaction_ignores_graph_padding_rows():
    impl = make_impl()
    seed_step0_rows(impl)
    source_before = impl.lim_topk_src.clone()
    destination_before = impl.lim_topk_dst.clone()
    padded_indices = torch.zeros(2048, dtype=torch.int32, device="npu:0")
    padded_indices[:2] = torch.tensor([1, 6], dtype=torch.int32, device="npu:0")

    impl.compact_lim_topk_metadata(padded_indices)

    torch.testing.assert_close(impl.lim_topk_src[:2], source_before[padded_indices[:2]])
    torch.testing.assert_close(impl.lim_topk_dst[:2], destination_before[padded_indices[:2]])


@pytest.mark.parametrize("graph", [False, True])
def test_later_draft_reuses_compacted_selection_without_copy(graph):
    enable_custom_op()
    torch.manual_seed(37)
    impl, md = make_impl(), metadata()
    seed_step0_rows(impl)
    impl.lim_topk_src[1].copy_(torch.arange(TOPK, dtype=torch.int32, device="npu:0").view(1, -1))
    impl.compact_lim_topk_metadata(torch.tensor([1, 6], dtype=torch.int32, device="npu:0"))

    hbm_k = torch.full((8 * STRIDE_BLOCKS, BLOCK, 1, 512), 7.0, dtype=torch.bfloat16, device="npu:0")
    hbm_r = torch.full((8 * STRIDE_BLOCKS, BLOCK, 1, 64), 9.0, dtype=torch.bfloat16, device="npu:0")
    selected_k = torch.randn((TOPK, 512), dtype=torch.bfloat16, device="npu:0")
    selected_r = torch.randn((TOPK, 64), dtype=torch.bfloat16, device="npu:0")
    hbm_k[STRIDE_BLOCKS : STRIDE_BLOCKS + TOPK // BLOCK].reshape(-1, 512).copy_(selected_k)
    hbm_r[STRIDE_BLOCKS : STRIDE_BLOCKS + TOPK // BLOCK].reshape(-1, 64).copy_(selected_r)

    # Poison the source cache and ordinary miss counts. Direct reuse must read
    # the populated step-0 HBM selection and the persistent zero-count buffers.
    source_k = torch.zeros((256, BLOCK, 512), dtype=torch.bfloat16, device="npu:0")
    source_r = torch.zeros((256, BLOCK, 64), dtype=torch.bfloat16, device="npu:0")
    manager = SimpleNamespace(
        topk_buffers_k=[hbm_k],
        topk_buffers_v=[hbm_r],
        k_caches_cpu=[source_k],
        v_caches_cpu=[source_r],
        _get_offload_layer_id=lambda _: 0,
    )
    query = torch.randn((2, 16, 512), dtype=torch.bfloat16, device="npu:0")
    rope = torch.randn((2, 16, 64), dtype=torch.bfloat16, device="npu:0")
    scores = (query[0].float() @ selected_k.float().T + rope[0].float() @ selected_r.float().T) * impl.scale
    expected = torch.softmax(scores, dim=-1) @ selected_k.float()

    def forward():
        return impl._copy_sfa_attention(query, rope, impl.lim_topk_src, md, manager, "mtp.attn")

    if graph:
        for _ in range(3):
            forward()
        captured = torch.npu.NPUGraph()
        with torch.npu.graph(captured):
            out = forward()
        captured.replay()
    else:
        out = forward()
    torch.npu.synchronize()

    torch.testing.assert_close(out[0].float(), expected, rtol=0.03, atol=0.08)
    assert out[1].count_nonzero().item() == 0
    assert impl.lim_reuse_topk_misses.count_nonzero().item() == 0
    assert impl.lim_reuse_misses.count_nonzero().item() == 0
    assert source_k.count_nonzero().item() == 0
    assert source_r.count_nonzero().item() == 0


@pytest.mark.parametrize("query_count", [1, 4])
@pytest.mark.parametrize("first_fill", [False, True])
def test_copy_sfa_graph_zero_lengths_follow_replay_and_preserve_inactive_cache(query_count, first_fill):
    """Exercise native output zeroing with MTP3 widths and both copy paths."""
    enable_custom_op()
    device = "npu:0"
    impl, md = make_impl(), metadata()
    impl.skip_topk = False
    md.num_decode_tokens = 2 * query_count
    md.copy_sfa_query_ends.copy_(torch.tensor([query_count, 2 * query_count], dtype=torch.int32, device=device))
    md.copy_sfa_logical_lens = torch.zeros(2, dtype=torch.int32, device=device)
    md.copy_sfa_cache_tokens = torch.full((2,), TOPK, dtype=torch.int32, device=device)
    impl.lim_topk_src.copy_(torch.arange(TOPK, dtype=torch.int32, device=device).view(1, 1, -1))
    impl.lim_topk_dst.copy_(impl.lim_topk_src)
    impl.lim_topk_misses.zero_()
    impl.lim_misses.zero_()
    impl.lim_miss_src[:, :TOPK].copy_(impl.lim_topk_src[0])
    impl.lim_miss_dst[:, :TOPK].copy_(impl.lim_topk_dst[0])

    hbm_k = torch.full((8 * STRIDE_BLOCKS, BLOCK, 1, 512), 7.0, dtype=torch.bfloat16, device=device)
    hbm_r = torch.full((8 * STRIDE_BLOCKS, BLOCK, 1, 64), 9.0, dtype=torch.bfloat16, device=device)
    source_k = torch.full((256, BLOCK, 512), 11.0, dtype=torch.bfloat16, device=device)
    source_r = torch.full((256, BLOCK, 64), 13.0, dtype=torch.bfloat16, device=device)
    manager = SimpleNamespace(
        topk_buffers_k=[hbm_k],
        topk_buffers_v=[hbm_r],
        k_caches_cpu=[source_k],
        v_caches_cpu=[source_r],
        _get_offload_layer_id=lambda _: 0,
    )
    # Zero queries make the active reference simply the mean of visible KV.
    query = torch.zeros((2 * query_count, 16, 512), dtype=torch.bfloat16, device=device)
    rope = torch.zeros((2 * query_count, 16, 64), dtype=torch.bfloat16, device=device)

    def forward():
        return impl._copy_sfa_attention(query, rope, impl.lim_topk_src, md, manager, "target.attn")

    # Capture all rows inactive, then change only device input contents.
    for _ in range(3):
        assert forward().count_nonzero().item() == 0
    captured = torch.npu.NPUGraph()
    with torch.npu.graph(captured):
        out = forward()
    for active_row in (0, 1, None):
        hbm_k.fill_(7)
        hbm_r.fill_(9)
        md.copy_sfa_logical_lens.zero_()
        impl.lim_misses.zero_()
        if active_row is not None:
            md.copy_sfa_logical_lens[active_row] = TOPK + query_count - 1
            if first_fill:
                impl.lim_misses[active_row] = TOPK
        eager = forward()
        captured.replay()
        torch.npu.synchronize()
        torch.testing.assert_close(out, eager)
        for row in range(2):
            actual = out[row * query_count : (row + 1) * query_count]
            if row != active_row:
                assert actual.count_nonzero().item() == 0
                pool = (1, 5)[row]
                assert torch.all(hbm_k[pool * STRIDE_BLOCKS : (pool + 1) * STRIDE_BLOCKS] == 7).item()
                assert torch.all(hbm_r[pool * STRIDE_BLOCKS : (pool + 1) * STRIDE_BLOCKS] == 9).item()
            else:
                values = [((11 if first_fill else 7) * TOPK + 7 * q) / (TOPK + q) for q in range(query_count)]
                expected = torch.tensor(values, dtype=torch.float32, device=device)[:, None, None].expand_as(actual)
                torch.testing.assert_close(actual.float(), expected, rtol=0.01, atol=0.08)
        assert torch.all(source_k == 11).item()
        assert torch.all(source_r == 13).item()
