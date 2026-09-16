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
    impl.nano_topk_src = torch.empty((8, 1, TOPK), dtype=torch.int32, device=device)
    impl.nano_topk_dst = torch.empty_like(impl.nano_topk_src)
    impl.nano_topk_misses = torch.full((8,), TOPK, dtype=torch.int32, device=device)
    impl.nano_miss_src = torch.zeros((4, 32768), dtype=torch.int32, device=device)
    impl.nano_miss_dst = torch.zeros_like(impl.nano_miss_src)
    impl.nano_misses = torch.full((4,), TOPK, dtype=torch.int32, device=device)
    impl.nano_reuse_cache_tokens = torch.full((4,), TOPK, dtype=torch.int32, device=device)
    impl.nano_reuse_topk_misses = torch.zeros(8, dtype=torch.int32, device=device)
    impl.nano_reuse_misses = torch.zeros(4, dtype=torch.int32, device=device)
    impl.nano_reuse_request_count = 2
    impl.nano_indexer_owner = impl
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
        nano_pool_entries=pools,
        nano_token_active=torch.tensor([True, False], device=device),
        nano_query_ends=torch.tensor([1, 2], dtype=torch.int32, device=device),
        nano_hbm_block_table=pools[:, None] * STRIDE_BLOCKS
        + torch.arange(STRIDE_BLOCKS, dtype=torch.int32, device=device)[None],
        nano_source_block_table=torch.arange(256, dtype=torch.int32, device=device).reshape(2, 128),
    )


def seed_step0_rows(impl):
    rows = torch.arange(8, dtype=torch.int32, device="npu:0")[:, None, None]
    sources = torch.arange(TOPK, dtype=torch.int32, device="npu:0")[None, None]
    impl.nano_topk_src.copy_(sources + rows * TOPK)
    impl.nano_topk_dst.copy_(sources.expand(8, -1, -1))
    impl.nano_topk_dst[6].fill_(-1)


def test_compaction_preserves_complete_step0_lim_rows():
    impl = make_impl()
    seed_step0_rows(impl)
    source_before = impl.nano_topk_src.clone()
    destination_before = impl.nano_topk_dst.clone()
    indices = torch.tensor([1, 6], dtype=torch.int32, device="npu:0")

    impl.compact_nano_topk_metadata(indices)

    torch.testing.assert_close(impl.nano_topk_src[:2], source_before[indices])
    torch.testing.assert_close(impl.nano_topk_dst[:2], destination_before[indices])
    assert impl.nano_reuse_topk_misses.count_nonzero().item() == 0
    assert impl.nano_reuse_misses.count_nonzero().item() == 0


def test_compaction_ignores_graph_padding_rows():
    impl = make_impl()
    seed_step0_rows(impl)
    source_before = impl.nano_topk_src.clone()
    destination_before = impl.nano_topk_dst.clone()
    padded_indices = torch.zeros(2048, dtype=torch.int32, device="npu:0")
    padded_indices[:2] = torch.tensor([1, 6], dtype=torch.int32, device="npu:0")

    impl.compact_nano_topk_metadata(padded_indices)

    torch.testing.assert_close(impl.nano_topk_src[:2], source_before[padded_indices[:2]])
    torch.testing.assert_close(impl.nano_topk_dst[:2], destination_before[padded_indices[:2]])


@pytest.mark.parametrize("graph", [False, True])
def test_later_draft_reuses_compacted_selection_without_copy(graph):
    enable_custom_op()
    torch.manual_seed(37)
    impl, md = make_impl(), metadata()
    seed_step0_rows(impl)
    impl.nano_topk_src[1].copy_(torch.arange(TOPK, dtype=torch.int32, device="npu:0").view(1, -1))
    impl.compact_nano_topk_metadata(torch.tensor([1, 6], dtype=torch.int32, device="npu:0"))

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
        return impl._nano_attention(query, rope, impl.nano_topk_src, md, manager, "mtp.attn")

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
    assert impl.nano_reuse_topk_misses.count_nonzero().item() == 0
    assert impl.nano_reuse_misses.count_nonzero().item() == 0
    assert source_k.count_nonzero().item() == 0
    assert source_r.count_nonzero().item() == 0
