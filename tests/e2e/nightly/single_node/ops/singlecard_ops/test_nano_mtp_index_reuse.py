# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MTP reuses logical TopK rows, including after prefill and on graph replay."""

import math
from types import SimpleNamespace

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.attention.sfa_kv_offload import AscendSFAKVOffloadImpl
from vllm_ascend.utils import enable_custom_op

TOPK = 2048
HOT = 8192
BLOCK = 128
STRIDE_BLOCKS = HOT // BLOCK + 2
INVALID_SLOT = -(1 << 31)


def make_impl():
    impl = AscendSFAKVOffloadImpl.__new__(AscendSFAKVOffloadImpl)
    device = "npu:0"
    impl.nano_slot_map = torch.full((8, 16384), INVALID_SLOT, dtype=torch.int32, device=device)
    impl.nano_last_generation = torch.full((8,), -1, dtype=torch.int64, device=device)
    impl.nano_last_cache = torch.zeros(8, dtype=torch.int32, device=device)
    impl.nano_last_prefix = torch.zeros_like(impl.nano_last_cache)
    impl.nano_topk_src = torch.empty((8, 1, TOPK), dtype=torch.int32, device=device)
    impl.nano_topk_dst = torch.empty_like(impl.nano_topk_src)
    impl.nano_topk_misses = torch.empty(8, dtype=torch.int32, device=device)
    impl.nano_miss_src = torch.empty((4, 32768), dtype=torch.int32, device=device)
    impl.nano_miss_dst = torch.empty_like(impl.nano_miss_src)
    impl.nano_misses = torch.empty(4, dtype=torch.int32, device=device)
    impl.nano_reuse_slots = torch.arange(TOPK, dtype=torch.int32, device=device)
    impl.nano_reuse_cache_tokens = torch.empty(4, dtype=torch.int32, device=device)
    impl.nano_indexer_owner = impl
    impl.skip_topk = True
    impl.has_indexer = True
    impl.kv_lora_rank = 512
    impl.qk_rope_head_dim = 64
    impl.scale = 1 / math.sqrt(576)
    return impl


def metadata(active=True, length=10368):
    device = "npu:0"
    pools = torch.tensor([1 if active else 4, 5], dtype=torch.int32, device=device)
    return SimpleNamespace(
        num_decode_tokens=2,
        nano_pool_entries=pools,
        nano_active=torch.tensor([active, False], device=device),
        nano_generations=torch.tensor([11 if active else -1, -1], dtype=torch.int64, device=device),
        nano_seq_lens=torch.tensor([length, 1], dtype=torch.int32, device=device),
        nano_prefix_lens=torch.tensor([(length - 1) // BLOCK * BLOCK, 0], dtype=torch.int32, device=device),
        nano_query_ends=torch.tensor([1, 2], dtype=torch.int32, device=device),
        nano_hbm_block_table=pools[:, None] * STRIDE_BLOCKS
        + torch.arange(STRIDE_BLOCKS, dtype=torch.int32, device=device)[None],
        nano_source_block_table=torch.arange(256, dtype=torch.int32, device=device).reshape(2, 128),
        nano_token_active=torch.tensor([active, False], device=device),
    )


def saved_indices():
    # Includes source zero, a former tail source, and one invalid slot. These
    # are already the compacted rows from the proposer's ordinary TopK buffer.
    src = torch.arange(TOPK, dtype=torch.int32, device="npu:0") * 3
    src[-2] = 10366
    src[-1] = -1
    return torch.stack((src, torch.full_like(src, -1))).unsqueeze(1)


def test_reuse_resolves_compacted_rows_without_repeating_misses():
    impl, md, src = make_impl(), metadata(), saved_indices()
    selected = src[0, 0, :-1].long()
    slots = torch.arange(TOPK - 1, dtype=torch.int32, device="npu:0") + 4096
    impl.nano_slot_map[1, selected] = slots
    impl.nano_last_generation[1] = 11
    impl.nano_last_cache[1] = HOT
    # Poison the old un-compacted LIM outputs and its consumed copy counts.
    impl.nano_topk_dst.fill_(777)
    impl.nano_misses.fill_(HOT)
    actual_cache = impl._prepare_nano_reused_topk(src, md)
    assert actual_cache.cpu().tolist() == [HOT, TOPK]
    assert impl.nano_misses[:2].cpu().tolist() == [0, 0]
    torch.testing.assert_close(impl.nano_topk_dst[0, 0, :-1], slots)
    assert impl.nano_topk_dst[0, 0, -1].item() == -1
    assert impl.nano_slot_map[1, 0].item() == 4096
    assert impl.nano_slot_map[5].eq(INVALID_SLOT).all().item()


@pytest.mark.parametrize("graph", [False, True])
def test_prefill_bootstrap_then_reuse_exact_attention_across_block_boundary(graph):
    enable_custom_op()
    torch.manual_seed(37)
    impl, md, src = make_impl(), metadata(active=not graph), saved_indices()
    hbm_k = torch.full((8 * STRIDE_BLOCKS, BLOCK, 1, 512), 7.0, dtype=torch.bfloat16, device="npu:0")
    hbm_r = torch.full((8 * STRIDE_BLOCKS, BLOCK, 1, 64), 9.0, dtype=torch.bfloat16, device="npu:0")
    # Device-backed source memory exercises the same copy-SFA contract as the
    # shared host/GVA cache, without requiring a separate MemFabric lifecycle.
    source_k = torch.randn((256, BLOCK, 512), dtype=torch.bfloat16, device="npu:0")
    source_r = torch.randn((256, BLOCK, 64), dtype=torch.bfloat16, device="npu:0")
    manager = SimpleNamespace(
        topk_buffers_k=[hbm_k],
        topk_buffers_v=[hbm_r],
        k_caches_cpu=[source_k],
        v_caches_cpu=[source_r],
        _get_offload_layer_id=lambda _: 0,
    )
    query = torch.randn((2, 16, 512), dtype=torch.bfloat16, device="npu:0")
    rope = torch.randn((2, 16, 64), dtype=torch.bfloat16, device="npu:0")
    indices = src[0, 0, :-1].long()
    selected_k = source_k.view(-1, 512)[indices].float()
    selected_r = source_r.view(-1, 64)[indices].float()
    scores = (query[0].float() @ selected_k.T + rope[0].float() @ selected_r.T) * impl.scale
    expected = torch.softmax(scores, dim=-1) @ selected_k

    def forward():
        return impl._nano_attention(query, rope, src, md, manager, "mtp.attn")

    if graph:
        for _ in range(3):
            forward()
        captured = torch.npu.NPUGraph()
        with torch.npu.graph(captured):
            out = forward()
        active = metadata()
        for name, value in vars(active).items():
            if isinstance(value, torch.Tensor):
                getattr(md, name).copy_(value)
        captured.replay()
    else:
        out = forward()
    torch.npu.synchronize()
    torch.testing.assert_close(out[0].float(), expected, rtol=0.03, atol=0.08)
    assert out[1].count_nonzero().item() == 0
    assert impl.nano_misses[:2].cpu().tolist() == [TOPK, 0]
    assert impl.nano_slot_map[1, 0].item() == 0
    assert impl.nano_slot_map[1, 10366].item() == TOPK - 2
    torch.testing.assert_close(
        hbm_k[STRIDE_BLOCKS : STRIDE_BLOCKS + TOPK // BLOCK].reshape(-1, 512)[:-1],
        selected_k.to(torch.bfloat16),
        rtol=0,
        atol=0,
    )
    assert hbm_k[5 * STRIDE_BLOCKS : 6 * STRIDE_BLOCKS].eq(7).all().item()

    # Cross a 128-token boundary and change source memory. The saved selection
    # must use the already loaded cache, with no H2D and no added dense tail.
    md.nano_seq_lens[0] += 1
    md.nano_prefix_lens[0] += BLOCK
    source_k.zero_()
    source_r.zero_()
    if graph:
        captured.replay()
    else:
        out = forward()
    torch.npu.synchronize()
    torch.testing.assert_close(out[0].float(), expected, rtol=0.03, atol=0.08)
    assert impl.nano_misses[:2].cpu().tolist() == [0, 0]

    # A new generation must discard the old map and refill from the now-zero
    # source cache, even when it reuses the same physical pool slot.
    md.nano_generations[0] += 1
    if graph:
        captured.replay()
    else:
        out = forward()
    torch.npu.synchronize()
    assert out.count_nonzero().item() == 0
    assert impl.nano_misses[:2].cpu().tolist() == [TOPK, 0]
