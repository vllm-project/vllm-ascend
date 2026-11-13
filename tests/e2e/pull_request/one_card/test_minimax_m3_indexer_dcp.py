# SPDX-License-Identifier: Apache-2.0
"""Check index cache ownership and causal scoring on an actual NPU."""

import pytest
import torch

import vllm_ascend.ops  # noqa: F401
from vllm_ascend.models.minimax_m3.ops.indexer_dcp import scatter_index_shard


@pytest.mark.parametrize("size", [2, 4])
@pytest.mark.parametrize("padded", [False, True])
@torch.no_grad()
def test_index_shard_scatter_skips_padding(size, padded):
    slots = torch.tensor([0, 1, 2, 3, 127, 128, 129, -1], dtype=torch.int64, device="npu")
    updates = torch.randn(8, 256, dtype=torch.bfloat16, device="npu")[:, ::2]
    for rank in range(size):
        storage = torch.zeros(3, 128 // size, 128, dtype=torch.bfloat16, device="npu")
        if padded:
            storage = torch.full((3, 128 // size + 7, 256), -777, dtype=torch.bfloat16, device="npu")
        expected_storage = storage.clone()
        cache = storage[:, : 128 // size, ::2] if padded else storage
        expected = expected_storage[:, : 128 // size, ::2] if padded else expected_storage
        scatter_index_shard(cache, updates, slots, size, rank)
        for row, slot in enumerate([0, 1, 2, 3, 127, 128, 129]):
            if slot % size == rank:
                expected[slot // 128, slot % 128 // size] = updates[row]
        torch.testing.assert_close(storage, expected_storage, rtol=0, atol=0)


@pytest.mark.parametrize("size", [2, 4])
@pytest.mark.parametrize("query_len", [1, 8, 17])
@pytest.mark.parametrize("context", [129, 601, 131077])
@pytest.mark.parametrize("padded", [False, True])
@torch.no_grad()
def test_compact_shards_reconstruct_exact_cann_scores(size, query_len, context, padded):
    from vllm_ascend.models.minimax_m3.ops.indexer_dcp import index_shard_scores
    from vllm_ascend.models.minimax_m3.ops.msa_m3_npu import _minimax_m3_index_score

    torch.manual_seed(123)
    blocks = (context + 127) // 128
    full = torch.randn(blocks + 7, 128, 128, dtype=torch.bfloat16, device="npu")
    table = torch.stack([torch.randperm(blocks + 7, device="npu")[:blocks] for _ in range(3)]).to(torch.int32)
    lengths = torch.tensor([context, context - 47, 0], dtype=torch.int32, device="npu")
    queries = torch.randn(3 * query_len, 4, 128, dtype=torch.bfloat16, device="npu")
    cu_seqlens = torch.arange(4, dtype=torch.int32, device="npu") * query_len
    starts = (lengths - query_len) // 128
    mask = torch.triu(torch.ones(2048, 2048, dtype=torch.int8, device="npu"), diagonal=1)
    expected = _minimax_m3_index_score(queries, full, table, cu_seqlens, lengths, starts, mask)
    parts = []
    for rank in range(size):
        local = full[:, rank::size, :].contiguous()
        if padded:
            storage = torch.empty(blocks + 9, 128 // size + 3, 256, dtype=full.dtype, device="npu")
            view = storage[1:-1, : 128 // size, ::2]
            view.copy_(local)
            local = view
        parts.append(index_shard_scores(queries, local, table, lengths, query_len, size, rank))
    torch.testing.assert_close(torch.stack(parts).amax(0), expected, rtol=0, atol=0)


@pytest.mark.parametrize("size", [2, 4])
@pytest.mark.parametrize("separate_index", [False, True])
@torch.no_grad()
def test_fused_prepare_preserves_index_shard_ownership_and_guards(size, separate_index):
    from tests.e2e.pull_request.one_card.test_minimax_qknorm_rope_cache import reference
    from vllm_ascend.ops.triton.msa.minimax_qknorm_rope_cache import minimax_qknorm_rope_cache

    torch.manual_seed(93)
    tokens, head_dim, q_heads, kv_heads, index_heads = 12, 128, 16, 1, 1
    width = (q_heads + 2 * kv_heads + index_heads + 1) * head_dim
    packed = torch.randn(tokens, width, dtype=torch.bfloat16)
    packed_storage = torch.empty(tokens, width + 32, dtype=packed.dtype, device="npu")
    packed_npu = packed_storage[:, :width]
    packed_npu.copy_(packed)
    weights = [torch.randn(head_dim, dtype=torch.bfloat16) * 0.1 for _ in range(4)]
    angles = torch.randn(256, 32)
    cos_sin = torch.cat((angles.cos(), angles.sin()), -1).to(torch.bfloat16)
    positions = torch.arange(tokens)
    slots = torch.tensor([0, 1, 31, 32, 63, 64, 127, 128, 129, 255, 256, -1])
    expected = reference(packed, cos_sin, positions, weights, q_heads, kv_heads, index_heads, 1e-6)
    main_width = (q_heads + 2 * kv_heads) * head_dim
    for rank in range(size):
        main_cache = torch.full((2, 3, 128, 1, 128), -7.0, dtype=packed.dtype, device="npu")
        # Include enough backing guard pages for the old, incorrect writer to
        # fail this regression without writing outside the backing allocation.
        backing = torch.full((14, 128 // size + 3, 128), -9.0, dtype=packed.dtype, device="npu")
        index_cache = backing[1:4, : 128 // size]
        q, iq = minimax_qknorm_rope_cache(
            packed_npu[:, :main_width] if separate_index else packed_npu,
            cos_sin.npu(),
            positions.npu(),
            *[w.npu() for w in weights],
            main_cache[0],
            main_cache[1],
            index_cache,
            slots.npu(),
            slots.npu(),
            tokens,
            q_heads,
            kv_heads,
            index_heads,
            1e-6,
            index_packed=packed_npu[:, main_width:] if separate_index else None,
            index_context_parallel_size=size,
            index_context_parallel_rank=rank,
        )
        gold = torch.full(backing.shape, -9.0, dtype=packed.dtype)
        for row, slot in enumerate(slots.tolist()):
            if slot >= 0 and slot % size == rank:
                gold[1 + slot // 128, slot % 128 // size] = expected[4][row].to(gold.dtype)
        torch.testing.assert_close(backing.cpu(), gold, atol=0.032, rtol=0.01)
        assert torch.equal(backing[0].cpu(), gold[0])
        assert torch.equal(backing[4:].cpu(), gold[4:])
        torch.testing.assert_close(q.cpu(), expected[0].to(q.dtype), atol=0.032, rtol=0.01)
        torch.testing.assert_close(iq.cpu(), expected[3].to(iq.dtype), atol=0.032, rtol=0.01)
        for cache, part in [(main_cache[0], expected[1]), (main_cache[1], expected[2])]:
            main_gold = torch.full(cache.shape, -7.0, dtype=cache.dtype).reshape(3 * 128, 128)
            main_gold[slots[:-1]] = part[:-1].to(cache.dtype)
            torch.testing.assert_close(cache.cpu().reshape_as(main_gold), main_gold, atol=0.032, rtol=0.01)


@pytest.mark.parametrize("size", [2, 4])
@torch.no_grad()
def test_compact_score_graph_replay_observes_new_lengths(size):
    from vllm_ascend.models.minimax_m3.ops.indexer_dcp import index_shard_scores
    from vllm_ascend.models.minimax_m3.ops.msa_m3_npu import _minimax_m3_index_score

    torch.manual_seed(41)
    full = torch.randn(8, 128, 128, dtype=torch.bfloat16, device="npu")
    table = torch.tensor([[3, 7, 1, 6], [0, 2, 4, 5]], dtype=torch.int32, device="npu")
    lengths = torch.tensor([509, 129], dtype=torch.int32, device="npu")
    query = torch.randn(2, 4, 128, dtype=torch.bfloat16, device="npu")
    cu = torch.arange(3, dtype=torch.int32, device="npu")
    mask = torch.triu(torch.ones(2048, 2048, dtype=torch.int8, device="npu"), diagonal=1)
    for rank in range(size):
        local = full[:, rank::size].contiguous()
        index_shard_scores(query, local, table, lengths, 1, size, rank)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        parts = [
            index_shard_scores(query, full[:, rank::size].contiguous(), table, lengths, 1, size, rank)
            for rank in range(size)
        ]
    for values in [[509, 129], [129, 0], [1, 257]]:
        lengths.copy_(torch.tensor(values, dtype=torch.int32, device="npu"))
        graph.replay()
        torch.npu.synchronize()
        starts = (lengths - 1) // 128
        expected = _minimax_m3_index_score(query, full, table, cu, lengths, starts, mask)
        torch.testing.assert_close(torch.stack(parts).amax(0), expected, rtol=0, atol=0)
