# SPDX-License-Identifier: Apache-2.0
"""Check index cache ownership and causal scoring on an actual NPU."""

import pytest
import torch

import vllm_ascend.ops  # noqa: F401
from vllm_ascend.models.minimax_m3.ops.indexer_dcp import scatter_index_shard
from vllm_ascend.ops.triton.msa.msa_m3_index_postprocess import index_decode_topk_fused


def assert_bf16_accuracy(actual, expected):
    """Apply both acceptance gates; untouched storage is checked separately."""
    assert actual.shape == expected.shape
    if actual.numel() == 0:
        return
    actual, expected = actual.float(), expected.float()
    error = (actual - expected).abs()
    passing = error <= (2**-6 + 2**-6 * expected.abs())
    assert passing.float().mean().item() >= 0.99
    assert error.max().item() <= 1.0


def assert_cann_scores_equal(actual, expected):
    # Compact buffers use FP16 -inf for invalid entries; CANN uses FP32 min.
    expected = expected.masked_fill(expected == torch.finfo(torch.float32).min, float("-inf"))
    torch.testing.assert_close(actual.float(), expected, rtol=0, atol=0)


def torch_index_score_reference(query, cache, table, lengths, query_len):
    """Independent CPU oracle, including causal tails and empty requests."""
    query, cache = query.float().cpu(), cache.float().cpu()
    table, lengths = table.cpu().long(), lengths.cpu().tolist()
    width = (table.shape[1] + 15) // 16 * 16
    result = torch.full((query.shape[1], query.shape[0], width), -torch.inf)
    positions = torch.arange(table.shape[1] * 128).reshape(table.shape[1], 128)
    for request, length in enumerate(lengths):
        start = request * query_len
        pages = table[request]
        keys = cache[pages.clamp_min(0)]
        dots = torch.einsum("qhd,bsd->hqbs", query[start : start + query_len], keys)
        visible = length - query_len + torch.arange(query_len) + 1
        valid = (positions[None, :, :] < visible[:, None, None]) & (pages[None, :, None] >= 0)
        maxima = dots.masked_fill(~valid[None, :, :, :], -torch.inf).amax(-1)
        result[:, start : start + query_len, : table.shape[1]] = maxima.half().float()
    return result


def assert_fp16_scores_accurate(actual, expected):
    actual = actual.float().cpu()
    assert torch.equal(torch.isneginf(actual), torch.isneginf(expected))
    assert torch.equal(torch.isposinf(actual), torch.isposinf(expected))
    assert not torch.isnan(actual).any()
    finite = torch.isfinite(expected)
    if finite.any():
        error = (actual[finite] - expected[finite]).abs()
        passing = error <= 2**-9 + 2**-9 * expected[finite].abs()
        assert passing.float().mean().item() >= 0.99
        assert error.max().item() <= 0.1


@pytest.mark.parametrize("size", [2, 4])
@pytest.mark.parametrize("padded", [False, True])
@pytest.mark.parametrize("slot_dtype", [torch.int32, torch.int64])
@torch.no_grad()
def test_index_shard_scatter_skips_padding(size, padded, slot_dtype):
    slots = torch.tensor([0, 1, 2, 3, 127, 128, 129, -1], dtype=slot_dtype, device="npu")
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
@torch.no_grad()
def test_compact_scores_preserve_cann_fp16_intermediate(size):
    from vllm_ascend.models.minimax_m3.ops.indexer_dcp import index_shard_scores
    from vllm_ascend.models.minimax_m3.ops.msa_m3_npu import _minimax_m3_index_score

    query = torch.zeros(1, 4, 128, dtype=torch.bfloat16, device="npu")
    cache = torch.zeros(1, 128, 128, dtype=query.dtype, device="npu")
    query[:, :, 0], query[:, :, 1] = 1, 3 / 64
    cache[0, 0, 0], cache[0, 0, 1] = 1, 1 / 64
    table = torch.zeros(1, 1, dtype=torch.int32, device="npu")
    lengths = torch.ones(1, dtype=torch.int32, device="npu")
    cu = torch.tensor([0, 1], dtype=torch.int32, device="npu")
    expected = _minimax_m3_index_score(query, cache, table, cu, lengths, lengths - 1, None)
    # Exact FP32 dot is 1 + 3/4096. CANN publishes its FP16 rounding in FP32.
    torch.testing.assert_close(expected[:, 0, 0], torch.full((4,), 1.0009765625, device="npu"), rtol=0, atol=0)
    parts = [
        index_shard_scores(query, cache[:, rank::size].contiguous(), table, lengths, 1, size, rank, causal=False)
        for rank in range(size)
    ]
    assert_cann_scores_equal(torch.stack(parts).amax(0), expected)


@pytest.mark.parametrize("size", [2, 4])
@pytest.mark.parametrize("metadata_dtype", [torch.int32, torch.int64])
@torch.no_grad()
def test_compact_pool_ignores_poisoned_unused_pages_and_tail_keys(size, metadata_dtype):
    from vllm_ascend.models.minimax_m3.ops.indexer_dcp import index_shard_scores
    from vllm_ascend.models.minimax_m3.ops.msa_m3_npu import _minimax_m3_index_score

    full = torch.full((10, 128, 128), float("nan"), dtype=torch.bfloat16, device="npu")
    full[3] = torch.randn_like(full[3])
    full[7, 0] = torch.randn_like(full[7, 0])
    query = torch.randn(2, 4, 128, dtype=full.dtype, device="npu")
    query[1].fill_(float("nan"))
    table = torch.tensor([[3, 7], [2, 9]], dtype=torch.int32, device="npu")
    lengths = torch.tensor([129, 0], dtype=torch.int32, device="npu")
    cu = torch.arange(3, dtype=torch.int32, device="npu")
    mask = torch.triu(torch.ones(2048, 2048, dtype=torch.int8, device="npu"), diagonal=1)
    expected = _minimax_m3_index_score(query, full, table, cu, lengths, (lengths - 1) // 128, mask)
    parts = [
        index_shard_scores(
            query, full[:, rank::size].contiguous(), table.to(metadata_dtype), lengths.to(metadata_dtype), 1, size, rank
        )
        for rank in range(size)
    ]
    assert torch.isfinite(expected[:, 0, :2]).all()
    assert_cann_scores_equal(torch.stack(parts).amax(0), expected)


@pytest.mark.parametrize("size", [2, 4])
@pytest.mark.parametrize("reference", ["torch", "cann"])
@pytest.mark.parametrize(
    "context,query_len,padded,dim_stride",
    [
        (129, 1, False, 1),
        (601, 8, True, 1),
        (601, 17, True, 2),
        (131077, 1, False, 1),
        (131077, 8, True, 1),
        (131077, 17, True, 2),
    ],
)
@torch.no_grad()
def test_compact_shards_reconstruct_reference_scores(size, reference, query_len, context, padded, dim_stride):
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
    expected = (
        _minimax_m3_index_score(queries, full, table, cu_seqlens, lengths, starts, mask)
        if reference == "cann"
        else torch_index_score_reference(queries, full, table, lengths, query_len)
    )
    if dim_stride > 1:
        storage = torch.empty(*queries.shape[:-1], 128 * dim_stride, dtype=queries.dtype, device="npu")
        view = storage[..., ::dim_stride]
        view.copy_(queries)
        queries = view
    parts = []
    for rank in range(size):
        local = full[:, rank::size, :].contiguous()
        if padded:
            storage = torch.empty(blocks + 9, 128 // size + 3, 128 * dim_stride, dtype=full.dtype, device="npu")
            view = storage[1:-1, : 128 // size, ::dim_stride]
            view.copy_(local)
            local = view
        parts.append(index_shard_scores(queries, local, table, lengths, query_len, size, rank))
    actual = torch.stack(parts).amax(0)
    if reference == "torch":
        assert_fp16_scores_accurate(actual, expected)
        return
    assert_cann_scores_equal(actual, expected)
    args = (lengths - query_len, query_len, 16, table.shape[1])
    actual_topk = index_decode_topk_fused(actual, *args, init_blocks=1, local_blocks=2, sort_indices=True)
    expected_topk = index_decode_topk_fused(expected, *args, init_blocks=1, local_blocks=2, sort_indices=True)
    for result, gold in zip(actual_topk, expected_topk):
        torch.testing.assert_close(result, gold, rtol=0, atol=0)


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
        untouched = torch.ones_like(gold, dtype=torch.bool)
        for slot in slots.tolist():
            if slot >= 0 and slot % size == rank:
                untouched[1 + slot // 128, slot % 128 // size] = False
        assert_bf16_accuracy(backing.cpu()[~untouched], gold[~untouched])
        assert torch.equal(backing.cpu()[untouched], gold[untouched])
        assert torch.equal(backing[0].cpu(), gold[0])
        assert torch.equal(backing[4:].cpu(), gold[4:])
        assert_bf16_accuracy(q.cpu(), expected[0].to(q.dtype))
        assert_bf16_accuracy(iq.cpu(), expected[3].to(iq.dtype))
        for cache, part in [(main_cache[0], expected[1]), (main_cache[1], expected[2])]:
            main_gold = torch.full(cache.shape, -7.0, dtype=cache.dtype).reshape(3 * 128, 128)
            main_gold[slots[:-1]] = part[:-1].to(cache.dtype)
            main_actual = cache.cpu().reshape_as(main_gold)
            written = torch.zeros(3 * 128, dtype=torch.bool)
            written[slots[:-1]] = True
            assert_bf16_accuracy(main_actual[written], main_gold[written])
            torch.testing.assert_close(main_actual[~written], main_gold[~written], rtol=0, atol=0)


@pytest.mark.parametrize("size", [2, 4])
@torch.no_grad()
def test_compact_score_graph_replay_observes_new_lengths(size):
    from vllm_ascend.models.minimax_m3.ops.indexer_dcp import index_shard_scores

    torch.manual_seed(41)
    full = torch.randn(8, 128, 128, dtype=torch.bfloat16, device="npu")
    table = torch.tensor([[3, 7, 1, 6], [0, 2, 4, 5]], dtype=torch.int32, device="npu")
    lengths = torch.tensor([509, 129], dtype=torch.int32, device="npu")
    query = torch.randn(2, 4, 128, dtype=torch.bfloat16, device="npu")
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
        expected = torch_index_score_reference(query, full, table, lengths, 1)
        assert_fp16_scores_accurate(torch.stack(parts).amax(0), expected)
