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
@pytest.mark.parametrize("query_len", [1, 8])
@pytest.mark.parametrize("context", [129, 601, 131077])
@torch.no_grad()
def test_cann_shards_reconstruct_exact_scores(size, query_len, context):
    from vllm_ascend.models.minimax_m3.ops.indexer_dcp import index_shard_scores_cann
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
    parts = [
        index_shard_scores_cann(
            queries, full[:, rank::size, :].contiguous(), table, cu_seqlens, lengths, starts, mask, size, rank
        )
        for rank in range(size)
    ]
    torch.testing.assert_close(torch.stack(parts).amax(0), expected, rtol=0, atol=0)
