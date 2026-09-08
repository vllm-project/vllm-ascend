# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the GLM-Next KPool lightning indexer op."""

import torch

from vllm_ascend.ops.glm5_next_lightning_indexer import (
    glm5_next_lightning_indexer,
)


def test_lightning_indexer_selects_pools_expands_tokens_and_appends_tail():
    index_topk = 4
    index_kpool = 2
    logical_keys = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [-1.0, 2.0],
            [3.0, -1.0],
        ],
        dtype=torch.bfloat16,
    )
    indexer_cache = torch.zeros((3, 2, 1, 2), dtype=torch.bfloat16)
    block_table = torch.tensor([[2, 0, 1]], dtype=torch.int32)
    for pool_id, logical_key in enumerate(logical_keys):
        logical_page, offset = divmod(pool_id, 2)
        physical_page = int(block_table[0, logical_page])
        indexer_cache[physical_page, offset, 0] = logical_key
    cache_before = indexer_cache.clone()

    query = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[1.0, 1.0], [2.0, 0.0]],
        ],
        dtype=torch.bfloat16,
    )
    weights = torch.tensor(
        [[1.0, 2.0], [1.0, -1.0]],
        dtype=torch.bfloat16,
    )
    result = glm5_next_lightning_indexer(
        query,
        indexer_cache,
        weights,
        cum_query_lens=torch.tensor([2], dtype=torch.int32),
        indexer_seq_lens=torch.tensor([5], dtype=torch.int32),
        indexer_block_table=block_table,
        positions=torch.tensor([4, 8], dtype=torch.int64),
        index_topk=index_topk,
        index_kpool=index_kpool,
        max_pool_seq_len=5,
    )

    assert result.dtype == torch.int32
    assert result.shape == (2, 1, index_topk + index_kpool - 1)
    assert sorted(result[0, 0, :4].tolist()) == [0, 1, 2, 3]
    assert result[0, 0, 4].item() == 4
    assert set(result[1, 0, :4].tolist()) == {2, 3, 6, 7}
    assert result[1, 0, 4].item() == 8
    torch.testing.assert_close(indexer_cache, cache_before)


def test_lightning_indexer_reads_block_major_pooled_cache_by_block_table():
    num_blocks = 4
    cache_block_size = 2
    head_dim = 2
    block_stride_bytes = 32
    page_offset_bytes = 8
    raw_cache = torch.zeros(
        num_blocks * block_stride_bytes,
        dtype=torch.int8,
    )
    indexer_cache = torch.as_strided(
        raw_cache.view(torch.bfloat16),
        size=(num_blocks, cache_block_size, 1, head_dim),
        stride=(block_stride_bytes // 2, head_dim, head_dim, 1),
        storage_offset=page_offset_bytes // 2,
    )
    block_table = torch.tensor(
        [[3, 1], [2, 0]],
        dtype=torch.int32,
    )
    request_keys = (
        ((1.0, 0.0), (4.0, 0.0), (2.0, 0.0)),
        ((0.0, 1.0), (0.0, 2.0), (0.0, 5.0)),
    )
    for request_id, logical_keys in enumerate(request_keys):
        for pool_id, key in enumerate(logical_keys):
            logical_page, offset = divmod(pool_id, cache_block_size)
            physical_page = int(block_table[request_id, logical_page])
            indexer_cache[physical_page, offset, 0] = torch.tensor(key)
    raw_before = raw_cache.clone()

    result = glm5_next_lightning_indexer(
        query=torch.tensor(
            [[[1.0, 0.0]], [[0.0, 1.0]]],
            dtype=torch.bfloat16,
        ),
        indexer_cache=indexer_cache,
        weights=torch.ones((2, 1), dtype=torch.bfloat16),
        cum_query_lens=torch.tensor([1, 2], dtype=torch.int32),
        indexer_seq_lens=torch.tensor([3, 3], dtype=torch.int32),
        indexer_block_table=block_table,
        positions=torch.tensor([5, 5], dtype=torch.int64),
        index_topk=2,
        index_kpool=2,
        max_pool_seq_len=3,
    )

    assert result[:, 0].tolist() == [[2, 3, -1], [4, 5, -1]]
    assert indexer_cache.stride(0) * indexer_cache.element_size() == 32
    torch.testing.assert_close(raw_cache, raw_before)
    assert raw_cache[page_offset_bytes + 8].item() == 0
