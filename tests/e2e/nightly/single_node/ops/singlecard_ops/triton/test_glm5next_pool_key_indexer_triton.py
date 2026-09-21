# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.ops.triton import glm5_next_lightning_indexer as indexer

# GLM-5.3-Flash indexer parameters, including its actual top-k width.
HEAD_DIM = 128
NUM_HEADS = 32
POOL_SIZE = 4
INDEX_TOPK = 2048
CACHE_BLOCK_SIZE = 16
OUTPUT_WIDTH = INDEX_TOPK + POOL_SIZE - 1


def _assert_selection(result, query, cache, weights, ends, pool_lens, table, positions):
    """CPU reference: score each query head before applying the head weights."""
    result, query, cache, weights = (x.cpu() for x in (result, query, cache, weights))
    start = 0
    for req, end in enumerate(ends.tolist()):
        for row in range(start, end):
            pos = int(positions[row])
            count = min((pos + 1) // POOL_SIZE, int(pool_lens[req]))
            ids = torch.arange(count)
            keys = cache[table[req, ids // CACHE_BLOCK_SIZE].long(), ids % CACHE_BLOCK_SIZE, 0].float()
            per_head_scores = query[row].float() @ keys.T
            scores = (per_head_scores * weights[row].float()[:, None]).sum(0)
            selected = torch.topk(scores, min(INDEX_TOPK // POOL_SIZE, count)).indices
            history = (selected[:, None] * POOL_SIZE + torch.arange(POOL_SIZE)).flatten()
            expected = torch.full((OUTPUT_WIDTH,), -1, dtype=torch.int32)
            expected[: history.numel()] = history.to(torch.int32)
            tail = torch.arange((pos + 1) // POOL_SIZE * POOL_SIZE, pos + 1, dtype=torch.int32)
            expected[INDEX_TOPK : INDEX_TOPK + tail.numel()] = tail
            actual = result[row, 0]
            # Top-k ordering may differ for tied scores; token membership,
            # multiplicity, fixed tail columns, and every padding lane must match.
            torch.testing.assert_close(
                actual[:INDEX_TOPK].sort().values,
                expected[:INDEX_TOPK].sort().values,
                rtol=0,
                atol=0,
            )
            torch.testing.assert_close(actual[INDEX_TOPK:], expected[INDEX_TOPK:], rtol=0, atol=0)
        start = end


@pytest.mark.parametrize("max_pool_seq_len", [0, 4, 512, 2050])
@pytest.mark.parametrize("use_graph", [False, True])
@torch.inference_mode()
def test_pool_selection_real_shape_paging_and_causal_tail(max_pool_seq_len, use_graph, monkeypatch):
    generator = torch.Generator().manual_seed(19)
    # Three requests exercise non-power-of-two bucketization. Two extra rows
    # model graph padding; the caller is responsible for ignoring their output.
    ends = torch.tensor([3, 5, 8], dtype=torch.int32)
    num_tokens = 10
    pages = (max_pool_seq_len + CACHE_BLOCK_SIZE - 1) // CACHE_BLOCK_SIZE
    num_blocks = max(1, 3 * pages)
    table = torch.randperm(num_blocks, generator=generator)[: 3 * pages].reshape(3, pages).to(torch.int32)
    backing = torch.randn(num_blocks, CACHE_BLOCK_SIZE + 2, 1, HEAD_DIM, generator=generator).bfloat16()
    cache = backing.npu()[:, :CACHE_BLOCK_SIZE]
    query = torch.randn(num_tokens, NUM_HEADS, HEAD_DIM, generator=generator).bfloat16().npu()
    weights = torch.randn(num_tokens, NUM_HEADS, generator=generator).bfloat16().npu()
    pool_lens = torch.tensor([0, min(4, max_pool_seq_len), max_pool_seq_len], dtype=torch.int32)
    last_pos = max(2, max_pool_seq_len * POOL_SIZE + 2)
    positions = torch.tensor([0, 1, 2, 1, 2, last_pos - 2, last_pos - 1, last_pos, 0, 0])
    device_lens, device_positions = pool_lens.npu(), positions.npu()
    args = (query, cache, weights, ends.npu(), device_lens, table.npu(), device_positions)
    kwargs = dict(index_topk=INDEX_TOPK, index_kpool=POOL_SIZE, max_pool_seq_len=max_pool_seq_len)
    # Force several token chunks without a large scratch allocation. This also
    # checks that request lookup uses the batch-global token offset.
    monkeypatch.setattr(indexer, "TRITON_SCORES_CHUNK_BYTES", max(1, max_pool_seq_len) * 4 * 3)

    if use_graph:
        indexer.glm5_next_lightning_indexer_triton(*args, **kwargs)
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            result = indexer.glm5_next_lightning_indexer_triton(*args, **kwargs)
    for step in range(2):
        if step:
            query.copy_(torch.randn(num_tokens, NUM_HEADS, HEAD_DIM, generator=generator).bfloat16().npu())
            weights.copy_(torch.randn(num_tokens, NUM_HEADS, generator=generator).bfloat16().npu())
            pool_lens[2] //= 2
            positions[3:5] = torch.tensor([15, 16])
            device_lens.copy_(pool_lens)
            device_positions.copy_(positions)
        if use_graph:
            graph.replay()
        else:
            result = indexer.glm5_next_lightning_indexer_triton(*args, **kwargs)
        assert result.shape == (num_tokens, 1, OUTPUT_WIDTH)
        assert result.dtype == torch.int32
        _assert_selection(result, query, cache, weights, ends, pool_lens, table, positions)
        torch.testing.assert_close(cache.cpu(), backing[:, :CACHE_BLOCK_SIZE], rtol=0, atol=0)


def test_empty_query():
    result = indexer.glm5_next_lightning_indexer_triton(
        torch.empty(0, NUM_HEADS, HEAD_DIM, dtype=torch.bfloat16, device="npu"),
        torch.empty(1, CACHE_BLOCK_SIZE, 1, HEAD_DIM, dtype=torch.bfloat16, device="npu"),
        torch.empty(0, NUM_HEADS, dtype=torch.bfloat16, device="npu"),
        torch.empty(0, dtype=torch.int32, device="npu"),
        torch.empty(0, dtype=torch.int32, device="npu"),
        torch.empty(0, 0, dtype=torch.int32, device="npu"),
        torch.empty(0, dtype=torch.int64, device="npu"),
        index_topk=INDEX_TOPK,
        index_kpool=POOL_SIZE,
        max_pool_seq_len=0,
    )
    assert result.shape == (0, 1, OUTPUT_WIDTH)
    assert result.dtype == torch.int32
