# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.models.glm5next.sparse_attn_indexer_kpool import append_causal_tail
from vllm_ascend.ops.triton import glm5_next_lightning_indexer as indexer


def reference(pool_ids, scores, positions, ends, index_topk, pool_size):
    """Independent row-wise contract, including arbitrary ranked pool IDs."""
    width = index_topk + pool_size - 1
    output = torch.full((positions.numel(), 1, width), -1, dtype=torch.int32)
    for row in range(int(ends[-1])):
        for slot, (pool, score) in enumerate(zip(pool_ids[row].tolist(), scores[row].tolist())):
            if score > torch.finfo(torch.float32).min:
                output[row, 0, slot * pool_size : (slot + 1) * pool_size] = torch.arange(
                    pool * pool_size, (pool + 1) * pool_size, dtype=torch.int32
                )
        pos = int(positions[row])
        tail_start = (pos + 1) // pool_size * pool_size
        tail_col = min(tail_start, index_topk)
        for offset in range(pool_size - 1):
            output[row, 0, tail_col + offset] = tail_start + offset if tail_start + offset <= pos else -1
    return output


@pytest.mark.parametrize("pool_size", [1, 4, 16])
@pytest.mark.parametrize("index_topk", [32, 2048])
@pytest.mark.parametrize("available_pools", [0, 2, 600])
@pytest.mark.parametrize("use_graph", [False, True])
@torch.inference_mode()
def test_fused_postprocess(pool_size, index_topk, available_pools, use_graph):
    topk = min(index_topk // pool_size, available_pools)
    # Short requests, exact pool boundaries, long requests, and padded rows.
    positions = torch.tensor([0, 1, pool_size - 1, pool_size, index_topk - 1, index_topk + 2, 999, 999])
    ends = torch.tensor([2, 6], dtype=torch.int32)
    ids = torch.arange(topk).flip(0).repeat(8, 1)
    scores = torch.ones(8, topk)
    if topk:
        scores[:, -1] = torch.finfo(torch.float32).min
        if topk > 1:
            scores[:, -2] = float("-inf")
    device_ids, device_scores, device_positions, device_ends = (x.npu() for x in (ids, scores, positions, ends))
    output = torch.empty(8, 1, index_topk + pool_size - 1, dtype=torch.int32, device="npu")

    def run():
        # Chunk-relative top-k buffers, global positions/output, nonzero offset.
        for start, stop in [(0, 3), (3, 8)]:
            indexer._write_compact_indices(
                device_ids[start:stop],
                device_scores[start:stop],
                device_positions,
                device_ends,
                output,
                start,
                stop - start,
                topk,
                index_topk,
                pool_size,
            )

    run()
    if use_graph:
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            run()
    for step in range(2):
        if step:
            positions[:6] += 1
            ends[-1] = 5
            device_positions.copy_(positions)
            device_ends.copy_(ends)
        if use_graph:
            graph.replay()
        else:
            run()
        torch.testing.assert_close(output.cpu(), reference(ids, scores, positions, ends, index_topk, pool_size))


@pytest.mark.parametrize("max_pools", [0, 4, 520])
@torch.inference_mode()
def test_full_indexer_matches_legacy_postprocess(max_pools, monkeypatch):
    torch.manual_seed(41)
    rows, pool_size, index_topk, block_size = 8, 4, 2048, 16
    pages = max(1, (max_pools + block_size - 1) // block_size)
    query = torch.randn(rows, 32, 128, device="npu", dtype=torch.bfloat16)
    cache = torch.randn(pages, block_size, 1, 128, device="npu", dtype=torch.bfloat16)
    weights = torch.randn(rows, 32, device="npu", dtype=torch.bfloat16)
    ends = torch.tensor([2, 6], dtype=torch.int32, device="npu")
    lengths = torch.tensor([0, max_pools], dtype=torch.int32, device="npu")
    table = torch.arange(pages, dtype=torch.int32, device="npu").repeat(2, 1)
    positions = torch.tensor([0, 2, 3, 4, 7, max_pools * pool_size + 2, 0, 0], device="npu")
    args = query, cache, weights, ends, lengths, table, positions
    kwargs = dict(index_topk=index_topk, index_kpool=pool_size, max_pool_seq_len=max_pools)
    monkeypatch.setattr(indexer, "TRITON_SCORES_CHUNK_BYTES", max(1, max_pools) * 4 * 3)
    expected = indexer.glm5_next_lightning_indexer_triton(*args, **kwargs)
    append_causal_tail(expected[:, 0], positions, index_topk, pool_size)
    expected.masked_fill_(~(torch.arange(rows, device="npu") < ends[-1])[:, None, None], -1)
    actual = indexer.glm5_next_lightning_indexer_triton(*args, **kwargs, compact_indices=True)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
