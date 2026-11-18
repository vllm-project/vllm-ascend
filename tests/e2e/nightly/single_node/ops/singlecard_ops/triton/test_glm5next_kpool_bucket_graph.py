# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.compilation.kpool_buckets import KPoolBucketProvider
from vllm_ascend.compilation.updatable_graph import ContextSource, UpdatableGraph
from vllm_ascend.ops.triton import glm5_next_lightning_indexer as indexer


@pytest.mark.parametrize("chunk_rows", [None, 2])
@torch.inference_mode()
def test_kpool_buckets_replay_preserves_million_token_history(chunk_rows, monkeypatch):
    torch.npu.set_device(0)
    torch.manual_seed(42)
    capacity = 262208
    if chunk_rows is not None:
        monkeypatch.setattr(indexer, "TRITON_SCORES_CHUNK_BYTES", chunk_rows * capacity * 4)
    rows = 6
    block = 16
    dim = 128
    pages = capacity // block
    cache = torch.randn(2 * pages, block, 1, dim, device="npu", dtype=torch.bfloat16)
    table = torch.arange(2 * pages, device="npu", dtype=torch.int32).reshape(2, pages)
    q = torch.randn(rows, 32, dim, device="npu", dtype=torch.bfloat16)
    weights = torch.randn(rows, 32, device="npu", dtype=torch.bfloat16)
    ends = torch.tensor([2, 5], device="npu", dtype=torch.int32)
    lens = torch.tensor([1024, 1024], device="npu", dtype=torch.int32)
    positions = torch.tensor([4094, 4095, 4093, 4094, 4095, 0], device="npu", dtype=torch.int64)
    args = (q, cache, weights, ends, lens, table, positions)
    kwargs = dict(index_topk=2048, index_kpool=4, max_pool_seq_len=capacity, pack_tail=True, allow_cache_packing=False)
    provider = KPoolBucketProvider("layer", capacity, 512)
    indexer.glm5_next_lightning_indexer_triton(*args, **kwargs, score_provider=provider)
    torch.npu.synchronize()
    graph = UpdatableGraph()
    stream = torch.npu.Stream()
    with torch.npu.graph(graph):
        out = indexer.glm5_next_lightning_indexer_triton(*args, **kwargs, score_provider=provider)
    for length in [1024, 2048, 2049, 8192, 262144, 262208, 1, 1024]:
        lens.copy_(torch.tensor([min(1024, length), length], dtype=torch.int32, device="npu"))
        positions.copy_(
            torch.tensor(
                [
                    max(0, min(1024, length) * 4 - 2),
                    min(1024, length) * 4 - 1,
                    max(0, length * 4 - 3),
                    max(0, length * 4 - 2),
                    length * 4 - 1,
                    0,
                ],
                device="npu",
                dtype=torch.int64,
            )
        )
        ref = indexer.glm5_next_lightning_indexer_triton(*args, **kwargs)
        # Clone before capture-pool output can be reused by another operation.
        ref = ref.clone()
        torch.npu.synchronize()
        graph.replay()
        graph.update(stream, graph.resolve_tasks(ContextSource({"layer": SimpleNamespace(max_pool_seq_len=length)})))
        torch.npu.synchronize()
        torch.testing.assert_close(out.sort(dim=-1).values, ref.sort(dim=-1).values, rtol=0, atol=0)
        assert (out[5] == -1).all()
