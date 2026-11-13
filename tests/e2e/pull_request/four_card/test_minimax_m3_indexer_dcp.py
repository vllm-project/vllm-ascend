# SPDX-License-Identifier: Apache-2.0
"""Real HCCL MAX reduce-scatter and graph replay for compact M3 index keys."""

from datetime import timedelta
from functools import partial
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch_npu  # noqa: F401

from tests.e2e.conftest import wait_until_npu_memory_free


def _worker(rank, size, rendezvous, capture):
    import vllm_ascend.ops  # noqa: F401
    from vllm_ascend.models.minimax_m3.ops.indexer_dcp import index_shard_scores, reduce_scatter_index_scores
    from vllm_ascend.models.minimax_m3.ops.msa_m3_npu import _minimax_m3_index_score

    torch.npu.set_device(rank)
    dist.init_process_group("hccl", init_method=rendezvous, rank=rank, world_size=size, timeout=timedelta(seconds=180))
    group = SimpleNamespace(world_size=size, device_group=dist.group.WORLD)

    def run(local_query, gathered, local_keys, table, lengths, query_len):
        dist.all_gather(gathered, local_query)
        full_query = torch.cat(gathered, dim=1)
        scores = index_shard_scores(full_query, local_keys, table, lengths, query_len, size, rank)
        return reduce_scatter_index_scores(scores, group)

    try:
        for context, query_len in [(601, 1), (131077, 1), (601, 8)]:
            torch.manual_seed(71)
            blocks = (context + 127) // 128
            full = torch.randn(blocks + 3, 128, 128, dtype=torch.bfloat16, device="npu")
            table = torch.stack([torch.randperm(blocks + 3, device="npu")[:blocks] for _ in range(3)]).int()
            lengths = torch.tensor([context, context - 37, 0], dtype=torch.int32, device="npu")
            query = torch.randn(3 * query_len, 4, 128, dtype=torch.bfloat16, device="npu")
            local_heads = 4 // size
            local_query = query[:, rank * local_heads : (rank + 1) * local_heads].contiguous()
            local_keys = full[:, rank::size].contiguous()
            gathered = [torch.empty_like(local_query) for _ in range(size)]

            run_case = partial(run, local_query, gathered, local_keys, table, lengths, query_len)

            cu = torch.arange(4, dtype=torch.int32, device="npu") * query_len
            mask = torch.triu(torch.ones(2048, 2048, dtype=torch.int8, device="npu"), diagonal=1)
            expected = _minimax_m3_index_score(query, full, table, cu, lengths, (lengths - query_len) // 128, mask)
            expected = expected[rank * local_heads : (rank + 1) * local_heads]
            expected = expected.masked_fill(expected == torch.finfo(torch.float32).min, float("-inf"))
            actual = run_case()
            torch.testing.assert_close(actual.float(), expected, rtol=0, atol=0)
            if capture:
                run_case()
                torch.npu.synchronize()
                graph = torch.npu.NPUGraph()
                with torch.npu.graph(graph):
                    replayed = run_case()
                graph.replay()
                torch.npu.synchronize()
                torch.testing.assert_close(replayed.float(), expected, rtol=0, atol=0)
                del graph
    finally:
        dist.destroy_process_group()


@wait_until_npu_memory_free()
@pytest.mark.parametrize("size", [2, 4])
@pytest.mark.parametrize("capture", [False, True])
def test_compact_indexer_hccl_scores_and_graphs(size, capture, tmp_path):
    rendezvous = (tmp_path / "store").resolve().as_uri()
    mp.spawn(_worker, args=(size, rendezvous, capture), nprocs=size, join=True)
