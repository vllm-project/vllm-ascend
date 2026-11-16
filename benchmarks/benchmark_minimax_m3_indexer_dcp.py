# SPDX-License-Identifier: Apache-2.0
"""Compare baseline, expanded DCP and compact DCP on an idle A3 TP group.

Example (four free NPUs; matching vLLM/CANN dependencies required)::

    torchrun --standalone --nproc-per-node=4 \
        benchmarks/benchmark_minimax_m3_indexer_dcp.py --graph

This measures one indexer layer's score stage, including query/score collectives. It does
not measure model serving, PD transfer or weight-memory savings. Temporary
peak bytes exclude the already allocated input and reference caches.
"""

import argparse
import json
import os
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch_npu  # noqa: F401

import vllm_ascend.ops  # noqa: F401
from vllm_ascend.models.minimax_m3.ops.indexer_dcp import index_shard_scores, reduce_scatter_index_scores
from vllm_ascend.models.minimax_m3.ops.msa_m3_npu import _minimax_m3_index_score


def measure(run, iterations, graph_mode):
    for _ in range(5):
        run()
    torch.npu.synchronize()
    capture_before = torch.npu.memory_allocated()
    torch.npu.reset_peak_memory_stats()
    graph = None
    if graph_mode:
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            output = run()
        replay = graph.replay
    else:
        replay = run
    capture_live = torch.npu.memory_allocated() - capture_before
    capture_peak = torch.npu.max_memory_allocated() - capture_before if graph_mode else 0
    # Allocator-reserved deltas can be negative when old cached pools are freed;
    # report allocated peaks instead of interpreting cache eviction as savings.
    torch.npu.synchronize()
    dist.barrier()
    before = torch.npu.memory_allocated()
    torch.npu.reset_peak_memory_stats()
    start, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        replay()
    end.record()
    end.synchronize()
    values = torch.tensor(
        [start.elapsed_time(end) / iterations, torch.npu.max_memory_allocated() - before, capture_peak, capture_live],
        device="npu",
        dtype=torch.float32,
    )
    dist.all_reduce(values, op=dist.ReduceOp.MAX)
    result = {
        "ms": values[0].item(),
        "temporary_peak_bytes": int(values[1].item()),
        "capture_peak_bytes": int(values[2].item()),
        "capture_live_bytes": int(values[3].item()),
    }
    if graph is not None:
        del output, graph
    return result


def run_case(batch, args, rank, size):
    # Every rank starts with identical reference keys, pages and all-head Q.
    # Keep the same logical cache pool and page table in all three variants.
    torch.manual_seed(109)
    blocks = (args.context + 127) // 128
    if args.cache_pages < blocks:
        raise ValueError("--cache-pages must cover --context")
    full_cache = torch.randn(args.cache_pages, 128, 128, dtype=torch.bfloat16, device="npu")
    local_cache = full_cache[:, rank::size].contiguous()
    table = torch.stack([torch.randperm(args.cache_pages, device="npu")[:blocks] for _ in range(batch)]).int()
    lengths = torch.full((batch,), args.context, dtype=torch.int32, device="npu")
    query = torch.randn(batch, 4, 128, dtype=torch.bfloat16, device="npu")
    local_heads = 4 // size
    local_query = query[:, rank * local_heads : (rank + 1) * local_heads].contiguous()
    gathered = [torch.empty_like(local_query) for _ in range(size)]
    cu = torch.arange(batch + 1, device="npu", dtype=torch.int32)
    start_loc = (lengths - 1) // 128
    shifted_lengths = (lengths - rank).clamp_min(0)
    mask = torch.triu(torch.ones(2048, 2048, dtype=torch.int8, device="npu"), diagonal=1)
    group = SimpleNamespace(world_size=size, device_group=dist.group.WORLD)

    def baseline():
        return _minimax_m3_index_score(local_query, full_cache, table, cu, lengths, start_loc, mask)

    def gather_query():
        dist.all_gather(gathered, local_query)
        return torch.cat(gathered, dim=1)

    def expanded():
        full_query = gather_query()
        keys = local_cache.repeat_interleave(size, dim=1)
        scores = _minimax_m3_index_score(full_query, keys, table, cu, shifted_lengths, start_loc, mask)
        dist.all_reduce(scores, op=dist.ReduceOp.MAX)
        return scores[rank * local_heads : (rank + 1) * local_heads].contiguous()

    def compact():
        scores = index_shard_scores(gather_query(), local_cache, table, lengths, 1, size, rank)
        return reduce_scatter_index_scores(scores, group)

    expected = baseline()
    # Never emit a speedup for a kernel that fails the numerical gate.
    torch.testing.assert_close(expanded(), expected, rtol=0, atol=0)
    expected_compact = expected.masked_fill(expected == torch.finfo(torch.float32).min, float("-inf"))
    torch.testing.assert_close(compact().float(), expected_compact, rtol=0, atol=0)
    del expected_compact
    del expected
    results = {
        name: measure(run, args.iterations, args.graph)
        for name, run in [("baseline", baseline), ("expanded_dcp", expanded), ("compact_dcp", compact)]
    }
    results["compact_vs_expanded"] = results["expanded_dcp"]["ms"] / results["compact_dcp"]["ms"]
    results["compact_vs_baseline"] = results["baseline"]["ms"] / results["compact_dcp"]["ms"]
    if rank == 0:
        print(
            json.dumps(
                {
                    "batch_per_tp_group": batch,
                    "context": args.context,
                    "tp": size,
                    "cache_pages": args.cache_pages,
                    "graph": args.graph,
                    "results": results,
                }
            ),
            flush=True,
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batches", nargs="+", type=int, default=[1, 8, 16, 32])
    parser.add_argument("--context", type=int, default=131077)
    parser.add_argument("--cache-pages", type=int, default=5221)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--graph", action="store_true")
    args = parser.parse_args()
    torch.npu.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("hccl")
    try:
        rank, size = dist.get_rank(), dist.get_world_size()
        if size not in (2, 4):
            raise ValueError("Launch with TP2 or TP4")
        for batch in args.batches:
            run_case(batch, args, rank, size)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
