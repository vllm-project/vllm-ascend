# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare DCP candidate packing with a baseline sharded_indexer.py on the same NPU.

Pass --baseline /path/to/baseline_sharded_indexer.py. With torchrun on multiple
ranks, the timed graph also gathers packed pairs and merges the owner's prefix.
"""

import argparse
import importlib.util
import json
import os
import statistics
import sys
from functools import partial
from pathlib import Path

import torch
import torch.distributed as dist
import torch_npu  # noqa: F401

from vllm_ascend.ops.triton import sharded_indexer


def load_baseline(path):
    spec = importlib.util.spec_from_file_location("baseline_sharded_indexer", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def run(module, indices, scores, world, rank):
    rows, _, topk = indices.shape
    interleave = 768
    packed = module.pack_local_candidates(
        indices, scores, world, rank, interleave, sort_candidates=True, fuse_sort=True
    )
    if not dist.is_initialized():
        return packed
    gathered = packed.new_empty((world * rows, topk, 2))
    dist.all_gather_into_tensor(gathered, packed.contiguous())
    return module.merge_packed_candidates(
        gathered, rows, world, topk, local_rank=rank, interleave=interleave, presorted=True, owner_sharded=True
    )


def capture(fn, expected):
    torch.testing.assert_close(fn(), expected, rtol=0, atol=0)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        output = fn()
    graph.replay()
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    return graph, output  # Keep the graph's output alive throughout timing.


def replay_us(graph, replays, warmups, device):
    for _ in range(warmups):
        graph.replay()
    torch.npu.synchronize()
    if dist.is_initialized():
        dist.barrier()
    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    start.record()
    for _ in range(replays):
        graph.replay()
    end.record()
    end.synchronize()
    elapsed = start.elapsed_time(end) * 1000 / replays
    if dist.is_initialized():
        slowest = torch.tensor([elapsed], device=device)
        dist.all_reduce(slowest, op=dist.ReduceOp.MAX)
        elapsed = slowest.item()
    return elapsed


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--rows", nargs="+", type=int, default=[112, 120, 128, 144, 160, 192, 256])
    parser.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16")
    args = parser.parse_args()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.npu.set_device(local_rank)
    distributed = int(os.environ.get("WORLD_SIZE", 1)) > 1
    if distributed:
        dist.init_process_group("hccl")
    world = dist.get_world_size() if distributed else 8
    rank = dist.get_rank() if distributed else 0
    baseline = load_baseline(args.baseline)
    for rows in args.rows:
        generator = torch.Generator().manual_seed(81000 + rank * 500 + rows)
        ids = torch.randint(1 << 20, (rows, 1, 2048), dtype=torch.int32, generator=generator).npu()
        scores = torch.randn(ids.shape, generator=generator).to(getattr(torch, args.dtype)).npu()
        fn = partial(run, indices=ids, scores=scores, world=world, rank=rank)
        expected = fn(baseline)
        graphs = {
            name: capture(partial(fn, mod), expected) for name, mod in (("base", baseline), ("patch", sharded_indexer))
        }
        samples = {"base": [], "patch": []}
        for trial in range(10):
            for name in ("base", "patch") if trial % 2 == 0 else ("patch", "base"):
                us = replay_us(graphs[name][0], 20 if distributed else 120, 5 if distributed else 12, ids.device)
                samples[name].append(us)
        if rank == 0:
            base, patch = samples["base"], samples["patch"]
            result = {
                "rows": rows,
                "dtype": args.dtype,
                "ranks": world if distributed else 1,
                "baseline_us": statistics.median(base),
                "candidate_us": statistics.median(patch),
                "paired_reduction_pct": 100 * statistics.median(1 - b / a for a, b in zip(base, patch)),
            }
            print(json.dumps(result), flush=True)
        del graphs
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
