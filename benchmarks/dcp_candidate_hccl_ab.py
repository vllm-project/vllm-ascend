# SPDX-License-Identifier: Apache-2.0
"""Measure the production owner-candidate chain with real HCCL AllGather."""

import argparse
import hashlib
import json
import os
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import torch_npu
from dcp_four_point_ab import HcclGroup, candidate_inputs, local_reference, measure

from vllm_ascend.attention.context_parallel.sharded_indexer import distributed_topk
from vllm_ascend.ops.triton import sharded_indexer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=128)
    parser.add_argument("--world", type=int, default=8)
    parser.add_argument("--dtype", choices=["fp32", "bf16"], default="bf16")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--profile", type=Path)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=15)
    args = parser.parse_args()
    torch.set_num_threads(2)
    rank = int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(rank)
    source = Path(sharded_indexer.__file__).resolve()
    assert source == (args.source_root / "vllm_ascend/ops/triton/sharded_indexer.py").resolve()
    dist.init_process_group("hccl", timeout=timedelta(seconds=120))
    try:
        assert dist.get_world_size() == args.world
        ids, scores, _, winners = candidate_inputs(
            args.rows, args.world, score_dtype=torch.float32 if args.dtype == "fp32" else torch.bfloat16
        )
        device_ids, device_scores = ids[rank].npu(), scores[rank].npu()
        expected = local_reference(winners, rank, args.world)
        group = HcclGroup(args.world, rank)

        def run():
            return distributed_topk(device_ids, device_scores, group, 2048, 768, return_local=True)

        for _ in range(3):
            torch.testing.assert_close(run().cpu(), expected, rtol=0, atol=0)
        before = group.calls
        run()
        assert group.calls - before == 1
        graph = torch.npu.NPUGraph()
        try:
            with torch.npu.graph(graph):
                captured = run()
            graph.replay()
            torch.npu.synchronize()
            torch.testing.assert_close(captured.cpu(), expected, rtol=0, atol=0)
        finally:
            graph.reset()
        dist.barrier()
        timing = measure(run, "graph", args.iterations, args.repeats)
        result = dict(
            rank=rank,
            world=args.world,
            rows=args.rows,
            dtype=args.dtype,
            correctness="exact",
            allgather_calls=1,
            iterations=args.iterations,
            repeats=args.repeats,
            source=str(source),
            source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            timing=timing,
        )
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / f"rank{rank}.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result), flush=True)
        if args.profile:
            dist.barrier()
            if rank == 0:
                with torch_npu.profiler.profile(
                    activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
                    schedule=torch_npu.profiler.schedule(wait=0, warmup=1, active=3, repeat=1),
                    on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(args.profile)),
                    record_shapes=True,
                    experimental_config=torch_npu.profiler._ExperimentalConfig(
                        profiler_level=torch_npu.profiler.ProfilerLevel.Level1
                    ),
                ) as profiler:
                    for _ in range(4):
                        run()
                        torch.npu.synchronize()
                        profiler.step()
            else:
                for _ in range(4):
                    run()
                    torch.npu.synchronize()
            dist.barrier()
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
