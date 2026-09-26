# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NPU A/B benchmark of the full indexer, with optional profiler traces."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import torch
import torch_npu

from vllm_ascend.models.glm5next.sparse_attn_indexer_kpool import append_causal_tail
from vllm_ascend.ops.triton.glm5_next_lightning_indexer import glm5_next_lightning_indexer_triton


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=32)
    parser.add_argument("--pools", type=int, default=1024)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--graph", action="store_true")
    parser.add_argument("--trace-dir", type=Path)
    args = parser.parse_args()
    if args.rows < 1 or args.pools < 0 or args.iterations < 1:
        parser.error("rows and iterations must be positive; pools must be nonnegative")
    torch.manual_seed(41)
    pages = max(1, (args.pools + 15) // 16)
    query = torch.randn(args.rows, 32, 128, device="npu", dtype=torch.bfloat16)
    cache = torch.randn(pages, 16, 1, 128, device="npu", dtype=torch.bfloat16)
    weights = torch.randn(args.rows, 32, device="npu", dtype=torch.bfloat16)
    ends = torch.tensor([args.rows], device="npu", dtype=torch.int32)
    lengths = torch.tensor([args.pools], device="npu", dtype=torch.int32)
    table = torch.arange(pages, device="npu", dtype=torch.int32).view(1, -1)
    # Consecutive positions in one prefill request, with a cached prefix.
    positions = torch.arange(args.rows, device="npu") + args.pools * 4

    def run(compact):
        output = glm5_next_lightning_indexer_triton(
            query,
            cache,
            weights,
            ends,
            lengths,
            table,
            positions,
            index_topk=2048,
            index_kpool=4,
            max_pool_seq_len=args.pools,
            compact_indices=compact,
        )
        if not compact:
            append_causal_tail(output[:, 0], positions, 2048, 4)
            output.masked_fill_(~(torch.arange(args.rows, device="npu") < ends[-1])[:, None, None], -1)
        return output

    torch.testing.assert_close(run(False), run(True), rtol=0, atol=0)
    measurements = {}
    for label, compact in [("baseline", False), ("fused", True)]:
        for _ in range(10):
            run(compact)
        torch.npu.synchronize()
        if args.graph:
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph):
                graph_output = run(compact)
            execute = graph.replay
        else:
            execute = lambda compact=compact: run(compact)
        for _ in range(10):
            execute()
        torch.npu.synchronize()
        samples = []
        for _ in range(5):
            start = perf_counter()
            for _ in range(args.iterations):
                execute()
            torch.npu.synchronize()
            samples.append((perf_counter() - start) * 1e6 / args.iterations)
        measurements[label] = {"wall_us_per_call": samples, "median_us": sorted(samples)[2]}
        if args.trace_dir:
            destination = args.trace_dir / label
            destination.mkdir(parents=True, exist_ok=False)
            with torch_npu.profiler.profile(
                activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
                record_shapes=True,
                with_stack=True,
                on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(destination)),
            ) as profiler:
                for _ in range(5):
                    execute()
                    profiler.step()
                torch.npu.synchronize()
        if args.graph:
            # Keep capture-owned output and graph alive through timing/tracing.
            del graph_output, graph
    print(
        json.dumps(
            {"rows": args.rows, "pools": args.pools, "graph": args.graph, "exact_match": True, "timing": measurements},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
