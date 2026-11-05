# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Compare DFlash2 edge scoring on one NPU; reports no model-level speedup."""

import argparse
import statistics
from functools import partial

import torch
from vllm.triton_utils import HAS_TRITON

from vllm_ascend.models.qwen3_dflash2 import _score_edges, _score_edges_reference


def measure(fn, iterations, use_graph):
    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(stream):
        for _ in range(5):
            fn()
    torch.npu.current_stream().wait_stream(stream)
    if use_graph:
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph, stream=stream):
            output = fn()
        # Keep captured output storage alive until timing completes.
        fn = graph.replay
    timings = []
    for _ in range(5):
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            fn()
        end.record()
        end.synchronize()
        timings.append(start.elapsed_time(end) * 1000 / iterations)
    if use_graph:
        del output
    return statistics.median(timings)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 4, 16, 64, 256])
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--vocab", type=int, default=151936)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--graph", action="store_true")
    args = parser.parse_args()
    if min(*args.batches, args.steps, args.top_k, args.rank, args.vocab, args.iterations) <= 0:
        parser.error("all dimensions and iteration counts must be positive")
    if args.top_k > 32 or args.rank > 256:
        parser.error("choose top-k <= 32 and rank <= 256 to benchmark the fused path")
    if not HAS_TRITON:
        parser.error("Triton-Ascend must be available to benchmark the fused path")
    torch.manual_seed(0)
    print(f"torch={torch.__version__}, device={torch.npu.get_device_name()}, graph={args.graph}")
    print("batch,reference_us,fused_us,ratio")
    for batch in args.batches:
        predecessor = torch.randn(args.vocab, args.rank, device="npu", dtype=torch.bfloat16) * 0.1
        successor = torch.randn_like(predecessor) * 0.1
        ids = torch.randint(args.vocab, (batch, args.steps, args.top_k), device="npu")
        unary = torch.randn(batch, args.steps, args.top_k, device="npu", dtype=torch.float32) * 0.1
        hidden = torch.randn(batch, args.steps, args.rank, device="npu", dtype=torch.bfloat16) * 0.1
        anchor = torch.randint(args.vocab, (batch,), device="npu")
        inputs = (predecessor, successor, ids, unary, hidden, anchor, args.top_k)
        reference = partial(_score_edges_reference, *inputs)
        fused = partial(_score_edges, *inputs)
        torch.testing.assert_close(fused(), reference(), rtol=2e-4, atol=2e-5)
        ref_us = measure(reference, args.iterations, args.graph)
        fused_us = measure(fused, args.iterations, args.graph)
        print(f"{batch},{ref_us:.3f},{fused_us:.3f},{ref_us / fused_us:.3f}")


if __name__ == "__main__":
    main()
