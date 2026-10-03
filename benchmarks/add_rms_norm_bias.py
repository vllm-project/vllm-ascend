# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark the fused add_rms_norm_bias custom op.

Run: python benchmarks/add_rms_norm_bias.py

Times only torch.ops._C_ascend.npu_add_rms_norm_bias (the op under
optimization). Times exclude compilation, graph capture and host tensor
allocation. Repetition counts adapt to a warmup measurement so slow shapes
stay bounded.
"""

import argparse
import json
import statistics

import torch
import torch_npu  # noqa: F401

from vllm_ascend.utils import enable_custom_op

enable_custom_op()


def graph_latency_us(fn, *args):
    for _ in range(3):
        fn(*args)
    torch.npu.synchronize()
    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    start.record()
    fn(*args)
    end.record()
    end.synchronize()
    estimate_ms = max(start.elapsed_time(end), 0.001)
    # Capture at most about 10 ms of work and measure about 50 ms per sample.
    batch = min(32, max(1, int(10 / estimate_ms)))
    repeats = min(20, max(1, int(50 / (batch * estimate_ms))))
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, capture_error_mode="thread_local",
                         auto_dispatch_capture=True):
        for _ in range(batch):
            output = fn(*args)
    for _ in range(3):
        graph.replay()
    torch.npu.synchronize()
    samples = []
    for _ in range(5):
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        start.record()
        for _ in range(repeats):
            graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000 / (batch * repeats))
    # Keep the captured outputs alive until timing completes.
    del output
    return statistics.median(samples)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden-size", type=int, default=7168)
    parser.add_argument("--tokens", type=int, nargs="+",
                        default=[1, 4, 16, 64, 128, 512, 1024, 2048, 4096])
    parser.add_argument("--dtype", type=str, nargs="+",
                        default=["bfloat16", "float16"])
    args = parser.parse_args()

    dtypes = [getattr(torch, d) for d in args.dtype]
    for dtype in dtypes:
        for tokens in args.tokens:
            torch.manual_seed(7)
            x1 = torch.randn(tokens, args.hidden_size, dtype=dtype,
                             device="npu")
            x2 = torch.randn(tokens, args.hidden_size, dtype=dtype,
                             device="npu")
            gamma = torch.randn(args.hidden_size, dtype=dtype, device="npu")
            beta = torch.randn(args.hidden_size, dtype=dtype, device="npu")

            def fused(x1, x2, gamma, beta):
                return torch.ops._C_ascend.npu_add_rms_norm_bias(
                    x1, x2, gamma, beta, 1e-6)

            def reference(x1, x2, gamma, beta):
                # Unfused composition producing the same outputs via the CANN
                # built-in add+rmsnorm (returns y, rstd, residual) plus a
                # separate bias add on y.
                y, _, residual = torch_npu.npu_add_rms_norm(x1, x2, gamma, 1e-6)
                y = y + beta
                return y, residual

            us = graph_latency_us(fused, x1, x2, gamma, beta)
            ref_us = graph_latency_us(reference, x1, x2, gamma, beta)
            print(json.dumps({
                "op": "add_rms_norm_bias",
                "dtype": str(dtype).removeprefix("torch."),
                "hidden_size": args.hidden_size,
                "tokens": tokens,
                "us": round(us, 3),
                "ref_us": round(ref_us, 3),
            }))


if __name__ == "__main__":
    main()
