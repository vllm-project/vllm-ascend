# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare indexer query quantization latency inside an NPU graph.

Run: python benchmarks/quantize_indexer_query.py
Times exclude compilation, graph capture and host tensor allocation.
"""

import argparse
import json
import statistics

import torch
import torch_npu  # noqa: F401

from vllm_ascend.ops.triton.quantize_indexer_query import quantize_indexer_query
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton


def reference(query):
    scale = (query.float().abs().amax(-1) / 127.0).half().clamp_min_(2.0**-24)
    quantized = (query.float() / scale.float().unsqueeze(-1)).round().clamp(-127, 127).to(torch.int8)
    return quantized, scale


def graph_latency_us(fn, query, batch=32, repeats=20):
    for _ in range(3):
        fn(query)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, capture_error_mode="thread_local", auto_dispatch_capture=True):
        for _ in range(batch):
            output = fn(query)
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
    assert output[0].shape == query.shape
    return statistics.median(samples)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, nargs="+", default=[1, 32, 256, 4096])
    parser.add_argument("--heads", type=int, nargs="+", default=[32, 64])
    args = parser.parse_args()
    torch.npu.set_device(0)
    init_device_properties_triton()
    torch.manual_seed(41)
    for tokens in args.tokens:
        for heads in args.heads:
            query = torch.randn(tokens, heads, 128, dtype=torch.bfloat16, device="npu")
            for actual, expected in zip(quantize_indexer_query(query), reference(query)):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            original_us = graph_latency_us(reference, query)
            fused_us = graph_latency_us(quantize_indexer_query, query)
            print(
                json.dumps(
                    {
                        "tokens": tokens,
                        "heads": heads,
                        "reference_us": original_us,
                        "triton_us": fused_us,
                        "speedup": original_us / fused_us,
                    }
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
