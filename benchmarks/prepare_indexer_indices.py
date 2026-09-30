# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare fused indexer postprocessing in an NPU graph.

Run: python benchmarks/prepare_indexer_indices.py
Times exclude compilation, graph capture and host tensor allocation. Repetition
counts adapt to a warmup measurement so slow INT32-sort baselines stay bounded.
"""

import argparse
import json
import statistics
from functools import partial

import torch
import torch_npu  # noqa: F401

from vllm_ascend.ops.triton.prepare_indexer_indices import prepare_indexer_indices
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton


def reference_indices(selected, positions, compress_ratio):
    visible = ((positions + 1) // compress_ratio).unsqueeze(-1)
    valid = (selected >= 0) & (selected < visible)
    sentinel = torch.iinfo(torch.int32).max
    selected = torch.where(valid, selected, sentinel).sort(dim=-1).values
    return torch.where(selected == sentinel, -1, selected)


def graph_latency_us(fn, value):
    for _ in range(3):
        fn(value)
    torch.npu.synchronize()
    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    start.record()
    fn(value)
    end.record()
    end.synchronize()
    estimate_ms = max(start.elapsed_time(end), 0.001)
    # Capture at most about 10 ms of work and measure about 50 ms per sample.
    batch = min(32, max(1, int(10 / estimate_ms)))
    repeats = min(20, max(1, int(50 / (batch * estimate_ms))))
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, capture_error_mode="thread_local", auto_dispatch_capture=True):
        for _ in range(batch):
            output = fn(value)
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


def benchmark(stage, value, reference, fused, **shape):
    expected, actual = reference(value), fused(value)
    if isinstance(expected, torch.Tensor):
        expected, actual = (expected,), (actual,)
    for output, ref in zip(actual, expected):
        torch.testing.assert_close(output, ref, rtol=0, atol=0)
    original_us = graph_latency_us(reference, value)
    fused_us = graph_latency_us(fused, value)
    print(
        json.dumps(
            {
                "stage": stage,
                **shape,
                "reference_us": original_us,
                "triton_us": fused_us,
                "speedup": original_us / fused_us,
            }
        ),
        flush=True,
    )


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, nargs="+", default=[1, 32, 256, 4096])
    parser.add_argument("--topk", type=int, nargs="+", default=[128, 2048])
    args = parser.parse_args()
    torch.npu.set_device(0)
    init_device_properties_triton()
    torch.manual_seed(41)
    for tokens in args.tokens:
        for topk in args.topk:
            selected = torch.randint(-1, 4096, (tokens, topk), dtype=torch.int32, device="npu")
            positions = torch.full((tokens,), 4095, dtype=torch.int64, device="npu")
            benchmark(
                "indices",
                selected,
                partial(reference_indices, positions=positions, compress_ratio=2),
                partial(prepare_indexer_indices, positions=positions, compress_ratio=2),
                tokens=tokens,
                topk=topk,
            )


if __name__ == "__main__":
    main()
