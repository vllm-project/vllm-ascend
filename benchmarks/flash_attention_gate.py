# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Output-gate device latency and eager host overhead, measured separately.

Run: python benchmarks/flash_attention_gate.py
Graph timing uses NPU events around 100 dependent calls captured in one graph;
the replay/event overhead is amortized. Eager timing includes Python dispatch.
Warmup, compilation, capture and allocation are excluded from both timings.
"""

import json
import statistics
import time

import torch
import torch_npu  # noqa: F401

from vllm_ascend.ops.triton.flash_attention_output import flash_attention_gate

CALLS_PER_SAMPLE = 100
PAIRED_TRIALS = 8
SHAPES = [
    (1, 1536),
    (4, 1536),
    (8, 1536),
    (32, 1536),
    (64, 1536),
    (128, 1536),
    (1024, 1536),
    (4096, 1536),
    (8, 768),
    (8, 3072),
    (8, 12288),
]


def native_gate(projected, gate):
    projected.mul_(torch.sigmoid(gate))


def sample_eager(func):
    torch.npu.synchronize()
    start = time.perf_counter()
    for _ in range(CALLS_PER_SAMPLE):
        func()
    torch.npu.synchronize()
    return (time.perf_counter() - start) * 1e6 / CALLS_PER_SAMPLE


def sample_graph(graph):
    start, end = (torch.npu.Event(enable_timing=True) for _ in range(2))
    start.record()
    graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000 / CALLS_PER_SAMPLE


@torch.inference_mode()
def compare(shape, methods):
    # BF16 sigmoid(10) rounds to one, keeping repeated in-place calls stable.
    gate = torch.full(shape, 10.0, device="npu", dtype=torch.bfloat16)
    inputs = {name: torch.ones_like(gate) for name in methods}
    calls, graphs = {}, {}
    samples = {name: {"graph_device_us": [], "eager_wall_us": []} for name in methods}
    for name, method in methods.items():
        calls[name] = lambda method=method, value=inputs[name]: method(value, gate)
        for _ in range(12):
            calls[name]()
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            for _ in range(CALLS_PER_SAMPLE):
                calls[name]()
        graphs[name] = graph
        for _ in range(3):
            graph.replay()
        torch.npu.synchronize()
    for trial in range(PAIRED_TRIALS):
        order = list(methods) if trial % 2 == 0 else list(methods)[::-1]
        for name in order:
            samples[name]["graph_device_us"].append(sample_graph(graphs[name]))
        for name in order:
            samples[name]["eager_wall_us"].append(sample_eager(calls[name]))
    return {
        "shape": shape,
        "calls_per_graph": CALLS_PER_SAMPLE,
        "median_us": {
            name: {key: statistics.median(values) for key, values in data.items()} for name, data in samples.items()
        },
        "samples_us": samples,
    }


if __name__ == "__main__":
    torch.set_num_threads(1)
    methods = {"native": native_gate, "fused": flash_attention_gate}
    print(json.dumps([compare(shape, methods) for shape in SHAPES], indent=2))
