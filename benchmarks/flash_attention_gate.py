# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare MLA output gating against the materialized sigmoid path on NPU.

Run: python benchmarks/flash_attention_gate.py
Times include Python dispatch and NPU execution, but exclude warmup, graph
capture, and tensor allocation. Each sample batches calls before synchronizing.
"""

import json
import statistics
import time

import torch
import torch_npu  # noqa: F401

from vllm_ascend.ops.triton.flash_attention_output import flash_attention_gate


def old_gate(projected, gate):
    projected.mul_(torch.sigmoid(gate))


def sample(func, iterations):
    torch.npu.synchronize()
    start = time.perf_counter()
    for _ in range(iterations):
        func()
    torch.npu.synchronize()
    return (time.perf_counter() - start) * 1e6 / iterations


def compare(shape, graph_mode):
    gate = torch.randn(shape, device="npu", dtype=torch.bfloat16)
    old_input = torch.ones_like(gate)
    fused_input = torch.ones_like(gate)
    old = lambda: old_gate(old_input, gate)
    fused = lambda: flash_attention_gate(fused_input, gate)

    for _ in range(12):
        old()
        fused()
    torch.npu.synchronize()

    if graph_mode:
        old_graph = torch.npu.NPUGraph()
        fused_graph = torch.npu.NPUGraph()
        with torch.npu.graph(old_graph):
            old()
        with torch.npu.graph(fused_graph):
            fused()
        old = old_graph.replay
        fused = fused_graph.replay
        iterations = 200
    else:
        iterations = 100

    old_samples = []
    fused_samples = []
    for trial in range(7):
        first, second = (
            ((old, old_samples), (fused, fused_samples))
            if trial % 2 == 0
            else ((fused, fused_samples), (old, old_samples))
        )
        for func, samples in (first, second):
            old_input.fill_(1)
            fused_input.fill_(1)
            torch.npu.synchronize()
            samples.append(sample(func, iterations))
    baseline = statistics.median(old_samples)
    optimized = statistics.median(fused_samples)
    return {
        "tokens": shape[0],
        "hidden": shape[1],
        "mode": "graph_replay" if graph_mode else "eager",
        "old_us": round(baseline, 3),
        "fused_us": round(optimized, 3),
        "delta_percent": round((baseline - optimized) / baseline * 100, 2),
        "old_samples_us": [round(x, 3) for x in old_samples],
        "fused_samples_us": [round(x, 3) for x in fused_samples],
    }


@torch.inference_mode()
def main():
    cases = [(1, 1536), (8, 1536), (32, 1536), (64, 1536), (128, 1536), (1, 12288), (8, 12288)]
    results = [compare(shape, graph_mode=False) for shape in cases]
    results.extend(compare(shape, graph_mode=True) for shape in cases[:4])
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
