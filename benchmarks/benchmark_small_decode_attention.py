# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Graph-replay microbenchmark; end-to-end gains require a model benchmark."""

import argparse
import json
import math
import statistics
from functools import partial
from pathlib import Path

import torch
import torch_npu

from vllm_ascend.ops.triton.small_decode_attention import small_decode_attention
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton

_CALLS_PER_GRAPH = 50


def capture(function):
    stream = torch.npu.Stream()
    with torch.npu.stream(stream):
        for _ in range(5):
            function()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, stream=stream):
        for _ in range(_CALLS_PER_GRAPH):
            function()
    for _ in range(5):
        graph.replay()
    torch.npu.synchronize()
    return graph


def measure(graph):
    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    start.record()
    graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000 / _CALLS_PER_GRAPH


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=20)
    args = parser.parse_args()
    if args.samples < 5:
        parser.error("Use at least five timing samples")
    torch.npu.set_device(0)
    init_device_properties_triton()
    torch.manual_seed(42)
    rows = []
    for length in (128, 160, 512, 1024):
        num_blocks = math.ceil(length / 128)
        # Match MRV2's combined K/V pages, including their non-dense stride.
        backing = torch.randn(num_blocks, 2, 128, 8, 128, device="npu", dtype=torch.bfloat16)
        key, value = backing[:, 0], backing[:, 1]
        query = torch.randn(1, 16, 128, device="npu", dtype=torch.bfloat16)
        table = torch.randperm(num_blocks, dtype=torch.int32, device="npu").reshape(1, -1)
        lengths = torch.tensor([length], dtype=torch.int32, device="npu")
        output = torch.empty_like(query)
        scale = 1 / math.sqrt(128)

        baseline = partial(
            torch_npu.npu_fused_infer_attention_score,
            query,
            key.view(num_blocks, 128, -1),
            value.view(num_blocks, 128, -1),
            block_table=table,
            block_size=128,
            actual_seq_lengths=[1],
            actual_seq_lengths_kv=[length],
            num_heads=16,
            num_key_value_heads=8,
            input_layout="TND",
            scale=scale,
            sparse_mode=0,
        )
        optimized = partial(small_decode_attention, query, key, value, table, lengths, scale, output)

        expected = baseline()[0]
        optimized()
        torch.testing.assert_close(output, expected, atol=0.004, rtol=0.01)
        graphs = {"baseline": capture(baseline), "optimized": capture(optimized)}
        samples = {name: [] for name in graphs}
        # Alternate order to reduce systematic warmup/frequency bias.
        for repeat in range(args.samples):
            order = ("baseline", "optimized") if repeat % 2 == 0 else ("optimized", "baseline")
            for name in order:
                samples[name].append(measure(graphs[name]))
        row = {
            "length": length,
            "key_stride": list(key.stride()),
            "calls_per_graph": _CALLS_PER_GRAPH,
            "samples_us": samples,
            "median_us": {name: statistics.median(values) for name, values in samples.items()},
        }
        rows.append(row)
        print(json.dumps(row), flush=True)
        args.output.write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
