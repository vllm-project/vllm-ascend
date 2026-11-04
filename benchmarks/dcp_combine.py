# SPDX-License-Identifier: Apache-2.0
"""Validate and measure BF16 DCP combine against the scalar fallback on A3.

Run with an idle NPU: python benchmarks/dcp_combine.py --output combine.json
The full correctness matrix must pass before any performance measurements.
"""

import argparse
import json
import statistics
from pathlib import Path
from unittest.mock import patch

import torch
import torch_npu  # noqa: F401

from vllm_ascend.device.device_config import AscendDeviceType, get_ascend_device_type
from vllm_ascend.ops.triton.dcp import dcp_a2a


def make_inputs(ranks, heads, tokens):
    torch.manual_seed(20261030)
    history = torch.randn(ranks, tokens, heads, 512, dtype=torch.bfloat16)
    history_lse = torch.randn(ranks, tokens, heads, 1) * 30
    local = torch.randn(tokens, heads, 512, dtype=torch.bfloat16)
    local_lse = torch.randn(tokens, heads, 1) * 30
    history[:, 0, 0] = torch.nan
    history_lse[:, 0, 0] = -torch.inf
    local[0, 0] = torch.nan
    local_lse[0, 0] = torch.nan
    history[0, 1, 1] = torch.nan
    history_lse[0, 1, 1] = torch.nan
    values = torch.cat((history.double(), local.double()[None]))
    logs = torch.cat((history_lse.double(), local_lse.double()[None]))
    valid = torch.isfinite(logs)
    maximum = torch.where(valid, logs, -torch.inf).amax(0)
    maximum = torch.where(torch.isfinite(maximum), maximum, 0.0)
    weights = torch.where(valid, (logs - maximum).exp(), 0.0)
    denominator = weights.sum(0)
    expected = (torch.where(valid, values, 0.0) * weights).sum(0)
    expected /= torch.where(denominator > 0, denominator, 1.0)
    recv = dcp_a2a.pack_dcp_output_lse(
        history.transpose(0, 1).reshape(tokens, ranks * heads, 512).npu(),
        history_lse.transpose(0, 1).reshape(tokens, ranks * heads, 1).npu(),
        ranks,
        1,
    )
    return (recv, local.npu(), local_lse.npu()), expected


def combine(data, scalar=False):
    # The private-base A3 dispatcher selects the unchanged scalar fallback.
    # Override only the selector for the paired comparison on the same NPU.
    with patch.object(
        dcp_a2a,
        "get_ascend_device_type",
        return_value=AscendDeviceType.A2 if scalar else AscendDeviceType.A3,
    ):
        return dcp_a2a.fused_dcp_lse_combine(data[0], 512, 1, local_output=data[1], local_lse=data[2])


def validate(data, expected):
    for scalar in (True, False):
        actual = combine(data, scalar)
        torch.testing.assert_close(actual.cpu().double(), expected, rtol=0.016, atol=0.001)
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            replayed = combine(data, scalar)
        original = data[1].clone()
        for scale in (0.75, 1.25):
            data[1].mul_(scale)
            graph.replay()
            torch.testing.assert_close(replayed, combine(data, scalar), rtol=0, atol=0)
        data[1].copy_(original)


def measure(data):
    graphs, outputs = {}, {}
    unroll, replays, rounds = 32, 5, 5
    for name, scalar in (("baseline", True), ("candidate", False)):
        for _ in range(5):
            combine(data, scalar)
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            outputs[name] = [combine(data, scalar) for _ in range(unroll)]
        graphs[name] = graph
        for _ in range(5):
            graph.replay()
        torch.npu.synchronize()
    samples = {name: [] for name in graphs}
    for iteration in range(rounds):
        order = list(graphs) if iteration % 2 == 0 else list(reversed(graphs))
        for name in order:
            start, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
            start.record()
            for _ in range(replays):
                graphs[name].replay()
            end.record()
            torch.npu.synchronize()
            samples[name].append(start.elapsed_time(end) * 1000 / (unroll * replays))
    return {name: {"median_us": statistics.median(values), "samples_us": values} for name, values in samples.items()}


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.npu.set_device(0)
    if get_ascend_device_type() != AscendDeviceType.A3:
        raise RuntimeError("This benchmark's baseline dispatch and shape matrix are qualified on A3")
    matrix = [
        {"ranks": ranks, "heads": heads, "tokens": tokens}
        for ranks in (4, 5, 8, 9)
        for heads in (24, 12, 16, 8)
        for tokens in range(1, 65)
        if tokens % 4 == 0 or tokens % 6 == 0
    ]
    results = {"device": torch.npu.get_device_name(0), "correctness": [], "performance": []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for case in matrix:
        data, expected = make_inputs(**case)
        validate(data, expected)
        results["correctness"].append(case)
        args.output.write_text(json.dumps(results, indent=2) + "\n")
    for case in matrix:
        data, _ = make_inputs(**case)
        results["performance"].append({"shape": case, "timings": measure(data)})
        args.output.write_text(json.dumps(results, indent=2) + "\n")
    print(f"Validated and measured {len(matrix)} shapes; results: {args.output}")


if __name__ == "__main__":
    main()
