# SPDX-License-Identifier: Apache-2.0
"""Graph-event regression benchmark for packed DCP LSE combining.

Run this same script with baseline and candidate source trees installed in turn:
    python benchmarks/benchmark_dcp_combine.py --output baseline.json
    python benchmarks/benchmark_dcp_combine.py --output candidate.json --baseline baseline.json

Use the same NPU, software, clock and exclusive device ownership for both runs.
This measures graph replay latency, separately from msprof Task Duration.
"""

import argparse
import json
import statistics
import subprocess
from pathlib import Path

import torch
import torch_npu

from vllm_ascend.ops.triton.dcp.dcp_a2a import fused_dcp_lse_combine


def make_inputs(shape, edge=False):
    tokens, heads, dim, world, has_local, head_major = shape
    generator = torch.Generator().manual_seed(20878 + tokens)
    values = torch.randn(world, tokens, heads, dim, generator=generator).bfloat16()
    lses = torch.randn(world, tokens, heads, generator=generator) * 4
    local = torch.randn(tokens, heads, dim, generator=generator).bfloat16()
    local_lse = torch.randn(tokens, heads, generator=generator) * 4
    if edge:
        lses[:, 0, 0] = -torch.inf
        local_lse[0, 0] = torch.nan
        values[:, 0, 0] = torch.nan
        local[0, 0] = torch.inf
        lses[0, -1, 0] = torch.nan
        lses[-1, -1, -1] = torch.inf
        values[0, -1, 0] = torch.nan
        lses[:, 2, 0], lses[:, 3, 0] = 200000.0, -200000.0
        local_lse[2, 0], local_lse[3, 0] = 200000.0, -200000.0
    raw = lses.contiguous().view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    exponent, fraction = (raw >> 23) & 255, raw & 0x7FFFFF
    finite = exponent < 255
    significand = fraction | torch.where(exponent == 0, 0, 0x800000)
    significand = torch.where(finite, significand, 0)
    code = (torch.maximum(exponent, torch.ones_like(exponent)) + 1).float()
    code = torch.where(raw >> 31 != 0, -code, code)
    code = torch.where((raw & 0x7FFFFFFF) == 0, 128.0, code)
    code = torch.where(finite, code, 0.0)
    fields = torch.stack(
        (
            code,
            ((significand >> 16) & 255).float(),
            ((significand >> 8) & 255).float(),
            (significand & 255).float(),
        ),
        -1,
    ).bfloat16()
    recv = torch.cat((values, fields), -1).permute(0, 2, 1, 3).contiguous().npu()
    all_lses = torch.cat((lses, local_lse[None]), 0) if has_local else lses
    all_values = torch.cat((values, local[None]), 0) if has_local else values
    valid = torch.isfinite(all_lses)
    maximum = torch.where(valid, all_lses.double(), -torch.inf).max(0).values
    maximum = torch.where(torch.isfinite(maximum), maximum, 0.0)
    weights = torch.where(valid, torch.exp(all_lses.double() - maximum), 0.0)
    numerator = (torch.where(valid[..., None], all_values.double(), 0.0) * weights[..., None]).sum(0)
    denominator = weights.sum(0)
    expected = numerator / torch.where(denominator > 0, denominator, 1.0)[..., None]
    if head_major:
        local = local.transpose(0, 1).contiguous().npu().transpose(0, 1)
    else:
        local = local.npu()
    return (
        recv,
        local if has_local else None,
        local_lse.unsqueeze(-1).npu() if has_local else None,
        expected,
    )


def measure(fn, iterations, repeats):
    for _ in range(10):
        fn()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        result = fn()
    for _ in range(10):
        graph.replay()
    torch.npu.synchronize()
    samples = []
    for _ in range(repeats):
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000 / iterations)
    # Keep the captured output alive until all graph replays complete.
    del result
    graph.reset()
    return {"median_us": statistics.median(samples), "samples_us": samples}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--max-slowdown", type=float, default=1.10)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=10)
    args = parser.parse_args()
    if args.iterations < 1 or args.repeats < 2 or args.max_slowdown <= 0:
        parser.error("iterations >= 1, repeats >= 2 and max-slowdown > 0 are required")
    torch.npu.set_device(0)
    torch.manual_seed(20878)
    report = {
        "protocol": "NPUGraph replay / NPU events",
        "device": torch.npu.get_device_name(0),
        "torch": torch.__version__,
        "torch_npu": torch_npu.__version__,
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "iterations": args.iterations,
        "repeats": args.repeats,
        "cases": {},
    }
    shapes = (
        (144, 12, 512, 8, True, False),
        (152, 12, 512, 8, True, False),
        (114, 8, 512, 8, False, False),
        (20, 8, 64, 4, False, False),
        (41, 7, 128, 3, True, False),
        (127, 16, 256, 16, True, False),
        (129, 12, 512, 4, False, False),
        (257, 12, 512, 8, True, False),
        (41, 12, 512, 8, True, True),
    )
    for shape in shapes:
        recv, local, local_lse, _ = make_inputs(shape)
        name = str(shape)
        report["cases"][name] = measure(
            lambda recv=recv, dim=shape[2], local=local, local_lse=local_lse: fused_dcp_lse_combine(
                recv, dim, 1, local_output=local, local_lse=local_lse
            ),
            args.iterations,
            args.repeats,
        )
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if args.baseline:
        baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
        for key in ("protocol", "device", "torch", "torch_npu", "iterations", "repeats"):
            if baseline[key] != report[key]:
                raise RuntimeError(f"Incompatible comparison metadata: {key}")
        if baseline["cases"].keys() != report["cases"].keys():
            raise RuntimeError("Baseline and candidate case sets differ")
        failures = []
        for name, current in report["cases"].items():
            previous = baseline["cases"][name]
            for side in (previous, current):
                spread = (max(side["samples_us"]) - min(side["samples_us"])) / side["median_us"]
                if spread > 0.10:
                    failures.append(f"{name} unstable samples: spread={spread:.3f}")
            ratio = current["median_us"] / previous["median_us"]
            print(f"{name}: {previous['median_us']:.3f} -> {current['median_us']:.3f} us ({ratio:.3f}x)")
            if ratio > args.max_slowdown:
                failures.append(f"{name} slowdown {ratio:.3f} > {args.max_slowdown:.3f}")
        if failures:
            raise RuntimeError("; ".join(failures))


if __name__ == "__main__":
    main()
