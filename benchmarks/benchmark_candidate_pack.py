# SPDX-License-Identifier: Apache-2.0
"""Graph-event regression benchmark for fused candidate packing.

Run this same script with baseline and candidate source trees installed in turn:
    python benchmarks/benchmark_candidate_pack.py --output baseline.json
    python benchmarks/benchmark_candidate_pack.py --output candidate.json --baseline baseline.json

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

from vllm_ascend.ops.triton.sharded_indexer import pack_local_candidates


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
    for rows in (1, 48, 127, 144, 152, 257):
        ids = torch.randint(0, 131072, (rows, 1, 2048), device="npu", dtype=torch.int32)
        for dtype in (torch.float32, torch.bfloat16):
            scores = torch.randn(ids.shape, device="npu", dtype=dtype)
            name = f"rows={rows},dtype={dtype}"
            report["cases"][name] = measure(
                lambda ids=ids, scores=scores: pack_local_candidates(
                    ids, scores, 8, 7, 768, sort_candidates=True, fuse_sort=True
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
