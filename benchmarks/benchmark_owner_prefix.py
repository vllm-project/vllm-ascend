# SPDX-License-Identifier: Apache-2.0
"""Graph-event regression benchmark for owner-prefix selection.

Run this same script with baseline and candidate source trees installed in turn:
    python benchmarks/benchmark_owner_prefix.py --output baseline.json
    python benchmarks/benchmark_owner_prefix.py --output candidate.json --baseline baseline.json

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

from vllm_ascend.ops.triton.sharded_indexer import merge_packed_candidates


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
    cases = [(t, 8, 3, "random") for t in (1, 39, 40, 41, 144, 152, 257)]
    cases += [(41, w, w - 1, "random") for w in (3, 4, 16)]
    cases += [(41, 8, 0, d) for d in ("ties", "own_high", "peer_high", "short", "mixed")]
    for rows, world, owner, distribution in cases:
        k, interleave = 2048, 768
        local = torch.arange(k, dtype=torch.int64)
        ids = (
            torch.stack([local + ((local // interleave) * (world - 1) + r) * interleave for r in range(world)])[
                :, None, :
            ]
            .expand(world, rows, k)
            .clone()
        )
        scores = torch.randn(ids.shape)
        if distribution == "ties":
            scores = scores.round()
        elif distribution == "own_high":
            scores[owner] += 20
        elif distribution == "peer_high":
            scores[(owner + 1) % world] += 20
        elif distribution in ("short", "mixed"):
            for rank in range(world):
                count = 97 + 23 * rank if distribution == "short" else rank * 307 % k
                ids[rank, :, count:] = -1
                scores[rank, :, count:] = -torch.inf
        by_id = torch.where(ids >= 0, ids, torch.iinfo(torch.int64).max).argsort(stable=True)
        order = by_id.gather(-1, scores.gather(-1, by_id).argsort(descending=True, stable=True))
        packed = (
            torch.stack((ids.gather(-1, order).int(), scores.gather(-1, order).contiguous().view(torch.int32)), -1)
            .reshape(world * rows, k, 2)
            .npu()
        )
        name = f"rows={rows},world={world},owner={owner},distribution={distribution}"
        report["cases"][name] = measure(
            lambda packed=packed,
            rows=rows,
            world=world,
            owner=owner,
            k=k,
            interleave=interleave: merge_packed_candidates(
                packed, rows, world, k, local_rank=owner, interleave=interleave, presorted=True, owner_sharded=True
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
