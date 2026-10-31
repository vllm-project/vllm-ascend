# SPDX-License-Identifier: Apache-2.0
"""Manual eager performance comparison of the complete old/new QSA selectors.

Run inside the existing NPU environment, after loading the V2 C++ bindings:

    python tests/manual/test_benchmark_qsa_lightning_indexer.py
    python tests/manual/test_benchmark_qsa_lightning_indexer.py --cases decode mixed
    python tests/manual/test_benchmark_qsa_lightning_indexer.py \
        --seq-lens 8192,16384 --query-lens 1,128 --heads 4 --page-size 384 \
        --page-stride 2 --iterations 50 --rounds 7 --output /tmp/qsa-li-perf.json

Input generation, warmup and output allocation are outside the timed region.
Both functions see the same BF16 query/cache and request metadata, with their
own preallocated INT32 output buffers, just as the QSA caller supplies `out`.
Timing INCLUDES each selector's validation, temporary allocations, input
preparation, LI execution (including V2 scheduling metadata), and packing.
This is a warm-cache eager microbenchmark, not a pure-kernel timer, graph-mode
benchmark, correctness test, or model/server end-to-end performance result.
"""

import argparse
import json
import math
import platform
import statistics
import time
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Callable

import torch
import torch_npu
import vllm_ascend.vllm_ascend_C  # noqa: F401

from vllm_ascend.models.qwen4_exp.lightning_indexer import (
    qsa_select_paged_tokens_lightning,
    qsa_select_paged_tokens_lightning_v2,
)

HEAD_DIM = 128
COMPRESS_RATIO = 4
TOKEN_TOPK = 2048
OUTPUT_WIDTH = TOKEN_TOPK + COMPRESS_RATIO - 1
MICROSECONDS_PER_SECOND = 1_000_000
MIB = 1024 * 1024
SEED = 1031
DEFAULT_HEADS = 4
DEFAULT_PAGE_SIZE = 384
DEFAULT_WARMUP = 5
DEFAULT_ITERATIONS = 20
DEFAULT_ROUNDS = 5


@dataclass(frozen=True)
class Shape:
    name: str
    sequence_lengths: tuple[int, ...]
    query_lengths: tuple[int, ...]


DEFAULT_SHAPES = (
    Shape("decode_1", (8192,), (1,)),
    Shape("decode_16", (8192,) * 16, (1,) * 16),
    Shape("prefill", (1024,), (1024,)),
    Shape("chunked", (8192,) * 4, (128,) * 4),
    Shape("mixed", (8192,) * 8, (1, 1, 1, 1, 32, 64, 128, 128)),
)


def integer_list(value: str) -> tuple[int, ...]:
    try:
        result = tuple(int(part.strip()) for part in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated positive integers") from exc
    if not result or any(item <= 0 for item in result):
        raise argparse.ArgumentTypeError("expected comma-separated positive integers")
    return result


def make_inputs(shape: Shape, heads: int, page_size: int, page_stride: int, device: torch.device):
    """Build contiguous query suffixes and distinct shuffled pages per request."""
    boundaries = [0]
    for size in shape.query_lengths:
        boundaries.append(boundaries[-1] + size)
    generator = torch.Generator().manual_seed(SEED)
    page_counts = tuple(
        max(1, math.ceil((length // COMPRESS_RATIO) / page_size))
        for length in shape.sequence_lengths
    )
    page_count = sum(page_counts)
    table_cpu = torch.zeros(len(page_counts), max(page_counts), dtype=torch.int32)
    permutation = torch.randperm(page_count, generator=generator).to(torch.int32)
    offset = 0
    for request, count in enumerate(page_counts):
        table_cpu[request, :count] = permutation[offset:offset + count]
        offset += count
    # Keep the backing allocation dense before taking a strided page view.
    # Only the page axis has gaps, matching the cache ABI both selectors accept.
    storage = torch.randn(
        page_count * page_stride, page_size, 1, HEAD_DIM,
        dtype=torch.bfloat16, device=device,
    )
    cache = storage[::page_stride]
    query = torch.randn(boundaries[-1], heads, HEAD_DIM, dtype=torch.bfloat16, device=device)
    requests_cpu = torch.repeat_interleave(
        torch.arange(len(shape.query_lengths)), torch.tensor(shape.query_lengths),
    )
    positions_cpu = torch.cat([
        torch.arange(length - count, length)
        for length, count in zip(shape.sequence_lengths, shape.query_lengths)
    ])
    arguments = (
        query,
        cache,
        table_cpu.to(device),
        requests_cpu.to(device),
        positions_cpu.to(device),
        torch.tensor(shape.sequence_lengths, dtype=torch.int32, device=device),
        torch.tensor(boundaries, dtype=torch.int32, device=device),
        TOKEN_TOPK,
        COMPRESS_RATIO,
    )
    buffers = {
        name: torch.empty(boundaries[-1], OUTPUT_WIDTH, dtype=torch.int32, device=device)
        for name in ("old", "v2")
    }
    functions = {
        "old": partial(qsa_select_paged_tokens_lightning, *arguments, out=buffers["old"]),
        "v2": partial(qsa_select_paged_tokens_lightning_v2, *arguments, out=buffers["v2"]),
    }
    geometry = {
        "requests": len(shape.query_lengths),
        "query_tokens": boundaries[-1],
        "sequence_lengths": shape.sequence_lengths,
        "query_lengths": shape.query_lengths,
        "heads": heads,
        "page_size": page_size,
        "page_stride": page_stride,
        "cache_shape": tuple(cache.shape),
        "cache_stride": cache.stride(),
        "backing_cache_mib": storage.numel() * storage.element_size() / MIB,
        "request_table_rows": len(shape.query_lengths),
        "old_expanded_table_rows": boundaries[-1],
    }
    return functions, geometry


def measure(function: Callable, iterations: int, mode: str) -> list[float]:
    """Return synchronized wall-time samples in microseconds per invocation."""
    samples = []
    torch.npu.synchronize()
    if mode == "latency":
        for _ in range(iterations):
            start = time.perf_counter()
            function()
            torch.npu.synchronize()
            samples.append((time.perf_counter() - start) * MICROSECONDS_PER_SECOND)
    else:
        start = time.perf_counter()
        for _ in range(iterations):
            function()
        torch.npu.synchronize()
        samples.append((time.perf_counter() - start) * MICROSECONDS_PER_SECOND / iterations)
    return samples


def percentile(samples: list[float], fraction: float) -> float:
    ordered = sorted(samples)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def summarize(samples: list[float]) -> dict:
    return {
        "samples_us": samples,
        "median_us": statistics.median(samples),
        "mean_us": statistics.mean(samples),
        "p95_us": percentile(samples, 0.95),
        "min_us": min(samples),
        "max_us": max(samples),
    }


def benchmark(functions: dict[str, Callable], warmup: int, iterations: int, rounds: int, mode: str):
    for _ in range(warmup):
        functions["old"]()
        functions["v2"]()
    torch.npu.synchronize()
    samples = {"old": [], "v2": []}
    for index in range(rounds):
        # Reverse order every round to reduce systematic first/second bias.
        order = ("old", "v2") if index % 2 == 0 else ("v2", "old")
        for name in order:
            # Refresh each path before its timed block, outside the measurement.
            functions[name]()
            torch.npu.synchronize()
            samples[name].extend(measure(functions[name], iterations, mode))
    results = {name: summarize(values) for name, values in samples.items()}
    results["speedup_old_over_v2"] = results["old"]["median_us"] / results["v2"]["median_us"]
    return results


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", type=int, default=0, help="NPU device index")
    parser.add_argument("--heads", type=int, default=DEFAULT_HEADS)
    parser.add_argument("--page-size", type=int, default=DEFAULT_PAGE_SIZE)
    parser.add_argument("--page-stride", type=int, choices=(1, 2), default=1)
    parser.add_argument("--cases", nargs="+", choices=("decode", "prefill", "chunked", "mixed"))
    parser.add_argument("--seq-lens", type=integer_list, help="custom request sequence lengths")
    parser.add_argument("--query-lens", type=integer_list, help="custom query suffix lengths")
    parser.add_argument("--mode", choices=("latency", "throughput", "both"), default="both")
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS)
    parser.add_argument("--rounds", type=int, default=DEFAULT_ROUNDS)
    parser.add_argument("--output", type=Path, help="optional JSON report including raw timing samples")
    args = parser.parse_args()
    if not 1 <= args.heads <= 64:
        parser.error("--heads must be in [1,64]")
    if not 16 <= args.page_size <= 1024 or args.page_size % 16:
        parser.error("--page-size must be a multiple of 16 in [16,1024]")
    if args.device < 0 or min(args.warmup, args.iterations, args.rounds) <= 0:
        parser.error("device must be nonnegative; warmup, iterations and rounds must be positive")
    if (args.seq_lens is None) != (args.query_lens is None):
        parser.error("--seq-lens and --query-lens must be supplied together")
    if args.seq_lens is not None:
        if args.cases:
            parser.error("custom lengths cannot be combined with --cases")
        if len(args.seq_lens) != len(args.query_lens):
            parser.error("custom sequence/query lists must have equal lengths")
        if any(query > length for length, query in zip(args.seq_lens, args.query_lens)):
            parser.error("each query length must not exceed its sequence length")
    return args


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    device = torch.device(f"npu:{args.device}")
    torch.npu.set_device(device)
    torch.manual_seed(SEED)
    if args.seq_lens is not None:
        shapes = (Shape("custom", args.seq_lens, args.query_lens),)
    else:
        shapes = tuple(
            shape for shape in DEFAULT_SHAPES
            if not args.cases or any(shape.name.startswith(name) for name in args.cases)
        )
    modes = ("latency", "throughput") if args.mode == "both" else (args.mode,)
    report = {
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_npu": torch_npu.__version__,
            "device": str(device),
            "device_name": torch.npu.get_device_name(args.device),
        },
        "warmup": args.warmup,
        "iterations": args.iterations,
        "rounds": args.rounds,
        "measurement": "complete eager selector, same inputs, preallocated outputs, warm cache",
        "cases": [],
    }
    print(json.dumps(report["environment"], ensure_ascii=False), flush=True)
    print("latency: synchronized wall time per call, including the final synchronization overhead.")
    print("throughput: synchronized block wall time / calls; samples are round averages, not call latencies.")
    print("Speedup = old median / V2 median (>1 means V2 is faster). No correctness assertions are run.")
    for shape in shapes:
        functions, geometry = make_inputs(shape, args.heads, args.page_size, args.page_stride, device)
        print(f"\n{shape.name}: {json.dumps(geometry)}", flush=True)
        case = {"name": shape.name, "geometry": geometry, "timings": {}}
        for mode in modes:
            results = benchmark(functions, args.warmup, args.iterations, args.rounds, mode)
            case["timings"][mode] = results
            for name in ("old", "v2"):
                values = results[name]
                print(
                    f"  {mode:10s} {name:3s}: median={values['median_us']:.2f} us "
                    f"mean={values['mean_us']:.2f} p95={values['p95_us']:.2f} "
                    f"min={values['min_us']:.2f} max={values['max_us']:.2f}",
                    flush=True,
                )
            print(f"  {mode:10s} speedup={results['speedup_old_over_v2']:.3f}x", flush=True)
        report["cases"].append(case)
        del functions
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"\nReport: {args.output}", flush=True)


if __name__ == "__main__":
    main()
