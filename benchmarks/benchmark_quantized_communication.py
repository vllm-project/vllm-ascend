# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Same-run BF16/INT8/MXFP8 full, serial-chunked and pipelined benchmark.

Uses the existing standalone algorithms, one HCCL group, identical per-rank
inputs and an interleaved measurement schedule. No vLLM imports or new kernels.
"""

import argparse
import hashlib
import importlib
import json
import math
import os
import socket
import statistics
import sys
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import benchmark_int8_all_reduce as int8_base
import benchmark_int8_pipeline as int8_pipeline
import benchmark_mxfp8_all_reduce as mx_base
import benchmark_mxfp8_pipeline as mx_pipeline
import torch
import torch.distributed as dist

_WORLD_SIZES = (2, 4, 8)
_BACKENDS = ("int8", "mxfp8")
_MIB = 1024 * 1024


@dataclass(frozen=True)
class Variant:
    name: str
    backend: str
    mode: str
    chunks: int
    op: Callable

    def metadata(self) -> dict:
        return {"name": self.name, "backend": self.backend, "mode": self.mode, "chunks": self.chunks}


def _build_variants(world_size: int, backends: list[str], chunks: list[int], torch_npu) -> list[Variant]:
    variants = [Variant("bf16", "bf16", "full", 1, dist.all_reduce)]
    for backend in backends:
        if backend == "int8":
            ops = int8_base.QuantizationOps("npu", "npu", torch_npu)
            full = int8_base.Int8AllReduce(world_size, ops, amax_method="norm")

            def chunked(count, pipelined, ops=ops):
                return int8_pipeline.ChunkedInt8AllReduce(world_size, ops, chunks=count, pipelined=pipelined)
        else:
            quantize = getattr(torch_npu, "npu_dynamic_mx_quant", None)
            dequantize = getattr(torch_npu, "npu_anti_mx_quant", None)
            if not callable(dequantize):
                dequantize = getattr(torch.ops.npu, "npu_anti_mx_quant", None)
            full = mx_base.Mxfp8AllToAllAllReduce(world_size, quantize, dequantize, dequant_dtype=torch.bfloat16)

            def chunked(count, pipelined, quantize=quantize, dequantize=dequantize):
                return mx_pipeline.ChunkedMxfp8AllToAllAllReduce(
                    world_size, quantize, dequantize, chunks=count, pipelined=pipelined, dequant_dtype=torch.bfloat16
                )

        variants.append(Variant(f"{backend}_full", backend, "full", 1, full))
        for count in chunks:
            for mode in ("serial", "pipeline"):
                variants.append(
                    Variant(
                        f"{backend}_{mode}_chunks_{count}", backend, mode, count, chunked(count, mode == "pipeline")
                    )
                )
    return variants


def _order(names: list[str], cycle: int) -> list[str]:
    """Rotate every cycle; reverse the next complete set of rotations."""
    offset = cycle % len(names)
    result = names[offset:] + names[:offset]
    return result if (cycle // len(names)) % 2 == 0 else result[::-1]


def _summary(samples: list[float]) -> dict:
    if not samples or any(not math.isfinite(value) or value <= 0 for value in samples):
        raise ValueError("Timing samples must be finite and positive.")
    result = int8_base.summarize(samples)
    ordered = sorted(samples)
    result["p99_ms"] = ordered[math.ceil(0.99 * len(ordered)) - 1]
    result["stddev_ms"] = statistics.stdev(samples) if len(samples) > 1 else 0.0
    result["cv_pct"] = 100 * result["stddev_ms"] / result["mean_ms"]
    result["sample_count"] = len(samples)
    return result


def _error_metrics(actual: torch.Tensor, reference: torch.Tensor) -> dict:
    error = actual.float() - reference
    values = int8_base._max_across_ranks(
        [(error.norm() / reference.norm().clamp_min(1e-12)).item(), error.abs().max().item()], actual.device
    )
    return {"relative_l2_vs_fp32": values[0], "max_abs_vs_fp32": values[1]}


def _check_mxfp8(variants: list[Variant], source: torch.Tensor, original_sum: torch.Tensor) -> dict:
    """Share one independent byte-decoded reference across all MXFP8 variants."""
    full = next(variant.op for variant in variants if variant.mode == "full")
    values, scales = full.quantize(source)
    decoded, reference, contribution = mx_base._reference_sum(values, scales)
    local = mx_base._check_output(full.dequantize(values, scales), decoded, decoded.abs(), torch.bfloat16, "MX decode")
    checked_chunks = set()
    accuracy = {}
    full_output = None
    for variant in variants:
        if variant.chunks > 1 and variant.chunks not in checked_chunks:
            mx_pipeline._check_chunk_quantization(full, source, variant.chunks)
            checked_chunks.add(variant.chunks)
        same_payload = torch.empty_like(source)
        variant.op.reduce_quantized(values, scales, same_payload)
        communication = mx_base._check_output(
            same_payload, reference, contribution, torch.bfloat16, variant.name + " same payload"
        )
        del same_payload
        actual = source.clone()
        variant.op(actual)
        complete = mx_base._check_output(actual, reference, contribution, torch.bfloat16, variant.name + " full path")
        if variant.mode == "full":
            full_output = actual
        full_difference = int8_base._max_across_ranks(
            [(actual.float() - full_output.float()).abs().max().item()], source.device
        )[0]
        accuracy[variant.name] = {
            "pass": True,
            **_error_metrics(actual, original_sum),
            "same_payload_max_abs": communication["max_abs_vs_dequantized_sum"],
            "max_abs_vs_full": full_difference,
            "local_dequantization": local,
            "full_path": complete,
            "same_quantized_bytes": True,
        }
    return accuracy


def _check_variants(variants: list[Variant], source: torch.Tensor) -> dict:
    original_sum = source.float()
    dist.all_reduce(original_sum)
    bf16 = source.clone()
    dist.all_reduce(bf16)
    accuracy = {"bf16": {"pass": True, **_error_metrics(bf16, original_sum)}}
    del bf16
    int8 = {variant.name: variant.op for variant in variants if variant.backend == "int8"}
    if int8:
        for name, result in int8_pipeline._check_variants(int8, source).items():
            accuracy[name] = {
                "pass": result["pass"],
                "relative_l2_vs_fp32": result["int8_relative_l2_vs_fp32"],
                "max_abs_vs_fp32": result["int8_max_abs_vs_fp32"],
                "same_payload_max_abs": result["same_payload_max_abs_vs_int32_sum"],
                "max_abs_vs_full": result["max_abs_vs_full"],
                "max_abs_vs_int32_dequantized": result["max_abs_vs_int32_dequantized"],
                "same_quantized_bytes": result["same_quantized_bytes"],
            }
    mxfp8 = [variant for variant in variants if variant.backend == "mxfp8"]
    if mxfp8:
        accuracy.update(_check_mxfp8(mxfp8, source, original_sum))
    return accuracy


def _aggregate(variants: list[Variant], measurements: dict, rounds: int) -> dict:
    result = {}
    for variant in variants:
        blocks = measurements[variant.name]
        samples = [sample for block in blocks for sample in block["samples_ms"]]
        result[variant.name] = {
            **_summary(samples),
            "samples_ms": samples,
            "blocks": blocks,
            "round_means_ms": [
                statistics.mean(sample for block in blocks if block["round"] == index for sample in block["samples_ms"])
                for index in range(rounds)
            ],
            "round_medians_ms": [
                statistics.median(
                    sample for block in blocks if block["round"] == index for sample in block["samples_ms"]
                )
                for index in range(rounds)
            ],
            "wall_ms_including_reset": sum(
                block["wall_ms_including_reset"] * len(block["samples_ms"]) for block in blocks
            )
            / len(samples),
            "extra_peak_allocated_mib": max(block["extra_peak_allocated_mib"] for block in blocks),
        }
    for variant in variants:
        item = result[variant.name]
        baseline = result["bf16"]
        item["speedup_vs_bf16"] = baseline["median_ms"] / item["median_ms"]
        item["mean_speedup_vs_bf16"] = baseline["mean_ms"] / item["mean_ms"]
        item["latency_reduction_vs_bf16_pct"] = 100 * (1 - item["median_ms"] / baseline["median_ms"])
        item["paired_round_mean_reduction_vs_bf16_pct"] = [
            100 * (1 - value / base) for value, base in zip(item["round_means_ms"], baseline["round_means_ms"])
        ]
        if variant.backend != "bf16":
            full = result[f"{variant.backend}_full"]
            item["speedup_vs_full"] = full["median_ms"] / item["median_ms"]
        if variant.mode == "pipeline":
            serial = result[f"{variant.backend}_serial_chunks_{variant.chunks}"]
            item["speedup_vs_serial"] = serial["median_ms"] / item["median_ms"]
    return result


def _cross_backend_comparisons(variants: list[Variant], times: dict) -> list[dict]:
    """Compare full paths, equal chunk counts, and explicitly chosen best p50."""
    lookup = {(variant.backend, variant.mode, variant.chunks): variant.name for variant in variants}
    pairs = []
    for variant in variants:
        other = lookup.get(("mxfp8", variant.mode, variant.chunks))
        if variant.backend == "int8" and other:
            pairs.append((variant.mode + f"_chunks_{variant.chunks}", variant.name, other))
    best = {}
    for backend in _BACKENDS:
        names = [variant.name for variant in variants if variant.backend == backend and variant.mode != "serial"]
        if names:
            best[backend] = min(names, key=lambda name: times[name]["median_ms"])
    if len(best) == len(_BACKENDS):
        pairs.append(("best_observed_p50", best["int8"], best["mxfp8"]))
    return [
        {
            "comparison": label,
            "int8_variant": left,
            "mxfp8_variant": right,
            "int8_p50_ms": times[left]["median_ms"],
            "mxfp8_p50_ms": times[right]["median_ms"],
            "mxfp8_latency_reduction_pct": 100 * (1 - times[right]["median_ms"] / times[left]["median_ms"]),
            "mxfp8_mean_latency_reduction_pct": 100 * (1 - times[right]["mean_ms"] / times[left]["mean_ms"]),
            "paired_round_mean_reduction_pct": [
                100 * (1 - right_time / left_time)
                for left_time, right_time in zip(times[left]["round_means_ms"], times[right]["round_means_ms"])
            ],
        }
        for label, left, right in pairs
    ]


def _benchmark_case(variants: list[Variant], source: torch.Tensor, args: argparse.Namespace) -> dict:
    case = {
        "shape": list(source.shape),
        "bf16_payload_mib": source.numel() * source.element_size() / _MIB,
        "accuracy": _check_variants(variants, source),
    }
    if args.check_only:
        return case
    names = [variant.name for variant in variants]
    lookup = {variant.name: variant.op for variant in variants}
    working = torch.empty_like(source)
    for repeat in range(args.warmup):
        for name in _order(names, repeat):
            working.copy_(source)
            lookup[name](working)
    torch.npu.synchronize()
    del working
    measurements = {name: [] for name in names}
    schedule = []
    cycle = 0
    for repeat in range(args.rounds):
        for first_sample in range(0, args.iterations, args.block_iterations):
            count = min(args.block_iterations, args.iterations - first_sample)
            order = _order(names, cycle)
            schedule.append({"round": repeat, "cycle": cycle, "iterations": count, "order": order})
            for name in order:
                block = int8_base._measure(lookup[name], source, count)
                measurements[name].append({"round": repeat, "cycle": cycle, **block})
            cycle += 1
    case["schedule"] = schedule
    case["measurements"] = _aggregate(variants, measurements, args.rounds)
    case["cross_backend_comparisons"] = _cross_backend_comparisons(variants, case["measurements"])
    return case


def format_report(report: dict) -> str:
    lines = [f"Same-run communication | TP={report['world_size']} | {report['device']} | {report['mode']}"]
    for case in report["cases"]:
        lines.append(f"\nShape={case['shape']}; BF16={case['bf16_payload_mib']:.2f} MiB")
        lines.append("variant | relative L2 / FP32 | max abs / FP32 | same-payload max abs | max abs vs full")
        for name, accuracy in case["accuracy"].items():
            lines.append(
                f"{name} | {accuracy['relative_l2_vs_fp32']:.6g} | {accuracy['max_abs_vs_fp32']:.6g} | "
                f"{accuracy.get('same_payload_max_abs', '-')} | {accuracy.get('max_abs_vs_full', '-')}"
            )
        if "measurements" not in case:
            continue
        lines.append(
            "variant | mean ms | p50 ms | p95 ms | p99 ms | CV % | vs BF16 | vs full | vs serial | "
            "extra peak MiB | wall/reset ms"
        )
        for name, item in case["measurements"].items():
            full = f"{item['speedup_vs_full']:.3f}x" if "speedup_vs_full" in item else "-"
            serial = f"{item['speedup_vs_serial']:.3f}x" if "speedup_vs_serial" in item else "-"
            lines.append(
                f"{name} | {item['mean_ms']:.4f} | {item['median_ms']:.4f} | {item['p95_ms']:.4f} | "
                f"{item['p99_ms']:.4f} | {item['cv_pct']:.2f} | {item['speedup_vs_bf16']:.3f}x | {full} | "
                f"{serial} | {item['extra_peak_allocated_mib']:.2f} | {item['wall_ms_including_reset']:.4f}"
            )
            paired = ", ".join(f"{value:.2f}%" for value in item["paired_round_mean_reduction_vs_bf16_pct"])
            lines.append(f"  {name}: paired round mean reductions vs BF16 = [{paired}]")
        lines.append("comparison | INT8 variant | MXFP8 variant | MXFP8 p50 reduction % | MXFP8 mean reduction %")
        for pair in case["cross_backend_comparisons"]:
            lines.append(
                f"{pair['comparison']} | {pair['int8_variant']} | {pair['mxfp8_variant']} | "
                f"{pair['mxfp8_latency_reduction_pct']:.2f} | {pair['mxfp8_mean_latency_reduction_pct']:.2f}"
            )
    return "\n".join(lines)


def _validate_shapes(args: argparse.Namespace, world_size: int) -> None:
    if world_size not in _WORLD_SIZES:
        raise ValueError("Use TP=2/4/8.")
    for rows in args.tokens:
        for count in args.chunks:
            int8_pipeline.chunk_ranges(rows, count)
            if "mxfp8" in args.backends:
                mx_pipeline.chunk_ranges(rows, world_size, count)
    if "mxfp8" in args.backends and any(hidden % 64 for hidden in args.hidden_sizes):
        raise ValueError("MXFP8 hidden sizes must be multiples of 64.")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", nargs="+", type=int, default=[8192, 16384, 32768])
    parser.add_argument("--hidden-sizes", nargs="+", type=int, default=[5120])
    parser.add_argument("--chunks", nargs="+", type=int, choices=(2, 4), default=[2, 4])
    parser.add_argument("--backends", nargs="+", choices=_BACKENDS, default=list(_BACKENDS))
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument(
        "--iterations", type=int, default=55, help="Samples per variant per round; default balances 11 variants."
    )
    parser.add_argument("--block-iterations", type=int, default=5, help="Switch variants after this many samples.")
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--timeout-seconds", type=int, default=2400)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--report-json", type=Path)
    args = parser.parse_args(argv)
    if (
        any(value < 1 for value in args.tokens)
        or any(value < 64 for value in args.hidden_sizes)
        or args.warmup < 0
        or min(args.iterations, args.block_iterations, args.rounds, args.timeout_seconds) < 1
    ):
        parser.error("Shapes/counts must be positive, hidden >= 64, and warmup nonnegative.")
    for values in (args.tokens, args.hidden_sizes, args.chunks, args.backends):
        if len(set(values)) != len(values):
            parser.error("Shape dimensions, chunk counts and backends must not contain duplicates.")
    return args


def _source_hashes() -> dict:
    directory = Path(__file__).resolve().parent
    names = (
        Path(__file__).name,
        "benchmark_int8_all_reduce.py",
        "benchmark_int8_pipeline.py",
        "benchmark_mxfp8_all_reduce.py",
        "benchmark_mxfp8_pipeline.py",
    )
    return {name: hashlib.sha256((directory / name).read_bytes()).hexdigest() for name in names}


def main() -> None:
    args = _parse_args()
    if args.report_json:
        print(format_report(json.loads(args.report_json.read_text(encoding="utf-8"))), flush=True)
        return
    world_size = int(os.environ["WORLD_SIZE"])
    _validate_shapes(args, world_size)
    torch_npu = importlib.import_module("torch_npu")
    if not torch.npu.is_available():
        raise RuntimeError("This benchmark requires Ascend NPUs and torch_npu.")
    torch.npu.set_device(int(os.environ["LOCAL_RANK"]))
    variants = _build_variants(world_size, args.backends, args.chunks, torch_npu)
    dist.init_process_group("hccl", timeout=timedelta(seconds=args.timeout_seconds))
    try:
        rank = dist.get_rank()
        device = torch.device("npu", torch.npu.current_device())
        report = {
            "schema_version": 1,
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
            "command": sys.argv,
            "source_sha256": _source_hashes(),
            "torch": torch.__version__,
            "torch_npu": torch_npu.__version__,
            "python": sys.version,
            "world_size": world_size,
            "device": torch.npu.get_device_name(),
            "hostname_rank0": socket.gethostname(),
            "mode": "correctness_only" if args.check_only else "benchmark",
            "backends": args.backends,
            "variants": [variant.metadata() for variant in variants],
            "dequant_dtype": "bf16",
            "int8_scale_mode": "one whole-input per-channel scale/MAX; max_q=127//TP",
            "lookahead": 2,
            "warmup": args.warmup,
            "iterations_per_round": args.iterations,
            "block_iterations": args.block_iterations,
            "rounds": args.rounds,
            "seed": args.seed,
            "timing": "Full path NPU events, including all async joins; reset excluded; slowest rank per sample; Eager",
            "schedule": (
                "Rotate variant order each block cycle; reverse each complete set of rotations; same on all ranks"
            ),
            "memory_scope": (
                "PyTorch allocated peak above block start; excludes reserved and external/persistent HCCL memory"
            ),
            "output_contract": "Every benchmark variant mutates its working BF16 tensor; runtime outputs may differ",
            "communication_domains": (
                "One default HCCL group for all data and control collectives; no new groups/user streams"
            ),
            # Record external launcher settings without changing them or importing vLLM.
            "environment": {
                name: os.environ.get(name)
                for name in (
                    "HCCL_OP_EXPANSION_MODE",
                    "HCCL_BUFFSIZE",
                    "HCCL_ALGO",
                    "HCCL_DETERMINISTIC",
                    "HCCL_NPU_SOCKET_PORT_RANGE",
                    "ASCEND_RT_VISIBLE_DEVICES",
                    "TASK_QUEUE_ENABLE",
                    "TORCH_HCCL_BLOCKING_WAIT",
                    "HCCL_BLOCKING_WAIT",
                    "ASCEND_LAUNCH_BLOCKING",
                )
            },
            "sanity_checks": [],
            "cases": [],
        }
        if rank == 0:
            print(json.dumps({key: value for key, value in report.items() if key != "cases"}, indent=2), flush=True)
        # MXFP8 tails in TP-sized units. INT8-only runs can also check odd rows.
        sanity_rows = (2 * max(args.chunks) + 1) * world_size
        hidden = args.hidden_sizes[0]
        for name, value in (
            ("zeros", 0.0),
            ("positive_boundary", 1.0),
            ("negative_boundary", -1.0),
            ("cancellation", (-1.0) ** rank),
            ("rank_constant", rank + 1.0),
        ):
            source = torch.full((sanity_rows, hidden), value, dtype=torch.bfloat16, device=device)
            _check_variants(variants, source)
            report["sanity_checks"].append({"case": name, "pass": True})
            if rank == 0:
                print(f"Correctness {name}: PASS", flush=True)
        indices = torch.arange(sanity_rows * hidden, device=device).reshape(sanity_rows, hidden)
        source = (((indices + rank * 3) % 31 - 15).float() / 8).to(torch.bfloat16)
        source[-1, 0] = 32 * (rank + 1)
        _check_variants(variants, source)
        report["sanity_checks"].append({"case": "rank_channel_chunk_outlier", "pass": True})
        if rank == 0:
            print("Correctness rank_channel_chunk_outlier: PASS", flush=True)
        for variant in variants:
            if variant.backend == "mxfp8":
                mx_pipeline._check_encoded(variant.op, rank, sanity_rows, hidden, device)
        if "mxfp8" in args.backends:
            report["sanity_checks"].append({"case": "rank_token_block_chunk_routing", "pass": True})
            if rank == 0:
                print("Correctness rank_token_block_chunk_routing: PASS", flush=True)
        del source, indices
        for hidden in args.hidden_sizes:
            for rows in args.tokens:
                torch.manual_seed(args.seed + rank)
                source = torch.randn(rows, hidden, dtype=torch.bfloat16, device=device)
                case = _benchmark_case(variants, source, args)
                report["cases"].append(case)
                del source
                if rank == 0:
                    print(format_report({**report, "cases": [case]}), flush=True)
                    if args.output:
                        args.output.parent.mkdir(parents=True, exist_ok=True)
                        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        if rank == 0:
            print("PASS: all requested same-run communication checks passed.", flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
