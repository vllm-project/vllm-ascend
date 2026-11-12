# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare full, serial-chunked and pipelined INT8 all-reduce (Eager, BF16).

Every variant computes one whole-input, shared per-channel scale per call.
Run with torchrun; only PyTorch and torch_npu are required, without vLLM.
"""

import argparse
import importlib
import json
import os
import socket
import statistics
from collections import deque
from collections.abc import Callable
from datetime import timedelta
from pathlib import Path
from typing import Any, NamedTuple

import benchmark_int8_all_reduce as base
import torch
import torch.distributed as dist

_LOOKAHEAD = 2
_BF16_RTOL = 0.008
_CHECK_ATOL = 1e-7


def chunk_ranges(rows: int, chunks: int) -> list[tuple[int, int]]:
    """Equal-size row chunks plus a tail; AllReduce does not require TP alignment."""
    if not 1 <= chunks <= rows:
        raise ValueError("The row count must be positive and at least the chunk count.")
    size, remainder = divmod(rows, chunks)
    ranges = []
    start = 0
    for index in range(chunks):
        end = start + size + int(index < remainder)
        ranges.append((start, end))
        start = end
    return ranges


class _Chunk(NamedTuple):
    start: int
    end: int
    quantized: torch.Tensor
    work: Any


class ChunkedInt8AllReduce(base.Int8AllReduce):
    """One shared scale followed by two-chunk lookahead on one HCCL group."""

    def __init__(self, *args, chunks: int, pipelined: bool, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if chunks < 2:
            raise ValueError("Use at least two chunks; int8_full is the unchunked control.")
        self.chunks = chunks
        self.pipelined = pipelined

    def _run(self, output: torch.Tensor, prepare: Callable, finish: Callable) -> torch.Tensor:
        ranges = iter(chunk_ranges(output.shape[0], self.chunks))

        def submit(start: int, end: int) -> _Chunk:
            quantized = prepare(start, end)
            work = dist.all_reduce(quantized, group=self.group, async_op=self.pipelined)
            return _Chunk(start, end, quantized, work)

        if not self.pipelined:
            for start, end in ranges:
                chunk = submit(start, end)
                output[start:end].copy_(finish(chunk.quantized))
                del chunk
            return output

        pending = deque()
        for _ in range(min(_LOOKAHEAD, self.chunks)):
            pending.append(submit(*next(ranges)))
        while pending:
            chunk = pending.popleft()
            # Q1/AR1 are submitted before joining AR0 onto the compute stream.
            chunk.work.wait()
            output[chunk.start : chunk.end].copy_(finish(chunk.quantized))
            del chunk
            next_range = next(ranges, None)
            if next_range is not None:
                pending.append(submit(*next_range))
        # All SUMs have been joined and BF16 copies enqueued on the caller's
        # stream. Its end event therefore includes every result dependency.
        return output

    def reduce_quantized(self, quantized: torch.Tensor) -> torch.Tensor:
        """Use the same schedule to check raw INT8 SUM against INT32 SUM."""
        output = torch.empty_like(quantized)
        return self._run(output, lambda start, end: quantized[start:end].clone(), lambda reduced: reduced)

    def __call__(self, x: torch.Tensor, markers=None) -> torch.Tensor:
        if markers is not None:
            raise ValueError("Serial stage markers cannot measure overlap; use --profile-dir.")
        chunk_ranges(x.shape[0], self.chunks)
        # Keep this whole-input scan/MAX in the primary end-to-end timing.
        scale = self.global_scale(x)
        return self._run(
            x,
            lambda start, end: self.ops.quantize(x[start:end], scale),
            lambda reduced: self.ops.dequantize(reduced, scale),
        )


def _require_all(condition: bool, message: str, device: torch.device) -> None:
    if base._max_across_ranks([int(not condition)], device)[0]:
        raise AssertionError(message)


def _check_variants(variants: dict, source: torch.Tensor) -> dict:
    """Check identical quantization, integer SUM, then independently decoded output."""
    full = variants["int8_full"]
    scale = full.global_scale(source)
    quantized = full.ops.quantize(source, scale)
    _require_all(
        quantized.dtype == torch.int8 and quantized.shape == source.shape and quantized.device == source.device,
        "Unexpected quantized shape/dtype/device.",
        source.device,
    )
    _require_all(
        bool(torch.isfinite(source).all())
        and bool(torch.isfinite(scale).all())
        and bool((quantized.float().abs() <= full.max_q).all()),
        "Nonfinite input/scale or quantized range exceeds the integer-safe bound.",
        source.device,
    )
    reference_int32 = quantized.to(torch.int32)
    dist.all_reduce(reference_int32, group=full.group)
    expected = (reference_int32.float() * scale).to(torch.bfloat16)
    original_sum = source.float()
    dist.all_reduce(original_sum, group=full.group)
    bf16 = source.clone()
    dist.all_reduce(bf16, group=full.group)
    denominator = original_sum.norm().clamp_min(1e-12)
    bf16_l2 = ((bf16.float() - original_sum).norm() / denominator).item()
    accuracy = {}
    full_output = None
    for name, op in variants.items():
        if name == "bf16":
            continue
        same_bytes = True
        if isinstance(op, ChunkedInt8AllReduce):
            for start, end in chunk_ranges(source.shape[0], op.chunks):
                same_bytes &= torch.equal(op.ops.quantize(source[start:end], scale), quantized[start:end])
            reduced = op.reduce_quantized(quantized)
        else:
            reduced = quantized.clone()
            dist.all_reduce(reduced, group=op.group)
        _require_all(same_bytes, f"{name}: chunked quantization bytes differ from full quantization.", source.device)
        _require_all(
            reduced.dtype == torch.int8 and torch.equal(reduced.to(torch.int32), reference_int32),
            f"{name}: INT8 SUM differs from INT32 SUM (overflow, ordering or chunk routing).",
            source.device,
        )
        actual = source.clone()
        op(actual)
        _require_all(
            actual.dtype == torch.bfloat16
            and bool(torch.isfinite(actual).all())
            and torch.allclose(actual, expected, rtol=_BF16_RTOL, atol=_CHECK_ATOL),
            f"{name}: full path differs from INT32 SUM + independent FP32 dequantization.",
            source.device,
        )
        if name == "int8_full":
            full_output = actual
        if full_output is None:
            raise ValueError("Place int8_full before chunked variants for the full-output comparison.")
        error = actual.float() - original_sum
        stats = base._max_across_ranks(
            [
                (error.norm() / denominator).item(),
                error.abs().max().item(),
                bf16_l2,
                (actual.float() - expected.float()).abs().max().item(),
                (actual.float() - full_output.float()).abs().max().item(),
            ],
            source.device,
        )
        accuracy[name] = {
            "pass": True,
            "same_quantized_bytes": True,
            "same_payload_max_abs_vs_int32_sum": 0,
            "int8_relative_l2_vs_fp32": stats[0],
            "int8_max_abs_vs_fp32": stats[1],
            "bf16_relative_l2_vs_fp32": stats[2],
            "max_abs_vs_int32_dequantized": stats[3],
            "max_abs_vs_full": stats[4],
        }
    return accuracy


def _benchmark_case(variants: dict, source: torch.Tensor, args: argparse.Namespace) -> dict:
    case = {
        "shape": list(source.shape),
        "bf16_payload_mib": source.numel() * source.element_size() / base._MIB,
        "chunk_rows": {
            str(chunks): [end - start for start, end in chunk_ranges(source.shape[0], chunks)] for chunks in args.chunks
        },
        "accuracy": _check_variants(variants, source),
    }
    if args.check_only:
        return case
    working = torch.empty_like(source)
    for _ in range(args.warmup):
        for op in variants.values():
            working.copy_(source)
            op(working)
    torch.npu.synchronize()
    del working
    measurements = {name: [] for name in variants}
    names = list(variants)
    for repeat in range(args.rounds):
        offset = repeat % len(names)
        order = names[offset:] + names[:offset]
        if repeat % 2:
            order.reverse()
        for name in order:
            measurements[name].append(base._measure(variants[name], source, args.iterations))
    times = {}
    for name, rounds in measurements.items():
        samples = [sample for item in rounds for sample in item["samples_ms"]]
        times[name] = {
            **base.summarize(samples),
            "round_medians_ms": [statistics.median(item["samples_ms"]) for item in rounds],
            "samples_ms": samples,
            "median_wall_ms_including_reset": statistics.median(item["wall_ms_including_reset"] for item in rounds),
            "extra_peak_allocated_mib": max(item["extra_peak_allocated_mib"] for item in rounds),
        }
    if any(item["median_ms"] <= 0 for item in times.values()):
        raise RuntimeError("Nonpositive NPU event duration; increase payload or verify event timing.")
    for name, item in times.items():
        item["speedup_vs_bf16"] = times["bf16"]["median_ms"] / item["median_ms"]
        item["speedup_vs_full"] = times["int8_full"]["median_ms"] / item["median_ms"]
        if name.startswith("pipeline_chunks_"):
            serial = name.replace("pipeline_", "serial_", 1)
            item["speedup_vs_serial_chunks"] = times[serial]["median_ms"] / item["median_ms"]
    case["measurements"] = times
    return case


def format_report(report: dict) -> str:
    lines = [f"INT8 pipeline | TP={report['world_size']} | device={report['device']} | {report['mode']}"]
    for case in report["cases"]:
        lines.append(
            f"\nShape={case['shape']}; BF16={case['bf16_payload_mib']:.2f} MiB; chunk rows={case['chunk_rows']}"
        )
        lines.append("variant | relative L2 / FP32 SUM | max abs / FP32 SUM | INT8 SUM vs INT32 | max abs vs full")
        for name, accuracy in case["accuracy"].items():
            lines.append(
                f"{name} | {accuracy['int8_relative_l2_vs_fp32']:.6g} | "
                f"{accuracy['int8_max_abs_vs_fp32']:.6g} | "
                f"{accuracy['same_payload_max_abs_vs_int32_sum']} | {accuracy['max_abs_vs_full']:.6g}"
            )
        if "measurements" not in case:
            continue
        lines.append("variant | p50 ms | p95 ms | vs BF16 | vs full | vs chunk serial | extra peak MiB | wall/reset ms")
        for name, item in case["measurements"].items():
            serial_gain = f"{item['speedup_vs_serial_chunks']:.3f}x" if "speedup_vs_serial_chunks" in item else "-"
            lines.append(
                f"{name} | {item['median_ms']:.4f} | {item['p95_ms']:.4f} | "
                f"{item['speedup_vs_bf16']:.3f}x | {item['speedup_vs_full']:.3f}x | {serial_gain} | "
                f"{item['extra_peak_allocated_mib']:.2f} | {item['median_wall_ms_including_reset']:.4f}"
            )
    return "\n".join(lines)


def _capture_profile(variants: dict, source: torch.Tensor, args: argparse.Namespace, rank: int) -> str:
    torch_npu = importlib.import_module("torch_npu")
    rows, hidden = source.shape
    directory = args.profile_dir / f"tp{dist.get_world_size()}_{rows}x{hidden}" / f"rank{rank}"
    directory.mkdir(parents=True, exist_ok=True)
    working = torch.empty_like(source)
    torch.npu.synchronize()
    dist.barrier()
    profiler = torch_npu.profiler
    with profiler.profile(
        activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.NPU],
        record_shapes=True,
        on_trace_ready=profiler.tensorboard_trace_handler(str(directory), analyse_flag=True),
    ) as capture:
        for _ in range(args.profile_steps):
            for name, op in variants.items():
                with torch.profiler.record_function("pipeline_benchmark_input_reset"):
                    working.copy_(source)
                with torch.profiler.record_function(name):
                    op(working)
                torch.npu.synchronize()
            capture.step()
    return str(directory)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", nargs="+", type=int, default=[8192, 16384, 32768])
    parser.add_argument("--hidden-sizes", nargs="+", type=int, default=[5120])
    parser.add_argument("--chunks", nargs="+", type=int, default=[2, 4])
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--quant-backend", choices=("npu", "torch"), default="npu")
    parser.add_argument("--dequant-backend", choices=("npu", "torch"), default="npu")
    parser.add_argument("--amax-method", choices=("norm", "abs"), default="norm")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--timeout-seconds", type=int, default=2400)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--report-json", type=Path)
    parser.add_argument("--profile-dir", type=Path)
    parser.add_argument("--profile-steps", type=int, default=2)
    args = parser.parse_args(argv)
    if any(value < 1 for value in (*args.tokens, *args.hidden_sizes)) or any(value < 2 for value in args.chunks):
        parser.error("Shapes must be positive and chunk counts must be at least two.")
    if len(set(args.chunks)) != len(args.chunks):
        parser.error("Chunk counts must be distinct.")
    if any(rows < max(args.chunks) for rows in args.tokens):
        parser.error("Each token count must be at least the largest chunk count.")
    if args.warmup < 0 or min(args.iterations, args.rounds, args.timeout_seconds, args.profile_steps) < 1:
        parser.error("Iterations, rounds, timeout and profile steps must be positive; warmup must be nonnegative.")
    if args.check_only and args.profile_dir:
        parser.error("Use --profile-dir in a separate performance run, without --check-only.")
    return args


def main() -> None:
    args = _parse_args()
    if args.report_json:
        print(format_report(json.loads(args.report_json.read_text(encoding="utf-8"))), flush=True)
        return
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size not in base._SUPPORTED_WORLD_SIZES:
        raise ValueError("Use TP=2/4/8.")
    torch_npu = importlib.import_module("torch_npu")
    if not torch.npu.is_available():
        raise RuntimeError("This benchmark requires Ascend NPUs and torch_npu.")
    torch.npu.set_device(int(os.environ["LOCAL_RANK"]))
    ops = base.QuantizationOps(args.quant_backend, args.dequant_backend, torch_npu)
    variants = {"bf16": dist.all_reduce, "int8_full": base.Int8AllReduce(world_size, ops, args.amax_method)}
    for chunks in args.chunks:
        for pipelined, name in ((False, "serial"), (True, "pipeline")):
            variants[f"{name}_chunks_{chunks}"] = ChunkedInt8AllReduce(
                world_size, ops, args.amax_method, chunks=chunks, pipelined=pipelined
            )
    dist.init_process_group("hccl", timeout=timedelta(seconds=args.timeout_seconds))
    try:
        rank = dist.get_rank()
        device = torch.device("npu", torch.npu.current_device())
        report = {
            "torch": torch.__version__,
            "torch_npu": torch_npu.__version__,
            "world_size": world_size,
            "device": torch.npu.get_device_name(),
            "hostname_rank0": socket.gethostname(),
            "mode": "correctness_only" if args.check_only else "benchmark",
            "scale_mode": "whole-input shared per-channel scale; one MAX collective per call",
            "max_q": variants["int8_full"].max_q,
            "quant_backend": args.quant_backend,
            "dequant_backend": args.dequant_backend,
            "amax_method": args.amax_method,
            "dequant_dtype": "bf16",
            "chunks": args.chunks,
            "lookahead": _LOOKAHEAD,
            "warmup": args.warmup,
            "iterations_per_round": args.iterations,
            "rounds": args.rounds,
            "seed": args.seed,
            "timing": "NPU events including amax/MAX, quantization, SUM joins and BF16 writeback; reset excluded; "
            "slowest rank per sample; Eager",
            "memory_scope": "PyTorch allocator peak; excludes reserved memory and external/persistent HCCL resources",
            "communication_domains": "One default HCCL group; no additional groups or user-created streams",
            # External launcher/HCCL settings; keep this standalone benchmark independent of vLLM initialization.
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
            print(json.dumps(report, indent=2), flush=True)
        # Odd row counts exercise uneven tails without requiring divisibility by TP.
        sanity_rows = 2 * max(args.chunks) + 1
        hidden = args.hidden_sizes[0]
        for name, value in (
            ("zeros", 0.0),
            ("same_sign_overflow_boundary", 1.0),
            ("negative_overflow_boundary", -1.0),
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
        del source, indices
        for hidden in args.hidden_sizes:
            for rows in args.tokens:
                torch.manual_seed(args.seed + rank)
                source = torch.randn(rows, hidden, dtype=torch.bfloat16, device=device)
                case = _benchmark_case(variants, source, args)
                if args.profile_dir:
                    case["profile_directory_rank0"] = _capture_profile(variants, source, args, rank)
                report["cases"].append(case)
                del source
                if rank == 0:
                    print(format_report({**report, "cases": [case]}), flush=True)
                    if args.output:
                        args.output.parent.mkdir(parents=True, exist_ok=True)
                        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        if rank == 0:
            print("PASS: all requested INT8 pipeline checks passed.", flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
