# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare full, serial-chunked and pipelined MXFP8 all-reduce (Eager, BF16).

Run with torchrun. Reuse benchmark_mxfp8_all_reduce's byte format, independent
accuracy reference and end-to-end event timer. No vLLM runtime is imported.
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

import benchmark_mxfp8_all_reduce as base
import torch
import torch.distributed as dist

_LOOKAHEAD = 2


def chunk_ranges(rows: int, world_size: int, chunks: int) -> list[tuple[int, int]]:
    """Split rows evenly in TP-sized units; every A2A has equal-size splits."""
    if world_size < 1 or rows % world_size or not 1 <= chunks <= rows // world_size:
        raise ValueError("Rows must be divisible by TP and contain at least one TP-sized unit per chunk.")
    units, remainder = divmod(rows // world_size, chunks)
    ranges = []
    start = 0
    for index in range(chunks):
        end = start + (units + int(index < remainder)) * world_size
        ranges.append((start, end))
        start = end
    return ranges


class _Exchange(NamedTuple):
    start: int
    end: int
    send: torch.Tensor
    received: torch.Tensor
    scale_dtype: torch.dtype
    work: Any


class ChunkedMxfp8AllToAllAllReduce(base.Mxfp8AllToAllAllReduce):
    """Two-chunk lookahead on one HCCL group, with explicit Work dependencies.

    HCCL uses its own communication stream. Quantization and local reduction use
    the caller's compute stream. Enqueue Q1/A1 before waiting for A0, so Q1 may
    overlap A0. G0 may overlap Q2/local work. Collectives themselves stay ordered
    on the same group. Actual device overlap must be verified in a trace.
    """

    def __init__(self, *args, chunks: int, pipelined: bool, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if chunks < 2:
            raise ValueError("Use at least two chunks; use the original benchmark for the unchunked control.")
        if self.dequant_dtype != torch.bfloat16:
            raise ValueError("This pipeline prototype uses BF16 dequantization and SUM output.")
        self.chunks = chunks
        self.pipelined = pipelined

    def _partial(self, received: torch.Tensor, scale_dtype: torch.dtype, hidden: int) -> torch.Tensor:
        values = received[:, :hidden].contiguous().view(torch.float8_e4m3fn)
        scales = received[:, hidden:].contiguous().view(scale_dtype)
        decoded = self.dequantize(values, scales)
        return decoded.reshape(self.world_size, -1, hidden).sum(dim=0, dtype=torch.bfloat16)

    def _run(self, output: torch.Tensor, prepare: Callable) -> torch.Tensor:
        ranges = iter(chunk_ranges(output.shape[0], self.world_size, self.chunks))

        def submit(start: int, end: int) -> _Exchange:
            values, scales = prepare(start, end)
            send, scale_dtype = self.pack_payload(values, scales)
            received = torch.empty_like(send)
            work = dist.all_to_all_single(received, send, group=self.group, async_op=self.pipelined)
            return _Exchange(start, end, send, received, scale_dtype, work)

        if not self.pipelined:
            for start, end in ranges:
                exchange = submit(start, end)
                partial = self._partial(exchange.received, exchange.scale_dtype, output.shape[1])
                dist.all_gather_into_tensor(output[start:end], partial, group=self.group)
                del exchange, partial
            return output

        pending = deque()
        gathers = []
        for _ in range(min(_LOOKAHEAD, self.chunks)):
            pending.append(submit(*next(ranges)))
        while pending:
            exchange = pending.popleft()
            # Join only after the next chunk's quantization/A2A is submitted.
            exchange.work.wait()
            partial = self._partial(exchange.received, exchange.scale_dtype, output.shape[1])
            target = output[exchange.start : exchange.end]
            work = dist.all_gather_into_tensor(target, partial, group=self.group, async_op=True)
            # Keep both tensors alive until the async gather has been joined.
            gathers.append((partial, target, work))
            del exchange
            next_range = next(ranges, None)
            if next_range is not None:
                pending.append(submit(*next_range))
        for partial, target, work in gathers:
            work.wait()
        # Every gather is joined onto the calling stream before its end event.
        return output

    def reduce_quantized(self, values, scales, output, markers=None) -> torch.Tensor:
        if markers is not None:
            raise ValueError("Serial stage markers cannot measure an overlapping pipeline; use --profile-dir.")
        row_scales = scales.reshape(values.shape[0], -1)
        return self._run(output, lambda start, end: (values[start:end], row_scales[start:end]))

    def __call__(self, x: torch.Tensor, markers=None) -> torch.Tensor:
        if markers is not None:
            raise ValueError("Serial stage markers cannot measure an overlapping pipeline; use --profile-dir.")
        return self._run(x, lambda start, end: self.quantize(x[start:end]))


def _check_encoded(pipeline, rank: int, rows: int, hidden: int, device: torch.device) -> None:
    # Distinct rank/token/block bytes detect owner routing and chunk-offset errors.
    indices = torch.arange(rows * hidden, device=device).reshape(rows, hidden)
    values = ((indices + rank) % 8 + 0x38).to(torch.uint8).view(torch.float8_e4m3fn)
    scale_indices = torch.arange(rows * hidden // base._MX_BLOCK, device=device).reshape(rows, -1)
    raw_scales = ((scale_indices + rank) % 7 + 124).to(torch.uint8)
    _, template = pipeline.quantize(torch.zeros(rows, hidden, dtype=torch.bfloat16, device=device))
    scales = raw_scales.view(template.dtype)
    _, reference, contribution = base._reference_sum(values, scales)
    output = torch.empty(rows, hidden, dtype=torch.bfloat16, device=device)
    pipeline.reduce_quantized(values, scales, output)
    base._check_output(output, reference, contribution, torch.bfloat16, "rank/token/block/chunk routing")


def _check_chunk_quantization(pipeline, source: torch.Tensor, chunks: int) -> dict:
    values, scales = pipeline.quantize(source)
    row_scales = scales.contiguous().view(torch.uint8).reshape(source.shape[0], -1)
    matches = True
    for start, end in chunk_ranges(source.shape[0], pipeline.world_size, chunks):
        chunk_values, chunk_scales = pipeline.quantize(source[start:end])
        matches &= torch.equal(chunk_values.view(torch.uint8), values[start:end].view(torch.uint8))
        matches &= torch.equal(
            chunk_scales.contiguous().view(torch.uint8).reshape(end - start, -1), row_scales[start:end]
        )
    base._require_all(matches, "Chunked quantization bytes differ from the unchunked control.", source.device)
    return {"pass": True, "same_quantized_bytes": True}


def _benchmark_case(variants: dict, source: torch.Tensor, args: argparse.Namespace) -> dict:
    case = {
        "shape": list(source.shape),
        "bf16_payload_mib": source.numel() * source.element_size() / base._MIB,
        "chunk_rows": {
            str(chunks): [end - start for start, end in chunk_ranges(source.shape[0], dist.get_world_size(), chunks)]
            for chunks in args.chunks
        },
        "chunk_quantization": {
            str(chunks): _check_chunk_quantization(variants["mxfp8_full"], source, chunks) for chunks in args.chunks
        },
        "accuracy": {name: base._check(op, source) for name, op in variants.items() if name != "bf16"},
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
        # All ranks use the same rotating order; reverse every other round.
        offset = repeat % len(names)
        order = names[offset:] + names[:offset]
        if repeat % 2:
            order.reverse()
        for name in order:
            measurements[name].append(base._measure(variants[name], source, args.iterations))
    case["measurements"] = {}
    for name, rounds in measurements.items():
        samples = [sample for item in rounds for sample in item["samples_ms"]]
        case["measurements"][name] = {
            **base.summarize(samples),
            "round_medians_ms": [statistics.median(item["samples_ms"]) for item in rounds],
            "samples_ms": samples,
            "median_wall_ms_including_reset": statistics.median(item["wall_ms_including_reset"] for item in rounds),
            "extra_peak_allocated_mib": max(item["extra_peak_allocated_mib"] for item in rounds),
        }
    times = case["measurements"]
    if any(item["median_ms"] <= 0 for item in times.values()):
        raise RuntimeError("Nonpositive NPU event duration; increase payload or verify event timing.")
    for name, item in times.items():
        item["speedup_vs_bf16"] = times["bf16"]["median_ms"] / item["median_ms"]
        item["speedup_vs_full"] = times["mxfp8_full"]["median_ms"] / item["median_ms"]
        if name.startswith("pipeline_chunks_"):
            serial = name.replace("pipeline_", "serial_", 1)
            item["speedup_vs_serial_chunks"] = times[serial]["median_ms"] / item["median_ms"]
    return case


def format_report(report: dict) -> str:
    lines = [f"MXFP8 pipeline | TP={report['world_size']} | device={report['device']} | {report['mode']}"]
    for case in report["cases"]:
        lines.append(
            f"\nShape={case['shape']}; BF16={case['bf16_payload_mib']:.2f} MiB; chunk rows={case['chunk_rows']}"
        )
        lines.append("variant | relative L2 / FP32 SUM | max abs / FP32 SUM | same-payload max abs")
        for name, accuracy in case["accuracy"].items():
            lines.append(
                f"{name} | {accuracy['mxfp8_relative_l2_vs_fp32']:.6g} | "
                f"{accuracy['mxfp8_max_abs_vs_fp32']:.6g} | "
                f"{accuracy['same_payload_collective']['max_abs_vs_dequantized_sum']:.6g}"
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
    parser.add_argument("--tokens", nargs="+", type=int, default=[16384, 32768])
    parser.add_argument("--hidden-sizes", nargs="+", type=int, default=[5120])
    parser.add_argument("--chunks", nargs="+", type=int, default=[2, 4])
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--timeout-seconds", type=int, default=2400)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--report-json", type=Path)
    parser.add_argument("--profile-dir", type=Path)
    parser.add_argument("--profile-steps", type=int, default=2)
    args = parser.parse_args(argv)
    if any(value < 1 for value in args.tokens) or any(value < 2 for value in args.chunks):
        parser.error("Tokens must be positive and chunk counts must be at least two.")
    if len(set(args.chunks)) != len(args.chunks):
        parser.error("Chunk counts must be distinct.")
    if any(value < base._HIDDEN_ALIGNMENT or value % base._HIDDEN_ALIGNMENT for value in args.hidden_sizes):
        parser.error("Hidden sizes must be positive multiples of 64.")
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
    for rows in args.tokens:
        for chunks in args.chunks:
            chunk_ranges(rows, world_size, chunks)
    torch_npu = importlib.import_module("torch_npu")
    if not torch.npu.is_available():
        raise RuntimeError("This benchmark requires Ascend NPUs.")
    torch.npu.set_device(int(os.environ["LOCAL_RANK"]))
    quantize = getattr(torch_npu, "npu_dynamic_mx_quant", None)
    dequantize = getattr(torch_npu, "npu_anti_mx_quant", None)
    if not callable(dequantize):
        dequantize = getattr(torch.ops.npu, "npu_anti_mx_quant", None)
    variants = {"bf16": dist.all_reduce, "mxfp8_full": base.Mxfp8AllToAllAllReduce(world_size, quantize, dequantize)}
    for chunks in args.chunks:
        for pipelined, name in ((False, "serial"), (True, "pipeline")):
            variants[f"{name}_chunks_{chunks}"] = ChunkedMxfp8AllToAllAllReduce(
                world_size, quantize, dequantize, chunks=chunks, pipelined=pipelined
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
            "dequant_dtype": "bf16",
            "chunks": args.chunks,
            "lookahead": _LOOKAHEAD,
            "warmup": args.warmup,
            "iterations_per_round": args.iterations,
            "rounds": args.rounds,
            "seed": args.seed,
            "timing": "NPU events including final gather joins; reset excluded; slowest rank per sample; Eager",
            "memory_scope": "PyTorch allocator peak; excludes reserved memory and external/persistent HCCL resources",
            "communication_domains": "One default HCCL group for every collective; no additional groups or streams",
            # External launcher/HCCL settings: keep this standalone benchmark independent of vLLM initialization.
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
        # Include uneven chunk tails in sanity tests (in TP-sized row units).
        sanity_rows = (2 * max(args.chunks) + 1) * world_size
        hidden = args.hidden_sizes[0]
        for name, value in (("zeros", 0.0), ("rank_constant", rank + 1.0), ("cancellation", (-1.0) ** rank)):
            source = torch.full((sanity_rows, hidden), value, dtype=torch.bfloat16, device=device)
            for variant, op in variants.items():
                if variant != "bf16":
                    base._check(op, source, exact=True)
            report["sanity_checks"].append({"case": name, "pass": True})
            if rank == 0:
                print(f"Correctness {name}: PASS", flush=True)
        for name, op in variants.items():
            if name != "bf16":
                _check_encoded(op, rank, sanity_rows, hidden, device)
        report["sanity_checks"].append({"case": "rank_token_block_chunk_routing", "pass": True})
        if rank == 0:
            print("Correctness rank_token_block_chunk_routing: PASS", flush=True)
        del source
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
            print("PASS: all requested MXFP8 pipeline checks passed.", flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
