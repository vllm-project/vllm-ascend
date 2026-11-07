# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Standalone eager INT8 all-reduce correctness test and Ascend microbenchmark.

Only torch and torch_npu are required; no vLLM imports or C++ builds.
Run with torchrun; see benchmark_int8_all_reduce.md for measurement details.
"""

import argparse
import json
import math
import os
import socket
import statistics
import time
import unicodedata
from collections.abc import Callable
from datetime import timedelta
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

_INT8_MAX = 127
_MIN_SCALE = 1e-8
_SUPPORTED_WORLD_SIZES = (2, 4, 8)
_MIB = 1024 * 1024
_STAGES = ("local_amax", "amax_all_reduce", "scale_and_quantize", "int8_all_reduce", "dequantize_and_copy")


class QuantizationOps:
    def __init__(self, quant_backend: str, dequant_backend: str, torch_npu: Any = None):
        self.quant_backend = quant_backend
        self.dequant_backend = dequant_backend
        self.quantize_op = getattr(torch_npu, "npu_quantize", None)
        self.dequantize_op = getattr(torch_npu, "npu_anti_quant", None)
        if quant_backend == "npu" and not callable(self.quantize_op):
            raise RuntimeError(
                "npu_quantize is unavailable. Use --quant-backend torch to test a different implementation."
            )
        if dequant_backend == "npu" and not callable(self.dequantize_op):
            raise RuntimeError(
                "npu_anti_quant is unavailable. Use --dequant-backend torch to test a different implementation."
            )

    def quantize(self, x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        if self.quant_backend == "npu":
            return self.quantize_op(x, scale, None, torch.int8, 1, True)
        return (x.float() / scale).round().to(torch.int8)

    def dequantize(self, q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        if self.dequant_backend == "npu":
            return self.dequantize_op(q, scale, dst_dtype=torch.bfloat16)
        return (q.float() * scale).to(torch.bfloat16)


class Int8AllReduce:
    """Shared per-channel scale, integer-safe SUM, BF16 in-place result."""

    def __init__(self, world_size: int, ops: QuantizationOps, amax_method: str = "norm", group=None):
        if world_size not in _SUPPORTED_WORLD_SIZES:
            raise ValueError("Use TP=2, 4 or 8 for this benchmark.")
        self.world_size, self.ops, self.group = world_size, ops, group
        self.max_q = _INT8_MAX // world_size
        self.amax_method = amax_method

    def local_amax(self, x: torch.Tensor) -> torch.Tensor:
        if self.amax_method == "norm":
            return torch.linalg.vector_norm(x, ord=math.inf, dim=0).float()
        return x.abs().amax(dim=0).float()

    def global_scale(self, x: torch.Tensor) -> torch.Tensor:
        amax = self.local_amax(x)
        dist.all_reduce(amax, op=dist.ReduceOp.MAX, group=self.group)
        return (amax / self.max_q).clamp_min(_MIN_SCALE)

    def __call__(self, x: torch.Tensor, markers: list | None = None) -> torch.Tensor:
        # Markers are only used for a separate diagnostic run, never for the
        # primary end-to-end timing. No device-to-host reads occur here.
        if markers is not None:
            markers[0].record()
        amax = self.local_amax(x)
        if markers is not None:
            markers[1].record()
        dist.all_reduce(amax, op=dist.ReduceOp.MAX, group=self.group)
        if markers is not None:
            markers[2].record()
        scale = (amax / self.max_q).clamp_min(_MIN_SCALE)
        q = self.ops.quantize(x, scale)
        if markers is not None:
            markers[3].record()
        dist.all_reduce(q, group=self.group)
        if markers is not None:
            markers[4].record()
        x.copy_(self.ops.dequantize(q, scale))
        if markers is not None:
            markers[5].record()
        return x


def summarize(samples: list[float]) -> dict:
    ordered = sorted(samples)
    if not ordered:
        raise ValueError("Cannot summarize an empty measurement.")
    return {
        "median_ms": statistics.median(ordered),
        "mean_ms": statistics.mean(ordered),
        "p95_ms": ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)],
        "min_ms": ordered[0],
        "max_ms": ordered[-1],
    }


def _max_across_ranks(values: list, device: torch.device) -> list:
    tensor = torch.tensor(values, dtype=torch.float32, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return tensor.cpu().tolist()


def _check(pipeline: Int8AllReduce, x: torch.Tensor) -> dict:
    """Validate INT8 SUM against INT32 SUM before judging quantization error."""
    scale = pipeline.global_scale(x)
    q = pipeline.ops.quantize(x, scale)
    reference_sum = q.to(torch.int32)
    dist.all_reduce(reference_sum, group=pipeline.group)
    actual_q = q.clone()
    dist.all_reduce(actual_q, group=pipeline.group)
    flags = [
        q.dtype != torch.int8,
        bool((q.float().abs().max() > pipeline.max_q).item()),
        not torch.equal(actual_q.to(torch.int32), reference_sum),
    ]
    failed = _max_across_ranks([int(flag) for flag in flags], x.device)
    if any(failed):
        raise AssertionError(f"Quantized dtype/range/INT8 SUM check failed: {failed}")
    expected = (reference_sum.float() * scale).to(torch.bfloat16)
    actual = x.clone()
    pipeline(actual)
    # Native dequantization may differ by one BF16 rounding step. The exact
    # integer comparison above catches overflow without a floating tolerance.
    correct = torch.allclose(actual, expected, rtol=0.008, atol=1e-7)
    if _max_across_ranks([int(not correct)], x.device)[0]:
        raise AssertionError("End-to-end result differs from INT32 SUM + independent FP32 dequantization.")
    del q, reference_sum, actual_q, expected, scale
    reference = x.float()
    dist.all_reduce(reference, group=pipeline.group)
    bf16 = x.clone()
    dist.all_reduce(bf16, group=pipeline.group)
    quant_error = actual.float() - reference
    bf16_error = bf16.float() - reference
    denominator = reference.norm().clamp_min(1e-12)
    errors = torch.stack((quant_error.norm() / denominator, quant_error.abs().max(), bf16_error.norm() / denominator))
    dist.all_reduce(errors, op=dist.ReduceOp.MAX)
    values = errors.cpu().tolist()
    return {
        "int8_relative_l2_vs_fp32": values[0],
        "int8_max_abs_vs_fp32": values[1],
        "bf16_relative_l2_vs_fp32": values[2],
    }


def _measure(op: Callable, source: torch.Tensor, iterations: int) -> dict:
    working = torch.empty_like(source)
    events = [(torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)) for _ in range(iterations)]
    torch.npu.synchronize()
    dist.barrier()
    torch.npu.synchronize()
    torch.npu.reset_peak_memory_stats()
    allocated = torch.npu.memory_allocated()
    start_wall = time.perf_counter()
    for start, end in events:
        # Reset outside the event interval. Both candidates mutate the input,
        # so omitting this reset would quickly cause repeated-SUM overflow.
        working.copy_(source)
        start.record()
        op(working)
        end.record()
    torch.npu.synchronize()
    wall_ms = (time.perf_counter() - start_wall) * 1000 / iterations
    extra_peak_mib = (torch.npu.max_memory_allocated() - allocated) / _MIB
    samples = _max_across_ranks([start.elapsed_time(end) for start, end in events], source.device)
    wall_ms, extra_peak_mib = _max_across_ranks([wall_ms, extra_peak_mib], source.device)
    return {"samples_ms": samples, "wall_ms_including_reset": wall_ms, "extra_peak_allocated_mib": extra_peak_mib}


def _breakdown(pipeline: Int8AllReduce, source: torch.Tensor, iterations: int) -> dict:
    working = torch.empty_like(source)
    markers = [[torch.npu.Event(enable_timing=True) for _ in range(len(_STAGES) + 1)] for _ in range(iterations)]
    torch.npu.synchronize()
    dist.barrier()
    for events in markers:
        working.copy_(source)
        pipeline(working, markers=events)
    torch.npu.synchronize()
    samples = [[events[i].elapsed_time(events[i + 1]) for i in range(len(_STAGES))] for events in markers]
    samples = _max_across_ranks(samples, source.device)
    return {name: summarize([row[index] for row in samples]) for index, name in enumerate(_STAGES)}


def _benchmark_case(pipeline: Int8AllReduce, source: torch.Tensor, args: argparse.Namespace) -> dict:
    accuracy = _check(pipeline, source)
    working = torch.empty_like(source)
    for _ in range(args.warmup):
        working.copy_(source)
        dist.all_reduce(working)
        working.copy_(source)
        pipeline(working)
    torch.npu.synchronize()
    del working
    results = {"bf16": [], "int8": []}
    for repeat in range(args.rounds):
        # Alternate measurement order so one candidate does not always run
        # earlier, colder, or at a different clock/thermal state.
        order = ("bf16", "int8") if repeat % 2 == 0 else ("int8", "bf16")
        for name in order:
            op = dist.all_reduce if name == "bf16" else pipeline
            results[name].append(_measure(op, source, args.iterations))
    case = {"shape": list(source.shape), "bf16_payload_mib": source.numel() * source.element_size() / _MIB}
    for name, rounds in results.items():
        samples = [sample for measurement in rounds for sample in measurement["samples_ms"]]
        case[name] = {
            **summarize(samples),
            "round_medians_ms": [summarize(measurement["samples_ms"])["median_ms"] for measurement in rounds],
            "median_wall_ms_including_reset": statistics.median(
                measurement["wall_ms_including_reset"] for measurement in rounds
            ),
            "max_wall_ms_including_reset": max(measurement["wall_ms_including_reset"] for measurement in rounds),
            "extra_peak_allocated_mib": max(measurement["extra_peak_allocated_mib"] for measurement in rounds),
            "samples_ms": samples,
        }
    if min(case["bf16"]["median_ms"], case["int8"]["median_ms"]) <= 0:
        raise RuntimeError("NPU events reported a nonpositive duration; increase the payload and verify event timing.")
    case["speedup"] = case["bf16"]["median_ms"] / case["int8"]["median_ms"]
    case["wall_speedup_including_reset"] = (
        case["bf16"]["median_wall_ms_including_reset"] / case["int8"]["median_wall_ms_including_reset"]
    )
    case["latency_reduction_pct"] = 100 * (1 - case["int8"]["median_ms"] / case["bf16"]["median_ms"])
    case["accuracy"] = accuracy
    if args.breakdown:
        case["stage_diagnostics"] = _breakdown(pipeline, source, args.iterations)
    return case


def _table(headers: list[str], rows: list[list[str]]) -> str:
    """Render aligned Markdown columns, accounting for Chinese display width."""

    def width(value):
        return sum(2 if unicodedata.east_asian_width(char) in ("W", "F") else 1 for char in value)

    widths = [max(width(row[index]) for row in [headers, *rows]) for index in range(len(headers))]

    def line(row):
        return "| " + " | ".join(value + " " * (size - width(value)) for value, size in zip(row, widths)) + " |"

    return "\n".join([line(headers), line(["-" * size for size in widths]), *(line(row) for row in rows)])


def format_report(report: dict, breakdown: bool = False) -> str:
    groups: dict[int, list] = {}
    for case in report["cases"]:
        groups.setdefault(case["shape"][1], []).append(case)
    lines = ["", f"INT8 all-reduce 测试结果 | TP={report['world_size']} | device={report['device']}"]
    lines.append("耗时为完整路径的中位数；输入重置不计入；每个样本取最慢 rank。")
    for hidden, cases in groups.items():
        lines.extend(["", f"hidden_size = {hidden}", ""])
        rows = []
        for case in cases:
            gain = case["latency_reduction_pct"]
            change = f"{'下降' if gain > 0 else '增加'} {abs(gain):.2f}%" if gain != 0 else "持平"
            rows.append(
                [
                    str(case["shape"][0]),
                    f"{case['bf16_payload_mib']:.2f}",
                    f"{case['bf16']['median_ms']:.4f}",
                    f"{case['int8']['median_ms']:.4f}",
                    f"{case['speedup']:.3f}x",
                    change,
                ]
            )
        lines.append(
            _table(["Tokens", "BF16 通信量(MiB)", "BF16 耗时(ms)", "INT8 完整耗时(ms)", "加速比", "延迟变化"], rows)
        )
        lines.extend(["", "误差与额外内存（误差相对 FP32 SUM）：", ""])
        lines.append(
            _table(
                ["Tokens", "相对 L2 误差", "最大绝对误差", "INT8 额外峰值(MiB)"],
                [
                    [
                        str(case["shape"][0]),
                        f"{case['accuracy']['int8_relative_l2_vs_fp32'] * 100:.3f}%",
                        f"{case['accuracy']['int8_max_abs_vs_fp32']:.6f}",
                        f"{case['int8']['extra_peak_allocated_mib']:.2f}",
                    ]
                    for case in cases
                ],
            )
        )
    if breakdown:
        lines.extend(["", "阶段耗时诊断（ms；独立测量，各列中位数之和不等于完整耗时）："])
        for hidden, cases in groups.items():
            diagnostic_cases = [case for case in cases if "stage_diagnostics" in case]
            if not diagnostic_cases:
                continue
            lines.extend(["", f"hidden_size = {hidden}", ""])
            lines.append(
                _table(
                    ["Tokens", "本地 amax", "amax 同步", "尺度+量化", "INT8 通信", "反量化+写回"],
                    [
                        [str(case["shape"][0])]
                        + [f"{case['stage_diagnostics'][stage]['median_ms']:.4f}" for stage in _STAGES]
                        for case in diagnostic_cases
                    ],
                )
            )
    return "\n".join(lines)


def _parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", nargs="+", type=int, default=[512, 2048, 8192, 16384])
    parser.add_argument("--hidden-sizes", nargs="+", type=int, default=[2048, 5120])
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--quant-backend", choices=("npu", "torch"), default="npu")
    parser.add_argument("--dequant-backend", choices=("npu", "torch"), default="npu")
    parser.add_argument("--amax-method", choices=("norm", "abs"), default="norm")
    parser.add_argument("--breakdown", action="store_true")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--timeout-seconds", type=int, default=300)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--report-json", type=Path, help="Print tables from an existing result JSON without running NPUs."
    )
    args = parser.parse_args(argv)
    if (
        any(value < 1 for value in (*args.tokens, *args.hidden_sizes))
        or args.warmup < 0
        or args.iterations < 1
        or args.rounds < 1
        or args.timeout_seconds < 1
    ):
        parser.error("Shapes, iterations, rounds and timeout must be positive; warmup must be nonnegative.")
    return args


def main() -> None:
    args = _parse_args()
    if args.report_json:
        print(format_report(json.loads(args.report_json.read_text(encoding="utf-8")), args.breakdown), flush=True)
        return
    # Load the backend in the NPU worker only: --help and CPU unit tests do
    # not need an Ascend installation.
    import torch_npu

    if not torch.npu.is_available():
        raise RuntimeError("This benchmark requires Ascend NPUs and torch_npu.")
    torch.npu.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("hccl", timeout=timedelta(seconds=args.timeout_seconds))
    try:
        rank, world_size = dist.get_rank(), dist.get_world_size()
        pipeline = Int8AllReduce(
            world_size, QuantizationOps(args.quant_backend, args.dequant_backend, torch_npu), args.amax_method
        )
        device = torch.device(f"npu:{torch.npu.current_device()}")
        report = {
            "torch": torch.__version__,
            "torch_npu": torch_npu.__version__,
            "device": torch.npu.get_device_name(),
            "hostname_rank0": socket.gethostname(),
            "world_size": world_size,
            "max_q": pipeline.max_q,
            "quant_backend": args.quant_backend,
            "dequant_backend": args.dequant_backend,
            "amax_method": args.amax_method,
            "warmup": args.warmup,
            "iterations_per_round": args.iterations,
            "rounds": args.rounds,
            "timing": "NPU events, input reset excluded, slowest rank per sample; Eager only",
            "cases": [],
        }
        if rank == 0:
            print(
                f"device={report['device']} | TP={world_size} | max_q={pipeline.max_q}\n"
                f"torch={report['torch']} | torch_npu={report['torch_npu']} | "
                f"quant={args.quant_backend} | dequant={args.dequant_backend} | amax={args.amax_method}\n"
                f"warmup={args.warmup} | iterations={args.iterations} | rounds={args.rounds}",
                flush=True,
            )
        for name, value in (("zeros", 0.0), ("same_sign_overflow_boundary", 1.0)):
            _check(pipeline, torch.full((64, 128), value, dtype=torch.bfloat16, device=device))
            if rank == 0:
                print(f"Correctness {name}: PASS", flush=True)
        total_cases = len(args.hidden_sizes) * len(args.tokens)
        for hidden in args.hidden_sizes:
            for tokens in args.tokens:
                torch.manual_seed(args.seed + rank)
                source = torch.randn(tokens, hidden, dtype=torch.bfloat16, device=device)
                case = _benchmark_case(pipeline, source, args)
                report["cases"].append(case)
                del source
                if rank == 0:
                    print(
                        f"完成 {len(report['cases'])}/{total_cases}：hidden_size={hidden}, tokens={tokens}",
                        flush=True,
                    )
                    if args.output:
                        args.output.parent.mkdir(parents=True, exist_ok=True)
                        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        if rank == 0:
            print(format_report(report, args.breakdown), flush=True)
            if args.output:
                print(f"\n完整结果已保存：{args.output}", flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()