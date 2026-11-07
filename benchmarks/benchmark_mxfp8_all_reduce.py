# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Standalone MXFP8 AllToAll/AllGather all-reduce benchmark for Atlas A5.

The candidate path uses only ordinary HCCL collectives:
BF16 -> local MXFP8 -> packed AllToAll -> BF16 dequantize and local SUM ->
BF16 AllGather. Use --dequant-dtype fp32 for the original FP32 staging path.
Run with torchrun; no vLLM or ops-transformer import is needed.
"""

import argparse
import importlib
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

_SUPPORTED_WORLD_SIZES = (2, 4, 8)
_MX_BLOCK = 32
_SCALE_PAIR = 2
_HIDDEN_ALIGNMENT = _MX_BLOCK * _SCALE_PAIR
_MIB = 1024 * 1024
_SANITY_TOKENS_PER_RANK = 2
_BFLOAT16_TYPE = 15
_FLOAT32_TYPE = 6
_FLOAT8_E4M3FN_TYPE = 292
_BF16_RTOL = 2**-7
_FP32_RTOL = 2e-5
_CHECK_ATOL = 1e-6
_STAGES = ("dynamic_mx_quant", "pack_payload", "all_to_all", "dequantize_and_sum", "bf16_all_gather")


class Mxfp8AllToAllAllReduce:
    """Approximate SUM with ordinary collectives and an in-place BF16 result."""

    def __init__(
        self,
        world_size: int,
        quantize_op: Callable,
        dequantize_op: Callable,
        group: Any = None,
        *,
        dequant_dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        if world_size not in _SUPPORTED_WORLD_SIZES:
            raise ValueError("Use TP=2, 4 or 8 for this benchmark.")
        if not callable(quantize_op) or not callable(dequantize_op):
            raise RuntimeError("Both npu_dynamic_mx_quant and npu_anti_mx_quant must be callable.")
        if dequant_dtype not in (torch.bfloat16, torch.float32):
            raise ValueError("Dequantization dtype must be torch.bfloat16 or torch.float32.")
        self.world_size = world_size
        self.quantize_op = quantize_op
        self.dequantize_op = dequantize_op
        self.group = group
        self.dequant_dtype = dequant_dtype

    def quantize(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        values, scales = self.quantize_op(
            x,
            axis=-1,
            dst_type=torch.float8_e4m3fn,
            block_size=_MX_BLOCK,
            round_mode="rint",
        )
        expected_scales = x.numel() // _MX_BLOCK
        valid = (
            values.dtype == torch.float8_e4m3fn
            and values.shape == x.shape
            and scales.element_size() == 1
            and scales.numel() == expected_scales
        )
        if not valid:
            raise RuntimeError(
                "Unexpected dynamic MX quantization result: "
                f"values={values.shape}/{values.dtype}, scales={scales.shape}/{scales.dtype}."
            )
        return values, scales

    def dequantize(self, values: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
        rows, hidden = values.shape
        paired_scales = scales.reshape(rows, hidden // (_MX_BLOCK * _SCALE_PAIR), _SCALE_PAIR)
        return self.dequantize_op(
            values,
            paired_scales,
            axis=-1,
            dst_type=_BFLOAT16_TYPE if self.dequant_dtype == torch.bfloat16 else _FLOAT32_TYPE,
            src_type=_FLOAT8_E4M3FN_TYPE,
        )

    @staticmethod
    def pack_payload(values: torch.Tensor, scales: torch.Tensor) -> tuple[torch.Tensor, torch.dtype]:
        rows = values.shape[0]
        scale_dtype = scales.dtype
        scale_bytes = scales.contiguous().view(torch.uint8).reshape(rows, -1)
        send = torch.cat((values.contiguous().view(torch.uint8), scale_bytes), dim=-1)
        return send, scale_dtype

    def exchange_payload(
        self,
        send: torch.Tensor,
        markers: list | None = None,
    ) -> torch.Tensor:
        received = torch.empty_like(send)
        dist.all_to_all_single(received, send, group=self.group)
        if markers is not None:
            markers[3].record()
        return received

    def finish_received(
        self,
        received: torch.Tensor,
        scale_dtype: torch.dtype,
        output: torch.Tensor,
        markers: list | None = None,
    ) -> torch.Tensor:
        rows, hidden = output.shape
        received_values = received[:, :hidden].contiguous().view(torch.float8_e4m3fn)
        received_scales = received[:, hidden:].contiguous().view(scale_dtype)
        dequantized = self.dequantize(received_values, received_scales)
        shard_rows = rows // self.world_size
        # Keep the sum's output in the selected staging dtype. In BF16 mode,
        # do not explicitly cast the full received payload back to FP32.
        # Internal accumulation precision/workspace remains backend-defined.
        partial = (
            dequantized.reshape(self.world_size, shard_rows, hidden)
            .sum(dim=0, dtype=self.dequant_dtype)
            .to(torch.bfloat16)
        )
        del received, received_values, received_scales, dequantized
        if markers is not None:
            markers[4].record()
        dist.all_gather_into_tensor(output.view(rows, hidden), partial, group=self.group)
        if markers is not None:
            markers[5].record()
        return output

    def reduce_quantized(
        self,
        values: torch.Tensor,
        scales: torch.Tensor,
        output: torch.Tensor,
        markers: list | None = None,
    ) -> torch.Tensor:
        rows = values.shape[0]
        if rows % self.world_size:
            raise ValueError("The row count must be divisible by TP for AllToAll sharding.")
        if markers is not None:
            markers[1].record()
        send, scale_dtype = self.pack_payload(values, scales)
        if markers is not None:
            markers[2].record()
        received = self.exchange_payload(send, markers)
        del send
        return self.finish_received(received, scale_dtype, output, markers)

    def __call__(self, x: torch.Tensor, markers: list | None = None) -> torch.Tensor:
        if markers is not None:
            markers[0].record()
        values, scales = self.quantize(x)
        if markers is not None:
            markers[1].record()
        send, scale_dtype = self.pack_payload(values, scales)
        del values, scales
        if markers is not None:
            markers[2].record()
        received = self.exchange_payload(send, markers)
        del send
        return self.finish_received(received, scale_dtype, x, markers)


def decode_mxfp8_cpu(values: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """Decode E4M3FN and E8M0 raw bytes without using an NPU anti-quant op."""
    raw_values = values.contiguous().view(torch.uint8).cpu()
    raw_scales = scales.contiguous().view(torch.uint8).cpu()
    value_table = []
    for code in range(256):
        sign = -1.0 if code & 0x80 else 1.0
        exponent, mantissa = (code >> 3) & 0x0F, code & 0x07
        if exponent == 0x0F and mantissa == 0x07:
            value = math.nan
        elif exponent == 0:
            value = sign * math.ldexp(float(mantissa), -9)
        else:
            value = sign * math.ldexp(1.0 + mantissa / 8, exponent - 7)
        value_table.append(value)
    value_table_tensor = torch.tensor(value_table, dtype=torch.float32)
    scale_table = torch.tensor([math.ldexp(1.0, code - 127) for code in range(255)] + [math.nan])
    rows, hidden = values.shape
    decoded = value_table_tensor[raw_values.long()].reshape(rows, hidden // _MX_BLOCK, _MX_BLOCK)
    factors = scale_table[raw_scales.long().reshape(rows, hidden // _MX_BLOCK)].unsqueeze(-1)
    return (decoded * factors).reshape(rows, hidden)


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


def _require_all(condition: bool, message: str, device: torch.device) -> None:
    if _max_across_ranks([int(not condition)], device)[0]:
        raise AssertionError(message)


def _reference_sum(values: torch.Tensor, scales: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    decoded_cpu = decode_mxfp8_cpu(values, scales)
    _require_all(bool(torch.isfinite(decoded_cpu).all()), "Decoded MXFP8 data contains NaN/Inf.", values.device)
    decoded = decoded_cpu.to(values.device)
    reference = decoded.clone()
    contribution = decoded.abs()
    dist.all_reduce(reference)
    dist.all_reduce(contribution)
    return decoded, reference, contribution


def _check_output(
    actual: torch.Tensor,
    reference: torch.Tensor,
    contribution: torch.Tensor,
    dtype: torch.dtype,
    label: str,
    *,
    exact: bool = False,
) -> dict:
    metadata_ok = actual.shape == reference.shape and actual.dtype == dtype and actual.device == reference.device
    _require_all(metadata_ok, f"Unexpected {label} shape/dtype/device.", reference.device)
    rounded_reference = reference.to(dtype).float()
    difference = (actual.float() - rounded_reference).abs()
    rtol = _BF16_RTOL if dtype == torch.bfloat16 else _FP32_RTOL
    tolerance = torch.zeros_like(reference) if exact else _CHECK_ATOL + rtol * contribution
    bad = (~torch.isfinite(actual)) | (difference > tolerance)
    stats = _max_across_ranks(
        [int(bad.any().item()), difference.nan_to_num(nan=math.inf).max().item()], reference.device
    )
    if stats[0]:
        raise AssertionError(f"{label} check failed; max_abs={stats[1]}")
    return {"pass": True, "max_abs_vs_dequantized_sum": stats[1]}


def _check(pipeline: Mxfp8AllToAllAllReduce, source: torch.Tensor, *, exact: bool = False) -> dict:
    """Validate local decode, the same payload collective, then the full path."""
    values, scales = pipeline.quantize(source)
    decoded, reference, contribution = _reference_sum(values, scales)
    if exact:
        _require_all(
            torch.equal(decoded.cpu(), source.float().cpu()),
            "Sanity input did not quantize exactly.",
            source.device,
        )

    npu_decoded = pipeline.dequantize(values, scales)
    local_accuracy = _check_output(
        npu_decoded,
        decoded,
        decoded.abs(),
        pipeline.dequant_dtype,
        "npu_anti_mx_quant",
        exact=exact,
    )
    same_payload = torch.empty_like(source)
    pipeline.reduce_quantized(values, scales, same_payload)
    collective_accuracy = _check_output(
        same_payload,
        reference,
        contribution,
        torch.bfloat16,
        "MXFP8 AllToAll/AllGather",
        exact=exact,
    )
    actual = source.clone()
    pipeline(actual)
    full_path_accuracy = _check_output(
        actual,
        reference,
        contribution,
        torch.bfloat16,
        "full MXFP8 AllToAll/AllGather path",
        exact=exact,
    )

    original_sum = source.float()
    dist.all_reduce(original_sum)
    bf16 = source.clone()
    dist.all_reduce(bf16)
    denominator = original_sum.norm().clamp_min(1e-12)
    error = actual.float() - original_sum
    metrics = _max_across_ranks(
        [
            (error.norm() / denominator).item(),
            error.abs().max().item(),
            ((bf16.float() - original_sum).norm() / denominator).item(),
            ((reference - original_sum).norm() / denominator).item(),
        ],
        source.device,
    )
    return {
        "dequantization": local_accuracy,
        "same_payload_collective": collective_accuracy,
        "full_path": full_path_accuracy,
        "mxfp8_relative_l2_vs_fp32": metrics[0],
        "mxfp8_max_abs_vs_fp32": metrics[1],
        "bf16_relative_l2_vs_fp32": metrics[2],
        "quantization_relative_l2_vs_fp32": metrics[3],
    }


def _check_encoded(
    pipeline: Mxfp8AllToAllAllReduce,
    rank: int,
    hidden: int,
    device: torch.device,
) -> None:
    """Exercise rank/token/block-dependent bytes without trusting dynamic quantization."""
    rows = _SANITY_TOKENS_PER_RANK * pipeline.world_size
    raw_values = torch.full((rows, hidden), 0x38 + rank, dtype=torch.uint8, device=device)
    values = raw_values.view(torch.float8_e4m3fn)
    indices = torch.arange(rows * hidden // _MX_BLOCK, dtype=torch.int64).reshape(
        rows, hidden // (_MX_BLOCK * _SCALE_PAIR), _SCALE_PAIR
    )
    raw_scales = ((indices + rank) % 7 + 124).to(device=device, dtype=torch.uint8)
    _, scale_template = pipeline.quantize(torch.zeros_like(values, dtype=torch.bfloat16))
    scales = raw_scales.view(scale_template.dtype)
    _, reference, contribution = _reference_sum(values, scales)
    actual = torch.empty(rows, hidden, dtype=torch.bfloat16, device=device)
    pipeline.reduce_quantized(values, scales, actual)
    _check_output(actual, reference, contribution, torch.bfloat16, "encoded AllToAll/AllGather")


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


def _breakdown(pipeline: Mxfp8AllToAllAllReduce, source: torch.Tensor, iterations: int) -> dict:
    working = torch.empty_like(source)
    markers = [[torch.npu.Event(enable_timing=True) for _ in range(len(_STAGES) + 1)] for _ in range(iterations)]
    torch.npu.synchronize()
    dist.barrier()
    for events in markers:
        working.copy_(source)
        pipeline(working, markers=events)
    torch.npu.synchronize()
    samples = [[events[index].elapsed_time(events[index + 1]) for index in range(len(_STAGES))] for events in markers]
    samples = _max_across_ranks(samples, source.device)
    return {name: summarize([row[index] for row in samples]) for index, name in enumerate(_STAGES)}


def _benchmark_case(
    pipeline: Mxfp8AllToAllAllReduce,
    source: torch.Tensor,
    args: argparse.Namespace,
) -> dict:
    bf16_payload_bytes = source.numel() * source.element_size()
    per_rank_factor = (pipeline.world_size - 1) / pipeline.world_size
    case = {
        "shape": list(source.shape),
        "bf16_payload_mib": bf16_payload_bytes / _MIB,
        "theoretical_bf16_ring_sent_mib_per_rank": 2 * per_rank_factor * bf16_payload_bytes / _MIB,
        "theoretical_mxfp8_alltoall_allgather_sent_mib_per_rank": (
            per_rank_factor * source.numel() * (1 + 1 / _MX_BLOCK + source.element_size()) / _MIB
        ),
        "accuracy": _check(pipeline, source),
    }
    if args.check_only:
        return case
    working = torch.empty_like(source)
    for _ in range(args.warmup):
        working.copy_(source)
        dist.all_reduce(working)
        working.copy_(source)
        pipeline(working)
    torch.npu.synchronize()
    del working
    results = {"bf16": [], "mxfp8_alltoall": []}
    for repeat in range(args.rounds):
        order = ("bf16", "mxfp8_alltoall") if repeat % 2 == 0 else ("mxfp8_alltoall", "bf16")
        for name in order:
            op = dist.all_reduce if name == "bf16" else pipeline
            results[name].append(_measure(op, source, args.iterations))
    for name, rounds in results.items():
        samples = [sample for measurement in rounds for sample in measurement["samples_ms"]]
        case[name] = {
            **summarize(samples),
            "round_medians_ms": [summarize(item["samples_ms"])["median_ms"] for item in rounds],
            "median_wall_ms_including_reset": statistics.median(item["wall_ms_including_reset"] for item in rounds),
            "max_wall_ms_including_reset": max(item["wall_ms_including_reset"] for item in rounds),
            "extra_peak_allocated_mib": max(item["extra_peak_allocated_mib"] for item in rounds),
            "samples_ms": samples,
        }
    if min(case["bf16"]["median_ms"], case["mxfp8_alltoall"]["median_ms"]) <= 0:
        raise RuntimeError("Nonpositive NPU event duration; verify event timing or increase the payload.")
    case["speedup"] = case["bf16"]["median_ms"] / case["mxfp8_alltoall"]["median_ms"]
    case["latency_reduction_pct"] = 100 * (
        1 - case["mxfp8_alltoall"]["median_ms"] / case["bf16"]["median_ms"]
    )
    case["wall_speedup_including_reset"] = (
        case["bf16"]["median_wall_ms_including_reset"]
        / case["mxfp8_alltoall"]["median_wall_ms_including_reset"]
    )
    if args.breakdown:
        case["stage_diagnostics"] = _breakdown(pipeline, source, args.iterations)
    return case


def _table(headers: list[str], rows: list[list[str]]) -> str:
    def width(value: str) -> int:
        return sum(2 if unicodedata.east_asian_width(char) in ("W", "F") else 1 for char in value)

    widths = [max(width(row[index]) for row in [headers, *rows]) for index in range(len(headers))]

    def line(row: list[str]) -> str:
        return "| " + " | ".join(value + " " * (size - width(value)) for value, size in zip(row, widths)) + " |"

    return "\n".join([line(headers), line(["-" * size for size in widths]), *(line(row) for row in rows)])


def format_report(report: dict, breakdown: bool = False) -> str:
    groups: dict[int, list] = {}
    for case in report["cases"]:
        groups.setdefault(case["shape"][1], []).append(case)
    mode = "精度检查" if report.get("mode") == "correctness_only" else "性能测试"
    lines = [
        "",
        f"MXFP8 AllToAll/AllGather {mode}结果 | TP={report['world_size']} | device={report['device']}",
        f"反量化类型={report.get('dequant_dtype', 'fp32')} | "
        f"求和输出类型={report.get('sum_output_dtype', report.get('dequant_dtype', 'fp32'))} | AllGather=bf16",
    ]
    has_performance = bool(report["cases"]) and all("mxfp8_alltoall" in case for case in report["cases"])
    if has_performance:
        lines.append("耗时为完整路径中位数；输入重置不计入；每个样本取最慢 rank。")
    else:
        lines.append("通信误差相对独立 MXFP8 字节解码后的 FP32 SUM；整体误差相对原始输入 FP32 SUM。")
    for hidden, cases in groups.items():
        lines.extend(["", f"hidden_size = {hidden}", ""])
        if has_performance:
            rows = []
            for case in cases:
                gain = case["latency_reduction_pct"]
                change = f"{'下降' if gain > 0 else '增加'} {abs(gain):.2f}%" if gain else "持平"
                rows.append(
                    [
                        str(case["shape"][0]),
                        f"{case['bf16_payload_mib']:.2f}",
                        f"{case['bf16']['median_ms']:.4f}",
                        f"{case['mxfp8_alltoall']['median_ms']:.4f}",
                        f"{case['speedup']:.3f}x",
                        change,
                    ]
                )
            lines.append(
                _table(
                    ["Tokens", "BF16输入(MiB)", "BF16耗时(ms)", "A2A完整耗时(ms)", "加速比", "延迟变化"],
                    rows,
                )
            )
        lines.extend(["", "整体误差（相对原始输入 FP32 SUM）：", ""])
        lines.append(
            _table(
                ["Tokens", "MXFP8相对L2", "量化相对L2", "BF16相对L2", "MXFP8最大绝对误差"],
                [
                    [
                        str(case["shape"][0]),
                        f"{case['accuracy']['mxfp8_relative_l2_vs_fp32'] * 100:.3f}%",
                        f"{case['accuracy']['quantization_relative_l2_vs_fp32'] * 100:.3f}%",
                        f"{case['accuracy']['bf16_relative_l2_vs_fp32'] * 100:.3f}%",
                        f"{case['accuracy']['mxfp8_max_abs_vs_fp32']:.6f}",
                    ]
                    for case in cases
                ],
            )
        )
        lines.extend(["", "分层正确性（相对独立 MXFP8 字节解码）：", ""])
        correctness_headers = ["Tokens", "反量化最大误差", "同payload通信最大误差", "完整路径最大误差"]
        if has_performance:
            correctness_headers.append("A2A额外峰值(MiB)")
        correctness_rows = []
        for case in cases:
            accuracy = case["accuracy"]
            # Reports produced before --dequant-dtype always used FP32.
            dequant_accuracy = accuracy.get("dequantization", accuracy.get("dequant_fp32"))
            row = [
                str(case["shape"][0]),
                f"{dequant_accuracy['max_abs_vs_dequantized_sum']:.6f}",
                f"{accuracy['same_payload_collective']['max_abs_vs_dequantized_sum']:.6f}",
                f"{accuracy['full_path']['max_abs_vs_dequantized_sum']:.6f}",
            ]
            if has_performance:
                row.append(f"{case['mxfp8_alltoall']['extra_peak_allocated_mib']:.2f}")
            correctness_rows.append(row)
        lines.append(_table(correctness_headers, correctness_rows))
    if breakdown and has_performance:
        lines.extend(["", "阶段耗时诊断（ms；独立测量，各列中位数之和不等于完整耗时）："])
        for hidden, cases in groups.items():
            diagnostic_cases = [case for case in cases if "stage_diagnostics" in case]
            if not diagnostic_cases:
                continue
            lines.extend(["", f"hidden_size = {hidden}", ""])
            lines.append(
                _table(
                    ["Tokens", "动态MX量化", "payload打包", "AllToAll", "反量化与求和", "BF16 AllGather"],
                    [
                        [str(case["shape"][0])]
                        + [f"{case['stage_diagnostics'][stage]['median_ms']:.4f}" for stage in _STAGES]
                        for case in diagnostic_cases
                    ],
                )
            )
    return "\n".join(lines)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", nargs="+", type=int, default=[512, 2048, 8192, 16384])
    parser.add_argument("--hidden-sizes", nargs="+", type=int, default=[2048, 5120])
    parser.add_argument(
        "--dequant-dtype",
        choices=("bf16", "fp32"),
        default="bf16",
        help="Dequantization and local SUM output dtype; bf16 reduces staging memory, fp32 restores the original path.",
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--breakdown", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--timeout-seconds", type=int, default=300)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--report-json",
        type=Path,
        help="Print tables from an existing result JSON without running NPUs.",
    )
    args = parser.parse_args(argv)
    if any(tokens < 1 for tokens in args.tokens) or any(
        hidden < _HIDDEN_ALIGNMENT or hidden % _HIDDEN_ALIGNMENT for hidden in args.hidden_sizes
    ):
        parser.error("Tokens must be positive; hidden sizes must be positive multiples of 64.")
    if args.warmup < 0 or min(args.iterations, args.rounds, args.timeout_seconds) < 1:
        parser.error("Iterations, rounds and timeout must be positive; warmup must be nonnegative.")
    return args


def main() -> None:
    args = _parse_args()
    if args.report_json:
        print(format_report(json.loads(args.report_json.read_text(encoding="utf-8")), args.breakdown), flush=True)
        return
    torch_npu = importlib.import_module("torch_npu")
    if not torch.npu.is_available():
        raise RuntimeError("This benchmark requires Ascend NPUs.")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(local_rank)
    quantize_op = getattr(torch_npu, "npu_dynamic_mx_quant", None)
    dequantize_op = getattr(torch_npu, "npu_anti_mx_quant", None)
    if not callable(dequantize_op):
        dequantize_op = getattr(torch.ops.npu, "npu_anti_mx_quant", None)
    if not callable(quantize_op) or not callable(dequantize_op):
        raise RuntimeError("Required npu_dynamic_mx_quant or npu_anti_mx_quant entry is unavailable.")
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size not in _SUPPORTED_WORLD_SIZES or any(tokens % world_size for tokens in args.tokens):
        raise ValueError("Use TP=2/4/8 and tokens divisible by TP.")
    dist.init_process_group("hccl", timeout=timedelta(seconds=args.timeout_seconds))
    try:
        rank = dist.get_rank()
        device = torch.device("npu", torch.npu.current_device())
        dequant_dtype = torch.bfloat16 if args.dequant_dtype == "bf16" else torch.float32
        pipeline = Mxfp8AllToAllAllReduce(world_size, quantize_op, dequantize_op, dequant_dtype=dequant_dtype)
        report = {
            "torch": torch.__version__,
            "torch_npu": torch_npu.__version__,
            "device": torch.npu.get_device_name(),
            "hostname_rank0": socket.gethostname(),
            "world_size": world_size,
            "mode": "correctness_only" if args.check_only else "benchmark",
            "quantize_module": getattr(quantize_op, "__module__", None),
            "dequantize_module": getattr(dequantize_op, "__module__", None),
            "dequant_dtype": args.dequant_dtype,
            "sum_output_dtype": args.dequant_dtype,
            "pipeline": (
                "dynamic_mx_quant -> packed all_to_all_single -> "
                f"{args.dequant_dtype.upper()} anti-quant/SUM output -> BF16 all_gather"
            ),
            "warmup": args.warmup,
            "iterations_per_round": args.iterations,
            "rounds": args.rounds,
            "seed": args.seed,
            "timing": "NPU events, input reset excluded, slowest rank per sample; Eager only",
            "memory_scope": "PyTorch allocator only; excludes allocator reserved and external HCCL resources",
            "communication_domains": "ordinary default HCCL for baseline, candidate, references and statistics",
            "sanity_checks": [],
            "cases": [],
        }
        if rank == 0:
            print(
                f"device={report['device']} | TP={world_size} | mode={report['mode']}\n"
                f"torch={report['torch']} | torch_npu={report['torch_npu']}\n"
                f"quantize={report['quantize_module']} | dequantize={report['dequantize_module']}\n"
                f"dequant_dtype={args.dequant_dtype} | sum_output_dtype={args.dequant_dtype}\n"
                f"warmup={args.warmup} | iterations={args.iterations} | rounds={args.rounds}",
                flush=True,
            )
        sanity_rows = _SANITY_TOKENS_PER_RANK * world_size
        sanity_hidden = args.hidden_sizes[0]
        for name, value in (
            ("zeros", 0.0),
            ("rank_constant", rank + 1.0),
            ("cancellation", 1.0 if rank % 2 == 0 else -1.0),
        ):
            source = torch.full((sanity_rows, sanity_hidden), value, dtype=torch.bfloat16, device=device)
            _check(pipeline, source, exact=True)
            report["sanity_checks"].append({"case": name, "pass": True})
            if rank == 0:
                print(f"Correctness {name}: PASS", flush=True)
        _check_encoded(pipeline, rank, sanity_hidden, device)
        report["sanity_checks"].append({"case": "rank_token_block_scales", "pass": True})
        if rank == 0:
            print("Correctness rank_token_block_scales: PASS", flush=True)

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
                        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        if rank == 0:
            print(format_report(report, args.breakdown), flush=True)
            print("PASS: all requested MXFP8 AllToAll/AllGather checks passed.", flush=True)
            if args.output:
                print(f"\n完整结果已保存：{args.output}", flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
