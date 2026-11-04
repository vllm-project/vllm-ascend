# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Standalone A5 native MXFP8 all-reduce correctness test and microbenchmark.

Run with torchrun; see benchmark_native_mxfp8_all_reduce.md. No vLLM imports,
local C++ wrapper, or sibling Python files are needed.
"""

import argparse
import importlib
import json
import math
import os
import socket
import statistics
import time
from collections.abc import Callable
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist

_SUPPORTED_WORLD_SIZES = (2, 4, 8)
_MX_BLOCK = 32
_SCALE_PAIR = 2
_HIDDEN_ALIGNMENT = 128
_MIN_HIDDEN = 1024
_MAX_HIDDEN = 8192
_MIB = 1024 * 1024
_SANITY_TOKENS = 128
_STAGES = ("dynamic_mx_quant", "scale_layout", "native_quant_all_reduce", "copy_result")
# torch extension enum values, not the raw ACL/GE dtype enums.
_DTYPE_BF16 = 15
_DTYPE_FP32 = 6
_DTYPE_E4M3FN = 292
_DTYPE_E8M0 = 293
_BF16_RTOL = 2**-7
_FP32_RTOL = 2e-5
_COMM_ATOL = 1e-6


class NativeMxfp8AllReduce:
    """BF16 -> local MXFP8 -> official QuantAllReduce -> BF16 in-place result."""

    def __init__(self, world_size: int, hcom: str, quantize_op: Callable, all_reduce_op: Callable):
        if world_size not in _SUPPORTED_WORLD_SIZES:
            raise ValueError("Use TP=2, 4 or 8 for this benchmark.")
        if not callable(quantize_op) or not callable(all_reduce_op):
            raise RuntimeError("Both npu_dynamic_mx_quant and official quant_all_reduce must be callable.")
        self.world_size, self.hcom = world_size, hcom
        self.quantize_op, self.all_reduce_op = quantize_op, all_reduce_op

    def quantize(self, x: torch.Tensor, markers: list | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        values, scales = self.quantize_op(x, dst_type=torch.float8_e4m3fn)
        if markers is not None:
            markers[1].record()
        # Preserve E8M0 bytes. A numeric cast to uint8 would corrupt the scales.
        scale_bytes = scales.view(torch.uint8).reshape(x.shape[0], x.shape[1] // (_MX_BLOCK * _SCALE_PAIR), 2)
        scale_bytes = scale_bytes.contiguous()
        if markers is not None:
            markers[2].record()
        return values, scale_bytes

    def reduce_quantized(self, values: torch.Tensor, scales: torch.Tensor, output_dtype: int = _DTYPE_BF16):
        # The public facade owns context and buffer setup. Only native quantized
        # communication uses this hcom; ordinary collectives use the default PG.
        return self.all_reduce_op(
            values,
            scales,
            self.hcom,
            self.world_size,
            reduce_op="sum",
            output_dtype=output_dtype,
            x_dtype=_DTYPE_E4M3FN,
            scales_dtype=_DTYPE_E8M0,
        )

    def __call__(self, x: torch.Tensor, markers: list | None = None) -> torch.Tensor:
        if markers is not None:
            markers[0].record()
        values, scales = self.quantize(x, markers)
        result = self.reduce_quantized(values, scales)
        if markers is not None:
            markers[3].record()
        x.copy_(result)
        if markers is not None:
            markers[4].record()
        return x


def decode_mxfp8_cpu(values: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """Independent raw-byte reference; no npu_anti_mx_quant or NPU casts."""
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
    value_table = torch.tensor(value_table, dtype=torch.float32)
    scale_table = torch.tensor([math.ldexp(1.0, code - 127) for code in range(255)] + [math.nan])
    tokens, hidden = values.shape
    decoded = value_table[raw_values.long()].reshape(tokens, hidden // _MX_BLOCK, _MX_BLOCK)
    factors = scale_table[raw_scales.long().reshape(tokens, hidden // _MX_BLOCK)].unsqueeze(-1)
    return (decoded * factors).reshape(tokens, hidden)


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
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX)  # ordinary default HCCL group
    return tensor.cpu().tolist()


def _require_all(condition: bool, message: str, device: torch.device) -> None:
    if _max_across_ranks([int(not condition)], device)[0]:
        raise AssertionError(message)


def _reference_sum(values: torch.Tensor, scales: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    decoded = decode_mxfp8_cpu(values, scales)
    _require_all(bool(torch.isfinite(decoded).all()), "Decoded MXFP8 data contains NaN/Inf.", values.device)
    reference = decoded.to(values.device)
    contribution = decoded.abs().to(values.device)
    dist.all_reduce(reference)
    dist.all_reduce(contribution)
    return reference, contribution


def _check_output(actual, reference, contribution, dtype, *, exact=False) -> dict:
    correct_metadata = actual.shape == reference.shape and actual.dtype == dtype and actual.device == reference.device
    _require_all(
        correct_metadata,
        f"Unexpected output shape/dtype/device: {actual.shape}, {actual.dtype}",
        reference.device,
    )
    rounded_reference = reference.to(dtype).float()
    difference = (actual.float() - rounded_reference).abs()
    rtol = _BF16_RTOL if dtype == torch.bfloat16 else _FP32_RTOL
    tolerance = torch.zeros_like(reference) if exact else _COMM_ATOL + rtol * contribution
    bad = (~torch.isfinite(actual)) | (difference > tolerance)
    stats = _max_across_ranks(
        [int(bad.any().item()), difference.nan_to_num(nan=math.inf).max().item()], reference.device
    )
    if stats[0]:
        raise AssertionError(f"Native MXFP8 {dtype} communication check failed; max_abs={stats[1]}")
    return {"pass": True, "max_abs_vs_dequantized_sum": stats[1]}


def _check(pipeline: NativeMxfp8AllReduce, source: torch.Tensor, *, exact=False) -> dict:
    """Check the same quantized payload first, then the full BF16 pipeline."""
    values, scales = pipeline.quantize(source)
    valid_values = values.dtype == torch.float8_e4m3fn and values.shape == source.shape
    _require_all(valid_values, "Unexpected dynamic MX quantization dtype/shape.", source.device)
    if exact:
        decoded = decode_mxfp8_cpu(values, scales)
        _require_all(
            torch.equal(decoded, source.float().cpu()), "Sanity input did not quantize exactly.", source.device
        )
    reference, contribution = _reference_sum(values, scales)
    accuracy = {}
    for name, enum, dtype in (("bf16", _DTYPE_BF16, torch.bfloat16), ("fp32", _DTYPE_FP32, torch.float32)):
        output = pipeline.reduce_quantized(values, scales, enum)
        accuracy[f"comm_{name}"] = _check_output(output, reference, contribution, dtype, exact=exact)
    actual = source.clone()
    pipeline(actual)
    accuracy["full_path"] = _check_output(actual, reference, contribution, torch.bfloat16, exact=exact)
    original_sum = source.float()
    dist.all_reduce(original_sum)
    bf16 = source.clone()
    dist.all_reduce(bf16)
    denominator = original_sum.norm().clamp_min(1e-12)
    error = actual.float() - original_sum
    metrics = torch.stack(
        (
            error.norm() / denominator,
            error.abs().max(),
            (bf16.float() - original_sum).norm() / denominator,
            (reference - original_sum).norm() / denominator,
        )
    )
    measurements = _max_across_ranks(metrics.cpu().tolist(), source.device)
    accuracy.update(
        mxfp8_relative_l2_vs_fp32=measurements[0],
        mxfp8_max_abs_vs_fp32=measurements[1],
        bf16_relative_l2_vs_fp32=measurements[2],
        quantization_relative_l2_vs_fp32=measurements[3],
    )
    return accuracy


def _check_encoded(pipeline: NativeMxfp8AllReduce, rank: int, hidden: int, device: torch.device) -> None:
    # Rank, token and adjacent block scales all differ. No dynamic quantizer is
    # involved, so a wrong layout or a no-op native API cannot hide behind it.
    raw = torch.full((_SANITY_TOKENS, hidden), 0x38 + rank, dtype=torch.uint8, device=device)
    indices = torch.arange(_SANITY_TOKENS * hidden // _MX_BLOCK).reshape(_SANITY_TOKENS, hidden // 64, 2)
    scales = ((indices + rank) % 7 + 124).to(device=device, dtype=torch.uint8)
    values = raw.view(torch.float8_e4m3fn)
    reference, contribution = _reference_sum(values, scales)
    for enum, dtype in ((_DTYPE_BF16, torch.bfloat16), (_DTYPE_FP32, torch.float32)):
        output = pipeline.reduce_quantized(values, scales, enum)
        _check_output(output, reference, contribution, dtype)


def _measure(op: Callable, source: torch.Tensor, iterations: int) -> dict:
    working = torch.empty_like(source)
    events = [(torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)) for _ in range(iterations)]
    torch.npu.synchronize()
    dist.barrier()  # default group only, never the quantized domain
    torch.npu.synchronize()
    torch.npu.reset_peak_memory_stats()
    allocated = torch.npu.memory_allocated()
    start_wall = time.perf_counter()
    for start, end in events:
        working.copy_(source)  # excluded from the event interval
        start.record()
        op(working)
        end.record()
    torch.npu.synchronize()
    wall_ms = (time.perf_counter() - start_wall) * 1000 / iterations
    extra_peak_mib = (torch.npu.max_memory_allocated() - allocated) / _MIB
    samples = _max_across_ranks([start.elapsed_time(end) for start, end in events], source.device)
    wall_ms, extra_peak_mib = _max_across_ranks([wall_ms, extra_peak_mib], source.device)
    return {"samples_ms": samples, "wall_ms_including_reset": wall_ms, "extra_peak_allocated_mib": extra_peak_mib}


def _breakdown(pipeline: NativeMxfp8AllReduce, source: torch.Tensor, iterations: int) -> dict:
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


def _benchmark_case(pipeline: NativeMxfp8AllReduce, source: torch.Tensor, args: argparse.Namespace) -> dict:
    case = {
        "shape": list(source.shape),
        "bf16_payload_mib": source.numel() * source.element_size() / _MIB,
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
    results = {"bf16": [], "mxfp8": []}
    for repeat in range(args.rounds):
        order = ("bf16", "mxfp8") if repeat % 2 == 0 else ("mxfp8", "bf16")
        for name in order:
            op = dist.all_reduce if name == "bf16" else pipeline
            results[name].append(_measure(op, source, args.iterations))
    for name, rounds in results.items():
        samples = [sample for measurement in rounds for sample in measurement["samples_ms"]]
        case[name] = {
            **summarize(samples),
            "round_medians_ms": [summarize(measurement["samples_ms"])["median_ms"] for measurement in rounds],
            "median_wall_ms_including_reset": statistics.median(item["wall_ms_including_reset"] for item in rounds),
            "extra_peak_allocated_mib": max(item["extra_peak_allocated_mib"] for item in rounds),
            "samples_ms": samples,
        }
    if min(case["bf16"]["median_ms"], case["mxfp8"]["median_ms"]) <= 0:
        raise RuntimeError("Nonpositive NPU event duration; verify event timing or increase the payload.")
    case["speedup"] = case["bf16"]["median_ms"] / case["mxfp8"]["median_ms"]
    case["latency_reduction_pct"] = 100 * (1 - case["mxfp8"]["median_ms"] / case["bf16"]["median_ms"])
    case["wall_speedup_including_reset"] = (
        case["bf16"]["median_wall_ms_including_reset"] / case["mxfp8"]["median_wall_ms_including_reset"]
    )
    if args.breakdown:
        case["stage_diagnostics"] = _breakdown(pipeline, source, args.iterations)
    return case


def _parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", nargs="+", type=int, default=[512, 2048, 8192, 16384])
    parser.add_argument("--hidden-sizes", nargs="+", type=int, default=[2048, 5120])
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--breakdown", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--timeout-seconds", type=int, default=300)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if any(value < 1 for value in args.tokens) or any(
        hidden < _MIN_HIDDEN or hidden > _MAX_HIDDEN or hidden % _HIDDEN_ALIGNMENT for hidden in args.hidden_sizes
    ):
        parser.error("Tokens must be positive; hidden sizes must be in [1024,8192] and aligned to 128.")
    if args.warmup < 0 or min(args.iterations, args.rounds, args.timeout_seconds) < 1:
        parser.error("Iterations, rounds and timeout must be positive; warmup must be nonnegative.")
    return args


def _print_case(case: dict, check_only: bool) -> None:
    tokens, hidden = case["shape"]
    if check_only:
        print(f"Correctness [{tokens},{hidden}]: PASS " + json.dumps(case["accuracy"]), flush=True)
        return
    print(
        f"{tokens:>7} {hidden:>6} {case['bf16_payload_mib']:>9.2f} "
        f"{case['bf16']['median_ms']:>9.4f} {case['mxfp8']['median_ms']:>13.4f} "
        f"{case['speedup']:>7.3f} {case['latency_reduction_pct']:>+8.2f} "
        f"{case['accuracy']['mxfp8_relative_l2_vs_fp32']:>12.6f} "
        f"{case['accuracy']['bf16_relative_l2_vs_fp32']:>11.6f} "
        f"{case['mxfp8']['extra_peak_allocated_mib']:>15.2f}",
        flush=True,
    )
    if "stage_diagnostics" in case:
        print(
            "Stage medians (diagnostic): "
            + json.dumps({key: value["median_ms"] for key, value in case["stage_diagnostics"].items()}),
            flush=True,
        )


def main() -> None:
    args = _parse_args()
    # Load backend packages only in the NPU worker; CPU tests and --help do not
    # need torch_npu, CANN, or vLLM.
    torch_npu = importlib.import_module("torch_npu")
    ops = importlib.import_module("cann_ops_transformer.ops")
    if not torch.npu.is_available():
        raise RuntimeError("This benchmark requires Ascend NPUs.")
    torch.npu.set_device(int(os.environ["LOCAL_RANK"]))
    quantize_op = getattr(torch_npu, "npu_dynamic_mx_quant", None)
    all_reduce_op = getattr(ops, "quant_all_reduce", None)
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size not in _SUPPORTED_WORLD_SIZES or any(tokens % world_size for tokens in args.tokens):
        raise ValueError("Use TP=2/4/8 and tokens divisible by TP.")
    if not callable(quantize_op) or not callable(all_reduce_op):
        raise RuntimeError("Required dynamic MX quantization or official QuantAllReduce entry is unavailable.")
    timeout = timedelta(seconds=args.timeout_seconds)
    dist.init_process_group("hccl", timeout=timeout)
    quant_group = None
    try:
        rank = dist.get_rank()
        device = torch.device("npu", torch.npu.current_device())
        quant_group = dist.new_group(ranks=list(range(world_size)), backend="hccl", timeout=timeout)
        hcom = quant_group._get_backend(device).get_hccl_comm_name(rank, init_comm=False)
        pipeline = NativeMxfp8AllReduce(world_size, hcom, quantize_op, all_reduce_op)
        report = {
            "torch": torch.__version__,
            "torch_npu": torch_npu.__version__,
            "device": torch.npu.get_device_name(),
            "hostname_rank0": socket.gethostname(),
            "world_size": world_size,
            "ops_package": ops.__file__,
            "operator_module": all_reduce_op.__module__,
            "mode": "correctness_only" if args.check_only else "benchmark",
            "dtype_tags": {"bf16": _DTYPE_BF16, "fp32": _DTYPE_FP32, "x": _DTYPE_E4M3FN, "scales": _DTYPE_E8M0},
            "warmup": args.warmup,
            "iterations_per_round": args.iterations,
            "rounds": args.rounds,
            "seed": args.seed,
            "timing": "NPU events, input reset excluded, slowest rank per sample; Eager only",
            "memory_scope": "PyTorch allocator only; excludes persistent/external HCCL context and buffers",
            "communication_domains": "ordinary default HCCL for baseline/control; dedicated HCCL for native MXFP8",
            "sanity_checks": [],
            "cases": [],
        }
        if rank == 0:
            print(json.dumps({key: value for key, value in report.items() if key != "cases"}, indent=2), flush=True)
        hidden = args.hidden_sizes[0]
        for name, value in (
            ("zeros", 0.0),
            ("rank_constant", rank + 1.0),
            ("cancellation", 1.0 if rank % 2 == 0 else -1.0),
        ):
            source = torch.full((_SANITY_TOKENS, hidden), value, dtype=torch.bfloat16, device=device)
            _check(pipeline, source, exact=True)
            report["sanity_checks"].append({"case": name, "pass": True})
            if rank == 0:
                print(f"Correctness {name}: PASS", flush=True)
        _check_encoded(pipeline, rank, hidden, device)
        report["sanity_checks"].append({"case": "rank_token_block_scales", "pass": True})
        if rank == 0:
            print("Correctness rank_token_block_scales: PASS", flush=True)
            if not args.check_only:
                print(
                    " tokens hidden  BF16_MiB   BF16_ms MXFP8_full_ms speedup gain_pct "
                    "MXFP8_rel_L2 BF16_rel_L2 extra_MXFP8_MiB",
                    flush=True,
                )
        for hidden in args.hidden_sizes:
            for tokens in args.tokens:
                torch.manual_seed(args.seed + rank)
                source = torch.randn(tokens, hidden, dtype=torch.bfloat16, device=device)
                case = _benchmark_case(pipeline, source, args)
                report["cases"].append(case)
                del source
                if rank == 0:
                    _print_case(case, args.check_only)
                    if args.output:
                        args.output.parent.mkdir(parents=True, exist_ok=True)
                        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        if rank == 0:
            print("PASS: all requested MXFP8 native communication and full-path checks passed.", flush=True)
    finally:
        if quant_group is not None:
            dist.destroy_process_group(quant_group)
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
