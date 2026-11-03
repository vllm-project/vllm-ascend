# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run with torchrun on 2/4/8 A5 NPUs; use Gloo exclusively for control traffic."""

import argparse
import json
import math
import os
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import torch_npu
from reference import BYTE_VALUES, MX_BLOCK_SIZE, decode_e4m3fn, decode_e8m0

BF16_COMM_RTOL = 2**-7
FP32_COMM_RTOL = 2e-5
COMM_ATOL = 1e-6
CASES = ("encoded", "zeros", "rank_constant", "random", "rank_scaled", "cancellation", "outliers")
OUTPUT_DTYPES = ("bf16", "fp32")


def dequantize_cpu(values: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """Decode raw bytes, not numeric casts of E8M0 or npu_anti_mx_quant."""
    values = values.to(device="cpu", dtype=torch.long)
    scales = scales.to(device="cpu", dtype=torch.long)
    value_table = torch.tensor([decode_e4m3fn(code) for code in range(BYTE_VALUES)], dtype=torch.float32)
    scale_table = torch.tensor([decode_e8m0(code) for code in range(BYTE_VALUES)], dtype=torch.float32)
    rows, hidden = values.shape
    return (
        value_table[values].reshape(rows, hidden // MX_BLOCK_SIZE, MX_BLOCK_SIZE)
        * scale_table[scales.reshape(rows, hidden // MX_BLOCK_SIZE)].unsqueeze(-1)
    ).reshape(rows, hidden)


def make_inputs(case: str, rows: int, hidden: int, rank: int, repeat: int, device: torch.device):
    # Cancellation partners use identical seeds and opposite signs.
    generator = torch.Generator().manual_seed(1234 + repeat * 100 + (rank // 2 if case == "cancellation" else rank))
    source = torch.randn(rows, hidden, generator=generator)
    if case == "encoded":
        # Distinct scales across ranks, rows and adjacent 32-element blocks
        # catch scale layout errors independently of dynamic quantization.
        codes = torch.tensor([0x38, 0xB8, 0x40, 0xC0, 0x30, 0xB0, 0x7E, 0xFE], dtype=torch.uint8)
        indices = torch.arange(rows * hidden).reshape(rows, hidden) + rank + repeat
        raw_values = codes[indices % codes.numel()]
        scale_indices = torch.arange(rows * hidden // MX_BLOCK_SIZE).reshape(rows, hidden // 64, 2)
        raw_scales = ((scale_indices + rank + repeat) % 7 + 124).to(torch.uint8)
        values = raw_values.to(device).view(torch.float8_e4m3fn)
        scales = raw_scales.to(device)
        source = dequantize_cpu(raw_values, raw_scales)
        return source, values, scales
    if case == "zeros":
        source.zero_()
    elif case == "rank_constant":
        source.fill_(rank + 1 + repeat)
    elif case == "rank_scaled":
        source *= 2.0 ** (rank - 2)
    elif case == "cancellation":
        source *= -1 if rank % 2 else 1
    elif case == "outliers":
        source *= torch.pow(2.0, (torch.arange(hidden) // MX_BLOCK_SIZE % 11 - 5).float())
        source[:, ::257] *= 32
    source = source.to(torch.bfloat16)
    values, scales = torch_npu.npu_dynamic_mx_quant(source.to(device), dst_type=torch.float8_e4m3fn)
    if values.dtype != torch.float8_e4m3fn or values.shape != source.shape:
        raise RuntimeError(f"Unexpected quantized tensor: {values.dtype}, {values.shape}")
    # Reinterpret storage; scales.to(uint8) would destroy E8M0 semantics.
    scales = scales.view(torch.uint8).reshape(rows, hidden // 64, 2).contiguous()
    return source.float(), values.contiguous(), scales


def error_metrics(actual: torch.Tensor, expected: torch.Tensor) -> dict:
    difference = (actual.double() - expected.double()).flatten()
    denominator = expected.double().norm().item()
    norm = difference.norm().item()
    return {
        "max_abs": difference.abs().max().item(),
        "rmse": (difference.square().mean().sqrt()).item(),
        "nrmse": norm / denominator if denominator else (0.0 if norm == 0 else math.inf),
    }


def evaluate(gathered: list[dict], case: str, rows: int, hidden: int, repeat: int, args) -> bool:
    decoded = torch.stack([dequantize_cpu(item["values"], item["scales"]) for item in gathered])
    if not torch.isfinite(decoded).all():
        raise RuntimeError("Non-finite decoded MXFP8 data: inspect quantizer output and E8M0 scale bytes")
    # Golden 1 isolates communication, Golden 2 includes quantization error.
    quantized_sum = decoded.sum(dim=0, dtype=torch.float32)
    original_sum = torch.stack([item["source"].float() for item in gathered]).sum(dim=0, dtype=torch.float32)
    contribution = decoded.abs().sum(dim=0)
    quant_error = error_metrics(quantized_sum, original_sum)
    passed = True
    for dtype in args.output_dtypes:
        worst_abs = 0.0
        bad_elements = 0
        nonfinite = 0
        rank_differences = 0
        rtol = args.bf16_comm_rtol if dtype == "bf16" else args.fp32_comm_rtol
        expected = quantized_sum.to(torch.bfloat16).float() if dtype == "bf16" else quantized_sum
        # Normalize to sum(abs(contributions)), so cancellation does not make
        # a harmless rounding error look like an unbounded relative error.
        tolerance = args.comm_atol + rtol * contribution
        if case in ("zeros", "rank_constant"):
            tolerance = torch.zeros_like(tolerance)
        for item in gathered:
            actual = item["outputs"][dtype]
            finite = torch.isfinite(actual)
            difference = (actual - expected).abs()
            nonfinite += (~finite).sum().item()
            bad_elements += ((difference > tolerance) | ~finite).sum().item()
            worst_abs = max(worst_abs, difference.nan_to_num(nan=math.inf).max().item())
            rank_differences += (actual != gathered[0]["outputs"][dtype]).sum().item()
        quant_passed = args.max_quant_nrmse is None or quant_error["nrmse"] <= args.max_quant_nrmse
        comm_passed = bad_elements == 0 and nonfinite == 0
        passed = passed and comm_passed and quant_passed
        report = {
            "case": case,
            "shape": [rows, hidden],
            "repeat": repeat,
            "output": dtype,
            "comm_pass": comm_passed,
            "comm_max_abs": worst_abs,
            "comm_bad_elements_all_ranks": bad_elements,
            "rank_difference_elements": rank_differences,
            "quantization_vs_original": quant_error,
            "end_to_end_rank0": error_metrics(gathered[0]["outputs"][dtype], original_sum),
            "quant_quality_pass": quant_passed if args.max_quant_nrmse is not None else "report_only",
        }
        print(json.dumps(report), flush=True)
    return passed


def run(args) -> None:
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size not in (2, 4, 8):
        raise ValueError("The native operator supports world sizes 2, 4 and 8")
    if any(rows <= 0 or rows % world_size for rows in args.rows):
        raise ValueError("Use positive rows divisible by world_size for this initial probe")
    if any(hidden < 1024 or hidden > 8192 or hidden % 128 for hidden in args.hidden_sizes):
        raise ValueError("Hidden sizes must be in [1024, 8192] and aligned to 128")
    torch.npu.set_device(local_rank)
    device = torch.device("npu", local_rank)
    manifest = json.loads(args.manifest.read_text())
    if manifest["torch"] != torch.__version__ or manifest["torch_npu"] != torch_npu.__version__:
        raise RuntimeError("The extension was built with different torch/torch_npu versions; rebuild it")
    torch.ops.load_library(manifest["library"])
    timeout = timedelta(seconds=args.timeout)
    # Never use ordinary collectives or barrier on the quantized HCCL group.
    dist.init_process_group(backend="gloo", timeout=timeout)
    quant_group = dist.new_group(ranks=list(range(world_size)), backend="hccl", timeout=timeout)
    hcom = quant_group._get_backend(device).get_hccl_comm_name(rank)
    if rank == 0:
        print(
            json.dumps(
                {
                    "torch": torch.__version__,
                    "torch_npu": torch_npu.__version__,
                    "device": torch.npu.get_device_name(local_rank),
                    "world_size": world_size,
                    "manifest": manifest,
                    "comm_atol": args.comm_atol,
                    "bf16_comm_rtol": args.bf16_comm_rtol,
                    "fp32_comm_rtol": args.fp32_comm_rtol,
                    "max_quant_nrmse": args.max_quant_nrmse,
                    "note": "Synchronous accuracy probe; timings are not performance measurements",
                }
            ),
            flush=True,
        )
    passed = True
    for rows in args.rows:
        for hidden in args.hidden_sizes:
            # CANN requires HCCL_BUFFSIZE >= 2*(values_bytes+scales_bytes+1).
            required_bytes = 2 * (rows * hidden + rows * hidden // MX_BLOCK_SIZE + 1)
            buffer_bytes = int(os.environ.get("HCCL_BUFFSIZE", "200")) * 1024 * 1024
            if buffer_bytes < required_bytes:
                raise ValueError(f"HCCL_BUFFSIZE too small: need at least {math.ceil(required_bytes / 1024**2)} MB")
            for repeat in range(args.repeats):
                for case in args.cases:
                    source, values, scales = make_inputs(case, rows, hidden, rank, repeat, device)
                    outputs = {}
                    for dtype in args.output_dtypes:
                        output = torch.ops.a5_quant_comm_probe.quant_all_reduce(values, scales, hcom, dtype == "fp32")
                        outputs[dtype] = output.float().cpu()
                    payload = {
                        "source": source.cpu(),
                        "values": values.view(torch.uint8).cpu(),
                        "scales": scales.cpu(),
                        "outputs": outputs,
                    }
                    gathered = [None] * world_size if rank == 0 else None
                    dist.gather_object(payload, gathered, dst=0)  # default Gloo group
                    if rank == 0:
                        passed = evaluate(gathered, case, rows, hidden, repeat, args) and passed
    verdict = [passed if rank == 0 else None]
    dist.broadcast_object_list(verdict, src=0)  # default Gloo group
    dist.destroy_process_group(quant_group)
    dist.destroy_process_group()
    if not verdict[0]:
        raise RuntimeError("Precision validation FAILED; inspect per-case JSON results")
    if rank == 0:
        print("PASS: native QuantAllReduce matched independently dequantized rank sums within configured tolerances.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("/tmp/a5_quant_comm_probe/manifest.json"))
    parser.add_argument("--rows", type=int, nargs="+", default=[128])
    parser.add_argument("--hidden-sizes", type=int, nargs="+", default=[5120])
    parser.add_argument("--cases", nargs="+", choices=CASES, default=list(CASES))
    parser.add_argument("--output-dtypes", nargs="+", choices=OUTPUT_DTYPES, default=list(OUTPUT_DTYPES))
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--comm-atol", type=float, default=COMM_ATOL)
    parser.add_argument("--bf16-comm-rtol", type=float, default=BF16_COMM_RTOL)
    parser.add_argument("--fp32-comm-rtol", type=float, default=FP32_COMM_RTOL)
    parser.add_argument("--max-quant-nrmse", type=float, help="optional workload-specific quantization quality gate")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    if any(
        value < 0 or not math.isfinite(value) for value in (args.comm_atol, args.bf16_comm_rtol, args.fp32_comm_rtol)
    ):
        parser.error("communication tolerances must be finite and nonnegative")
    if args.max_quant_nrmse is not None and (args.max_quant_nrmse < 0 or not math.isfinite(args.max_quant_nrmse)):
        parser.error("--max-quant-nrmse must be finite and nonnegative")
    run(args)


if __name__ == "__main__":
    main()
