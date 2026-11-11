# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Capture BF16, full MXFP8 and pipelined MXFP8 together on one timeline.

Plain python launches torchrun automatically; an existing torchrun also works.
Reuse the benchmark algorithms and the standalone vLLM-style NPU profiler.
"""

import argparse
import importlib
import json
import os
import shlex
import socket
import subprocess
import sys
import uuid
from datetime import datetime, timedelta
from pathlib import Path

import benchmark_all_reduce_modes as profiling
import benchmark_mxfp8_all_reduce as base
import benchmark_mxfp8_pipeline as pipeline
import torch
import torch.distributed as dist


class _StageAnnotations:
    """Profiler-only labels; leave the benchmark's hot paths unchanged."""

    def quantize(self, source):
        start = source.storage_offset() // source.shape[1]
        with torch.profiler.record_function(f"mxfp8/quantize/rows_{start}_{start + source.shape[0]}"):
            return super().quantize(source)

    def pack_payload(self, values, scales):
        with torch.profiler.record_function("mxfp8/pack_payload"):
            return super().pack_payload(values, scales)

    def dequantize(self, values, scales):
        with torch.profiler.record_function("mxfp8/dequantize"):
            return super().dequantize(values, scales)


class _AnnotatedFull(_StageAnnotations, base.Mxfp8AllToAllAllReduce):
    def exchange_payload(self, *args, **kwargs):
        with torch.profiler.record_function("mxfp8/all_to_all"):
            return super().exchange_payload(*args, **kwargs)

    def finish_received(self, *args, **kwargs):
        with torch.profiler.record_function("mxfp8/dequantize_sum_and_gather"):
            return super().finish_received(*args, **kwargs)


class _AnnotatedPipeline(_StageAnnotations, pipeline.ChunkedMxfp8AllToAllAllReduce):
    def _partial(self, *args, **kwargs):
        with torch.profiler.record_function("mxfp8/dequantize_and_sum"):
            return super()._partial(*args, **kwargs)


def _variants(world_size: int, quantize, dequantize, chunks: list[int]) -> dict:
    variants = {
        "bf16_all_reduce": dist.all_reduce,
        "mxfp8_full": _AnnotatedFull(world_size, quantize, dequantize),
    }
    for count in chunks:
        variants[f"mxfp8_pipeline_chunks_{count}"] = _AnnotatedPipeline(
            world_size, quantize, dequantize, chunks=count, pipelined=True
        )
    return variants


def _capture(variants: dict, source: torch.Tensor, args: argparse.Namespace, rank: int) -> Path:
    rows, hidden = source.shape
    directory = args.output_dir / f"tp{dist.get_world_size()}" / f"{rows}x{hidden}" / f"rank{rank}"
    directory.mkdir(parents=True, exist_ok=False)
    working = torch.empty_like(source)
    for _ in range(args.warmup):
        for op in variants.values():
            working.copy_(source)
            op(working)
    torch.npu.synchronize()
    dist.barrier()
    torch.npu.synchronize()
    worker_name = f"{socket.gethostname()}_rank{rank}"
    with profiling._create_profiler(directory, worker_name, args) as capture:
        for step in range(args.steps):
            with torch.profiler.record_function(f"comparison/step_{step}"):
                for name, op in variants.items():
                    with torch.profiler.record_function(f"input_reset/{name}"):
                        working.copy_(source)
                        torch.npu.synchronize()
                    with torch.profiler.record_function(f"variant/{name}"):
                        op(working)
                        # Complete the entire variant, never synchronize between chunks.
                        torch.npu.synchronize()
            capture.step()
    # Exclude rank-dependent trace export/analysis time from the next case.
    dist.barrier()
    return directory


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tp", nargs="+", type=int, choices=base._SUPPORTED_WORLD_SIZES)
    parser.add_argument("--tokens", nargs="+", type=int, default=[16384])
    parser.add_argument("--hidden-sizes", nargs="+", type=int, default=[5120])
    parser.add_argument("--chunks", nargs="+", type=int, default=[2])
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--steps", type=int, default=3, help="Capture this many repetitions of the full comparison.")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--timeout-seconds", type=int, default=2400)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--profile-with-stack", action="store_true")
    parser.add_argument("--profile-memory", action="store_true")
    args = parser.parse_args(argv)
    for name in ("tp", "tokens", "hidden_sizes", "chunks"):
        values = getattr(args, name)
        if values is not None and len(set(values)) != len(values):
            parser.error(f"{name} values must be distinct.")
    if any(value < 1 for value in args.tokens) or any(value < 2 for value in args.chunks):
        parser.error("Tokens must be positive and chunk counts must be at least two.")
    if any(value < base._HIDDEN_ALIGNMENT or value % base._HIDDEN_ALIGNMENT for value in args.hidden_sizes):
        parser.error("Hidden sizes must be positive multiples of 64.")
    if args.warmup < 0 or min(args.steps, args.timeout_seconds) < 1:
        parser.error("Steps and timeout must be positive; warmup must be nonnegative.")
    return args


def _launch(args: argparse.Namespace) -> None:
    world_sizes = args.tp or [2]
    # Validate the complete matrix before launching any workers.
    for world_size in world_sizes:
        for rows in args.tokens:
            for chunks in args.chunks:
                pipeline.chunk_ranges(rows, world_size, chunks)
    output_dir = args.output_dir or Path("results") / (
        f"mxfp8_compare_{datetime.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:8]}"
    )
    for world_size in world_sizes:
        command = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc-per-node={world_size}",
            str(Path(__file__).resolve()),
            "--tp",
            str(world_size),
            "--output-dir",
            str(output_dir.resolve()),
        ]
        for name in ("tokens", "hidden_sizes", "chunks", "warmup", "steps", "seed", "timeout_seconds"):
            value = getattr(args, name)
            command.append("--" + name.replace("_", "-"))
            command.extend(str(item) for item in (value if isinstance(value, list) else [value]))
        for name in ("profile_with_stack", "profile_memory"):
            if getattr(args, name):
                command.append("--" + name.replace("_", "-"))
        print(f"Launching: {shlex.join(command)}", flush=True)
        subprocess.run(command, check=True)
    print(f"All traces saved under: {output_dir.resolve()}", flush=True)


def _worker(args: argparse.Namespace) -> None:
    # Standard torchrun and external HCCL environment, independent of vLLM initialization.
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size not in base._SUPPORTED_WORLD_SIZES or (args.tp is not None and args.tp != [world_size]):
        raise ValueError("Under torchrun, --tp must match WORLD_SIZE and contain one value (2/4/8).")
    for rows in args.tokens:
        for chunks in args.chunks:
            pipeline.chunk_ranges(rows, world_size, chunks)
    args.output_dir = args.output_dir or Path("results/mxfp8_compare")
    torch_npu = importlib.import_module("torch_npu")
    if not torch.npu.is_available():
        raise RuntimeError("This profiler requires Ascend NPUs.")
    torch.npu.set_device(int(os.environ["LOCAL_RANK"]))
    quantize = getattr(torch_npu, "npu_dynamic_mx_quant", None)
    dequantize = getattr(torch_npu, "npu_anti_mx_quant", None)
    if not callable(dequantize):
        dequantize = getattr(torch.ops.npu, "npu_anti_mx_quant", None)
    variants = _variants(world_size, quantize, dequantize, args.chunks)
    dist.init_process_group("hccl", timeout=timedelta(seconds=args.timeout_seconds))
    try:
        rank = dist.get_rank()
        device = torch.device("npu", torch.npu.current_device())
        report = {
            "torch": torch.__version__,
            "torch_npu": torch_npu.__version__,
            "device": torch.npu.get_device_name(),
            "hostname_rank0": socket.gethostname(),
            "world_size": world_size,
            "variants": list(variants),
            "dequant_dtype": "bf16",
            "warmup": args.warmup,
            "steps": args.steps,
            "seed": args.seed,
            "profile_with_stack": args.profile_with_stack,
            "profile_memory": args.profile_memory,
            "capture": "One trace per shape/rank; all variants in order per step; synchronize only between variants",
            "profiler": "CPU/NPU, Level1, Text, PipeUtilization; synchronous export and analysis",
            "communication_domains": "One default HCCL group; Eager; same algorithms as the MXFP8 benchmarks",
            "environment": {
                name: os.environ.get(name)
                for name in (
                    "ASCEND_RT_VISIBLE_DEVICES",
                    "HCCL_NPU_SOCKET_PORT_RANGE",
                    "HCCL_OP_EXPANSION_MODE",
                    "HCCL_BUFFSIZE",
                    "HCCL_ALGO",
                    "HCCL_DETERMINISTIC",
                    "TASK_QUEUE_ENABLE",
                    "TORCH_HCCL_BLOCKING_WAIT",
                    "HCCL_BLOCKING_WAIT",
                    "ASCEND_LAUNCH_BLOCKING",
                )
            },
            "cases": [],
        }
        if rank == 0:
            print(json.dumps(report, indent=2), flush=True)
        for hidden in args.hidden_sizes:
            for rows in args.tokens:
                torch.manual_seed(args.seed + rank)
                source = torch.randn(rows, hidden, dtype=torch.bfloat16, device=device)
                # Independent byte-decode/FP32 SUM reference, entirely outside the trace.
                accuracy = {name: base._check(op, source) for name, op in variants.items() if name != "bf16_all_reduce"}
                directory = _capture(variants, source, args, rank)
                del source
                case = {
                    "shape": [rows, hidden],
                    "bf16_payload_mib": rows * hidden * 2 / base._MIB,
                    "chunk_rows": {
                        str(chunks): [end - start for start, end in pipeline.chunk_ranges(rows, world_size, chunks)]
                        for chunks in args.chunks
                    },
                    "accuracy": accuracy,
                    "rank_directories": [str(directory.parent / f"rank{owner}") for owner in range(world_size)],
                }
                report["cases"].append(case)
                if rank == 0:
                    manifest = args.output_dir / f"tp{world_size}" / "manifest.json"
                    manifest.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
                    print(
                        f"Captured {rows}x{hidden}, TP={world_size}: {directory.parent}\nManifest: {manifest}",
                        flush=True,
                    )
    finally:
        dist.destroy_process_group()


def main() -> None:
    args = _parse_args()
    if "WORLD_SIZE" in os.environ:
        _worker(args)
    else:
        _launch(args)


if __name__ == "__main__":
    main()
