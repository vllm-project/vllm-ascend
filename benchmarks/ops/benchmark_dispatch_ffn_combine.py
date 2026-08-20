#!/usr/bin/env python3
"""Multi-rank benchmark for the W8A8 DispatchFFNCombine operator.

Example (two local NPUs):
  python benchmarks/ops/benchmark_dispatch_ffn_combine.py \
      --devices 0,1 --m 1,8,32,64,128 --k 7168 --n 4096 \
      --experts-per-rank 32 --topk 8 --max-output-size 4096 \
      --warmup 10 --iterations 50

For A/B correctness, first save baseline snapshots with ``--snapshot-dir`` and
then pass that directory to the optimized build through ``--reference-dir``.
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import queue
import socket
import statistics
import sys
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch_npu
from torch.distributed.distributed_c10d import _get_default_group

from vllm_ascend.utils import enable_custom_op

enable_custom_op()


@dataclass(frozen=True)
class Shape:
    m: int
    k: int
    n: int
    experts_per_rank: int
    topk: int
    max_output_size: int


@dataclass
class RankResult:
    rank: int
    shape: dict[str, int]
    times_ms: list[float]
    deterministic: bool
    reference_ok: bool | None
    output_sum: float
    output_abs_max: float


def parse_int_list(value: str) -> list[int]:
    values = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected a comma-separated integer list")
    return values


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def make_packed_dequant_scale(length: int, generator: torch.Generator) -> torch.Tensor:
    """Pack positive FP32 dequant scales into the INT64 format used by CANN."""
    scale = torch.rand(length, dtype=torch.float32, generator=generator) * 0.009 + 0.001
    packed = scale.numpy().view(np.uint32).astype(np.int64)
    return torch.from_numpy(packed).npu()


def hcomm_name(rank: int) -> str:
    group = _get_default_group()
    backend = group._get_backend(torch.device("npu"))
    return backend.get_hccl_comm_name(rank)


def make_weight_list(
    experts: int,
    rows: int,
    cols: int,
    generator: torch.Generator,
) -> list[torch.Tensor]:
    result = []
    for _ in range(experts):
        weight = torch.randint(-16, 16, (rows, cols), dtype=torch.int8, generator=generator).npu()
        result.append(torch_npu.npu_format_cast(weight, 29))
    return result


def make_static_inputs(
    shape: Shape,
    rank: int,
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(20260731 + rank)
    weight1 = make_weight_list(shape.experts_per_rank, shape.k, shape.n, generator)
    weight2 = make_weight_list(shape.experts_per_rank, shape.n // 2, shape.k, generator)
    scale1 = [make_packed_dequant_scale(shape.n, generator) for _ in range(shape.experts_per_rank)]
    scale2 = [make_packed_dequant_scale(shape.k, generator) for _ in range(shape.experts_per_rank)]
    return weight1, weight2, scale1, scale2


def make_case_inputs(shape: Shape, rank: int, world_size: int) -> dict[str, Any]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(20260731 + rank * 100003 + shape.m)
    x = torch.randn((shape.m, shape.k), dtype=torch.bfloat16, generator=generator).npu()
    expert_idx = torch.randint(
        0,
        world_size * shape.experts_per_rank,
        (shape.m, shape.topk),
        dtype=torch.int32,
        generator=generator,
    ).npu()
    logits = torch.randn((shape.m, shape.topk), dtype=torch.float32, generator=generator)
    probs = torch.softmax(logits, dim=-1).npu()
    return {
        "x": x,
        "expert_idx": expert_idx,
        "probs": probs,
        "out": torch.empty((shape.m, shape.k), dtype=torch.bfloat16, device="npu"),
        "expert_token_nums": torch.empty((1, shape.experts_per_rank), dtype=torch.int32, device="npu"),
        "empty_bias": torch.tensor([]),
    }


def call_op(
    case: dict[str, Any],
    static: tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]],
    group: str,
    shape: Shape,
    use_active_mask: bool,
) -> None:
    weight1, weight2, scale1, scale2 = static
    active_mask = torch.ones(shape.m, dtype=torch.bool, device="npu") if use_active_mask else None
    torch.ops._C_ascend.dispatch_ffn_combine(
        x=case["x"],
        weight1=weight1,
        weight2=weight2,
        expert_idx=case["expert_idx"],
        bias1=case["empty_bias"],
        bias2=case["empty_bias"],
        scale1=scale1,
        scale2=scale2,
        probs=case["probs"],
        group=group,
        max_output_size=shape.max_output_size,
        x_active_mask=active_mask,
        out=case["out"],
        expert_token_nums=case["expert_token_nums"],
    )


def snapshot_path(directory: str, rank: int, shape: Shape) -> Path:
    return Path(directory) / (
        f"rank{rank}_m{shape.m}_k{shape.k}_n{shape.n}_"
        f"e{shape.experts_per_rank}_topk{shape.topk}_maxout{shape.max_output_size}.pt"
    )


def check_or_save_snapshot(
    case: dict[str, Any],
    rank: int,
    shape: Shape,
    snapshot_dir: str | None,
    reference_dir: str | None,
    rtol: float,
    atol: float,
) -> bool | None:
    payload = {
        "out": case["out"].detach().cpu(),
        "expert_token_nums": case["expert_token_nums"].detach().cpu(),
    }
    if snapshot_dir:
        path = snapshot_path(snapshot_dir, rank, shape)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, path)
    if not reference_dir:
        return None
    reference = torch.load(snapshot_path(reference_dir, rank, shape), map_location="cpu")
    torch.testing.assert_close(payload["out"], reference["out"], rtol=rtol, atol=atol)
    torch.testing.assert_close(payload["expert_token_nums"], reference["expert_token_nums"], rtol=0, atol=0)
    return True


def worker(
    rank: int,
    world_size: int,
    device: int,
    port: int,
    args: argparse.Namespace,
    result_queue: mp.Queue,
) -> None:
    try:
        torch_npu.npu.set_device(device)
        dist.init_process_group(
            backend="hccl",
            rank=rank,
            world_size=world_size,
            init_method=f"tcp://127.0.0.1:{port}",
        )
        group = hcomm_name(rank)
        torch_npu.npu.config.allow_internal_format = True

        base = Shape(0, args.k, args.n, args.experts_per_rank, args.topk, args.max_output_size)
        static = make_static_inputs(base, rank)
        for m in args.m:
            shape = Shape(m, args.k, args.n, args.experts_per_rank, args.topk, args.max_output_size)
            case = make_case_inputs(shape, rank, world_size)

            for _ in range(args.warmup):
                call_op(case, static, group, shape, args.use_active_mask)
            torch.npu.synchronize()
            dist.barrier()

            times_ms: list[float] = []
            start = torch.npu.Event(enable_timing=True)
            end = torch.npu.Event(enable_timing=True)
            for _ in range(args.iterations):
                dist.barrier()
                start.record()
                call_op(case, static, group, shape, args.use_active_mask)
                end.record()
                torch.npu.synchronize()
                times_ms.append(float(start.elapsed_time(end)))

            call_op(case, static, group, shape, args.use_active_mask)
            torch.npu.synchronize()
            out_first = case["out"].clone()
            counts_first = case["expert_token_nums"].clone()
            call_op(case, static, group, shape, args.use_active_mask)
            torch.npu.synchronize()
            deterministic = torch.allclose(case["out"], out_first, rtol=args.rtol, atol=args.atol)
            deterministic = deterministic and torch.equal(case["expert_token_nums"], counts_first)
            reference_ok = check_or_save_snapshot(
                case,
                rank,
                shape,
                args.snapshot_dir,
                args.reference_dir,
                args.rtol,
                args.atol,
            )
            result_queue.put(
                RankResult(
                    rank=rank,
                    shape=asdict(shape),
                    times_ms=times_ms,
                    deterministic=bool(deterministic),
                    reference_ok=reference_ok,
                    output_sum=float(case["out"].float().sum().item()),
                    output_abs_max=float(case["out"].float().abs().max().item()),
                )
            )
        dist.destroy_process_group()
    except Exception:
        result_queue.put({"rank": rank, "error": traceback.format_exc()})
        raise


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(fraction * len(ordered)) - 1))
    return ordered[index]


def aggregate(results: list[RankResult], iterations: int) -> list[dict[str, Any]]:
    grouped: dict[int, list[RankResult]] = {}
    for result in results:
        grouped.setdefault(result.shape["m"], []).append(result)
    summary = []
    for m, ranks in sorted(grouped.items()):
        rank_count = len(ranks)
        if any(len(item.times_ms) != iterations for item in ranks):
            raise RuntimeError(f"incomplete timing data for m={m}")
        max_rank_times = [max(item.times_ms[i] for item in ranks) for i in range(iterations)]
        summary.append(
            {
                "shape": ranks[0].shape,
                "rank_count": rank_count,
                "latency_ms": {
                    "min": min(max_rank_times),
                    "mean": statistics.fmean(max_rank_times),
                    "p50": statistics.median(max_rank_times),
                    "p90": percentile(max_rank_times, 0.90),
                    "p99": percentile(max_rank_times, 0.99),
                    "max": max(max_rank_times),
                },
                "deterministic": all(item.deterministic for item in ranks),
                "reference_ok": all(item.reference_ok is not False for item in ranks),
                "rank_outputs": [
                    {
                        "rank": item.rank,
                        "sum": item.output_sum,
                        "abs_max": item.output_abs_max,
                    }
                    for item in sorted(ranks, key=lambda value: value.rank)
                ],
            }
        )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--devices", type=parse_int_list, default=[0, 1])
    parser.add_argument("--m", type=parse_int_list, default=[1, 8, 32, 64, 128, 256])
    parser.add_argument("--k", type=int, default=7168)
    parser.add_argument("--n", type=int, default=4096)
    parser.add_argument("--experts-per-rank", type=int, default=32)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--max-output-size", type=int, default=4096)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--use-active-mask", action="store_true")
    parser.add_argument("--snapshot-dir")
    parser.add_argument("--reference-dir")
    parser.add_argument("--output-json")
    parser.add_argument("--rtol", type=float, default=1e-2)
    parser.add_argument("--atol", type=float, default=1e-2)
    args = parser.parse_args()
    if args.n % 2:
        parser.error("--n must be even because SwiGLU halves the GMM1 output")
    if args.k <= 0 or args.n <= 0 or args.experts_per_rank <= 0 or args.topk <= 0:
        parser.error("shape arguments must be positive")
    if args.warmup < 0 or args.iterations <= 0:
        parser.error("warmup must be non-negative and iterations must be positive")
    return args


def main() -> None:
    args = parse_args()
    world_size = len(args.devices)
    port = free_port()
    context = mp.get_context("spawn")
    result_queue = context.Queue()
    processes = [
        context.Process(target=worker, args=(rank, world_size, device, port, args, result_queue))
        for rank, device in enumerate(args.devices)
    ]
    for process in processes:
        process.start()

    expected = world_size * len(args.m)
    raw_results: list[RankResult] = []
    errors: list[dict[str, Any]] = []
    received = 0
    while received < expected:
        try:
            item = result_queue.get(timeout=5)
        except queue.Empty:
            if any(process.is_alive() for process in processes):
                continue
            errors.append({"error": "benchmark workers exited before returning all results"})
            break
        received += 1
        if isinstance(item, dict):
            errors.append(item)
            break
        raw_results.append(item)

    if errors:
        for process in processes:
            if process.is_alive():
                process.terminate()
    for process in processes:
        process.join()
    if errors or any(process.exitcode != 0 for process in processes):
        for error in errors:
            print(error["error"], file=sys.stderr)
        raise SystemExit("one or more benchmark workers failed")

    summary = aggregate(raw_results, args.iterations)
    payload = {"config": vars(args), "results": summary}
    print(json.dumps(payload, indent=2))
    if args.output_json:
        output = Path(args.output_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
