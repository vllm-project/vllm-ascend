# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from collections.abc import Callable
from typing import Any

import torch
import torch.distributed as dist
import torch_npu  # noqa: F401

from vllm_ascend.ops.triton.sfa_cp import sfa_dcp_a2a_fused_combine


def _legacy_all_to_all(
    tensor: torch.Tensor,
    scatter_dim: int,
    group: dist.ProcessGroup,
) -> torch.Tensor:
    world_size = dist.get_world_size(group)
    scatter_size = tensor.shape[scatter_dim]
    send = tensor.movedim(scatter_dim, 0).contiguous()
    recv = torch.empty_like(send)
    dist.all_to_all_single(recv, send, group=group)
    return recv.view(world_size, scatter_size // world_size, *send.shape[1:])


def _legacy_two_a2a_merge(
    output: torch.Tensor,
    lse: torch.Tensor,
    scatter_dim: int,
    group: dist.ProcessGroup,
) -> torch.Tensor:
    output_recv = _legacy_all_to_all(output, scatter_dim, group)
    lse_recv = _legacy_all_to_all(lse, scatter_dim, group).squeeze(-1)
    lse_recv = lse_recv.masked_fill(~torch.isfinite(lse_recv), float("-inf"))
    weights = torch.nan_to_num(torch.softmax(lse_recv, dim=0), nan=0.0)
    merged = (output_recv.float() * weights.unsqueeze(-1)).sum(dim=0)
    token_dim = 1 if scatter_dim == 0 else 2
    return merged.movedim(token_dim - 1, 0).contiguous().to(output.dtype)


def _run_once(fn: Callable[[], torch.Tensor]) -> tuple[float, float]:
    torch.npu.synchronize()
    start_event = torch.npu.Event(enable_timing=True)
    end_event = torch.npu.Event(enable_timing=True)
    wall_start = time.perf_counter_ns()
    start_event.record()
    result = fn()
    end_event.record()
    end_event.synchronize()
    wall_ms = (time.perf_counter_ns() - wall_start) / 1_000_000.0
    device_ms = start_event.elapsed_time(end_event)
    # Keep the output live until the measured dependency has completed.
    del result
    return float(device_ms), wall_ms


def _summary(values: list[float]) -> dict[str, float | list[float]]:
    mean = statistics.fmean(values)
    stdev = statistics.stdev(values) if len(values) > 1 else 0.0
    return {
        "raw_ms": values,
        "median_ms": statistics.median(values),
        "mean_ms": mean,
        "stdev_ms": stdev,
        "cv": stdev / mean if mean else 0.0,
        "min_ms": min(values),
        "max_ms": max(values),
    }


def _measure_alternating(
    baseline: Callable[[], torch.Tensor],
    candidate: Callable[[], torch.Tensor],
    warmup: int,
    repeats: int,
) -> dict[str, Any]:
    for _ in range(warmup):
        baseline()
        candidate()
    torch.npu.synchronize()

    samples = {
        "baseline": {"device": [], "wall": []},
        "candidate": {"device": [], "wall": []},
    }
    for repeat in range(repeats):
        order = (("baseline", baseline), ("candidate", candidate))
        if repeat % 2:
            order = tuple(reversed(order))
        for name, fn in order:
            device_ms, wall_ms = _run_once(fn)
            samples[name]["device"].append(device_ms)
            samples[name]["wall"].append(wall_ms)
    return samples


def _max_across_ranks(values: list[float], device: torch.device) -> list[float]:
    timing = torch.tensor(values, dtype=torch.float32, device=device)
    dist.all_reduce(timing, op=dist.ReduceOp.MAX)
    return [float(value) for value in timing.cpu().tolist()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare legacy two-A2A SFA DCP merge with the fused one-A2A path.")
    parser.add_argument("--scatter-tokens", action="store_true")
    parser.add_argument("--tokens", type=int, default=8)
    parser.add_argument("--heads", type=int, default=64)
    parser.add_argument("--head-dim", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=30)
    args = parser.parse_args()

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(local_rank)
    device = torch.device(f"npu:{local_rank}")
    dist.init_process_group("hccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    scatter_dim = 0 if args.scatter_tokens else 1
    scatter_size = args.tokens if scatter_dim == 0 else args.heads
    if scatter_size % world_size:
        raise ValueError(f"scatter size {scatter_size} must be divisible by world size {world_size}.")

    torch.manual_seed(2026 + rank)
    output = torch.randn(
        args.tokens,
        args.heads,
        args.head_dim,
        dtype=torch.bfloat16,
        device=device,
    )
    lse = torch.randn(
        args.tokens,
        args.heads,
        1,
        dtype=torch.float32,
        device=device,
    )
    baseline = lambda: _legacy_two_a2a_merge(output, lse, scatter_dim, dist.group.WORLD)
    candidate = lambda: sfa_dcp_a2a_fused_combine(
        output,
        lse,
        world_size,
        scatter_dim,
        dist.group.WORLD,
    )

    expected = baseline()
    actual = candidate()
    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)
    error = (actual.float() - expected.float()).abs()
    max_abs = float(error.max().item())
    max_rel = float((error / expected.float().abs().clamp_min(1e-8)).max().item())

    dist.barrier()
    samples = _measure_alternating(
        baseline,
        candidate,
        args.warmup,
        args.repeats,
    )
    dist.barrier()
    for implementation in ("baseline", "candidate"):
        for timing_kind in ("device", "wall"):
            samples[implementation][timing_kind] = _max_across_ranks(
                samples[implementation][timing_kind],
                device,
            )

    result = {
        "schema_version": 1,
        "world_size": world_size,
        "scatter_dim": scatter_dim,
        "scatter_mode": "tokens" if scatter_dim == 0 else "heads",
        "shape": [args.tokens, args.heads, args.head_dim],
        "dtype": "bfloat16",
        "warmup": args.warmup,
        "repeats": args.repeats,
        "correctness": {
            "atol": 0.02,
            "rtol": 0.02,
            "max_abs": max_abs,
            "max_rel": max_rel,
        },
        "baseline_two_a2a": {
            "device": _summary(samples["baseline"]["device"]),
            "wall": _summary(samples["baseline"]["wall"]),
        },
        "candidate_one_a2a": {
            "device": _summary(samples["candidate"]["device"]),
            "wall": _summary(samples["candidate"]["wall"]),
        },
    }
    baseline_median = result["baseline_two_a2a"]["wall"]["median_ms"]
    candidate_median = result["candidate_one_a2a"]["wall"]["median_ms"]
    result["wall_speedup"] = baseline_median / candidate_median
    result["wall_relative_improvement"] = 1.0 - candidate_median / baseline_median

    dist.destroy_process_group()
    if rank == 0:
        print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
