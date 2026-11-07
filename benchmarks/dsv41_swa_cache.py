# SPDX-License-Identifier: Apache-2.0
"""Microbenchmark DeepSeek-V4.1 SWA cache preparation on Ascend A2/A3.

Example:
    python benchmarks/dsv41_swa_cache.py --tokens 1 8 32 128 512 2048 8192
"""

import argparse
import json
import statistics

import torch
import torch_npu

from vllm_ascend.device.device_op import DeviceOperator
from vllm_ascend.ops.triton.dsv41_swa_cache import dsv41_rmsnorm_rope_cache

HEAD_DIM = 512
NOPE_DIM = 448
ROPE_DIM = HEAD_DIM - NOPE_DIM
EPS = 1e-6


def _measure(function, warmups: int, repeats: int) -> float:
    for _ in range(warmups):
        function()
    torch.npu.synchronize()
    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        function()
    end.record()
    torch.npu.synchronize()
    return start.elapsed_time(end) * 1000 / repeats


def _case(tokens: int, warmups: int, repeats: int, samples: int) -> dict[str, float | int | list[float]]:
    projected = torch.randn(tokens, HEAD_DIM, device="npu", dtype=torch.bfloat16)
    weight = torch.randn(HEAD_DIM, device="npu", dtype=torch.bfloat16)
    angles = torch.randn(tokens, 1, 1, ROPE_DIM // 2, device="npu")
    cos = angles.cos().repeat_interleave(2, dim=-1)
    sin = angles.sin().repeat_interleave(2, dim=-1)
    block_size = 32
    blocks = (tokens + block_size - 1) // block_size
    cache = torch.empty(blocks, block_size, 1, HEAD_DIM, device="npu", dtype=torch.bfloat16)
    flat_slots = torch.arange(tokens, device="npu", dtype=torch.int32)
    slots = torch.stack((flat_slots // block_size, flat_slots % block_size), -1)

    def baseline():
        value, _ = torch_npu.npu_rms_norm(projected, weight, EPS)
        value = value.view(tokens, 1, HEAD_DIM)
        DeviceOperator.apply_partial_rotary_inplace(value, cos, sin, start=NOPE_DIM, end=HEAD_DIM)
        torch.ops._C_ascend.npu_scatter_nd_update_sk(cache.squeeze(-2), slots, value.squeeze(1).contiguous())

    def fused():
        dsv41_rmsnorm_rope_cache(projected, weight, cos, sin, cache, slots, EPS, NOPE_DIM)

    baseline_samples = []
    fused_samples = []
    for sample in range(samples):
        ordered = ((baseline, baseline_samples), (fused, fused_samples))
        if sample % 2:
            ordered = ordered[::-1]
        for function, results in ordered:
            results.append(_measure(function, warmups, repeats))
    baseline_us = statistics.median(baseline_samples)
    fused_us = statistics.median(fused_samples)
    return {
        "tokens": tokens,
        "baseline_us": baseline_us,
        "fused_us": fused_us,
        "speedup": baseline_us / fused_us,
        "baseline_samples_us": baseline_samples,
        "fused_samples_us": fused_samples,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, nargs="+", default=[1, 4, 8, 16, 32, 64, 128, 256, 512, 2048, 8192])
    parser.add_argument("--warmups", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--samples", type=int, default=5)
    args = parser.parse_args()
    results = [_case(tokens, args.warmups, args.repeats, args.samples) for tokens in args.tokens]
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
