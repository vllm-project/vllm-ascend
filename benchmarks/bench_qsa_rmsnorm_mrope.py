#!/usr/bin/env python3
import argparse
import json
import statistics
import time

import torch
import torch_npu

from vllm_ascend.ops.triton.linearnorm.qsa_rmsnorm_mrope import triton_qsa_rmsnorm_mrope


def reference(tensor: torch.Tensor, weight: torch.Tensor, cos_sin: torch.Tensor, eps: float) -> torch.Tensor:
    values = tensor.float()
    reciprocal_std = torch.rsqrt(values.square().mean(dim=-1, keepdim=True) + eps)
    normalized = (values * reciprocal_std * (weight.float() + 1.0)).to(torch.bfloat16)
    rotary_offsets = torch.arange(32)
    h_mask = (rotary_offsets % 3 == 1) & (rotary_offsets < 33)
    w_mask = (rotary_offsets % 3 == 2) & (rotary_offsets < 30)
    axis = torch.where(h_mask, 1, torch.where(w_mask, 2, 0))
    token = torch.arange(tensor.shape[0])[:, None]
    offset = rotary_offsets[None, :]
    cos = cos_sin[axis[None, :], token, offset].float()
    sin = cos_sin[axis[None, :], token, offset + 32].float()
    first = normalized[..., :32]
    second = normalized[..., 32:64]
    out = normalized.clone()
    out[..., :32] = (
        (first.float() * cos[:, None, :]).to(torch.bfloat16) - (second.float() * sin[:, None, :]).to(torch.bfloat16)
    ).to(torch.bfloat16)
    out[..., 32:64] = (
        (second.float() * cos[:, None, :]).to(torch.bfloat16) + (first.float() * sin[:, None, :]).to(torch.bfloat16)
    ).to(torch.bfloat16)
    return out


def synchronize() -> None:
    torch_npu.npu.synchronize()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=4096)
    parser.add_argument("--heads", type=int, choices=(1, 4), default=4)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--inner", type=int, default=100)
    parser.add_argument("--profile-repeats", type=int, default=0)
    args = parser.parse_args()
    torch.manual_seed(17)
    tensor_cpu = torch.randn(args.tokens, args.heads, 128, dtype=torch.bfloat16)
    weight_cpu = torch.randn(128, dtype=torch.bfloat16) * 0.05
    cos_sin_cpu = torch.randn(3, args.tokens, 64, dtype=torch.bfloat16)
    expected = reference(tensor_cpu, weight_cpu, cos_sin_cpu, 1e-6)
    tensor = tensor_cpu.npu()
    weight = weight_cpu.npu()
    cos_sin = cos_sin_cpu.npu()

    def run() -> torch.Tensor:
        return triton_qsa_rmsnorm_mrope(tensor, weight, cos_sin, 1e-6, [11, 11, 10], 64)

    actual = run()
    synchronize()
    actual_cpu = actual.cpu()
    diff = (actual_cpu.float() - expected.float()).abs()
    torch.testing.assert_close(actual_cpu, expected, rtol=2e-2, atol=2e-2)
    for _ in range(args.warmup):
        run()
    synchronize()
    if args.profile_repeats:
        for _ in range(args.profile_repeats):
            run()
        synchronize()
    samples_us = []
    for _ in range(args.samples):
        synchronize()
        start = time.perf_counter_ns()
        for _ in range(args.inner):
            run()
        synchronize()
        samples_us.append((time.perf_counter_ns() - start) / args.inner / 1000.0)
    median_us = statistics.median(samples_us)
    mean_us = statistics.mean(samples_us)
    cv = statistics.pstdev(samples_us) / mean_us if mean_us else 0.0
    print(
        json.dumps(
            {
                "tokens": args.tokens,
                "heads": args.heads,
                "dtype": "bfloat16",
                "shape": list(tensor.shape),
                "max_abs": diff.max().item(),
                "median_us": median_us,
                "mean_us": mean_us,
                "cv": cv,
                "samples_us": samples_us,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
