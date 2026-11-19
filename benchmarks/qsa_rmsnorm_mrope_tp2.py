import argparse
import json
import time

import torch
import torch_npu
from vllm.triton_utils import triton

from vllm_ascend.ops.triton.linearnorm.qsa_rmsnorm_mrope import (
    _qsa_rmsnorm_mrope_fused_heads_kernel,
    _qsa_rmsnorm_mrope_kernel,
)


def launch_baseline(tensor, weight, cache, positions, output):
    tokens, heads, head_dim = tensor.shape
    _qsa_rmsnorm_mrope_kernel[(tokens, heads)](
        tensor,
        weight,
        cache,
        output,
        tensor.stride(0),
        tensor.stride(1),
        0,
        cache.stride(-2),
        output.stride(0),
        output.stride(1),
        num_heads=heads,
        head_dim=head_dim,
        eps=1e-6,
        mrope_section_h=11,
        mrope_section_w=10,
        rope_dim=64,
        has_mrope=True,
        block_dim=triton.next_power_of_2(head_dim),
        positions_ptr=positions,
        position_stride_axis=positions.stride(0),
        position_stride_token=positions.stride(-1),
        cache_rows=cache.shape[0],
        GATHER_CACHE=True,
        num_warps=1,
        num_stages=1,
    )


def launch_fused(tensor, weight, cache, positions, output):
    tokens, heads, head_dim = tensor.shape
    _qsa_rmsnorm_mrope_fused_heads_kernel[(tokens,)](
        tensor,
        weight,
        cache,
        output,
        tensor.stride(0),
        tensor.stride(1),
        cache.stride(-2),
        output.stride(0),
        output.stride(1),
        num_heads=heads,
        head_dim=head_dim,
        eps=1e-6,
        mrope_section_h=11,
        mrope_section_w=10,
        rope_dim=64,
        block_dim=triton.next_power_of_2(head_dim),
        positions_ptr=positions,
        position_stride_axis=positions.stride(0),
        position_stride_token=positions.stride(-1),
        cache_rows=cache.shape[0],
        num_warps=1,
        num_stages=1,
    )


def measure(fn, args, warmup, repeats):
    for _ in range(warmup):
        fn(*args)
    torch_npu.npu.synchronize()
    samples = []
    for _ in range(repeats):
        begin = time.perf_counter_ns()
        fn(*args)
        torch_npu.npu.synchronize()
        samples.append((time.perf_counter_ns() - begin) / 1000.0)
    samples.sort()
    return {
        "p50_us": samples[len(samples) // 2],
        "p90_us": samples[int(len(samples) * 0.9)],
        "mean_us": sum(samples) / len(samples),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=20)
    args = parser.parse_args()
    torch.manual_seed(23)
    results = []
    for tokens in (8, 16, 24, 26, 32, 64):
        packed = torch.randn(tokens, 3, 128, dtype=torch.bfloat16, device="npu")
        tensor = packed[:, :2]
        weight = torch.randn(128, dtype=torch.bfloat16, device="npu") * 0.05
        cache = torch.randn(257, 64, dtype=torch.bfloat16, device="npu")
        positions = torch.randint(0, 257, (3, tokens), dtype=torch.int64, device="npu")
        baseline_out, fused_out = torch.empty_like(tensor), torch.empty_like(tensor)
        launch_baseline(tensor, weight, cache, positions, baseline_out)
        launch_fused(tensor, weight, cache, positions, fused_out)
        torch_npu.npu.synchronize()
        torch.testing.assert_close(fused_out, baseline_out, rtol=0, atol=0)
        baseline = measure(launch_baseline, (tensor, weight, cache, positions, baseline_out), args.warmup, args.repeats)
        fused = measure(launch_fused, (tensor, weight, cache, positions, fused_out), args.warmup, args.repeats)
        results.append(
            {
                "batch": tokens,
                "heads": 2,
                "baseline": baseline,
                "fused": fused,
                "speedup": baseline["p50_us"] / fused["p50_us"],
            }
        )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
