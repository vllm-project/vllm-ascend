# SPDX-License-Identifier: Apache-2.0
"""Compare fused and separate Kimi latent RMSNorm/MXFP8 quantization on A5.

Run on one A5 NPU:
    python benchmark_kimi_rmsnorm_mxquant.py
"""

import argparse
from statistics import median

import torch
import torch_npu


def _latency_us(fn, warmup: int, repeats: int) -> float:
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()

    samples = []
    for _ in range(repeats):
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000)
    return median(samples)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, nargs="+", default=[8, 32, 64])
    parser.add_argument("--hidden-size", type=int, default=3584)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=100)
    args = parser.parse_args()

    torch.npu.set_device(0)
    gamma = torch.ones(args.hidden_size, device="npu", dtype=torch.bfloat16)
    print("tokens scale_alg fused_us separate_us speedup")
    with torch.inference_mode():
        for tokens in args.tokens:
            x = torch.randn(tokens, args.hidden_size, device="npu", dtype=torch.bfloat16)
            for scale_alg in (0, 1):

                def fused(x=x, scale_alg=scale_alg):
                    return torch.ops.npu.npu_rms_norm_dynamic_mx_quant(
                        x, gamma, epsilon=1e-6, scale_alg=scale_alg, dst_type=torch.float8_e4m3fn
                    )

                def separate(x=x, scale_alg=scale_alg):
                    normalized = torch_npu.npu_rms_norm(x, gamma, 1e-6)[0]
                    return torch_npu.npu_dynamic_mx_quant(normalized, dst_type=torch.float8_e4m3fn, scale_alg=scale_alg)

                fused_us = _latency_us(fused, args.warmup, args.repeats)
                separate_us = _latency_us(separate, args.warmup, args.repeats)
                print(
                    f"{tokens:>6} {scale_alg:>9} {fused_us:>8.2f} {separate_us:>11.2f} {separate_us / fused_us:>7.2f}x"
                )


if __name__ == "__main__":
    main()
