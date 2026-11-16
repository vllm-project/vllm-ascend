# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
"""Compare the pre-cache and cached Gemma Norm paths on the same NPU.

Run from an installed vllm-ascend environment:
    python benchmarks/benchmark_gemma_norm_weight.py --tokens 1 4 16 128 4096
Results are local Norm latency, not model throughput or end-to-end latency.
"""

import argparse
import json
import statistics
import time

import torch
import torch_npu
from vllm.config import VllmConfig, set_current_vllm_config

from vllm_ascend.device.device_op import DeviceOperator
from vllm_ascend.ops.layernorm import AscendGemmaRMSNorm


def measure(fn, iterations):
    start, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
    torch.npu.synchronize()
    begin = time.perf_counter()
    start.record()
    for _ in range(iterations):
        fn()
    end.record()
    end.synchronize()
    wall_us = (time.perf_counter() - begin) * 1e6 / iterations
    return start.elapsed_time(end) * 1e3 / iterations, wall_us


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, nargs="+", default=[1, 4, 16, 128, 4096])
    parser.add_argument("--width", type=int, default=2048)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--dtype", choices=["bfloat16", "float16"], default="bfloat16")
    args = parser.parse_args()
    if min(*args.tokens, args.width, args.iterations, args.rounds) <= 0:
        parser.error("tokens, width, iterations and rounds must be positive")
    dtype = getattr(torch, args.dtype)
    torch.manual_seed(17)
    with set_current_vllm_config(VllmConfig()):
        norm = AscendGemmaRMSNorm(args.width).to(device="npu", dtype=dtype)
    norm.weight.weight_loader(norm.weight, torch.randn_like(norm.weight))
    print(
        json.dumps(
            {
                "device": torch.npu.get_device_name(),
                "torch": torch.__version__,
                "torch_npu": torch_npu.__version__,
                "dtype": args.dtype,
                "width": args.width,
                "cache_bytes": norm._effective_weight.nbytes,
            }
        )
    )
    for rows in args.tokens:
        x = torch.randn(rows, args.width, device="npu", dtype=dtype)
        for with_residual in (False, True):
            residual = torch.randn_like(x) if with_residual else None

            def baseline(x=x, residual=residual):
                if residual is not None:
                    return torch_npu.npu_add_rms_norm(x, residual, 1.0 + norm.weight, norm.variance_epsilon)[0]
                return DeviceOperator.npu_gemma_rms_norm(x, norm.weight, norm.variance_epsilon)

            def cached(x=x, residual=residual):
                output = norm.forward_oot(x, residual)
                return output[0] if residual is not None else output

            torch.testing.assert_close(cached(), baseline(), rtol=0, atol=0)
            for _ in range(10):
                baseline()
                cached()
            times = {"baseline": [], "cached": []}
            for index in range(args.rounds):
                order = [("baseline", baseline), ("cached", cached)]
                if index % 2:
                    order.reverse()
                for name, fn in order:
                    times[name].append(measure(fn, args.iterations))
            print(
                json.dumps(
                    {
                        "rows": rows,
                        "residual": with_residual,
                        "event_median_us": {k: statistics.median(t[0] for t in v) for k, v in times.items()},
                        "wall_median_us": {k: statistics.median(t[1] for t in v) for k, v in times.items()},
                        "rounds_event_wall_us": times,
                    }
                )
            )


if __name__ == "__main__":
    main()
