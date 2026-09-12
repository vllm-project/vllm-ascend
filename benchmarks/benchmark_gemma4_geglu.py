# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Compare Gemma4's unfused and A5 fused activation using device graph timing.

Example: ASCEND_RT_VISIBLE_DEVICES=2 python benchmarks/benchmark_gemma4_geglu.py
Results cover only activation (optionally followed by MXFP4 quantization),
not a full expert layer or end-to-end serving throughput.
"""

import argparse
import json
import statistics

import torch
import torch_npu  # noqa: F401

from vllm_ascend.device.device_config import is_950
from vllm_ascend.utils import enable_custom_op


@torch.inference_mode()
def measure(fn, x, with_quant, iterations, repeats):
    def compute():
        y = fn(x)
        if with_quant:
            return torch_npu.npu_dynamic_mx_quant(y, dst_type=torch_npu.float4_e2m1fn_x2)
        return y

    for _ in range(5):
        compute()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        for _ in range(iterations):
            compute()
    graph.replay()
    torch.npu.synchronize()
    durations = []
    for _ in range(repeats):
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        durations.append(start.elapsed_time(end) * 1000 / iterations)
    return statistics.median(durations)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, nargs="+", default=[1, 8, 32, 128, 512])
    parser.add_argument("--width", type=int, default=704)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=7)
    args = parser.parse_args()
    if min(*args.rows, args.width, args.iterations, args.repeats) <= 0:
        parser.error("rows, width, iterations, and repeats must be positive")
    if not is_950():
        parser.error("this benchmark requires Ascend 950")
    torch.npu.set_device(0)
    enable_custom_op()
    # Import ops first to avoid the existing
    # device_op -> ops -> moe_mlp -> device_op import cycle.
    import vllm_ascend.ops  # noqa: F401
    from vllm_ascend.device.device_op import A5DeviceAdaptor, BaseDeviceAdaptor

    torch.manual_seed(0)
    for rows in args.rows:
        x = torch.randn(rows, 2 * args.width, dtype=torch.bfloat16).npu()
        for with_quant in (False, True):
            native = measure(BaseDeviceAdaptor.gelu_tanh_and_mul, x, with_quant, args.iterations, args.repeats)
            fused = measure(A5DeviceAdaptor.gelu_tanh_and_mul, x, with_quant, args.iterations, args.repeats)
            print(
                json.dumps(
                    {
                        "rows": rows,
                        "width": args.width,
                        "with_mxfp4_quant": with_quant,
                        "native_us": native,
                        "fused_us": fused,
                        "speedup": native / fused,
                    }
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
