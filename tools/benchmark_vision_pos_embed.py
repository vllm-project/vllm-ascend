# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Ascend-only operator A/B; optionally use a real checkpoint position table."""

import argparse
import json
import math
import time

import torch
import torch_npu  # noqa: F401
from safetensors import safe_open
from vllm.model_executor.models.qwen3_vl import pos_embed_interpolate_native

from vllm_ascend.device.device_config import is_310p
from vllm_ascend.ops.triton.vision_pos_embed import vision_pos_embed_interpolate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weight-file", help="safetensors shard containing the learned position table")
    parser.add_argument("--weight-key", default="visual.pos_embed.weight")
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="bfloat16")
    parser.add_argument("--grid", type=int, nargs=3, default=[1, 32, 32], metavar=("T", "H", "W"))
    parser.add_argument("--merge", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()
    if not torch.npu.is_available() or is_310p():
        parser.error("requires a Triton-capable Ascend NPU")
    if args.iterations <= 0:
        parser.error("iterations must be positive")
    dtype = getattr(torch, args.dtype)
    if args.weight_file:
        with safe_open(args.weight_file, framework="pt", device="cpu") as checkpoint:
            weight = checkpoint.get_tensor(args.weight_key).to(device="npu", dtype=dtype).contiguous()
    else:
        torch.manual_seed(37948)
        weight = torch.randn(48 * 48, 1152, device="npu", dtype=dtype)
    side = math.isqrt(weight.shape[0])
    if side * side != weight.shape[0]:
        parser.error("position table must be square")
    inputs = (weight, *args.grid, side, args.merge, dtype)
    functions = {"native": pos_embed_interpolate_native, "fused": vision_pos_embed_interpolate}
    start = time.perf_counter()
    actual = functions["fused"](*inputs)
    torch.npu.synchronize()
    first_call_ms = (time.perf_counter() - start) * 1000
    expected = functions["native"](*inputs)
    error = (actual.float() - expected.float()).abs()
    rtol, atol, max_error = {
        torch.float32: (2**-10, 2**-16, 0.01),
        torch.float16: (2**-9, 2**-9, 0.1),
        torch.bfloat16: (2**-6, 2**-6, 1.0),
    }[dtype]
    fraction = (error <= atol + rtol * expected.float().abs()).float().mean().item()
    maximum = error.max().item()
    if fraction < 0.99 or not maximum <= max_error:
        raise AssertionError(f"accuracy failed: fraction={fraction}, max_abs={maximum}")
    for function in functions.values():
        for _ in range(5):
            function(*inputs)
    torch.npu.synchronize()
    samples = {name: [] for name in functions}
    # Alternate order to reduce drift. Event intervals include device idle time
    # caused by host launches; they are not isolated profiler kernel durations.
    for iteration in range(args.iterations):
        for name in list(functions) if iteration % 2 == 0 else list(reversed(functions)):
            begin = torch.npu.Event(enable_timing=True)
            end = torch.npu.Event(enable_timing=True)
            torch.npu.synchronize()
            start = time.perf_counter()
            begin.record()
            result = functions[name](*inputs)
            end.record()
            end.synchronize()
            samples[name].append((begin.elapsed_time(end), (time.perf_counter() - start) * 1000))
            del result
    print(
        json.dumps(
            {
                "device": torch.npu.get_device_name(),
                "torch": torch.__version__,
                "cann": getattr(torch.version, "cann", None),
                "weight_source": args.weight_file or "synthetic (not model accuracy evidence)",
                "weight_key": args.weight_key,
                "shape": list(weight.shape),
                "grid": args.grid,
                "dtype": args.dtype,
                "first_fused_call_wall_ms_including_possible_jit": first_call_ms,
                "accuracy_fraction": fraction,
                "max_abs_error": maximum,
                "mean_ms": {
                    name: {
                        "device_event_interval": sum(s[0] for s in values) / len(values),
                        "synchronized_host_wall": sum(s[1] for s in values) / len(values),
                    }
                    for name, values in samples.items()
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
