# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Measure block-FP8 scale resolution on an NPU with native FP8 tensor support.

Run the same script on base and candidate checkouts. The measurement includes
the dense output allocation and excludes subsequent MXFP8 quantization/model
loading. Incremental peak memory is the PyTorch allocator peak, including output.
"""

import argparse
import json
import statistics

import torch
import torch_npu  # noqa: F401

from vllm_ascend.quantization.methods.w8a8.fp8_block import resolve_block_scales


def benchmark(rows: int, cols: int, dtype: torch.dtype, repetitions: int) -> dict:
    block_size = 128
    weight = torch.randn((rows, cols), device="npu").to(torch.float8_e4m3fn)
    scale_shape = ((rows + block_size - 1) // block_size, (cols + block_size - 1) // block_size)
    scales = torch.rand(scale_shape, device="npu") + 0.001

    def run():
        return resolve_block_scales(weight, scales, block_size, block_size, dtype)

    for _ in range(5):
        run()
    torch.npu.synchronize()
    samples = []
    for _ in range(5):
        start, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
        start.record()
        for _ in range(repetitions):
            run()
        end.record()
        torch.npu.synchronize()
        samples.append(start.elapsed_time(end) / repetitions)
    before = torch.npu.memory_allocated()
    torch.npu.reset_peak_memory_stats()
    output = run()
    torch.npu.synchronize()
    peak = torch.npu.max_memory_allocated() - before
    assert output.shape == weight.shape
    return {
        "shape": [rows, cols],
        "dtype": str(dtype),
        "median_ms": statistics.median(samples),
        "samples_ms": samples,
        "incremental_peak_bytes": peak,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shapes", nargs="+", default=["34816x5120", "5120x17408", "16384x5120"])
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--repetitions", type=int, default=3)
    args = parser.parse_args()
    if args.repetitions <= 0:
        parser.error("Repetitions must be positive.")
    torch.manual_seed(42)
    for shape in args.shapes:
        try:
            rows, cols = map(int, shape.split("x"))
        except ValueError:
            parser.error(f"Expected ROWSxCOLS, got {shape!r}.")
        if min(rows, cols) <= 0:
            parser.error("Shape dimensions must be positive.")
        print(json.dumps(benchmark(rows, cols, getattr(torch, args.dtype), args.repetitions)), flush=True)


if __name__ == "__main__":
    main()
