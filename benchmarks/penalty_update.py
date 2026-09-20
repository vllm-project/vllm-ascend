# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark the sampling penalty update with precomputed occurrence masks/counts.

Run this script against the base and candidate revisions on the same idle NPU:
    python benchmarks/penalty_update.py --batches 1 4 16 64 --vocab-sizes 32000 151936 248320

Neutral runtime penalty tensors keep repeated in-place updates finite. Histogram
construction and model inference are excluded. Reported times include eager
dispatch; this sampling operation normally runs outside the captured model graph.
"""

import argparse
import json
import statistics

import torch
import torch_npu  # noqa: F401

from vllm_ascend.ops.triton.penalty import _apply_all_penalties_triton
from vllm_ascend.ops.triton.triton_utils import get_vectorcore_num, init_device_properties_triton


def benchmark(batch: int, vocab: int, dtype: torch.dtype, repetitions: int, groups: int) -> dict:
    logits = torch.randn((batch, vocab), dtype=dtype, device="npu")
    prompt_mask = torch.rand((batch, vocab), device="npu") > 0.5
    counts = torch.randint(0, 4, (batch, vocab), dtype=torch.int32, device="npu")
    output_mask = counts > 0
    repetition = torch.ones(batch, device="npu")
    zero_penalty = torch.zeros(batch, device="npu")

    def run():
        _apply_all_penalties_triton(logits, prompt_mask, output_mask, counts, repetition, zero_penalty, zero_penalty)

    for _ in range(10):
        run()
    torch.npu.synchronize()
    samples = []
    for _ in range(groups):
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        start.record()
        for _ in range(repetitions):
            run()
        end.record()
        torch.npu.synchronize()
        samples.append(start.elapsed_time(end) * 1000 / repetitions)
    return {
        "batch": batch,
        "vocab": vocab,
        "dtype": str(dtype),
        "median_us": statistics.median(samples),
        "samples_us": samples,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batches", nargs="+", type=int, default=[1, 4, 16, 64])
    parser.add_argument("--vocab-sizes", nargs="+", type=int, default=[32000, 151936, 248320])
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float32")
    parser.add_argument("--repetitions", type=int, default=30)
    parser.add_argument("--groups", type=int, default=5)
    args = parser.parse_args()
    if min(args.batches + args.vocab_sizes + [args.repetitions, args.groups]) <= 0:
        parser.error("Batch sizes, vocabulary sizes, repetitions and groups must be positive.")
    torch.manual_seed(42)
    init_device_properties_triton()
    print(json.dumps({"vector_cores": get_vectorcore_num()}), flush=True)
    for batch in args.batches:
        for vocab in args.vocab_sizes:
            result = benchmark(batch, vocab, getattr(torch, args.dtype), args.repetitions, args.groups)
            print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
