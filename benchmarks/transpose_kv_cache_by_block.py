# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Measure in-place KV transpose latency for dense and interleaved caches.

Run on the same NPU/build configuration before and after the change:
    python benchmarks/transpose_kv_cache_by_block.py --layout dense
The old operator rejects --layout interleaved. Compare both layouts on the fix.
Reports synchronized wall time per call, including host dispatch overhead.
"""

import argparse
import json
import statistics
import time

import torch
import torch_npu  # noqa: F401

from vllm_ascend.utils import enable_custom_op


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layout", choices=("dense", "interleaved"), default="interleaved")
    parser.add_argument("--layers", type=int, default=16)
    parser.add_argument("--blocks", type=int, default=128)
    parser.add_argument("--selected-blocks", type=int, default=33)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--split", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=100)
    args = parser.parse_args()
    if min(vars(args)[key] for key in vars(args) if key != "layout") <= 0:
        parser.error("all dimensions and iteration counts must be positive")
    if args.selected_blocks > args.blocks or args.heads % args.split:
        parser.error("selected blocks must fit the cache and split must divide heads")
    torch.npu.set_device(0)
    enable_custom_op()
    k_caches, v_caches = [], []
    shape = (args.blocks, args.block_size, args.heads, 128)
    for _ in range(args.layers):
        if args.layout == "dense":
            k_caches.append(torch.randn(shape, device="npu", dtype=torch.bfloat16))
            v_caches.append(torch.randn(shape, device="npu", dtype=torch.bfloat16))
        else:
            backing = torch.randn(args.blocks, 2, *shape[1:], device="npu", dtype=torch.bfloat16)
            k_caches.append(backing[:, 0])
            v_caches.append(backing[:, 1])
    ids = torch.randperm(args.blocks, device="npu", dtype=torch.int64)[: args.selected_blocks]

    def run():
        torch.ops._C_ascend.transpose_kv_cache_by_block(
            k_caches, v_caches, ids, args.block_size, args.heads, 128, args.split, args.layers
        )

    for _ in range(5):
        run()
    torch.npu.synchronize()
    samples = []
    for _ in range(5):
        start = time.perf_counter()
        for _ in range(args.repeats):
            run()
        torch.npu.synchronize()
        samples.append((time.perf_counter() - start) * 1e6 / args.repeats)
    print(json.dumps({**vars(args), "median_us": statistics.median(samples), "samples_us": samples}))


if __name__ == "__main__":
    main()
