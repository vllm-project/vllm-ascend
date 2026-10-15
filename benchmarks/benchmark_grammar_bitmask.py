"""Benchmark the Ascend MRV2 grammar bitmask kernel.

Example:
    python benchmarks/benchmark_grammar_bitmask.py --masks 64 --vocab-size 151936
"""

import argparse
import time

import torch
import torch_npu  # noqa: F401
from vllm.triton_utils import triton

from vllm_ascend.ops.triton.v2.apply_grammar_bitmask import _apply_grammar_bitmask_kernel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--masks", type=int, default=64)
    parser.add_argument("--vocab-size", type=int, default=151936)
    parser.add_argument("--block-size", type=int, default=8192)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()

    device = torch.device("npu:0")
    mask_stride = 8
    masks = args.masks
    vocab_size = args.vocab_size
    words = (vocab_size + 31) // 32
    logits = torch.randn(masks, vocab_size, dtype=torch.bfloat16, device=device)
    mapping = torch.arange(masks, dtype=torch.int32, device=device) * mask_stride
    cu_num_logits = torch.arange(0, masks + 1, dtype=torch.int32, device=device)
    bitmask = torch.randint(
        -(2**31),
        2**31 - 1,
        (masks, words),
        dtype=torch.int32,
        device=device,
    )
    grid = (masks, triton.cdiv(vocab_size, args.block_size))

    def launch() -> None:
        _apply_grammar_bitmask_kernel[grid](
            logits,
            logits.stride(0),
            mapping,
            cu_num_logits,
            bitmask,
            bitmask.stride(0),
            vocab_size,
            MASK_STRIDE=mask_stride,
            BLOCK_SIZE=args.block_size,
        )

    for _ in range(args.warmup):
        launch()
    torch.npu.synchronize()

    start = time.perf_counter()
    for _ in range(args.iterations):
        launch()
    torch.npu.synchronize()
    elapsed_ms = (time.perf_counter() - start) * 1000 / args.iterations

    print(
        f"masks={masks} vocab_size={vocab_size} block_size={args.block_size} "
        f"iterations={args.iterations} avg_ms={elapsed_ms:.4f}"
    )


if __name__ == "__main__":
    main()
