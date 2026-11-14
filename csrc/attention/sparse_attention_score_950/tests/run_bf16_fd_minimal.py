#!/usr/bin/env python3
"""Run one BF16 shape that automatically selects FlashDecoding."""

import math
import sys
from pathlib import Path

import torch

# Make the repository's Python wrapper importable when this file is run directly.
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "torch_extension"))

from cann_ops_transformer.ops.sparse_attention_score import (  # noqa: E402
    npu_sparse_attention_score,
)


def run_once(
    query,
    key,
    value,
    select_idx,
    block_table,
    select_num_idx,
    actual_seq_lengths,
    actual_seq_lengths_kv,
    kv_heads,
    head_dim,
    block_size,
    top_k,
):
    return npu_sparse_attention_score(
        query,
        key,
        value,
        select_idx,
        block_table,
        select_num_idx=select_num_idx,
        actual_seq_lengths=actual_seq_lengths,
        actual_seq_lengths_kv=actual_seq_lengths_kv,
        num_key_value_heads=kv_heads,
        scale_value=1.0 / math.sqrt(head_dim),
        block_size=block_size,
        top_k=top_k,
        inner_precise=4,
    )


def main():
    torch.manual_seed(2026)

    # This shape has one base task and 16 selected KV blocks, so Arch35 Host
    # tiling automatically selects the BF16 FD path (key 10006).
    q_tokens = 1
    q_heads = 16
    kv_heads = 1
    head_dim = 128
    block_size = 128
    top_k = 16
    kv_seq_len = block_size * top_k

    query = torch.randn(q_tokens, q_heads, head_dim, dtype=torch.bfloat16).npu()
    key = torch.randn(top_k, block_size, kv_heads, head_dim, dtype=torch.bfloat16).npu()
    value = torch.randn(top_k, block_size, kv_heads, head_dim, dtype=torch.bfloat16).npu()

    # Logical KV block IDs and physical block IDs both use 0..15 here.
    select_idx = torch.arange(top_k, dtype=torch.int32).reshape(kv_heads, q_tokens, top_k).npu()
    block_table = torch.arange(top_k, dtype=torch.int32).reshape(1, top_k).npu()
    select_num_idx = torch.full((kv_heads, q_tokens), top_k, dtype=torch.int32).npu()
    actual_seq_lengths = torch.tensor([q_tokens], dtype=torch.int32).npu()
    actual_seq_lengths_kv = torch.tensor([kv_seq_len], dtype=torch.int32).npu()

    fd_output = run_once(
        query,
        key,
        value,
        select_idx,
        block_table,
        select_num_idx,
        actual_seq_lengths,
        actual_seq_lengths_kv,
        kv_heads,
        head_dim,
        block_size,
        top_k,
    )

    torch.npu.synchronize()
    fd_cpu = fd_output.cpu()

    print(f"output.shape={tuple(fd_cpu.shape)}")
    print(f"output.dtype={fd_cpu.dtype}")
    print(f"fd[0, 0, :8]={fd_cpu[0, 0, :8].float()}")
    assert torch.isfinite(fd_cpu.float()).all()
    print("PASS: automatic FD output is finite.")


if __name__ == "__main__":
    main()
