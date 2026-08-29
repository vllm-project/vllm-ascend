# Copyright (c) 2026.
"""Shared helpers for the FA3 standalone UTs (not collected by pytest).

All correctness tests validate against an unambiguous float64 CPU reference:
per request b, gather its KV blocks in block-table order, compare with the
single query token under causal (optionally windowed) attention.
"""

import torch

try:
    from flash_attn_npu_3 import (
        flash_attn_with_kvcache as fa3_kvcache,
        get_scheduler_metadata,
    )
    HAS_FA3 = True
except ImportError:
    HAS_FA3 = False

# Dimensions mirror the production decode shape (Qwen3-style GQA, bf16).
# Note: FA3's kernel tiling is dimension-sensitive — smaller head counts take
# different internal paths; keep these at production-like values.
DTYPE = torch.bfloat16
BLOCK_SIZE = 128
HEAD_SIZE = 128
NUM_HEADS = 32
NUM_KV_HEADS = 8
GROUP = NUM_HEADS // NUM_KV_HEADS
SCALE = 1.0 / (HEAD_SIZE ** 0.5)


def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def make_block_table(batch: int, seqlens, width: int, pool: int, seed: int) -> torch.Tensor:
    """Block table with reordered (non-identity) physical block ids and
    vllm's -1 sentinel in each row's unallocated tail."""
    g = torch.Generator().manual_seed(seed)
    bt = torch.full((batch, width), -1, dtype=torch.int32)
    for b, s in enumerate(seqlens):
        nblk = ceil_div(s, BLOCK_SIZE)
        ids = torch.randperm(pool, generator=g)[:nblk]
        bt[b, :nblk] = ids
    return bt


def cpu_ref_decode(
    q: torch.Tensor,
    k_pool: torch.Tensor,
    v_pool: torch.Tensor,
    block_table: torch.Tensor,
    seqlens,
    window: int | None = None,
) -> torch.Tensor:
    """Float64 CPU reference for single-query-token decode attention.

    q: (batch, NUM_HEADS, HEAD_SIZE); k/v pool: (pool, BLOCK_SIZE,
    NUM_KV_HEADS, HEAD_SIZE). With *window* set, token i only attends
    positions [max(0, i - window + 1), i].
    """
    outs = []
    for b, seq_len in enumerate(seqlens):
        nblk = ceil_div(seq_len, BLOCK_SIZE)
        ids = block_table[b, :nblk].tolist()
        k_flat = torch.cat([k_pool[i] for i in ids], dim=0)[:seq_len]
        v_flat = torch.cat([v_pool[i] for i in ids], dim=0)[:seq_len]
        k_g = k_flat.repeat_interleave(GROUP, dim=1)
        v_g = v_flat.repeat_interleave(GROUP, dim=1)
        scores = torch.einsum("hd,thd->ht", q[b], k_g) * SCALE
        if window is not None:
            pos = torch.arange(seq_len, dtype=torch.float64)
            scores[:, pos < (seq_len - window)] = float("-inf")
        attn = torch.softmax(scores, dim=-1)
        outs.append(torch.einsum("ht,thd->hd", attn, v_g))
    return torch.stack(outs, dim=0)


def max_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.float().cpu() - b.float().cpu()).abs().max().item())
