# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Standalone MiniMax M3 prefill index block-score kernel + wrapper.

Extracted from vllm/models/minimax_m3/common/ops/index_topk.py — only
``minimax_m3_index_score`` and its direct dependencies.
"""

import torch
from vllm.triton_utils import tl, triton

SPARSE_BLOCK_SIZE = 128


def _round_up(x: int, y: int) -> int:
    return ((x + y - 1) // y) * y


def _as_triton_index_kv_cache(
    index_kv_cache: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    """Normalize an index KV cache to ``[num_blocks, 128, head_dim]``.

    Accepts an already-3-D tensor, a ``(k, v)`` pair/list (K is used), or a
    4-D/5-D tensor with a singleton/leading-2 head axis that Ascend's layout
    produces; squeezes it down to the 3-D shape the kernel expects.
    """
    if isinstance(index_kv_cache, (tuple, list)):
        index_kv_cache = index_kv_cache[0]
    if index_kv_cache.ndim == 5 and index_kv_cache.shape[0] == 2:
        index_kv_cache = index_kv_cache[0]
    if index_kv_cache.ndim == 4:
        if index_kv_cache.shape[2] != 1:
            raise ValueError(f"Unexpected index cache head dim: {tuple(index_kv_cache.shape)}")
        index_kv_cache = index_kv_cache.squeeze(2)
    if index_kv_cache.ndim != 3:
        raise ValueError(f"Unexpected index cache ndim: {index_kv_cache.ndim}")
    return index_kv_cache


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE_Q": 4}, num_warps=1),
        triton.Config({"BLOCK_SIZE_Q": 8}, num_warps=1),
        triton.Config({"BLOCK_SIZE_Q": 16}, num_warps=1),
        triton.Config({"BLOCK_SIZE_Q": 32}, num_warps=1),
        triton.Config({"BLOCK_SIZE_Q": 64}, num_warps=1),
        triton.Config({"BLOCK_SIZE_Q": 128}, num_warps=1),
    ],
    key=["head_dim", "num_q_blocks"],
)
@triton.jit(do_not_specialize_on_alignment=["seq_lens", "prefix_lens"])
def _index_block_score_kernel(
    q_ptr,
    ik_cache_ptr,
    score_ptr,
    block_table_ptr,
    cu_seqlens,
    seq_lens,
    prefix_lens,
    num_idx_heads,
    head_dim: tl.constexpr,
    sm_scale_log2e,
    stride_q_n,
    stride_q_h,
    stride_q_d,
    stride_ik_blk,
    stride_ik_pos,
    stride_ik_d,
    stride_s_h,
    stride_s_n,
    stride_s_k,
    stride_bt_b,
    num_q_blocks: tl.constexpr,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_q = tl.program_id(0)
    pid_bh = tl.program_id(1)
    pid_b = pid_bh // num_idx_heads
    pid_h = pid_bh % num_idx_heads

    seq_start = tl.load(cu_seqlens + pid_b)
    q_len = tl.load(cu_seqlens + pid_b + 1) - seq_start
    seq_len = tl.load(seq_lens + pid_b)
    prefix_len = tl.load(prefix_lens + pid_b)
    if BLOCK_SIZE_Q * pid_q >= q_len:
        return

    q_ptrs = tl.make_block_ptr(
        base=q_ptr + seq_start * stride_q_n + pid_h * stride_q_h,
        shape=(q_len, head_dim),
        strides=(stride_q_n, stride_q_d),
        offsets=(pid_q * BLOCK_SIZE_Q, 0),
        block_shape=(BLOCK_SIZE_Q, head_dim),
        order=(1, 0),
    )
    q = tl.load(q_ptrs, boundary_check=(0,), padding_option="zero")

    off_q = tl.arange(0, BLOCK_SIZE_Q) + pid_q * BLOCK_SIZE_Q + prefix_len
    off_k = tl.arange(0, BLOCK_SIZE_K)
    bt_row = block_table_ptr + pid_b * stride_bt_b
    hi = min(seq_len, prefix_len + (pid_q + 1) * BLOCK_SIZE_Q)
    q_store_mask = (pid_q * BLOCK_SIZE_Q + tl.arange(0, BLOCK_SIZE_Q)) < q_len
    s_ptrs_base = (
        score_ptr + pid_h * stride_s_h + (seq_start + pid_q * BLOCK_SIZE_Q + tl.arange(0, BLOCK_SIZE_Q)) * stride_s_n
    )
    # Process four sparse blocks per outer iteration to minimize loop overhead
    # and expose multiple independent dot-product streams for scheduling.
    # Hoist all 4 page loads to the top of each iteration to expose scalar-load
    # latency to the compiler: adjacent addresses (bt_row+blk+0..3) let the MTE
    # coalesce them, and the page values are available earlier for K-load setup.
    for i in tl.range(0, hi, BLOCK_SIZE_K * 4):
        blk = i // BLOCK_SIZE_K

        # Pre-load all 4 page indices for this iteration
        page0 = tl.load(bt_row + blk).to(tl.int64)
        page1 = tl.load(bt_row + blk + 1).to(tl.int64)
        page2 = tl.load(bt_row + blk + 2).to(tl.int64)
        page3 = tl.load(bt_row + blk + 3).to(tl.int64)

        # --- Block 0 ---
        pos = i + off_k
        k = tl.load(
            tl.make_block_ptr(
                base=ik_cache_ptr + page0 * stride_ik_blk,
                shape=(head_dim, BLOCK_SIZE_K),
                strides=(stride_ik_d, stride_ik_pos),
                offsets=(0, 0),
                block_shape=(head_dim, BLOCK_SIZE_K),
                order=(0, 1),
            )
        )
        qk = tl.dot(q, k)
        qk = tl.where(off_q[:, None] >= pos[None, :], qk * sm_scale_log2e, float("-inf"))
        tl.store(s_ptrs_base + blk * stride_s_k, tl.max(qk, axis=1), mask=q_store_mask)

        # --- Block 1 ---
        i1 = i + BLOCK_SIZE_K
        if i1 < hi:
            pos = i1 + off_k
            blk1 = blk + 1
            k = tl.load(
                tl.make_block_ptr(
                    base=ik_cache_ptr + page1 * stride_ik_blk,
                    shape=(head_dim, BLOCK_SIZE_K),
                    strides=(stride_ik_d, stride_ik_pos),
                    offsets=(0, 0),
                    block_shape=(head_dim, BLOCK_SIZE_K),
                    order=(0, 1),
                )
            )
            qk = tl.dot(q, k)
            qk = tl.where(off_q[:, None] >= pos[None, :], qk * sm_scale_log2e, float("-inf"))
            tl.store(s_ptrs_base + blk1 * stride_s_k, tl.max(qk, axis=1), mask=q_store_mask)

        # --- Block 2 ---
        i2 = i + 2 * BLOCK_SIZE_K
        if i2 < hi:
            pos = i2 + off_k
            blk2 = blk + 2
            k = tl.load(
                tl.make_block_ptr(
                    base=ik_cache_ptr + page2 * stride_ik_blk,
                    shape=(head_dim, BLOCK_SIZE_K),
                    strides=(stride_ik_d, stride_ik_pos),
                    offsets=(0, 0),
                    block_shape=(head_dim, BLOCK_SIZE_K),
                    order=(0, 1),
                )
            )
            qk = tl.dot(q, k)
            qk = tl.where(off_q[:, None] >= pos[None, :], qk * sm_scale_log2e, float("-inf"))
            tl.store(s_ptrs_base + blk2 * stride_s_k, tl.max(qk, axis=1), mask=q_store_mask)

        # --- Block 3 ---
        i3 = i + 3 * BLOCK_SIZE_K
        if i3 < hi:
            pos = i3 + off_k
            blk3 = blk + 3
            k = tl.load(
                tl.make_block_ptr(
                    base=ik_cache_ptr + page3 * stride_ik_blk,
                    shape=(head_dim, BLOCK_SIZE_K),
                    strides=(stride_ik_d, stride_ik_pos),
                    offsets=(0, 0),
                    block_shape=(head_dim, BLOCK_SIZE_K),
                    order=(0, 1),
                )
            )
            qk = tl.dot(q, k)
            qk = tl.where(off_q[:, None] >= pos[None, :], qk * sm_scale_log2e, float("-inf"))
            tl.store(s_ptrs_base + blk3 * stride_s_k, tl.max(qk, axis=1), mask=q_store_mask)


@torch.no_grad()
def minimax_m3_index_score(
    idx_q: torch.Tensor,
    index_kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    seq_lens: torch.Tensor,
    prefix_lens: torch.Tensor,
    max_query_len: int,
    max_seq_len: int,
    num_kv_heads: int,
    sm_scale: float,
) -> torch.Tensor:
    """Compute per-token index scores for each visible sparse block.

    Returns score [num_kv_heads, total_q, max_block], where each score is the
    max over a 128-token index-K block.
    """
    index_kv_cache = _as_triton_index_kv_cache(index_kv_cache)
    total_q, num_idx_heads, head_dim = idx_q.shape
    assert num_idx_heads == num_kv_heads, "M3 expects num_idx_heads == num_kv_heads (no topk index reduce)"
    batch = cu_seqlens_q.shape[0] - 1
    max_block = triton.cdiv(max_seq_len, SPARSE_BLOCK_SIZE)

    score_block_stride = _round_up(max_block, 16)
    score = torch.empty(
        (num_idx_heads, total_q, score_block_stride),
        dtype=torch.float32,
        device=idx_q.device,
    )
    BLOCK_SIZE_Q = 64  # default, overridden by autotune
    num_q_blocks = triton.cdiv(max_query_len, BLOCK_SIZE_Q)
    sm_scale_log2e = sm_scale * 1.4426950409
    grid_score = lambda META: (triton.cdiv(max_query_len, META["BLOCK_SIZE_Q"]), batch * num_idx_heads)
    _index_block_score_kernel[grid_score](
        idx_q,
        index_kv_cache,
        score,
        block_table,
        cu_seqlens_q,
        seq_lens,
        prefix_lens,
        num_idx_heads,
        head_dim,
        sm_scale_log2e,
        idx_q.stride(0),
        idx_q.stride(1),
        idx_q.stride(2),
        index_kv_cache.stride(0),
        index_kv_cache.stride(1),
        index_kv_cache.stride(2),
        score.stride(0),
        score.stride(1),
        score.stride(2),
        block_table.stride(0),
        num_q_blocks=num_q_blocks,
        BLOCK_SIZE_K=SPARSE_BLOCK_SIZE,
        multibuffer=True,
    )
    return score
