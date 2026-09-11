# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton kernels for MiniMax M3 block-sparse GQA attention on Ascend.

Migrated from reference/vllm_cp/vllm/models/minimax_m3/common/ops/sparse_attn.py.
The Python wrappers adapt vLLM Ascend's KV cache layout to the paged layout
expected by the migrated kernels.
"""

# TODO: Unify the A3 and A5 sparse-attention kernels once their operator
# interfaces and implementations are compatible. They must remain separate
# for now.

from __future__ import annotations

import os
from collections.abc import Callable
from functools import lru_cache
from typing import Any

import torch
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils.math_utils import round_up

get_aicore_num: Callable[[], Any] | None
init_device_properties_triton: Callable[[], Any] | None

try:
    from vllm_ascend.ops.triton.triton_utils import (
        get_aicore_num,
        init_device_properties_triton,
    )
except ImportError:
    get_aicore_num = None
    init_device_properties_triton = None

# One sparse block == one KV page.
SPARSE_BLOCK_SIZE = 128


def _as_triton_main_kv_cache(
    kv_cache: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    if isinstance(kv_cache, (tuple, list)):
        kv_cache = torch.stack((kv_cache[0], kv_cache[1]), dim=1)
    if kv_cache.ndim != 5:
        raise ValueError(f"Unexpected main kv cache ndim: {kv_cache.ndim}")
    if kv_cache.shape[0] == 2:
        return kv_cache.permute(1, 0, 2, 3, 4)
    if kv_cache.shape[1] == 2:
        return kv_cache
    raise ValueError(f"Unexpected main kv cache shape: {tuple(kv_cache.shape)}")


def _as_triton_index_kv_cache(
    index_kv_cache: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    """Normalize Ascend indexer cache to ``[num_blocks, 128, head_dim]``."""
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


def _is_arch_support_pdl() -> bool:
    if current_platform.device_name == "npu":
        return False
    is_supported = getattr(current_platform, "is_arch_support_pdl", None)
    return bool(is_supported()) if callable(is_supported) else False


_SPARSE_ATTN_NUM_STAGES_KWARG: dict | None = None


def _sparse_attn_num_stages_kwarg() -> dict:
    """Triton ``num_stages`` override for the sparse-attn GEMM kernels."""
    global _SPARSE_ATTN_NUM_STAGES_KWARG
    if _SPARSE_ATTN_NUM_STAGES_KWARG is None:
        kwarg: dict = {}
        if current_platform.is_rocm():
            from vllm.platforms.rocm import on_gfx942

            if on_gfx942():
                kwarg = {"num_stages": 1}
        _SPARSE_ATTN_NUM_STAGES_KWARG = kwarg
    return _SPARSE_ATTN_NUM_STAGES_KWARG


# Set VLLM_MINIMAX_M3_DECODE_INDEX_SCORE_KQ=1 to use dot(k, q) with [N,D] K load.
_USE_DECODE_INDEX_SCORE_KQ = os.environ.get("VLLM_MINIMAX_M3_DECODE_INDEX_SCORE_KQ", "0") == "1"

_DECODE_INDEX_SCORE_MAX_GRID = 512

# Per-query decode-score launch policy. This kernel uses tl.dot and runs on
# AI_CORE, so size the grid from the detected AICore count rather than AIV count.
# Keep 32 only as a conservative fallback/minimum target.
_DECODE_QK_SCORE_MIN_PROGRAMS = 32
_DECODE_QK_SCORE_MAX_CHUNKS = 256


@lru_cache(maxsize=1)
def _decode_qk_score_target_programs() -> int:
    if get_aicore_num is None or init_device_properties_triton is None:
        return _DECODE_QK_SCORE_MIN_PROGRAMS

    try:
        aicore_count = int(get_aicore_num())
    except AssertionError:
        init_device_properties_triton()
        aicore_count = int(get_aicore_num())
    except Exception:
        return _DECODE_QK_SCORE_MIN_PROGRAMS

    return max(_DECODE_QK_SCORE_MIN_PROGRAMS, aicore_count)


_DECODE_INDEX_SCORE_AUTOTUNE_CONFIGS = [
    triton.Config({"num_kv_chunks": num_kv_chunks}, num_stages=num_stages)
    for num_kv_chunks in [1, 2, 4, 8, 16, 32, 64, 128, 256]
    for num_stages in [1, 2]
]


def _prune_decode_index_score_configs(configs, nargs, **kwargs):
    num_reqs = max(1, nargs["num_reqs"])
    max_chunks = max(1, _DECODE_INDEX_SCORE_MAX_GRID // num_reqs)
    max_chunks = 1 << (max_chunks.bit_length() - 1)
    pruned = [c for c in configs if c.kwargs["num_kv_chunks"] <= max_chunks]
    return pruned or configs[:1]


@triton.jit(do_not_specialize_on_alignment=["seq_lens", "prefix_lens"])
def _index_block_score_kernel(
    q_ptr,
    ik_cache_ptr,
    score_ptr,
    block_table_ptr,
    cu_seqlens,
    seq_lens,
    prefix_lens,
    num_idx_heads: tl.constexpr,
    head_dim: tl.constexpr,  # 128
    sm_scale,
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
    BLOCK_SIZE_Q: tl.constexpr,  # 64
    BLOCK_SIZE_K: tl.constexpr,  # 128
    # CUBE_N: tl.constexpr,
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
    q_start = prefix_len + pid_q * BLOCK_SIZE_Q

    off_q = tl.arange(0, BLOCK_SIZE_Q) + pid_q * BLOCK_SIZE_Q + prefix_len
    off_k = tl.arange(0, BLOCK_SIZE_K)
    off_d = tl.arange(0, head_dim)
    bt_row = block_table_ptr + pid_b * stride_bt_b
    hi = tl.minimum(seq_len, prefix_len + (pid_q + 1) * BLOCK_SIZE_Q)

    middle = tl.maximum(
        0,
        (q_start - BLOCK_SIZE_K) // BLOCK_SIZE_K * BLOCK_SIZE_K,
    )

    for key_start in tl.range(0, middle, BLOCK_SIZE_K):
        block_id = key_start // BLOCK_SIZE_K
        page_id = tl.load(bt_row + block_id).to(tl.int64)
        key_positions = key_start + off_k
        key = tl.load(
            ik_cache_ptr + page_id * stride_ik_blk + off_k[None, :] * stride_ik_pos + off_d[:, None] * stride_ik_d,
        )
        query_key = tl.dot(q, key)
        block_score = tl.max(query_key, axis=1)
        query_offsets = tl.arange(0, BLOCK_SIZE_Q)
        score_ptrs = (
            score_ptr
            + pid_h * stride_s_h
            + (seq_start + pid_q * BLOCK_SIZE_Q + query_offsets) * stride_s_n
            + block_id * stride_s_k
        )
        query_mask = pid_q * BLOCK_SIZE_Q + query_offsets < q_len
        tl.store(score_ptrs, block_score, mask=query_mask)

    for key_start in tl.range(middle, hi, BLOCK_SIZE_K):
        block_id = key_start // BLOCK_SIZE_K
        page_id = tl.load(bt_row + block_id).to(tl.int64)
        key_positions = key_start + off_k
        key = tl.load(
            ik_cache_ptr + page_id * stride_ik_blk + off_k[None, :] * stride_ik_pos + off_d[:, None] * stride_ik_d,
        )
        query_key = tl.dot(q, key)
        query_key = tl.where(
            off_q[:, None] >= key_positions[None, :],
            query_key,
            float("-inf"),
        )
        block_score = tl.max(query_key, axis=1)
        query_offsets = tl.arange(0, BLOCK_SIZE_Q)
        score_ptrs = (
            score_ptr
            + pid_h * stride_s_h
            + (seq_start + pid_q * BLOCK_SIZE_Q + query_offsets) * stride_s_n
            + block_id * stride_s_k
        )
        query_mask = pid_q * BLOCK_SIZE_Q + query_offsets < q_len
        tl.store(score_ptrs, block_score, mask=query_mask)


# ---------------------------------------------------------------------------
# Decode index-score kernel (split-K over seq blocks). Decode batches are
# flattened request-major, with a runtime query length used to map each query
# token back to its request metadata. Chunk counts depend only on shape
# constants so the grid is fixed within a cuda graph. The score scale is omitted
# because decode only consumes block ordering.
# ---------------------------------------------------------------------------
@triton.jit(do_not_specialize=["decode_query_len"])
def _decode_qk_score_kernel(
    q_ptr,
    ik_cache_ptr,
    score_ptr,
    block_table_ptr,
    seq_lens_ptr,
    num_idx_heads: tl.constexpr,
    head_dim: tl.constexpr,
    decode_query_len,
    init_blocks: tl.constexpr,
    local_blocks: tl.constexpr,
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
    BLOCK_SIZE_K: tl.constexpr,
    NUM_CHUNKS: tl.constexpr,
):
    """Compute decode QK block scores with one program per query/chunk.

    Visible-range metadata is scalar because each program owns exactly one
    flattened query. Fully visible pages skip the token-position mask; at most
    one causal boundary page uses it. The score buffer is pre-filled with
    ``-inf``, so unwritten tail entries are already masked.
    """
    query_id = tl.program_id(0)
    chunk_id = tl.program_id(1)

    request_id = query_id // decode_query_len
    query_offset = query_id - request_id * decode_query_len

    seq_len = tl.load(seq_lens_ptr + request_id)
    kv_length = tl.maximum(
        seq_len - decode_query_len + query_offset + 1,
        0,
    )
    valid_blocks = (kv_length + BLOCK_SIZE_K - 1) // BLOCK_SIZE_K
    full_block_count = kv_length // BLOCK_SIZE_K
    local_start = tl.maximum(valid_blocks - local_blocks, 0)

    # Split only the visible range. Empty chunks return before loading Q.
    chunk_size_blocks = (valid_blocks + NUM_CHUNKS - 1) // NUM_CHUNKS
    chunk_start = chunk_id * chunk_size_blocks
    chunk_end = tl.minimum(chunk_start + chunk_size_blocks, valid_blocks)
    if chunk_start >= chunk_end:
        return

    head_offsets = tl.arange(0, num_idx_heads)
    dim_offsets = tl.arange(0, head_dim)
    key_offsets = tl.arange(0, BLOCK_SIZE_K)

    query = tl.load(
        q_ptr + query_id * stride_q_n + head_offsets[:, None] * stride_q_h + dim_offsets[None, :] * stride_q_d,
    )
    block_table_row = block_table_ptr + request_id * stride_bt_b

    # Fully visible blocks require no token-position generation or causal mask.
    full_end = tl.minimum(chunk_end, full_block_count)
    for block_id in tl.range(chunk_start, full_end):
        page_id = tl.load(block_table_row + block_id).to(tl.int64)
        key = tl.load(
            ik_cache_ptr
            + page_id * stride_ik_blk
            + key_offsets[None, :] * stride_ik_pos
            + dim_offsets[:, None] * stride_ik_d,
        )
        block_score = tl.max(
            tl.dot(query, key, out_dtype=tl.float32),
            axis=1,
        )

        # Preserve source priority: local is applied after init and overrides it
        # when the two forced regions overlap.
        block_score = tl.where(block_id < init_blocks, 1e30, block_score)
        block_score = tl.where(block_id >= local_start, 1e29, block_score)

        tl.store(
            score_ptr + head_offsets * stride_s_h + query_id * stride_s_n + block_id * stride_s_k,
            block_score,
        )

    # A non-block-aligned kv_length has exactly one causal boundary block.
    boundary_block = full_block_count
    boundary_in_chunk = (
        (full_block_count < valid_blocks) & (boundary_block >= chunk_start) & (boundary_block < chunk_end)
    )
    if boundary_in_chunk:
        page_id = tl.load(block_table_row + boundary_block).to(tl.int64)
        key = tl.load(
            ik_cache_ptr
            + page_id * stride_ik_blk
            + key_offsets[None, :] * stride_ik_pos
            + dim_offsets[:, None] * stride_ik_d,
        )
        query_key = tl.dot(query, key, out_dtype=tl.float32)
        key_positions = boundary_block * BLOCK_SIZE_K + key_offsets
        query_key = tl.where(
            key_positions[None, :] < kv_length,
            query_key,
            float("-inf"),
        )
        block_score = tl.max(query_key, axis=1)
        block_score = tl.where(boundary_block < init_blocks, 1e30, block_score)
        block_score = tl.where(boundary_block >= local_start, 1e29, block_score)

        tl.store(
            score_ptr + head_offsets * stride_s_h + query_id * stride_s_n + boundary_block * stride_s_k,
            block_score,
        )


# ---------------------------------------------------------------------------
# Decode top-k: torch.topk on scores, then mask invalid block ids per token.
# Forced init/local blocks are already encoded in the scores.
# ---------------------------------------------------------------------------
@triton.heuristics({"BLOCK_SIZE_T": lambda args: triton.next_power_of_2(args["topk"])})
@triton.jit(do_not_specialize=["decode_query_len"])
def _index_topk_postprocess_kernel(
    ti_in_ptr,  # [num_idx_heads, total_q, topk] int64 input
    ti_out_ptr,  # [num_idx_heads, total_q, topk] int32 output
    select_num_idx_ptr,  # [num_idx_heads, total_q] int32 output
    seq_lens,  # [num_reqs]
    block_size: tl.constexpr,  # sparse block size (128)
    topk: tl.constexpr,
    decode_query_len,
    stride_in_h,
    stride_in_b,
    stride_in_t,
    stride_out_h,
    stride_out_b,
    stride_out_t,
    stride_select_h,
    stride_select_b,
    BLOCK_SIZE_T: tl.constexpr,
):
    pid_b = tl.program_id(0)  # flattened query-token id
    pid_h = tl.program_id(1)
    req_id = pid_b // decode_query_len
    q_offset = pid_b - req_id * decode_query_len

    seq_len = tl.load(seq_lens + req_id)
    query_pos = seq_len - decode_query_len + q_offset
    # Full-CG padding uses zero-length request rows. Clamp to an empty
    # attention range instead of letting padded rows produce negative lengths.
    kv_len = tl.maximum(query_pos + 1, 0)
    num_blocks = (kv_len + block_size - 1) // block_size

    off_t = tl.arange(0, BLOCK_SIZE_T)
    ti_in_ptrs = ti_in_ptr + pid_h * stride_in_h + pid_b * stride_in_b + off_t * stride_in_t
    store_mask = off_t < topk
    idx = tl.load(ti_in_ptrs, mask=store_mask, other=0)
    valid_slot = off_t < tl.minimum(topk, num_blocks)
    valid_idx = (idx >= 0) & (idx < num_blocks)
    masked_idx = tl.where(valid_slot & valid_idx, idx, -1)
    ti_out_ptrs = ti_out_ptr + pid_h * stride_out_h + pid_b * stride_out_b + off_t * stride_out_t
    tl.store(ti_out_ptrs, masked_idx.to(tl.int32), mask=store_mask)
    valid_count = tl.sum((valid_slot & valid_idx).to(tl.int32))
    tl.store(select_num_idx_ptr + pid_h * stride_select_h + pid_b * stride_select_b, valid_count)


# ---------------------------------------------------------------------------
# Prefill top-k: prepare scores (pad tail, force init/local), then torch.topk.
# ---------------------------------------------------------------------------
@triton.jit(do_not_specialize=["max_block", "chunk_blocks", "num_prep_chunks"])
def _prefill_index_score_prepare_for_topk_kernel(
    score_ptr,  # [num_idx_heads, total_q, score_block_stride] fp32 in/out
    cu_seqlens,  # [batch+1]
    prefix_lens,  # [batch]
    init_blocks: tl.constexpr,
    local_blocks: tl.constexpr,
    max_block,
    chunk_blocks,
    num_prep_chunks,
    stride_s_h,
    stride_s_n,
    stride_s_k,
    block_size: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_q = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_h = tl.program_id(2)
    seq_start = tl.load(cu_seqlens + pid_b)
    q_len = tl.load(cu_seqlens + pid_b + 1) - seq_start
    if pid_q >= q_len:
        return
    token_idx = seq_start + pid_q
    prefix_len = tl.load(prefix_lens + pid_b)
    valid_blocks = (prefix_len + pid_q + block_size) // block_size
    local_start = tl.maximum(0, valid_blocks - local_blocks)

    for pid_chunk in tl.range(0, num_prep_chunks):
        chunk_start = pid_chunk * chunk_blocks
        chunk_end = tl.minimum(chunk_start + chunk_blocks, max_block)
        if chunk_start < chunk_end:
            num_blks = chunk_end - chunk_start
            off_k = tl.arange(0, BLOCK_SIZE_K)
            for i in tl.range(0, num_blks, BLOCK_SIZE_K):
                blk = chunk_start + i + off_k
                mask = (i + off_k) < num_blks
                s_ptrs = score_ptr + pid_h * stride_s_h + token_idx * stride_s_n + blk * stride_s_k
                score = tl.load(s_ptrs, mask=mask, other=float("-inf"))
                blk_valid = blk < valid_blocks
                score = tl.where(blk_valid, score, float("-inf"))
                is_init = (blk < init_blocks) & blk_valid
                is_local = (blk >= local_start) & blk_valid
                score = tl.where(is_local, 1e29, tl.where(is_init, 1e30, score))
                tl.store(s_ptrs, score, mask=mask)


@triton.heuristics({"BLOCK_SIZE_T": lambda args: triton.next_power_of_2(args["topk"])})
@triton.jit(do_not_specialize_on_alignment=["prefix_lens"])
def _topk_index_mask_invalid_prefill_kernel(
    ti_ptr,  # [num_idx_heads, total_q, topk] int32 in/out
    cu_seqlens,  # [batch+1]
    prefix_lens,  # [batch]
    block_size: tl.constexpr,  # sparse block size (128)
    topk: tl.constexpr,
    stride_ti_h,
    stride_ti_n,
    stride_ti_t,
    BLOCK_SIZE_T: tl.constexpr,
):
    pid_q = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_h = tl.program_id(2)
    seq_start = tl.load(cu_seqlens + pid_b)
    q_len = tl.load(cu_seqlens + pid_b + 1) - seq_start
    if pid_q >= q_len:
        return
    token_idx = seq_start + pid_q
    prefix_len = tl.load(prefix_lens + pid_b)
    valid_blocks = (prefix_len + pid_q + block_size) // block_size

    off_t = tl.arange(0, BLOCK_SIZE_T)
    ti_ptrs = ti_ptr + pid_h * stride_ti_h + token_idx * stride_ti_n + off_t * stride_ti_t
    store_mask = off_t < topk
    idx = tl.load(ti_ptrs, mask=store_mask, other=0)
    valid_slot = off_t < tl.minimum(topk, valid_blocks)
    valid_idx = (idx >= 0) & (idx < valid_blocks)
    masked_idx = tl.where(valid_slot & valid_idx, idx, -1)
    tl.store(ti_ptrs, masked_idx.to(ti_ptr.dtype.element_ty), mask=store_mask)


# ---------------------------------------------------------------------------
# Decode init/local bool masks for index scoring. fp32 intermediates; split-K
# over max_block with shape-constant chunk count (cudagraph-safe).
# ---------------------------------------------------------------------------
@triton.jit(do_not_specialize=["decode_query_len", "max_block", "chunk_blocks"])
def _decode_index_score_masks_kernel(
    init_mask_ptr,  # [total_q, score_block_stride] bool out
    local_mask_ptr,  # [total_q, score_block_stride] bool out
    seq_lens,  # [num_reqs] int32
    block_size: tl.constexpr,  # sparse block size (128)
    max_block,
    decode_query_len,
    chunk_blocks,
    init_blocks: tl.constexpr,
    local_blocks: tl.constexpr,
    stride_mask_q,
    stride_mask_k,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_q = tl.program_id(0)
    pid_chunk = tl.program_id(1)
    req_id = pid_q // decode_query_len
    q_offset = pid_q - req_id * decode_query_len

    seq_len = tl.load(seq_lens + req_id).to(tl.float32)
    query_pos = seq_len - decode_query_len + q_offset
    kv_len = tl.maximum(query_pos + 1.0, 0.0)
    valid_blocks = tl.floor((query_pos + block_size * 1.0) / (block_size * 1.0))
    local_start = tl.maximum(
        tl.floor((kv_len + (block_size - 1) * 1.0) / (block_size * 1.0)) - local_blocks * 1.0,
        0.0,
    )

    chunk_start = pid_chunk * chunk_blocks
    chunk_end = tl.minimum(chunk_start + chunk_blocks, max_block)
    if chunk_start >= chunk_end:
        return

    num_blks = chunk_end - chunk_start
    off_k = tl.arange(0, BLOCK_SIZE_K)
    for i in tl.range(0, num_blks, BLOCK_SIZE_K):
        blk = chunk_start + i + off_k
        store_mask = (i + off_k) < num_blks
        blk_f = blk * 1.0
        blk_valid = blk_f < valid_blocks
        is_init = (blk_f < init_blocks * 1.0) & blk_valid
        is_local = (blk_f >= local_start) & blk_valid
        mask_ptrs = init_mask_ptr + pid_q * stride_mask_q + blk * stride_mask_k
        tl.store(mask_ptrs, is_init, mask=store_mask)
        tl.store(
            local_mask_ptr + pid_q * stride_mask_q + blk * stride_mask_k,
            is_local,
            mask=store_mask,
        )


# ---------------------------------------------------------------------------
# Python wrappers
# ---------------------------------------------------------------------------
@torch.no_grad()
def _build_decode_index_score_masks(
    seq_lens: torch.Tensor,
    decode_query_len: int,
    init_blocks: int,
    local_blocks: int,
    max_block: int,
    score_block_stride: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build bool init/local masks for decode index scoring via Triton."""
    total_q = seq_lens.shape[0] * decode_query_len
    init_mask = torch.zeros(
        (total_q, score_block_stride),
        dtype=torch.bool,
        device=seq_lens.device,
    )
    local_mask = torch.zeros(
        (total_q, score_block_stride),
        dtype=torch.bool,
        device=seq_lens.device,
    )
    MASK_TARGET_GRID = 64
    MAX_MASK_CHUNKS = 16
    mask_target = max(1, min(MAX_MASK_CHUNKS, MASK_TARGET_GRID // max(1, total_q)))
    chunk_blocks = (max_block + mask_target - 1) // mask_target
    _decode_index_score_masks_kernel[(total_q, mask_target)](
        init_mask,
        local_mask,
        seq_lens,
        SPARSE_BLOCK_SIZE,
        max_block,
        decode_query_len,
        chunk_blocks,
        init_blocks,
        local_blocks,
        init_mask.stride(0),
        init_mask.stride(1),
        BLOCK_SIZE_K=2048,
    )
    return init_mask, local_mask


@torch.no_grad()
def minimax_m3_index_score(
    idx_q: torch.Tensor,  # [total_q, num_idx_heads, head_dim]
    index_kv_cache: torch.Tensor,  # [num_blocks, 128, head_dim]
    block_table: torch.Tensor,  # [batch, max_blocks]
    cu_seqlens_q: torch.Tensor,  # [batch+1] int32
    seq_lens: torch.Tensor,  # [batch] int32
    prefix_lens: torch.Tensor,  # [batch] int32
    max_query_len: int,
    max_seq_len: int,
    num_kv_heads: int,
    sm_scale=None,
) -> torch.Tensor:
    """Compute per-token index scores for each visible sparse block.

    Returns score [num_kv_heads, total_q, max_block], where each score is the
    max over a 128-token index-K block. M3 has num_idx_heads == num_kv_heads.
    """
    index_kv_cache = _as_triton_index_kv_cache(index_kv_cache)
    total_q, num_idx_heads, head_dim = idx_q.shape
    assert num_idx_heads == num_kv_heads, "M3 expects num_idx_heads == num_kv_heads (no topk index reduce)"
    batch = cu_seqlens_q.shape[0] - 1
    max_block = triton.cdiv(max_seq_len, SPARSE_BLOCK_SIZE)

    # Keep score strides 16-divisible to avoid Triton recompiles.
    score_block_stride = round_up(max_block, 16)
    score = torch.empty(
        (num_idx_heads, total_q, score_block_stride),
        dtype=torch.float32,
        device=idx_q.device,
    )
    BLOCK_SIZE_Q = 128
    grid_score = (triton.cdiv(max_query_len, BLOCK_SIZE_Q), batch * num_idx_heads)
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
        sm_scale,
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
        BLOCK_SIZE_Q=BLOCK_SIZE_Q,
        BLOCK_SIZE_K=SPARSE_BLOCK_SIZE,
    )
    return score


@torch.no_grad()
def minimax_m3_index_topk(
    score: torch.Tensor,  # [num_idx_heads, total_q, max_block]
    cu_seqlens_q: torch.Tensor,  # [batch+1] int32
    prefix_lens: torch.Tensor,  # [batch] int32
    max_query_len: int,
    topk: int,
    init_blocks: int,
    local_blocks: int,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Select index top-k from a precomputed score tensor (torch.topk).

    When ``out`` is provided (a ``[num_idx_heads, >=total_q, topk]`` buffer), the
    result is written into ``out[:, :total_q, :]`` instead of a fresh tensor --
    used to keep the top-k output at a stable address for cudagraph capture.
    """
    num_idx_heads = score.shape[0]
    batch = cu_seqlens_q.shape[0] - 1
    total_q = score.shape[1]
    max_block = score.shape[2]
    prep_chunk_blocks = 2048
    num_prep_chunks = (max_block + prep_chunk_blocks - 1) // prep_chunk_blocks
    _prefill_index_score_prepare_for_topk_kernel[(max_query_len, batch, num_idx_heads)](
        score,
        cu_seqlens_q,
        prefix_lens,
        init_blocks,
        local_blocks,
        max_block,
        prep_chunk_blocks,
        num_prep_chunks,
        score.stride(0),
        score.stride(1),
        score.stride(2),
        block_size=SPARSE_BLOCK_SIZE,
        BLOCK_SIZE_K=2048,
    )
    score_rows = score[:, :total_q, :max_block]
    _, topk_idx_raw = torch.topk(score_rows, k=min(topk, max_block), dim=-1)
    if out is not None:
        topk_idx = out[:, :total_q, :]
        topk_idx.copy_(topk_idx_raw)
    else:
        if max_block < topk:
            topk_idx = torch.empty(
                num_idx_heads,
                total_q,
                topk,
                dtype=torch.int32,
                device=score.device,
            )
            topk_idx[..., :max_block].copy_(topk_idx_raw.to(torch.int32))
        else:
            topk_idx = topk_idx_raw.to(torch.int32)
    _topk_index_mask_invalid_prefill_kernel[(max_query_len, batch, num_idx_heads)](
        topk_idx,
        cu_seqlens_q,
        prefix_lens,
        SPARSE_BLOCK_SIZE,
        topk,
        topk_idx.stride(0),
        topk_idx.stride(1),
        topk_idx.stride(2),
    )
    return topk_idx


@torch.no_grad()
def minimax_m3_index_decode(
    idx_q: torch.Tensor,  # [total_q, num_idx_heads, head_dim]
    index_kv_cache: torch.Tensor,  # [num_blocks, 128, head_dim]
    block_table: torch.Tensor,  # [num_reqs, max_blocks]
    seq_lens: torch.Tensor,  # [num_reqs] int32
    max_seq_len: int,
    topk: int,
    init_blocks: int,
    local_blocks: int,
    num_kv_heads: int,
    decode_query_len: int,
    max_decode_query_len: int | None = None,
    out: torch.Tensor | None = None,
    sm_scale=None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode index block-score + top-k (torch.topk + invalid-index mask).

    Returns ``(topk_idx, select_num_idx)``. Invalid block IDs are ``-1`` and
    ``select_num_idx`` contains the number of valid IDs for each head/query.
    When ``out`` ([num_kv_heads, >=total_q, topk]) is given, writes into
    ``out[:, :total_q, :]`` (stable address for cudagraph) instead of allocating.
    """
    index_kv_cache = _as_triton_index_kv_cache(index_kv_cache)
    total_q, num_idx_heads, head_dim = idx_q.shape
    assert num_idx_heads == num_kv_heads, "M3 expects num_idx_heads == num_kv_heads (no topk index reduce)"
    if max_decode_query_len is None:
        max_decode_query_len = decode_query_len
    assert decode_query_len <= max_decode_query_len
    assert total_q == seq_lens.shape[0] * decode_query_len
    batch = total_q
    max_block = triton.cdiv(max_seq_len, SPARSE_BLOCK_SIZE)
    del sm_scale

    # Pre-fill the tail so torch.topk can run directly on one contiguous tensor
    # without a separate pad kernel. Keep the stride 16-divisible and at least
    # topk wide so the output shape is stable for short sequences.
    score_block_stride = round_up(max(max_block, topk), 16)
    score = torch.full(
        (num_idx_heads, total_q, score_block_stride),
        float("-inf"),
        dtype=torch.float32,
        device=idx_q.device,
    )

    # Shape/device-constant per-query split-K. The detected AICore count is
    # cached for the process, so the launch grid remains stable for cudagraph.
    target_programs = _decode_qk_score_target_programs()
    num_kv_chunks = max(
        1,
        min(
            _DECODE_QK_SCORE_MAX_CHUNKS,
            max(1, max_block),
            triton.cdiv(target_programs, max(1, total_q)),
        ),
    )
    _decode_qk_score_kernel[(total_q, num_kv_chunks)](
        idx_q,
        index_kv_cache,
        score,
        block_table,
        seq_lens,
        num_idx_heads,
        head_dim,
        decode_query_len,
        init_blocks,
        local_blocks,
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
        BLOCK_SIZE_K=SPARSE_BLOCK_SIZE,
        NUM_CHUNKS=num_kv_chunks,
    )
    _, topk_idx_raw = torch.topk(score, k=topk, dim=-1)
    if out is not None:
        topk_idx = out[:, :total_q, :]
    else:
        topk_idx = torch.empty(
            num_idx_heads,
            total_q,
            topk,
            dtype=torch.int32,
            device=idx_q.device,
        )
    select_num_idx = torch.empty(
        (num_idx_heads, total_q),
        dtype=torch.int32,
        device=idx_q.device,
    )
    _index_topk_postprocess_kernel[(batch, num_idx_heads)](
        topk_idx_raw,
        topk_idx,
        select_num_idx,
        seq_lens,
        SPARSE_BLOCK_SIZE,
        topk,
        decode_query_len,
        topk_idx_raw.stride(0),
        topk_idx_raw.stride(1),
        topk_idx_raw.stride(2),
        topk_idx.stride(0),
        topk_idx.stride(1),
        topk_idx.stride(2),
        select_num_idx.stride(0),
        select_num_idx.stride(1),
    )
    return topk_idx, select_num_idx


# ---------------------------------------------------------------------------
# GQA block-sparse attention (paged). Main heads attend only to the selected
# blocks. BLOCK_SIZE_K == 128 so each selected block is one page.
# ---------------------------------------------------------------------------
# since prefill metadata is sliced from mixed batch metadata, seq_lens and prefix_lens
# might lose pointer alignment, which trigger Triton recompiles. we don't actually
# need pointer alignment for those tensors anyway because we do scalar load.
@triton.heuristics(
    {
        "BLOCK_SIZE_D": lambda args: triton.next_power_of_2(args["head_dim"]),
        "BLOCK_SIZE_H": lambda args: triton.next_power_of_2(args["gqa_group_size"]),
        "BLOCK_SIZE_T": lambda args: triton.next_power_of_2(args["max_topk"]),
        "BLOCK_SIZE_QH": lambda args: args["BLOCK_SIZE_Q"] * triton.next_power_of_2(args["gqa_group_size"]),
    }
)
@triton.jit(do_not_specialize_on_alignment=["seq_lens", "prefix_lens"])
def _gqa_sparse_fwd_kernel(
    q_ptr,  # [total_q, num_heads, head_dim]
    kv_cache_ptr,  # main cache: [num_blocks, 2, 128, num_kv_heads, head_dim]
    t_ptr,  # topk_idx: [num_kv_heads, total_q, topk]
    o_ptr,  # [total_q, num_heads, head_dim]
    block_table_ptr,  # [num_reqs, max_blocks]
    cu_seqlens_q,
    cu_seqblocks_q,
    seq_lens,
    prefix_lens,
    num_kv_heads,
    gqa_group_size,
    head_dim,
    max_topk,
    num_q_loop,
    sm_scale,
    stride_qn,
    stride_qh,
    stride_qd,
    stride_kv_blk,
    stride_kv_kv,
    stride_kv_pos,
    stride_kv_h,
    stride_kv_d,
    stride_th,
    stride_tn,
    stride_tk,
    stride_on,
    stride_oh,
    stride_od,
    stride_bt_b,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,  # == SPARSE_BLOCK_SIZE (128)
    BLOCK_SIZE_D: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_T: tl.constexpr,
    BLOCK_SIZE_QH: tl.constexpr,
    USE_FP8: tl.constexpr,  # fp8 KV cache: dequantize K/V to q.dtype on load
):
    sm_scale_log2e = sm_scale * 1.4426950409
    pid_q = tl.program_id(0)
    pid_kh = tl.program_id(1)
    pid_b = tl.program_id(2)
    pid_h = pid_kh * gqa_group_size
    q_start = tl.load(cu_seqlens_q + pid_b)
    q_len = tl.load(cu_seqlens_q + pid_b + 1) - q_start
    q_block_start = tl.load(cu_seqblocks_q + pid_b)
    q_block_len = tl.load(cu_seqblocks_q + pid_b + 1) - q_block_start
    seq_len = tl.load(seq_lens + pid_b)
    prefix_len = tl.load(prefix_lens + pid_b)
    if pid_q * num_q_loop >= q_block_len:
        return
    real_q_loop = min(num_q_loop, q_block_len - pid_q * num_q_loop)
    bt_row = block_table_ptr + pid_b * stride_bt_b
    off_n = tl.arange(0, BLOCK_SIZE_K)
    off_d = tl.arange(0, BLOCK_SIZE_D)
    d_mask = off_d < head_dim
    for j in range(real_q_loop):
        pid_q_j = pid_q * num_q_loop + j
        t_ptr_j = t_ptr + (q_block_start + pid_q_j) * stride_tn + pid_kh * stride_th
        off_t = tl.arange(0, BLOCK_SIZE_T)
        topk_idx = tl.load(t_ptr_j + off_t * stride_tk, mask=off_t < max_topk, other=-1)
        real_topk = tl.sum((topk_idx >= 0).to(tl.int32), axis=0)
        q_ptrs = tl.make_block_ptr(
            base=q_ptr + q_start * stride_qn + pid_h * stride_qh,
            shape=(q_len, gqa_group_size, head_dim),
            strides=(stride_qn, stride_qh, stride_qd),
            offsets=(pid_q_j * BLOCK_SIZE_Q, 0, 0),
            block_shape=(BLOCK_SIZE_Q, BLOCK_SIZE_H, BLOCK_SIZE_D),
            order=(2, 1, 0),
        )
        q = tl.load(q_ptrs, boundary_check=(0, 1, 2), padding_option="zero")
        off_q = (
            tl.arange(0, BLOCK_SIZE_Q)[:, None]
            + pid_q_j * BLOCK_SIZE_Q
            + prefix_len
            - tl.arange(0, BLOCK_SIZE_K)[None, :]
        )
        m_i = tl.full((BLOCK_SIZE_QH,), float("-inf"), dtype=tl.float32)
        lse_i = tl.full((BLOCK_SIZE_QH,), float("-inf"), dtype=tl.float32)
        acc_o = tl.zeros((BLOCK_SIZE_QH, BLOCK_SIZE_D), dtype=tl.float32)
        q = tl.reshape(q, BLOCK_SIZE_QH, BLOCK_SIZE_D)
        for _ in range(real_topk):
            blk = tl.load(t_ptr_j).to(tl.int32)
            t_ptr_j = t_ptr_j + stride_tk
            c = blk * BLOCK_SIZE_K
            page = tl.load(bt_row + blk).to(tl.int64)
            pos = c + off_n
            pos_mask = pos < seq_len
            k = tl.load(
                kv_cache_ptr
                + page * stride_kv_blk
                + 0 * stride_kv_kv
                + off_n[None, :] * stride_kv_pos
                + pid_kh * stride_kv_h
                + off_d[:, None] * stride_kv_d,
                mask=d_mask[:, None] & pos_mask[None, :],
                other=0.0,
            )
            if USE_FP8:
                k = k.to(q.dtype)
            qk = tl.zeros((BLOCK_SIZE_Q, BLOCK_SIZE_H, BLOCK_SIZE_K), dtype=tl.float32)
            # causal: q_abs_pos - k_off >= block_start (c)
            qk += tl.where(off_q[:, None, :] >= c, 0, float("-inf"))
            qk = tl.reshape(qk, BLOCK_SIZE_QH, BLOCK_SIZE_K)
            qk += tl.dot(q, k) * sm_scale_log2e
            qk += tl.where(pos_mask[None, :], 0, float("-inf"))
            m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
            p = tl.exp2(qk - m_ij[:, None])
            l_ij = tl.sum(p, axis=1)
            acc_o = acc_o * tl.exp2(m_i - m_ij)[:, None]
            v = tl.load(
                kv_cache_ptr
                + page * stride_kv_blk
                + 1 * stride_kv_kv
                + off_n[:, None] * stride_kv_pos
                + pid_kh * stride_kv_h
                + off_d[None, :] * stride_kv_d,
                mask=pos_mask[:, None] & d_mask[None, :],
                other=0.0,
            )
            if USE_FP8:
                v = v.to(q.dtype)
            acc_o += tl.dot(p.to(v.dtype), v)
            m_i = m_ij
            lse_i = m_ij + tl.log2(tl.exp2(lse_i - m_ij) + l_ij)
        acc_o = acc_o * tl.exp2(m_i - lse_i)[:, None]
        acc_o = tl.reshape(acc_o, BLOCK_SIZE_Q, BLOCK_SIZE_H, BLOCK_SIZE_D)
        o_ptrs = tl.make_block_ptr(
            base=o_ptr + q_start * stride_on + pid_h * stride_oh,
            shape=(q_len, gqa_group_size, head_dim),
            strides=(stride_on, stride_oh, stride_od),
            offsets=(pid_q_j * BLOCK_SIZE_Q, 0, 0),
            block_shape=(BLOCK_SIZE_Q, BLOCK_SIZE_H, BLOCK_SIZE_D),
            order=(2, 1, 0),
        )
        tl.store(o_ptrs, acc_o.to(o_ptr.dtype.element_ty), boundary_check=(0, 1, 2))


# ---------------------------------------------------------------------------
# Decode kernels (split-K). Decode batches are flattened request-major, with a
# runtime query length used to map each query token back to its request metadata.
# This parallelizes over the selected top-k blocks, producing partials that the
# merge kernel combines (flash-decoding). All chunk counts depend only on shape
# constants so the grid is fixed within a cuda graph. Base-2 (exp2/log2)
# softmax matches the prefill kernel.
# ---------------------------------------------------------------------------
@triton.heuristics(
    {
        "BLOCK_SIZE_H": lambda args: max(16, triton.next_power_of_2(args["gqa_group_size"])),
        "BLOCK_SIZE_D": lambda args: triton.next_power_of_2(args["head_dim"]),
        "BLOCK_SIZE_T": lambda args: triton.next_power_of_2(args["max_topk"]),
    }
)
@triton.jit(do_not_specialize=["decode_query_len"])
def _gqa_sparse_decode_kernel(
    q_ptr,  # [total_q, num_heads, head_dim]
    kv_cache_ptr,  # main cache: [num_blocks, 2, 128, num_kv_heads, head_dim]
    t_ptr,  # topk_idx: [num_kv_heads, total_q, topk]
    o_ptr,  # partial out: [NUM_TOPK_CHUNKS, total_q, num_heads, head_dim]
    lse_ptr,  # partial lse (log2): [NUM_TOPK_CHUNKS, total_q, num_heads]
    block_table_ptr,  # [num_reqs, max_blocks]
    seq_lens,  # [num_reqs]
    total_q,
    gqa_group_size,
    head_dim,
    max_topk,
    sm_scale,
    decode_query_len,
    stride_qn,
    stride_qh,
    stride_qd,
    stride_kv_blk,
    stride_kv_kv,
    stride_kv_pos,
    stride_kv_h,
    stride_kv_d,
    stride_th,
    stride_tn,
    stride_tk,
    stride_o_c,
    stride_o_b,
    stride_o_h,
    stride_o_d,
    stride_l_c,
    stride_l_b,
    stride_l_h,
    stride_bt_b,
    BLOCK_SIZE_K: tl.constexpr,  # == SPARSE_BLOCK_SIZE (128)
    NUM_TOPK_CHUNKS: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
    BLOCK_SIZE_T: tl.constexpr,
    USE_FP8: tl.constexpr,  # fp8 KV cache: dequantize K/V to q.dtype on load
    USE_PDL: tl.constexpr,
):
    sm_scale_log2e = sm_scale * 1.4426950409
    # split-K over the topk dimension: pid(0) folds (query-token, chunk).
    pid_bc, pid_kh = tl.program_id(0), tl.program_id(1)
    pid_b = pid_bc % total_q
    pid_c = pid_bc // total_q
    req_id = pid_b // decode_query_len
    q_offset = pid_b - req_id * decode_query_len
    pid_h = pid_kh * gqa_group_size
    chunk_size_topk = (max_topk + NUM_TOPK_CHUNKS - 1) // NUM_TOPK_CHUNKS
    chunk_start_topk = pid_c * chunk_size_topk
    chunk_end_compiletime = chunk_start_topk + chunk_size_topk

    if USE_PDL:
        tl.extra.cuda.gdc_wait()

    seq_len = tl.load(seq_lens + req_id)
    query_pos = seq_len - decode_query_len + q_offset
    # Full-CG padding uses zero-length request rows. Clamp to an empty
    # attention range instead of letting padded rows produce negative lengths.
    kv_len = tl.maximum(query_pos + 1, 0)

    # number of valid (non-padded) selected blocks for this query token
    off_t = tl.arange(0, BLOCK_SIZE_T)
    idx_base = t_ptr + pid_kh * stride_th + pid_b * stride_tn
    topk_idx = tl.load(idx_base + off_t * stride_tk, mask=off_t < max_topk, other=-1)
    real_topk = tl.sum((topk_idx >= 0).to(tl.int32), axis=0)
    chunk_end_topk = tl.minimum(chunk_end_compiletime, real_topk)

    off_n = tl.arange(0, BLOCK_SIZE_K)
    off_d = tl.arange(0, BLOCK_SIZE_D)
    d_mask = off_d < head_dim
    bt_row = block_table_ptr + req_id * stride_bt_b

    m_i = tl.full((BLOCK_SIZE_H,), float("-inf"), dtype=tl.float32)
    lse_i = tl.full((BLOCK_SIZE_H,), float("-inf"), dtype=tl.float32)
    acc_o = tl.zeros((BLOCK_SIZE_H, BLOCK_SIZE_D), dtype=tl.float32)
    q_ptrs = tl.make_block_ptr(
        base=q_ptr + pid_b * stride_qn + pid_h * stride_qh,
        shape=(gqa_group_size, head_dim),
        strides=(stride_qh, stride_qd),
        offsets=(0, 0),
        block_shape=(BLOCK_SIZE_H, BLOCK_SIZE_D),
        order=(1, 0),
    )
    q = tl.load(q_ptrs, boundary_check=(0, 1), padding_option="zero")

    cur_idx_ptr = idx_base + chunk_start_topk * stride_tk
    for _ in tl.range(chunk_start_topk, chunk_end_topk):
        blk = tl.load(cur_idx_ptr).to(tl.int32)
        cur_idx_ptr = cur_idx_ptr + stride_tk
        c = blk * BLOCK_SIZE_K
        page = tl.load(bt_row + blk).to(tl.int64)
        pos = c + off_n
        pos_mask = pos < kv_len
        k = tl.load(
            kv_cache_ptr
            + page * stride_kv_blk
            + 0 * stride_kv_kv
            + off_n[None, :] * stride_kv_pos
            + pid_kh * stride_kv_h
            + off_d[:, None] * stride_kv_d,
            mask=d_mask[:, None] & pos_mask[None, :],
            other=0.0,
        )
        if USE_FP8:
            k = k.to(q.dtype)
        qk = tl.zeros((BLOCK_SIZE_H, BLOCK_SIZE_K), dtype=tl.float32)
        qk += tl.where(pos_mask[None, :], 0, float("-inf"))
        qk += tl.dot(q, k) * sm_scale_log2e
        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
        p = tl.exp2(qk - m_ij[:, None])
        l_ij = tl.sum(p, axis=1)
        acc_o = acc_o * tl.exp2(m_i - m_ij)[:, None]
        v = tl.load(
            kv_cache_ptr
            + page * stride_kv_blk
            + 1 * stride_kv_kv
            + off_n[:, None] * stride_kv_pos
            + pid_kh * stride_kv_h
            + off_d[None, :] * stride_kv_d,
            mask=pos_mask[:, None] & d_mask[None, :],
            other=0.0,
        )
        if USE_FP8:
            v = v.to(q.dtype)
        acc_o += tl.dot(p.to(v.dtype), v)
        m_i = m_ij
        lse_i = m_ij + tl.log2(tl.exp2(lse_i - m_ij) + l_ij)

    if USE_PDL:
        tl.extra.cuda.gdc_launch_dependents()

    # Empty chunks for active rows must store zero output; otherwise the merge
    # can hit 0 * NaN. All-empty padded rows may still produce NaNs in merge.
    scale = tl.where(lse_i > float("-inf"), tl.exp2(m_i - lse_i), tl.zeros_like(lse_i))
    acc_o = acc_o * scale[:, None]
    o_ptrs = tl.make_block_ptr(
        base=o_ptr + pid_c * stride_o_c + pid_b * stride_o_b + pid_h * stride_o_h,
        shape=(gqa_group_size, head_dim),
        strides=(stride_o_h, stride_o_d),
        offsets=(0, 0),
        block_shape=(BLOCK_SIZE_H, BLOCK_SIZE_D),
        order=(1, 0),
    )
    tl.store(o_ptrs, acc_o.to(o_ptr.dtype.element_ty), boundary_check=(0, 1))
    lse_ptrs = tl.make_block_ptr(
        base=lse_ptr + pid_c * stride_l_c + pid_b * stride_l_b + pid_h * stride_l_h,
        shape=(gqa_group_size,),
        strides=(stride_l_h,),
        offsets=(0,),
        block_shape=(BLOCK_SIZE_H,),
        order=(0,),
    )
    tl.store(lse_ptrs, lse_i.to(lse_ptr.dtype.element_ty), boundary_check=(0,))


@triton.heuristics({"BLOCK_SIZE_D": lambda args: triton.next_power_of_2(args["head_dim"])})
@triton.jit
def _merge_topk_attn_out_kernel(
    o_ptr,  # partials: [NUM_TOPK_CHUNKS, total_q, num_heads, head_dim]
    lse_ptr,  # partials (log2): [NUM_TOPK_CHUNKS, total_q, num_heads]
    out_ptr,  # merged out: [total_q, num_heads, head_dim]
    head_dim,
    stride_o_c,
    stride_o_b,
    stride_o_h,
    stride_o_d,
    stride_l_c,
    stride_l_b,
    stride_l_h,
    stride_out_n,
    stride_out_h,
    stride_out_d,
    NUM_TOPK_CHUNKS: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
    USE_PDL: tl.constexpr,
):
    pid_b, pid_h = tl.program_id(0), tl.program_id(1)

    # NOTE: assume seq_lens is safe to load before gdc_wait()
    if USE_PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()

    off_c = tl.arange(0, NUM_TOPK_CHUNKS)
    off_d = tl.arange(0, BLOCK_SIZE_D)
    o_ptrs = tl.make_block_ptr(
        base=o_ptr + pid_b * stride_o_b + pid_h * stride_o_h,
        shape=(NUM_TOPK_CHUNKS, head_dim),
        strides=(stride_o_c, stride_o_d),
        offsets=(0, 0),
        block_shape=(NUM_TOPK_CHUNKS, BLOCK_SIZE_D),
        order=(1, 0),
    )
    lse_ptrs = lse_ptr + pid_b * stride_l_b + pid_h * stride_l_h + off_c * stride_l_c
    o = tl.load(o_ptrs, boundary_check=(0, 1), padding_option="zero")
    lse = tl.load(lse_ptrs)  # empty chunks contribute -inf -> weight 0
    lse_max = tl.max(lse, axis=0)
    weights = tl.exp2(lse - lse_max)
    weights = weights / tl.sum(weights, axis=0)
    o_merged = tl.sum(o * weights[:, None], axis=0)
    out_ptrs = out_ptr + pid_b * stride_out_n + pid_h * stride_out_h + off_d * stride_out_d
    tl.store(out_ptrs, o_merged.to(out_ptr.dtype.element_ty), mask=off_d < head_dim)


# ---------------------------------------------------------------------------
# Python wrappers
# ---------------------------------------------------------------------------
@torch.no_grad()
def minimax_m3_sparse_attn(
    q: torch.Tensor,  # [total_q, num_heads, head_dim]
    kv_cache: torch.Tensor,  # [num_blocks, 2, 128, num_kv_heads, head_dim]
    topk_idx: torch.Tensor,  # [num_kv_heads, total_q, topk]
    block_table: torch.Tensor,  # [batch, max_blocks]
    cu_seqlens_q: torch.Tensor,  # [batch+1] int32
    seq_lens: torch.Tensor,  # [batch] int32
    prefix_lens: torch.Tensor,  # [batch] int32
    max_query_len: int,
    num_kv_heads: int,
    sm_scale: float,
    output: torch.Tensor,  # [total_q, num_heads, head_dim]
) -> None:
    """GQA block-sparse attention over the selected blocks. block_size_q == 1."""
    kv_cache = _as_triton_main_kv_cache(kv_cache)
    total_q, num_heads, head_dim = q.shape
    batch = cu_seqlens_q.shape[0] - 1
    topk = topk_idx.shape[-1]
    gqa_group_size = num_heads // num_kv_heads
    use_fp8 = kv_cache.dtype in (torch.float8_e4m3fn, torch.float8_e5m2)
    grid = (max_query_len, num_kv_heads, batch)
    _gqa_sparse_fwd_kernel[grid](
        q,
        kv_cache,
        topk_idx,
        output,
        block_table,
        cu_seqlens_q,
        cu_seqlens_q,  # cu_seqblocks_q == cu_seqlens_q when block_size_q == 1
        seq_lens,
        prefix_lens,
        num_kv_heads,
        gqa_group_size,
        head_dim,
        topk,
        1,  # num_q_loop
        sm_scale,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        kv_cache.stride(0),
        kv_cache.stride(1),
        kv_cache.stride(2),
        kv_cache.stride(3),
        kv_cache.stride(4),
        topk_idx.stride(0),
        topk_idx.stride(1),
        topk_idx.stride(2),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        block_table.stride(0),
        BLOCK_SIZE_Q=1,
        BLOCK_SIZE_K=SPARSE_BLOCK_SIZE,
        USE_FP8=use_fp8,
        **_sparse_attn_num_stages_kwarg(),
    )


@torch.no_grad()
def minimax_m3_sparse_attn_decode(
    q: torch.Tensor,  # [total_q, num_heads, head_dim]
    kv_cache: torch.Tensor,  # [num_blocks, 2, 128, num_kv_heads, head_dim]
    topk_idx: torch.Tensor,  # [num_kv_heads, total_q, topk]
    block_table: torch.Tensor,  # [num_reqs, max_blocks]
    seq_lens: torch.Tensor,  # [num_reqs] int32
    num_kv_heads: int,
    sm_scale: float,
    output: torch.Tensor,  # [total_q, num_heads, head_dim]
    decode_query_len: int,
) -> None:
    """GQA block-sparse attention for decode (split-K over the top-k blocks)."""
    kv_cache = _as_triton_main_kv_cache(kv_cache)
    total_q, num_heads, head_dim = q.shape
    assert total_q == seq_lens.shape[0] * decode_query_len
    max_topk = topk_idx.shape[-1]
    gqa_group_size = num_heads // num_kv_heads
    use_fp8 = kv_cache.dtype in (torch.float8_e4m3fn, torch.float8_e5m2)
    use_pdl = _is_arch_support_pdl()
    # `launch_pdl` is a Triton runtime kwarg only some backends accept (CUDA
    # SM9+); this ROCm Triton rejects it even when False ("Keyword argument
    # launch_pdl was specified but unrecognised"). Only pass it when PDL is
    # actually supported -- on ROCm use_pdl is always False, so it's omitted.
    pdl_launch = {"launch_pdl": True} if use_pdl else {}
    # split-K over the selected blocks; chunk count is shape-constant (cuda graph).
    TARGET_GRID = 256
    target = max(1, min(max_topk, TARGET_GRID // max(1, total_q * num_kv_heads)))
    num_topk_chunks = 1 << (target.bit_length() - 1)
    o_partial = torch.empty(num_topk_chunks, total_q, num_heads, head_dim, dtype=q.dtype, device=q.device)
    lse_partial = torch.empty(num_topk_chunks, total_q, num_heads, dtype=torch.float32, device=q.device)
    grid = (total_q * num_topk_chunks, num_kv_heads)
    _gqa_sparse_decode_kernel[grid](
        q,
        kv_cache,
        topk_idx,
        o_partial,
        lse_partial,
        block_table,
        seq_lens,
        total_q,
        gqa_group_size,
        head_dim,
        max_topk,
        sm_scale,
        decode_query_len,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        kv_cache.stride(0),
        kv_cache.stride(1),
        kv_cache.stride(2),
        kv_cache.stride(3),
        kv_cache.stride(4),
        topk_idx.stride(0),
        topk_idx.stride(1),
        topk_idx.stride(2),
        o_partial.stride(0),
        o_partial.stride(1),
        o_partial.stride(2),
        o_partial.stride(3),
        lse_partial.stride(0),
        lse_partial.stride(1),
        lse_partial.stride(2),
        block_table.stride(0),
        BLOCK_SIZE_K=SPARSE_BLOCK_SIZE,
        NUM_TOPK_CHUNKS=num_topk_chunks,
        USE_FP8=use_fp8,
        USE_PDL=use_pdl,
        **_sparse_attn_num_stages_kwarg(),
        **pdl_launch,
    )
    merge_grid = (total_q, num_heads)
    _merge_topk_attn_out_kernel[merge_grid](
        o_partial,
        lse_partial,
        output,
        head_dim,
        o_partial.stride(0),
        o_partial.stride(1),
        o_partial.stride(2),
        o_partial.stride(3),
        lse_partial.stride(0),
        lse_partial.stride(1),
        lse_partial.stride(2),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        NUM_TOPK_CHUNKS=num_topk_chunks,
        USE_PDL=use_pdl,
        **pdl_launch,
    )
