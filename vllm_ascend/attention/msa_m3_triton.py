# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton kernels for MiniMax M3 block-sparse GQA attention on Ascend.

Migrated from reference/vllm_cp/vllm/models/minimax_m3/common/ops/index_topk.py
and sparse_attn.py. The kernels accept K/V cache tensors directly so Ascend's
split cache layout does not need to be materialized into the GPU
``[num_blocks, 2, ...]`` layout.
"""

from __future__ import annotations

import torch

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils.math_utils import round_up

# One sparse block == one KV page.
SPARSE_BLOCK_SIZE = 128

TOPK_SELECTION_TILE = 128
TOPK_COMPUTE_MIN_TILE = 16
TOPK_NUM_WARPS = 4
TOPK_NUM_STAGES = 2


_FP8_DTYPES = (torch.float8_e4m3fn, torch.float8_e5m2)


def _split_triton_main_kv_cache(
    kv_cache: torch.Tensor | tuple[torch.Tensor, ...] | list[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(kv_cache, (tuple, list)):
        if len(kv_cache) < 2:
            raise ValueError("Main kv cache tuple must contain K and V tensors")
        k_cache, v_cache = kv_cache[0], kv_cache[1]
    else:
        if kv_cache.ndim != 5:
            raise ValueError(f"Unexpected main kv cache ndim: {kv_cache.ndim}")
        if kv_cache.shape[0] == 2:
            k_cache, v_cache = kv_cache[0], kv_cache[1]
        elif kv_cache.shape[1] == 2:
            k_cache, v_cache = kv_cache[:, 0], kv_cache[:, 1]
        else:
            raise ValueError(f"Unexpected main kv cache shape: {tuple(kv_cache.shape)}")
    if k_cache.ndim != 4 or v_cache.ndim != 4:
        raise ValueError(
            "Unexpected split main kv cache shapes: "
            f"{tuple(k_cache.shape)}, {tuple(v_cache.shape)}"
        )
    return k_cache, v_cache


def _is_arch_support_pdl() -> bool:
    if current_platform.device_name == "npu":
        return False
    is_supported = getattr(current_platform, "is_arch_support_pdl", None)
    return bool(is_supported()) if callable(is_supported) else False


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
            raise ValueError(
                f"Unexpected index cache head dim: {tuple(index_kv_cache.shape)}"
            )
        index_kv_cache = index_kv_cache.squeeze(2)
    if index_kv_cache.ndim != 3:
        raise ValueError(f"Unexpected index cache ndim: {index_kv_cache.ndim}")
    return index_kv_cache


_SPARSE_ATTN_NUM_STAGES_KWARG: dict | None = None


def _sparse_attn_num_stages_kwarg() -> dict:
    """Triton ``num_stages`` override for the sparse-attn GEMM kernels.

    Forced only where required: CDNA3 (gfx942) caps LDS at
    64 KB, and the default 2-stage pipeline double-buffers the 128x128 K/V tiles
    to ~66 KB ("out of resource: shared memory"), so pin gfx942 to a single
    stage (~32 KB, which fits). Everywhere else (NVIDIA, CDNA4 gfx950) return an
    empty kwarg and let Triton keep its own default -- don't second-guess it.
    Cached: the arch is fixed per process.
    """
    global _SPARSE_ATTN_NUM_STAGES_KWARG
    if _SPARSE_ATTN_NUM_STAGES_KWARG is None:
        kwarg: dict = {}
        if current_platform.is_rocm():
            from vllm.platforms.rocm import on_gfx942

            if on_gfx942():
                kwarg = {"num_stages": 1}
        _SPARSE_ATTN_NUM_STAGES_KWARG = kwarg
    return _SPARSE_ATTN_NUM_STAGES_KWARG


@triton.jit
def _select_topk_pairs(
    scores,
    indices,
    valid_mask,
    topk_size: tl.constexpr,
    block_size: tl.constexpr,
):
    off_k = tl.arange(0, block_size)
    off_t = tl.arange(0, topk_size)

    valid_i32 = valid_mask.to(tl.int32)
    work_valid_i32 = valid_i32
    work_scores = tl.where(work_valid_i32 != 0, scores, float("-inf"))
    selected_scores = tl.full((topk_size,), -1e30, dtype=tl.float32)
    selected_indices = tl.full((topk_size,), 0, dtype=tl.int32)

    for rank in tl.static_range(0, topk_size):
        best_score = tl.max(work_scores, axis=0)
        best_offset = tl.argmax(work_scores, axis=0).to(tl.int32)

        selected_i32 = (
            (off_k == best_offset).to(tl.int32) * work_valid_i32
        )
        best_index = tl.sum(
            tl.where(selected_i32 != 0, indices, 0),
            axis=0,
        ).to(tl.int32)
        has_best_i32 = (best_index > 0).to(tl.int32)

        selected_scores = tl.where(
            off_t == rank,
            tl.where(has_best_i32 != 0, best_score, -1e30),
            selected_scores,
        )
        selected_indices = tl.where(
            off_t == rank,
            tl.where(has_best_i32 != 0, best_index, 0),
            selected_indices,
        )

        selected_lane_i32 = (off_k == best_offset).to(tl.int32)
        work_valid_i32 = work_valid_i32 * (1 - selected_lane_i32)
        work_scores = tl.where(
            selected_lane_i32 != 0,
            float("-inf"),
            work_scores,
        )

    return selected_scores, selected_indices


@triton.jit
def _merge_topk_pairs(
    left_scores,
    left_indices,
    right_scores,
    right_indices,
    topk_size: tl.constexpr,
):
    off_t = tl.arange(0, topk_size)

    left_valid_i32 = (left_indices > 0).to(tl.int32)
    right_valid_i32 = (right_indices > 0).to(tl.int32)
    left_work = tl.where(left_valid_i32 != 0, left_scores, float("-inf"))
    right_work = tl.where(right_valid_i32 != 0, right_scores, float("-inf"))

    out_scores = tl.full((topk_size,), -1e30, dtype=tl.float32)
    out_indices = tl.full((topk_size,), 0, dtype=tl.int32)

    for rank in tl.static_range(0, topk_size):
        left_best = tl.max(left_work, axis=0)
        left_offset = tl.argmax(left_work, axis=0).to(tl.int32)
        right_best = tl.max(right_work, axis=0)
        right_offset = tl.argmax(right_work, axis=0).to(tl.int32)

        take_left_i32 = (left_best >= right_best).to(tl.int32)

        left_selected_i32 = (
            (off_t == left_offset).to(tl.int32) * left_valid_i32
        )
        right_selected_i32 = (
            (off_t == right_offset).to(tl.int32) * right_valid_i32
        )
        left_index = tl.sum(
            tl.where(left_selected_i32 != 0, left_indices, 0),
            axis=0,
        ).to(tl.int32)
        right_index = tl.sum(
            tl.where(right_selected_i32 != 0, right_indices, 0),
            axis=0,
        ).to(tl.int32)

        best_score = tl.where(take_left_i32 != 0, left_best, right_best)
        best_index = tl.where(take_left_i32 != 0, left_index, right_index)
        has_best_i32 = (best_index > 0).to(tl.int32)

        out_scores = tl.where(
            off_t == rank,
            tl.where(has_best_i32 != 0, best_score, -1e30),
            out_scores,
        )
        out_indices = tl.where(
            off_t == rank,
            tl.where(has_best_i32 != 0, best_index, 0),
            out_indices,
        )

        left_remove_i32 = left_selected_i32 * take_left_i32
        right_remove_i32 = right_selected_i32 * (1 - take_left_i32)
        left_valid_i32 = left_valid_i32 * (1 - left_remove_i32)
        right_valid_i32 = right_valid_i32 * (1 - right_remove_i32)
        left_work = tl.where(left_remove_i32 != 0, float("-inf"), left_work)
        right_work = tl.where(right_remove_i32 != 0, float("-inf"), right_work)

    return out_scores, out_indices


# ---------------------------------------------------------------------------
# Index block-score kernel (paged). score[h, token, block] = max over the
# 128-token block of (idx_q . index_k), causal-masked. BLOCK_SIZE_K == 128 so
# each K-tile is exactly one page (BLOCKS_PER_K_BLOCK == 1).
# ---------------------------------------------------------------------------
# since prefill metadata is sliced from mixed batch metadata, seq_lens and prefix_lens
# might lose pointer alignment, which trigger Triton recompiles. we don't actually
# need pointer alignment for those tensors anyway because we do scalar load.
@triton.jit(do_not_specialize_on_alignment=["seq_lens", "prefix_lens"])
def _index_block_score_kernel(
    q_ptr,  # idx_q: [total_q, num_idx_heads, head_dim]
    ik_cache_ptr,  # index-K cache: [num_blocks, 128, head_dim]
    score_ptr,  # [num_idx_heads, total_q, max_block]
    block_table_ptr,  # [num_reqs, max_blocks]
    cu_seqlens,  # [batch+1] query start offsets
    seq_lens,  # [batch] total K length
    prefix_lens,  # [batch] context length before this chunk's queries
    num_idx_heads,
    head_dim: tl.constexpr,
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
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,  # == SPARSE_BLOCK_SIZE (128)
):
    sm_scale_log2e = sm_scale * 1.4426950409
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
    # Block table row for this request.
    bt_row = block_table_ptr + pid_b * stride_bt_b
    # Causal window: only blocks up to the last query token's position.
    hi = min(seq_len, prefix_len + (pid_q + 1) * BLOCK_SIZE_Q)
    for i in tl.range(0, hi, BLOCK_SIZE_K):
        blk = i // BLOCK_SIZE_K
        page = tl.load(bt_row + blk).to(tl.int64)
        valid_page = page >= 0
        safe_page = tl.maximum(page, 0)
        pos = i + off_k
        # index-K for this page: [BLOCK_SIZE_D, BLOCK_SIZE_K] (transposed)
        # we don't need masked load for K, because KV cache ensures
        # allocation is multiple of BLOCK_SIZE_K.
        # for tokens beyond seqlen, they will be masked in qk later.
        k = tl.load(
            ik_cache_ptr
            + safe_page * stride_ik_blk
            + off_k[None, :] * stride_ik_pos
            + off_d[:, None] * stride_ik_d,
            mask=valid_page,
            other=0.0,
        )
        qk = tl.dot(q, k) * sm_scale_log2e
        qk = tl.where(valid_page, qk, float("-inf"))
        # apply causal mask as needed
        if q_start < i + BLOCK_SIZE_K:
            qk = tl.where(off_q[:, None] >= pos[None, :], qk, float("-inf"))
        # one sparse block per K-tile -> max over the 128 positions
        score = tl.max(qk, axis=1)  # [BLOCK_SIZE_Q]
        score = tl.where(valid_page, score, -1e30)
        s_ptrs = (
            score_ptr
            + pid_h * stride_s_h
            + (seq_start + pid_q * BLOCK_SIZE_Q + tl.arange(0, BLOCK_SIZE_Q))
            * stride_s_n
            + blk * stride_s_k
        )
        q_store_mask = (pid_q * BLOCK_SIZE_Q + tl.arange(0, BLOCK_SIZE_Q)) < q_len
        tl.store(s_ptrs, score, mask=q_store_mask)


# ---------------------------------------------------------------------------
# Prefill and decode top-k kernels.
# ---------------------------------------------------------------------------
@triton.jit(do_not_specialize_on_alignment=["prefix_lens"])
def _prefill_topk_partial_kernel(
    s_ptr,
    scores_partial_ptr,
    indices_partial_ptr,
    cu_seqlens,
    prefix_lens,
    init_blocks: tl.constexpr,
    local_blocks: tl.constexpr,
    chunk_blocks,
    num_chunks,
    stride_s_h,
    stride_s_n,
    stride_s_k,
    stride_ps_c,
    stride_ps_h,
    stride_ps_n,
    stride_ps_t,
    stride_pi_c,
    stride_pi_h,
    stride_pi_n,
    stride_pi_t,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_T: tl.constexpr,
    BLOCK_SIZE_BLOCK: tl.constexpr,
    MASK_INIT: tl.constexpr,
    MASK_LOCAL: tl.constexpr,
):
    pid_q = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_hc = tl.program_id(2)
    pid_h = pid_hc // num_chunks
    pid_chunk = pid_hc - pid_h * num_chunks

    seq_start = tl.load(cu_seqlens + pid_b)
    q_len = tl.load(cu_seqlens + pid_b + 1) - seq_start
    prefix_len = tl.load(prefix_lens + pid_b)
    if pid_q >= q_len:
        return

    off_k = tl.arange(0, BLOCK_SIZE_K)
    off_t = tl.arange(0, BLOCK_SIZE_T)
    query_idx = seq_start + pid_q
    valid_blocks = (prefix_len + pid_q + BLOCK_SIZE_BLOCK) // BLOCK_SIZE_BLOCK
    chunk_start = pid_chunk * chunk_blocks
    block_ids = chunk_start + off_k
    valid_i32 = (
        ((block_ids < valid_blocks) & (off_k < chunk_blocks)).to(tl.int32)
    )

    score = tl.load(
        s_ptr
        + pid_h * stride_s_h
        + query_idx * stride_s_n
        + block_ids * stride_s_k,
        mask=valid_i32 != 0,
        other=-1e30,
    ).to(tl.float32)
    score = tl.where(score != score, -1e30, score)

    init_i32 = (block_ids < init_blocks).to(tl.int32)
    local_i32 = (
        block_ids >= tl.maximum(0, valid_blocks - local_blocks)
    ).to(tl.int32)
    if MASK_INIT:
        score = tl.where(
            (valid_i32 * init_i32) != 0,
            score - 1e29,
            score,
        )
    else:
        score = tl.where(
            (valid_i32 * init_i32) != 0,
            1e30,
            score,
        )
    if MASK_LOCAL:
        score = tl.where(
            (valid_i32 * local_i32) != 0,
            score - 1e28,
            score,
        )
    else:
        score = tl.where(
            (valid_i32 * local_i32) != 0,
            1e29,
            score,
        )

    indices = tl.where(valid_i32 != 0, block_ids + 1, 0).to(tl.int32)
    selected_scores, selected_indices = _select_topk_pairs(
        score,
        indices,
        valid_i32 != 0,
        BLOCK_SIZE_T,
        BLOCK_SIZE_K,
    )

    tl.store(
        scores_partial_ptr
        + pid_chunk * stride_ps_c
        + pid_h * stride_ps_h
        + query_idx * stride_ps_n
        + off_t * stride_ps_t,
        selected_scores,
    )
    tl.store(
        indices_partial_ptr
        + pid_chunk * stride_pi_c
        + pid_h * stride_pi_h
        + query_idx * stride_pi_n
        + off_t * stride_pi_t,
        selected_indices,
    )


@triton.jit(do_not_specialize=["num_input_chunks"])
def _topk_pair_merge_kernel(
    in_scores_ptr,
    in_indices_ptr,
    out_scores_ptr,
    out_indices_ptr,
    num_input_chunks,
    stride_is_c,
    stride_is_h,
    stride_is_n,
    stride_is_t,
    stride_ii_c,
    stride_ii_h,
    stride_ii_n,
    stride_ii_t,
    stride_os_c,
    stride_os_h,
    stride_os_n,
    stride_os_t,
    stride_oi_c,
    stride_oi_h,
    stride_oi_n,
    stride_oi_t,
    BLOCK_SIZE_T: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_out_chunk = tl.program_id(2)

    off_t = tl.arange(0, BLOCK_SIZE_T)
    left_chunk = 2 * pid_out_chunk
    right_chunk = left_chunk + 1
    right_exists_i32 = (right_chunk < num_input_chunks).to(tl.int32)

    left_scores = tl.load(
        in_scores_ptr
        + left_chunk * stride_is_c
        + pid_h * stride_is_h
        + pid_n * stride_is_n
        + off_t * stride_is_t,
    ).to(tl.float32)
    left_indices = tl.load(
        in_indices_ptr
        + left_chunk * stride_ii_c
        + pid_h * stride_ii_h
        + pid_n * stride_ii_n
        + off_t * stride_ii_t,
    ).to(tl.int32)
    right_scores = tl.load(
        in_scores_ptr
        + right_chunk * stride_is_c
        + pid_h * stride_is_h
        + pid_n * stride_is_n
        + off_t * stride_is_t,
        mask=right_exists_i32 != 0,
        other=-1e30,
    ).to(tl.float32)
    right_indices = tl.load(
        in_indices_ptr
        + right_chunk * stride_ii_c
        + pid_h * stride_ii_h
        + pid_n * stride_ii_n
        + off_t * stride_ii_t,
        mask=right_exists_i32 != 0,
        other=0,
    ).to(tl.int32)

    merged_scores, merged_indices = _merge_topk_pairs(
        left_scores,
        left_indices,
        right_scores,
        right_indices,
        BLOCK_SIZE_T,
    )

    tl.store(
        out_scores_ptr
        + pid_out_chunk * stride_os_c
        + pid_h * stride_os_h
        + pid_n * stride_os_n
        + off_t * stride_os_t,
        merged_scores,
    )
    tl.store(
        out_indices_ptr
        + pid_out_chunk * stride_oi_c
        + pid_h * stride_oi_h
        + pid_n * stride_oi_n
        + off_t * stride_oi_t,
        merged_indices,
    )


@triton.jit
def _topk_finalize_kernel(
    indices_partial_ptr,
    indices_final_ptr,
    topk,
    stride_pi_c,
    stride_pi_h,
    stride_pi_n,
    stride_pi_t,
    stride_f_h,
    stride_f_n,
    stride_f_t,
    BLOCK_SIZE_T: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_h = tl.program_id(1)
    off_t = tl.arange(0, BLOCK_SIZE_T)

    indices = tl.load(
        indices_partial_ptr
        + pid_h * stride_pi_h
        + pid_n * stride_pi_n
        + off_t * stride_pi_t,
    ).to(tl.int32)
    output = tl.where(
        (off_t < topk) & (indices > 0),
        indices - 1,
        -1,
    )
    tl.store(
        indices_final_ptr
        + pid_h * stride_f_h
        + pid_n * stride_f_n
        + off_t * stride_f_t,
        output.to(indices_final_ptr.dtype.element_ty),
        mask=off_t < topk,
    )


# ---------------------------------------------------------------------------
# Decode index-score kernel (split-K over seq blocks). Decode batches are
# flattened request-major, with a runtime query length used to map each query
# token back to its request metadata. The score stage uses a fixed split-K
# grid, while the top-k stage is NPU-oriented. Base-2 score scaling matches
# prefill.
# ---------------------------------------------------------------------------
@triton.jit(do_not_specialize=["num_kv_chunks", "decode_query_len"])
def _decode_index_score_kernel(
    q_ptr,  # idx_q: [total_q, num_idx_heads, head_dim]
    ik_cache_ptr,  # index-K cache: [num_blocks, 128, head_dim]
    score_ptr,  # [num_idx_heads, total_q, max_block]
    block_table_ptr,  # [num_reqs, max_blocks]
    seq_lens,  # [num_reqs]
    num_idx_heads: tl.constexpr,
    head_dim: tl.constexpr,
    init_blocks,
    local_blocks,
    sm_scale,
    decode_query_len,
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
    BLOCK_SIZE_K: tl.constexpr,  # == SPARSE_BLOCK_SIZE (128)
    num_kv_chunks,
):
    sm_scale_log2e = sm_scale * 1.4426950409
    pid_b = tl.program_id(0)  # flattened query-token id
    pid_c = tl.program_id(1)
    req_id = pid_b // decode_query_len
    q_offset = pid_b - req_id * decode_query_len

    seq_len = tl.load(seq_lens + req_id)
    query_pos = seq_len - decode_query_len + q_offset
    # Full-shape padding uses zero-length request rows. Clamp to an empty
    # attention range instead of letting padded rows produce negative lengths.
    kv_len = tl.maximum(query_pos + 1, 0)
    num_blocks = (kv_len + BLOCK_SIZE_K - 1) // BLOCK_SIZE_K

    # Block-aligned fixed-count split.  The grid depends only on the launch
    # shape, while each program masks to its request's actual valid block range.
    chunk_size_blocks = (num_blocks + num_kv_chunks - 1) // num_kv_chunks
    chunk_start_block = pid_c * chunk_size_blocks
    chunk_end_block = tl.minimum(chunk_start_block + chunk_size_blocks, num_blocks)
    if chunk_start_block >= chunk_end_block:
        return

    off_k = tl.arange(0, BLOCK_SIZE_K)
    off_d = tl.arange(0, head_dim)
    bt_row = block_table_ptr + req_id * stride_bt_b
    local_start = tl.maximum(0, num_blocks - local_blocks)

    q = tl.load(
        q_ptr
        + pid_b * stride_q_n
        + tl.arange(0, num_idx_heads) * stride_q_h
        + off_d[:, None] * stride_q_d,
    )  # [D, H]

    for blk in tl.range(chunk_start_block, chunk_end_block):
        page = tl.load(bt_row + blk).to(tl.int64)
        valid_page = page >= 0
        safe_page = tl.maximum(page, 0)
        pos = blk * BLOCK_SIZE_K + off_k
        pos_mask = pos < kv_len
        k = tl.load(
            ik_cache_ptr
            + safe_page * stride_ik_blk
            + off_k[:, None] * stride_ik_pos
            + off_d * stride_ik_d,
            mask=valid_page,
            other=0.0,
        )  # [N, D]

        kq = tl.dot(k, q) * sm_scale_log2e  # [N, H]
        kq = tl.where(pos_mask[:, None], kq, float("-inf"))
        score = tl.max(kq, axis=0)  # [H]

        is_init = blk < init_blocks
        is_local = (blk >= local_start) & (blk < num_blocks)
        score = tl.where(valid_page, score, -1e30)
        score = tl.where(is_local & valid_page, 1e29, tl.where(is_init & valid_page, 1e30, score))
        tl.store(
            score_ptr
            + tl.arange(0, num_idx_heads) * stride_s_h
            + pid_b * stride_s_n
            + blk * stride_s_k,
            score,
        )


# ---------------------------------------------------------------------------
# Decode local top-k.
# ---------------------------------------------------------------------------
@triton.jit(do_not_specialize=["chunk_blocks", "decode_query_len"])
def _decode_topk_partial_kernel(
    s_ptr,
    scores_partial_ptr,
    indices_partial_ptr,
    seq_lens,
    chunk_blocks,
    decode_query_len,
    stride_s_h,
    stride_s_n,
    stride_s_k,
    stride_ps_c,
    stride_ps_h,
    stride_ps_n,
    stride_ps_t,
    stride_pi_c,
    stride_pi_h,
    stride_pi_n,
    stride_pi_t,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_T: tl.constexpr,
    BLOCK_SIZE_BLOCK: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_chunk = tl.program_id(2)

    req_id = pid_n // decode_query_len
    q_offset = pid_n - req_id * decode_query_len
    seq_len = tl.load(seq_lens + req_id)
    query_pos = seq_len - decode_query_len + q_offset
    kv_len = tl.maximum(query_pos + 1, 0)
    num_blocks = (kv_len + BLOCK_SIZE_BLOCK - 1) // BLOCK_SIZE_BLOCK

    off_k = tl.arange(0, BLOCK_SIZE_K)
    off_t = tl.arange(0, BLOCK_SIZE_T)
    chunk_start = pid_chunk * chunk_blocks
    block_ids = chunk_start + off_k
    valid_i32 = (
        ((block_ids < num_blocks) & (off_k < chunk_blocks)).to(tl.int32)
    )

    scores = tl.load(
        s_ptr
        + pid_h * stride_s_h
        + pid_n * stride_s_n
        + block_ids * stride_s_k,
        mask=valid_i32 != 0,
        other=-1e30,
    ).to(tl.float32)
    scores = tl.where(scores != scores, -1e30, scores)
    indices = tl.where(valid_i32 != 0, block_ids + 1, 0).to(tl.int32)

    selected_scores, selected_indices = _select_topk_pairs(
        scores,
        indices,
        valid_i32 != 0,
        BLOCK_SIZE_T,
        BLOCK_SIZE_K,
    )

    tl.store(
        scores_partial_ptr
        + pid_chunk * stride_ps_c
        + pid_h * stride_ps_h
        + pid_n * stride_ps_n
        + off_t * stride_ps_t,
        selected_scores,
    )
    tl.store(
        indices_partial_ptr
        + pid_chunk * stride_pi_c
        + pid_h * stride_pi_h
        + pid_n * stride_pi_n
        + off_t * stride_pi_t,
        selected_indices,
    )


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
        "BLOCK_SIZE_QH": lambda args: args["BLOCK_SIZE_Q"]
        * triton.next_power_of_2(args["gqa_group_size"]),
    }
)
@triton.jit(do_not_specialize_on_alignment=["seq_lens", "prefix_lens"])
def _gqa_sparse_fwd_kernel(
    q_ptr,  # [total_q, num_heads, head_dim]
    k_cache_ptr,  # [num_blocks, 128, num_kv_heads, head_dim]
    v_cache_ptr,  # [num_blocks, 128, num_kv_heads, head_dim]
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
    stride_k_blk,
    stride_k_pos,
    stride_k_h,
    stride_k_d,
    stride_v_blk,
    stride_v_pos,
    stride_v_h,
    stride_v_d,
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
            valid_blk = blk >= 0
            safe_blk = tl.maximum(blk, 0)
            c = safe_blk * BLOCK_SIZE_K
            page = tl.load(bt_row + safe_blk, mask=valid_blk, other=-1).to(tl.int64)
            valid_page = valid_blk & (page >= 0)
            safe_page = tl.maximum(page, 0)
            pos = c + off_n
            pos_mask = (pos < seq_len) & valid_page
            k = tl.load(
                k_cache_ptr
                + safe_page * stride_k_blk
                + off_n[None, :] * stride_k_pos
                + pid_kh * stride_k_h
                + off_d[:, None] * stride_k_d,
                mask=d_mask[:, None],
                other=0.0,
            )
            if USE_FP8:
                k = k.to(q.dtype)
            qk = tl.zeros((BLOCK_SIZE_Q, BLOCK_SIZE_H, BLOCK_SIZE_K), dtype=tl.float32)
            # causal: q_abs_pos - k_off >= block_start (c)
            qk += tl.where(off_q[:, None, :] >= c, 0, float("-inf"))
            qk = tl.reshape(qk, BLOCK_SIZE_QH, BLOCK_SIZE_K)
            qk += tl.dot(q, k) * sm_scale_log2e
            qk = tl.where(pos_mask[None, :], qk, float("-inf"))
            m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
            active_i = m_ij > float("-inf")
            p = tl.exp2(qk - m_ij[:, None])
            p = tl.where(active_i[:, None], p, 0.0)
            l_ij = tl.sum(p, axis=1)
            acc_scale = tl.where(active_i, tl.exp2(m_i - m_ij), tl.zeros_like(m_i))
            acc_o = acc_o * acc_scale[:, None]
            v = tl.load(
                v_cache_ptr
                + safe_page * stride_v_blk
                + off_n[:, None] * stride_v_pos
                + pid_kh * stride_v_h
                + off_d[None, :] * stride_v_d,
                mask=d_mask[None, :],
                other=0.0,
            )
            if USE_FP8:
                v = v.to(q.dtype)
            acc_o += tl.dot(p.to(v.dtype), v)
            m_i = m_ij
            lse_next = m_ij + tl.log2(tl.exp2(lse_i - m_ij) + l_ij)
            lse_i = tl.where(active_i, lse_next, lse_i)
        has_lse = lse_i > float("-inf")
        acc_o = tl.where(
            has_lse[:, None],
            acc_o * tl.exp2(m_i - lse_i)[:, None],
            tl.zeros_like(acc_o),
        )
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
        "BLOCK_SIZE_H": lambda args: max(
            16, triton.next_power_of_2(args["gqa_group_size"])
        ),
        "BLOCK_SIZE_D": lambda args: triton.next_power_of_2(args["head_dim"]),
        "BLOCK_SIZE_T": lambda args: triton.next_power_of_2(args["max_topk"]),
    }
)
@triton.jit(do_not_specialize=["decode_query_len"])
def _gqa_sparse_decode_kernel(
    q_ptr,  # [total_q, num_heads, head_dim]
    k_cache_ptr,  # [num_blocks, 128, num_kv_heads, head_dim]
    v_cache_ptr,  # [num_blocks, 128, num_kv_heads, head_dim]
    t_ptr,  # topk_idx: [num_kv_heads, total_q, topk]
    o_ptr,  # partial out: [NUM_TOPK_CHUNKS, total_q, num_heads, head_dim]
    lse_ptr,  # partial lse (log2): [NUM_TOPK_CHUNKS, total_q, num_heads]
    block_table_ptr,  # [num_reqs, max_blocks]
    seq_lens,  # [num_reqs]
    max_blocks,
    total_q,
    gqa_group_size,
    head_dim,
    max_topk,
    sm_scale,
    decode_query_len,
    stride_qn,
    stride_qh,
    stride_qd,
    stride_k_blk,
    stride_k_pos,
    stride_k_h,
    stride_k_d,
    stride_v_blk,
    stride_v_pos,
    stride_v_h,
    stride_v_d,
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
    if USE_PDL:
        tl.extra.cuda.gdc_wait()

    seq_len = tl.load(seq_lens + req_id)
    query_pos = seq_len - decode_query_len + q_offset
    # Full-CG padding uses zero-length request rows. Clamp to an empty
    # attention range instead of letting padded rows produce negative lengths.
    kv_len = tl.maximum(query_pos + 1, 0)
    num_valid_blocks = (kv_len + BLOCK_SIZE_K - 1) // BLOCK_SIZE_K

    idx_base = t_ptr + pid_kh * stride_th + pid_b * stride_tn
    off_t = tl.arange(0, BLOCK_SIZE_T)
    topk_idx = tl.load(
        idx_base + off_t * stride_tk,
        mask=off_t < max_topk,
        other=-1,
    )
    topk_valid_blk = (
        (topk_idx >= 0)
        & (topk_idx < num_valid_blocks)
        & (topk_idx < max_blocks)
    )
    valid_topk = (off_t < max_topk) & topk_valid_blk
    # Keep all positions up to the last valid selected block. This skips only
    # a trailing invalid suffix; the hot loop still has PR33 per-block guards,
    # so interleaved invalid entries remain precision-safe.
    real_topk = tl.max(tl.where(valid_topk, off_t + 1, 0), axis=0)
    chunk_size_topk = tl.maximum(
        1,
        (real_topk + NUM_TOPK_CHUNKS - 1) // NUM_TOPK_CHUNKS,
    )
    chunk_start_topk = pid_c * chunk_size_topk
    chunk_end_topk = tl.minimum(chunk_start_topk + chunk_size_topk, real_topk)

    off_n = tl.arange(0, BLOCK_SIZE_K)
    off_d = tl.arange(0, BLOCK_SIZE_D)
    d_mask = off_d < head_dim
    bt_row = block_table_ptr + req_id * stride_bt_b

    if chunk_start_topk >= chunk_end_topk:
        lse_ptrs = tl.make_block_ptr(
            base=lse_ptr + pid_c * stride_l_c + pid_b * stride_l_b + pid_h * stride_l_h,
            shape=(gqa_group_size,),
            strides=(stride_l_h,),
            offsets=(0,),
            block_shape=(BLOCK_SIZE_H,),
            order=(0,),
        )
        empty_lse = tl.full((BLOCK_SIZE_H,), float("-inf"), dtype=tl.float32)
        tl.store(lse_ptrs, empty_lse, boundary_check=(0,))
        if USE_PDL:
            tl.extra.cuda.gdc_launch_dependents()
        return

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
        cur_idx_ptr += stride_tk
        valid_blk = (blk >= 0) & (blk < num_valid_blocks) & (blk < max_blocks)
        safe_blk = tl.minimum(tl.maximum(blk, 0), max_blocks - 1)
        c = safe_blk * BLOCK_SIZE_K
        page = tl.load(bt_row + safe_blk, mask=valid_blk, other=-1).to(tl.int64)
        valid_page = valid_blk & (page >= 0)
        safe_page = tl.maximum(page, 0)
        pos = c + off_n
        pos_mask = (pos < kv_len) & valid_page
        k = tl.load(
            k_cache_ptr
            + safe_page * stride_k_blk
            + off_n[None, :] * stride_k_pos
            + pid_kh * stride_k_h
            + off_d[:, None] * stride_k_d,
            mask=d_mask[:, None],
            other=0.0,
        )
        if USE_FP8:
            k = k.to(q.dtype)
        qk = tl.dot(q, k) * sm_scale_log2e
        qk = tl.where(pos_mask[None, :], qk, float("-inf"))
        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
        active_i = m_ij > float("-inf")
        p = tl.exp2(qk - m_ij[:, None])
        p = tl.where(active_i[:, None], p, 0.0)
        l_ij = tl.sum(p, axis=1)
        acc_scale = tl.where(active_i, tl.exp2(m_i - m_ij), tl.zeros_like(m_i))
        acc_o = acc_o * acc_scale[:, None]
        v = tl.load(
            v_cache_ptr
            + safe_page * stride_v_blk
            + off_n[:, None] * stride_v_pos
            + pid_kh * stride_v_h
            + off_d[None, :] * stride_v_d,
            mask=d_mask[None, :],
            other=0.0,
        )
        if USE_FP8:
            v = v.to(q.dtype)
        acc_o += tl.dot(p.to(v.dtype), v)
        m_i = m_ij
        lse_next = m_ij + tl.log2(tl.exp2(lse_i - m_ij) + l_ij)
        lse_i = tl.where(active_i, lse_next, lse_i)

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


@triton.heuristics(
    {"BLOCK_SIZE_D": lambda args: triton.next_power_of_2(args["head_dim"])}
)
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
    lse_ptrs = lse_ptr + pid_b * stride_l_b + pid_h * stride_l_h + off_c * stride_l_c
    lse = tl.load(lse_ptrs)  # empty chunks contribute -inf -> weight 0
    valid_chunk = lse > float("-inf")
    o = tl.load(
        o_ptr
        + off_c[:, None] * stride_o_c
        + pid_b * stride_o_b
        + pid_h * stride_o_h
        + off_d[None, :] * stride_o_d,
        mask=valid_chunk[:, None] & (off_d[None, :] < head_dim),
        other=0.0,
    )
    lse_max = tl.max(lse, axis=0)
    has_lse = lse_max > float("-inf")
    safe_lse_max = tl.where(has_lse, lse_max, 0.0)
    weights = tl.where(lse > float("-inf"), tl.exp2(lse - safe_lse_max), 0.0)
    denom = tl.sum(weights, axis=0)
    denom_safe = tl.where(denom > 0.0, denom, 1.0)
    o_merged = tl.sum(o * weights[:, None], axis=0) / denom_safe
    o_merged = tl.where(has_lse, o_merged, tl.zeros((BLOCK_SIZE_D,), dtype=tl.float32))
    out_ptrs = (
        out_ptr + pid_b * stride_out_n + pid_h * stride_out_h + off_d * stride_out_d
    )
    tl.store(out_ptrs, o_merged.to(out_ptr.dtype.element_ty), mask=off_d < head_dim)


def _topk_compute_width(topk: int) -> int:
    return max(TOPK_COMPUTE_MIN_TILE, triton.next_power_of_2(topk))


def _topk_select_width(topk: int) -> int:
    return max(TOPK_SELECTION_TILE, _topk_compute_width(topk))


def _merge_topk_levels(
    scores: torch.Tensor,
    indices: torch.Tensor,
    num_heads: int,
    total_q: int,
    topk_width: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    current_scores = scores
    current_indices = indices
    current_chunks = scores.shape[0]
    while current_chunks > 1:
        next_chunks = triton.cdiv(current_chunks, 2)
        next_scores = torch.empty(
            (next_chunks, num_heads, total_q, topk_width),
            dtype=torch.float32,
            device=scores.device,
        )
        next_indices = torch.empty(
            (next_chunks, num_heads, total_q, topk_width),
            dtype=torch.int32,
            device=indices.device,
        )
        _topk_pair_merge_kernel[(total_q, num_heads, next_chunks)](
            current_scores,
            current_indices,
            next_scores,
            next_indices,
            current_chunks,
            current_scores.stride(0),
            current_scores.stride(1),
            current_scores.stride(2),
            current_scores.stride(3),
            current_indices.stride(0),
            current_indices.stride(1),
            current_indices.stride(2),
            current_indices.stride(3),
            next_scores.stride(0),
            next_scores.stride(1),
            next_scores.stride(2),
            next_scores.stride(3),
            next_indices.stride(0),
            next_indices.stride(1),
            next_indices.stride(2),
            next_indices.stride(3),
            BLOCK_SIZE_T=topk_width,
            num_warps=TOPK_NUM_WARPS,
            num_stages=TOPK_NUM_STAGES,
        )
        current_scores = next_scores
        current_indices = next_indices
        current_chunks = next_chunks
    return current_scores, current_indices


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
    index_kv_cache = _as_triton_index_kv_cache(index_kv_cache)
    total_q, num_idx_heads, head_dim = idx_q.shape
    assert num_idx_heads == num_kv_heads
    batch = cu_seqlens_q.shape[0] - 1
    max_block = triton.cdiv(max_seq_len, SPARSE_BLOCK_SIZE)
    score_block_stride = round_up(max_block, 16)
    score = torch.empty(
        (num_idx_heads, total_q, score_block_stride),
        dtype=torch.float32,
        device=idx_q.device,
    )
    block_size_q = 64
    _index_block_score_kernel[
        (triton.cdiv(max_query_len, block_size_q), batch * num_idx_heads)
    ](
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
        BLOCK_SIZE_Q=block_size_q,
        BLOCK_SIZE_K=SPARSE_BLOCK_SIZE,
    )
    return score


@torch.no_grad()
def minimax_m3_index_topk(
    score: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    prefix_lens: torch.Tensor,
    max_query_len: int,
    topk: int,
    init_blocks: int,
    local_blocks: int,
) -> torch.Tensor:
    assert topk > 0

    num_heads, total_q, score_block_stride = score.shape
    batch = cu_seqlens_q.shape[0] - 1
    topk_width = _topk_compute_width(topk)
    select_width = _topk_select_width(topk)
    num_chunks = max(1, triton.cdiv(score_block_stride, select_width))
    chunk_blocks = triton.cdiv(score_block_stride, num_chunks)

    partial_scores = torch.empty(
        (num_chunks, num_heads, total_q, topk_width),
        dtype=torch.float32,
        device=score.device,
    )
    partial_indices = torch.empty(
        (num_chunks, num_heads, total_q, topk_width),
        dtype=torch.int32,
        device=score.device,
    )

    _prefill_topk_partial_kernel[
        (max_query_len, batch, num_heads * num_chunks)
    ](
        score,
        partial_scores,
        partial_indices,
        cu_seqlens_q,
        prefix_lens,
        init_blocks,
        local_blocks,
        chunk_blocks,
        num_chunks,
        score.stride(0),
        score.stride(1),
        score.stride(2),
        partial_scores.stride(0),
        partial_scores.stride(1),
        partial_scores.stride(2),
        partial_scores.stride(3),
        partial_indices.stride(0),
        partial_indices.stride(1),
        partial_indices.stride(2),
        partial_indices.stride(3),
        BLOCK_SIZE_K=select_width,
        BLOCK_SIZE_T=topk_width,
        BLOCK_SIZE_BLOCK=SPARSE_BLOCK_SIZE,
        MASK_INIT=False,
        MASK_LOCAL=False,
        num_warps=TOPK_NUM_WARPS,
        num_stages=TOPK_NUM_STAGES,
    )

    _, final_indices = _merge_topk_levels(
        partial_scores,
        partial_indices,
        num_heads,
        total_q,
        topk_width,
    )
    topk_idx = torch.empty(
        (num_heads, total_q, topk),
        dtype=torch.int32,
        device=score.device,
    )
    _topk_finalize_kernel[(total_q, num_heads)](
        final_indices,
        topk_idx,
        topk,
        final_indices.stride(0),
        final_indices.stride(1),
        final_indices.stride(2),
        final_indices.stride(3),
        topk_idx.stride(0),
        topk_idx.stride(1),
        topk_idx.stride(2),
        BLOCK_SIZE_T=topk_width,
        num_warps=TOPK_NUM_WARPS,
        num_stages=TOPK_NUM_STAGES,
    )
    return topk_idx


@torch.no_grad()
def minimax_m3_index_decode(
    idx_q: torch.Tensor,
    index_kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    max_seq_len: int,
    topk: int,
    init_blocks: int,
    local_blocks: int,
    num_kv_heads: int,
    sm_scale: float,
    decode_query_len: int,
) -> torch.Tensor:
    index_kv_cache = _as_triton_index_kv_cache(index_kv_cache)
    total_q, num_idx_heads, head_dim = idx_q.shape
    assert num_idx_heads == num_kv_heads
    assert total_q == seq_lens.shape[0] * decode_query_len
    assert topk > 0

    max_block = triton.cdiv(max_seq_len, SPARSE_BLOCK_SIZE)
    score_block_stride = round_up(max_block, 16)
    score = torch.empty(
        (num_idx_heads, total_q, score_block_stride),
        dtype=torch.float32,
        device=idx_q.device,
    )

    # On Ascend, over-splitting the decode score pass is dominated by scalar
    # scheduling and many empty programs. Keep a small split-K count and let
    # each program score multiple 128-token blocks with q reused in UB.
    target_grid = 4096
    max_num_kv_chunks = 8
    target = max(
        1,
        min(
            max_num_kv_chunks,
            target_grid // max(1, total_q * num_idx_heads),
        ),
    )
    num_kv_chunks = 1 << (target.bit_length() - 1)
    _decode_index_score_kernel[(total_q, num_kv_chunks)](
        idx_q,
        index_kv_cache,
        score,
        block_table,
        seq_lens,
        num_idx_heads,
        head_dim,
        init_blocks,
        local_blocks,
        sm_scale,
        decode_query_len,
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
        num_kv_chunks=num_kv_chunks,
    )

    topk_width = _topk_compute_width(topk)
    select_width = _topk_select_width(topk)
    num_chunks = max(1, triton.cdiv(score_block_stride, select_width))
    chunk_blocks = triton.cdiv(score_block_stride, num_chunks)

    partial_scores = torch.empty(
        (num_chunks, num_idx_heads, total_q, topk_width),
        dtype=torch.float32,
        device=idx_q.device,
    )
    partial_indices = torch.empty(
        (num_chunks, num_idx_heads, total_q, topk_width),
        dtype=torch.int32,
        device=idx_q.device,
    )
    _decode_topk_partial_kernel[(total_q, num_idx_heads, num_chunks)](
        score,
        partial_scores,
        partial_indices,
        seq_lens,
        chunk_blocks,
        decode_query_len,
        score.stride(0),
        score.stride(1),
        score.stride(2),
        partial_scores.stride(0),
        partial_scores.stride(1),
        partial_scores.stride(2),
        partial_scores.stride(3),
        partial_indices.stride(0),
        partial_indices.stride(1),
        partial_indices.stride(2),
        partial_indices.stride(3),
        BLOCK_SIZE_K=select_width,
        BLOCK_SIZE_T=topk_width,
        BLOCK_SIZE_BLOCK=SPARSE_BLOCK_SIZE,
        num_warps=TOPK_NUM_WARPS,
        num_stages=TOPK_NUM_STAGES,
    )

    _, final_indices = _merge_topk_levels(
        partial_scores,
        partial_indices,
        num_idx_heads,
        total_q,
        topk_width,
    )
    topk_idx = torch.empty(
        (num_idx_heads, total_q, topk),
        dtype=torch.int32,
        device=idx_q.device,
    )
    _topk_finalize_kernel[(total_q, num_idx_heads)](
        final_indices,
        topk_idx,
        topk,
        final_indices.stride(0),
        final_indices.stride(1),
        final_indices.stride(2),
        final_indices.stride(3),
        topk_idx.stride(0),
        topk_idx.stride(1),
        topk_idx.stride(2),
        BLOCK_SIZE_T=topk_width,
        num_warps=TOPK_NUM_WARPS,
        num_stages=TOPK_NUM_STAGES,
    )
    return topk_idx
@torch.no_grad()
def minimax_m3_sparse_attn(
    q: torch.Tensor,  # [total_q, num_heads, head_dim]
    kv_cache: torch.Tensor | tuple[torch.Tensor, ...] | list[torch.Tensor],
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
    k_cache, v_cache = _split_triton_main_kv_cache(kv_cache)
    total_q, num_heads, head_dim = q.shape
    batch = cu_seqlens_q.shape[0] - 1
    topk = topk_idx.shape[-1]
    gqa_group_size = num_heads // num_kv_heads
    use_fp8 = k_cache.dtype in _FP8_DTYPES or v_cache.dtype in _FP8_DTYPES
    grid = (max_query_len, num_kv_heads, batch)
    _gqa_sparse_fwd_kernel[grid](
        q,
        k_cache,
        v_cache,
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
        k_cache.stride(0),
        k_cache.stride(1),
        k_cache.stride(2),
        k_cache.stride(3),
        v_cache.stride(0),
        v_cache.stride(1),
        v_cache.stride(2),
        v_cache.stride(3),
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
    kv_cache: torch.Tensor | tuple[torch.Tensor, ...] | list[torch.Tensor],
    topk_idx: torch.Tensor,  # [num_kv_heads, total_q, topk]
    block_table: torch.Tensor,  # [num_reqs, max_blocks]
    seq_lens: torch.Tensor,  # [num_reqs] int32
    num_kv_heads: int,
    sm_scale: float,
    output: torch.Tensor,  # [total_q, num_heads, head_dim]
    decode_query_len: int,
) -> None:
    """GQA block-sparse attention for decode (split-K over the top-k blocks)."""
    k_cache, v_cache = _split_triton_main_kv_cache(kv_cache)
    total_q, num_heads, head_dim = q.shape
    assert total_q == seq_lens.shape[0] * decode_query_len
    max_topk = topk_idx.shape[-1]
    gqa_group_size = num_heads // num_kv_heads
    use_fp8 = k_cache.dtype in _FP8_DTYPES or v_cache.dtype in _FP8_DTYPES
    use_pdl = _is_arch_support_pdl()
    # `launch_pdl` is a Triton runtime kwarg only some backends accept (CUDA
    # SM9+); this ROCm Triton rejects it even when False ("Keyword argument
    # launch_pdl was specified but unrecognised"). Only pass it when PDL is
    # actually supported -- on ROCm use_pdl is always False, so it's omitted.
    pdl_launch = {"launch_pdl": True} if use_pdl else {}
    # split-K over the selected blocks; keep enough programs for small decode
    # batches without regressing long-context sparse decode.
    TARGET_GRID = 64
    MAX_TOPK_CHUNKS = 4
    target = max(
        1,
        min(
            max_topk,
            MAX_TOPK_CHUNKS,
            TARGET_GRID // max(1, total_q * num_kv_heads),
        ),
    )
    num_topk_chunks = 1 << (target.bit_length() - 1)
    o_partial = torch.empty(
        num_topk_chunks,
        total_q,
        num_heads,
        head_dim,
        dtype=q.dtype,
        device=q.device,
    )
    lse_partial = torch.empty(
        num_topk_chunks, total_q, num_heads, dtype=torch.float32, device=q.device
    )
    grid = (total_q * num_topk_chunks, num_kv_heads)
    _gqa_sparse_decode_kernel[grid](
        q,
        k_cache,
        v_cache,
        topk_idx,
        o_partial,
        lse_partial,
        block_table,
        seq_lens,
        block_table.shape[-1],
        total_q,
        gqa_group_size,
        head_dim,
        max_topk,
        sm_scale,
        decode_query_len,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k_cache.stride(0),
        k_cache.stride(1),
        k_cache.stride(2),
        k_cache.stride(3),
        v_cache.stride(0),
        v_cache.stride(1),
        v_cache.stride(2),
        v_cache.stride(3),
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
