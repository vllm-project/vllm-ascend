# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Standalone MiniMax M3 prefill index top-k kernel + wrapper.

Extracted from vllm/models/minimax_m3/common/ops/index_topk.py — only
``minimax_m3_index_topk`` and its direct dependencies.

NPU-adapted: the bitonic top-k kernel is replaced with a chunk-wise
select/merge top-k (one ``_select_topk_pairs`` per chunk, then pairwise
``_merge_topk_pairs`` reductions), which avoids the reshape/bitcast ops the
Ascend BiShengIR backend does not support. PDL (CUDA SM9+) is dropped since
this build runs on NPU.
"""

import torch
from vllm.triton_utils import tl, triton

SPARSE_BLOCK_SIZE = 128

TOPK_SELECTION_TILE = 128
TOPK_COMPUTE_MIN_TILE = 16
TOPK_NUM_WARPS = 4
TOPK_NUM_STAGES = 2


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

        selected_i32 = (off_k == best_offset).to(tl.int32) * work_valid_i32
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
def _select_topk_indices_direct(
    scores,
    valid_i32,
    valid_count,
    base_offset,
    topk_size: tl.constexpr,
    block_size: tl.constexpr,
):
    """Select top-k indices directly from scores where index = base_offset + pos + 1.

    valid_i32 and valid_count are pre-computed by the caller to avoid
    redundant mask conversion and reduction inside this function.
    Returns only indices.
    """
    off_k = tl.arange(0, block_size)
    off_t = tl.arange(0, topk_size)
    idx_ramp = off_k.to(tl.float32)

    work_scores = tl.where(valid_i32 != 0, scores, float("-inf"))
    selected_indices = tl.full((topk_size,), 0, dtype=tl.int32)

    for rank in tl.static_range(0, topk_size):
        # Replace tl.argmax with tl.max + tl.sum(tl.where(...))
        # to decouple the max-value and max-index reductions.
        # tl.argmax couples scalar/index tracking with vector reduction,
        # creating pipeline bubbles (38% idle per profiling).
        # tl.max is a pure vector reduction; tl.sum over the one-hot
        # mask gives the index.
        max_val = tl.max(work_scores, axis=0)
        best_offset = tl.sum(tl.where(work_scores == max_val, idx_ramp, 0.0), axis=0).to(tl.int32)

        # Direct index computation — no reduction needed.
        best_index = base_offset + best_offset + 1
        valid_rank = (rank < valid_count).to(tl.int32)

        selected_indices = tl.where(
            off_t == rank,
            tl.where(valid_rank != 0, best_index, 0),
            selected_indices,
        )

        # Inline the single-position mask: avoid intermediate variable
        # to reduce register pressure in the hot loop.
        work_scores = tl.where(
            off_k == best_offset,
            float("-inf"),
            work_scores,
        )

    return selected_indices


@triton.jit
def _select_topk_pairs_direct(
    scores,
    valid_i32,
    valid_count,
    base_offset,
    topk_size: tl.constexpr,
    block_size: tl.constexpr,
):
    """Select top-k scores+indices where index = base_offset + pos + 1.

    Like _select_topk_indices_direct but also returns scores (needed
    by the partial-kernel merge path). valid_i32 and valid_count are
    pre-computed by the caller.
    """
    off_k = tl.arange(0, block_size)
    off_t = tl.arange(0, topk_size)
    idx_ramp = off_k.to(tl.float32)

    work_scores = tl.where(valid_i32 != 0, scores, float("-inf"))
    selected_scores = tl.full((topk_size,), -1e30, dtype=tl.float32)
    selected_indices = tl.full((topk_size,), 0, dtype=tl.int32)

    for rank in tl.static_range(0, topk_size):
        # Replace tl.argmax+tl.max with tl.max+tl.sum(tl.where(...)).
        # The single tl.max gives both the value (reuse as best_score)
        # and the mask (use with sum for index).
        best_score = tl.max(work_scores, axis=0)
        best_offset = tl.sum(tl.where(work_scores == best_score, idx_ramp, 0.0), axis=0).to(tl.int32)

        # Direct index computation — no reduction needed.
        best_index = base_offset + best_offset + 1
        valid_rank = (rank < valid_count).to(tl.int32)

        selected_scores = tl.where(
            off_t == rank,
            tl.where(valid_rank != 0, best_score, -1e30),
            selected_scores,
        )
        selected_indices = tl.where(
            off_t == rank,
            tl.where(valid_rank != 0, best_index, 0),
            selected_indices,
        )

        work_scores = tl.where(
            off_k == best_offset,
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

        left_selected_i32 = (off_t == left_offset).to(tl.int32) * left_valid_i32
        right_selected_i32 = (off_t == right_offset).to(tl.int32) * right_valid_i32
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
# Prefill top-k: one program per (query, head, chunk) selects the chunk's
# top-k; a tree of pairwise merges reduces the chunks to a single top-k.
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
    valid_i32 = ((block_ids < valid_blocks) & (off_k < chunk_blocks)).to(tl.int32)

    score = tl.load(
        s_ptr + pid_h * stride_s_h + query_idx * stride_s_n + block_ids * stride_s_k,
        mask=valid_i32 != 0,
        other=-1e30,
    ).to(tl.float32)
    score = tl.where(score != score, -1e30, score)

    init_i32 = (block_ids < init_blocks).to(tl.int32)
    local_i32 = (block_ids >= tl.maximum(0, valid_blocks - local_blocks)).to(tl.int32)
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

    valid_count = tl.sum(valid_i32, axis=0)

    selected_scores, selected_indices = _select_topk_pairs_direct(
        score,
        valid_i32,
        valid_count,
        chunk_start,
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


@triton.jit(do_not_specialize_on_alignment=["prefix_lens"])
def _prefill_topk_fused_kernel(
    s_ptr,
    topk_idx_ptr,
    cup_seqlens,
    prefix_lens,
    init_blocks: tl.constexpr,
    local_blocks: tl.constexpr,
    chunk_blocks,
    num_chunks,
    topk: tl.constexpr,
    stride_s_h,
    stride_s_n,
    stride_s_k,
    stride_o_h,
    stride_o_n,
    stride_o_t,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_T: tl.constexpr,
    BLOCK_SIZE_BLOCK: tl.constexpr,
    MASK_INIT: tl.constexpr,
    MASK_LOCAL: tl.constexpr,
):
    """Fused prefill top-k + finalize: write final indices directly.

    Identical to _prefill_topk_partial_kernel except that instead of
    storing intermediate (scores, indices) to partial buffers, it writes
    the final topk_idx output directly (indices - 1 for valid, -1 otherwise).
    Only used when num_chunks == 1 (no merge needed).
    """
    pid_q = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_hc = tl.program_id(2)
    pid_h = pid_hc // num_chunks
    pid_chunk = pid_hc - pid_h * num_chunks

    seq_start = tl.load(cup_seqlens + pid_b)
    q_len = tl.load(cup_seqlens + pid_b + 1) - seq_start
    prefix_len = tl.load(prefix_lens + pid_b)
    if pid_q >= q_len:
        return

    off_k = tl.arange(0, BLOCK_SIZE_K)
    off_t = tl.arange(0, BLOCK_SIZE_T)
    query_idx = seq_start + pid_q
    valid_blocks = (prefix_len + pid_q + BLOCK_SIZE_BLOCK) // BLOCK_SIZE_BLOCK
    chunk_start = pid_chunk * chunk_blocks
    block_ids = chunk_start + off_k
    valid_i32 = ((block_ids < valid_blocks) & (off_k < chunk_blocks)).to(tl.int32)

    score = tl.load(
        s_ptr + pid_h * stride_s_h + query_idx * stride_s_n + block_ids * stride_s_k,
        mask=valid_i32 != 0,
        other=-1e30,
    ).to(tl.float32)
    score = tl.where(score != score, -1e30, score)

    init_i32 = (block_ids < init_blocks).to(tl.int32)
    local_i32 = (block_ids >= tl.maximum(0, valid_blocks - local_blocks)).to(tl.int32)
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

    # Hoist valid_count to kernel level: valid_i32 is already computed,
    # so tl.sum here avoids a redundant vreduce<sum> inside the select
    # function. Passing valid_i32 directly also skips a bool↔int32 round-trip.
    valid_count = tl.sum(valid_i32, axis=0)

    selected_indices = _select_topk_indices_direct(
        score,
        valid_i32,
        valid_count,
        chunk_start,
        BLOCK_SIZE_T,
        BLOCK_SIZE_K,
    )

    # Write final output directly: valid index -> index-1, else -1
    output = tl.where(
        (off_t < topk) & (selected_indices > 0),
        selected_indices - 1,
        -1,
    )
    tl.store(
        topk_idx_ptr + pid_h * stride_o_h + query_idx * stride_o_n + off_t * stride_o_t,
        output.to(topk_idx_ptr.dtype.element_ty),
        mask=off_t < topk,
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
        in_scores_ptr + left_chunk * stride_is_c + pid_h * stride_is_h + pid_n * stride_is_n + off_t * stride_is_t,
    ).to(tl.float32)
    left_indices = tl.load(
        in_indices_ptr + left_chunk * stride_ii_c + pid_h * stride_ii_h + pid_n * stride_ii_n + off_t * stride_ii_t,
    ).to(tl.int32)
    right_scores = tl.load(
        in_scores_ptr + right_chunk * stride_is_c + pid_h * stride_is_h + pid_n * stride_is_n + off_t * stride_is_t,
        mask=right_exists_i32 != 0,
        other=-1e30,
    ).to(tl.float32)
    right_indices = tl.load(
        in_indices_ptr + right_chunk * stride_ii_c + pid_h * stride_ii_h + pid_n * stride_ii_n + off_t * stride_ii_t,
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
        out_scores_ptr + pid_out_chunk * stride_os_c + pid_h * stride_os_h + pid_n * stride_os_n + off_t * stride_os_t,
        merged_scores,
    )
    tl.store(
        out_indices_ptr + pid_out_chunk * stride_oi_c + pid_h * stride_oi_h + pid_n * stride_oi_n + off_t * stride_oi_t,
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
        indices_partial_ptr + pid_h * stride_pi_h + pid_n * stride_pi_n + off_t * stride_pi_t,
    ).to(tl.int32)
    output = tl.where(
        (off_t < topk) & (indices > 0),
        indices - 1,
        -1,
    )
    tl.store(
        indices_final_ptr + pid_h * stride_f_h + pid_n * stride_f_n + off_t * stride_f_t,
        output.to(indices_final_ptr.dtype.element_ty),
        mask=off_t < topk,
    )


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
def minimax_m3_index_topk(
    score: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    prefix_lens: torch.Tensor,
    max_query_len: int,
    topk: int,
    init_blocks: int,
    local_blocks: int,
) -> torch.Tensor:
    """Select index top-k from a precomputed score tensor."""
    assert topk > 0

    num_heads, total_q, score_block_stride = score.shape
    batch = cu_seqlens_q.shape[0] - 1
    topk_width = _topk_compute_width(topk)
    select_width = _topk_select_width(topk)
    num_chunks = max(1, triton.cdiv(score_block_stride, select_width))
    chunk_blocks = triton.cdiv(score_block_stride, num_chunks)

    topk_idx = torch.empty(
        (num_heads, total_q, topk),
        dtype=torch.int32,
        device=score.device,
    )

    if num_chunks == 1:
        # Fused fast path: skip partial buffers, merge, and finalize kernel.
        _prefill_topk_fused_kernel[(max_query_len, batch, num_heads)](
            score,
            topk_idx,
            cu_seqlens_q,
            prefix_lens,
            init_blocks,
            local_blocks,
            chunk_blocks,
            num_chunks,
            topk,
            score.stride(0),
            score.stride(1),
            score.stride(2),
            topk_idx.stride(0),
            topk_idx.stride(1),
            topk_idx.stride(2),
            BLOCK_SIZE_K=select_width,
            BLOCK_SIZE_T=topk_width,
            BLOCK_SIZE_BLOCK=SPARSE_BLOCK_SIZE,
            MASK_INIT=False,
            MASK_LOCAL=False,
            num_warps=TOPK_NUM_WARPS,
            num_stages=TOPK_NUM_STAGES,
        )
    else:
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

        _prefill_topk_partial_kernel[(max_query_len, batch, num_heads * num_chunks)](
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
