# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MiniMax M3 lightning-indexer block scoring and top-k selection.

The prefill path computes one score per visible 128-token KV block, applies
forced init/local-block priorities, and selects block IDs with ``torch.topk``.
The score-finalization and invalid-index-mask kernels tile consecutive queries
so their launch grids remain small on Ascend NPUs.

The decode path keeps the existing split-K scoring and post-top-k masking flow.
One sparse block maps to one KV-cache page.
"""

import os
from collections.abc import Sequence

import torch

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils.math_utils import round_up

# Data-layout constants.
SPARSE_BLOCK_SIZE = 128
SCORE_BLOCK_STRIDE_ALIGNMENT = 16

# Prefill score-kernel configuration.
PREFILL_SCORE_QUERY_TILE_SIZE = 64

# Prefill score-finalization launch policy.
PREFILL_FINALIZE_QUERY_TILE_CANDIDATES = (8, 16, 32, 64)
PREFILL_FINALIZE_TARGET_PROGRAM_COUNT = 128
PREFILL_FINALIZE_TAIL_TILE_SIZE = 16

# Prefill invalid-index-mask launch policy.
PREFILL_INVALID_MASK_QUERY_TILE_CANDIDATES = (1, 2, 4, 8, 16, 32, 64)
PREFILL_INVALID_MASK_TARGET_PROGRAM_COUNT = 128
PREFILL_INVALID_MASK_MAX_TILE_ELEMENTS = 2048

# Decode score autotuning and auxiliary-kernel launch limits.
DECODE_SCORE_MAX_PROGRAM_COUNT = 512
DECODE_SCORE_CHUNK_CANDIDATES = (1, 2, 4, 8, 16, 32, 64, 128, 256)
DECODE_SCORE_STAGE_CANDIDATES = (1, 2)
DECODE_MASK_TARGET_PROGRAM_COUNT = 64
DECODE_MASK_MAX_CHUNK_COUNT = 16
DECODE_TAIL_FILL_TARGET_PROGRAM_COUNT = 64
DECODE_TAIL_FILL_MAX_CHUNK_COUNT = 16
DECODE_MASK_BLOCK_TILE_SIZE = 2048
DECODE_TAIL_FILL_BLOCK_TILE_SIZE = 2048

# Environment-variable names used only for explicit tuning/debug overrides.
ENV_PREFILL_FINALIZE_QUERY_TILE = (
    "VLLM_MINIMAX_M3_PREFILL_FINALIZE_BLOCK_Q"
)
ENV_PREFILL_FINALIZE_TARGET_PROGRAMS = (
    "VLLM_MINIMAX_M3_PREFILL_FINALIZE_TARGET_PROGRAMS"
)
ENV_PREFILL_INVALID_MASK_QUERY_TILE = "VLLM_MINIMAX_M3_PREFILL_INVALID_BLOCK_Q"
ENV_PREFILL_INVALID_MASK_TARGET_PROGRAMS = (
    "VLLM_MINIMAX_M3_PREFILL_INVALID_TARGET_PROGRAMS"
)
ENV_PREFILL_INVALID_MASK_MAX_TILE_ELEMENTS = (
    "VLLM_MINIMAX_M3_PREFILL_INVALID_MAX_TILE_ELEMENTS"
)


DECODE_SCORE_AUTOTUNE_CONFIGS = [
    triton.Config({"num_kv_chunks": chunk_count}, num_stages=stage_count)
    for chunk_count in DECODE_SCORE_CHUNK_CANDIDATES
    for stage_count in DECODE_SCORE_STAGE_CANDIDATES
]


def _as_triton_index_kv_cache(
    index_kv_cache: torch.Tensor | tuple[torch.Tensor, ...] | list[torch.Tensor],
) -> torch.Tensor:
    """Normalize Ascend indexer cache to [num_blocks, 128, head_dim]."""
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


def _read_positive_int_env(name: str, default: int) -> int:
    """Returns a positive integer environment override."""
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        parsed_value = int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if parsed_value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return parsed_value


def _read_int_choice_env(
    name: str,
    choices: Sequence[int],
) -> int | None:
    """Returns an integer environment override restricted to ``choices``."""
    value = os.environ.get(name)
    if value is None:
        return None
    try:
        parsed_value = int(value)
    except ValueError as exc:
        allowed_values = ", ".join(str(choice) for choice in choices)
        raise ValueError(f"{name} must be one of {allowed_values}") from exc
    if parsed_value not in choices:
        allowed_values = ", ".join(str(choice) for choice in choices)
        raise ValueError(f"{name} must be one of {allowed_values}")
    return parsed_value


def _prune_decode_score_autotune_configs(configs, named_args, **unused_kwargs):
    """Limits decode split-K variants so the launch grid stays bounded."""
    request_count = max(1, named_args["num_reqs"])
    max_chunk_count = max(1, DECODE_SCORE_MAX_PROGRAM_COUNT // request_count)
    max_chunk_count = 1 << (max_chunk_count.bit_length() - 1)
    pruned_configs = [
        config
        for config in configs
        if config.kwargs["num_kv_chunks"] <= max_chunk_count
    ]
    return pruned_configs or configs[:1]


# ---------------------------------------------------------------------------
# Index block-score kernel (paged). score[h, token, block] = max over the
# 128-token block of (idx_q . index_k), causal-masked. BLOCK_SIZE_K == 128 so
# each K-tile is exactly one page (one K tile == one sparse block).
# ---------------------------------------------------------------------------
# Scalar metadata loads do not require pointer-alignment specialization.
@triton.jit(
    do_not_specialize_on_alignment=[
        "sequence_lengths_ptr",
        "prefix_lengths_ptr",
    ]
)
def _prefill_index_block_score_kernel(
    query_ptr,
    index_key_cache_ptr,
    score_ptr,
    block_table_ptr,
    query_start_offsets_ptr,
    sequence_lengths_ptr,
    prefix_lengths_ptr,
    index_head_count: tl.constexpr,
    head_dim: tl.constexpr,
    query_token_stride,
    query_head_stride,
    query_dim_stride,
    key_block_stride,
    key_position_stride,
    key_dim_stride,
    score_head_stride,
    score_token_stride,
    score_block_stride,
    block_table_batch_stride,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """Computes one max dot-product score per visible sparse block."""

    tl.static_assert(
        BLOCK_SIZE_Q <= BLOCK_SIZE_K,
        "BLOCK_SIZE_Q must not exceed BLOCK_SIZE_K",
    )

    query_tile_id = tl.program_id(0)
    batch_head_id = tl.program_id(1)

    batch_id = batch_head_id // index_head_count
    head_id = batch_head_id % index_head_count

    sequence_start = tl.load(
        query_start_offsets_ptr + batch_id
    )
    sequence_end = tl.load(
        query_start_offsets_ptr + batch_id + 1
    )
    query_length = sequence_end - sequence_start

    sequence_length = tl.load(
        sequence_lengths_ptr + batch_id
    )
    prefix_length = tl.load(
        prefix_lengths_ptr + batch_id
    )

    query_tile_start = query_tile_id * BLOCK_SIZE_Q
    if query_tile_start >= query_length:
        return

    query_lane_offsets = tl.arange(0, BLOCK_SIZE_Q)
    key_lane_offsets = tl.arange(0, BLOCK_SIZE_K)
    dim_offsets = tl.arange(0, head_dim)

    query_offsets = query_tile_start + query_lane_offsets
    query_mask = query_offsets < query_length
    query_positions = prefix_length + query_offsets

    query = tl.load(
        query_ptr
        + (
            sequence_start + query_offsets[:, None]
        ) * query_token_stride
        + head_id * query_head_stride
        + dim_offsets[None, :] * query_dim_stride,
        mask=query_mask[:, None].broadcast_to(
            (BLOCK_SIZE_Q, head_dim)
        ),
        other=0.0,
    )

    block_table_row_ptr = (
        block_table_ptr
        + batch_id * block_table_batch_stride
    )

    score_row_ptrs = (
        score_ptr
        + head_id * score_head_stride
        + (
            sequence_start + query_offsets
        ) * score_token_stride
    )

    # Exclusive upper bound of key positions visible to at least one valid
    # query in this query tile.
    query_tile_valid_end = tl.minimum(
        query_length,
        query_tile_start + BLOCK_SIZE_Q,
    )
    visible_key_end = tl.minimum(
        sequence_length,
        prefix_length + query_tile_valid_end,
    )

    # A block is fully visible to every query in the tile when its final
    # position is no greater than the earliest query position.
    earliest_query_position = (
        prefix_length + query_tile_start
    )
    causally_full_block_count = (
        earliest_query_position + 1
    ) // BLOCK_SIZE_K

    # Only complete sequence blocks may use an unmasked key load.
    complete_sequence_block_count = (
        sequence_length // BLOCK_SIZE_K
    )

    full_block_count = tl.minimum(
        causally_full_block_count,
        complete_sequence_block_count,
    )

    key_position_offsets = (
        key_lane_offsets[None, :]
        * key_position_stride
    )
    key_dim_offsets = (
        dim_offsets[:, None]
        * key_dim_stride
    )

    # Fully visible historical blocks.
    #
    # These blocks require neither a key-position mask nor a causal mask.
    for block_id in tl.range(0, full_block_count):
        page_id = tl.load(
            block_table_row_ptr + block_id
        ).to(tl.int64)

        key = tl.load(
            index_key_cache_ptr
            + page_id * key_block_stride
            + key_position_offsets
            + key_dim_offsets,
        )

        query_key = tl.dot(query, key)
        block_score = tl.max(query_key, axis=1)

        tl.store(
            score_row_ptrs
            + block_id * score_block_stride,
            block_score,
            mask=query_mask,
        )

    # Blocks intersecting the query tile's causal boundary or the sequence
    # boundary.
    #
    # When prefix_length is block aligned, this normally processes one block.
    # Keeping it as a loop also correctly handles an unaligned prefix, where
    # the query tile can overlap two sparse blocks.
    boundary_key_start = (
        full_block_count * BLOCK_SIZE_K
    )

    for key_block_start in tl.range(
        boundary_key_start,
        visible_key_end,
        BLOCK_SIZE_K,
    ):
        block_id = key_block_start // BLOCK_SIZE_K

        page_id = tl.load(
            block_table_row_ptr + block_id
        ).to(tl.int64)

        key_positions = (
            key_block_start + key_lane_offsets
        )
        key_mask = key_positions < sequence_length

        key = tl.load(
            index_key_cache_ptr
            + page_id * key_block_stride
            + key_position_offsets
            + key_dim_offsets,
            mask=key_mask[None, :].broadcast_to(
                (head_dim, BLOCK_SIZE_K)
            ),
            other=0.0,
        )

        query_key = tl.dot(query, key)

        causal_mask = (
            query_positions[:, None]
            >= key_positions[None, :]
        )
        query_key = tl.where(
            causal_mask & key_mask[None, :],
            query_key,
            float("-inf"),
        )

        block_score = tl.max(
            query_key,
            axis=1,
        )

        tl.store(
            score_row_ptrs
            + block_id * score_block_stride,
            block_score,
            mask=query_mask,
        )

# ---------------------------------------------------------------------------
# Decode index-score kernel (split-K over seq blocks). Decode batches are
# flattened request-major, with a runtime query length used to map each query
# token back to its request metadata. Chunk counts depend only on shape
# constants so the grid is fixed within a captured graph. The score scale is omitted
# because decode only consumes block ordering.
# ---------------------------------------------------------------------------
@triton.autotune(
    configs=DECODE_SCORE_AUTOTUNE_CONFIGS,
    key=["num_idx_heads", "BLOCK_SIZE_Q", "head_dim", "num_reqs"],
    prune_configs_by={"early_config_prune": _prune_decode_score_autotune_configs},
)
@triton.jit(do_not_specialize=["decode_query_len"])
def _decode_index_score_kernel(
    q_ptr,  # idx_q: [total_q, num_idx_heads, head_dim]
    ik_cache_ptr,  # index-K cache: [num_blocks, 128, head_dim]
    score_ptr,  # [num_idx_heads, total_q, max_block]
    init_mask_ptr,  # [total_q, score_block_stride] bool
    local_mask_ptr,  # [total_q, score_block_stride] bool
    block_table_ptr,  # [num_reqs, max_blocks]
    seq_lens,  # [num_reqs]
    num_idx_heads: tl.constexpr,
    head_dim: tl.constexpr,
    num_reqs: tl.constexpr,
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
    stride_mask_q,
    stride_mask_k,
    stride_bt_b,
    BLOCK_SIZE_K: tl.constexpr,  # == SPARSE_BLOCK_SIZE (128)
    BLOCK_SIZE_Q: tl.constexpr,
    num_kv_chunks,
    USE_PDL: tl.constexpr,
):
    BLOCK_SIZE_HQ: tl.constexpr = num_idx_heads * BLOCK_SIZE_Q
    pid_r = tl.program_id(0)
    pid_c = tl.program_id(1)
    hq_offsets = tl.arange(0, BLOCK_SIZE_HQ)
    h_offsets = hq_offsets // BLOCK_SIZE_Q
    q_offsets = hq_offsets % BLOCK_SIZE_Q
    q_mask = q_offsets < decode_query_len
    q_ids = pid_r * decode_query_len + q_offsets

    if USE_PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()

    seq_len = tl.load(seq_lens + pid_r)
    query_pos = seq_len - decode_query_len + q_offsets
    # Full-CG padding uses zero-length request rows. Clamp to an empty
    # attention range instead of letting padded rows produce negative lengths.
    kv_len = tl.maximum(query_pos + 1, 0)
    kv_len_max = tl.max(tl.where(q_mask, kv_len, 0), axis=0)
    num_blocks = (kv_len_max + BLOCK_SIZE_K - 1) // BLOCK_SIZE_K

    # block-aligned fixed-count split: grid independent of seq_len (captured graph).
    chunk_size_blocks = (num_blocks + num_kv_chunks - 1) // num_kv_chunks
    chunk_start_block = pid_c * chunk_size_blocks
    chunk_end_block = tl.minimum(chunk_start_block + chunk_size_blocks, num_blocks)
    if chunk_start_block >= chunk_end_block:
        return
    off_k = tl.arange(0, BLOCK_SIZE_K)  # positions within a 128-block
    off_d = tl.arange(0, head_dim)
    bt_row = block_table_ptr + pid_r * stride_bt_b
    # Query vectors for all index heads in a small spec-decode block.
    q = tl.load(
        q_ptr
        + q_ids[:, None] * stride_q_n
        + h_offsets[:, None] * stride_q_h
        + off_d[None, :] * stride_q_d,
        mask=q_mask[:, None],
        other=0.0,
    )  # [HQ,D]
    for blk in tl.range(chunk_start_block, chunk_end_block):
        page = tl.load(bt_row + blk).to(tl.int64)
        pos = blk * BLOCK_SIZE_K + off_k
        pos_mask = pos[None, :] < kv_len[:, None]
        # index-K for this page: [D,N] (transposed), same layout as prefill.
        k = tl.load(
            ik_cache_ptr
            + page * stride_ik_blk
            + off_k[None, :] * stride_ik_pos
            + off_d[:, None] * stride_ik_d,
        )  # [D,N]
        # fp32 accumulation is required for the fp8 (e4m3) index cache: q/k are
        # loaded in their stored dtype (bf16 or e4m3) and the MMA accumulates in
        # fp32 so the per-block max score is exact for the fp8 indexer too.
        qk = tl.dot(q, k, out_dtype=tl.float32)  # [HQ,N]
        qk = tl.where(pos_mask & q_mask[:, None], qk, float("-inf"))
        score = tl.max(qk, axis=1)  # [HQ]
        mask_off = q_ids * stride_mask_q + blk * stride_mask_k
        is_init = tl.load(init_mask_ptr + mask_off)
        is_local = tl.load(local_mask_ptr + mask_off)
        score = tl.where(is_local, 1e29, tl.where(is_init, 1e30, score))
        tl.store(
            score_ptr
            + h_offsets * stride_s_h
            + q_ids * stride_s_n
            + blk * stride_s_k,
            score,
            mask=q_mask,
        )


# ---------------------------------------------------------------------------
# Pad unwritten score tail with -inf so torch.topk ignores [num_blocks, max_block).
# _decode_index_score_kernel only writes [0, row_num_blocks); torch.empty leaves
# the rest as garbage. Per-token num_blocks matches the top-k invalid mask.
# Split-K over max_block with a shape-constant chunk count (captured graph-safe).
# ---------------------------------------------------------------------------
@triton.jit(do_not_specialize=["decode_query_len", "max_block", "chunk_blocks"])
def _decode_score_tail_fill_kernel(
    score_ptr,  # [num_idx_heads, total_q, score_block_stride] fp32
    seq_lens,  # [num_reqs]
    block_size: tl.constexpr,  # sparse block size (128)
    max_block,
    decode_query_len,
    chunk_blocks,  # max_block split count per chunk (shape-constant)
    stride_s_h,
    stride_s_b,
    stride_s_k,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_b = tl.program_id(0)  # flattened query-token id
    pid_h = tl.program_id(1)
    pid_chunk = tl.program_id(2)
    req_id = pid_b // decode_query_len
    q_offset = pid_b - req_id * decode_query_len

    seq_len = tl.load(seq_lens + req_id)
    query_pos = seq_len - decode_query_len + q_offset
    kv_len = tl.maximum(query_pos + 1, 0)
    num_blocks = (kv_len + block_size - 1) // block_size

    chunk_start = pid_chunk * chunk_blocks
    chunk_end = tl.minimum(chunk_start + chunk_blocks, max_block)
    fill_start = tl.maximum(chunk_start, num_blocks)
    if fill_start >= chunk_end:
        return

    num_to_fill = chunk_end - fill_start
    off_k = tl.arange(0, BLOCK_SIZE_K)
    for i in tl.range(0, num_to_fill, BLOCK_SIZE_K):
        blk = fill_start + i + off_k
        store_mask = (i + off_k) < num_to_fill
        s_ptrs = (
            score_ptr
            + pid_h * stride_s_h
            + pid_b * stride_s_b
            + blk * stride_s_k
        )
        tl.store(s_ptrs, float("-inf"), mask=store_mask)


# ---------------------------------------------------------------------------
# Decode top-k: torch.topk on scores, then mask invalid block ids per token.
# Forced init/local blocks are already encoded in the scores.
# ---------------------------------------------------------------------------
@triton.heuristics({"BLOCK_SIZE_T": lambda args: triton.next_power_of_2(args["topk"])})
@triton.jit(do_not_specialize=["decode_query_len"])
def _decode_topk_invalid_index_mask_kernel(
    ti_ptr,  # [num_idx_heads, total_q, topk] int32 in/out
    seq_lens,  # [num_reqs]
    block_size: tl.constexpr,  # sparse block size (128)
    topk: tl.constexpr,
    decode_query_len,
    stride_ti_h,
    stride_ti_b,
    stride_ti_t,
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
    ti_ptrs = (
        ti_ptr + pid_h * stride_ti_h + pid_b * stride_ti_b + off_t * stride_ti_t
    )
    store_mask = off_t < topk
    idx = tl.load(ti_ptrs, mask=store_mask, other=0)
    valid_slot = off_t < tl.minimum(topk, num_blocks)
    valid_idx = (idx >= 0) & (idx < num_blocks)
    masked_idx = tl.where(valid_slot & valid_idx, idx, -1)
    tl.store(ti_ptrs, masked_idx.to(ti_ptr.dtype.element_ty), mask=store_mask)


# ---------------------------------------------------------------------------
# Prefill score finalization before torch.topk.
#
# A program handles one query tile and one index head. BLOCK_SIZE_Q is selected
# on the host from {8, 16, 32, 64} so the number of programs stays near a
# configurable target. Since BLOCK_SIZE_Q <= 64 and one sparse block contains
# 128 tokens, a tile can contain at most two distinct valid-block counts.
#
# Init blocks use one regular [Q, K] store. Local blocks are split into the two
# possible valid-block groups, so every store still has a scalar block base and
# regular vector addresses. The invalid tail is also written with a common
# column vector and a row mask; no per-query dynamic scatter is used.
# ---------------------------------------------------------------------------
@triton.jit(do_not_specialize=["score_block_count"])
def _prefill_score_finalize_kernel(
    score_ptr,
    query_start_offsets_ptr,
    prefix_lengths_ptr,
    index_head_count,
    init_block_count: tl.constexpr,
    local_block_count: tl.constexpr,
    score_block_count,
    score_head_stride,
    score_token_stride,
    score_block_stride,
    sparse_block_size: tl.constexpr,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_FORCE: tl.constexpr,
    BLOCK_SIZE_TAIL: tl.constexpr,
):
    """Applies init/local priorities and fills the invalid score tail."""
    tl.static_assert(BLOCK_SIZE_Q <= sparse_block_size)

    query_tile_id = tl.program_id(0)
    batch_head_id = tl.program_id(1)
    batch_id = batch_head_id // index_head_count
    head_id = batch_head_id % index_head_count

    sequence_start = tl.load(query_start_offsets_ptr + batch_id)
    sequence_end = tl.load(query_start_offsets_ptr + batch_id + 1)
    query_length = sequence_end - sequence_start
    query_tile_start = query_tile_id * BLOCK_SIZE_Q
    if query_tile_start >= query_length:
        return

    query_lane_offsets = tl.arange(0, BLOCK_SIZE_Q)
    query_offsets = query_tile_start + query_lane_offsets
    query_mask = query_offsets < query_length
    token_indices = sequence_start + query_offsets
    prefix_length = tl.load(prefix_lengths_ptr + batch_id)

    valid_block_counts = (
        prefix_length + query_offsets + sparse_block_size
    ) // sparse_block_size
    valid_block_counts = tl.minimum(
        valid_block_counts,
        score_block_count,
    )

    # Because BLOCK_SIZE_Q <= sparse_block_size, one query tile contains at
    # most two consecutive valid-block counts.
    min_valid_block_count = (
        prefix_length + query_tile_start + sparse_block_size
    ) // sparse_block_size
    min_valid_block_count = tl.minimum(
        min_valid_block_count,
        score_block_count,
    )
    max_valid_block_count = tl.minimum(
        min_valid_block_count + 1,
        score_block_count,
    )

    score_row_ptrs = (
        score_ptr
        + head_id * score_head_stride
        + token_indices[:, None] * score_token_stride
    )
    forced_block_offsets = tl.arange(0, BLOCK_SIZE_FORCE)

    if init_block_count > 0:
        init_mask = (
            query_mask[:, None]
            & (forced_block_offsets[None, :] < init_block_count)
            & (
                forced_block_offsets[None, :]
                < valid_block_counts[:, None]
            )
        )
        tl.store(
            score_row_ptrs
            + forced_block_offsets[None, :] * score_block_stride,
            1e30,
            mask=init_mask,
        )

    rows_with_min_count = query_mask & (
        valid_block_counts == min_valid_block_count
    )
    rows_with_max_count = query_mask & (
        valid_block_counts > min_valid_block_count
    )

    if local_block_count > 0:
        min_local_start = tl.maximum(
            0,
            min_valid_block_count - local_block_count,
        )
        min_local_count = min_valid_block_count - min_local_start
        min_local_blocks = min_local_start + forced_block_offsets
        tl.store(
            score_row_ptrs
            + min_local_blocks[None, :] * score_block_stride,
            1e29,
            mask=(
                rows_with_min_count[:, None]
                & (forced_block_offsets[None, :] < min_local_count)
            ),
        )

        max_local_start = tl.maximum(
            0,
            max_valid_block_count - local_block_count,
        )
        max_local_count = max_valid_block_count - max_local_start
        max_local_blocks = max_local_start + forced_block_offsets
        tl.store(
            score_row_ptrs
            + max_local_blocks[None, :] * score_block_stride,
            1e29,
            mask=(
                rows_with_max_count[:, None]
                & (forced_block_offsets[None, :] < max_local_count)
            ),
        )

    tail_lane_offsets = tl.arange(0, BLOCK_SIZE_TAIL)
    tail_block_count = score_block_count - min_valid_block_count
    for tail_offset in tl.range(
        0,
        tail_block_count,
        BLOCK_SIZE_TAIL,
    ):
        block_ids = (
            min_valid_block_count + tail_offset + tail_lane_offsets
        )
        block_mask = block_ids < score_block_count
        row_mask = (
            rows_with_min_count[:, None]
            | (
                rows_with_max_count[:, None]
                & (block_ids[None, :] >= max_valid_block_count)
            )
        )
        tl.store(
            score_row_ptrs + block_ids[None, :] * score_block_stride,
            float("-inf"),
            mask=row_mask & block_mask[None, :],
        )


def _choose_prefill_finalize_query_tile_size(
    max_query_length: int,
    batch_size: int,
    index_head_count: int,
) -> int:
    """Chooses the query tile for prefill score finalization."""
    override = _read_int_choice_env(
        ENV_PREFILL_FINALIZE_QUERY_TILE,
        PREFILL_FINALIZE_QUERY_TILE_CANDIDATES,
    )
    if override is not None:
        return override

    target_program_count = _read_positive_int_env(
        ENV_PREFILL_FINALIZE_TARGET_PROGRAMS,
        PREFILL_FINALIZE_TARGET_PROGRAM_COUNT,
    )
    total_query_rows = max(
        1,
        max_query_length * batch_size * index_head_count,
    )
    required_tile_size = triton.cdiv(
        total_query_rows,
        target_program_count,
    )
    for tile_size in PREFILL_FINALIZE_QUERY_TILE_CANDIDATES:
        if required_tile_size <= tile_size:
            return tile_size
    return PREFILL_FINALIZE_QUERY_TILE_CANDIDATES[-1]


def _choose_prefill_invalid_mask_query_tile_size(
    max_query_length: int,
    batch_size: int,
    index_head_count: int,
    topk: int,
) -> int:
    """Chooses a query tile while bounding the 2-D mask tile size."""
    override = _read_int_choice_env(
        ENV_PREFILL_INVALID_MASK_QUERY_TILE,
        PREFILL_INVALID_MASK_QUERY_TILE_CANDIDATES,
    )
    if override is not None:
        return override

    target_program_count = _read_positive_int_env(
        ENV_PREFILL_INVALID_MASK_TARGET_PROGRAMS,
        PREFILL_INVALID_MASK_TARGET_PROGRAM_COUNT,
    )
    max_tile_elements = _read_positive_int_env(
        ENV_PREFILL_INVALID_MASK_MAX_TILE_ELEMENTS,
        PREFILL_INVALID_MASK_MAX_TILE_ELEMENTS,
    )
    total_query_rows = max(
        1,
        max_query_length * batch_size * index_head_count,
    )
    required_tile_size = triton.cdiv(
        total_query_rows,
        target_program_count,
    )

    desired_tile_size = PREFILL_INVALID_MASK_QUERY_TILE_CANDIDATES[-1]
    for tile_size in PREFILL_INVALID_MASK_QUERY_TILE_CANDIDATES:
        if required_tile_size <= tile_size:
            desired_tile_size = tile_size
            break

    topk_tile_size = triton.next_power_of_2(max(1, topk))
    max_query_tile_size = max(1, max_tile_elements // topk_tile_size)
    bounded_tile_size = PREFILL_INVALID_MASK_QUERY_TILE_CANDIDATES[0]
    for tile_size in PREFILL_INVALID_MASK_QUERY_TILE_CANDIDATES:
        if tile_size > max_query_tile_size:
            break
        bounded_tile_size = tile_size
    return min(desired_tile_size, bounded_tile_size)


@triton.heuristics(
    {"BLOCK_SIZE_T": lambda args: triton.next_power_of_2(args["topk"])}
)
@triton.jit(do_not_specialize_on_alignment=["prefix_lengths_ptr"])
def _prefill_topk_invalid_index_mask_kernel(
    topk_indices_ptr,
    query_start_offsets_ptr,
    prefix_lengths_ptr,
    index_head_count: tl.constexpr,
    sparse_block_size: tl.constexpr,
    topk: tl.constexpr,
    index_head_stride,
    index_token_stride,
    index_topk_stride,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_T: tl.constexpr,
):
    """Replaces invalid prefill top-k block IDs with ``-1``."""
    query_tile_id = tl.program_id(0)
    batch_head_id = tl.program_id(1)
    batch_id = batch_head_id // index_head_count
    head_id = batch_head_id % index_head_count

    sequence_start = tl.load(query_start_offsets_ptr + batch_id)
    sequence_end = tl.load(query_start_offsets_ptr + batch_id + 1)
    query_length = sequence_end - sequence_start
    query_tile_start = query_tile_id * BLOCK_SIZE_Q
    if query_tile_start >= query_length:
        return

    query_lane_offsets = tl.arange(0, BLOCK_SIZE_Q)
    topk_lane_offsets = tl.arange(0, BLOCK_SIZE_T)
    query_offsets = query_tile_start + query_lane_offsets
    query_mask = query_offsets < query_length
    token_indices = sequence_start + query_offsets
    prefix_length = tl.load(prefix_lengths_ptr + batch_id)
    valid_block_counts = (
        prefix_length + query_offsets + sparse_block_size
    ) // sparse_block_size

    index_ptrs = (
        topk_indices_ptr
        + head_id * index_head_stride
        + token_indices[:, None] * index_token_stride
        + topk_lane_offsets[None, :] * index_topk_stride
    )
    access_mask = (
        query_mask[:, None]
        & (topk_lane_offsets[None, :] < topk)
    )
    block_ids = tl.load(index_ptrs, mask=access_mask, other=0)

    valid_rank_mask = topk_lane_offsets[None, :] < tl.minimum(
        topk,
        valid_block_counts[:, None],
    )
    valid_block_mask = (
        (block_ids >= 0)
        & (block_ids < valid_block_counts[:, None])
    )
    output_block_ids = tl.where(
        valid_rank_mask & valid_block_mask,
        block_ids,
        -1,
    )
    tl.store(
        index_ptrs,
        output_block_ids.to(topk_indices_ptr.dtype.element_ty),
        mask=access_mask,
    )


# ---------------------------------------------------------------------------
# Decode init/local bool masks for index scoring. fp32 intermediates; split-K
# over max_block with shape-constant chunk count (captured graph-safe).
# ---------------------------------------------------------------------------
@triton.jit(do_not_specialize=["decode_query_len", "max_block", "chunk_blocks"])
def _decode_init_local_mask_kernel(
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
        tl.floor((kv_len + (block_size - 1) * 1.0) / (block_size * 1.0))
        - local_blocks * 1.0,
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
# Public Python API
# ---------------------------------------------------------------------------
def _copy_topk_indices(
    raw_indices: torch.Tensor,
    requested_topk: int,
    output: torch.Tensor | None,
) -> torch.Tensor:
    """Copies top-k indices into an int32 result and pads missing slots."""
    head_count, total_query_tokens, selected_count = raw_indices.shape
    if output is None and selected_count == requested_topk:
        return raw_indices.to(torch.int32)

    if output is None:
        result = torch.empty(
            (head_count, total_query_tokens, requested_topk),
            dtype=torch.int32,
            device=raw_indices.device,
        )
    else:
        result = output[:, :total_query_tokens, :requested_topk]

    if selected_count < requested_topk:
        result.fill_(-1)
    result[..., :selected_count].copy_(raw_indices)
    return result


@torch.no_grad()
def minimax_m3_index_score(
    idx_q: torch.Tensor,
    index_kv_cache: torch.Tensor | tuple[torch.Tensor, ...] | list[torch.Tensor],
    block_table: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    seq_lens: torch.Tensor,
    prefix_lens: torch.Tensor,
    max_query_len: int,
    max_seq_len: int,
    num_kv_heads: int,
    sm_scale: float | None = None,
) -> torch.Tensor:
    """Computes one block score for every visible prefill KV block.

    ``sm_scale`` is accepted for API compatibility. A positive global scale
    does not change block ordering, so this score-only path intentionally omits
    it.
    """
    index_kv_cache = _as_triton_index_kv_cache(index_kv_cache)
    total_q, num_idx_heads, head_dim = idx_q.shape
    assert num_idx_heads == num_kv_heads
    total_query_tokens, index_head_count, head_dim = idx_q.shape
    assert index_head_count == num_kv_heads, (
        "M3 requires num_idx_heads == num_kv_heads"
    )

    batch_size = cu_seqlens_q.shape[0] - 1
    max_block_count = triton.cdiv(max_seq_len, SPARSE_BLOCK_SIZE)
    score_block_stride = round_up(
        max_block_count,
        SCORE_BLOCK_STRIDE_ALIGNMENT,
    )
    score = torch.empty(
        (index_head_count, total_query_tokens, score_block_stride),
        dtype=torch.float32,
        device=idx_q.device,
    )

    score_grid = (
        triton.cdiv(max_query_len, PREFILL_SCORE_QUERY_TILE_SIZE),
        batch_size * index_head_count,
    )
    _prefill_index_block_score_kernel[score_grid](
        idx_q,
        index_kv_cache,
        score,
        block_table,
        cu_seqlens_q,
        seq_lens,
        prefix_lens,
        index_head_count,
        head_dim,
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
        BLOCK_SIZE_Q=PREFILL_SCORE_QUERY_TILE_SIZE,
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
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Finalizes prefill scores and returns zero-based block IDs."""
    assert topk > 0
    index_head_count, total_query_tokens, score_block_count = score.shape
    batch_size = cu_seqlens_q.shape[0] - 1

    force_tile_size = triton.next_power_of_2(
        max(1, init_blocks, local_blocks)
    )
    finalize_query_tile_size = _choose_prefill_finalize_query_tile_size(
        max_query_len,
        batch_size,
        index_head_count,
    )
    finalize_grid = (
        triton.cdiv(max_query_len, finalize_query_tile_size),
        batch_size * index_head_count,
    )
    _prefill_score_finalize_kernel[finalize_grid](
        score,
        cu_seqlens_q,
        prefix_lens,
        index_head_count,
        init_blocks,
        local_blocks,
        score_block_count,
        score.stride(0),
        score.stride(1),
        score.stride(2),
        sparse_block_size=SPARSE_BLOCK_SIZE,
        BLOCK_SIZE_Q=finalize_query_tile_size,
        BLOCK_SIZE_FORCE=force_tile_size,
        BLOCK_SIZE_TAIL=PREFILL_FINALIZE_TAIL_TILE_SIZE,
    )

    selected_count = min(topk, score_block_count)
    score_rows = score[:, :total_query_tokens, :score_block_count]
    raw_indices = torch.topk(
        score_rows,
        k=selected_count,
        dim=-1,
    ).indices
    topk_indices = _copy_topk_indices(raw_indices, topk, out)

    mask_query_tile_size = _choose_prefill_invalid_mask_query_tile_size(
        max_query_len,
        batch_size,
        index_head_count,
        topk,
    )
    mask_grid = (
        triton.cdiv(max_query_len, mask_query_tile_size),
        batch_size * index_head_count,
    )
    _prefill_topk_invalid_index_mask_kernel[mask_grid](
        topk_indices,
        cu_seqlens_q,
        prefix_lens,
        index_head_count,
        SPARSE_BLOCK_SIZE,
        topk,
        topk_indices.stride(0),
        topk_indices.stride(1),
        topk_indices.stride(2),
        BLOCK_SIZE_Q=mask_query_tile_size,
    )
    return topk_indices


@torch.no_grad()
def minimax_m3_index_decode(
    idx_q: torch.Tensor,
    index_kv_cache: torch.Tensor | tuple[torch.Tensor, ...] | list[torch.Tensor],
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    max_seq_len: int,
    topk: int,
    init_blocks: int,
    local_blocks: int,
    num_kv_heads: int,
    decode_query_len: int,
    max_decode_query_len: int | None = None,
    out: torch.Tensor | None = None,
    sm_scale: float | None = None,
) -> torch.Tensor:
    """Computes decode block scores and returns zero-based top-k block IDs.

    ``sm_scale`` is accepted for API compatibility and intentionally omitted
    because this score-only path consumes block ordering.
    """

    index_kv_cache = _as_triton_index_kv_cache(index_kv_cache)
    total_q, num_idx_heads, head_dim = idx_q.shape
    assert topk > 0
    total_query_tokens, index_head_count, head_dim = idx_q.shape
    assert index_head_count == num_kv_heads, (
        "M3 requires num_idx_heads == num_kv_heads"
    )

    if max_decode_query_len is None:
        max_decode_query_len = decode_query_len
    assert decode_query_len <= max_decode_query_len

    request_count = seq_lens.shape[0]
    assert total_query_tokens == request_count * decode_query_len

    max_block_count = triton.cdiv(max_seq_len, SPARSE_BLOCK_SIZE)
    score_block_stride = round_up(
        max_block_count,
        SCORE_BLOCK_STRIDE_ALIGNMENT,
    )
    score = torch.empty(
        (index_head_count, total_query_tokens, score_block_stride),
        dtype=torch.float32,
        device=idx_q.device,
    )

    init_mask = torch.zeros(
        (total_query_tokens, score_block_stride),
        dtype=torch.bool,
        device=seq_lens.device,
    )
    local_mask = torch.zeros_like(init_mask)
    mask_chunk_count = max(
        1,
        min(
            DECODE_MASK_MAX_CHUNK_COUNT,
            DECODE_MASK_TARGET_PROGRAM_COUNT
            // max(1, total_query_tokens),
        ),
    )
    mask_chunk_blocks = triton.cdiv(max_block_count, mask_chunk_count)
    _decode_init_local_mask_kernel[
        (total_query_tokens, mask_chunk_count)
    ](
        init_mask,
        local_mask,
        seq_lens,
        SPARSE_BLOCK_SIZE,
        max_block_count,
        decode_query_len,
        mask_chunk_blocks,
        init_blocks,
        local_blocks,
        init_mask.stride(0),
        init_mask.stride(1),
        BLOCK_SIZE_K=DECODE_MASK_BLOCK_TILE_SIZE,
    )

    use_pdl = current_platform.is_arch_support_pdl()
    launch_kwargs = {"launch_pdl": True} if use_pdl else {}
    decode_query_tile_size = triton.next_power_of_2(
        max_decode_query_len
    )
    decode_score_grid = lambda metadata: (
        request_count,
        metadata["num_kv_chunks"],
    )
    _decode_index_score_kernel[decode_score_grid](
        idx_q,
        index_kv_cache,
        score,
        init_mask,
        local_mask,
        block_table,
        seq_lens,
        index_head_count,
        head_dim,
        request_count,
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
        init_mask.stride(0),
        init_mask.stride(1),
        block_table.stride(0),
        BLOCK_SIZE_K=SPARSE_BLOCK_SIZE,
        BLOCK_SIZE_Q=decode_query_tile_size,
        USE_PDL=use_pdl,
        **launch_kwargs,
    )

    tail_chunk_count = max(
        1,
        min(
            DECODE_TAIL_FILL_MAX_CHUNK_COUNT,
            DECODE_TAIL_FILL_TARGET_PROGRAM_COUNT
            // max(1, total_query_tokens * index_head_count),
        ),
    )
    tail_chunk_blocks = triton.cdiv(max_block_count, tail_chunk_count)
    _decode_score_tail_fill_kernel[
        (total_query_tokens, index_head_count, tail_chunk_count)
    ](
        score,
        seq_lens,
        SPARSE_BLOCK_SIZE,
        max_block_count,
        decode_query_len,
        tail_chunk_blocks,
        score.stride(0),
        score.stride(1),
        score.stride(2),
        BLOCK_SIZE_K=DECODE_TAIL_FILL_BLOCK_TILE_SIZE,
    )

    selected_count = min(topk, max_block_count)
    score_rows = score[:, :total_query_tokens, :max_block_count]
    raw_indices = torch.topk(
        score_rows,
        k=selected_count,
        dim=-1,
    ).indices
    topk_indices = _copy_topk_indices(raw_indices, topk, out)

    _decode_topk_invalid_index_mask_kernel[
        (total_query_tokens, index_head_count)
    ](
        topk_indices,
        seq_lens,
        SPARSE_BLOCK_SIZE,
        topk,
        decode_query_len,
        topk_indices.stride(0),
        topk_indices.stride(1),
        topk_indices.stride(2),
    )
    return topk_indices
