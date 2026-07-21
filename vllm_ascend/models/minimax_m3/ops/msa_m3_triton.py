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

# Data-layout constants.
SPARSE_BLOCK_SIZE = 128
SCORE_BLOCK_STRIDE_ALIGNMENT = 16

# Index-score kernel configuration.
PREFILL_SCORE_QUERY_TILE_SIZE = 96
PREFILL_SCALAR_SCORE_BLOCK_TILE_SIZE = 32

_FP8_DTYPES = (torch.float8_e4m3fn, torch.float8_e5m2)


def _as_triton_index_kv_cache(
    index_kv_cache: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    """Normalizes the index-key cache to [num_blocks, 128, head_dim]."""
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
        raise ValueError(f"Unexpected split main kv cache shapes: {tuple(k_cache.shape)}, {tuple(v_cache.shape)}")
    return k_cache, v_cache


def _is_arch_support_pdl() -> bool:
    if current_platform.device_name == "npu":
        return False
    is_supported = getattr(current_platform, "is_arch_support_pdl", None)
    return bool(is_supported()) if callable(is_supported) else False


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


def _prune_decode_score_configs(configs, named_args, **_):
    """Keeps decode split-K launches within the configured program budget."""
    request_count = max(1, named_args["num_reqs"])
    chunk_limit = max(1, 512 // request_count)
    chunk_limit = 1 << (chunk_limit.bit_length() - 1)
    valid_configs = [config for config in configs if config.kwargs["num_kv_chunks"] <= chunk_limit]
    return valid_configs or configs[:1]


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
def _prefill_index_score_kernel(
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

    sequence_start = tl.load(query_start_offsets_ptr + batch_id)
    sequence_end = tl.load(query_start_offsets_ptr + batch_id + 1)
    query_length = sequence_end - sequence_start

    sequence_length = tl.load(sequence_lengths_ptr + batch_id)
    prefix_length = tl.load(prefix_lengths_ptr + batch_id)

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
        + (sequence_start + query_offsets[:, None]) * query_token_stride
        + head_id * query_head_stride
        + dim_offsets[None, :] * query_dim_stride,
        mask=query_mask[:, None].broadcast_to((BLOCK_SIZE_Q, head_dim)),
        other=0.0,
    )

    block_table_row_ptr = block_table_ptr + batch_id * block_table_batch_stride

    score_row_ptrs = score_ptr + head_id * score_head_stride + (sequence_start + query_offsets) * score_token_stride

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
    earliest_query_position = prefix_length + query_tile_start
    causally_full_block_count = (earliest_query_position + 1) // BLOCK_SIZE_K

    # Only complete sequence blocks may use an unmasked key load.
    complete_sequence_block_count = sequence_length // BLOCK_SIZE_K

    full_block_count = tl.minimum(
        causally_full_block_count,
        complete_sequence_block_count,
    )

    key_position_offsets = key_lane_offsets[None, :] * key_position_stride
    key_dim_offsets = dim_offsets[:, None] * key_dim_stride

    # Fully visible historical blocks.
    #
    # These blocks require neither a key-position mask nor a causal mask.
    for block_id in tl.range(0, full_block_count):
        page_id = tl.load(block_table_row_ptr + block_id).to(tl.int64)

        key = tl.load(
            index_key_cache_ptr + page_id * key_block_stride + key_position_offsets + key_dim_offsets,
        )

        query_key = tl.dot(query, key)
        block_score = tl.max(query_key, axis=1)

        tl.store(
            score_row_ptrs + block_id * score_block_stride,
            block_score,
            mask=query_mask,
        )

    # Blocks intersecting the query tile's causal boundary or the sequence
    # boundary.
    #
    # When prefix_length is block aligned, this normally processes one block.
    # Keeping it as a loop also correctly handles an unaligned prefix, where
    # the query tile can overlap two sparse blocks.
    boundary_key_start = full_block_count * BLOCK_SIZE_K

    for key_block_start in tl.range(
        boundary_key_start,
        visible_key_end,
        BLOCK_SIZE_K,
    ):
        block_id = key_block_start // BLOCK_SIZE_K

        page_id = tl.load(block_table_row_ptr + block_id).to(tl.int64)

        key_positions = key_block_start + key_lane_offsets
        key_mask = key_positions < sequence_length

        key = tl.load(
            index_key_cache_ptr + page_id * key_block_stride + key_position_offsets + key_dim_offsets,
            mask=key_mask[None, :].broadcast_to((head_dim, BLOCK_SIZE_K)),
            other=0.0,
        )

        query_key = tl.dot(query, key)

        causal_mask = query_positions[:, None] >= key_positions[None, :]
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
            score_row_ptrs + block_id * score_block_stride,
            block_score,
            mask=query_mask,
        )


# ---------------------------------------------------------------------------
# Scalar (head_dim == 1) prefill specialization.
#
# For a fully visible block and scalar q:
#   max_j(q * k_j) = q * max_j(k_j), q >= 0
#                    q * min_j(k_j), q < 0
#
# Page extrema are therefore computed once and reused by all query tiles and
# index heads. Causal-boundary blocks still use the original per-token path.
# ---------------------------------------------------------------------------
@triton.jit
def _prefill_scalar_key_extrema_kernel(
    index_key_cache_ptr,
    page_extrema_ptr,
    page_count,
    key_block_stride,
    key_position_stride,
    extrema_page_stride,
    extrema_value_stride,
    BLOCK_SIZE_K: tl.constexpr,
):
    """Computes one minimum and maximum for each scalar index-key page."""
    page_id = tl.program_id(0)
    if page_id >= page_count:
        return

    key_lane_offsets = tl.arange(0, BLOCK_SIZE_K)
    key_values = tl.load(
        index_key_cache_ptr + page_id * key_block_stride + key_lane_offsets * key_position_stride,
    ).to(tl.float32)

    page_minimum = tl.min(key_values, axis=0)
    page_maximum = tl.max(key_values, axis=0)
    page_extrema_base = page_extrema_ptr + page_id * extrema_page_stride
    tl.store(
        page_extrema_base,
        page_minimum,
    )
    tl.store(
        page_extrema_base + extrema_value_stride,
        page_maximum,
    )


@triton.jit(
    do_not_specialize_on_alignment=[
        "sequence_lengths_ptr",
        "prefix_lengths_ptr",
    ]
)
def _prefill_scalar_index_score_kernel(
    query_ptr,
    index_key_cache_ptr,
    page_extrema_ptr,
    score_ptr,
    block_table_ptr,
    query_start_offsets_ptr,
    sequence_lengths_ptr,
    prefix_lengths_ptr,
    index_head_count: tl.constexpr,
    query_token_stride,
    query_head_stride,
    key_block_stride,
    key_position_stride,
    extrema_page_stride,
    extrema_value_stride,
    score_head_stride,
    score_token_stride,
    score_block_stride,
    block_table_batch_stride,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_B: tl.constexpr,
):
    """Computes scalar prefill scores using reusable page extrema."""
    tl.static_assert(
        BLOCK_SIZE_Q <= BLOCK_SIZE_K,
        "BLOCK_SIZE_Q must not exceed BLOCK_SIZE_K",
    )

    query_tile_id = tl.program_id(0)
    batch_head_id = tl.program_id(1)
    batch_id = batch_head_id // index_head_count
    head_id = batch_head_id % index_head_count

    sequence_start = tl.load(query_start_offsets_ptr + batch_id)
    sequence_end = tl.load(query_start_offsets_ptr + batch_id + 1)
    query_length = sequence_end - sequence_start
    sequence_length = tl.load(sequence_lengths_ptr + batch_id)
    prefix_length = tl.load(prefix_lengths_ptr + batch_id)

    query_tile_start = query_tile_id * BLOCK_SIZE_Q
    if query_tile_start >= query_length:
        return

    query_lane_offsets = tl.arange(0, BLOCK_SIZE_Q)
    query_offsets = query_tile_start + query_lane_offsets
    query_mask = query_offsets < query_length
    query_positions = prefix_length + query_offsets
    query_values = tl.load(
        query_ptr + (sequence_start + query_offsets) * query_token_stride + head_id * query_head_stride,
        mask=query_mask,
        other=0.0,
    ).to(tl.float32)

    block_table_row_ptr = block_table_ptr + batch_id * block_table_batch_stride
    score_row_ptrs = score_ptr + head_id * score_head_stride + (sequence_start + query_offsets) * score_token_stride

    query_tile_valid_end = tl.minimum(
        query_length,
        query_tile_start + BLOCK_SIZE_Q,
    )
    visible_key_end = tl.minimum(
        sequence_length,
        prefix_length + query_tile_valid_end,
    )

    earliest_query_position = prefix_length + query_tile_start
    causally_full_block_count = (earliest_query_position + 1) // BLOCK_SIZE_K
    complete_sequence_block_count = sequence_length // BLOCK_SIZE_K
    full_block_count = tl.minimum(
        causally_full_block_count,
        complete_sequence_block_count,
    )

    # Process several fully visible logical blocks per loop iteration. Each
    # logical block maps through block_table to a physical page whose extrema
    # were computed once by _prefill_scalar_key_extrema_kernel.
    block_lane_offsets = tl.arange(0, BLOCK_SIZE_B)
    for block_tile_start in tl.range(0, full_block_count, BLOCK_SIZE_B):
        block_ids = block_tile_start + block_lane_offsets
        block_mask = block_ids < full_block_count
        page_ids = tl.load(
            block_table_row_ptr + block_ids,
            mask=block_mask,
            other=0,
        ).to(tl.int64)

        extrema_base_ptrs = page_extrema_ptr + page_ids * extrema_page_stride
        page_minimums = tl.load(
            extrema_base_ptrs,
            mask=block_mask,
            other=0.0,
        )
        page_maximums = tl.load(
            extrema_base_ptrs + extrema_value_stride,
            mask=block_mask,
            other=0.0,
        )

        selected_extrema = tl.where(
            query_values[:, None] >= 0.0,
            page_maximums[None, :],
            page_minimums[None, :],
        )
        block_scores = query_values[:, None] * selected_extrema
        tl.store(
            score_row_ptrs[:, None] + block_ids[None, :] * score_block_stride,
            block_scores,
            mask=query_mask[:, None] & block_mask[None, :],
        )

    # The one or two blocks intersecting the causal/sequence boundary cannot
    # use whole-page extrema, because some positions in the page are future
    # tokens for part of the query tile. Keep the exact original computation.
    key_lane_offsets = tl.arange(0, BLOCK_SIZE_K)
    key_position_offsets = key_lane_offsets * key_position_stride
    boundary_key_start = full_block_count * BLOCK_SIZE_K
    for key_block_start in tl.range(
        boundary_key_start,
        visible_key_end,
        BLOCK_SIZE_K,
    ):
        block_id = key_block_start // BLOCK_SIZE_K
        page_id = tl.load(block_table_row_ptr + block_id).to(tl.int64)

        key_positions = key_block_start + key_lane_offsets
        key_mask = key_positions < sequence_length
        key_values = tl.load(
            index_key_cache_ptr + page_id * key_block_stride + key_position_offsets,
            mask=key_mask,
            other=0.0,
        ).to(tl.float32)

        query_key = query_values[:, None] * key_values[None, :]
        causal_mask = query_mask[:, None] & key_mask[None, :] & (query_positions[:, None] >= key_positions[None, :])
        query_key = tl.where(
            causal_mask,
            query_key,
            float("-inf"),
        )
        block_score = tl.max(query_key, axis=1)
        tl.store(
            score_row_ptrs + block_id * score_block_stride,
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
    configs=[
        triton.Config(
            {"num_kv_chunks": chunk_count},
            num_stages=stage_count,
        )
        for chunk_count in (1, 2, 4, 8, 16, 32, 64, 128, 256)
        for stage_count in (1, 2)
    ],
    key=["num_idx_heads", "BLOCK_SIZE_Q", "head_dim", "num_reqs"],
    prune_configs_by={"early_config_prune": _prune_decode_score_configs},
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
        q_ptr + q_ids[:, None] * stride_q_n + h_offsets[:, None] * stride_q_h + off_d[None, :] * stride_q_d,
        mask=q_mask[:, None],
        other=0.0,
    )  # [HQ,D]
    for blk in tl.range(chunk_start_block, chunk_end_block):
        page = tl.load(bt_row + blk).to(tl.int64)
        pos = blk * BLOCK_SIZE_K + off_k
        pos_mask = pos[None, :] < kv_len[:, None]
        # index-K for this page: [D,N] (transposed), same layout as prefill.
        k = tl.load(
            ik_cache_ptr + page * stride_ik_blk + off_k[None, :] * stride_ik_pos + off_d[:, None] * stride_ik_d,
        )  # [D,N]
        # fp32 accumulation is required for the fp8 (e4m3) index cache: q/k are
        # loaded in their stored dtype (bf16 or e4m3) and the MMA accumulates in
        # fp32 so the per-block max score is exact for the fp8 indexer too.
        qk = tl.dot(q, k, out_dtype=tl.float32)  # [HQ,N]
        qk = tl.where(pos_mask & q_mask[:, None], qk, float("-inf"))
        score = tl.max(qk, axis=1)  # [HQ]
        mask_off = q_ids * stride_mask_q + blk * stride_mask_k
        is_init = tl.load(init_mask_ptr + mask_off) != 0
        is_local = tl.load(local_mask_ptr + mask_off) != 0
        score = tl.where(is_local, 1e29, tl.where(is_init, 1e30, score))
        tl.store(
            score_ptr + h_offsets * stride_s_h + q_ids * stride_s_n + blk * stride_s_k,
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
def _fill_decode_score_tail_kernel(
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
        s_ptrs = score_ptr + pid_h * stride_s_h + pid_b * stride_s_b + blk * stride_s_k
        tl.store(s_ptrs, float("-inf"), mask=store_mask)


# ---------------------------------------------------------------------------
# Decode top-k: torch.topk on scores, then mask invalid block ids per token.
# Forced init/local blocks are already encoded in the scores.
# ---------------------------------------------------------------------------
@triton.heuristics({"BLOCK_SIZE_T": lambda args: triton.next_power_of_2(args["topk"])})
@triton.jit(do_not_specialize=["decode_query_len"])
def _mask_decode_topk_indices_kernel(
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
    ti_ptrs = ti_ptr + pid_h * stride_ti_h + pid_b * stride_ti_b + off_t * stride_ti_t
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
def _prepare_prefill_topk_scores_kernel(
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
    tl.static_assert(sparse_block_size >= BLOCK_SIZE_Q)

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

    valid_block_counts = (prefix_length + query_offsets + sparse_block_size) // sparse_block_size
    valid_block_counts = tl.minimum(
        valid_block_counts,
        score_block_count,
    )

    # Because BLOCK_SIZE_Q <= sparse_block_size, one query tile contains at
    # most two consecutive valid-block counts.
    min_valid_block_count = (prefix_length + query_tile_start + sparse_block_size) // sparse_block_size
    min_valid_block_count = tl.minimum(
        min_valid_block_count,
        score_block_count,
    )
    max_valid_block_count = tl.minimum(
        min_valid_block_count + 1,
        score_block_count,
    )

    score_row_ptrs = score_ptr + head_id * score_head_stride + token_indices[:, None] * score_token_stride
    forced_block_offsets = tl.arange(0, BLOCK_SIZE_FORCE)

    if init_block_count > 0:
        init_mask = (
            query_mask[:, None]
            & (forced_block_offsets[None, :] < init_block_count)
            & (forced_block_offsets[None, :] < valid_block_counts[:, None])
        )
        tl.store(
            score_row_ptrs + forced_block_offsets[None, :] * score_block_stride,
            1e30,
            mask=init_mask,
        )

    rows_with_min_count = query_mask & (valid_block_counts == min_valid_block_count)
    rows_with_max_count = query_mask & (valid_block_counts > min_valid_block_count)

    if local_block_count > 0:
        min_local_start = tl.maximum(
            0,
            min_valid_block_count - local_block_count,
        )
        min_local_count = min_valid_block_count - min_local_start
        min_local_blocks = min_local_start + forced_block_offsets
        tl.store(
            score_row_ptrs + min_local_blocks[None, :] * score_block_stride,
            1e29,
            mask=(rows_with_min_count[:, None] & (forced_block_offsets[None, :] < min_local_count)),
        )

        max_local_start = tl.maximum(
            0,
            max_valid_block_count - local_block_count,
        )
        max_local_count = max_valid_block_count - max_local_start
        max_local_blocks = max_local_start + forced_block_offsets
        tl.store(
            score_row_ptrs + max_local_blocks[None, :] * score_block_stride,
            1e29,
            mask=(rows_with_max_count[:, None] & (forced_block_offsets[None, :] < max_local_count)),
        )

    tail_lane_offsets = tl.arange(0, BLOCK_SIZE_TAIL)
    tail_block_count = score_block_count - min_valid_block_count
    for tail_offset in tl.range(
        0,
        tail_block_count,
        BLOCK_SIZE_TAIL,
    ):
        block_ids = min_valid_block_count + tail_offset + tail_lane_offsets
        block_mask = block_ids < score_block_count
        row_mask = rows_with_min_count[:, None] | (
            rows_with_max_count[:, None] & (block_ids[None, :] >= max_valid_block_count)
        )
        tl.store(
            score_row_ptrs + block_ids[None, :] * score_block_stride,
            float("-inf"),
            mask=row_mask & block_mask[None, :],
        )


@triton.heuristics({"BLOCK_SIZE_T": lambda args: triton.next_power_of_2(args["topk"])})
@triton.jit(do_not_specialize_on_alignment=["prefix_lengths_ptr"])
def _mask_prefill_topk_indices_kernel(
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
    valid_block_counts = (prefix_length + query_offsets + sparse_block_size) // sparse_block_size

    index_ptrs = (
        topk_indices_ptr
        + head_id * index_head_stride
        + token_indices[:, None] * index_token_stride
        + topk_lane_offsets[None, :] * index_topk_stride
    )
    access_mask = query_mask[:, None] & (topk_lane_offsets[None, :] < topk)
    block_ids = tl.load(index_ptrs, mask=access_mask, other=0)

    valid_rank_mask = topk_lane_offsets[None, :] < tl.minimum(
        topk,
        valid_block_counts[:, None],
    )
    valid_block_mask = (block_ids >= 0) & (block_ids < valid_block_counts[:, None])
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
def _prepare_decode_score_masks_kernel(
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
        "BLOCK_SIZE_H": lambda args: max(16, triton.next_power_of_2(args["gqa_group_size"])),
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
    topk_valid_blk = (topk_idx >= 0) & (topk_idx < num_valid_blocks) & (topk_idx < max_blocks)
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
    lse_ptrs = lse_ptr + pid_b * stride_l_b + pid_h * stride_l_h + off_c * stride_l_c
    lse = tl.load(lse_ptrs)  # empty chunks contribute -inf -> weight 0
    valid_chunk = lse > float("-inf")
    o = tl.load(
        o_ptr + off_c[:, None] * stride_o_c + pid_b * stride_o_b + pid_h * stride_o_h + off_d[None, :] * stride_o_d,
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
    out_ptrs = out_ptr + pid_b * stride_out_n + pid_h * stride_out_h + off_d * stride_out_d
    tl.store(out_ptrs, o_merged.to(out_ptr.dtype.element_ty), mask=off_d < head_dim)


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
    index_kv_cache: torch.Tensor,
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
    total_query_tokens, index_head_count, head_dim = idx_q.shape
    assert index_head_count == num_kv_heads, "M3 requires num_idx_heads == num_kv_heads"

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
    if head_dim == 1:
        page_count = index_kv_cache.shape[0]
        page_extrema = torch.empty(
            (page_count, 2),
            dtype=torch.float32,
            device=index_kv_cache.device,
        )
        _prefill_scalar_key_extrema_kernel[(page_count,)](
            index_kv_cache,
            page_extrema,
            page_count,
            index_kv_cache.stride(0),
            index_kv_cache.stride(1),
            page_extrema.stride(0),
            page_extrema.stride(1),
            BLOCK_SIZE_K=SPARSE_BLOCK_SIZE,
        )
        _prefill_scalar_index_score_kernel[score_grid](
            idx_q,
            index_kv_cache,
            page_extrema,
            score,
            block_table,
            cu_seqlens_q,
            seq_lens,
            prefix_lens,
            index_head_count,
            idx_q.stride(0),
            idx_q.stride(1),
            index_kv_cache.stride(0),
            index_kv_cache.stride(1),
            page_extrema.stride(0),
            page_extrema.stride(1),
            score.stride(0),
            score.stride(1),
            score.stride(2),
            block_table.stride(0),
            BLOCK_SIZE_Q=PREFILL_SCORE_QUERY_TILE_SIZE,
            BLOCK_SIZE_K=SPARSE_BLOCK_SIZE,
            BLOCK_SIZE_B=PREFILL_SCALAR_SCORE_BLOCK_TILE_SIZE,
        )
    else:
        _prefill_index_score_kernel[score_grid](
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

    force_tile_size = triton.next_power_of_2(max(1, init_blocks, local_blocks))
    total_query_rows = max(
        1,
        max_query_len * batch_size * index_head_count,
    )
    required_query_tile_size = triton.cdiv(total_query_rows, 128)
    base_query_tile_size = min(
        64,
        triton.next_power_of_2(required_query_tile_size),
    )
    finalize_query_tile_size = max(8, base_query_tile_size)
    finalize_grid = (
        triton.cdiv(max_query_len, finalize_query_tile_size),
        batch_size * index_head_count,
    )
    _prepare_prefill_topk_scores_kernel[finalize_grid](
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
        BLOCK_SIZE_TAIL=16,
    )

    selected_count = min(topk, score_block_count)
    score_rows = score[:, :total_query_tokens, :score_block_count]
    raw_indices = torch.topk(
        score_rows,
        k=selected_count,
        dim=-1,
    ).indices
    topk_indices = _copy_topk_indices(raw_indices, topk, out)

    topk_tile_size = triton.next_power_of_2(max(1, topk))
    mask_tile_limit = max(1, 2048 // topk_tile_size)
    mask_query_tile_size = min(
        base_query_tile_size,
        1 << (mask_tile_limit.bit_length() - 1),
    )
    mask_grid = (
        triton.cdiv(max_query_len, mask_query_tile_size),
        batch_size * index_head_count,
    )
    _mask_prefill_topk_indices_kernel[mask_grid](
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
    index_kv_cache: torch.Tensor,
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
    assert topk > 0
    total_query_tokens, index_head_count, head_dim = idx_q.shape
    assert index_head_count == num_kv_heads, "M3 requires num_idx_heads == num_kv_heads"

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
        min(16, 64 // max(1, total_query_tokens)),
    )
    mask_chunk_blocks = triton.cdiv(max_block_count, mask_chunk_count)
    _prepare_decode_score_masks_kernel[(total_query_tokens, mask_chunk_count)](
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
        BLOCK_SIZE_K=2048,
    )

    use_pdl = current_platform.is_arch_support_pdl()
    launch_kwargs = {"launch_pdl": True} if use_pdl else {}
    decode_query_tile_size = triton.next_power_of_2(max_decode_query_len)
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
            16,
            64 // max(1, total_query_tokens * index_head_count),
        ),
    )
    tail_chunk_blocks = triton.cdiv(max_block_count, tail_chunk_count)
    _fill_decode_score_tail_kernel[(total_query_tokens, index_head_count, tail_chunk_count)](
        score,
        seq_lens,
        SPARSE_BLOCK_SIZE,
        max_block_count,
        decode_query_len,
        tail_chunk_blocks,
        score.stride(0),
        score.stride(1),
        score.stride(2),
        BLOCK_SIZE_K=2048,
    )

    selected_count = min(topk, max_block_count)
    score_rows = score[:, :total_query_tokens, :max_block_count]
    raw_indices = torch.topk(
        score_rows,
        k=selected_count,
        dim=-1,
    ).indices
    topk_indices = _copy_topk_indices(raw_indices, topk, out)

    _mask_decode_topk_indices_kernel[(total_query_tokens, index_head_count)](
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
    lse_partial = torch.empty(num_topk_chunks, total_q, num_heads, dtype=torch.float32, device=q.device)
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
