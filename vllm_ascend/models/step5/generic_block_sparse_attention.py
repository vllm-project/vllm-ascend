# SPDX-License-Identifier: Apache-2.0
"""aclnn GenericBlockSparseAttention adapter.

Wraps ``generic_block_sparse_attention`` (+ its AICPU metadata op) from
ops-transformer as the exact-attention replacement for the Triton
``sparse_attention_decode`` / ``sparse_attention_prefill`` kernels.

Contract parity with the previous pack + Triton chain:

- ``ids`` is the ``indexer_score_topk`` output: ``[G*T, topk]`` int32 logical
  region ids, ascending, ``-1`` tail padding, current region NOT included
  (the fused indexer's mask3 bound keeps it out of the top-k pool).
- The current region ``floor(position / region_size)`` is appended here into
  a ``topk + 1`` wide tensor at slot ``count`` -- the same slot the Triton
  pack kernels (``decode_pack_from_ids`` / ``prefill_pack_from_ids``) used
  for their extra column -- so a full top-k selection plus the current
  region still fits without evicting a selected id.
- ``mask_mode=CAUSAL`` with right-aligned varlen (``cu_seqlens_q`` +
  ``seqused_kv``) reproduces the per-token causal truncation the pack
  kernels expressed through per-region ``valid`` counts: query ``i`` of a
  request sits at kv position ``seqused_kv - q_len + i`` and attends only
  tokens at positions ``<=`` its own.
- K/V stay in the native paged layout: the Step5 KV cache is
  ``[num_blocks, page_size, G, D]``, which is exactly the operator's
  ``PA_BBND`` layout, so the 4-D views are passed through untouched and the
  operator resolves logical region -> physical page through ``block_table``.

The AICPU metadata op is split out (``gbsa_sparse_attention_metadata``): it
schedules from the per-row block COUNTS and the seq-len tensors only and
never reads the id values, and the counts are a function of the query
positions alone (``min(topk, floor(pos / region_size)) + 1``), identical for
every layer in a step. So the metadata is generated once per step in the
attention metadata builder -- outside the model graph, cf. the FA3 scheduler
tiling -- and the per-layer forward passes it straight through.

``is_consistent_topk`` is fixed to ``False``: live counts vary per row in
serving, and the operator's consistent-count fast path does not apply.

``gbsa_union_sparse_attention`` is the union (mask_mode=3) variant of
``gbsa_sparse_attention`` for prefill: the aclnn ``UnionSparseMeta``
operator (ops-nn) deduplicates each region's per-token selections into a
union that GenericBlockSparseAttention computes once, restoring per-token
visibility through ``exactMask`` bitmasks. It is selected per request by
``VLLM_ASCEND_ENABLE_STEP4_UNION_SPARSE`` and runs its union
preprocessing and AICPU metadata per layer (union counts are
selection-dependent, unlike the position-derived base counts).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from cann_ops_transformer.ops.attention.generic_block_sparse_attention import (
    generic_block_sparse_attention,
    generic_block_sparse_attention_metadata,
)

_SPARSE_PATTERN_PER_GROUP_PER_TOKEN = 4
_MASK_CAUSAL = 1
_MASK_UNION = 3
_QUANT_NONE = 0
_SOFTMAX_PRECISION_FP32 = 1
_SOFTMAX_PRECISION_BF16 = 0
_WINDOW_DISABLED = -1
_RESIDUAL_BLOCK_MODE_DISABLED = 0


@triton.jit
def _append_current_region_kernel(
    ids_ptr,
    positions_ptr,
    wide_ptr,
    counts_ptr,
    tokens,
    region_size,
    topk: tl.constexpr,
    block_k: tl.constexpr,
    block_t: tl.constexpr,
):
    """Pack ``block_t`` rows of one group, streaming ``topk`` in chunks."""
    group = tl.program_id(0)
    tile = tl.program_id(1)
    row = tile * block_t + tl.arange(0, block_t)
    row_mask = row < tokens
    rows = group * tokens + row

    live = tl.zeros((block_t,), dtype=tl.int32)
    for k0 in range(0, topk, block_k):
        col = k0 + tl.arange(0, block_k)
        mask = row_mask[:, None] & (col < topk)[None, :]
        ids = tl.load(
            ids_ptr + rows[:, None] * topk + col[None, :],
            mask=mask,
            other=-1,
        )
        live += tl.sum((ids != -1).to(tl.int32), axis=1)
        tl.store(
            wide_ptr + rows[:, None] * (topk + 1) + col[None, :],
            ids,
            mask=mask,
        )

    pad = tl.full((block_t,), -1, dtype=tl.int32)
    tl.store(wide_ptr + rows * (topk + 1) + topk, pad, mask=row_mask)

    positions = tl.load(positions_ptr + row, mask=row_mask, other=0)
    current = (positions // region_size).to(tl.int32)
    slot = tl.minimum(live, topk)
    tl.store(wide_ptr + rows * (topk + 1) + slot, current, mask=row_mask)

    tl.store(counts_ptr + rows, live + 1, mask=row_mask)


def _append_current_region(
    ids: torch.Tensor,
    positions: torch.Tensor,
    *,
    groups: int,
    region_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Append each row's current region into a ``topk + 1`` wide id tensor.

    ``ids`` is ``[G*T, topk]`` ascending logical ids with ``-1`` tail padding;
    the selection pool is strictly below ``floor(position / region_size)``,
    so the appended current region is larger than every selected id and the
    row stays ascending. It lands at slot ``count`` (the extra ``topk``-th
    column when the selection is full), matching the Triton pack kernels'
    current-region append. Counts become ``live + 1``.

    The whole pack (count + copy + append) runs as a single fused Triton
    kernel.

    Returns ``(idx [G, T, topk + 1], counts [G, T])`` ready for the operator.
    """
    rows, topk = ids.shape
    tokens = rows // groups
    wide = torch.empty(
        (groups, tokens, topk + 1), device=ids.device, dtype=ids.dtype
    )
    counts = torch.empty(
        (groups, tokens), device=ids.device, dtype=torch.int32
    )
    block_k = triton.next_power_of_2(min(topk, 512))
    block_t = max(1, min(64, 8192 // block_k))
    grid = (groups, triton.cdiv(tokens, block_t))
    _append_current_region_kernel[grid](
        ids,
        positions,
        wide,
        counts,
        tokens,
        region_size,
        topk=topk,
        block_k=block_k,
        block_t=block_t,
    )
    return wide, counts


def gbsa_expected_counts(
    positions: torch.Tensor,
    *,
    groups: int,
    region_size: int,
    topk: int,
) -> torch.Tensor:
    """Live block counts the fused path yields for these positions.

    The fused indexer fills exactly ``min(top_k, floor(pos / region_size))``
    slots per row (its mask3 pool is the regions strictly below the current
    one) and the current-region append adds one more, so the per-row count is
    ``min(topk, floor(pos / region_size)) + 1`` -- a function of the position
    only. This is what ``_append_current_region`` produces from the real ids,
    and it is identical for every layer in a step, which is what lets the
    AICPU metadata be built once per step outside the model.

    Returns ``counts [G, T]`` int32 (contiguous, group-major).
    """
    live = (positions.to(torch.int32) // region_size).clamp(max=topk)
    return (live + 1).unsqueeze(0).expand(groups, -1).contiguous()


def gbsa_sparse_attention_metadata(
    counts: torch.Tensor,
    *,
    num_heads: int,
    num_kv_groups: int,
    head_dim: int,
    region_size: int,
    topk: int,
    cu_seqlens_q: torch.Tensor,
    seqused_kv: torch.Tensor,
    max_query_len: int,
    max_pages: int,
    page_size: int,
) -> torch.Tensor:
    """Run the AICPU scheduling op once per step, outside the model graph.

    The AICPU kernel schedules from ``sparse_block_count`` (values), the
    seq-len tensors and the idx SHAPE (block capacity ``topk + 1``); the id
    values themselves are only resolved by the AI Core attention kernel at
    run time, so a zero-filled idx of the right shape is sufficient here.
    Must be called with the same layout / mask / quant attributes as the
    subsequent ``gbsa_sparse_attention`` calls that consume the result.
    """
    groups, tokens = counts.shape
    idx_shape = torch.zeros(
        (groups, tokens, topk + 1), dtype=torch.int32, device=counts.device
    )
    return generic_block_sparse_attention_metadata(
        idx_shape,
        counts,
        num_heads,
        num_kv_groups,
        head_dim,
        [1, region_size],
        cu_seqlens_q=cu_seqlens_q.to(torch.int64),
        seqused_kv=seqused_kv.to(torch.int32),
        max_seqlen_q=max_query_len,
        max_seqlen_kv=max_pages * page_size,
        layout_q="TND",
        layout_kv="PA_BBND",
        layout_sparse_pattern=_SPARSE_PATTERN_PER_GROUP_PER_TOKEN,
        mask_mode=_MASK_CAUSAL,
        quant_mode=_QUANT_NONE,
        softmax_precision=_SOFTMAX_PRECISION_FP32,
        is_consistent_topk=False,
    )


def _union_step_geometry(
    seq_lens_host: list,
    context_lens_host: list,
    *,
    region_size: int,
    topk: int,
) -> tuple[int, int, int, int]:
    """Host-side union geometry shared by every layer of one prefill step.

    Returns ``(max_query_len, num_union_groups, union_capacity,
    max_kv_blocks)``. All four derive from the step's seq-lens and the
    selection width only -- never from device data -- so the attention
    metadata builder can compute them before the model runs and every
    layer's union tensors come out with exactly these shapes.
    """
    query_lens = [s - c for s, c in zip(seq_lens_host, context_lens_host)]
    max_query_len = max(query_lens)
    max_kv_blocks = max(1, max(-(-s // region_size) for s in seq_lens_host))
    num_union_groups = sum(-(-q // region_size) for q in query_lens)
    union_capacity = min(region_size * (topk + 1), max_kv_blocks)
    return max_query_len, num_union_groups, union_capacity, max_kv_blocks


def gbsa_union_sparse_metadata(
    *,
    num_heads: int,
    num_kv_groups: int,
    head_dim: int,
    region_size: int,
    topk: int,
    cu_seqlens_q: torch.Tensor,
    seqused_kv: torch.Tensor,
    seq_lens_host: list,
    context_lens_host: list,
    max_pages: int,
    page_size: int,
) -> torch.Tensor:
    """Run the union (``mask_mode=3``) AICPU scheduling op once per step.

    The scheduler tiles the step's query geometry and hands each core a
    range of union-group block slots; it is fed the union *capacity* bound
    -- the largest union any layer of the step can produce -- instead of
    one layer's actual counts. Every layer's real union count is at most
    that bound by construction (the union builder op validates it), and
    the AI Core attention kernel clamps its scheduled ranges to the
    runtime ``sparse_block_count`` tensor, so a single schedule serves all
    layers of the step. This is the same upper-bound-scheduling contract
    the base path's ``gbsa_expected_counts`` relies on, and it is why the
    metadata is tiling, not per-layer data.

    Only tensor shapes and the count values are read on the host; the
    zero-filled idx never reaches the AI Core kernel.
    """
    max_query_len, num_union_groups, union_capacity, _ = _union_step_geometry(
        seq_lens_host, context_lens_host, region_size=region_size, topk=topk
    )
    idx_shape = torch.zeros(
        (num_kv_groups, num_union_groups, union_capacity),
        dtype=torch.int32,
        device=seqused_kv.device,
    )
    counts = torch.full(
        (num_kv_groups, num_union_groups),
        union_capacity,
        dtype=torch.int32,
        device=seqused_kv.device,
    )
    return generic_block_sparse_attention_metadata(
        idx_shape,
        counts,
        num_heads,
        num_kv_groups,
        head_dim,
        [1, region_size],
        cu_seqlens_q=cu_seqlens_q.to(torch.int64),
        seqused_kv=seqused_kv.to(torch.int32),
        max_seqlen_q=max_query_len,
        max_seqlen_kv=max_pages * page_size,
        layout_q="TND",
        layout_kv="PA_BBND",
        layout_sparse_pattern=_SPARSE_PATTERN_PER_GROUP_PER_TOKEN,
        mask_mode=_MASK_UNION,
        quant_mode=_QUANT_NONE,
        softmax_precision=_SOFTMAX_PRECISION_BF16,
        win_left=_WINDOW_DISABLED,
        win_right=_WINDOW_DISABLED,
        residual_block_mode=_RESIDUAL_BLOCK_MODE_DISABLED,
        is_consistent_topk=False,
    )


def gbsa_sparse_attention(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    ids: torch.Tensor,
    positions: torch.Tensor,
    block_table: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    seqused_kv: torch.Tensor,
    metadata: torch.Tensor,
    *,
    num_kv_groups: int,
    region_size: int,
    softmax_scale: float,
) -> torch.Tensor:
    """Sparse attention over per-(group, token) selected logical regions.

    Args:
        query: ``[total_q, num_heads, head_dim]`` (``TND``).
        key_cache, value_cache: ``[num_blocks, page_size, G, head_dim]`` paged
            cache views (``PA_BBND``), passed through to the operator.
        ids: ``[G * total_q, topk]`` int32 logical region ids, ascending,
            ``-1`` tail padding, current region not included.
        positions: ``[total_q]`` absolute position of each query token; the
            current region is derived per token and appended internally.
        block_table: ``[num_reqs, max_pages]`` int32 logical-to-physical map.
        cu_seqlens_q: ``[num_reqs + 1]`` cumulative query lengths (1 per
            request for decode, ``K+1`` per request for the multi-token MTP
            verify path).
        seqused_kv: ``[num_reqs]`` used KV length per request (right-aligned
            causal anchor; must be ``>=`` the request's query length).
        metadata: per-step AICPU scheduling tensor from
            ``gbsa_sparse_attention_metadata`` (built once in the attention
            metadata builder and shared across layers; the counts it was
            scheduled from are the ``gbsa_expected_counts`` of ``positions``).
        num_kv_groups: G, the number of KV head groups.
        region_size: tokens per region (sparse block height).
        softmax_scale: attention softmax scale.

    Returns:
        ``out`` shaped ``[total_q, num_heads, head_dim]``, same dtype as
        ``query``.
    """
    idx, counts = _append_current_region(
        ids, positions, groups=num_kv_groups, region_size=region_size
    )
    common = dict(
        cu_seqlens_q=cu_seqlens_q.to(torch.int64),
        seqused_kv=seqused_kv.to(torch.int32),
        layout_q="TND",
        layout_kv="PA_BBND",
        layout_sparse_pattern=_SPARSE_PATTERN_PER_GROUP_PER_TOKEN,
        mask_mode=_MASK_CAUSAL,
        quant_mode=_QUANT_NONE,
        softmax_precision=_SOFTMAX_PRECISION_FP32,
        is_consistent_topk=False,
    )
    out, _lse = generic_block_sparse_attention(
        query,
        key_cache,
        value_cache,
        idx,
        counts,
        [1, region_size],
        metadata=metadata,
        block_table=block_table,
        softmax_scale=softmax_scale,
        return_softmax_lse=False,
        **common,
    )
    return out


def gbsa_union_sparse_attention(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    ids: torch.Tensor,
    positions: torch.Tensor,
    block_table: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    seqused_kv: torch.Tensor,
    *,
    num_kv_groups: int,
    region_size: int,
    softmax_scale: float,
    seq_lens_host: list,
    context_lens_host: list,
    metadata: torch.Tensor,
) -> torch.Tensor:
    """Union (mask_mode=3) variant of :func:`gbsa_sparse_attention`.

    The aclnn ``UnionSparseMeta`` operator (ops-nn) merges the appended
    per-token selections of each ``region_size``-token region into one
    deduplicated union; GenericBlockSparseAttention then gathers and
    computes the union once and restores every token's exact selection
    through the union ``exactMask`` bitmasks. This removes the redundant
    KV gathers / QK^T GEMMs of heavily-overlapping adjacent selections
    (measured on real Step-5 prefill: union ~555 blocks per region vs
    ~3592 per-token block visits, 2.7x on the attention op).

    ``metadata`` is the step's union AICPU schedule from
    :func:`gbsa_union_sparse_metadata`, built once in the attention
    metadata builder. It is pure tiling -- scheduled from the union
    capacity bound, with every layer's actual union counts resolved
    against it at run time by the AI Core kernel -- so all layers of the
    step share it, exactly like the base path's per-step metadata.

    Everything on the hot path stays on device: the operator consumes the
    packed int32 seq-len tensors directly, and the union geometry (group
    count, capacity) comes from the builder's CPU seq-len copies
    (``seq_lens_host`` / ``context_lens_host``) -- no tensor is read back
    to the host and no shape value is derived from device data.

    Returns ``out`` shaped ``[total_q, num_heads, head_dim]``.
    """
    if num_kv_groups != 1:
        raise NotImplementedError(
            "The union sparse path currently requires a single KV group "
            f"(MQA), got num_kv_groups={num_kv_groups}."
        )
    if region_size != 8:
        raise NotImplementedError(
            "The union sparse path currently requires region_size=8 "
            f"(UnionSparseMeta v1 block/union granularity), got {region_size}."
        )
    try:
        # Registers torch.ops.cann_ops_nn.union_sparse_meta_launch
        # (schema + NPU dispatch); a sys.modules hit after the first call.
        import cann_ops_nn_custom_nn  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "VLLM_ASCEND_ENABLE_STEP4_UNION_SPARSE=1 requires the "
            "UnionSparseMeta operator (ops-nn): install the custom_nn "
            "vendor package and the cann_ops_nn_custom_nn wheel."
        ) from exc

    idx, counts = _append_current_region(
        ids, positions, groups=num_kv_groups, region_size=region_size
    )
    # Union geometry from the builder's CPU copies (host ints, no sync):
    # the capacity must cover min(region selection width, every logical
    # block of the longest request) -- the safe bound the operator
    # enforces -- and the group count is one union per region. These are
    # step constants, identical to what gbsa_union_sparse_metadata
    # scheduled the step's shared tiling from.
    _, num_union_groups, union_capacity, max_kv_blocks = _union_step_geometry(
        seq_lens_host,
        context_lens_host,
        region_size=region_size,
        topk=idx.shape[2] - 1,
    )

    seqused_kv_i32 = seqused_kv.to(torch.int32)
    cu_seqlens_kv = torch.cat(
        (
            torch.zeros((1,), device=seqused_kv_i32.device, dtype=torch.int32),
            seqused_kv_i32.cumsum(0, dtype=torch.int32),
        )
    )
    union_idx, union_counts, exact_mask = (
        torch.ops.cann_ops_nn.union_sparse_meta_launch(
            idx,
            counts,
            cu_seqlens_q.to(torch.int32),
            cu_seqlens_kv,
            None,
            seqused_kv_i32,
            1,
            region_size,
            region_size,
            True,
            union_capacity,
            num_union_groups,
            max_kv_blocks,
        )
    )
    common = dict(
        cu_seqlens_q=cu_seqlens_q.to(torch.int64),
        cu_seqlens_kv=None,
        seqused_kv=seqused_kv_i32,
        layout_q="TND",
        layout_kv="PA_BBND",
        layout_sparse_pattern=_SPARSE_PATTERN_PER_GROUP_PER_TOKEN,
        mask_mode=_MASK_UNION,
        quant_mode=_QUANT_NONE,
        softmax_precision=_SOFTMAX_PRECISION_BF16,
        win_left=_WINDOW_DISABLED,
        win_right=_WINDOW_DISABLED,
        residual_block_mode=_RESIDUAL_BLOCK_MODE_DISABLED,
        is_consistent_topk=False,
    )
    # The step's shared tiling: scheduled once in the metadata builder
    # from the union capacity bound; the kernel clamps its ranges to this
    # layer's runtime union counts.
    out, _lse = generic_block_sparse_attention(
        query,
        key_cache,
        value_cache,
        union_idx,
        union_counts,
        [1, region_size],
        metadata=metadata,
        block_table=block_table,
        attn_mask=exact_mask,
        softmax_scale=softmax_scale,
        return_softmax_lse=False,
        **common,
    )
    return out
