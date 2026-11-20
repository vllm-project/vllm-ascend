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

"""

from __future__ import annotations

import torch

from vllm_ascend.ops.triton.step5.summary_update import append_current_region

_SPARSE_PATTERN_PER_GROUP_PER_TOKEN = 4
_MASK_CAUSAL = 1
_QUANT_NONE = 0
_SOFTMAX_PRECISION_FP32 = 1


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
    from cann_ops_transformer.ops.attention.generic_block_sparse_attention import (
        generic_block_sparse_attention_metadata,
    )

    groups, tokens = counts.shape
    idx_shape = torch.zeros((groups, tokens, topk + 1), dtype=torch.int32, device=counts.device)
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
    from cann_ops_transformer.ops.attention.generic_block_sparse_attention import generic_block_sparse_attention

    idx, counts = append_current_region(ids, positions, groups=num_kv_groups, region_size=region_size)
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
