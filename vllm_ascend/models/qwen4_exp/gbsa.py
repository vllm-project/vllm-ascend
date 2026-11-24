# SPDX-License-Identifier: Apache-2.0
"""Adapt the generic CANN GBSA operator to Qwen4Exp QSA semantics."""

import torch

from vllm_ascend.ops.gbsa import (
    generic_block_sparse_attention,
    generic_block_sparse_attention_metadata,
)
from vllm_ascend.ops.triton.qwen4_exp.gbsa import pack_resident_gbsa_inputs, prepare_qsa_gbsa_inputs

GBSA_COMPRESS_RATIO = 4
GBSA_RESIDUAL_TOKENS = GBSA_COMPRESS_RATIO - 1
GBSA_PAGE_SIZE = 128
GBSA_MAX_GROUPS = 512


def expand_qsa_selected_tokens(
    compact_indices: torch.Tensor, visible_lens: torch.Tensor, *, out: torch.Tensor | None = None
) -> torch.Tensor:
    """Expand compact LI-v2 groups to request-local token IDs under the GBSA tail policy.

    Invalid lanes are -1 and may occur between valid groups. A caller can supply
    an INT32 resolver buffer to retain its address across decode steps.
    """
    if compact_indices.ndim != 2 or compact_indices.shape[1] <= GBSA_RESIDUAL_TOKENS:
        raise ValueError("Expected compact QSA group indices followed by three residual slots")
    groups = compact_indices[:, :-GBSA_RESIDUAL_TOKENS]
    rows, num_groups = groups.shape
    if visible_lens.shape != (rows,):
        raise ValueError("Expected one visible length per QSA query row")
    if out is not None:
        shape = (rows, num_groups * GBSA_COMPRESS_RATIO + GBSA_RESIDUAL_TOKENS)
        if out.shape != shape or out.dtype != torch.int32 or out.device != groups.device or out.stride(1) != 1:
            raise ValueError(
                "QSA resolver output requires INT32 token rows with contiguous columns on the input device"
            )

    # Graph padding has no visible tokens. Every unused lane must carry the
    # resolver sentinel, including tail lanes of a padded row.
    visible_lens = visible_lens.clamp_min(0)
    full_groups = visible_lens.floor_divide(GBSA_COMPRESS_RATIO)
    valid = (groups >= 0) & (groups < full_groups.unsqueeze(-1))
    if num_groups >= GBSA_MAX_GROUPS:
        capacity = num_groups - visible_lens.remainder(GBSA_COMPRESS_RATIO).ne(0).to(torch.int32)
        valid &= valid.to(torch.int32).cumsum(-1) <= capacity.unsqueeze(-1)
    offsets = torch.arange(GBSA_COMPRESS_RATIO, device=groups.device, dtype=groups.dtype)
    tokens = groups.unsqueeze(-1) * GBSA_COMPRESS_RATIO + offsets
    tokens = torch.where(valid.unsqueeze(-1), tokens, -1).flatten(1)
    tail_offsets = torch.arange(GBSA_RESIDUAL_TOKENS, device=groups.device, dtype=groups.dtype)
    tail = full_groups.unsqueeze(-1) * GBSA_COMPRESS_RATIO + tail_offsets
    tail = torch.where(tail < visible_lens.unsqueeze(-1), tail, -1)
    return torch.cat((tokens, tail), dim=-1, out=out)


def qsa_gbsa_attention(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    compact_indices: torch.Tensor,
    block_table: torch.Tensor,
    token_to_req: torch.Tensor,
    logical_positions: torch.Tensor,
    seq_lens: torch.Tensor,
    output: torch.Tensor,
    prepared_metadata: torch.Tensor | None = None,
) -> torch.Tensor:
    if key_cache.ndim != 4 or value_cache.ndim != 4:
        raise ValueError("QSA GBSA requires 4-D K/V caches")
    if key_cache.shape != value_cache.shape:
        raise ValueError("QSA GBSA requires matching K/V cache shapes")
    # The table width counts logical kernel pages; pool rows also include
    # packed gaps and therefore do not describe the sequence-length bound.
    logical_max_seqlen_kv = block_table.shape[1] * GBSA_PAGE_SIZE
    num_kv_heads = key_cache.shape[2]
    if query.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("QSA GBSA requires FP16/BF16 query")
    if key_cache.dtype != query.dtype or value_cache.dtype != query.dtype:
        raise ValueError("QSA GBSA requires matching Q/K/V dtypes")
    if compact_indices.shape[1] <= GBSA_RESIDUAL_TOKENS:
        raise ValueError("QSA GBSA requires compact LI-v2 indices")
    if min(key_cache.shape[:3]) == 0 or min(block_table.shape) == 0:
        raise ValueError("QSA GBSA requires nonempty KV cache and block table")
    if key_cache.shape != value_cache.shape or key_cache.shape[-1] != query.shape[-1]:
        raise ValueError("QSA GBSA requires matching KV shapes and Q/K head dimensions")
    if key_cache.shape[1] != GBSA_PAGE_SIZE:
        raise ValueError("QSA GBSA requires 128-token paged KV cache")
    if query.shape[1] % num_kv_heads:
        raise ValueError("QSA GBSA requires query heads divisible by KV heads")
    if seq_lens.shape != (block_table.shape[0],):
        raise ValueError("QSA GBSA sequence lengths must match block table rows")
    if token_to_req.shape != (query.shape[0],):
        raise ValueError("QSA GBSA request mapping must match query rows")
    if logical_positions.shape != (query.shape[0],):
        raise ValueError("QSA GBSA logical positions must match query rows")

    sparse_idx, sparse_count, row_visible_lens, cu_q, row_block_table = prepare_qsa_gbsa_inputs(
        compact_indices, logical_positions, seq_lens, token_to_req, block_table, num_kv_heads
    )
    return _run_prepared_gbsa(
        query,
        key_cache,
        value_cache,
        output,
        sparse_idx,
        sparse_count,
        row_visible_lens,
        cu_q,
        row_block_table,
        logical_max_seqlen_kv,
        prepared_metadata,
    )


def qsa_resident_gbsa_attention(
    query: torch.Tensor,
    resident: torch.Tensor,
    compact_indices: torch.Tensor,
    block_table: torch.Tensor,
    token_to_req: torch.Tensor,
    logical_positions: torch.Tensor,
    seq_lens: torch.Tensor,
    output: torch.Tensor,
) -> torch.Tensor:
    head_dim = query.shape[-1]
    num_kv_heads = resident.shape[-1] // (2 * head_dim)
    selected, counts, visible, cu_q, row_table = prepare_qsa_gbsa_inputs(
        compact_indices, logical_positions, seq_lens, token_to_req, block_table, num_kv_heads
    )
    keys, values, indices, lengths, pages = pack_resident_gbsa_inputs(
        resident, selected, counts, visible, row_table, head_dim
    )
    return _run_prepared_gbsa(
        query, keys, values, output, indices, counts, lengths, cu_q, pages, pages.shape[1] * GBSA_PAGE_SIZE
    )


def _run_prepared_gbsa(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    output: torch.Tensor,
    sparse_idx: torch.Tensor,
    sparse_count: torch.Tensor,
    row_visible_lens: torch.Tensor,
    cu_q: torch.Tensor,
    row_block_table: torch.Tensor,
    logical_max_seqlen_kv: int,
    prepared_metadata: torch.Tensor | None = None,
) -> torch.Tensor:
    common = dict(
        cu_seqlens_q=cu_q,
        seqused_kv=row_visible_lens,
        layout_q="TND",
        layout_kv="PA_BBND",
        layout_sparse_pattern=4,
        mask_mode=1,
        quant_mode=0,
        softmax_precision=0,
        residual_block_mode=0,
        is_consistent_topk=False,
    )
    metadata = prepared_metadata
    if metadata is None:
        metadata = generic_block_sparse_attention_metadata(
            sparse_idx,
            sparse_count,
            query.shape[1],
            key_cache.shape[2],
            query.shape[-1],
            [1, GBSA_COMPRESS_RATIO],
            max_seqlen_q=1,
            max_seqlen_kv=int(logical_max_seqlen_kv),
            **common,
        )
    result, _ = generic_block_sparse_attention(
        query.contiguous(),
        key_cache,
        value_cache,
        sparse_idx,
        sparse_count,
        [1, GBSA_COMPRESS_RATIO],
        metadata=metadata,
        block_table=row_block_table,
        softmax_scale=query.shape[-1] ** -0.5,
        **common,
    )
    output.copy_(result)
    return output


__all__ = ["qsa_gbsa_attention", "qsa_resident_gbsa_attention"]
