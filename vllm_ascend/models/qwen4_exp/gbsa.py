# SPDX-License-Identifier: Apache-2.0
"""Adapt the generic CANN GBSA operator to Qwen4Exp QSA semantics."""

import torch

from vllm_ascend.ops.gbsa import (
    generic_block_sparse_attention,
    generic_block_sparse_attention_metadata,
)
from vllm_ascend.ops.triton.qwen4_exp.gbsa import prepare_qsa_gbsa_inputs

GBSA_COMPRESS_RATIO = 4
GBSA_RESIDUAL_TOKENS = GBSA_COMPRESS_RATIO - 1
GBSA_PAGE_SIZE = 128


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
    result, _ = qsa_gbsa_local_attention(
        query,
        key_cache,
        value_cache,
        compact_indices,
        block_table,
        token_to_req,
        logical_positions,
        seq_lens,
        prepared_metadata=prepared_metadata,
    )
    return output.copy_(result)


def qsa_gbsa_local_attention(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    compact_indices: torch.Tensor,
    block_table: torch.Tensor,
    token_to_req: torch.Tensor,
    logical_positions: torch.Tensor,
    seq_lens: torch.Tensor,
    *,
    prepared_metadata: torch.Tensor | None = None,
    dcp_size: int = 1,
    dcp_rank: int = 0,
    dcp_interleave: int = GBSA_COMPRESS_RATIO,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute attention over this rank's KV. DCP also needs the LSE for merging."""
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

    if dcp_size > 1 and prepared_metadata is not None:
        raise ValueError("QSA DCP GBSA metadata must be built after distributed TopK")
    sparse_idx, sparse_count, row_visible_lens, cu_q, row_block_table = prepare_qsa_gbsa_inputs(
        compact_indices,
        logical_positions,
        seq_lens,
        token_to_req,
        block_table,
        num_kv_heads,
        dcp_size=dcp_size,
        dcp_rank=dcp_rank,
        dcp_interleave=dcp_interleave,
    )
    common = dict(
        cu_seqlens_q=cu_q,
        seqused_kv=row_visible_lens,
        layout_q="TND",
        layout_kv="PA_BBND",
        layout_sparse_pattern=4,
        mask_mode=1,
        quant_mode=0,
        softmax_precision=0,
        residual_block_mode=1,
        is_consistent_topk=False,
    )
    metadata = prepared_metadata
    if metadata is None:
        metadata = generic_block_sparse_attention_metadata(
            sparse_idx,
            sparse_count,
            query.shape[1],
            num_kv_heads,
            query.shape[-1],
            [1, GBSA_COMPRESS_RATIO],
            max_seqlen_q=1,
            max_seqlen_kv=int(logical_max_seqlen_kv),
            **common,
        )
    result, lse = generic_block_sparse_attention(
        query.contiguous(),
        key_cache,
        value_cache,
        sparse_idx,
        sparse_count,
        [1, GBSA_COMPRESS_RATIO],
        metadata=metadata,
        block_table=row_block_table,
        softmax_scale=query.shape[-1] ** -0.5,
        return_softmax_lse=dcp_size > 1,
        **common,
    )
    if dcp_size > 1:
        # Some devices leave empty sparse rows unspecified. They must be the
        # neutral contribution to the global softmax, including graph padding.
        empty = (sparse_count == 0) & (row_visible_lens % GBSA_COMPRESS_RATIO == 0)
        empty = empty.transpose(0, 1).repeat_interleave(query.shape[1] // num_kv_heads, dim=1)
        result.masked_fill_(empty.unsqueeze(-1), 0)
        lse.masked_fill_(empty.unsqueeze(-1), -torch.inf)
    return result, lse


__all__ = ["qsa_gbsa_attention", "qsa_gbsa_local_attention"]
