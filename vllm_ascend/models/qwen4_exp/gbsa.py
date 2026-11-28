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
    *,
    query_start_loc: torch.Tensor | None = None,
    max_query_len: int | None = None,
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

    if query_start_loc is not None:
        if query_start_loc.shape != (block_table.shape[0] + 1,):
            raise ValueError("QSA GBSA needs one query boundary per request plus the final offset")
        if query_start_loc.dtype != torch.int64:
            raise ValueError("QSA GBSA query boundaries must be int64")
        if query_start_loc.device != query.device:
            raise ValueError("QSA GBSA query boundaries must match query device")
    if max_query_len is None:
        if query_start_loc is not None and prepared_metadata is None:
            raise ValueError("QSA GBSA packed query layout requires max_query_len")
        effective_max_query_len = 1
    elif max_query_len < 1:
        raise ValueError("QSA GBSA max_query_len must be positive")
    else:
        effective_max_query_len = max_query_len

    sparse_idx, sparse_count, kv_lengths, cu_q, page_table = prepare_qsa_gbsa_inputs(
        compact_indices, logical_positions, seq_lens, token_to_req, block_table, num_kv_heads, query_start_loc
    )
    common = dict(
        cu_seqlens_q=cu_q,
        seqused_kv=kv_lengths,
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
            num_kv_heads,
            query.shape[-1],
            [1, GBSA_COMPRESS_RATIO],
            max_seqlen_q=effective_max_query_len,
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
        block_table=page_table,
        softmax_scale=query.shape[-1] ** -0.5,
        **common,
    )
    output.copy_(result)
    return output


__all__ = ["qsa_gbsa_attention"]
