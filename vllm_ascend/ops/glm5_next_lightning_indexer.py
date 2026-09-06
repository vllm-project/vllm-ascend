# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Narrow GLM5 Next KPool lightning indexer op.

The op starts from an already-populated compressed indexer cache. It selects
top-k pool ids, expands them to token ids, appends the current tail tokens, and
returns sparse indices for GLM5 Next sparse attention.
"""

from __future__ import annotations

import torch
from vllm.triton_utils import HAS_TRITON
from vllm.utils.torch_utils import direct_register_custom_op

INDEXER_KPOOL_QUERY_CHUNK_SIZE = 16
INDEXER_KPOOL_KEY_CHUNK_SIZE = 2048
TRITON_HEAD_DIM = 128
TRITON_MAX_POOL_TOPK = 512

if HAS_TRITON:
    from vllm_ascend.ops.triton.glm5_next_lightning_indexer import (
        glm5_next_lightning_indexer_triton,
    )
else:
    glm5_next_lightning_indexer_triton = None


def _align_key_chunk_size(cache_block_size: int) -> int:
    return max(
        cache_block_size,
        INDEXER_KPOOL_KEY_CHUNK_SIZE // cache_block_size * cache_block_size,
    )


def _validate_common_inputs(
    query: torch.Tensor,
    indexer_cache: torch.Tensor,
    weights: torch.Tensor,
    cum_query_lens: torch.Tensor,
    indexer_seq_lens: torch.Tensor,
    indexer_block_table: torch.Tensor,
    positions: torch.Tensor,
    index_topk: int,
    index_kpool: int,
    max_pool_seq_len: int,
) -> None:
    if index_topk <= 0:
        raise ValueError(f"index_topk must be positive, got {index_topk}.")
    if index_kpool <= 0:
        raise ValueError(f"index_kpool must be positive, got {index_kpool}.")
    if index_topk % index_kpool:
        raise ValueError(f"index_topk ({index_topk}) must be divisible by index_kpool ({index_kpool}).")
    if max_pool_seq_len < 0:
        raise ValueError(f"max_pool_seq_len must be non-negative, got {max_pool_seq_len}.")
    if query.ndim != 3:
        raise ValueError(f"GLM5 Next indexer query must be [T,H,D], got {query.shape}.")
    if query.dtype != torch.bfloat16:
        raise TypeError(f"GLM5 Next indexer query must use bfloat16, got {query.dtype}.")
    for name, tensor in (
        ("indexer_cache", indexer_cache),
        ("weights", weights),
        ("cum_query_lens", cum_query_lens),
        ("indexer_seq_lens", indexer_seq_lens),
        ("indexer_block_table", indexer_block_table),
        ("positions", positions),
    ):
        if tensor.device != query.device:
            raise ValueError(f"{name} must be on {query.device}, got {tensor.device}.")
    if weights.shape != query.shape[:2]:
        raise ValueError(f"GLM5 Next indexer weights must be [T,H], got {weights.shape} for query {query.shape}.")
    if weights.dtype != query.dtype:
        raise TypeError(f"GLM5 Next indexer weights must use {query.dtype}, got {weights.dtype}.")
    if indexer_cache.ndim != 4 or indexer_cache.shape[2] != 1:
        raise ValueError(f"GLM5 Next indexer cache must be [blocks,block,1,D], got {indexer_cache.shape}.")
    if indexer_cache.dtype != torch.bfloat16:
        raise TypeError(f"GLM5 Next indexer cache must use bfloat16, got {indexer_cache.dtype}.")
    if indexer_cache.shape[-1] != query.shape[-1]:
        raise ValueError(
            "GLM5 Next indexer query/cache dims differ: "
            f"{query.shape[-1]} and {indexer_cache.shape[-1]}."
        )
    if cum_query_lens.ndim != 1:
        raise ValueError(f"cum_query_lens must be 1-D, got {cum_query_lens.shape}.")
    if cum_query_lens.dtype not in (torch.int32, torch.int64):
        raise TypeError(f"cum_query_lens must use int32/int64, got {cum_query_lens.dtype}.")
    if indexer_seq_lens.shape != cum_query_lens.shape:
        raise ValueError(
            "cum_query_lens and indexer_seq_lens must have the same shape, "
            f"got {cum_query_lens.shape} and {indexer_seq_lens.shape}."
        )
    if indexer_seq_lens.dtype not in (torch.int32, torch.int64):
        raise TypeError(f"indexer_seq_lens must use int32/int64, got {indexer_seq_lens.dtype}.")
    if indexer_block_table.ndim != 2 or indexer_block_table.shape[0] != indexer_seq_lens.shape[0]:
        raise ValueError(
            "indexer_block_table rows must match request count, got "
            f"{indexer_block_table.shape} and {indexer_seq_lens.shape}."
        )
    if indexer_block_table.dtype not in (torch.int32, torch.int64):
        raise TypeError(f"indexer_block_table must use int32/int64, got {indexer_block_table.dtype}.")
    if positions.ndim != 1 or positions.shape[0] != query.shape[0]:
        raise ValueError(
            "positions must provide one value per query, got "
            f"{positions.shape} and {query.shape[0]} queries."
        )
    if positions.dtype not in (torch.int32, torch.int64):
        raise TypeError(f"positions must use int32/int64, got {positions.dtype}.")
    if query.shape[0] > 0 and cum_query_lens.shape[0] == 0:
        raise ValueError("cum_query_lens must not be empty when query has rows.")
    if indexer_cache.shape[0] == 0 and max_pool_seq_len:
        raise ValueError("indexer_cache has no physical blocks for a non-empty cache.")
    if indexer_cache.shape[1] == 0 and max_pool_seq_len:
        raise ValueError("indexer_cache block size is zero for a non-empty cache.")
    if max_pool_seq_len:
        required_pages = (max_pool_seq_len + indexer_cache.shape[1] - 1) // indexer_cache.shape[1]
        if indexer_block_table.shape[1] < required_pages:
            raise ValueError(
                "indexer_block_table has insufficient pages for max_pool_seq_len, got "
                f"{indexer_block_table.shape[1]} pages and need {required_pages}."
            )


def _gather_indexer_k_cache(
    indexer_cache: torch.Tensor,
    dst_k: torch.Tensor,
    block_table: torch.Tensor,
    cu_seq_lens: torch.Tensor,
) -> None:
    if dst_k.shape[0] == 0:
        return

    output_rows = torch.arange(
        dst_k.shape[0],
        dtype=cu_seq_lens.dtype,
        device=dst_k.device,
    )
    request_ids = torch.bucketize(
        output_rows,
        cu_seq_lens[1:],
        right=True,
    )
    logical_indices = output_rows - cu_seq_lens[request_ids]
    cache_block_size = indexer_cache.shape[1]
    logical_pages = torch.div(
        logical_indices,
        cache_block_size,
        rounding_mode="floor",
    )
    page_offsets = torch.remainder(logical_indices, cache_block_size)
    physical_blocks = block_table[
        request_ids,
        logical_pages,
    ].to(torch.int64)
    safe_physical_blocks = physical_blocks.clamp(
        min=0,
        max=indexer_cache.shape[0] - 1,
    )
    gathered_k = indexer_cache[
        safe_physical_blocks,
        page_offsets,
        0,
        :,
    ]
    dst_k.copy_(gathered_k)


def _bf16_mqa_logits(
    query: torch.Tensor,
    key: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    weighted_query = (query * weights.to(torch.bfloat16).unsqueeze(-1)).sum(dim=1)
    return torch.matmul(weighted_query, key.transpose(0, 1)).float()


def _top_k_per_row(
    logits: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
    sparse_count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    columns = torch.arange(
        logits.shape[1],
        dtype=cu_seqlen_ks.dtype,
        device=logits.device,
    )
    valid = (columns[None, :] >= cu_seqlen_ks[:, None]) & (columns[None, :] < cu_seqlen_ke[:, None])
    score_mask_value = torch.finfo(logits.dtype).min
    scores = logits.masked_fill(~valid, score_mask_value)
    if scores.shape[1] < sparse_count:
        scores = torch.nn.functional.pad(
            scores,
            (0, sparse_count - scores.shape[1]),
            value=score_mask_value,
        )
    values, absolute_indices = torch.topk(
        scores,
        k=sparse_count,
        dim=1,
        largest=True,
        sorted=False,
    )
    relative_indices = absolute_indices - cu_seqlen_ks[:, None]
    selected_valid = values != score_mask_value
    relative_indices = torch.where(
        selected_valid,
        relative_indices,
        torch.full_like(relative_indices, -1),
    )
    return relative_indices.to(torch.int32), values


def _pool_topk(
    query: torch.Tensor,
    indexer_cache: torch.Tensor,
    weights: torch.Tensor,
    cum_query_lens: torch.Tensor,
    indexer_seq_lens: torch.Tensor,
    indexer_block_table: torch.Tensor,
    positions: torch.Tensor,
    pool_topk: int,
    index_kpool: int,
    max_pool_seq_len: int,
) -> torch.Tensor:
    output = torch.full(
        (query.shape[0], pool_topk),
        -1,
        dtype=torch.int32,
        device=query.device,
    )
    if query.shape[0] == 0 or max_pool_seq_len == 0:
        return output

    token_ids = torch.arange(
        query.shape[0],
        dtype=cum_query_lens.dtype,
        device=query.device,
    )
    # Padded rows beyond the last request bucketize to num_reqs; clamp them
    # back in range (their outputs are unused downstream).
    request_ids = torch.bucketize(token_ids, cum_query_lens, right=True).clamp_max(
        cum_query_lens.shape[0] - 1
    )
    request_pool_lens = indexer_seq_lens[request_ids].to(torch.int64)
    causal_pool_lens = torch.div(
        positions.to(torch.int64) + 1,
        index_kpool,
        rounding_mode="floor",
    )
    causal_pool_lens = torch.minimum(causal_pool_lens, request_pool_lens)

    cache_block_size = indexer_cache.shape[1]
    key_chunk_size = _align_key_chunk_size(cache_block_size)
    score_mask_value = torch.finfo(torch.float32).min

    for query_start in range(0, query.shape[0], INDEXER_KPOOL_QUERY_CHUNK_SIZE):
        query_end = min(query_start + INDEXER_KPOOL_QUERY_CHUNK_SIZE, query.shape[0])
        chunk_query = query[query_start:query_end]
        chunk_weights = weights[query_start:query_end]
        chunk_request_ids = request_ids[query_start:query_end]
        chunk_pool_lens = causal_pool_lens[query_start:query_end]
        chunk_rows = query_end - query_start
        best_values = torch.full(
            (chunk_rows, pool_topk),
            score_mask_value,
            dtype=torch.float32,
            device=query.device,
        )
        best_indices = torch.full(
            (chunk_rows, pool_topk),
            -1,
            dtype=torch.int64,
            device=query.device,
        )

        for key_start in range(0, max_pool_seq_len, key_chunk_size):
            key_end = min(key_start + key_chunk_size, max_pool_seq_len)
            keys_per_row = key_end - key_start
            if key_start % cache_block_size:
                raise ValueError(
                    "GLM5 Next indexer key chunks must start on compressed cache "
                    f"block boundaries, got key_start={key_start} and block_size={cache_block_size}."
                )

            gather_cu_seq_lens = (
                torch.arange(
                    chunk_rows + 1,
                    dtype=torch.int32,
                    device=query.device,
                )
                * keys_per_row
            )
            gathered_key = torch.empty(
                (chunk_rows * keys_per_row, query.shape[-1]),
                dtype=torch.bfloat16,
                device=query.device,
            )
            first_page = key_start // cache_block_size
            last_page = (key_end + cache_block_size - 1) // cache_block_size
            chunk_block_table = indexer_block_table[
                chunk_request_ids,
                first_page:last_page,
            ]
            _gather_indexer_k_cache(
                indexer_cache,
                gathered_key,
                chunk_block_table,
                gather_cu_seq_lens,
            )

            cu_seqlen_ks = gather_cu_seq_lens[:-1]
            valid_counts = (chunk_pool_lens - key_start).clamp(min=0, max=keys_per_row).to(torch.int32)
            cu_seqlen_ke = cu_seqlen_ks + valid_counts
            logits = _bf16_mqa_logits(
                chunk_query,
                gathered_key,
                chunk_weights,
            )
            chunk_topk, chunk_values = _top_k_per_row(
                logits,
                cu_seqlen_ks,
                cu_seqlen_ke,
                pool_topk,
            )
            chunk_valid = chunk_topk >= 0
            safe_chunk_topk = chunk_topk.clamp_min(0).to(torch.int64)
            absolute_workspace_indices = cu_seqlen_ks[:, None].to(torch.int64) + safe_chunk_topk
            chunk_values = torch.gather(
                logits,
                1,
                absolute_workspace_indices,
            )
            chunk_values = chunk_values.masked_fill(
                ~chunk_valid,
                score_mask_value,
            )
            candidate_indices = torch.where(
                chunk_valid,
                safe_chunk_topk + key_start,
                torch.full_like(safe_chunk_topk, -1),
            )
            merged_values = torch.cat([best_values, chunk_values], dim=1)
            merged_indices = torch.cat([best_indices, candidate_indices], dim=1)
            best_values, selected = torch.topk(
                merged_values,
                k=pool_topk,
                dim=1,
                largest=True,
                sorted=False,
            )
            best_indices = torch.gather(merged_indices, 1, selected)

        best_indices = torch.where(
            best_values == score_mask_value,
            torch.full_like(best_indices, -1),
            best_indices,
        )
        output[query_start:query_end] = best_indices.to(torch.int32)

    return output


def _expand_pools_to_tokens(
    pool_ids: torch.Tensor,
    index_topk: int,
    index_kpool: int,
) -> torch.Tensor:
    offsets = torch.arange(
        index_kpool,
        device=pool_ids.device,
        dtype=torch.int64,
    )
    token_ids = pool_ids.to(torch.int64).unsqueeze(-1) * index_kpool + offsets
    token_ids = token_ids.reshape(pool_ids.shape[0], index_topk)
    valid = (pool_ids >= 0).unsqueeze(-1).expand(-1, -1, index_kpool).reshape(pool_ids.shape[0], index_topk)
    output = token_ids.to(torch.int32)
    return torch.where(valid, output, torch.full_like(output, -1))


def _append_tail_to_topk(
    topk_result: torch.Tensor,
    positions: torch.Tensor,
    index_kpool: int,
) -> torch.Tensor:
    tail_width = index_kpool - 1
    if tail_width == 0:
        return topk_result

    rows, history_width = topk_result.shape
    output_width = history_width + tail_width
    columns = torch.arange(
        output_width,
        device=topk_result.device,
    )[None, :]
    is_history = columns < history_width
    tail_offsets = columns - history_width

    seq_lens = positions.to(torch.int32) + 1
    pool_lens = torch.div(
        seq_lens,
        index_kpool,
        rounding_mode="floor",
    ).to(torch.int32)
    tail_start = pool_lens * index_kpool
    tail_count = seq_lens - tail_start
    is_tail = (tail_offsets >= 0) & (tail_offsets < tail_count[:, None])

    safe_history = torch.minimum(
        columns,
        torch.full_like(columns, history_width - 1),
    ).expand(rows, output_width)
    history_values = torch.gather(topk_result, 1, safe_history)
    tail_values = (tail_start[:, None] + tail_offsets).to(torch.int32)

    output = torch.where(is_history, history_values, -1)
    return torch.where(is_tail, tail_values, output)


def glm5_next_lightning_indexer(
    query: torch.Tensor,
    indexer_cache: torch.Tensor,
    weights: torch.Tensor,
    cum_query_lens: torch.Tensor,
    indexer_seq_lens: torch.Tensor,
    indexer_block_table: torch.Tensor,
    positions: torch.Tensor,
    *,
    index_topk: int,
    index_kpool: int,
    max_pool_seq_len: int,
) -> torch.Tensor:
    _validate_common_inputs(
        query,
        indexer_cache,
        weights,
        cum_query_lens,
        indexer_seq_lens,
        indexer_block_table,
        positions,
        index_topk,
        index_kpool,
        max_pool_seq_len,
    )
    if _can_use_triton(
        query,
        index_topk,
        index_kpool,
        max_pool_seq_len,
    ):
        assert glm5_next_lightning_indexer_triton is not None
        return glm5_next_lightning_indexer_triton(
            query,
            indexer_cache,
            weights,
            cum_query_lens,
            indexer_seq_lens,
            indexer_block_table,
            positions,
            index_topk=index_topk,
            index_kpool=index_kpool,
            max_pool_seq_len=max_pool_seq_len,
        )

    pool_topk = index_topk // index_kpool
    pool_ids = _pool_topk(
        query,
        indexer_cache,
        weights,
        cum_query_lens,
        indexer_seq_lens,
        indexer_block_table,
        positions,
        pool_topk,
        index_kpool,
        max_pool_seq_len,
    )
    expanded = _expand_pools_to_tokens(
        pool_ids,
        index_topk,
        index_kpool,
    )
    expanded = _append_tail_to_topk(
        expanded,
        positions,
        index_kpool,
    )
    return expanded.unsqueeze(1)


def _can_use_triton(
    query: torch.Tensor,
    index_topk: int,
    index_kpool: int,
    max_pool_seq_len: int,
) -> bool:
    if not HAS_TRITON or glm5_next_lightning_indexer_triton is None:
        return False
    if query.device.type != "npu":
        return False
    if query.shape[0] == 0 or max_pool_seq_len == 0:
        return False
    if query.shape[-1] != TRITON_HEAD_DIM:
        return False
    return index_topk // index_kpool <= TRITON_MAX_POOL_TOPK


def glm5_next_lightning_indexer_fake(
    query: torch.Tensor,
    indexer_cache: torch.Tensor,
    weights: torch.Tensor,
    cum_query_lens: torch.Tensor,
    indexer_seq_lens: torch.Tensor,
    indexer_block_table: torch.Tensor,
    positions: torch.Tensor,
    *,
    index_topk: int,
    index_kpool: int,
    max_pool_seq_len: int,
) -> torch.Tensor:
    del indexer_cache, weights, cum_query_lens, indexer_seq_lens, indexer_block_table, positions, max_pool_seq_len
    if index_topk <= 0 or index_kpool <= 0 or index_topk % index_kpool:
        raise ValueError(f"Invalid GLM5 Next indexer top-k/pool attrs: {index_topk}, {index_kpool}.")
    return torch.empty(
        (query.shape[0], 1, index_topk + index_kpool - 1),
        dtype=torch.int32,
        device=query.device,
    )


direct_register_custom_op(
    op_name="glm5_next_lightning_indexer",
    op_func=glm5_next_lightning_indexer,
    mutates_args=[],
    fake_impl=glm5_next_lightning_indexer_fake,
    dispatch_key="PrivateUse1",
)
