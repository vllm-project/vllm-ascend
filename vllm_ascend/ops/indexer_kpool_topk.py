# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pure-PyTorch GLM-5 kpool indexer scoring path.

Implements the kpool indexer scoring without the CANN ``pool_key_indexer``
wheel. All math is plain torch on the paged BF16 indexer K cache:

* ``cp_gather_indexer_k_cache`` — gather logical K rows from the paged cache
* ``bf16_mqa_logits``           — weighted multi-head MQA logits
* ``top_k_per_row_prefill``     — per-row causal top-k over chunk logits
* ``indexer_kpool_topk_pytorch``— full scoring + topk over the paged cache
"""

from __future__ import annotations

import torch

INDEXER_KPOOL_HEAD_DIM = 128
INDEXER_KPOOL_QUERY_CHUNK_SIZE = 16
INDEXER_KPOOL_KEY_CHUNK_SIZE = 2048


def cp_gather_indexer_k_cache(
    kv_cache: torch.Tensor,
    dst_k: torch.Tensor,
    block_table: torch.Tensor,
    cu_seq_lens: torch.Tensor,
) -> None:
    """Gather contiguous logical K rows from the paged BF16 indexer cache."""
    paged_k = kv_cache
    if paged_k.ndim != 4 or paged_k.shape[2] != 1:
        raise ValueError(f"Paged indexer K must be [blocks,block,1,dim], got {paged_k.shape}.")
    if paged_k.dtype != torch.bfloat16:
        raise TypeError(f"Paged indexer K must use bfloat16, got {paged_k.dtype}.")
    if dst_k.ndim != 2 or dst_k.shape[1] != paged_k.shape[-1]:
        raise ValueError(f"dst_k must be [total_k,{paged_k.shape[-1]}], got {dst_k.shape}.")
    if dst_k.dtype != torch.bfloat16:
        raise TypeError(f"dst_k must use bfloat16, got {dst_k.dtype}.")
    if block_table.ndim != 2:
        raise ValueError(f"block_table must be 2-D, got {block_table.shape}.")
    if cu_seq_lens.shape != (block_table.shape[0] + 1,):
        raise ValueError(
            "cu_seq_lens must contain one boundary per block-table row, "
            f"got {cu_seq_lens.shape} and {block_table.shape[0]} rows."
        )
    if block_table.shape[1] == 0 and dst_k.shape[0]:
        raise ValueError("block_table has no columns for a non-empty gather.")
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
    cache_block_size = paged_k.shape[1]
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
        max=paged_k.shape[0] - 1,
    )
    gathered_k = paged_k[
        safe_physical_blocks,
        page_offsets,
        0,
        :,
    ]
    dst_k.copy_(gathered_k)


def bf16_mqa_logits(
    query: torch.Tensor,
    key: torch.Tensor,
    weights: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
    clean_logits: bool,
) -> torch.Tensor:
    """Weighted multi-head MQA logits with BF16 Q/K."""
    if query.ndim != 3:
        raise ValueError(f"Indexer query must be [M,H,D], got {query.shape}.")
    if query.dtype != torch.bfloat16:
        raise TypeError(f"Indexer query must use bfloat16, got {query.dtype}.")
    if weights.shape != query.shape[:2]:
        raise ValueError(
            f"Indexer weights must be [M,H], got {weights.shape} for query {query.shape}."
        )
    if key.ndim != 2 or key.shape[1] != query.shape[2]:
        raise ValueError(f"Indexer K must be [N,D] with D={query.shape[2]}, got {key.shape}.")
    if key.dtype != torch.bfloat16:
        raise TypeError(f"Indexer K must use bfloat16, got {key.dtype}.")
    if cu_seqlen_ks.shape != (query.shape[0],) or cu_seqlen_ke.shape != cu_seqlen_ks.shape:
        raise ValueError(
            "cu_seqlen_ks/cu_seqlen_ke must have one entry per query, "
            f"got {cu_seqlen_ks.shape}, {cu_seqlen_ke.shape}, and M={query.shape[0]}."
        )

    # sum_h(weight_h * dot(q_h, k)) == dot(sum_h(weight_h * q_h), k)
    weighted_q = (query * weights.to(torch.bfloat16).unsqueeze(-1)).sum(dim=1)
    logits = torch.matmul(weighted_q, key.transpose(0, 1)).float()
    if clean_logits:
        columns = torch.arange(
            logits.shape[1],
            dtype=cu_seqlen_ks.dtype,
            device=logits.device,
        )
        valid = (columns[None, :] >= cu_seqlen_ks[:, None]) & (columns[None, :] < cu_seqlen_ke[:, None])
        logits = logits.masked_fill(
            ~valid,
            torch.finfo(logits.dtype).min,
        )
    return logits


def top_k_per_row_prefill(
    logits: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
    raw_topk_indices: torch.Tensor,
    num_rows: int,
    stride0: int,
    stride1: int,
    topk_tokens: int,
) -> None:
    """Per-row causal top-k with the upstream operator's interface."""
    if logits.ndim != 2:
        raise ValueError(f"Indexer logits must be 2-D, got {logits.shape}.")
    if num_rows != logits.shape[0]:
        raise ValueError(f"num_rows={num_rows} does not match logits rows {logits.shape[0]}.")
    if stride0 != logits.stride(0) or stride1 != logits.stride(1):
        raise ValueError(
            f"Explicit logits strides do not match the tensor: {(stride0, stride1)} vs {logits.stride()}."
        )
    if cu_seqlen_ks.shape != (num_rows,) or cu_seqlen_ke.shape != (num_rows,):
        raise ValueError("cu_seqlen_ks/cu_seqlen_ke must have one entry per logits row.")
    if raw_topk_indices.shape[0] < num_rows or raw_topk_indices.shape[1] < topk_tokens:
        raise ValueError(f"raw_topk_indices {raw_topk_indices.shape} cannot hold [{num_rows}, {topk_tokens}].")
    if topk_tokens <= 0:
        raise ValueError(f"topk_tokens must be positive, got {topk_tokens}.")

    columns = torch.arange(
        logits.shape[1],
        dtype=cu_seqlen_ks.dtype,
        device=logits.device,
    )
    valid = (columns[None, :] >= cu_seqlen_ks[:, None]) & (columns[None, :] < cu_seqlen_ke[:, None])
    scores = logits.masked_fill(
        ~valid,
        torch.finfo(logits.dtype).min,
    )
    if scores.shape[1] < topk_tokens:
        scores = torch.nn.functional.pad(
            scores,
            (0, topk_tokens - scores.shape[1]),
            value=torch.finfo(scores.dtype).min,
        )
    values, absolute_indices = torch.topk(
        scores,
        k=topk_tokens,
        dim=1,
        largest=True,
        sorted=False,
    )
    relative_indices = absolute_indices - cu_seqlen_ks[:, None]
    selected_valid = values != torch.finfo(scores.dtype).min
    relative_indices = torch.where(
        selected_valid,
        relative_indices,
        torch.full_like(relative_indices, -1),
    )
    raw_topk_indices[:num_rows, :topk_tokens].copy_(relative_indices.to(torch.int32))


def indexer_kpool_topk_pytorch(
    query: torch.Tensor,
    key: torch.Tensor,
    weights: torch.Tensor,
    actual_seq_lengths_query: torch.Tensor,
    actual_seq_lengths_key: torch.Tensor,
    block_table: torch.Tensor,
    query_positions: torch.Tensor,
    sparse_count: int,
    pool_size: int,
    max_key_seq_len: int,
    query_chunk_size: int = INDEXER_KPOOL_QUERY_CHUNK_SIZE,
    key_chunk_size: int = INDEXER_KPOOL_KEY_CHUNK_SIZE,
) -> torch.Tensor:
    """MQA logits + per-row top-k over the paged BF16 K cache."""
    if query.ndim != 3:
        raise ValueError(f"Indexer query must be [T,H,D], got {query.shape}.")
    if weights.shape != query.shape[:2]:
        raise ValueError(
            f"Indexer weights must match query T/H dimensions, got {weights.shape} and {query.shape}."
        )
    if key.ndim != 4 or key.shape[2] != 1:
        raise ValueError(f"Indexer K cache must be [blocks,block,1,dim], got {key.shape}.")
    if key.shape[-1] != query.shape[-1]:
        raise ValueError(f"Indexer query/K dims differ: {query.shape[-1]} and {key.shape[-1]}.")
    if query.dtype != torch.bfloat16 or key.dtype != torch.bfloat16:
        raise TypeError(f"Indexer query/K cache must both use bfloat16, got {query.dtype} and {key.dtype}.")
    if actual_seq_lengths_query.ndim != 1:
        raise ValueError("actual_seq_lengths_query must be cumulative 1-D.")
    if actual_seq_lengths_key.shape != actual_seq_lengths_query.shape:
        raise ValueError(
            "Indexer query/key sequence metadata must have the same "
            f"request count, got {actual_seq_lengths_query.shape} and "
            f"{actual_seq_lengths_key.shape}."
        )
    if block_table.ndim != 2 or block_table.shape[0] != actual_seq_lengths_key.shape[0]:
        raise ValueError(
            "Indexer block-table rows must match request count, got "
            f"{block_table.shape} and {actual_seq_lengths_key.shape}."
        )
    if query_positions.ndim != 1 or query_positions.shape[0] != query.shape[0]:
        raise ValueError(
            "Indexer positions must provide one value per query, got "
            f"{query_positions.shape} and {query.shape[0]} queries."
        )
    if sparse_count <= 0:
        raise ValueError(f"sparse_count must be positive, got {sparse_count}.")
    if pool_size <= 0:
        raise ValueError(f"pool_size must be positive, got {pool_size}.")
    if max_key_seq_len < 0:
        raise ValueError(f"max_key_seq_len must be non-negative, got {max_key_seq_len}.")
    if query_chunk_size <= 0 or key_chunk_size <= 0:
        raise ValueError(
            f"Indexer query/key chunk sizes must be positive, got {query_chunk_size} and {key_chunk_size}."
        )
    if block_table.shape[1] == 0 and max_key_seq_len:
        raise ValueError("Indexer block table has no columns for a non-empty cache.")

    output = torch.full(
        (query.shape[0], sparse_count),
        -1,
        dtype=torch.int32,
        device=query.device,
    )
    if query.shape[0] == 0 or max_key_seq_len == 0:
        return output

    query_ends = actual_seq_lengths_query
    token_ids = torch.arange(
        query.shape[0],
        dtype=query_ends.dtype,
        device=query.device,
    )
    request_ids = torch.bucketize(token_ids, query_ends, right=True)
    request_pool_lens = actual_seq_lengths_key[request_ids].to(torch.int64)
    causal_pool_lens = torch.div(
        query_positions.to(torch.int64) + 1,
        pool_size,
        rounding_mode="floor",
    )
    causal_pool_lens = torch.minimum(causal_pool_lens, request_pool_lens)

    cache_block_size = key.shape[1]
    key_chunk_size = max(
        cache_block_size,
        key_chunk_size // cache_block_size * cache_block_size,
    )
    score_mask_value = torch.finfo(torch.float32).min

    for query_start in range(0, query.shape[0], query_chunk_size):
        query_end = min(query_start + query_chunk_size, query.shape[0])
        chunk_query = query[query_start:query_end]
        chunk_weights = weights[query_start:query_end]
        chunk_request_ids = request_ids[query_start:query_end]
        chunk_pool_lens = causal_pool_lens[query_start:query_end]
        chunk_rows = query_end - query_start
        best_values = torch.full(
            (chunk_rows, sparse_count),
            score_mask_value,
            dtype=torch.float32,
            device=query.device,
        )
        best_indices = torch.full(
            (chunk_rows, sparse_count),
            -1,
            dtype=torch.int64,
            device=query.device,
        )

        for key_start in range(0, max_key_seq_len, key_chunk_size):
            key_end = min(key_start + key_chunk_size, max_key_seq_len)
            keys_per_row = key_end - key_start
            if key_start % cache_block_size:
                raise ValueError(
                    "Indexer key chunks must start on compressed cache "
                    f"block boundaries, got key_start={key_start} and "
                    f"block_size={cache_block_size}."
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
            chunk_block_table = block_table[
                chunk_request_ids,
                first_page:last_page,
            ]
            cp_gather_indexer_k_cache(
                key,
                gathered_key,
                chunk_block_table,
                gather_cu_seq_lens,
            )
            cu_seqlen_ks = gather_cu_seq_lens[:-1]
            valid_counts = (chunk_pool_lens - key_start).clamp(min=0, max=keys_per_row).to(torch.int32)
            cu_seqlen_ke = cu_seqlen_ks + valid_counts
            logits = bf16_mqa_logits(
                chunk_query,
                gathered_key,
                chunk_weights,
                cu_seqlen_ks,
                cu_seqlen_ke,
                clean_logits=False,
            )
            chunk_topk = torch.empty(
                (chunk_rows, sparse_count),
                dtype=torch.int32,
                device=query.device,
            )
            top_k_per_row_prefill(
                logits,
                cu_seqlen_ks,
                cu_seqlen_ke,
                chunk_topk,
                chunk_rows,
                logits.stride(0),
                logits.stride(1),
                sparse_count,
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
                k=sparse_count,
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


def scatter_paged_cache(
    cache: torch.Tensor,
    slots: torch.Tensor,
    values: torch.Tensor,
    block_size: int,
) -> None:
    """Scatter token rows into a paged cache, graph-safe (slot 0 sentinel)."""
    if cache.shape[1] != block_size:
        raise ValueError(
            f"Cache block size mismatch: expected {block_size}, got {cache.shape[1]}."
        )
    values = values.reshape(values.shape[0], *cache.shape[2:])
    valid = (slots >= 0) & (slots < cache.shape[0] * block_size)
    safe_slots = torch.where(valid, slots, torch.zeros_like(slots))
    block_ids = torch.div(
        safe_slots,
        block_size,
        rounding_mode="floor",
    )
    block_offsets = torch.remainder(safe_slots, block_size)
    row_mask = valid.view(-1, *([1] * (values.ndim - 1)))
    row_zero = cache[0, 0].clone()
    safe_values = torch.where(row_mask, values, row_zero.unsqueeze(0))
    row_zero_mask = valid & (slots == 0)
    update_zero = torch.where(
        row_zero_mask.view(-1, *([1] * (values.ndim - 1))),
        values,
        torch.zeros_like(values),
    ).sum(dim=0)
    expected_zero = torch.where(
        row_zero_mask.any(),
        update_zero,
        row_zero,
    )
    cache[block_ids, block_offsets] = safe_values
    cache[0, 0].copy_(expected_zero)


def gather_compressor_state(
    state_cache: torch.Tensor,
    state_block_table: torch.Tensor,
    state_block_size: int,
    end_positions: torch.Tensor,
    request_ids: torch.Tensor,
    index_kpool: int,
) -> torch.Tensor:
    """Gather the trailing ``index_kpool`` compressor states per token."""
    offsets = torch.arange(
        index_kpool - 1,
        -1,
        -1,
        device=end_positions.device,
    )
    logical = end_positions[:, None] - offsets[None, :]
    safe_logical = logical.clamp_min(0)
    pages = torch.div(
        safe_logical,
        state_block_size,
        rounding_mode="floor",
    ).clamp_max(state_block_table.shape[1] - 1)
    page_offsets = torch.remainder(safe_logical, state_block_size)
    physical_blocks = state_block_table[
        request_ids[:, None],
        pages,
    ].clamp(min=0, max=state_cache.shape[0] - 1)
    return state_cache[
        physical_blocks.long(),
        page_offsets,
    ]


def indexer_kpool_topk_static(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    weights: torch.Tensor,
    cum_query_lens: torch.Tensor,
    seq_lens: torch.Tensor,
    block_table: torch.Tensor,
    positions: torch.Tensor,
    sparse_count: int,
    pool_size: int,
) -> torch.Tensor:
    """Graph-capturable pool top-k over the paged BF16 K cache.

    Same math as ``indexer_kpool_topk_pytorch`` (weighted-q MQA logits +
    per-row causal bound) but with fully static shapes: no ``nonzero``, no
    ``.item()``, no data-dependent Python loops. Scores each query row
    against its request's pools in LOGICAL space by gathering the paged keys
    through the block table, so the top-k indices are logical pool ids just
    like the eager path. Intended for FULL-graph decode capture (small T);
    prefill keeps the chunked eager path (this version materializes
    [T, L, D] keys, too large for prefill T).
    """
    if query.ndim != 3:
        raise ValueError(f"Indexer query must be [T,H,D], got {query.shape}.")
    if key_cache.ndim != 4 or key_cache.shape[2] != 1:
        raise ValueError(f"Indexer K cache must be [blocks,block,1,dim], got {key_cache.shape}.")
    T = query.shape[0]
    cache_block = key_cache.shape[1]
    num_pools = key_cache.shape[0] * cache_block
    device = query.device

    weighted_q = (query * weights.to(query.dtype).unsqueeze(-1)).sum(dim=1)  # [T, D]

    token_ids = torch.arange(T, device=device)
    request_ids = torch.bucketize(token_ids, cum_query_lens, right=True).clamp_max(
        seq_lens.shape[0] - 1
    )
    bound = torch.minimum(
        torch.div(positions.to(torch.int64) + 1, pool_size, rounding_mode="floor"),
        seq_lens.to(torch.int64)[request_ids],
    )

    # Logical pool l of request r lives at physical pool
    # block_table[r, l // cache_block] * cache_block + l % cache_block.
    num_logical = block_table.shape[1] * cache_block
    ell = torch.arange(num_logical, device=device)
    phys = (
        block_table.to(torch.int64)[request_ids][:, ell // cache_block] * cache_block
        + (ell % cache_block)[None, :]
    ).clamp(0, num_pools - 1)  # [T, L]; out-of-table columns are masked below
    keys = key_cache.reshape(num_pools, query.shape[-1])
    keys_logical = keys[phys]  # [T, L, D]
    logits = torch.bmm(
        weighted_q.unsqueeze(1), keys_logical.transpose(1, 2)
    ).squeeze(1).float()  # [T, L]

    valid = ell[None, :] < bound[:, None]
    scores = logits.masked_fill(~valid, torch.finfo(logits.dtype).min)
    # aclnnTopk on this CANN caps k at 256; sort+slice has no such cap and is
    # graph-capturable. [T, L] with T <= max_num_seqs and L bounded by the
    # block table width, so the sort cost is negligible at decode widths.
    sorted_scores, sorted_idx = torch.sort(scores, dim=1, descending=True)
    idx = sorted_idx[:, :sparse_count]
    # Entries past the causal bound (or on rows with no pools yet) hold -inf
    # scores; report them as -1 exactly like the eager padding.
    out = torch.where(idx < bound[:, None], idx, -1).to(torch.int32)
    # Narrow block tables yield fewer than sparse_count logical columns;
    # pad to sparse_count so the width contract matches the eager path.
    if out.shape[1] < sparse_count:
        pad = torch.full(
            (out.shape[0], sparse_count - out.shape[1]),
            -1,
            dtype=out.dtype,
            device=device,
        )
        out = torch.cat([out, pad], dim=1)
    return out
