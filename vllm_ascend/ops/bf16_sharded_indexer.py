# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

# Bound the FP32 gathered key and score tiles independently of context length.
DEFAULT_QUERY_ROWS_PER_TILE = 32
_KEY_TOKENS_PER_TILE = 2048


def bf16_paged_indexer_candidates(
    query: torch.Tensor,
    weights: torch.Tensor,
    key_cache: torch.Tensor,
    block_table: torch.Tensor,
    query_ends: torch.Tensor,
    local_visible_lengths: torch.Tensor,
    topk: int,
    *,
    first_valid_page: int = 1,
    full_graph: bool = False,
    query_rows_per_tile: int = DEFAULT_QUERY_ROWS_PER_TILE,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select paged, rank-local Lightning Indexer candidates with FP32 scores.

    The query and K cache contain the indexer's projected BF16 values.
    Query ends identify requests and local visible lengths encode causal
    visibility for each query row. Tiling uses tensor shapes only, including
    during graph replay. Equal scores prefer the earlier local token ID.
    """
    if query.dtype != torch.bfloat16 or key_cache.dtype != torch.bfloat16:
        raise ValueError("BF16 sharded indexer requires BF16 query and key cache.")
    rows, heads, head_dim = query.shape
    num_pages, page_size, kv_heads, key_head_dim = key_cache.shape
    if kv_heads != 1 or head_dim != key_head_dim or weights.shape != (rows, heads):
        raise ValueError("Invalid BF16 sharded indexer Q/K/weights geometry.")
    if (
        query_ends.ndim != 1
        or block_table.ndim != 2
        or block_table.shape[0] != query_ends.numel()
        or local_visible_lengths.numel() != rows
        or topk <= 0
    ):
        raise ValueError("Invalid BF16 sharded indexer row metadata or candidate count.")

    indices = torch.full((rows, 1, topk), -1, dtype=torch.int32, device=query.device)
    scores = torch.full((rows, 1, topk), -torch.inf, dtype=torch.float32, device=query.device)
    if not rows or not query_ends.numel() or not block_table.shape[1] or not num_pages:
        return indices, scores

    local_visible_lengths = local_visible_lengths.reshape(-1)
    query_rows = torch.arange(rows, dtype=query_ends.dtype, device=query.device)
    request_rows = torch.searchsorted(query_ends.contiguous(), query_rows, right=True, out_int32=True)
    valid_rows = request_rows < query_ends.numel()
    request_rows = request_rows.clamp_max(query_ends.numel() - 1)
    token_offsets = torch.arange(page_size, dtype=torch.int32, device=query.device)
    pages_per_tile = max(1, _KEY_TOKENS_PER_TILE // page_size)

    for row_start in range(0, rows, query_rows_per_tile):
        row_end = min(row_start + query_rows_per_tile, rows)
        row_slice = slice(row_start, row_end)
        query_tile = query[row_slice].float()
        weights_tile = weights[row_slice].float().unsqueeze(-1)
        visible = local_visible_lengths[row_slice, None]
        row_is_valid = valid_rows[row_slice, None]
        requests = request_rows[row_slice]
        best_ids = indices[row_slice, 0]
        best_scores = scores[row_slice, 0]
        score_tiles = []

        for page_start in range(0, block_table.shape[1], pages_per_tile):
            page_end = min(page_start + pages_per_tile, block_table.shape[1])
            pages = block_table[:, page_start:page_end].index_select(0, requests)
            valid_pages = (pages >= first_valid_page) & (pages < num_pages)
            safe_pages = torch.where(valid_pages, pages, 0).reshape(-1)
            keys = key_cache.index_select(0, safe_pages)
            keys = keys.reshape(row_end - row_start, -1, head_dim).float()
            page_starts = torch.arange(page_start, page_end, dtype=torch.int32, device=query.device) * page_size
            page_tokens = page_starts[:, None] + token_offsets
            valid_tokens = row_is_valid & ((page_tokens < visible.unsqueeze(-1)) & valid_pages.unsqueeze(-1)).flatten(1)
            # Preserve FP32 score accumulation under an outer BF16 autocast.
            with torch.autocast(device_type=query.device.type, enabled=False):
                tile_scores = torch.bmm(query_tile, keys.transpose(1, 2)).relu_()
            tile_scores.mul_(weights_tile)
            tile_scores = tile_scores.sum(dim=1).masked_fill_(~valid_tokens, -torch.inf)
            if full_graph:
                # Score tiles stay small, while a single stable sort over the
                # static page extent avoids recording one sort per page tile.
                score_tiles.append(tile_scores)
                continue
            tile_ids = page_tokens.flatten().expand_as(tile_scores).masked_fill(~valid_tokens, -1)

            # All IDs in the old winners precede the new tile's IDs.
            # Stable descending sort therefore preserves score/ID ordering.
            merged_scores = torch.cat((best_scores, tile_scores), dim=-1)
            merged_ids = torch.cat((best_ids, tile_ids), dim=-1)
            sorted_scores, order = torch.sort(merged_scores, dim=-1, descending=True, stable=True)
            best_scores = sorted_scores[:, :topk]
            best_ids = merged_ids.gather(-1, order[:, :topk])

        if full_graph:
            all_scores = torch.cat(score_tiles, dim=-1)
            sorted_scores, order = torch.sort(all_scores, dim=-1, descending=True, stable=True)
            width = min(topk, all_scores.shape[-1])
            best_scores = sorted_scores[:, :width]
            best_ids = order[:, :width].to(torch.int32).masked_fill(best_scores == -torch.inf, -1)
            indices[row_slice, 0, :width].copy_(best_ids)
            scores[row_slice, 0, :width].copy_(best_scores)
            continue

        indices[row_slice, 0].copy_(best_ids)
        scores[row_slice, 0].copy_(best_scores)

    return indices, scores
