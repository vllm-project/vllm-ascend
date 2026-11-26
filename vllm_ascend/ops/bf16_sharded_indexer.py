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
    request_rows: torch.Tensor | None = None,
    first_valid_page: int = 1,
    full_graph: bool = False,
    query_rows_per_tile: int = DEFAULT_QUERY_ROWS_PER_TILE,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select paged, rank-local Lightning Indexer candidates with FP32 scores.

    The query and K cache contain the indexer's projected BF16 values.
    Query ends identify requests and local visible lengths encode causal
    visibility for each query row. Tiling uses tensor shapes only, including
    during graph replay. Equal scores prefer the earlier local token ID.
    The metadata builder can supply one safe request index per row, paired
    with a zero visible length for every graph-padding row.
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
        or (request_rows is not None and request_rows.shape != (rows,))
        or topk <= 0
        or query_rows_per_tile <= 0
    ):
        raise ValueError("Invalid BF16 sharded indexer row metadata or candidate count.")

    indices = torch.empty((rows, 1, topk), dtype=torch.int32, device=query.device)
    scores = torch.empty((rows, 1, topk), dtype=torch.float32, device=query.device)
    if not rows or not query_ends.numel() or not block_table.shape[1] or not num_pages:
        return indices.fill_(-1), scores.fill_(-torch.inf)

    visible_lengths = local_visible_lengths.reshape(-1)
    if request_rows is None:
        query_rows = torch.arange(rows, dtype=query_ends.dtype, device=query.device)
        request_rows = torch.searchsorted(query_ends.contiguous(), query_rows, right=True, out_int32=True)
        visible_lengths = torch.where(request_rows < query_ends.numel(), visible_lengths, 0)
        request_rows = request_rows.clamp_max(query_ends.numel() - 1)
    num_columns = block_table.shape[1]
    token_ids = torch.arange(num_columns * page_size, dtype=torch.int32, device=query.device)
    width = min(topk, num_columns * page_size)
    if width < topk:
        indices[:, :, width:].fill_(-1)
        scores[:, :, width:].fill_(-torch.inf)
    pages_per_tile = min(num_columns, max(1, _KEY_TOKENS_PER_TILE // page_size))
    page_padding = -num_columns % pages_per_tile
    if page_padding:
        block_table = torch.nn.functional.pad(block_table, (0, page_padding))
    # Gather page IDs in tile-major order so a complete tile can be passed
    # directly to index_select on the cache with a contiguous 1-D index.
    table_tiles = block_table.view(block_table.shape[0], -1, pages_per_tile).transpose(0, 1)

    for row_start in range(0, rows, query_rows_per_tile):
        row_end = min(row_start + query_rows_per_tile, rows)
        row_slice = slice(row_start, row_end)
        query_tile = query[row_slice].float()
        weights_tile = weights[row_slice].float().unsqueeze(-1)
        visible = visible_lengths[row_slice, None]
        # Resolve the small page table once for all key tiles in these rows.
        # Only the paged K payload and scores need token-wise tiling.
        pages = table_tiles.index_select(1, request_rows[row_slice])
        valid_pages = (pages >= first_valid_page) & (pages < num_pages)
        safe_pages = torch.where(valid_pages, pages, 0)
        previous = None
        score_tiles = []

        for tile, page_start in enumerate(range(0, num_columns, pages_per_tile)):
            page_end = min(page_start + pages_per_tile, num_columns)
            page_count = page_end - page_start
            keys = key_cache.index_select(0, safe_pages[tile, :, :page_count].reshape(-1))
            keys = keys.reshape(row_end - row_start, -1, head_dim).float()
            tile_ids = token_ids[page_start * page_size : page_end * page_size]
            # Preserve FP32 score accumulation under an outer BF16 autocast.
            with torch.autocast(device_type=query.device.type, enabled=False):
                tile_scores = torch.bmm(query_tile, keys.transpose(1, 2)).relu_()
            tile_scores.mul_(weights_tile)
            tile_scores = tile_scores.sum(dim=1)
            if full_graph:
                # Score tiles stay small, while a single stable sort over the
                # static page extent avoids recording one sort per page tile.
                score_tiles.append(tile_scores)
                continue
            valid_tokens = (tile_ids < visible).view(row_end - row_start, -1, page_size)
            valid_tokens &= valid_pages[tile, :, :page_count, None]
            tile_scores.masked_fill_(~valid_tokens.flatten(1), -torch.inf)
            if previous is not None:
                # Earlier tiles precede new IDs when scores tie.
                previous_ids, previous_scores = previous
                tile_ids = torch.cat((previous_ids, tile_ids.expand_as(tile_scores)), dim=-1)
                tile_scores = torch.cat((previous_scores, tile_scores), dim=-1)
            sorted_scores, order = torch.sort(tile_scores, dim=-1, descending=True, stable=True)
            best_scores = sorted_scores[:, :topk]
            best_ids = tile_ids.gather(-1, order[:, :topk]) if page_start else order[:, :topk].to(torch.int32)
            previous = best_ids, best_scores

        if full_graph:
            all_scores = score_tiles[0] if len(score_tiles) == 1 else torch.cat(score_tiles, dim=-1)
            # All scores are resident for the single graph sort. Mask once
            # per query tile, in the same logical page order as the scores.
            page_mask = valid_pages.transpose(0, 1).reshape(row_end - row_start, -1)[:, :num_columns]
            invalid_tokens = (token_ids >= visible).view(row_end - row_start, num_columns, page_size)
            invalid_tokens.logical_or_(~page_mask[:, :, None])
            all_scores.masked_fill_(invalid_tokens.flatten(1), -torch.inf)
            sorted_scores, order = torch.sort(all_scores, dim=-1, descending=True, stable=True)
            best_scores = sorted_scores[:, :topk]
            best_ids = order[:, :topk].to(torch.int32)

        indices[row_slice, 0, :width].copy_(best_ids.masked_fill_(best_scores == -torch.inf, -1))
        scores[row_slice, 0, :width].copy_(best_scores)

    return indices, scores
