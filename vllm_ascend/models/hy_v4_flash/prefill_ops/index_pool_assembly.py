# SPDX-License-Identifier: Apache-2.0
"""Assemble pooled-key input rows and map each group to its first token."""

import torch
from vllm.triton_utils import tl, triton


@triton.jit(do_not_specialize=["First", "Past", "Current", "Rows", "Destination", "HistoryIndex"])
def _assemble_index_pool_groups(
    Keys,
    CacheKeys,
    CachePositions,
    BlockTable,
    Raw,
    GroupMap,
    HistoryPositions,
    First,
    Past,
    Current,
    Rows,
    Destination,
    HistoryIndex,
    PAGE_ROWS: tl.constexpr,
    KEY_ROW: tl.constexpr,
    POSITION_ROW: tl.constexpr,
    KEY_START: tl.constexpr,
    POSITION_START: tl.constexpr,
    AXES: tl.constexpr,
    WITH_HISTORY: tl.constexpr,
    TILE: tl.constexpr,
):
    tile = tl.program_id(0)
    row = tile * TILE + tl.arange(0, TILE)
    channel = tl.arange(0, 128)
    if WITH_HISTORY and tile == 0:
        is_history = row < Past
        token = First + row
        page = tl.load(BlockTable + token // PAGE_ROWS, is_history, other=0).to(tl.int64)
        slot = page * PAGE_ROWS + token % PAGE_ROWS
        history = tl.load(
            CacheKeys + slot[:, None] * KEY_ROW + KEY_START + channel[None, :],
            is_history[:, None],
            other=0,
        )
        current = tl.load(
            Keys + (Current + row[:, None] - Past) * 128 + channel[None, :],
            ((row < Rows) & ~is_history)[:, None],
            other=0,
        )
        key = tl.where(is_history[:, None], history, current)
        first_page = tl.load(BlockTable + First // PAGE_ROWS).to(tl.int64)
        first_slot = first_page * PAGE_ROWS + First % PAGE_ROWS
        axis = tl.arange(0, 1 if AXES == 1 else 4)
        coordinate = tl.load(
            CachePositions + first_slot * POSITION_ROW + POSITION_START + axis,
            axis < AXES,
            other=0,
        )
        tl.store(HistoryPositions + HistoryIndex * AXES + axis, coordinate, axis < AXES)
    else:
        key = tl.load(
            Keys + (Current + row[:, None] - Past) * 128 + channel[None, :],
            row[:, None] < Rows,
            other=0,
        )
    tl.store(Raw + (Destination + row[:, None]) * 128 + channel[None, :], key.to(tl.float32), row[:, None] < Rows)

    group = tile * (TILE // 4) + tl.arange(0, TILE // 4)
    first_row = group * 4
    # Negative entries address the compact history instead of the current batch.
    source = tl.where(first_row < Past, -1 - HistoryIndex, Current + first_row - Past)
    tl.store(GroupMap + Destination // 4 + group, source, first_row < Rows)


def assemble_index_pool_groups(cache, block_table, block_size, key_offset, keys, positions_ndim, segments):
    axes = 1 if positions_ndim == 1 else 3
    total_rows = sum(rows for _, _, _, _, rows in segments)
    history_count = sum(past > 0 for _, _, past, _, _ in segments)
    raw = torch.empty((total_rows, 128), dtype=torch.float32, device=keys.device)
    group_map = torch.empty(total_rows // 4, dtype=torch.int32, device=keys.device)
    history = torch.empty((history_count, axes), dtype=torch.int64, device=keys.device)
    key_cache = cache.view(torch.bfloat16)
    position_cache = cache.view(torch.int64)
    tile = 32
    destination = history_index = 0
    for request, first_token, past, first_new, rows in segments:
        _assemble_index_pool_groups[(triton.cdiv(rows, tile),)](
            keys,
            key_cache,
            position_cache,
            block_table[request],
            raw,
            group_map,
            history,
            first_token,
            past,
            first_new,
            rows,
            destination,
            history_index,
            block_size,
            key_cache.stride(0),
            position_cache.stride(0),
            key_offset // 2,
            (key_offset + 256) // 8,
            axes,
            past > 0,
            tile,
        )
        destination += rows
        history_index += past > 0
    return raw, group_map, history[:, 0] if axes == 1 else history.T
