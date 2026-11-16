# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Prepare paged metadata for noncausal DSpARK FlashMLA."""

from typing import TYPE_CHECKING

import torch
import triton
import triton.language as tl

if TYPE_CHECKING:
    from vllm_ascend.attention.flashmla import FlashMLADecode


@triton.jit(do_not_specialize=["num_reqs", "num_tokens", "num_slots", "num_positions"])
def _prepare_draft(
    starts,
    seq_lens,
    in_table,
    in_slots,
    in_positions,
    cu,
    used_q,
    cache_lens,
    block_table,
    slots,
    positions,
    token_live,
    window_used,
    window_lens,
    window_table,
    num_reqs,
    num_tokens,
    num_slots,
    num_positions,
    TOKENS: tl.constexpr,
    ROWS: tl.constexpr,
    COLS: tl.constexpr,
    TABLE_STRIDE: tl.constexpr,
    WINDOW: tl.constexpr,
    WINDOW_COLS: tl.constexpr,
    PAGE: tl.constexpr,
    SEARCH_STEPS: tl.constexpr,
    REQ_TILE: tl.constexpr,
    TABLE_TILE: tl.constexpr,
    TOKEN_TILE: tl.constexpr,
    WINDOW_TILE: tl.constexpr,
):
    task = tl.program_id(0)
    req_tiles: tl.constexpr = (ROWS + REQ_TILE) // REQ_TILE
    table_tiles: tl.constexpr = (ROWS * COLS + TABLE_TILE - 1) // TABLE_TILE

    if task < req_tiles:
        request_row = task * REQ_TILE + tl.arange(0, REQ_TILE)
        req_start = tl.load(starts + request_row, request_row <= num_reqs, TOKENS).to(tl.int32)
        req_start = tl.where(request_row <= num_reqs, tl.minimum(req_start, num_tokens), TOKENS)
        req_end = tl.load(starts + request_row + 1, request_row < num_reqs, 0).to(tl.int32)
        req_length = tl.load(seq_lens + request_row, request_row < num_reqs, 0).to(tl.int32)
        used = tl.where(req_length > 0, tl.minimum(req_end, num_tokens) - req_start, 0)
        tl.store(cu + request_row, req_start, request_row <= ROWS)
        tl.store(used_q + request_row, used, request_row < ROWS)
        tl.store(cache_lens + request_row, tl.where(used != 0, req_length, 0), request_row < ROWS)
    elif task < req_tiles + table_tiles:
        offset = (task - req_tiles) * TABLE_TILE + tl.arange(0, TABLE_TILE)
        table_req = offset // COLS
        column = offset % COLS
        base_page_id = tl.load(
            in_table + table_req * TABLE_STRIDE + column, (table_req < num_reqs) & (offset < ROWS * COLS), 0
        )
        tl.store(block_table + offset, base_page_id, offset < ROWS * COLS)
    else:
        token = (task - req_tiles - table_tiles) * TOKEN_TILE + tl.arange(0, TOKEN_TILE)
        valid_token = token < TOKENS
        # Upper bound over real query ends. Equal or empty ranges map to the
        # next request, just as searchsorted(right=True) does.
        lower = tl.full((TOKEN_TILE,), 0, tl.int32)
        upper = lower + num_reqs
        for _ in tl.range(SEARCH_STEPS):
            middle = (lower + upper) // 2
            end = tl.load(starts + middle + 1, valid_token & (middle < num_reqs), 0).to(tl.int32)
            advance = (middle < num_reqs) & (tl.minimum(end, num_tokens) <= token)
            lower = tl.where(advance, middle + 1, lower)
            upper = tl.where(advance, upper, middle)
        req = lower
        active_req = valid_token & (req < num_reqs)
        length = tl.load(seq_lens + req, active_req, 0).to(tl.int32)
        start = tl.load(starts + req, active_req, 0).to(tl.int32)
        live = active_req & (length > 0) & (token >= start) & (token < num_tokens)
        slot = tl.load(in_slots + token, live & (token < num_slots), -1)
        pos = tl.load(in_positions + token, valid_token & (token < num_positions), 0)
        tl.store(slots + token, slot, valid_token)
        tl.store(positions + token, pos, valid_token)
        tl.store(token_live + token, live, valid_token)
        tl.store(window_used + token, live, valid_token)

        first_page = tl.maximum(length - WINDOW, 0) // PAGE
        effective = tl.where(live, length - first_page * PAGE, 0)
        tl.store(window_lens + token, effective, valid_token)
        page = tl.arange(0, WINDOW_TILE)
        source_column = tl.minimum(first_page[:, None] + page[None, :], COLS - 1)
        output_mask = valid_token[:, None] & (page[None, :] < WINDOW_COLS)
        window_page_id = tl.load(
            in_table + req[:, None] * TABLE_STRIDE + source_column, active_req[:, None] & output_mask, 0
        )
        tl.store(window_table + token[:, None] * WINDOW_COLS + page[None, :], window_page_id, output_mask)


def try_prepare_flashmla_draft(
    flash: "FlashMLADecode", common, num_reqs: int, num_tokens: int, window: int, page_size: int
) -> bool:
    """Write the existing graph buffers, falling back for unsupported layouts."""
    draft = flash.draft_window
    tokens = flash.slots.numel()
    rows, cols = flash.block_table.shape
    if (
        draft is None
        or flash.slots.device.type != "npu"
        or not 0 < num_reqs < rows
        or not 0 < num_tokens <= tokens
        or cols == 0
    ):
        return False
    starts, lengths, table, slots, positions = (
        common.query_start_loc,
        common.seq_lens,
        common.block_table_tensor,
        common.slot_mapping,
        common.positions,
    )
    if (
        starts.numel() <= num_reqs
        or lengths.numel() < num_reqs
        or table.shape[0] < num_reqs
        or table.shape[1] != cols
        or table.stride(1) != 1
        or table.dtype != torch.int32
        or table.device != flash.slots.device
        or not all(
            t.ndim == 1
            and t.is_contiguous()
            and t.device == flash.slots.device
            and t.dtype in (torch.int32, torch.int64)
            for t in (starts, lengths, slots, positions)
        )
    ):
        return False
    # Keep parallel writes on 32-byte boundaries, including bool token_live
    # and the variable-width rows of the window block table.
    req_tile, table_tile, token_tile = 32, 512, 32
    window_cols = draft.block_table.shape[1]
    grid = (triton.cdiv(rows + 1, req_tile) + triton.cdiv(rows * cols, table_tile) + triton.cdiv(tokens, token_tile),)
    _prepare_draft[grid](
        starts,
        lengths,
        table,
        slots,
        positions,
        flash.cu,
        flash.used_q,
        flash.cache_lens,
        flash.block_table,
        flash.slots,
        flash.positions,
        flash.token_live,
        draft.used_q,
        draft.cache_lens,
        draft.block_table,
        num_reqs,
        num_tokens,
        min(num_tokens, slots.numel()),
        min(num_tokens, positions.numel()),
        tokens,
        rows,
        cols,
        table.stride(0),
        window,
        window_cols,
        page_size,
        num_reqs.bit_length(),
        req_tile,
        table_tile,
        token_tile,
        triton.next_power_of_2(window_cols),
    )
    return True
