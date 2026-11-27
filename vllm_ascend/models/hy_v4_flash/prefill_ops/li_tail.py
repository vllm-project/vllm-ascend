# SPDX-License-Identifier: Apache-2.0
"""Build the visible portion of each current four-token IndexPool group."""

import torch
from vllm.triton_utils import tl, triton


@triton.jit(do_not_specialize=["rows"])
def _build_li_tail(positions, output, rows, TILE: tl.constexpr):
    row = tl.program_id(0) * TILE + tl.arange(0, TILE)
    pos = tl.load(positions + row, row < rows, other=0).to(tl.int32)
    lane = tl.arange(0, 4)
    group_start = pos & -4
    candidates = group_start[:, None] + lane[None, :]
    tail = tl.where(candidates <= pos[:, None], candidates, -1)
    tl.store(output + row[:, None] * 4 + lane[None, :], tail, row[:, None] < rows)


def build_li_tail(positions):
    rows = positions.shape[0]
    output = torch.empty((rows, 4), dtype=torch.int32, device=positions.device)
    tile = 128
    _build_li_tail[(triton.cdiv(rows, tile),)](positions, output, rows, tile)
    return output


@triton.jit(do_not_specialize=["count", "sentinel"])
def _mask_group_ids(groups, output, count, sentinel, TILE: tl.constexpr):
    index = tl.program_id(0) * TILE + tl.arange(0, TILE)
    valid = index < count
    group = tl.load(groups + index, valid, other=-1)
    sortable = tl.where(group >= 0, group, sentinel)
    tl.store(output + index, sortable, valid)


def mask_group_ids(groups, sentinel, dtype):
    count = groups.numel()
    result = torch.empty_like(groups, dtype=dtype)
    tile = 4096
    _mask_group_ids[(triton.cdiv(count, tile),)](groups, result, count, sentinel, tile)
    return result


@triton.jit(do_not_specialize=["rows", "sentinel"])
def _expand_group_tile(groups, tail, output, rows, sentinel, ROW_TILE: tl.constexpr):
    row = tl.program_id(0) * ROW_TILE + tl.arange(0, ROW_TILE)
    column = tl.arange(0, 512)
    lane = tl.arange(0, 4)
    group = tl.load(
        groups + row[:, None] * 511 + column[None, :], (row[:, None] < rows) & (column[None, :] < 511), other=sentinel
    ).to(tl.int32)
    current = tl.load(tail + row[:, None] * 4 + lane[None, :], row[:, None] < rows, other=-1)
    history = tl.where(group[:, :, None] == sentinel, -1, group[:, :, None] * 4 + lane[None, None, :])
    index = tl.where(column[None, :, None] < 511, history, current[:, None, :])
    out = row[:, None, None] * 2048 + column[None, :, None] * 4 + lane[None, None, :]
    tl.store(output + out, index, row[:, None, None] < rows)


@triton.jit(do_not_specialize=["rows", "sentinel"])
def _expand_group_rows(groups, tail, output, rows, sentinel, TILE: tl.constexpr):
    row = tl.program_id(0) * TILE + tl.arange(0, TILE)
    col = tl.arange(0, 512)
    valid_row = row < rows
    history = col < 511
    valid = valid_row[:, None] & history[None, :]
    group = tl.load(groups + row[:, None] * 511 + col[None, :], valid, other=sentinel).to(tl.int32)
    start = group * 4
    has_group = group != sentinel
    tail0 = tl.load(tail + row * 4, valid_row, other=-1)
    tail1 = tl.load(tail + row * 4 + 1, valid_row, other=-1)
    tail2 = tl.load(tail + row * 4 + 2, valid_row, other=-1)
    tail3 = tl.load(tail + row * 4 + 3, valid_row, other=-1)
    p0 = tl.where(history[None, :], tl.where(has_group, start, -1), tail0[:, None])
    p1 = tl.where(history[None, :], tl.where(has_group, start + 1, -1), tail1[:, None])
    p2 = tl.where(history[None, :], tl.where(has_group, start + 2, -1), tail2[:, None])
    p3 = tl.where(history[None, :], tl.where(has_group, start + 3, -1), tail3[:, None])
    packed = tl.join(tl.join(p0, p2), tl.join(p1, p3)).reshape(TILE, 2048)
    lane = tl.arange(0, 2048)
    tl.store(output + row[:, None] * 2048 + lane[None, :], packed, valid_row[:, None])


def expand_sorted_group_ids(groups, sentinel, tail):
    rows = groups.shape[0]
    output = torch.empty((rows, 2048), dtype=torch.int32, device=groups.device)
    row_tile = 8
    kernel = _expand_group_rows if rows >= 768 else _expand_group_tile
    kernel[(triton.cdiv(rows, row_tile),)](groups, tail, output, rows, sentinel, row_tile)
    return output
