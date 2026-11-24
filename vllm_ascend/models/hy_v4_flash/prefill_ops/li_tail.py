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
def _mask_group_ids_to_float(groups, output, count, sentinel, TILE: tl.constexpr):
    index = tl.program_id(0) * TILE + tl.arange(0, TILE)
    valid = index < count
    group = tl.load(groups + index, valid, other=-1)
    sortable = tl.where(group >= 0, group, sentinel).to(tl.float32)
    tl.store(output + index, sortable, valid)


def mask_group_ids_to_float(groups, sentinel):
    count = groups.numel()
    result = torch.empty_like(groups, dtype=torch.float32)
    tile = 4096
    _mask_group_ids_to_float[(triton.cdiv(count, tile),)](groups, result, count, sentinel, tile)
    return result
