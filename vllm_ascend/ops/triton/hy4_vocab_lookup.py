# SPDX-License-Identifier: Apache-2.0
"""Prepare rank-local embedding indices for eager HY4 tensor parallelism."""

import torch
from vllm.triton_utils import tl, triton


@triton.jit
def rank_local_vocab_index(token, org_start, org_end, added_start, added_end, added_local, invalid_index):
    original = (token >= org_start) & (token < org_end)
    added = (token >= added_start) & (token < added_end)
    slot = tl.where(original, token - org_start, token - added_start + added_local)
    valid = original | added
    return tl.where(valid, slot, invalid_index), valid


@triton.jit(do_not_specialize=["rows", "org_start", "org_end", "added_start", "added_end", "added_local"])
def _hy4_vocab_indices(
    tokens,
    indices,
    padding,
    rows,
    org_start,
    org_end,
    added_start,
    added_end,
    added_local,
    invalid_index,
    TILE: tl.constexpr,
    WRITE_MASK: tl.constexpr,
):
    row = tl.program_id(0) * TILE + tl.arange(0, TILE)
    token = tl.load(tokens + row, row < rows, other=-1)
    index, valid = rank_local_vocab_index(
        token,
        org_start,
        org_end,
        added_start,
        added_end,
        added_local,
        invalid_index,
    )
    tl.store(indices + row, index, row < rows)
    if WRITE_MASK:
        tl.store(padding + row, ~valid, row < rows)


def vocab_indices(tokens, shard, *, padding_index=None):
    rows = tokens.numel()
    write_mask = padding_index is None
    indices = torch.empty_like(tokens, dtype=torch.int64 if write_mask else torch.int32)
    padding = torch.empty_like(tokens, dtype=torch.bool) if write_mask else None
    added_local = shard.org_vocab_end_index - shard.org_vocab_start_index + shard.num_org_vocab_padding
    tile = 512
    _hy4_vocab_indices[(triton.cdiv(rows, tile),)](
        tokens,
        indices,
        padding,
        rows,
        shard.org_vocab_start_index,
        shard.org_vocab_end_index,
        shard.added_vocab_start_index,
        shard.added_vocab_end_index,
        added_local,
        0 if write_mask else padding_index,
        tile,
        write_mask,
    )
    return indices, padding
