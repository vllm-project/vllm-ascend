# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Maintain the 8-token fused K+scale view required by A5 QSLI."""

import torch
from vllm.triton_utils import tl, triton

FOLDED_GROUP_ROWS = 8
INDEX_KEY_BYTES = 64
INDEX_SCALE_BYTES = 4
FOLDED_ROW_BYTES = FOLDED_GROUP_ROWS * (INDEX_KEY_BYTES + INDEX_SCALE_BYTES)


@triton.jit
def _fold_indexer_rows(
    slots_ptr,
    key_ptr,
    scale_ptr,
    folded_ptr,
    key_page_stride,
    key_row_stride,
    scale_page_stride,
    scale_row_stride,
    folded_page_stride,
    folded_row_stride,
    FOLDED_GROUP_ROWS: tl.constexpr,
    INDEX_KEY_BYTES: tl.constexpr,
    INDEX_SCALE_BYTES: tl.constexpr,
):
    token = tl.program_id(0)
    page = tl.load(slots_ptr + token * 2).to(tl.int64)
    row = tl.load(slots_ptr + token * 2 + 1).to(tl.int64)
    valid = (page >= 0) & (row >= 0)
    group = row // FOLDED_GROUP_ROWS
    within = row % FOLDED_GROUP_ROWS
    key_offset = tl.arange(0, INDEX_KEY_BYTES)
    scale_offset = tl.arange(0, INDEX_SCALE_BYTES)
    key = tl.load(
        key_ptr + page * key_page_stride + row * key_row_stride + key_offset,
        mask=valid,
        other=0,
    )
    scale = tl.load(
        scale_ptr + page * scale_page_stride + row * scale_row_stride + scale_offset,
        mask=valid,
        other=0,
    )
    destination = folded_ptr + page * folded_page_stride + group * folded_row_stride
    tl.store(destination + within * INDEX_KEY_BYTES + key_offset, key, mask=valid)
    tl.store(
        destination + FOLDED_GROUP_ROWS * INDEX_KEY_BYTES + within * INDEX_SCALE_BYTES + scale_offset,
        scale,
        mask=valid,
    )


def fold_indexer_cache_rows(
    cache: tuple[torch.Tensor, torch.Tensor],
    folded: torch.Tensor,
    slots: torch.Tensor,
) -> None:
    """Copy freshly written token rows into their candidate-block group."""
    key, scale = cache
    if slots.ndim != 2 or slots.shape[1] != 2 or slots.dtype != torch.int32:
        raise ValueError("folded A5 index cache requires int32 [T,2] page/row slots")
    if not slots.is_contiguous():
        raise ValueError("folded A5 index cache requires contiguous slots")
    if key.shape[-1] != INDEX_KEY_BYTES or scale.shape[-1] != INDEX_SCALE_BYTES:
        raise ValueError("A5 index cache must contain 64 key and 4 scale bytes per token")
    if key.shape[1] % FOLDED_GROUP_ROWS:
        raise ValueError("A5 index page size must be divisible by eight")
    if folded.shape != (key.shape[0], key.shape[1] // FOLDED_GROUP_ROWS, 1, FOLDED_ROW_BYTES):
        raise ValueError("A5 folded index cache has incompatible shape")
    if slots.shape[0] == 0:
        return
    _fold_indexer_rows[(slots.shape[0],)](
        slots,
        key,
        scale,
        folded,
        key.stride(0),
        key.stride(1),
        scale.stride(0),
        scale.stride(1),
        folded.stride(0),
        folded.stride(1),
        FOLDED_GROUP_ROWS,
        INDEX_KEY_BYTES,
        INDEX_SCALE_BYTES,
    )
