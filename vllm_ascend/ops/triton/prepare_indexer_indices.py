# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Convert score-ordered QLI indices into visible, chronological positions."""

import math

import torch
from vllm.triton_utils import tl, triton

from vllm_ascend.ops.triton.triton_utils import get_vectorcore_num


@triton.jit(do_not_specialize=["num_rows", "blocks_per_core"])
def _prepare_indexer_indices_kernel(
    selected_ptr,
    positions_ptr,
    output_ptr,
    num_rows,
    TOPK: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    blocks_per_core,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_COLS: tl.constexpr,
    SENTINEL: tl.constexpr,
    SORT_KEY_SHIFT: tl.constexpr,
    NEGATIVE_KEY_BASE: tl.constexpr,
):
    first_block = tl.program_id(0) * blocks_per_core
    last_block = tl.minimum(first_block + blocks_per_core, tl.cdiv(num_rows, BLOCK_ROWS))
    columns = tl.arange(0, BLOCK_COLS)
    for block in range(first_block, last_block):
        rows = block * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
        offsets = rows[:, None] * TOPK + columns[None, :]
        mask = (rows[:, None] < num_rows) & (columns[None, :] < TOPK)
        selected = tl.load(selected_ptr + offsets, mask, other=SENTINEL)
        positions = tl.load(positions_ptr + rows, rows < num_rows, other=-1)
        visible = (positions + 1) // COMPRESS_RATIO
        valid = (selected >= 0) & (selected < visible[:, None])
        selected = tl.where(valid, selected, SENTINEL)
        # Use native FP32 sort even when the INT32 implementation is absent.
        # Encode ordered, normal FP32 bit patterns without numeric conversion:
        # [0, 2**24) uses negative keys; larger indices use positive keys.
        key_bits = tl.where(selected < 2 * SORT_KEY_SHIFT, NEGATIVE_KEY_BASE - selected, selected - SORT_KEY_SHIFT)
        sort_keys = key_bits.to(tl.float32, bitcast=True)
        sort_keys = tl.extra.cann.extension.sort(sort_keys, dim=1, descending=False)
        key_bits = sort_keys.to(tl.int32, bitcast=True)
        selected = tl.where(key_bits < 0, NEGATIVE_KEY_BASE - key_bits, key_bits + SORT_KEY_SHIFT)
        selected = tl.where(selected == SENTINEL, -1, selected)
        tl.store(output_ptr + offsets, selected, mask)


def prepare_indexer_indices(selected: torch.Tensor, positions: torch.Tensor, compress_ratio: int) -> torch.Tensor:
    """Filter and sort [tokens, topk] INT32 indices, with invalid slots last."""
    assert selected.ndim == 2 and selected.dtype == torch.int32
    assert positions.ndim == 1 and positions.shape[0] == selected.shape[0]
    assert positions.dtype in (torch.int32, torch.int64)
    assert compress_ratio in (1, 2)
    num_rows, topk = selected.shape
    assert 1 <= topk <= 2048
    selected = selected.contiguous()
    positions = positions.contiguous()
    output = torch.empty_like(selected)
    if num_rows == 0:
        return output

    num_cores = get_vectorcore_num()
    block_cols = triton.next_power_of_2(topk)
    # Budget 128 KiB for INT32 sort data and scratch (eight buffers).
    max_block_rows = 128 * 1024 // (block_cols * 4 * 8)
    # Core boundaries must also align to 32 bytes for arbitrary TopK widths.
    aligned_rows = 8 // math.gcd(topk, 8)
    block_rows = min(max_block_rows, triton.next_power_of_2(triton.cdiv(num_rows, num_cores)))
    num_blocks = triton.cdiv(num_rows, block_rows)
    aligned_blocks = triton.cdiv(aligned_rows, block_rows)
    grid = min(triton.cdiv(num_blocks, aligned_blocks), num_cores)
    blocks_per_core = triton.cdiv(triton.cdiv(num_blocks, grid), aligned_blocks) * aligned_blocks
    _prepare_indexer_indices_kernel[(grid,)](
        selected,
        positions,
        output,
        num_rows,
        TOPK=topk,
        COMPRESS_RATIO=compress_ratio,
        blocks_per_core=blocks_per_core,
        BLOCK_ROWS=block_rows,
        BLOCK_COLS=block_cols,
        SENTINEL=torch.iinfo(torch.int32).max,
        SORT_KEY_SHIFT=1 << 23,
        NEGATIVE_KEY_BASE=0x81800000 - (1 << 32),
        multibuffer=False,
        unit_flag=False,
    )
    return output
