# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

"""NPU gather of graph-stable, per-group Mamba physical state indices.

Uses the contiguous GM load followed by UB gather from Triton-Ascend's
``third_party/ascend/tutorials/10-gather-2d-simd.py``. Since state slots are
consecutive, only a bounded column window is needed instead of the full table.
"""

import math

import torch
from vllm.triton_utils import tl, triton

from vllm_ascend.ops.triton.triton_utils import get_vectorcore_num, init_device_properties_triton

_ALIGN_ELEMENTS = 8  # 32 bytes of INT32.
_FLAT_BLOCK = 256
_MAX_WINDOW_ELEMENTS = 4096


@triton.jit(do_not_specialize=["num_reqs"])
def _aligned_indices_window_kernel(
    table_ptrs,
    seq_lens,
    output,
    table_stride: tl.int64,
    seq_stride: tl.constexpr,
    output_group_stride: tl.constexpr,
    output_row_stride: tl.constexpr,
    num_reqs,
    TABLE_COLS: tl.constexpr,
    CACHE_BLOCK_SIZE: tl.constexpr,
    STATE_SLOTS: tl.constexpr,
    BLOCK_SLOTS: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    WINDOW: tl.constexpr,
    TILES_PER_PROGRAM: tl.constexpr,
):
    group = tl.program_id(0)
    # A scalar base per group, cast outside loops for Ascend AxisInfo analysis.
    table_base = tl.load(table_ptrs + group).to(tl.pointer_type(tl.int32))
    columns = tl.arange(0, WINDOW)
    slots = tl.arange(0, BLOCK_SLOTS)
    first_tile = tl.program_id(1) * TILES_PER_PROGRAM
    last_tile = tl.minimum(first_tile + TILES_PER_PROGRAM, tl.cdiv(num_reqs, BLOCK_ROWS))
    for tile in range(first_tile, last_tile):
        rows = tile * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
        valid_rows = rows < num_reqs
        # Masked tl.load/store lanes do not access GM; tail offsets stay unclamped.
        if BLOCK_ROWS == 1:
            # A true scalar keeps max/div out of Ascend's single-element
            # tensor pointer analysis, which does not support tensor maxsi.
            lengths = tl.load(seq_lens + tile * seq_stride, tile < num_reqs, other=1)
        else:
            lengths = tl.load(seq_lens + rows * seq_stride, valid_rows, other=1)
        # Preserve the length dtype: Ascend BlockPtrAnalysis cannot parse an
        # int64-to-int32 truncation in GM pointer offsets.
        first_slot = tl.maximum((lengths - 1) // CACHE_BLOCK_SIZE, 0)
        window_start = first_slot // 8 * 8
        if BLOCK_ROWS == 1:
            source_columns = window_start + columns[None, :]
            local_indices = first_slot - window_start + tl.minimum(slots[None, :], STATE_SLOTS - 1)
        else:
            source_columns = window_start[:, None] + columns[None, :]
            local_indices = first_slot[:, None] - window_start[:, None] + tl.minimum(slots[None, :], STATE_SLOTS - 1)
        # Only a small contiguous window enters UB, rather than the whole table.
        source = tl.load(
            table_base + rows[:, None].to(tl.int64) * table_stride + source_columns,
            valid_rows[:, None] & (source_columns < TABLE_COLS),
            other=0,
        )
        # Masked output lanes must also have valid UB gather indices.
        local_indices = local_indices.to(tl.int32)
        # Ascend gather supports fp32 but not int32; preserve physical ID bits.
        values = tl.gather(source.to(tl.float32, bitcast=True), local_indices, axis=1).to(tl.int32, bitcast=True)
        tl.store(
            output + group * output_group_stride + rows[:, None] * output_row_stride + slots[None, :],
            values,
            valid_rows[:, None] & (slots[None, :] < STATE_SLOTS),
        )


@triton.jit(do_not_specialize=["num_reqs"])
def _aligned_indices_flat_kernel(
    table_ptrs,
    seq_lens,
    output,
    table_stride: tl.int64,
    seq_stride: tl.constexpr,
    num_reqs,
    MAX_REQS: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    STATE_SLOTS: tl.constexpr,
    CACHE_BLOCK_SIZE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # Whole 32-byte output spans avoid cross-core partial-write overlap when
    # max_reqs * state_slots leaves adjacent group bases unaligned.
    program_start = tl.program_id(0) * BLOCK
    offsets = program_start + tl.arange(0, BLOCK)
    groups = offsets // (MAX_REQS * STATE_SLOTS)
    rows = offsets // STATE_SLOTS % MAX_REQS
    slots = offsets % STATE_SLOTS
    valid = (groups < NUM_GROUPS) & (rows < num_reqs)
    lengths = tl.load(seq_lens + rows * seq_stride, valid, other=1)
    first_slot = tl.maximum((lengths - 1) // CACHE_BLOCK_SIZE, 0).to(tl.int32)
    first_group = program_start // (MAX_REQS * STATE_SLOTS)
    last_group = tl.minimum(tl.cdiv(program_start + BLOCK, MAX_REQS * STATE_SLOTS), NUM_GROUPS)
    values = tl.full((BLOCK,), 0, tl.int32)
    for group in range(first_group, last_group):
        # A vector int-to-pointer base aborts Ascend OffsetAnalysis. Keep the
        # base scalar and visit only groups intersecting this program's output.
        table_base = tl.load(table_ptrs + group).to(tl.pointer_type(tl.int32))
        group_mask = valid & (groups == group)
        group_values = tl.load(table_base + rows.to(tl.int64) * table_stride + first_slot + slots, group_mask, other=0)
        values = tl.where(group_mask, group_values, values)
    # One store retains disjoint 32-byte ownership across group boundaries.
    tl.store(output + offsets, values, valid)


def _aligned_index_launch_config(num_groups, num_reqs, max_reqs, state_slots, num_cores):
    """Select small UB tiles and disjoint 32-byte output regions on the CPU."""
    window = triton.next_power_of_2(state_slots + _ALIGN_ELEMENTS - 1)
    aligned_rows = _ALIGN_ELEMENTS // math.gcd(state_slots, _ALIGN_ELEMENTS)
    max_rows = min(32, _MAX_WINDOW_ELEMENTS // window)
    if max_reqs * state_slots % _ALIGN_ELEMENTS or max_rows < aligned_rows:
        return None
    rows = triton.next_power_of_2(triton.cdiv(num_reqs * num_groups, num_cores))
    block_rows = max(aligned_rows, min(rows, max_rows))
    num_tiles = triton.cdiv(num_reqs, block_rows)
    row_programs = min(num_tiles, max(1, num_cores // num_groups))
    tiles_per_program = triton.cdiv(num_tiles, row_programs)
    return (num_groups, row_programs), block_rows, window, tiles_per_program


def compute_aligned_state_indices(
    table_ptrs: torch.Tensor,
    seq_lens: torch.Tensor,
    output: torch.Tensor,
    num_reqs: int,
    table_stride: int,
    table_cols: int,
    cache_block_size: int,
) -> torch.Tensor:
    """Gather each group's state slots in one launch, into existing storage.

    Inputs and output must share a device. Table columns are contiguous; rows
    may have physical padding. Callers guarantee that each real request's
    selected state slots fit within table_cols. Inactive output rows stay intact.
    """
    assert output.ndim == 3 and output.dtype == torch.int32 and output.is_contiguous()
    assert output.data_ptr() % 32 == 0
    num_groups, max_reqs, state_slots = output.shape
    assert num_groups > 0 and state_slots > 0 and cache_block_size > 0
    assert 0 <= num_reqs <= min(max_reqs, seq_lens.shape[0])
    assert state_slots <= table_cols <= table_stride
    assert seq_lens.ndim == 1 and seq_lens.dtype in (torch.int32, torch.int64)
    assert (
        table_ptrs.ndim == 1
        and table_ptrs.shape[0] == num_groups
        and table_ptrs.dtype == torch.int64
        and table_ptrs.is_contiguous()
    )
    assert table_ptrs.device == seq_lens.device == output.device
    if num_reqs == 0:
        return output[:, :0]
    init_device_properties_triton()
    config = _aligned_index_launch_config(num_groups, num_reqs, max_reqs, state_slots, get_vectorcore_num())
    if config is None:
        grid = (triton.cdiv(output.numel(), _FLAT_BLOCK),)
        assert grid[0] <= 65535
        _aligned_indices_flat_kernel[grid](
            table_ptrs,
            seq_lens,
            output,
            table_stride,
            seq_lens.stride(0),
            num_reqs,
            MAX_REQS=max_reqs,
            NUM_GROUPS=num_groups,
            STATE_SLOTS=state_slots,
            CACHE_BLOCK_SIZE=cache_block_size,
            BLOCK=_FLAT_BLOCK,
            multibuffer=False,
            unit_flag=False,
        )
    else:
        grid, block_rows, window, tiles_per_program = config
        assert math.prod(grid) <= 65535
        _aligned_indices_window_kernel[grid](
            table_ptrs,
            seq_lens,
            output,
            table_stride,
            seq_lens.stride(0),
            output.stride(0),
            output.stride(1),
            num_reqs,
            TABLE_COLS=table_cols,
            CACHE_BLOCK_SIZE=cache_block_size,
            STATE_SLOTS=state_slots,
            BLOCK_SLOTS=triton.next_power_of_2(state_slots),
            BLOCK_ROWS=block_rows,
            WINDOW=window,
            TILES_PER_PROGRAM=tiles_per_program,
            multibuffer=False,
            unit_flag=False,
        )
    return output[:, :num_reqs]
