# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

"""Gather each Mamba group's physical state indices into its persistent output."""

from typing import TYPE_CHECKING

import torch
from vllm.triton_utils import tl, triton

if TYPE_CHECKING:
    from vllm.v1.worker.mamba_utils import MambaSpecDecodeGPUContext

# A 128-byte tile keeps stores aligned and limits how many group pointers it visits.
_TILE_SIZE = 32
_MAX_PROGRAMS = 65535


@triton.jit(do_not_specialize=["num_reqs"])
def _aligned_indices_kernel(
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
    TILES_PER_PROGRAM: tl.constexpr,
):
    offset_dtype: tl.constexpr = tl.int64 if NUM_GROUPS * MAX_REQS * STATE_SLOTS > 2**31 - BLOCK else tl.int32
    for tile in range(TILES_PER_PROGRAM):
        program_start = (tl.program_id(0) + tile * tl.num_programs(0)).to(offset_dtype) * BLOCK
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
            # Triton-Ascend requires a scalar table base for each load.
            table_base = tl.load(table_ptrs + group).to(tl.pointer_type(tl.int32))
            group_mask = valid & (groups == group)
            group_values = tl.load(
                table_base + rows.to(tl.int64) * table_stride + first_slot + slots, group_mask, other=0
            )
            values = tl.where(group_mask, group_values, values)
        tl.store(output + offsets, values, valid)


def compute_aligned_state_indices(
    self: "MambaSpecDecodeGPUContext", seq_lens: torch.Tensor, num_reqs: int
) -> torch.Tensor:
    """Gather the live rows of every group's state IDs into the context's output."""
    assert self.is_initialized and self.aligned_state_indices is not None
    output = self.aligned_state_indices
    # Newer contexts keep batch-ordered gather pointers separate from state-copy pointers.
    table_ptrs = getattr(self, "aligned_index_block_table_ptrs", self.block_table_ptrs)
    table_stride = self.block_table_stride_req
    cache_block_size = self.block_size
    assert output.ndim == 3 and output.dtype == torch.int32 and output.is_contiguous()
    assert output.data_ptr() % 32 == 0
    num_groups, max_reqs, state_slots = output.shape
    assert num_groups > 0 and state_slots > 0 and cache_block_size > 0
    assert 0 <= num_reqs <= min(max_reqs, seq_lens.shape[0])
    assert state_slots <= table_stride
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
    num_tiles = triton.cdiv(output.numel(), _TILE_SIZE)
    num_programs = min(num_tiles, _MAX_PROGRAMS)
    _aligned_indices_kernel[(num_programs,)](
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
        BLOCK=_TILE_SIZE,
        TILES_PER_PROGRAM=triton.cdiv(num_tiles, num_programs),
        multibuffer=False,
        unit_flag=False,
    )
    return output[:, :num_reqs]
