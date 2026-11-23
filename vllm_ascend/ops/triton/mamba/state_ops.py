# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Recurrent-state gather that preserves padded cache storage on Ascend."""

import math

import torch
from vllm.triton_utils import tl, triton

STATE_COPY_BLOCK_BYTES = 16384


@triton.jit
def _gather_initial_states_kernel(
    state,
    output,
    indices,
    has_initial_state,
    stride_state_batch,
    stride_indices,
    stride_has_initial_state,
    ROW_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(1)
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    has_state = tl.load(has_initial_state + row * stride_has_initial_state).to(tl.int1)
    # Mask the index load too: fresh rows may carry invalid sentinel indices.
    source_row = tl.load(indices + row * stride_indices, mask=has_state, other=0).to(tl.int64)
    values = tl.load(
        state + source_row * stride_state_batch + offsets,
        mask=has_state & (offsets < ROW_SIZE),
        other=0,
    )
    tl.store(output + row * ROW_SIZE + offsets, values, offsets < ROW_SIZE)


def gather_initial_states(state: torch.Tensor, indices: torch.Tensor, has_initial_state: torch.Tensor) -> torch.Tensor:
    """Read selected rows, zeroing fresh sequences without reading their cache."""
    if state.device.type != "npu":
        idx = indices.to(torch.int64) * has_initial_state.to(torch.int64)
        out = state.index_select(0, idx)
        keep = has_initial_state.view([-1] + [1] * (state.dim() - 1)).to(torch.bool)
        return torch.where(keep, out, 0)
    output = torch.empty((indices.numel(), *state.shape[1:]), dtype=state.dtype, device=state.device)
    row_elements = math.prod(state.shape[1:])
    if indices.numel() == 0 or row_elements == 0:
        return output
    assert state.ndim >= 2 and state[0].is_contiguous()
    assert indices.ndim == has_initial_state.ndim == 1
    assert indices.shape == has_initial_state.shape
    assert indices.device == has_initial_state.device == state.device
    block_size = min(triton.next_power_of_2(row_elements), STATE_COPY_BLOCK_BYTES // state.element_size())
    _gather_initial_states_kernel[(triton.cdiv(row_elements, block_size), indices.numel())](
        state,
        output,
        indices,
        has_initial_state,
        state.stride(0),
        indices.stride(0),
        has_initial_state.stride(0),
        row_elements,
        block_size,
    )
    return output
