# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

"""Triton kernels used by DeepSeek-V4 Compressor sequence parallelism."""

import torch
from vllm.triton_utils import HAS_TRITON, tl, triton

_MAX_LINEAR_OFFSET = (1 << 31) - 1
_MAX_GRID_ROWS = 65535
_UB_BUDGET_BYTES = 192 * 1024 // 2


@triton.jit
def _compressor_state_gather_kernel(
    state_ptr,
    block_indices_ptr,
    offset_indices_ptr,
    output_ptr,
    state_block_stride,
    state_offset_stride,
    output_row_stride,
    STATE_DIM: tl.constexpr,
    BLOCK_DIM: tl.constexpr,
):
    """Gather one state row for each physical ``(block, offset)`` pair."""
    row = tl.program_id(0)
    block = tl.load(block_indices_ptr + row).to(tl.int32)
    offset = tl.load(offset_indices_ptr + row).to(tl.int32)
    cols = tl.arange(0, BLOCK_DIM)
    mask = cols < STATE_DIM

    state_row = block * state_block_stride + offset * state_offset_stride
    values = tl.load(state_ptr + state_row + cols, mask=mask)
    tl.store(output_ptr + row * output_row_stride + cols, values, mask=mask)


def can_use_triton_compressor_state_gather(
    state: torch.Tensor,
    block_indices: torch.Tensor,
    offset_indices: torch.Tensor,
    output: torch.Tensor,
) -> bool:
    """Return whether the state rows can use the direct Triton gather."""
    if not HAS_TRITON:
        return False
    if state.ndim != 3 or output.ndim != 2:
        return False
    if block_indices.ndim != 1 or offset_indices.ndim != 1:
        return False
    if block_indices.shape != offset_indices.shape:
        return False
    if output.shape != (block_indices.shape[0], state.shape[-1]):
        return False
    if state.dtype != torch.float32 or output.dtype != state.dtype:
        return False
    if block_indices.dtype not in (torch.int32, torch.int64):
        return False
    if offset_indices.dtype not in (torch.int32, torch.int64):
        return False
    if block_indices.stride(0) != 1 or offset_indices.stride(0) != 1:
        return False
    if state.stride(-1) != 1 or output.stride(-1) != 1:
        return False
    if block_indices.shape[0] > _MAX_GRID_ROWS:
        return False

    max_linear_offset = (
        (state.shape[0] - 1) * state.stride(0) + (state.shape[1] - 1) * state.stride(1) + state.shape[2] - 1
    )
    if max_linear_offset > _MAX_LINEAR_OFFSET:
        return False

    block_dim = 1
    while block_dim < state.shape[-1]:
        block_dim <<= 1
    return block_dim * state.element_size() <= _UB_BUDGET_BYTES


def triton_compressor_state_gather(
    state: torch.Tensor,
    block_indices: torch.Tensor,
    offset_indices: torch.Tensor,
    output: torch.Tensor,
) -> None:
    """Gather state rows directly into the fixed SP communication buffer."""
    num_rows = block_indices.shape[0]
    if num_rows == 0:
        return

    state_dim = state.shape[-1]
    block_dim = 1
    while block_dim < state_dim:
        block_dim <<= 1
    _compressor_state_gather_kernel[(num_rows,)](
        state,
        block_indices,
        offset_indices,
        output,
        state.stride(0),
        state.stride(1),
        output.stride(0),
        STATE_DIM=state_dim,
        BLOCK_DIM=block_dim,
    )
