# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from __future__ import annotations

import math

import torch
from vllm.triton_utils import tl, triton

STATE_IO_BLOCK_SIZE = 1024


@triton.jit
def _gather_ssm_states_kernel(
    state_ptr,
    indices_ptr,
    has_initial_state_ptr,
    output_ptr,
    stride_state_batch: tl.int64,
    stride_indices,
    stride_has_initial_state,
    row_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    block_idx = tl.program_id(0)
    batch_idx = tl.program_id(1)
    offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < row_size

    has_initial_state = tl.load(has_initial_state_ptr + batch_idx * stride_has_initial_state).to(tl.int1)
    state_idx = tl.load(
        indices_ptr + batch_idx * stride_indices,
        mask=has_initial_state,
        other=0,
    ).to(tl.int64)

    values = tl.load(
        state_ptr + state_idx * stride_state_batch + offsets,
        mask=mask & has_initial_state,
        other=0.0,
    )
    tl.store(output_ptr + batch_idx * row_size + offsets, values, mask=mask)


@triton.jit
def _scatter_ssm_states_kernel(
    state_ptr,
    indices_ptr,
    source_ptr,
    stride_state_batch: tl.int64,
    stride_indices,
    stride_source_batch: tl.int64,
    row_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    block_idx = tl.program_id(0)
    batch_idx = tl.program_id(1)
    offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < row_size

    state_idx = tl.load(indices_ptr + batch_idx * stride_indices).to(tl.int64)
    values = tl.load(
        source_ptr + batch_idx * stride_source_batch + offsets,
        mask=mask,
    )
    tl.store(
        state_ptr + state_idx * stride_state_batch + offsets,
        values,
        mask=mask,
    )


def _validate_state_and_indices(
    state: torch.Tensor,
    indices: torch.Tensor,
) -> None:
    if state.ndim < 2:
        raise ValueError(f"state must have at least 2 dimensions, got {state.ndim}")
    if state.shape[0] == 0:
        raise ValueError("state must contain at least one row")
    if indices.ndim != 1:
        raise ValueError(f"indices must be 1D, got shape={tuple(indices.shape)}")
    if indices.dtype not in (torch.int32, torch.int64):
        raise TypeError(f"indices must be int32 or int64, got {indices.dtype}")
    if indices.device != state.device:
        raise ValueError("indices and state must be on the same device")
    row_size = math.prod(state.shape[1:])
    if state.stride(0) < row_size:
        raise ValueError("state stride(0) must be at least the flattened row size")
    if not state[0].is_contiguous():
        raise ValueError("each state row must be contiguous")


def _validate_gather_inputs(
    state: torch.Tensor,
    indices: torch.Tensor,
    has_initial_state: torch.Tensor,
) -> None:
    _validate_state_and_indices(state, indices)
    if has_initial_state.ndim != 1 or has_initial_state.shape != indices.shape:
        raise ValueError("has_initial_state must be 1D and match indices")
    if has_initial_state.dtype != torch.bool:
        raise TypeError("has_initial_state must have bool dtype")
    if has_initial_state.device != state.device:
        raise ValueError("has_initial_state and state must be on the same device")


def _validate_scatter_inputs(
    state: torch.Tensor,
    indices: torch.Tensor,
    source: torch.Tensor,
) -> None:
    _validate_state_and_indices(state, indices)
    expected_shape = (indices.numel(), *state.shape[1:])
    if source.shape != expected_shape:
        raise ValueError(f"source shape must be {expected_shape}, got {tuple(source.shape)}")
    if source.device != state.device:
        raise ValueError("source and state must be on the same device")
    if source.numel() > 0 and not source[0].is_contiguous():
        raise ValueError("each source row must be contiguous")


def gather_ssm_states(
    state: torch.Tensor,
    indices: torch.Tensor,
    has_initial_state: torch.Tensor,
    *,
    output_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Gather selected rows from a state cache with a padded batch stride.

    Unlike ``state[indices]`` or ``index_select``, this kernel reads only the
    requested rows. ``has_initial_state=False`` produces a zero row without
    reading the state cache.

    ``state`` has shape ``[num_state_rows, *inner_dims]``. The inner
    dimensions must be contiguous; only ``state.stride(0)`` may include
    physical page padding. ``indices`` select physical state rows and may be
    unsorted or non-contiguous. Callers must provide valid row IDs. The
    optional ``output_dtype`` only changes the gathered output dtype.
    """
    _validate_gather_inputs(state, indices, has_initial_state)

    output = torch.empty(
        (indices.numel(), *state.shape[1:]),
        dtype=state.dtype if output_dtype is None else output_dtype,
        device=state.device,
    )
    if indices.numel() == 0:
        return output

    row_size = math.prod(state.shape[1:])
    grid = (triton.cdiv(row_size, STATE_IO_BLOCK_SIZE), indices.numel())
    _gather_ssm_states_kernel[grid](
        state,
        indices,
        has_initial_state,
        output,
        state.stride(0),
        indices.stride(0),
        has_initial_state.stride(0),
        row_size=row_size,
        BLOCK_SIZE=STATE_IO_BLOCK_SIZE,
        num_warps=8,
    )
    return output


def scatter_ssm_states_(
    state: torch.Tensor,
    indices: torch.Tensor,
    source: torch.Tensor,
) -> None:
    """Write dense state rows into selected slots of a padded state cache.

    The state layout follows :func:`gather_ssm_states`: inner dimensions are
    contiguous and padding is represented by ``state.stride(0)``. ``indices``
    select physical rows and may be unsorted or non-contiguous. ``source``
    must have the logical state shape and contiguous inner dimensions.
    """
    _validate_scatter_inputs(state, indices, source)
    if indices.numel() == 0:
        return

    row_size = math.prod(state.shape[1:])
    grid = (triton.cdiv(row_size, STATE_IO_BLOCK_SIZE), indices.numel())
    _scatter_ssm_states_kernel[grid](
        state,
        indices,
        source,
        state.stride(0),
        indices.stride(0),
        source.stride(0),
        row_size=row_size,
        BLOCK_SIZE=STATE_IO_BLOCK_SIZE,
        num_warps=8,
    )
