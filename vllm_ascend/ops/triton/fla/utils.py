# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Songlin Yang, Yu Zhang
#
# This file contains code copied from the flash-linear-attention project.
# The original source code was licensed under the MIT license and included
# the following copyright notice:
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
# ruff: noqa: E501
import contextlib
import functools
from collections.abc import Callable

import torch
from vllm.triton_utils import tl, triton


def prepare_lens(cu_seqlens: torch.LongTensor) -> torch.LongTensor:
    return cu_seqlens[1:] - cu_seqlens[:-1]


def prepare_chunk_indices(cu_seqlens: torch.LongTensor, chunk_size: int) -> torch.LongTensor:
    indices = torch.cat([torch.arange(n) for n in triton.cdiv(prepare_lens(cu_seqlens), chunk_size).tolist()])
    return torch.stack([indices.eq(0).cumsum(0) - 1, indices], 1).to(cu_seqlens)


def prepare_chunk_indices_dev(
    cu_seqlens: torch.LongTensor,
    total_tokens: int,
    chunk_size: int,
) -> torch.LongTensor:
    """Device-side equivalent of :func:`prepare_chunk_indices`.

    ``prepare_chunk_indices`` ends with ``.tolist()``, i.e. a device->host synchronisation. On a
    serving hot path that stalls the NPU pipeline once per kernel launch, and it makes the
    surrounding op impossible to capture into an ACLGraph (host synchronisation is forbidden
    during capture). This variant never copies anything back to the host.

    The row count must be known on the host, so it is the shape-derived upper bound
    ``total_tokens // chunk_size + num_seqs`` instead of the exact
    ``sum(ceil(len_s / chunk_size))``. Rows past the real chunk count are phantom rows: their
    ``seq_id`` clamps to 0 while ``chunk_in_seq`` runs past the end of the first sequence, so
    every lane they address is fully masked and they store nothing (grid-only over-allocation,
    see the per-kernel masking audit in the PR description).

    Zero-length segments are not supported by either helper: ``prepare_chunk_indices`` numbers
    its ``seq_id`` over the non-empty sequences only, which does not match ``cu_seqlens``
    indexing, so callers have to drop empty segments first (the prefill path already does,
    see ``_compact_empty_segments``).

    Args:
        cu_seqlens: ``[num_seqs + 1]`` cumulative sequence lengths, on device.
        total_tokens: number of tokens in the packed batch, i.e. ``cu_seqlens[-1]``.
        chunk_size: chunk size the indices are built for.

    Returns:
        ``[total_chunks, 2]`` int tensor of ``(seq_id, chunk_in_seq)`` with the same dtype as
        ``cu_seqlens``. Its first ``sum(ceil(len_s / chunk_size))`` rows are element-wise
        identical to :func:`prepare_chunk_indices`.
    """
    num_seqs = cu_seqlens.numel() - 1
    if num_seqs <= 0:
        return torch.zeros((0, 2), dtype=cu_seqlens.dtype, device=cu_seqlens.device)
    total_chunks = total_tokens // chunk_size + num_seqs
    lens = (cu_seqlens[1:] - cu_seqlens[:-1]).to(torch.int32)
    chunks_per_seq = (lens + (chunk_size - 1)) // chunk_size
    chunk_ends = torch.cumsum(chunks_per_seq, 0)
    row_ids = torch.arange(total_chunks, dtype=torch.int32, device=cu_seqlens.device)
    seq_ids = torch.searchsorted(chunk_ends, row_ids, right=True)
    seq_ids = torch.where(seq_ids >= num_seqs, torch.zeros_like(seq_ids), seq_ids)
    chunk_in_seq = row_ids - (chunk_ends - chunks_per_seq)[seq_ids]
    return torch.stack((seq_ids, chunk_in_seq), dim=1).to(cu_seqlens)


def prepare_final_chunk_indices(cu_seqlens: torch.LongTensor, chunk_size: int) -> torch.LongTensor:
    indices = triton.cdiv(prepare_lens(cu_seqlens), chunk_size) + 1
    return torch.cumsum(indices, 0) - 1


def prepare_chunk_offsets(cu_seqlens: torch.LongTensor, chunk_size: int) -> torch.LongTensor:
    return torch.cat([cu_seqlens.new_tensor([0]), triton.cdiv(prepare_lens(cu_seqlens), chunk_size)]).cumsum(-1)


def prepare_update_chunk_offsets(cu_seqlens: torch.LongTensor, chunk_size: int) -> torch.LongTensor:
    return torch.cat([cu_seqlens.new_tensor([0]), triton.cdiv(prepare_lens(cu_seqlens), chunk_size) + 1]).cumsum(-1)


def input_guard(fn: Callable[..., torch.Tensor]) -> Callable[..., torch.Tensor]:
    """
    A decorator to make sure all input tensors are contiguous and set the device based on input tensors.
    """

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        contiguous_args = (i if not isinstance(i, torch.Tensor) else i.contiguous() for i in args)
        contiguous_kwargs = {k: (v if not isinstance(v, torch.Tensor) else v.contiguous()) for k, v in kwargs.items()}

        tensor = None
        for arg in args:
            if isinstance(arg, torch.Tensor):
                tensor = arg
                break
        if tensor is None:
            for value in kwargs.values():
                if isinstance(value, torch.Tensor):
                    tensor = value
                    break

        if tensor is not None:
            ctx = torch.npu.device(tensor.device.index)
        else:
            ctx = contextlib.nullcontext()

        with ctx:
            return fn(*contiguous_args, **contiguous_kwargs)

    return wrapper


@triton.jit
def safe_exp(x):
    return tl.exp(tl.where(x <= 0, x, float("-inf")))


@triton.jit(do_not_specialize=["inner_size", "row_stride"])
def _clear_ssm_states_kernel(
    states_ptr,
    has_initial_state_ptr,
    inner_size,
    row_stride,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(axis=0)
    col_block_idx = tl.program_id(axis=1)

    has_state = tl.load(has_initial_state_ptr + row_idx).to(tl.int1)
    if has_state:
        return

    cols = col_block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = cols < inner_size
    row_ptr = states_ptr + row_idx * row_stride + cols
    tl.store(row_ptr, tl.zeros((BLOCK_SIZE,), dtype=states_ptr.dtype.element_ty), mask=mask)


def clear_ssm_states(ssm_states: torch.Tensor, has_initial_state: torch.Tensor) -> None:
    """Zero out specific rows for the SSM states

    Args:
        ssm_states (torch.Tensor): input SSM states
        has_initial_state (torch.Tensor): indicates whether the row has initial states already
    """
    if ssm_states.numel() == 0:
        return

    if has_initial_state.device != ssm_states.device:
        has_initial_state = has_initial_state.to(ssm_states.device, non_blocking=True)
    if has_initial_state.dtype != torch.bool:
        has_initial_state = has_initial_state.to(torch.bool)

    has_initial_state = has_initial_state.reshape(-1).contiguous()
    num_rows = ssm_states.shape[0]
    if num_rows == 0:
        return
    if has_initial_state.numel() != num_rows:
        raise ValueError(
            f"clear_ssm_states: has_initial_state size mismatch: expected {num_rows}, got {has_initial_state.numel()}"
        )
    inner_size = ssm_states.numel() // num_rows
    if inner_size == 0:
        return

    block_size = 4096
    grid = (num_rows, triton.cdiv(inner_size, block_size))
    _clear_ssm_states_kernel[grid](
        ssm_states,
        has_initial_state,
        inner_size,
        ssm_states.stride(0),
        BLOCK_SIZE=block_size,
    )
