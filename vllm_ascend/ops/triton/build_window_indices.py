# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build fixed-width causal window indices in one NPU launch."""

from __future__ import annotations

import torch
from vllm.triton_utils import tl, triton


@triton.jit
def _build_window_indices_kernel(
    positions_ptr,
    indices_ptr,
    lengths_ptr,
    num_tokens,
    WINDOW_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    token = tl.program_id(0)
    columns = tl.arange(0, BLOCK_SIZE)
    valid_token = token < num_tokens
    position = tl.load(positions_ptr + token, mask=valid_token, other=-1)
    length = tl.minimum(position + 1, WINDOW_SIZE)
    start = position + 1 - length
    indices = tl.where(columns < length, start + columns, -1)
    tl.store(
        indices_ptr + token * WINDOW_SIZE + columns,
        indices,
        mask=valid_token & (columns < WINDOW_SIZE),
    )
    tl.store(lengths_ptr + token, length, mask=valid_token)


def build_window_indices_triton(
    positions: torch.Tensor,
    window_size: int,
    *,
    indices_output: torch.Tensor | None = None,
    lengths_output: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return causal indices ``[T,1,W]`` and lengths ``[T,1]`` as INT32."""
    if positions.ndim != 1:
        raise ValueError(f"positions must be rank 1, got {tuple(positions.shape)}")
    if window_size <= 0:
        raise ValueError(f"window_size must be positive, got {window_size}")
    tokens = positions.shape[0]
    expected_indices = (tokens, 1, window_size)
    expected_lengths = (tokens, 1)
    if indices_output is None:
        indices_output = torch.empty(
            expected_indices,
            dtype=torch.int32,
            device=positions.device,
        )
    elif tuple(indices_output.shape) != expected_indices or indices_output.dtype != torch.int32:
        raise ValueError(
            f"indices_output must be INT32{expected_indices}, got {indices_output.dtype}{tuple(indices_output.shape)}"
        )
    if lengths_output is None:
        lengths_output = torch.empty(
            expected_lengths,
            dtype=torch.int32,
            device=positions.device,
        )
    elif tuple(lengths_output.shape) != expected_lengths or lengths_output.dtype != torch.int32:
        raise ValueError(
            f"lengths_output must be INT32{expected_lengths}, got {lengths_output.dtype}{tuple(lengths_output.shape)}"
        )
    if tokens:
        block_size = triton.next_power_of_2(window_size)
        _build_window_indices_kernel[(tokens,)](
            positions.contiguous(),
            indices_output,
            lengths_output,
            tokens,
            WINDOW_SIZE=window_size,
            BLOCK_SIZE=block_size,
        )
    return indices_output, lengths_output
