# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import torch


def remap_sparse_indices_pytorch(
    indices: torch.Tensor,
    output: torch.Tensor,
    dcp_size: int,
    dcp_rank: int,
    interleave_size: int,
) -> torch.Tensor:
    """PyTorch fallback for SFA DCP index remap."""
    if indices.dtype != torch.int32:
        raise TypeError(f"indices must have dtype int32, got {indices.dtype}")
    if not indices.is_contiguous():
        raise ValueError("indices must be contiguous")
    if output.shape != indices.shape or output.dtype != indices.dtype:
        raise ValueError("output must match indices shape and dtype")
    if not output.is_contiguous():
        raise ValueError("output must be contiguous")

    top_k = indices.shape[-1]
    blocks = torch.div(indices, interleave_size, rounding_mode="floor")
    is_local = (indices >= 0) & (blocks.remainder(dcp_size) == dcp_rank)
    remapped = (
        torch.div(
            indices,
            dcp_size * interleave_size,
            rounding_mode="floor",
        )
        * interleave_size
        + indices.remainder(interleave_size)
    )

    compact_positions = torch.cumsum(is_local.to(torch.int64), dim=-1) - 1
    tail_positions = torch.arange(
        top_k,
        dtype=torch.int64,
        device=indices.device,
    ).view((1,) * (indices.ndim - 1) + (top_k,)).expand_as(indices)
    scatter_positions = torch.where(is_local, compact_positions, tail_positions + top_k)
    scatter_values = torch.where(is_local, remapped, torch.full_like(remapped, -1))

    scratch = torch.full(
        (*indices.shape[:-1], top_k * 2),
        -1,
        dtype=indices.dtype,
        device=indices.device,
    )
    scratch.scatter_(-1, scatter_positions, scatter_values)
    output.copy_(scratch[..., :top_k])
    return output
