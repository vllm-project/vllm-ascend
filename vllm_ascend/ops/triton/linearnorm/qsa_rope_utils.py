# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared QSA rotary-cache preparation utilities."""

from __future__ import annotations

import torch
from vllm.triton_utils import tl, triton


@triton.jit
def _prepare_qsa_rope_cache_kernel(
    positions_ptr,
    cache_ptr,
    output_ptr,
    position_stride_axis,
    position_stride_token,
    cache_stride_row,
    cache_stride_col,
    output_stride_row,
    num_tokens: tl.constexpr,
    cache_rows: tl.constexpr,
    cache_width: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    axis = row // num_tokens
    token = row % num_tokens
    position = tl.load(positions_ptr + axis * position_stride_axis + token * position_stride_token)
    position = tl.minimum(tl.maximum(position, 0), cache_rows - 1).to(tl.int64)
    offsets = tl.arange(0, BLOCK)
    mask = offsets < cache_width
    values = tl.load(
        cache_ptr + position * cache_stride_row + offsets * cache_stride_col,
        mask=mask,
        other=0.0,
    )
    tl.store(output_ptr + row * output_stride_row + offsets, values, mask=mask)


def validate_qsa_rope_positions(
    positions: torch.Tensor,
    cos_sin: torch.Tensor,
    qkv: torch.Tensor,
    rope_dim: int,
) -> None:
    """Validate inputs for gathering the full rotary cache in-kernel."""
    if positions.ndim not in (1, 2) or (positions.ndim == 2 and positions.shape[0] != 3):
        raise ValueError("QSA positions must have shape [tokens] or [3, tokens]")
    if positions.dtype not in (torch.int32, torch.int64):
        raise ValueError("QSA positions must use int32 or int64")
    if positions.shape[-1] != qkv.shape[0]:
        raise ValueError("QSA positions and QKV projection must have the same token count")
    if positions.device != qkv.device or cos_sin.device != qkv.device:
        raise ValueError("QSA positions, rotary cache and QKV projection must share a device")
    if cos_sin.ndim != 2 or cos_sin.shape[0] == 0 or cos_sin.shape[1] != rope_dim:
        raise ValueError("QSA full rotary cache must have shape [cache_rows, rope_dim]")
    if cos_sin.stride(1) != 1:
        raise ValueError("QSA full rotary cache must be contiguous in its last dimension")


def prepare_qsa_rope_cache(
    positions: torch.Tensor,
    cos_sin: torch.Tensor,
    output: torch.Tensor,
) -> None:
    """Clamp and gather QSA rotary rows directly into a persistent buffer."""
    if positions.ndim not in (1, 2) or (positions.ndim == 2 and positions.shape[0] != 3):
        raise ValueError("QSA positions must have shape [tokens] or [3, tokens]")
    if positions.dtype not in (torch.int32, torch.int64):
        raise ValueError("QSA positions must use int32 or int64")
    if cos_sin.ndim != 2 or cos_sin.shape[0] == 0 or cos_sin.stride(1) != 1:
        raise ValueError("QSA rotary cache must be a non-empty row-major matrix")
    expected_shape = (*positions.shape, cos_sin.shape[1])
    if output.shape != expected_shape or not output.is_contiguous():
        raise ValueError(f"QSA prepared rotary buffer must be contiguous with shape {expected_shape}")
    if positions.device != cos_sin.device or output.device != cos_sin.device:
        raise ValueError("QSA positions, rotary cache and output must share a device")
    if positions.numel() == 0:
        return

    num_tokens = positions.shape[-1]
    position_stride_axis = positions.stride(0) if positions.ndim == 2 else 0
    position_stride_token = positions.stride(-1)
    cache_width = cos_sin.shape[1]
    _prepare_qsa_rope_cache_kernel[(positions.numel(),)](
        positions,
        cos_sin,
        output,
        position_stride_axis,
        position_stride_token,
        cos_sin.stride(0),
        cos_sin.stride(1),
        output.stride(-2),
        num_tokens=num_tokens,
        cache_rows=cos_sin.shape[0],
        cache_width=cache_width,
        BLOCK=triton.next_power_of_2(cache_width),
    )


__all__ = ["prepare_qsa_rope_cache", "validate_qsa_rope_positions"]
