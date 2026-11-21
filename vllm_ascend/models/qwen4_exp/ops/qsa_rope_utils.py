# SPDX-License-Identifier: Apache-2.0

import torch


def validate_qsa_rope_positions(
    positions: torch.Tensor,
    cache: torch.Tensor,
    tensor: torch.Tensor,
    rope_dim: int,
) -> None:
    """Check the direct-gather ABI using metadata only (safe during capture)."""
    if positions.ndim not in (1, 2) or (positions.ndim == 2 and positions.shape[0] != 3):
        raise ValueError("QSA positions must have one or three axes")
    if positions.shape[-1] != tensor.shape[0] or positions.dtype not in (torch.int32, torch.int64):
        raise ValueError("QSA positions must be integer positions for every token")
    if cache.ndim != 2 or cache.shape[0] == 0 or cache.shape[1] != rope_dim or cache.stride(1) != 1:
        raise ValueError("QSA rotary cache must have nonempty, contiguous rotary rows")
    if tensor.dtype != torch.bfloat16 or cache.dtype != tensor.dtype:
        raise ValueError("QSA direct rotary gathering requires a BF16 cache and input")
    if positions.device != tensor.device or cache.device != tensor.device:
        raise ValueError("QSA positions, rotary cache and input must share a device")
