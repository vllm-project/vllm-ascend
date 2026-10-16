# SPDX-License-Identifier: Apache-2.0
"""Shared MX group alignment and scale pairing for linear and MoE weights."""

import torch
import torch.nn.functional as F


def get_mxfp_tp_padding(input_size: int, group_size: int, tp_rank: int) -> tuple[int, int]:
    """Return logical K padding that preserves a shard's global MX group phase.

    Callers decide whether the layer is input-sharded and convert the returned
    element counts to storage bytes when weights are packed.
    """
    prefix = (tp_rank * input_size) % group_size
    suffix = -(prefix + input_size) % group_size
    return prefix, suffix


def pair_mxfp_scales(scale: torch.Tensor) -> torch.Tensor:
    """Pair adjacent scale bytes on the last axis, padding an odd tail with zero.

    Preserve all leading dimensions and leave operator-specific transposes to
    the caller. Even, contiguous inputs only require a view.
    """
    if scale.shape[-1] % 2:
        scale = F.pad(scale, (0, 1))
    return scale.reshape(*scale.shape[:-1], scale.shape[-1] // 2, 2)
