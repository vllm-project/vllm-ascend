# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tensor expansion for LoRA outputs narrower than the native kernel rank."""

import torch


def expand_slice(
    output: torch.Tensor,
    inputs: torch.Tensor,
    weights: torch.Tensor,
    indices: torch.Tensor,
    offset: int,
    width: int,
    add_inputs: bool,
) -> None:
    """Accumulate in float32 and leave inactive adapter rows unchanged."""
    weights = weights.view(weights.shape[0], -1, weights.shape[-1])
    selected = weights.index_select(0, indices.clamp_min(0))
    delta = torch.bmm(selected.float(), inputs.float().unsqueeze(-1)).squeeze(-1)
    target = output[:, offset : offset + width]
    result = target + delta if add_inputs else delta
    target.copy_(torch.where(indices[:, None] >= 0, result, target))
