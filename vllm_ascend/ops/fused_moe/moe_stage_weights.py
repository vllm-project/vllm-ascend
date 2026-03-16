from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

import torch

MoEWeightTensor: TypeAlias = torch.Tensor | list[torch.Tensor]
MoEOptionalQuantTensor: TypeAlias = list[torch.Tensor] | torch.Tensor | None


@dataclass(frozen=True, slots=True)
class MoEWeights:
    """Dense and quantized weight payloads consumed by MoE execution."""

    w1: MoEWeightTensor
    w2: MoEWeightTensor
    w1_bias: torch.Tensor | None = None
    w2_bias: torch.Tensor | None = None
    w1_scale: MoEOptionalQuantTensor = None
    w2_scale: MoEOptionalQuantTensor = None
    w1_scale_bias: torch.Tensor | None = None
    w2_scale_bias: torch.Tensor | None = None
    w1_offset: torch.Tensor | None = None
    w2_offset: torch.Tensor | None = None


__all__ = [
    "MoEOptionalQuantTensor",
    "MoEWeights",
    "MoEWeightTensor",
]
