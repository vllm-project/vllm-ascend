# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ExpertTensorSpec:
    name: str
    shape: tuple[int, ...]
    dtype: torch.dtype


@dataclass(frozen=True)
class PreparedExpertTensor:
    """Pinned physical bytes plus the native tensor layout they encode."""

    data: torch.Tensor
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    dtype: torch.dtype
    device: torch.device
    npu_format: int
    expert_bytes: int


class ExpertWeightView:
    """Pass one consistent weight bundle to both GEMMs without rebinding Parameters."""

    def __init__(self, layer, tensors: dict[str, torch.Tensor]):
        self._layer = layer
        self.__dict__.update(tensors)
        self.ascend_expert_map = None

    def __getattr__(self, name):
        return getattr(self._layer, name)
