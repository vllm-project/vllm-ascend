# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Triton-free temperature (mainline ``gumbel.apply_temperature`` is Triton)."""

from __future__ import annotations

import torch


def apply_temperature(
    logits: torch.Tensor,
    expanded_idx_mapping: torch.Tensor,
    temperature: torch.Tensor,
) -> None:
    """In-place temperature scaling (Triton-free; skips temp in {0, 1})."""
    temps = temperature[expanded_idx_mapping].to(dtype=torch.float32)
    need_scale = (temps != 0.0) & (temps != 1.0)
    if not bool(need_scale.any().item()):
        return
    # Work in FP32 for numerical stability, write back to logits dtype.
    scaled = logits.to(torch.float32) / temps.unsqueeze(-1)
    logits.copy_(torch.where(need_scale.unsqueeze(-1), scaled.to(logits.dtype), logits))
