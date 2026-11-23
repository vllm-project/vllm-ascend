# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Triton-free min_p (MRV1 MinPLogitsProcessor probability-space semantics)."""

from __future__ import annotations

import torch


def apply_min_p(
    logits: torch.Tensor,
    expanded_idx_mapping: torch.Tensor,
    min_p: torch.Tensor,
) -> None:
    min_p_rows = min_p[expanded_idx_mapping].to(dtype=torch.float32)
    need = min_p_rows != 0.0
    if not bool(need.any().item()):
        return
    probs = torch.softmax(logits, dim=-1, dtype=torch.float32)
    thresholds = probs.amax(dim=-1, keepdim=True) * min_p_rows.unsqueeze(-1)
    logits.masked_fill_((probs < thresholds) & need.unsqueeze(-1), float("-inf"))
