# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Triton-free allowed_token_ids mask helpers for 310P MRV2."""

from __future__ import annotations

import torch


def materialize_allowed_mask(
    mask: torch.Tensor,
    req_idx: int,
    allowed: list[int],
    vocab_size: int,
) -> None:
    """Fill ``mask[req_idx]``: True = disallowed (masked to -inf)."""
    mask[req_idx] = True
    ids = torch.tensor(allowed, dtype=torch.int64, device=mask.device)
    ids = ids[(ids >= 0) & (ids < vocab_size)]
    if ids.numel():
        mask[req_idx, ids] = False


def apply_allowed_token_ids(
    logits: torch.Tensor,
    allowed_mask: torch.Tensor,
    expanded_idx_mapping: torch.Tensor,
) -> None:
    """Mask disallowed tokens to -inf (MRV1 allowed_token_ids_mask)."""
    row_mask = allowed_mask[expanded_idx_mapping]
    logits.masked_fill_(row_mask, float("-inf"))
