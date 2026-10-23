# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Triton-free min_p for 310P MRV2.

Signature matches mainline ``vllm_ascend.worker.v2.sample.min_p.apply_min_p``.
Semantics follow MRV1 ``MinPLogitsProcessor`` (probability-space threshold).
"""

from __future__ import annotations

import torch


def apply_min_p(
    logits: torch.Tensor,
    expanded_idx_mapping: torch.Tensor,
    min_p: torch.Tensor,
) -> None:
    """In-place min_p: keep tokens with ``p >= min_p * max_p``.

    Equivalent to the upstream MRV2 log-space form
    ``logit >= max_logit + log(min_p)`` when probabilities are softmax(logits).
    """
    min_p_rows = min_p[expanded_idx_mapping].to(dtype=torch.float32)
    need = min_p_rows != 0.0
    if not bool(need.any().item()):
        return
    probs = torch.softmax(logits, dim=-1, dtype=torch.float32)
    max_probs = probs.amax(dim=-1, keepdim=True)
    thresholds = max_probs * min_p_rows.unsqueeze(-1)
    invalid = (probs < thresholds) & need.unsqueeze(-1)
    logits.masked_fill_(invalid, float("-inf"))
