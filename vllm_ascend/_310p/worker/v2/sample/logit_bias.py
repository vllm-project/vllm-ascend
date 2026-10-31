# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Triton-free logit_bias."""

from __future__ import annotations

import numpy as np
import torch


def apply_logit_bias(
    logits: torch.Tensor,
    expanded_idx_mapping: torch.Tensor,
    *,
    logit_bias: dict[int, dict[int, float]],
    use_logit_bias: np.ndarray,
) -> None:
    expanded_np = expanded_idx_mapping.detach().to("cpu").numpy()
    for row_idx, req_idx in enumerate(expanded_np.tolist()):
        req_idx = int(req_idx)
        if req_idx < 0 or not bool(use_logit_bias[req_idx]):
            continue
        bias_map = logit_bias.get(req_idx)
        if not bias_map:
            continue
        for token_id, bias in bias_map.items():
            if 0 <= token_id < logits.shape[-1]:
                logits[row_idx, token_id] += bias
