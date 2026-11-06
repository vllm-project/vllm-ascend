# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# mypy: ignore-errors
"""Triton-free bad_words filtering for 310P MRV2.

The algorithm matches ``vllm.v1.sample.ops.bad_words``, which is removed with
Model Runner V1. 310P keeps a local copy so sampler import does not depend on it.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

_SMALLEST_LOGIT = float("-inf")


def _ban_completed_bad_words(row_logits: torch.Tensor, bad_words: list[list[int]], past_tokens: list[int]) -> None:
    """Mask the last token of a bad word once its prefix matches ``past_tokens``."""
    for bad_word in bad_words:
        if len(bad_word) > len(past_tokens) + 1:
            continue
        prefix = bad_word[:-1]
        if prefix and past_tokens[-len(prefix) :] != prefix:
            continue
        last_token_id = bad_word[-1]
        # Slice assignment avoids a scalar write sync.
        row_logits[last_token_id : last_token_id + 1] = _SMALLEST_LOGIT


def apply_bad_words(
    logits: torch.Tensor,
    *,
    req_states: Any,
    bad_words_token_ids: dict[int, list[list[int]]],
    use_bad_words: np.ndarray,
    expanded_idx_mapping: torch.Tensor,
    input_ids: torch.Tensor | None = None,
    expanded_local_pos: torch.Tensor | None = None,
) -> None:
    """Ban completing bad-word sequences (past tokens from CPU req_states)."""
    expanded_np = expanded_idx_mapping.detach().to("cpu").numpy()
    local_pos_np = None if expanded_local_pos is None else expanded_local_pos.detach().to("cpu").numpy()
    input_ids_cpu = None if input_ids is None else input_ids.detach().to("cpu")
    all_token_ids_cpu = req_states.all_token_ids.cpu
    total_lens = req_states.total_len.np

    for row_idx, req_idx in enumerate(expanded_np.tolist()):
        req_idx = int(req_idx)
        if req_idx < 0 or not bool(use_bad_words[req_idx]):
            continue
        total_len = int(total_lens[req_idx])
        past = all_token_ids_cpu[req_idx, :total_len].tolist()
        if local_pos_np is not None and input_ids_cpu is not None:
            pos = int(local_pos_np[row_idx])
            if pos > 0:
                start_idx = row_idx - pos
                for prev_pos in range(pos):
                    past.append(int(input_ids_cpu[start_idx + prev_pos + 1].item()))
        _ban_completed_bad_words(logits[row_idx], bad_words_token_ids[req_idx], past)
