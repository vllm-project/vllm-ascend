# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# mypy: ignore-errors
"""Triton-free bad_words; reuses ``vllm.v1.sample.ops.bad_words.apply_bad_words``."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from vllm.v1.sample.ops.bad_words import apply_bad_words as apply_bad_words_vllm


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

    bad_by_row: dict[int, list[list[int]]] = {}
    past_tokens: list[list[int]] = []
    for row_idx, req_idx in enumerate(expanded_np.tolist()):
        req_idx = int(req_idx)
        if req_idx < 0 or not bool(use_bad_words[req_idx]):
            past_tokens.append([])
            continue
        total_len = int(total_lens[req_idx])
        past = all_token_ids_cpu[req_idx, :total_len].tolist()
        if local_pos_np is not None and input_ids_cpu is not None:
            pos = int(local_pos_np[row_idx])
            if pos > 0:
                start_idx = row_idx - pos
                for prev_pos in range(pos):
                    past.append(int(input_ids_cpu[start_idx + prev_pos + 1].item()))
        past_tokens.append(past)
        bad_by_row[row_idx] = bad_words_token_ids[req_idx]
    apply_bad_words_vllm(logits, bad_by_row, past_tokens)
