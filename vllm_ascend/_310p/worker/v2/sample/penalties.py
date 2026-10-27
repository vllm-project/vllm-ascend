# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# mypy: ignore-errors
"""Triton-free penalties; reuses upstream ``use_penalty`` and vLLM ``apply_penalties``."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from vllm.model_executor.layers.utils import apply_penalties
from vllm.v1.worker.gpu.sample.penalties import use_penalty

__all__ = [
    "apply_penalties",
    "build_penalty_token_tensors",
    "use_penalty",
]


def build_penalty_token_tensors(
    *,
    req_states: Any,
    use_penalty_mask: np.ndarray,
    expanded_idx_mapping_np: np.ndarray,
    vocab_size: int,
    device: torch.device,
    input_ids: torch.Tensor | None = None,
    expanded_local_pos: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build padded prompt/output ids from CPU req_states (MRV1 pad=vocab_size)."""
    vocab_size = max(int(vocab_size), 1)
    num_rows = int(expanded_idx_mapping_np.shape[0])
    all_token_ids_cpu = req_states.all_token_ids.cpu
    prompt_lens = req_states.prompt_len.np
    total_lens = req_states.total_len.np
    local_pos_np = None if expanded_local_pos is None else expanded_local_pos.detach().to("cpu").numpy()
    input_ids_cpu = None if input_ids is None else input_ids.detach().to("cpu")

    prompt_lists: list[list[int]] = []
    output_lists: list[list[int]] = []
    max_prompt = 0
    max_output = 0
    for row_idx, req_idx in enumerate(expanded_idx_mapping_np.tolist()):
        req_idx = int(req_idx)
        if req_idx < 0 or not bool(use_penalty_mask[req_idx]):
            prompt_lists.append([])
            output_lists.append([])
            continue
        prompt_len = int(prompt_lens[req_idx])
        total_len = int(total_lens[req_idx])
        prompt = all_token_ids_cpu[req_idx, :prompt_len].tolist()
        output = all_token_ids_cpu[req_idx, prompt_len:total_len].tolist()
        if local_pos_np is not None and input_ids_cpu is not None:
            pos = int(local_pos_np[row_idx])
            if pos > 0:
                start_idx = row_idx - pos
                for prev_pos in range(pos):
                    output.append(int(input_ids_cpu[start_idx + prev_pos + 1].item()))
        prompt_lists.append(prompt)
        output_lists.append(output)
        max_prompt = max(max_prompt, len(prompt))
        max_output = max(max_output, len(output))

    max_prompt = max(max_prompt, 1)
    max_output = max(max_output, 1)
    prompt_t = torch.full((num_rows, max_prompt), fill_value=vocab_size, dtype=torch.int64, device=device)
    output_t = torch.full((num_rows, max_output), fill_value=vocab_size, dtype=torch.int64, device=device)
    for row_idx, (prompt, output) in enumerate(zip(prompt_lists, output_lists)):
        if prompt:
            prompt_t[row_idx, : len(prompt)] = torch.as_tensor(prompt, dtype=torch.int64)
        if output:
            output_t[row_idx, : len(output)] = torch.as_tensor(output, dtype=torch.int64)
    return prompt_t, output_t
