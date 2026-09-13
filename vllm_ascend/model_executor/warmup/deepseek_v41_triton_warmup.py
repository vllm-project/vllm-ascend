# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Warm the finite tile variants used by the V4.1 indexer."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from vllm.triton_utils import HAS_TRITON

from vllm_ascend.ops.triton.prepare_indexer_indices import prepare_indexer_indices
from vllm_ascend.ops.triton.quantize_indexer_query import quantize_indexer_query
from vllm_ascend.ops.triton.triton_utils import get_vectorcore_num

if TYPE_CHECKING:
    from vllm_ascend.worker.worker import NPUWorker


def collect_indexer_warmup_token_counts(topk: int, num_cores: int, max_tokens: int) -> list[int]:
    """One token count per reachable ``BLOCK_ROWS`` in index postprocessing."""
    # Match the 128 KiB, eight-buffer sort budget in prepare_indexer_indices.
    padded_topk = 1 << (topk - 1).bit_length()
    max_block_rows = 128 * 1024 // (padded_topk * 4 * 8)
    token_counts = [1]
    block_rows = 1
    while block_rows < max_block_rows:
        tokens = block_rows * num_cores + 1
        if tokens > max_tokens:
            break
        token_counts.append(tokens)
        block_rows *= 2
    return token_counts


@torch.inference_mode()
def deepseek_v41_triton_warmup(worker: NPUWorker) -> None:
    """Precompile indexer tiles before serving arbitrary eager token counts."""
    if not HAS_TRITON:
        return
    config = worker.model_config.hf_text_config
    if getattr(config, "model_type", None) not in (
        "deepseek_v4.1",
        "deepseek_v41",
        "deepseek_v4.1_text",
        "deepseek_v41_text",
    ):
        return
    ratios = sorted(set(config.compress_ratios[: config.num_hidden_layers]) - {0})
    if not ratios:
        return

    device = worker.device
    query = torch.zeros(1, config.index_n_heads, config.index_head_dim, dtype=worker.model_config.dtype, device=device)
    quantize_indexer_query(query)
    token_counts = collect_indexer_warmup_token_counts(
        config.index_topk, get_vectorcore_num(), worker.scheduler_config.max_num_batched_tokens
    )
    for tokens in token_counts:
        selected = torch.zeros(tokens, config.index_topk, dtype=torch.int32, device=device)
        positions = torch.zeros(tokens, dtype=torch.int64, device=device)
        for ratio in ratios:
            prepare_indexer_indices(selected, positions, ratio)
