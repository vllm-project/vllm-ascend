# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Warm up Triton kernels used by ``apply_penalties_triton`` (bincount + penalties)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.triton_utils import HAS_TRITON

from vllm_ascend.ops.triton.penalty import apply_penalties_triton

if TYPE_CHECKING:
    from vllm_ascend.worker.worker import NPUWorker

# Must match ``get_token_bin_counts_and_mask_triton`` (bincount.py).
_BINCOUNT_SEQ_BLOCK = 256


def _local_vocab_size(model_config) -> int:
    vocab_size = model_config.get_vocab_size()
    tp_size = get_tensor_model_parallel_world_size()
    return max(1, vocab_size // tp_size)


def _make_history_tokens(
    num_seqs: int,
    seq_len: int,
    vocab_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Build dummy prompt/output token ids for penalty warmup."""
    if seq_len == 0:
        return torch.empty(num_seqs, 0, dtype=torch.int64, device=device)

    tokens = torch.randint(
        0,
        vocab_size,
        (num_seqs, seq_len),
        dtype=torch.int64,
        device=device,
    )
    # Sentinel used by bincount/penalty path for padded positions.
    tokens[:, -1:] = vocab_size
    return tokens


@torch.inference_mode()
def penalties_triton_warmup(worker: NPUWorker) -> None:
    """JIT bincount and penalty Triton kernels before the first sampling with penalties."""
    if not HAS_TRITON:
        return

    device = worker.device
    num_seqs = max(worker.scheduler_config.max_num_seqs, 1)
    max_num_batched_tokens = max(worker.scheduler_config.max_num_batched_tokens, 1)
    vocab_size = _local_vocab_size(worker.model_config)

    # ``num_seqs`` / seq lens are dynamic in the kernels; penalty BLOCK_SIZE is fixed.
    # Warm with seq_len past one bincount block to cover the multi-block path.
    seq_len = min(_BINCOUNT_SEQ_BLOCK + 1, max_num_batched_tokens)

    logits = torch.randn(num_seqs, vocab_size, dtype=torch.float32, device=device)
    prompt_tokens = _make_history_tokens(num_seqs, seq_len, vocab_size, device)
    output_tokens = _make_history_tokens(num_seqs, seq_len, vocab_size, device)
    presence_penalties = torch.zeros(num_seqs, dtype=torch.float32, device=device)
    frequency_penalties = torch.zeros(num_seqs, dtype=torch.float32, device=device)
    repetition_penalties = torch.ones(num_seqs, dtype=torch.float32, device=device)

    apply_penalties_triton(
        logits,
        prompt_tokens,
        output_tokens,
        presence_penalties,
        frequency_penalties,
        repetition_penalties,
    )
