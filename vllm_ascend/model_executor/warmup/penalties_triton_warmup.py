# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Warm up Triton kernels used by ``apply_penalties_triton`` (bincount + penalties)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.logger import logger
from vllm.triton_utils import HAS_TRITON

from vllm_ascend.ops.triton.penalty import apply_penalties_triton

if TYPE_CHECKING:
    from vllm_ascend.worker.worker import NPUWorker

# Must match ``get_token_bin_counts_and_mask_triton`` (bincount.py).
_BINCOUNT_SEQ_BLOCK = 256

_PENALTIES_TENSOR_DTYPES: dict[str, torch.dtype] = {
    "logits": torch.float32,
    "history_tokens": torch.int64,
    "repetition_penalties": torch.float32,
    "frequency_penalties": torch.float32,
    "presence_penalties": torch.float32,
}


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
    token_dtype = _PENALTIES_TENSOR_DTYPES["history_tokens"]
    if seq_len == 0:
        return torch.empty(num_seqs, 0, dtype=token_dtype, device=device)
    tokens = torch.randint(
        0,
        vocab_size,
        (num_seqs, seq_len),
        dtype=token_dtype,
        device=device,
    )
    tokens[:, -1:] = vocab_size
    return tokens


# ``num_seqs``, ``prompt_len``, and ``output_len`` are dynamic in the Triton kernels;
# ``BLOCK_SIZE`` is fixed at 2048 in ``apply_all_penalties_kernel``.
def _warm_apply_penalties_triton(
    device: torch.device,
    num_seqs: int,
    vocab_size: int,
    prompt_len: int,
    output_len: int,
) -> None:
    dtypes = _PENALTIES_TENSOR_DTYPES
    logits = torch.randn(
        num_seqs,
        vocab_size,
        dtype=dtypes["logits"],
        device=device,
    )
    prompt_tokens = _make_history_tokens(num_seqs, prompt_len, vocab_size, device)
    output_tokens = _make_history_tokens(num_seqs, output_len, vocab_size, device)
    repetition_penalties = torch.ones(
        num_seqs,
        dtype=dtypes["repetition_penalties"],
        device=device,
    )
    frequency_penalties = torch.zeros(
        num_seqs,
        dtype=dtypes["frequency_penalties"],
        device=device,
    )
    presence_penalties = torch.zeros(
        num_seqs,
        dtype=dtypes["presence_penalties"],
        device=device,
    )

    apply_penalties_triton(
        logits,
        prompt_tokens,
        output_tokens,
        presence_penalties,
        frequency_penalties,
        repetition_penalties,
    )


@torch.inference_mode()
def penalties_triton_warmup(worker: NPUWorker) -> None:
    """JIT bincount and penalty Triton kernels before the first sampling with penalties."""
    if not HAS_TRITON:
        return

    device = worker.device
    max_num_reqs = max(worker.scheduler_config.max_num_seqs, 1)
    max_num_batched_tokens = max(worker.scheduler_config.max_num_batched_tokens, 1)
    vocab_size = _local_vocab_size(worker.model_config)
    seq_len = min(_BINCOUNT_SEQ_BLOCK + 1, max_num_batched_tokens)

    logger.info(
        "Warming up penalties Triton kernels: local_vocab_size=%d, "
        "num_seqs=%d, prompt_len=%d, output_len=%d, dtypes=%s",
        vocab_size,
        max_num_reqs,
        seq_len,
        seq_len,
        _PENALTIES_TENSOR_DTYPES,
    )

    _warm_apply_penalties_triton(
        device,
        max_num_reqs,
        vocab_size,
        seq_len,
        seq_len,
    )

    if device.type == "npu":
        torch.npu.synchronize()
