# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Warm up Triton kernels used by ``apply_penalties_triton`` (bincount + penalties)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.logger import logger
from vllm.triton_utils import HAS_TRITON, triton

from vllm_ascend.ops.triton.penalty import apply_penalties_triton
from vllm_ascend.ops.triton.triton_utils import get_vectorcore_num

if TYPE_CHECKING:
    from vllm_ascend.worker.worker import NPUWorker

# Must match ``get_token_bin_counts_and_mask_triton`` (bincount.py).
_BINCOUNT_SEQ_BLOCK = 256

# Runtime uses padded prompt/output history; cover empty-one-side cases from e2e grid.
_EXTRA_PROMPT_OUTPUT_LEN_PAIRS: tuple[tuple[int, int], ...] = (
    (0, 16),
    (32, 0),
    (0, 0),
)


def bincount_total_blocks(
    num_seqs: int,
    seq_len: int,
    seq_block: int = _BINCOUNT_SEQ_BLOCK,
) -> int:
    """Work items for ``token_bin_counts_and_mask_kernel`` (matches host launch)."""
    if num_seqs <= 0 or seq_len <= 0:
        return 0
    return num_seqs * triton.cdiv(seq_len, seq_block)


def bincount_fixed_launch_grid_size(core_num: int) -> int:
    """Bincount always launches ``core_num`` programs (see bincount.py)."""
    return max(core_num, 1)


def collect_warmup_bincount_seq_lens(
    max_seq_len: int,
    seq_block: int = _BINCOUNT_SEQ_BLOCK,
) -> list[int]:
    """Representative seq lengths (smoke tests; kernel uses dynamic seq_len)."""
    if max_seq_len <= 0:
        return [0]

    sizes = {1, max_seq_len}
    if seq_block > 1:
        sizes.add(seq_block - 1)
    sizes.add(seq_block)
    sizes.add(seq_block + 1)
    if max_seq_len >= seq_block * 2:
        sizes.add(seq_block * 2)
    return sorted(s for s in sizes if 0 <= s <= max_seq_len)


def collect_warmup_bincount_shapes(
    max_num_reqs: int,
    max_num_batched_tokens: int,
) -> list[tuple[int, int]]:
    """Minimal (num_seqs, seq_len) pairs to JIT bincount once per process.

    Launch grid is fixed at ``core_num``; batch/seq/total_blocks/vocab are
    dynamic in the kernel. ``seq_len`` follows ``--max-num-batched-tokens``
    (``SchedulerConfig.max_num_batched_tokens``), which bounds prompt history
    work in typical serving configs.
    """
    if max_num_reqs <= 0 or max_num_batched_tokens <= 0:
        return []

    typical_seq_len = min(_BINCOUNT_SEQ_BLOCK + 1, max_num_batched_tokens)
    pairs = {(1, 1), (max_num_reqs, typical_seq_len)}
    if max_num_batched_tokens != typical_seq_len:
        pairs.add((max_num_reqs, max_num_batched_tokens))
    return sorted(pairs)


def collect_warmup_penalty_cases(
    max_num_reqs: int,
    max_num_batched_tokens: int,
) -> list[tuple[int, int, int]]:
    """(num_seqs, prompt_len, output_len) cases for ``apply_penalties_triton`` warmup."""
    max_num_reqs = max(max_num_reqs, 1)
    cases: list[tuple[int, int, int]] = []
    seen: set[tuple[int, int, int]] = set()

    def add_case(num_seqs: int, prompt_len: int, output_len: int) -> None:
        key = (num_seqs, prompt_len, output_len)
        if key not in seen:
            seen.add(key)
            cases.append(key)

    for num_seqs, prompt_len in collect_warmup_bincount_shapes(
        max_num_reqs, max_num_batched_tokens
    ):
        add_case(num_seqs, prompt_len, prompt_len)

    for prompt_len, output_len in _EXTRA_PROMPT_OUTPUT_LEN_PAIRS:
        add_case(max_num_reqs, prompt_len, output_len)

    return cases


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
    if seq_len == 0:
        return torch.empty(num_seqs, 0, dtype=torch.int64, device=device)
    tokens = torch.randint(0, vocab_size, (num_seqs, seq_len), dtype=torch.int64, device=device)
    tokens[:, -1:] = vocab_size
    return tokens


def _warm_apply_penalties_triton(
    device: torch.device,
    num_seqs: int,
    vocab_size: int,
    prompt_len: int,
    output_len: int,
    logits_dtype: torch.dtype,
) -> None:
    logits = torch.randn(num_seqs, vocab_size, dtype=logits_dtype, device=device)
    prompt_tokens = _make_history_tokens(num_seqs, prompt_len, vocab_size, device)
    output_tokens = _make_history_tokens(num_seqs, output_len, vocab_size, device)
    repetition_penalties = torch.ones(num_seqs, dtype=torch.float32, device=device)
    frequency_penalties = torch.zeros(num_seqs, dtype=torch.float32, device=device)
    presence_penalties = torch.zeros(num_seqs, dtype=torch.float32, device=device)

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
    max_num_batched_tokens = worker.scheduler_config.max_num_batched_tokens
    vocab_size = _local_vocab_size(worker.model_config)
    logits_dtype = worker.model_config.dtype
    core_num = get_vectorcore_num()
    launch_grid = bincount_fixed_launch_grid_size(core_num)
    penalty_cases = collect_warmup_penalty_cases(max_num_reqs, max_num_batched_tokens)

    logger.info(
        "Warming up penalties Triton kernels: local_vocab_size=%d, "
        "bincount_launch_grid=%d, penalty_cases=%s, logits_dtype=%s",
        vocab_size,
        launch_grid,
        penalty_cases,
        logits_dtype,
    )

    for num_seqs, prompt_len, output_len in penalty_cases:
        logger.info("Warming up penalties Triton kernels: num_seqs=%d, prompt_len=%d, output_len=%d",
                    num_seqs, prompt_len, output_len
                    )
        _warm_apply_penalties_triton(
            device,
            num_seqs,
            vocab_size,
            prompt_len,
            output_len,
            logits_dtype,
        )

    if device.type == "npu":
        torch.npu.synchronize()
