# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Register and materialize DeepSeek V4.1 indexer Triton specializations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from vllm.triton_utils import HAS_TRITON

from vllm_ascend.ops.triton.prepare_indexer_indices import (
    _PREPARE_INDEXER_INDICES_KERNEL,
)
from vllm_ascend.ops.triton.quantize_indexer_query import (
    _QUANTIZE_INDEXER_QUERY_KERNEL,
)
from vllm_ascend.ops.triton.triton_utils import get_vectorcore_num
from vllm_ascend.utils import is_deepseek_v41

if TYPE_CHECKING:
    from vllm_ascend.worker.worker import NPUWorker


@dataclass(frozen=True)
class IndexerWarmupContext:
    topk: int
    token_counts: tuple[int, ...]
    num_cores: int
    compress_ratios: tuple[int, ...]


def collect_indexer_warmup_token_counts(topk: int, num_cores: int, max_tokens: int) -> list[int]:
    """One token count per reachable BLOCK_ROWS in index postprocessing."""
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


def _enabled(worker: NPUWorker) -> bool:
    if not HAS_TRITON:
        return False
    kernel_config = getattr(worker.vllm_config, "kernel_config", None)
    return kernel_config is None or bool(kernel_config.enable_jit_warmup)


def _make_context(worker: NPUWorker) -> IndexerWarmupContext | None:
    config = worker.model_config.hf_text_config
    if not is_deepseek_v41(config):
        return None

    ratios = tuple(sorted(set(config.compress_ratios[: config.num_hidden_layers]) - {0}))
    if not ratios:
        return None

    num_cores = max(get_vectorcore_num(), 1)
    token_counts = tuple(
        collect_indexer_warmup_token_counts(
            config.index_topk,
            num_cores,
            worker.scheduler_config.max_num_batched_tokens,
        )
    )
    return IndexerWarmupContext(
        topk=config.index_topk,
        token_counts=token_counts,
        num_cores=num_cores,
        compress_ratios=ratios,
    )


def register_indexer_triton_warmup(worker: NPUWorker) -> bool:
    """Register indexer wrappers with the active warmup registry."""
    if not _enabled(worker):
        return False
    context = _make_context(worker)
    if context is None:
        return False
    _QUANTIZE_INDEXER_QUERY_KERNEL.register_warmup(worker.vllm_config)
    _PREPARE_INDEXER_INDICES_KERNEL.register_warmup(context)
    return True


def indexer_triton_warmup(worker: NPUWorker) -> None:
    """Materialize indexer specializations directly for early/fallback paths."""
    if not _enabled(worker):
        return
    context = _make_context(worker)
    if context is None:
        return
    _QUANTIZE_INDEXER_QUERY_KERNEL.warmup(worker.vllm_config)
    _PREPARE_INDEXER_INDICES_KERNEL.warmup(context)
