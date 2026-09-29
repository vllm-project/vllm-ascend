# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Register and materialize speculative-decoding Triton specializations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from vllm.triton_utils import HAS_TRITON

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.ops.triton.reject_sample import (
    _EXPAND_KERNEL,
    _REJECTION_GREEDY_KERNEL,
    _REJECTION_GREEDY_SPEC_LEN_1_KERNEL,
    _REJECTION_RANDOM_SAMPLE_BLOCK_VERIFY_KERNEL,
    _REJECTION_RANDOM_SAMPLE_KERNEL,
    _SAMPLE_RECOVERED_TOKENS_KERNEL,
    cal_grid_and_block_size,
)
from vllm_ascend.ops.triton.spec_decode.utils import (
    _PREPARE_INPUTS_PADDED_KERNEL,
)

if TYPE_CHECKING:
    from vllm_ascend.worker.worker import NPUWorker


@dataclass(frozen=True)
class RejectionWarmupContext:
    block_sizes: tuple[int, ...]
    max_spec_len: int
    no_draft_probs_values: tuple[bool, ...]
    enable_reduce_sampling: bool
    entropy_verify: bool
    synthetic_mode: bool
    block_verify: bool
    vocab_block_size: int = 512
    posterior_threshold: float = 0.95
    posterior_alpha: float = 0.4
    sub_block: int = 4096
    epsilon: float = 1e-10


def _is_ngram_spec_method(spec_config: Any) -> bool:
    method = getattr(spec_config, "method", None)
    if method in ("ngram", "ngram_gpu"):
        return True
    use_ngram_gpu = getattr(spec_config, "use_ngram_gpu", None)
    return bool(callable(use_ngram_gpu) and use_ngram_gpu())


def _collect_no_draft_probs_values(spec_config: Any, pipeline_parallel_size: int) -> tuple[bool, ...]:
    if _is_ngram_spec_method(spec_config) or pipeline_parallel_size <= 1:
        return (True,)
    return (False, True)


def collect_warmup_rejection_block_sizes(max_num_reqs: int) -> tuple[int, ...]:
    """Return every BLOCK_SIZE reachable from 1..max_num_reqs."""
    if max_num_reqs <= 0:
        return ()
    block_sizes: set[int] = set()
    for batch_size in range(1, max_num_reqs + 1):
        _, block_size = cal_grid_and_block_size(batch_size)
        block_sizes.add(block_size)
    return tuple(sorted(block_sizes))


def _make_context(worker: NPUWorker) -> RejectionWarmupContext | None:
    spec_config = worker.vllm_config.speculative_config
    if spec_config is None:
        return None

    max_spec_len = spec_config.num_speculative_tokens
    if max_spec_len <= 0:
        return None

    ascend_config = get_ascend_config()
    rejection_config = ascend_config.rejection_sampler_config
    rejection_sampler = getattr(getattr(worker, "model_runner", None), "rejection_sampler", None)

    return RejectionWarmupContext(
        block_sizes=collect_warmup_rejection_block_sizes(worker.scheduler_config.max_num_seqs),
        max_spec_len=max_spec_len,
        no_draft_probs_values=_collect_no_draft_probs_values(
            spec_config,
            worker.vllm_config.parallel_config.pipeline_parallel_size,
        ),
        enable_reduce_sampling=bool(ascend_config.enable_reduce_sample),
        entropy_verify=bool(rejection_config.enable_entropy_verify),
        synthetic_mode=bool(getattr(rejection_sampler, "synthetic_mode", False)),
        block_verify=(max_spec_len >= 3 and bool(rejection_config.enable_block_verify)),
    )


def _selected_kernels(context: RejectionWarmupContext) -> tuple[Any, ...]:
    kernels = [
        _PREPARE_INPUTS_PADDED_KERNEL,
        _EXPAND_KERNEL,
        _REJECTION_GREEDY_SPEC_LEN_1_KERNEL,
        _REJECTION_GREEDY_KERNEL,
        _SAMPLE_RECOVERED_TOKENS_KERNEL,
    ]
    if context.block_verify:
        kernels.append(_REJECTION_RANDOM_SAMPLE_BLOCK_VERIFY_KERNEL)
    else:
        kernels.append(_REJECTION_RANDOM_SAMPLE_KERNEL)
    return tuple(kernels)


def _enabled(worker: NPUWorker) -> bool:
    if not HAS_TRITON:
        return False
    kernel_config = getattr(worker.vllm_config, "kernel_config", None)
    return kernel_config is None or bool(kernel_config.enable_jit_warmup)


def register_rejection_sampler_triton_warmup(worker: NPUWorker) -> bool:
    """Register the selected rejection wrappers with the active registry."""
    if not _enabled(worker):
        return False
    context = _make_context(worker)
    if context is None:
        return False
    for kernel in _selected_kernels(context):
        kernel.register_warmup(context)
    return True


def rejection_sampler_triton_warmup(worker: NPUWorker) -> None:
    """Materialize rejection specializations directly for early/fallback paths."""
    if not _enabled(worker):
        return
    context = _make_context(worker)
    if context is None:
        return
    for kernel in _selected_kernels(context):
        kernel.warmup(context)
