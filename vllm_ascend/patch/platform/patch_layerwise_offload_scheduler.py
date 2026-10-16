# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2025 Huawei Technologies Co., Ltd.
"""Align layerwise Prefill chunks at the scheduler allocation hook."""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any

from vllm.v1.core.sched.async_scheduler import AsyncScheduler


_HOOK_NAME = "_mamba_block_aligned_split"
_PATCH_MARKER = "_vllm_ascend_layerwise_offload_patched"
_LAYERWISE_SCHEDULER_MARKER = "_vllm_ascend_layerwise_offload_enabled"


def align_prefill_chunk_for_store(
    request: Any,
    effective_computed_tokens: int,
    proposed_tokens: int,
    granularity: int,
) -> int:
    """Align non-final prompt chunks to Store's block granularity."""

    if proposed_tokens <= 0:
        return proposed_tokens
    if granularity <= 0:
        raise ValueError(f"invalid Store granularity: {granularity}")

    prompt_tokens = request.num_prompt_tokens
    if effective_computed_tokens >= prompt_tokens:
        return proposed_tokens

    prompt_part = min(proposed_tokens, prompt_tokens - effective_computed_tokens)
    # Keep the final prompt chunk, including any speculative tokens.
    if effective_computed_tokens + prompt_part >= prompt_tokens:
        return proposed_tokens

    aligned_end = (
        (effective_computed_tokens + prompt_part) // granularity
    ) * granularity
    return max(aligned_end - effective_computed_tokens, 0)


def _make_block_alignment_hook(original_hook: Callable[..., int]) -> Callable[..., int]:
    @functools.wraps(original_hook)
    def layerwise_block_alignment_hook(
        self: Any,
        request: Any,
        num_new_tokens: int,
        num_new_local_computed_tokens: int = 0,
        num_external_computed_tokens: int = 0,
    ) -> int:
        if not getattr(self, _LAYERWISE_SCHEDULER_MARKER, False):
            return original_hook(
                self,
                request,
                num_new_tokens,
                num_new_local_computed_tokens,
                num_external_computed_tokens,
            )

        effective_computed_tokens = (
            request.num_computed_tokens
            + num_new_local_computed_tokens
            + num_external_computed_tokens
        )
        return align_prefill_chunk_for_store(
            request,
            effective_computed_tokens,
            num_new_tokens,
            int(self.store_granularity),
        )

    setattr(layerwise_block_alignment_hook, _PATCH_MARKER, True)
    setattr(layerwise_block_alignment_hook, "_vllm_ascend_original", original_hook)
    return layerwise_block_alignment_hook


def apply_layerwise_offload_scheduler_patch() -> None:
    """Install the allocation hook once and fail if it is unavailable."""

    current_hook = getattr(AsyncScheduler, _HOOK_NAME, None)
    if not callable(current_hook):
        raise RuntimeError(
            "Layerwise Host KV offload requires vLLM AsyncScheduler to expose "
            f"a callable {_HOOK_NAME} pre-allocation hook"
        )
    if getattr(current_hook, _PATCH_MARKER, False):
        return
    setattr(AsyncScheduler, _HOOK_NAME, _make_block_alignment_hook(current_hook))


class LayerwiseOffloadAsyncScheduler(AsyncScheduler):
    """AsyncScheduler variant that aligns Prefill chunks for Store writes."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        setattr(self, _LAYERWISE_SCHEDULER_MARKER, True)
        self.need_mamba_block_aligned_split = True
        self.store_granularity = int(self.cache_config.block_size)
