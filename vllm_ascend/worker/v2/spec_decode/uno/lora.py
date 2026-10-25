# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Adapted from https://github.com/vllm-project/vllm/blob/main/vllm/v1/worker/gpu/spec_decode/uno_lora.py
"""LoRA token mappings for UNO's shared-model drafting.

UNO drafts with the target model plus a gated LoRA adapter. Per request the
draft batch holds one base-weighted seed row and ``k - 1`` rows routed through
the adapter, so a single forward produces the whole draft block while the seed
row keeps the unmodified target distribution.
"""

from __future__ import annotations


def draft_lora_mapping(
    num_input_tokens: int,
    batch_size: int,
    k: int,
    uno_lora_id: int,
) -> tuple[int, ...]:
    """Return the draft-token LoRA mapping, including graph padding.

    Rows after ``batch_size * k`` are graph padding and stay on the base model
    so a padded replay cannot sample from a stale adapter row.

    Args:
        num_input_tokens: Length of the flat draft token buffer.
        batch_size: Number of requests in the draft batch.
        k: Draft rows per request (``num_speculative_tokens``).
        uno_lora_id: LoRA id UNO's gated adapter is registered under.

    Returns:
        One LoRA id per draft row: ``0`` (base weights) for the seed row and
        ``uno_lora_id`` for the remaining ``k - 1`` rows of each request.
    """
    if num_input_tokens < 0 or batch_size < 1 or k < 1:
        raise ValueError("UNO draft mapping dimensions must be positive")
    actual = batch_size * k
    if num_input_tokens < actual:
        raise ValueError("UNO draft mapping cannot truncate real query rows")
    per_request = (0,) + (int(uno_lora_id),) * (k - 1)
    return per_request * batch_size + (0,) * (num_input_tokens - actual)


def base_prompt_mapping(num_requests: int) -> tuple[int, ...]:
    """Return an all-base prompt/sampler mapping.

    Draft logits are produced by the base vocabulary head, so prompt and sampler
    rows never route through the adapter.
    """
    if num_requests < 0:
        raise ValueError("base prompt mapping length must be nonnegative")
    return (0,) * num_requests


__all__ = [
    "base_prompt_mapping",
    "draft_lora_mapping",
]
