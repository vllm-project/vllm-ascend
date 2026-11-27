# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Profiler ranges for service-side MoE A/B analysis."""

from contextlib import AbstractContextManager, nullcontext

import torch

from vllm_ascend import envs as ascend_envs

_MOE_PROFILE_RANGES_ENABLED = ascend_envs.VLLM_ASCEND_MOE_PROFILE_RANGES
_MOE_PROFILE_SYNC_BOUNDARIES_ENABLED = ascend_envs.VLLM_ASCEND_MOE_PROFILE_SYNC_BOUNDARIES


def moe_profile_sync_boundaries_enabled() -> bool:
    """Return whether comparable synchronized MoE boundaries are enabled."""
    return _MOE_PROFILE_RANGES_ENABLED and _MOE_PROFILE_SYNC_BOUNDARIES_ENABLED


def moe_profile_range(
    backend: str,
    stage: str,
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
) -> AbstractContextManager:
    """Create a shape-qualified profiler range when service profiling is enabled."""
    if not _MOE_PROFILE_RANGES_ENABLED:
        return nullcontext()

    tokens = hidden_states.shape[0]
    hidden_size = hidden_states.shape[-1]
    top_k = topk_ids.shape[-1]
    name = f"vllm_ascend.moe.{backend}.{stage}[M={tokens},H={hidden_size},topK={top_k}]"
    return torch.profiler.record_function(name)


def moe_profile_phase(
    backend: str,
    stage: str,
    *,
    global_tokens: int,
    rank_local_tokens: int,
    hidden_size: int,
    top_k: int,
) -> AbstractContextManager:
    """Identify both token layouts around prepare and finalize, without a sync."""
    if not _MOE_PROFILE_RANGES_ENABLED:
        return nullcontext()
    name = (
        f"vllm_ascend.moe.{backend}.{stage}"
        f"[global_M={global_tokens},rank_M={rank_local_tokens},H={hidden_size},topK={top_k}]"
    )
    return torch.profiler.record_function(name)


def moe_profile_component(
    backend: str,
    component: str,
    *,
    global_tokens: int,
    rank_local_tokens: int,
    hidden_size: int,
    top_k: int,
) -> AbstractContextManager:
    """Record a non-overlapping MoE component with explicit token metadata.

    Component ranges are intentionally separate from the larger prepare,
    finalize, and MoE-body ranges.  The profiling summarizer uses them to
    estimate the contribution of allocation, collective, copy, and kernel
    stages without changing the execution order.
    """
    return moe_profile_phase(
        backend,
        component,
        global_tokens=global_tokens,
        rank_local_tokens=rank_local_tokens,
        hidden_size=hidden_size,
        top_k=top_k,
    )
