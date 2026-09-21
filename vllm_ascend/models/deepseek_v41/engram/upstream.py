# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Resolve the Engram API from native vLLM or the v0.29 compatibility layer."""

_UPSTREAM_MODULE = "vllm.models.deepseek_v4_1.common.engram"

try:
    from vllm.models.deepseek_v4_1.common.engram import (
        DEAD_ID,
        EngramLayout,
        NgramHashState,
        ParallelEngramEmbedding,
        _write_hash_cache_kernel,
    )
except ModuleNotFoundError as exc:
    # vLLM v0.29 predates DeepSeek V4.1. Do not hide an unrelated missing
    # dependency from a native Engram module that is present but broken.
    if exc.name is None or not _UPSTREAM_MODULE.startswith(exc.name):
        raise
    from vllm_ascend.compat.deepseek_v41.engram import (
        DEAD_ID,
        EngramLayout,
        NgramHashState,
        ParallelEngramEmbedding,
        _write_hash_cache_kernel,
    )

__all__ = [
    "DEAD_ID",
    "EngramLayout",
    "NgramHashState",
    "ParallelEngramEmbedding",
    "_write_hash_cache_kernel",
]
