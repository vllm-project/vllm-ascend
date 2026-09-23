# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Lane-stable re-exports for the upstream DeepSeek-V4.1 modules.

The ``vllm.models.deepseek_v41`` package (renamed from ``deepseek_v4_1`` on
main) has no counterpart on the pinned release, so the V4.1 model code is
main-only and is never imported on the release lane. The imports carry
``# type: ignore[import-not-found]`` so the release-lane mypy pass resolves.
"""

from vllm.models.deepseek_v41.common.engram import (  # type: ignore[import-not-found]
    DEAD_ID,
    EngramLayout,
    NgramHashState,
    ParallelEngramEmbedding,
    _write_hash_cache_kernel,
)
from vllm.models.deepseek_v41.common.mm_preprocess import (  # type: ignore[import-not-found]
    IMAGE,
    IMAGE_END,
    IMAGE_NEW_LINE,
    IMAGE_PLACEHOLDER,
    IMAGE_SENTINEL_BASE_ID,
    IMAGE_START,
    DeepseekV4VLDummyInputsBuilder,
    DeepseekV4VLMultiModalProcessor,
    DeepseekV4VLProcessingInfo,
    DeepseekV4VLProcessor,
    image_sentinel_mask,
    image_token_types,
)

__all__ = [
    "DEAD_ID",
    "EngramLayout",
    "NgramHashState",
    "ParallelEngramEmbedding",
    "_write_hash_cache_kernel",
    "IMAGE",
    "IMAGE_END",
    "IMAGE_NEW_LINE",
    "IMAGE_PLACEHOLDER",
    "IMAGE_SENTINEL_BASE_ID",
    "IMAGE_START",
    "DeepseekV4VLDummyInputsBuilder",
    "DeepseekV4VLMultiModalProcessor",
    "DeepseekV4VLProcessingInfo",
    "DeepseekV4VLProcessor",
    "image_sentinel_mask",
    "image_token_types",
]
