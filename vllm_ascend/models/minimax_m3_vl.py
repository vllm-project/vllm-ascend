# SPDX-License-Identifier: Apache-2.0
"""Compatibility exports for the merged MiniMax M3 VL adapter."""

from vllm_ascend.models.minimax_m3 import (
    MiniMaxM3SparseForConditionalGeneration,
    MiniMaxM3VLDummyInputsBuilder,
    MiniMaxM3VLModel,
    MiniMaxM3VLMultiModalProcessor,
    MiniMaxM3VLProcessingInfo,
    MiniMaxVLVisionModel,
)

__all__ = [
    "MiniMaxM3SparseForConditionalGeneration",
    "MiniMaxM3VLDummyInputsBuilder",
    "MiniMaxM3VLMultiModalProcessor",
    "MiniMaxM3VLModel",
    "MiniMaxM3VLProcessingInfo",
    "MiniMaxVLVisionModel",
]
