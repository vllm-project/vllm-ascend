"""Ascend-specific model parameter offloading."""

from vllm_ascend.model_executor.offloader.base import create_offloader
from vllm_ascend.model_executor.offloader.prefetch import AscendPrefetchOffloader

__all__ = [
    "AscendPrefetchOffloader",
    "create_offloader",
]
