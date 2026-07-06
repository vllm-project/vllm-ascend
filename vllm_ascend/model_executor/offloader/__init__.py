"""Ascend-specific model parameter offloading."""

from vllm.model_executor.offloader.base import (
    BaseOffloader,
    NoopOffloader,
    get_offloader,
    set_offloader,
)
from vllm.model_executor.offloader.uva import UVAOffloader

from vllm_ascend.model_executor.offloader.base import create_offloader
from vllm_ascend.model_executor.offloader.prefetch import AscendPrefetchOffloader

__all__ = [
    "AscendPrefetchOffloader",
    "BaseOffloader",
    "NoopOffloader",
    "UVAOffloader",
    "create_offloader",
    "get_offloader",
    "set_offloader",
]
