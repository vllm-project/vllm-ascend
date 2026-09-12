# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KV cache layers and metadata helpers for the GLM-Next pooled indexer."""

from typing import Any

import torch
from torch import nn
from vllm.config import CacheConfig, VllmConfig, get_current_vllm_config
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.v1.kv_cache_interface import KVCacheSpec

from vllm_ascend.core.kv_cache_interface import (
    AscendIndexerKPoolStateSpec,
    AscendMLAAttentionSpec,
)
from vllm_ascend.utils import vllm_version_is


def is_glm5_next_cache_spec(spec: KVCacheSpec) -> bool:
    return getattr(spec, "model_version", None) == "glm5_next"


def format_indexer_kpool_slot_mapping(
    slot_mapping: torch.Tensor,
    positions: torch.Tensor,
    logical_block_size: int,
    compress_ratio: int,
) -> torch.Tensor:
    """Map completed token pools onto the compressed indexer cache."""
    if compress_ratio <= 1 or logical_block_size <= 0 or logical_block_size % compress_ratio:
        raise ValueError(
            f"logical_block_size={logical_block_size} must be divisible by compress_ratio={compress_ratio}."
        )
    valid = (slot_mapping >= 0) & (torch.remainder(positions + 1, compress_ratio) == 0)
    safe_slots = slot_mapping.clamp_min(0)
    block_ids = torch.div(safe_slots, logical_block_size, rounding_mode="floor")
    offsets = torch.remainder(safe_slots, logical_block_size)
    compressed_slots = block_ids * (logical_block_size // compress_ratio) + torch.div(
        offsets,
        compress_ratio,
        rounding_mode="floor",
    )
    return torch.where(valid, compressed_slots, torch.full_like(compressed_slots, -1))


class Glm5NextIndexerCache(nn.Module, AttentionLayerBase):
    """Independently allocated compressed-K cache for the GLM-Next indexer."""

    # Auxiliary caches use the GLM-specific small-page class instead of the
    # generic attention/Mamba page-size class.
    align_kv_cache_with_mamba = False

    def __init__(
        self,
        *,
        head_dim: int,
        dtype: torch.dtype,
        cache_role: str,
        cache_config: CacheConfig,
        prefix: str,
        compress_ratio: int,
    ) -> None:
        super().__init__()
        if compress_ratio <= 1 or cache_config.block_size % compress_ratio:
            raise ValueError(
                "GLM-Next indexer cache requires block_size divisible by a "
                f"compress_ratio greater than one, got {cache_config.block_size} "
                f"and {compress_ratio}."
            )
        self.head_dim = head_dim
        self.dtype = dtype
        self.cache_role = cache_role
        self.cache_config = cache_config
        self.compress_ratio = compress_ratio
        self.prefix = prefix
        current_config = get_current_vllm_config()
        self.kv_cache = [torch.tensor([]) for _ in range(current_config.parallel_config.pipeline_parallel_size)]
        static_context = current_config.compilation_config.static_forward_context
        if prefix in static_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        static_context[prefix] = self

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        del vllm_config
        ratio_kwargs: dict[str, Any] = (
            {"compress_ratio": self.compress_ratio}
            if vllm_version_is("0.28.0")
            else {"tokens_per_state": self.compress_ratio}
        )
        return AscendMLAAttentionSpec(
            block_size=self.cache_config.block_size,
            num_kv_heads=1,
            head_size=self.head_dim,
            dtype=self.dtype,
            cache_dtype_str=None,
            model_version="glm5_next",
            indexes_kv_by_block_stride=True,
            **ratio_kwargs,
        )

    def get_attn_backend(self):
        from vllm_ascend.attention.indexer_kpool import (
            AscendIndexerKPoolBackend,
        )

        return AscendIndexerKPoolBackend

    def forward(self): ...


class Glm5NextStateCache(nn.Module, AttentionLayerBase):
    """Paged FP32 ``[K, gate]`` state for incomplete GLM-Next pools."""

    align_kv_cache_with_mamba = False

    def __init__(
        self,
        *,
        state_dim: int,
        dtype: torch.dtype,
        compress_ratio: int,
        cache_config: CacheConfig,
        prefix: str,
    ) -> None:
        super().__init__()
        if dtype != torch.float32:
            raise ValueError(f"GLM-Next compressor state must use torch.float32, got {dtype}.")
        if compress_ratio <= 1:
            raise ValueError(
                f"GLM-Next compressor state requires compress_ratio greater than one, got {compress_ratio}."
            )
        self.state_dim = state_dim
        self.dtype = dtype
        self.prefix = prefix
        self.compress_ratio = compress_ratio
        self.block_size = compress_ratio
        self.sliding_window = compress_ratio
        self.cache_config = cache_config
        self.cache_role = "indexer_state"
        current_config = get_current_vllm_config()
        self.kv_cache = [torch.tensor([]) for _ in range(current_config.parallel_config.pipeline_parallel_size)]
        static_context = current_config.compilation_config.static_forward_context
        if prefix in static_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        static_context[prefix] = self

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        del vllm_config
        return AscendIndexerKPoolStateSpec(
            block_size=self.block_size,
            num_kv_heads=1,
            head_size=self.state_dim,
            dtype=self.dtype,
            sliding_window=self.sliding_window,
            cache_dtype_str=None,
            model_version="glm5_next",
            cache_role=self.cache_role,
            indexes_kv_by_block_stride=True,
        )

    def get_attn_backend(self):
        from vllm_ascend.attention.indexer_kpool import (
            AscendIndexerKPoolStateBackend,
        )

        return AscendIndexerKPoolStateBackend

    def forward(self): ...
