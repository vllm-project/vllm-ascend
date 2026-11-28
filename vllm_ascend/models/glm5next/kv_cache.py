# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KV cache layers and metadata helpers for the GLM-Next pooled indexer."""

from typing import Any

import torch
from torch import nn
from vllm.config import CacheConfig, VllmConfig, get_current_vllm_config
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.utils.math_utils import cdiv, next_power_of_2
from vllm.v1.kv_cache_interface import CircularBufferSpec, KVCacheSpec

from vllm_ascend.core.kv_cache_interface import (
    AscendMLAAttentionSpec,
)


def is_glm5_next_cache_spec(spec: KVCacheSpec) -> bool:
    return getattr(spec, "model_version", None) == "glm5_next"


def get_kpool_tail_ring_capacity(vllm_config: VllmConfig, compress_ratio: int) -> int:
    """Keep the open pool plus every not-yet-accepted speculative token.

    Target verification writes all draft rows before rejection sampling is
    known.  The first row is the already-sampled token and is committed; the
    remaining ``num_speculative_tokens`` rows are lookahead.  Retaining that
    lookahead in addition to one full pool prevents rejected rows from wrapping
    around and destroying committed history that a replayed boundary token
    still needs. Round to a power-of-two number of pools, as upstream does,
    so the ring divides the model's attention block size.
    """
    speculative_config = getattr(vllm_config, "speculative_config", None)
    lookahead = int(getattr(speculative_config, "num_speculative_tokens", 0) or 0)
    if lookahead < 0:
        raise ValueError(f"num_speculative_tokens must be nonnegative, got {lookahead}.")
    return compress_ratio * next_power_of_2(cdiv(compress_ratio + lookahead, compress_ratio))


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
        if dtype not in (torch.bfloat16, torch.float8_e4m3fn, torch.int8):
            raise ValueError(f"GLM KPool cache requires BF16, FP8 or INT8, got {dtype}.")
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
        return AscendMLAAttentionSpec(
            block_size=self.cache_config.block_size,
            num_kv_heads=1,
            head_size=self.head_dim,
            dtype=self.dtype,
            scale_dim=1 if self.dtype in (torch.float8_e4m3fn, torch.int8) else 0,
            scale_dtype={torch.float8_e4m3fn: torch.float32, torch.int8: torch.float16}.get(self.dtype, torch.int8),
            cache_dtype_str=None,
            model_version="glm5_next",
            indexes_kv_by_block_stride=True,
            tokens_per_state=self.compress_ratio,
        )

    def get_attn_backend(self):
        from vllm_ascend.attention.indexer_kpool import (
            AscendIndexerKPoolBackend,
        )

        return AscendIndexerKPoolBackend

    def forward(self): ...


class Glm5NextTailCache(nn.Module, AttentionLayerBase):
    """Fixed per-request FP32 ring containing raw keys and gate scores."""

    align_kv_cache_with_mamba = False

    def __init__(
        self,
        *,
        head_dim: int,
        dtype: torch.dtype,
        compress_ratio: int,
        prefix: str,
        ring_capacity: int | None = None,
    ) -> None:
        super().__init__()
        if dtype != torch.float32:
            raise ValueError(f"GLM-Next tail must use torch.float32, got {dtype}.")
        if compress_ratio <= 1:
            raise ValueError(f"GLM-Next tail requires compress_ratio greater than one, got {compress_ratio}.")
        self.ring_capacity = compress_ratio if ring_capacity is None else ring_capacity
        if self.ring_capacity < compress_ratio or head_dim <= 0:
            raise ValueError("Tail requires a positive head_dim and ring_capacity >= compress_ratio.")
        self.head_dim = head_dim
        self.dtype = dtype
        self.prefix = prefix
        self.compress_ratio = compress_ratio
        current_config = get_current_vllm_config()
        self.kv_cache = [torch.tensor([]) for _ in range(current_config.parallel_config.pipeline_parallel_size)]
        static_context = current_config.compilation_config.static_forward_context
        if prefix in static_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        static_context[prefix] = self

    def bind_kv_cache(self, kv_cache: torch.Tensor) -> None:
        """Keep the model's engine container separate from the registered view."""
        self.kv_cache = [kv_cache]

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        del vllm_config
        return CircularBufferSpec(
            block_size=self.ring_capacity,
            num_kv_heads=1,
            head_size=2 * self.head_dim,
            head_size_v=0,
            dtype=self.dtype,
            dcp_sharded=False,
        )

    def get_attn_backend(self):
        from vllm_ascend.attention.indexer_kpool import (
            AscendIndexerKPoolTailBackend,
        )

        return AscendIndexerKPoolTailBackend

    def forward(self): ...
