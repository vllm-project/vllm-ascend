# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM-5 kpool indexer cache layers.

Two independently allocated BF16 caches replacing the CUDA path's fp8
indexer cache + circular tail buffer:

* ``AscendGlm5NextIndexerKPoolCache`` — compressed pool K, one row per
  completed kpool pool: MLAAttentionSpec -> ``[blocks, block, 1, head_dim]``
* ``AscendGlm5NextCompressorStateCache`` — per-token ``[K, gate]`` sliding
  state: AscendIndexerKPoolStateSpec -> ``[blocks, block, 1, 2*head_dim]``
"""

from __future__ import annotations

import torch
from torch import nn
from vllm.config import CacheConfig, VllmConfig, get_current_vllm_config
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.v1.attention.backend import AttentionBackend
from vllm.v1.kv_cache_interface import KVCacheSpec, MLAAttentionSpec

from vllm_ascend.core.kv_cache_interface import AscendIndexerKPoolStateSpec


class AscendGlm5NextIndexerKPoolCache(nn.Module, AttentionLayerBase):
    """One independently allocated physical cache in GLM-5 Indexer KPool MLA."""

    def __init__(
        self,
        *,
        head_dim: int,
        dtype: torch.dtype,
        cache_role: str,
        cache_config: CacheConfig,
        prefix: str,
        compress_ratio: int = 1,
    ) -> None:
        super().__init__()
        if cache_config.block_size % compress_ratio:
            raise ValueError(
                "Indexer KPool MLA cache block size "
                f"{cache_config.block_size} must be divisible by "
                f"compress ratio {compress_ratio}."
            )
        self.head_dim = head_dim
        self.dtype = dtype
        self.cache_role = cache_role
        self.cache_config = cache_config
        self.compress_ratio = compress_ratio
        self.prefix = prefix
        self.kv_cache = [
            torch.tensor([]) for _ in range(get_current_vllm_config().parallel_config.pipeline_parallel_size)
        ]
        compilation_config = get_current_vllm_config().compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        # Spec shape mirroring the CUDA path's indexer cache:
        # tokens_per_state=kpool keeps the indexer layers
        # (head_size=head_dim) in their own group apart from the main
        # 512-dim MLA layers during merge/uniform grouping, and lets the
        # GLM-5 grouping path (_get_kv_cache_groups_glm5_next /
        # _glm5_next_tensor_layout) recognize them. AscendMLAAttentionSpec
        # cannot be used here: its merge() ignores head_size, which would
        # merge the 128-dim indexer layers into the 512-dim MLA group and
        # break GLM layout recognition.
        return MLAAttentionSpec(
            block_size=self.cache_config.block_size,
            num_kv_heads=1,
            head_size=self.head_dim,
            dtype=self.dtype,
            cache_dtype_str=None,
            model_version="glm5_next",
            tokens_per_state=self.compress_ratio,
        )

    def get_attn_backend(self) -> type[AttentionBackend]:
        # The compressed indexer K cache uses the cache-only backend; the
        # main-MLA AscendIndexerKPoolMLABackend is for the NoPE attention
        # layers themselves (its builder requires uncompressed specs and
        # ".attn" layer names).
        from vllm_ascend.attention.indexer_kpool_mla_v1 import (
            AscendIndexerKPoolBackend,
        )

        return AscendIndexerKPoolBackend

    def forward(self): ...


class AscendGlm5NextCompressorStateCache(nn.Module, AttentionLayerBase):
    """GLM-5 kpool tail state: one ``[K, gate]`` vector per token.

    This is a sliding tail that occupies one page per request, not a tensor
    ring buffer indexed by position modulo. The state block table maps each
    request's absolute logical pool position to the physical page owned by
    the allocator. The compressor consumes BF16 per-token state.
    """

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
        if dtype != torch.bfloat16:
            raise ValueError(f"GLM-5 compressor state must use bfloat16, got {dtype}.")
        self.state_dim = state_dim
        self.dtype = dtype
        self.prefix = prefix
        self.compress_ratio = compress_ratio
        self.sliding_window = compress_ratio
        self.block_size = compress_ratio
        self.cache_config = cache_config
        self.cache_role = "indexer_state"
        self.kv_cache = [
            torch.tensor([]) for _ in range(get_current_vllm_config().parallel_config.pipeline_parallel_size)
        ]
        compilation_config = get_current_vllm_config().compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        return AscendIndexerKPoolStateSpec(
            block_size=self.block_size,
            num_kv_heads=1,
            head_size=self.state_dim,
            dtype=self.dtype,
            sliding_window=self.sliding_window,
            cache_dtype_str=None,
            model_version="glm5_next",
            cache_role=self.cache_role,
        )

    def get_attn_backend(self) -> type[AttentionBackend]:
        from vllm_ascend.attention.indexer_kpool_mla_v1 import (
            AscendIndexerKPoolStateBackend,
        )

        return AscendIndexerKPoolStateBackend

    def forward(self): ...
