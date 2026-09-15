# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KV cache layers and metadata helpers for the GLM-Next pooled indexer."""

from collections.abc import Sequence
from typing import Any, ClassVar

import torch
from torch import nn
from vllm.config import CacheConfig, VllmConfig, get_current_vllm_config
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import BlockHashList, KVCacheBlock
from vllm.v1.core.single_type_kv_cache_manager import FullAttentionManager
from vllm.v1.kv_cache_interface import KVCacheSpec
from vllm.v1.request import Request

from vllm_ascend.core.kv_cache_interface import (
    AscendIndexerKPoolTailSpec,
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
        self.block_size = compress_ratio if ring_capacity is None else ring_capacity
        if self.block_size < compress_ratio or head_dim <= 0:
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

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        del vllm_config
        return AscendIndexerKPoolTailSpec(
            block_size=self.block_size,
            num_kv_heads=1,
            head_size=self.head_dim,
            dtype=self.dtype,
            sliding_window=self.compress_ratio,
            compress_ratio=self.compress_ratio,
        )

    def get_attn_backend(self):
        from vllm_ascend.attention.indexer_kpool import (
            AscendIndexerKPoolTailBackend,
        )

        return AscendIndexerKPoolTailBackend

    def forward(self): ...


class KpoolTailManager(FullAttentionManager):
    """Own one unshared ring block until request completion or preemption.

    Prefix hits never initialize this transient cache. Logical pool-aligned
    prefix lookup lets the next forward seed it without historical tail reads.
    """

    supports_fine_grained_hash_lookup: ClassVar[bool] = False

    def __init__(
        self,
        kv_cache_spec: KVCacheSpec,
        block_pool: BlockPool,
        enable_caching: bool,
        kv_cache_group_id: int,
        scheduler_block_size: int,
        **kwargs: Any,
    ) -> None:
        # The global pool can cache other groups; this manager never does.
        # Disable it before entering the base constructor, independently of
        # coordinator-side filtering and the caller's global caching setting.
        super().__init__(
            kv_cache_spec=kv_cache_spec,
            block_pool=block_pool,
            enable_caching=False,
            kv_cache_group_id=kv_cache_group_id,
            scheduler_block_size=scheduler_block_size,
            **kwargs,
        )

    @classmethod
    def find_longest_cache_hit(
        cls,
        block_hashes: BlockHashList,
        max_length: int,
        kv_cache_group_ids: list[int],
        block_pool: BlockPool,
        kv_cache_spec: KVCacheSpec,
        drop_eagle_block: bool,
        alignment_tokens: int,
        dcp_world_size: int = 1,
        pcp_world_size: int = 1,
    ) -> tuple[tuple[list[KVCacheBlock], ...], int]:
        return tuple([] for _ in kv_cache_group_ids), 0

    def cache_blocks(
        self,
        request: Request,
        num_tokens: int,
        retention_interval: int | None = None,
        *,
        replay_boundary: int | None = None,
        replay_boundaries: Sequence[int] | None = None,
    ) -> None:
        # Upstream uses either replay boundary spelling; neither path publishes
        # this request-private ring to the shared block cache.
        return

    def get_num_common_prefix_blocks(self, running_request_id: str) -> int:
        return 0

    def get_num_blocks_to_allocate(
        self,
        request_id: str,
        num_tokens: int,
        new_computed_blocks: Sequence[KVCacheBlock],
        total_computed_tokens: int,
        num_local_computed_tokens: int,
        num_tokens_main_model: int,
        apply_admission_cap: bool = False,
    ) -> int:
        return 0 if self.req_to_blocks.get(request_id) else 1

    def allocate_new_blocks(self, request_id: str, num_tokens: int, num_tokens_main_model: int) -> list[KVCacheBlock]:
        req_blocks = self.req_to_blocks[request_id]
        if req_blocks:
            return []
        new_blocks = self.block_pool.get_new_blocks(1)
        req_blocks.extend(new_blocks)
        if self._record_new_block_ids:
            self.new_block_ids.extend(block.block_id for block in new_blocks)
        return new_blocks

    def add_local_computed_blocks(
        self,
        request_id: str,
        new_computed_blocks: Sequence[KVCacheBlock],
        num_local_computed_tokens: int,
        num_external_computed_tokens: int,
    ) -> None:
        assert not new_computed_blocks, "Tail blocks cannot be shared through prefix caching."

    def allocate_external_computed_blocks(
        self, request_id: str, num_local_computed_tokens: int, num_external_computed_tokens: int
    ) -> None:
        self.allocate_new_blocks(request_id, num_local_computed_tokens + num_external_computed_tokens, 0)
