# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass, replace
from typing import Any

import torch
from typing_extensions import Self
from vllm.config import VllmConfig
from vllm.utils.math_utils import cdiv
from vllm.utils.torch_utils import get_dtype_size
from vllm.v1.core.single_type_kv_cache_manager import FullAttentionManager, SlidingWindowManager
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheSpec,
    MLAAttentionSpec,
    SlidingWindowMLASpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.kv_cache_spec_registry import KVCacheSpecRegistry

from vllm_ascend.utils import vllm_version_is


def get_storage_block_size(kv_cache_spec: KVCacheSpec) -> int:
    """Return the physical token rows represented by one scheduler block."""
    if isinstance(kv_cache_spec, UniformTypeKVCacheSpecs):
        storage_block_sizes = {get_storage_block_size(spec) for spec in kv_cache_spec.kv_cache_specs.values()}
        assert len(storage_block_sizes) == 1, "All specs in one KV cache group must use the same storage block size."
        return storage_block_sizes.pop()
    if not vllm_version_is("0.27.1"):
        # vLLM #53906 added an optional MLA storage-view override. It is not
        # Ascend's derived number of physical rows per logical block.
        if isinstance(kv_cache_spec, AscendMLAAttentionSpec):
            return kv_cache_spec.block_size // kv_cache_spec.tokens_per_state
        if isinstance(kv_cache_spec, MLAAttentionSpec):
            storage_block_size = kv_cache_spec.storage_block_size
            return kv_cache_spec.block_size if storage_block_size is None else storage_block_size
    return getattr(kv_cache_spec, "storage_block_size", kv_cache_spec.block_size)


@dataclass(frozen=True, kw_only=True)
class AscendMLAAttentionSpec(MLAAttentionSpec):
    """MLA cache spec with Ascend-specific layout metadata.

    For SFA, this spec describes only the main MLA cache. The indexer K
    tensor, its quantization scale, and DCP replication are described by a
    separate :class:`AscendSFAIndexerCacheSpec`.
    """

    scale_dim: int = 0
    scale_dtype: torch.dtype = torch.int8
    # Sparse C8 changes the main cache into one packed byte tensor. Keep that
    # main-cache property here; indexer-specific C8 properties belong to the
    # indexer spec.
    cache_sparse_sfa_c8: bool = False
    store_on_host: bool = False

    if vllm_version_is("0.27.1"):

        @property
        def storage_block_size(self) -> int:
            """Legacy physical geometry; main uses get_storage_block_size.

            On main, #53906 initializes a dataclass field with this name.
            A read-only property would reject that constructor assignment.
            """
            return self.block_size // self.compress_ratio

    @property
    def real_page_size_bytes(self) -> int:
        return (
            get_storage_block_size(self)
            * self.num_kv_heads
            * (self.head_size * get_dtype_size(self.dtype) + self.scale_dim * get_dtype_size(self.scale_dtype))
        )

    @property
    def unpadded_page_size_bytes(self) -> int:
        return self.real_page_size_bytes

    @classmethod
    def merge(cls, specs: list[Self]) -> Self:
        assert all(isinstance(spec, MLAAttentionSpec) for spec in specs), (
            "All attention layers in the same KV cache group must be MLAAttentionSpec."
        )
        ascend_layouts = {
            (
                spec.scale_dim,
                spec.scale_dtype,
                spec.cache_sparse_sfa_c8,
                spec.store_on_host,
                spec.alignment,
                spec.compress_ratio if vllm_version_is("0.27.1") else spec.tokens_per_state,
            )
            for spec in specs
        }
        assert len(ascend_layouts) == 1, (
            "All attention layers in the same KV cache group must use the same Ascend KV cache layout."
        )
        non_causal_multi_token_decode_set = set(spec.non_causal_multi_token_decode for spec in specs)
        assert len(non_causal_multi_token_decode_set) == 1, (
            "Causal target layers and non-causal multi-token draft layers must use separate KV cache groups."
        )
        first_spec = specs[0]
        merged = super().merge(specs)
        # vLLM #51718 removed AttentionSpec.indexes_kv_by_block_stride on main;
        # only carry it through on the legacy lane.
        merged_kwargs: dict[str, Any] = {}
        if vllm_version_is("0.27.1"):
            merged_kwargs["indexes_kv_by_block_stride"] = first_spec.indexes_kv_by_block_stride
        return replace(
            merged,
            scale_dim=first_spec.scale_dim,
            scale_dtype=first_spec.scale_dtype,
            alignment=first_spec.alignment,
            cache_sparse_sfa_c8=first_spec.cache_sparse_sfa_c8,
            store_on_host=first_spec.store_on_host,
            **merged_kwargs,
        )

    def max_memory_usage_bytes(self, vllm_config: VllmConfig) -> int:
        max_model_len = vllm_config.model_config.max_model_len
        dcp_world_size = vllm_config.parallel_config.decode_context_parallel_size
        # Note(hc): each dcp rank only need save
        # (max_model_len//dcp_world_size) tokens locally.
        if dcp_world_size > 1:
            max_model_len = cdiv(max_model_len, dcp_world_size)
        return cdiv(max_model_len, self.block_size) * self.page_size_bytes


@dataclass(frozen=True, kw_only=True)
class AscendSFAIndexerCacheSpec(MLAAttentionSpec):
    """KV cache spec for SFA indexer K/scale cache.

    The scheduler should treat this as a full-attention-compatible cache so it
    can share block ids with the MLA cache in the same UniformType group. The
    model runner still allocates it as an independent physical cache tensor.
    """

    scale_dim: int = 0
    scale_dtype: torch.dtype = torch.int8
    cache_sparse_li_c8: bool = False
    cache_dtype_str: str | None = None
    sfa_dcp_replicated_indexer_size: int = 1

    @property
    def page_size_bytes(self) -> int:
        return self.real_page_size_bytes

    @property
    def real_page_size_bytes(self) -> int:
        num_heads_per_page = self.block_size * self.num_kv_heads
        return (
            self.sfa_dcp_replicated_indexer_size
            * num_heads_per_page
            * (self.head_size * get_dtype_size(self.dtype) + self.scale_dim * get_dtype_size(self.scale_dtype))
        )

    @classmethod
    def merge(cls, specs: list[Self]) -> Self:
        assert all(isinstance(spec, AscendSFAIndexerCacheSpec) for spec in specs), (
            "All attention layers in the same KV cache group must be AscendSFAIndexerCacheSpec."
        )
        cache_dtype_str_set = set(spec.cache_dtype_str for spec in specs)
        dtype_set = set(spec.dtype for spec in specs)
        scale_dim_set = set(spec.scale_dim for spec in specs)
        scale_dtype_set = set(spec.scale_dtype for spec in specs)
        cache_sparse_li_c8_set = set(spec.cache_sparse_li_c8 for spec in specs)
        sfa_dcp_replicated_indexer_size_set = set(spec.sfa_dcp_replicated_indexer_size for spec in specs)
        assert (
            len(cache_dtype_str_set) == 1
            and len(dtype_set) == 1
            and len(scale_dim_set) == 1
            and len(scale_dtype_set) == 1
            and len(cache_sparse_li_c8_set) == 1
            and len(sfa_dcp_replicated_indexer_size_set) == 1
        ), (
            "All SFA indexer cache layers in the same KV cache group must use "
            "the same dtype, scale layout, quantization method, sparse LI C8 "
            "setting and DCP replication size."
        )
        return cls(
            block_size=specs[0].block_size,
            num_kv_heads=specs[0].num_kv_heads,
            head_size=specs[0].head_size,
            dtype=dtype_set.pop(),
            cache_dtype_str=cache_dtype_str_set.pop(),
            scale_dim=scale_dim_set.pop(),
            scale_dtype=scale_dtype_set.pop(),
            cache_sparse_li_c8=cache_sparse_li_c8_set.pop(),
            sfa_dcp_replicated_indexer_size=sfa_dcp_replicated_indexer_size_set.pop(),
        )


@dataclass(frozen=True, kw_only=True)
class AscendSlidingWindowMLASpec(SlidingWindowMLASpec):
    """Sliding window attention with MLA cache format."""

    cache_dtype_str: str | None = None
    # DeepseekV4-only: see MLAAttentionSpec.model_version.
    alignment: int | None = None  # Default to None for no padding.
    compress_ratio: int = 1
    model_version: str | None = None

    def __post_init__(self):
        pass

    @property
    def storage_block_size(self) -> int:
        return self.block_size // self.compress_ratio

    @property
    def real_page_size_bytes(self) -> int:
        return self.storage_block_size * self.num_kv_heads * self.head_size * get_dtype_size(self.dtype)

    @classmethod
    def merge(cls, specs: list[Self]) -> Self:
        assert all(isinstance(spec, AscendSlidingWindowMLASpec) for spec in specs), (
            "All attention layers in the same KV cache group must be AscendSlidingWindowMLASpec."
        )
        cache_dtype_str_set = set(spec.cache_dtype_str for spec in specs)
        compress_ratio_set = set(spec.compress_ratio for spec in specs)
        model_version_set = set(spec.model_version for spec in specs)
        sliding_window_set = set(spec.sliding_window for spec in specs)
        assert (
            len(cache_dtype_str_set) == 1
            and len(compress_ratio_set) == 1
            and len(model_version_set) == 1
            and len(sliding_window_set) == 1
        ), (
            "All attention layers in the same KV cache group must use the same "
            "quantization method, compress ratio, model version and sliding "
            "window size."
        )
        return cls(
            block_size=specs[0].block_size,
            num_kv_heads=specs[0].num_kv_heads,
            head_size=specs[0].head_size,
            dtype=specs[0].dtype,
            page_size_padded=specs[0].page_size_padded,
            sliding_window=sliding_window_set.pop(),
            cache_dtype_str=cache_dtype_str_set.pop(),
            compress_ratio=compress_ratio_set.pop(),
            model_version=model_version_set.pop(),
        )


def register_ascend_kv_cache_specs() -> None:
    KVCacheSpecRegistry.register(
        kvcache_spec_cls=AscendMLAAttentionSpec,
        manager_class=FullAttentionManager,
        uniform_type_base_spec=FullAttentionSpec,
    )
    KVCacheSpecRegistry.register(
        kvcache_spec_cls=AscendSFAIndexerCacheSpec,
        manager_class=FullAttentionManager,
        uniform_type_base_spec=FullAttentionSpec,
    )
    KVCacheSpecRegistry.register(
        kvcache_spec_cls=AscendSlidingWindowMLASpec,
        manager_class=SlidingWindowManager,
        uniform_type_base_spec=SlidingWindowMLASpec,
    )

    # Imported lazily so this module stays independent of any single model.
    from vllm_ascend.models.glm5next.kv_cache import KpoolTailManager, KpoolTailSpec

    KVCacheSpecRegistry.register(
        kvcache_spec_cls=KpoolTailSpec,
        manager_class=KpoolTailManager,
        uniform_type_base_spec=KpoolTailSpec,
    )
