# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
from collections.abc import Iterable
from dataclasses import dataclass

import torch
from typing_extensions import Self
from vllm.config import VllmConfig
from vllm.utils.math_utils import cdiv
from vllm.utils.torch_utils import get_dtype_size
from vllm.v1.core.single_type_kv_cache_manager import FullAttentionManager, SlidingWindowManager
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheSpec,
    MambaSpec,
    MLAAttentionSpec,
    SlidingWindowMLASpec,
)
from vllm.v1.kv_cache_spec_registry import KVCacheSpecRegistry

from vllm_ascend.core.single_type_kv_cache_manager import CompressAttentionManager


@dataclass(frozen=True)
class AscendMLAMambaGQALayout:
    """Four contiguous planes inside each existing shared raw tensor."""

    block_size: int
    conv_page_size_bytes: int
    recurrent_page_size_bytes: int
    shared_k_page_size_bytes: int
    gqa_v_page_size_bytes: int

    @property
    def mature_page_size_bytes(self) -> int:
        return self.conv_page_size_bytes + self.recurrent_page_size_bytes + self.shared_k_page_size_bytes

    @property
    def page_size_bytes(self) -> int:
        return self.mature_page_size_bytes + self.gqa_v_page_size_bytes


def get_mla_mamba_gqa_layout(
    kv_cache_specs: Iterable[KVCacheSpec],
) -> AscendMLAMambaGQALayout | None:
    """Recognize the MLA/Mamba/GQA geometry that needs one appended V plane.

    The existing Ascend shared tensor already contains contiguous conv,
    recurrent/MLA-NoPE and MLA-RoPE planes. When compact GQA K and V each
    match the RoPE plane, K can reuse that plane and only V needs to extend
    the existing raw page. No new pool, block table or block-ID mapping is
    required.
    """

    specs = list(kv_cache_specs)
    mla_specs = [spec for spec in specs if isinstance(spec, AscendMLAAttentionSpec)]
    mamba_specs = [spec for spec in specs if isinstance(spec, MambaSpec)]
    gqa_specs = [
        spec
        for spec in specs
        if isinstance(spec, FullAttentionSpec)
        and not isinstance(spec, MLAAttentionSpec)
        and spec.page_size_padded is not None
        and spec.page_size_bytes > spec.unpadded_page_size_bytes
    ]
    if (
        not mla_specs
        or not mamba_specs
        or not gqa_specs
        or len(specs) != len(mla_specs) + len(mamba_specs) + len(gqa_specs)
        or any(spec.mamba_cache_mode != "align" for spec in mamba_specs)
    ):
        return None

    block_sizes = {spec.block_size for spec in (*mla_specs, *mamba_specs, *gqa_specs)}
    if len(block_sizes) != 1:
        return None
    block_size = next(iter(block_sizes))

    mamba_layouts = {
        tuple(math.prod(shape) * get_dtype_size(dtype) for shape, dtype in zip(spec.shapes, spec.dtypes, strict=True))
        for spec in mamba_specs
    }
    if len(mamba_layouts) != 1:
        return None
    state_sizes = next(iter(mamba_layouts))
    if len(state_sizes) != 2:
        return None
    conv_page_size, recurrent_page_size = state_sizes

    mla_page_sizes = {spec.unpadded_page_size_bytes for spec in mla_specs}
    if len(mla_page_sizes) != 1:
        return None
    mla_page_size = next(iter(mla_page_sizes))
    shared_k_page_size = mla_page_size - recurrent_page_size
    if shared_k_page_size <= 0:
        return None

    gqa_layouts = {
        (
            spec.block_size * spec.num_kv_heads * spec.head_size * get_dtype_size(spec.dtype),
            spec.block_size * spec.num_kv_heads * (spec.head_size_v or spec.head_size) * get_dtype_size(spec.dtype),
        )
        for spec in gqa_specs
    }
    if gqa_layouts != {(shared_k_page_size, shared_k_page_size)}:
        return None

    layout = AscendMLAMambaGQALayout(
        block_size=block_size,
        conv_page_size_bytes=conv_page_size,
        recurrent_page_size_bytes=recurrent_page_size,
        shared_k_page_size_bytes=shared_k_page_size,
        gqa_v_page_size_bytes=shared_k_page_size,
    )
    current_page_sizes = {spec.page_size_bytes for spec in (*mla_specs, *mamba_specs, *gqa_specs)}
    if current_page_sizes not in (
        {layout.mature_page_size_bytes},
        {layout.page_size_bytes},
    ):
        return None
    return layout


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

    @property
    def real_page_size_bytes(self) -> int:
        return (
            self.block_size
            * self.num_kv_heads
            * (self.head_size * get_dtype_size(self.dtype) + self.scale_dim * get_dtype_size(self.scale_dtype))
        )

    @classmethod
    def merge(cls, specs: list[Self]) -> Self:
        assert all(isinstance(spec, cls) for spec in specs), (
            "All attention layers in the same KV cache group must use AscendMLAAttentionSpec."
        )
        layout_set = {
            (
                spec.block_size,
                spec.num_kv_heads,
                spec.head_size,
                spec.scale_dim,
                spec.scale_dtype,
                spec.dtype,
            )
            for spec in specs
        }
        assert len(layout_set) == 1, (
            "All attention layers in the same KV cache group must use the same KV cache layout."
        )
        cache_dtype_str_set = set(spec.cache_dtype_str for spec in specs)
        assert len(cache_dtype_str_set) == 1, (
            "All attention layers in the same KV cache group must use the same quantization method."
        )
        cache_sparse_sfa_c8_set = set(spec.cache_sparse_sfa_c8 for spec in specs)
        assert len(cache_sparse_sfa_c8_set) == 1, (
            "All attention layers in the same KV cache group must use the same sparse SFA C8 setting."
        )
        first_spec = specs[0]
        return cls(
            block_size=first_spec.block_size,
            num_kv_heads=first_spec.num_kv_heads,
            head_size=first_spec.head_size,
            scale_dim=first_spec.scale_dim,
            scale_dtype=first_spec.scale_dtype,
            dtype=first_spec.dtype,
            kv_quant_mode=first_spec.kv_quant_mode,
            page_size_padded=first_spec.page_size_padded,
            indexes_kv_by_block_stride=first_spec.indexes_kv_by_block_stride,
            cache_dtype_str=first_spec.cache_dtype_str,
            alignment=first_spec.alignment,
            compress_ratio=first_spec.compress_ratio,
            model_version=first_spec.model_version,
            cache_sparse_sfa_c8=first_spec.cache_sparse_sfa_c8,
        )

    def max_memory_usage_bytes(self, vllm_config: VllmConfig) -> int:
        max_model_len = vllm_config.model_config.max_model_len
        dcp_world_size = vllm_config.parallel_config.decode_context_parallel_size
        # Note(hc): each dcp rank only need save
        # (max_model_len//dcp_world_size) tokens locally.
        if dcp_world_size > 1:
            max_model_len = cdiv(max_model_len, dcp_world_size)
        return cdiv(max_model_len, self.block_size * self.compress_ratio) * self.page_size_bytes


@dataclass(frozen=True, kw_only=True)
class AscendSFAIndexerCacheSpec(FullAttentionSpec):
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
        return self.block_size

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
        manager_class=CompressAttentionManager,
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
