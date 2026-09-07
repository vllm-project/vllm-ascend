# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, fields, replace
from typing import Any

import torch
from typing_extensions import Self
from vllm.config import VllmConfig
from vllm.utils.math_utils import cdiv
from vllm.utils.torch_utils import get_dtype_size
from vllm.v1.core.single_type_kv_cache_manager import FullAttentionManager, SlidingWindowManager
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheSpec,
    MLAAttentionSpec,
    SlidingWindowMLASpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.kv_cache_spec_registry import KVCacheSpecRegistry

_MLA_SPEC_FIELDS = {field.name for field in fields(MLAAttentionSpec)}
# vLLM reworked how MLA specs describe cache compression: ``compress_ratio``
# became ``tokens_per_state``, ``storage_block_size`` turned from a derived row
# count into an optional token width, and the ``indexes_kv_by_block_stride``
# opt-in was dropped once padded pages started indexing by block stride
# unconditionally. Detect the layout from the dataclass rather than from the
# version string, which containers routinely patch out of sync with the code.
SPEC_USES_TOKENS_PER_STATE = "tokens_per_state" in _MLA_SPEC_FIELDS
SPEC_HAS_BLOCK_STRIDE_INDEXING = "indexes_kv_by_block_stride" in _MLA_SPEC_FIELDS

_COMPRESSION_FIELD = "tokens_per_state" if SPEC_USES_TOKENS_PER_STATE else "compress_ratio"


def spec_compress_ratio(kv_cache_spec: KVCacheSpec) -> int:
    """Return how many tokens one physical KV row of ``kv_cache_spec`` holds."""
    return getattr(kv_cache_spec, _COMPRESSION_FIELD, 1)


def optional_spec_compress_ratio(kv_cache_spec: KVCacheSpec) -> int | None:
    """Return the compression ratio, or ``None`` if the spec declares none.

    Only MLA specs carry cache compression. Callers treat ``None`` as "ask the
    model config instead", which a plain ratio of 1 must not be confused with.
    """
    if not isinstance(kv_cache_spec, (MLAAttentionSpec, SlidingWindowMLASpec)):
        return None
    return spec_compress_ratio(kv_cache_spec)


def compression_kwargs(compress_ratio: int) -> dict[str, Any]:
    """Spec kwargs declaring ``compress_ratio`` tokens per physical KV row."""
    return {_COMPRESSION_FIELD: compress_ratio}


def serialized_spec_compress_ratio(serialized_spec: Mapping[str, Any]) -> int | None:
    """Return the compression ratio of a spec serialized for a peer.

    A PD peer may run either spec layout, so both field names are accepted.
    Returns ``None`` when the spec declares no compression at all.
    """
    for field_name in ("tokens_per_state", "compress_ratio"):
        value = serialized_spec.get(field_name)
        if isinstance(value, int):
            return max(1, value)
    return None


def block_stride_indexing_kwargs(enabled: bool) -> dict[str, bool]:
    """Spec kwargs opting padded pages into block-stride KV indexing."""
    return {"indexes_kv_by_block_stride": enabled} if SPEC_HAS_BLOCK_STRIDE_INDEXING else {}


def _spec_storage_block_size(kv_cache_spec: KVCacheSpec) -> int:
    """Return the physical KV rows one scheduler block addresses."""
    token_width = getattr(kv_cache_spec, "storage_block_size", None)
    if not SPEC_USES_TOKENS_PER_STATE:
        # Here ``storage_block_size`` is already a row count, exposed by vLLM
        # for every spec kind.
        return token_width if token_width is not None else kv_cache_spec.block_size
    if token_width is None:
        # ``None`` means the storage is viewed in whole kernel blocks.
        token_width = kv_cache_spec.block_size
    compress_ratio = spec_compress_ratio(kv_cache_spec)
    # Some models tag "uncompressed" with a non-positive ratio, matching the
    # guard in vLLM's own ``get_num_kernel_states``.
    return token_width // compress_ratio if compress_ratio > 0 else token_width


@dataclass(frozen=True)
class AliasedKVCacheTensor:
    """One allocation together with every layer that shares its bytes."""

    size: int
    shared_by: list[str]


def aliased_kv_cache_tensors(
    kv_cache_config: KVCacheConfig,
) -> list[AliasedKVCacheTensor]:
    """Describe the KV cache allocation as sets of layers that share bytes.

    vLLM used to say this directly: a ``KVCacheTensor`` was one allocation of
    ``size`` bytes plus the ``shared_by`` layers that alias it. It now describes
    placements into a common backing allocation instead -- layer ``l`` of an
    entry starts at ``offset + l * layer_stride``, and aliasing is expressed by
    two placements landing on the same address rather than by a shared list.

    Recover the old view, which is what the allocator below still wants: expand
    each entry to one placement per layer, then group the placements by address.
    Layers at the same address are exactly the old ``shared_by`` set -- for
    GLM-5.3-Flash that is each MLA layer overlaid with its Mamba partner, and
    each indexer layer overlaid with its kpool tail.
    """
    tensors = kv_cache_config.kv_cache_tensors
    if not tensors or hasattr(tensors[0], "shared_by"):
        return tensors

    num_blocks = kv_cache_config.num_blocks
    layers_at: dict[int, list[str]] = defaultdict(list)
    bytes_at: dict[int, int] = {}
    for tensor in tensors:
        layer_bytes = num_blocks * tensor.block_stride
        assert len(tensor.layers) == 1 or tensor.layer_stride == layer_bytes, (
            "Ascend's KV cache allocator only handles layer-outermost layouts, "
            f"got layer_stride={tensor.layer_stride} for {len(tensor.layers)} "
            f"layers of {layer_bytes} bytes each."
        )
        for index, layer_name in enumerate(tensor.layers):
            offset = tensor.offset + index * tensor.layer_stride
            layers_at[offset].append(layer_name)
            bytes_at[offset] = max(bytes_at.get(offset, 0), layer_bytes)

    return [AliasedKVCacheTensor(size=bytes_at[offset], shared_by=layers) for offset, layers in layers_at.items()]


def get_storage_block_size(kv_cache_spec: KVCacheSpec) -> int:
    """Return the physical token rows represented by one scheduler block."""
    if isinstance(kv_cache_spec, UniformTypeKVCacheSpecs):
        storage_block_sizes = {_spec_storage_block_size(spec) for spec in kv_cache_spec.kv_cache_specs.values()}
        assert len(storage_block_sizes) == 1, "All specs in one KV cache group must use the same storage block size."
        return storage_block_sizes.pop()
    return _spec_storage_block_size(kv_cache_spec)


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
                spec_compress_ratio(spec),
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
        return replace(
            merged,
            scale_dim=first_spec.scale_dim,
            scale_dtype=first_spec.scale_dtype,
            alignment=first_spec.alignment,
            cache_sparse_sfa_c8=first_spec.cache_sparse_sfa_c8,
            store_on_host=first_spec.store_on_host,
            **block_stride_indexing_kwargs(getattr(first_spec, "indexes_kv_by_block_stride", False)),
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
    model_version: str | None = None

    def __post_init__(self):
        pass

    @property
    def real_page_size_bytes(self) -> int:
        return get_storage_block_size(self) * self.num_kv_heads * self.head_size * get_dtype_size(self.dtype)

    @classmethod
    def merge(cls, specs: list[Self]) -> Self:
        assert all(isinstance(spec, AscendSlidingWindowMLASpec) for spec in specs), (
            "All attention layers in the same KV cache group must be AscendSlidingWindowMLASpec."
        )
        cache_dtype_str_set = set(spec.cache_dtype_str for spec in specs)
        compress_ratio_set = set(spec_compress_ratio(spec) for spec in specs)
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
            model_version=model_version_set.pop(),
            **compression_kwargs(compress_ratio_set.pop()),
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
