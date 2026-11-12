# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared cache types, configuration values, byte views and allocation."""

from __future__ import annotations

from collections.abc import Callable, Collection, Mapping
from dataclasses import dataclass
from math import lcm
from typing import Any

import regex as re
import torch
from vllm.utils.torch_utils import get_dtype_size
from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    KVCacheSpec,
    KVCacheTensor,
    UniformTypeKVCacheSpecs,
)

from vllm_ascend.core.kv_cache_interface import AscendSFAIndexerCacheSpec

SpecReuseRules = Mapping[type[KVCacheSpec], bool | Callable[[KVCacheSpec], bool]]


@dataclass(frozen=True)
class LayerwiseReuseLayout:
    layers: tuple[int, ...]
    cache_layer_indices: dict[str, int]
    kv_cache_tensors: tuple[KVCacheTensor, ...]
    # Preserve each native physical window, including padding and components.
    cache_view_sizes: dict[str, int]
    # Target layer -> source layers whose saves must finish before buffer reuse.
    # A target may reuse several buffers, e.g. {4: (0, 2)} waits for both saves.
    # Empty saves signal completion independently, so waiting only for the
    # largest source index does not guarantee that earlier saves have finished.
    prefetch_layer_map: dict[int, tuple[int, ...]]
    independent_layers: list[int]
    num_prefetch_layers: int
    reused_cache_names: frozenset[str]

    @property
    def has_layer_reuse(self) -> bool:
        return bool(self.prefetch_layer_map)


_NUM_SHARED_BUFFERS = "layerwise_num_shared_buffers"
_PREFETCH_LAYERS = "layerwise_prefetch_layers"
_INDEPENDENT_LAYERS = "layerwise_independent_layers"
_DEFAULT_MAX_PREFETCH_LAYERS = 8


def allocate_shared_indexer_cache(
    specs: dict[str, AscendSFAIndexerCacheSpec],
    num_blocks: int,
    allocate: Callable[[int], torch.Tensor],
) -> dict[str, tuple[torch.Tensor, ...]]:
    """Share raw bytes while retaining each layer's data and scale layout.

    The caller must fence reuse between layers. Returned byte views contain
    exactly the layer's logical capacity, so reshape and transfer code never
    mistake spare backing capacity for extra blocks.
    """
    regions: dict[str, tuple[int, int, int]] = {}
    capacity = 0
    for name, spec in specs.items():
        rows = num_blocks * spec.sfa_dcp_replicated_indexer_size * spec.block_size * spec.num_kv_heads
        data_bytes = rows * spec.head_size * get_dtype_size(spec.dtype)
        scale_bytes = rows * spec.scale_dim * get_dtype_size(spec.scale_dtype)
        scale_alignment = get_dtype_size(spec.scale_dtype)
        scale_offset = ((data_bytes + scale_alignment - 1) // scale_alignment) * scale_alignment
        regions[name] = (data_bytes, scale_offset, scale_bytes)
        capacity = max(capacity, scale_offset + scale_bytes if scale_bytes else data_bytes)
    backing = allocate(capacity)
    views: dict[str, tuple[torch.Tensor, ...]] = {}
    region_views: dict[tuple[int, int, int], tuple[torch.Tensor, ...]] = {}
    for name, region in regions.items():
        if region not in region_views:
            data_bytes, scale_offset, scale_bytes = region
            data = backing[:data_bytes]
            region_views[region] = (
                (data, backing[scale_offset : scale_offset + scale_bytes]) if scale_bytes else (data,)
            )
        views[name] = region_views[region]
    return views


def cache_view_alignment(spec: KVCacheSpec) -> int:
    dtypes = getattr(spec, "dtypes", (getattr(spec, "dtype", torch.uint8),))
    alignment = lcm(*(get_dtype_size(dtype) for dtype in dtypes))
    if getattr(spec, "scale_dim", 0):
        alignment = lcm(alignment, get_dtype_size(spec.scale_dtype))
    return alignment


def get_cache_views(config: KVCacheConfig, specs: dict[str, KVCacheSpec]) -> dict[str, tuple[int, int, int]]:
    """Validate physical byte windows without collecting packs or planning reuse."""
    if config.num_blocks <= 0:
        raise ValueError("Cache views require at least one block")
    layout = getattr(config, "layerwise_reuse_layout", None)
    view_sizes = layout.cache_view_sizes if getattr(config, "layerwise_kv_cache_reuse", False) else None
    # Cache name -> (descriptor index, byte start, byte end).
    views = {}
    for index, tensor in enumerate(config.kv_cache_tensors):
        if tensor.host_resident:
            continue
        if not tensor.layers or tensor.size <= 0:
            raise ValueError("Cache descriptors require layers and positive backing capacity")
        if tensor.layer_stride < 0 or tensor.block_stride <= 0 or tensor.offset < 0:
            raise ValueError("Cache descriptors require nonnegative offsets and positive block strides")
        for ordinal, name in enumerate(tensor.layers):
            start = tensor.offset + ordinal * tensor.layer_stride
            spec = specs[name]
            alignment = cache_view_alignment(spec)
            if start % alignment or tensor.block_stride % alignment:
                raise ValueError(f"Cache view for {name} is not aligned to its element sizes")
            end = (
                start + view_sizes[name]
                if view_sizes is not None
                else min(start + config.num_blocks * tensor.block_stride, tensor.size)
            )
            required = (
                start + (config.num_blocks - 1) * tensor.block_stride + min(spec.page_size_bytes, tensor.block_stride)
            )
            if end <= start or end > tensor.size or required > end:
                raise ValueError(f"Cache view for {name} exceeds its backing")
            if name in views:
                raise ValueError("Each cache view must occur in exactly one descriptor")
            views[name] = (index, start, end)
    return views


def allocate_cache_backing(
    config: KVCacheConfig, specs: dict[str, KVCacheSpec], allocate: Callable[[int], torch.Tensor]
) -> dict[str, torch.Tensor]:
    """Allocate one backing and preserve identical byte views for aliased layers."""
    if any(tensor.host_resident for tensor in config.kv_cache_tensors):
        raise NotImplementedError("Explicit cache backing allocation requires device-resident tensors")
    views = get_cache_views(config, specs)
    sizes = {tensor.size for tensor in config.kv_cache_tensors}
    if len(sizes) != 1:
        raise ValueError("Layerwise cache descriptors must share one backing capacity")
    backing = allocate(sizes.pop())
    regions = {}
    result = {}
    for name, (_, start, end) in views.items():
        region = (start, end)
        if region not in regions:
            regions[region] = backing[start:end]
        result[name] = regions[region]
    return result


def get_physical_layer_index(layer_name: str, base_layers: int) -> int:
    match = re.search(
        r"(?:^|\.)mtp(?:\.layers)?\.(\d+)(?:\.|$)",
        layer_name,
    )
    if match:
        return base_layers + int(match.group(1))
    match = re.search(r"layers\.(\d+)", layer_name)
    if match:
        return int(match.group(1))
    match = re.search(r"(\d+)", layer_name)
    return int(match.group(1)) if match else 0


@dataclass(frozen=True)
class LayerwiseCacheConfig:
    num_shared_buffers: int | None
    num_prefetch_layers: int
    independent_layers: list[int]

    @classmethod
    def parse(
        cls,
        num_layers: int,
        extra_config: dict[str, Any] | None = None,
        *,
        stage_layers: Collection[int] | None = None,
    ) -> LayerwiseCacheConfig:
        """Parse model-global layer indices and project them onto the stage, if supplied."""
        stage_num_layers = num_layers if stage_layers is None else len(stage_layers)
        shared_buffers_value = extra_config.get(_NUM_SHARED_BUFFERS) if extra_config else None
        num_shared_buffers = None
        if shared_buffers_value is None:
            if num_layers < 1:
                raise ValueError("num_layers must be at least 1")
        else:
            num_shared_buffers = _parse_int_config(shared_buffers_value, _NUM_SHARED_BUFFERS)
            if num_shared_buffers < 1:
                raise ValueError(f"{_NUM_SHARED_BUFFERS} must be at least 1")

        prefetch_value = extra_config.get(_PREFETCH_LAYERS) if extra_config else None
        if prefetch_value is None:
            num_prefetch_layers = min(num_shared_buffers or max(1, stage_num_layers), _DEFAULT_MAX_PREFETCH_LAYERS)
        else:
            num_prefetch_layers = _parse_int_config(prefetch_value, _PREFETCH_LAYERS)
            if num_prefetch_layers < 1:
                raise ValueError(f"{_PREFETCH_LAYERS} must be at least 1")

        independent_value = extra_config.get(_INDEPENDENT_LAYERS) if extra_config else None
        if independent_value is None:
            layer_indices = [0]
        elif isinstance(independent_value, str) and independent_value.strip().lower() == "all":
            layer_indices = list(range(num_layers))
        elif isinstance(independent_value, list):
            layer_indices = [_parse_int_config(index, _INDEPENDENT_LAYERS) for index in independent_value]
        else:
            raise TypeError(f"{_INDEPENDENT_LAYERS} must be a list of integers or 'all'")

        normalized_indices = set()
        for layer_index in layer_indices:
            if layer_index < 0:
                layer_index += num_layers
            if layer_index < 0 or layer_index >= num_layers:
                raise ValueError(
                    f"{_INDEPENDENT_LAYERS} contains out-of-range layer index "
                    f"{layer_index}; valid range is [0, {num_layers - 1}]"
                )
            normalized_indices.add(layer_index)
        if stage_layers is not None:
            normalized_indices.intersection_update(stage_layers)
        independent_layers = sorted(normalized_indices)

        return cls(
            num_shared_buffers=num_shared_buffers,
            num_prefetch_layers=num_prefetch_layers,
            independent_layers=independent_layers,
        )


def _parse_int_config(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, got bool")
    try:
        return int(value)
    except (TypeError, ValueError) as err:
        raise TypeError(f"{name} must be an integer, got {value!r}") from err


def get_layerwise_kv_cache_specs(
    kv_cache_config: KVCacheConfig,
) -> dict[str, KVCacheSpec]:
    """Expand group specs into a cache spec for every logical layer."""
    layer_specs: dict[str, KVCacheSpec] = {}
    for group in kv_cache_config.kv_cache_groups:
        group_spec = group.kv_cache_spec
        for layer_name in group.layer_names:
            if isinstance(group_spec, UniformTypeKVCacheSpecs):
                layer_specs[layer_name] = group_spec.kv_cache_specs[layer_name]
            else:
                layer_specs[layer_name] = group_spec
    return layer_specs
