# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Serializable slot assignments; geometry remains in cache descriptors."""

from copy import copy
from dataclasses import dataclass, fields, replace
from functools import cached_property

from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheSpec, KVCacheTensor, UniformTypeKVCacheSpecs

import vllm_ascend.patch.patch_layerwise_cache_fields  # noqa: F401


def get_scheduler_spec(spec: KVCacheSpec) -> KVCacheSpec:
    if isinstance(spec, UniformTypeKVCacheSpecs):
        return next(iter(spec.kv_cache_specs.values()))
    return spec


def scheduler_geometry_config(config: KVCacheConfig) -> KVCacheConfig:
    """Give scalar readers their normal group shape without touching published specs."""
    if not any(isinstance(group.kv_cache_spec, UniformTypeKVCacheSpecs) for group in config.kv_cache_groups):
        return config
    geometry = copy(config)
    geometry.kv_cache_groups = [
        replace(group, kv_cache_spec=get_scheduler_spec(group.kv_cache_spec)) for group in config.kv_cache_groups
    ]
    clear_cached_group_properties(geometry)
    return geometry


def clear_cached_group_properties(config: KVCacheConfig) -> None:
    """Invalidate derived group lists after changing the group specs."""
    for cls in type(config).__mro__:
        for name, attr in vars(cls).items():
            if isinstance(attr, cached_property):
                config.__dict__.pop(name, None)


def kv_cache_allocation_bytes(tensors: list[KVCacheTensor]) -> int:
    """Count each explicitly shared backing once, never deduplicate by size."""
    allocations = {}
    for index, tensor in enumerate(tensors):
        key = ("private", index) if tensor.backing_id is None else ("shared", tensor.backing_id)
        if key in allocations and allocations[key] != tensor.size:
            raise ValueError("Descriptors for one backing disagree about capacity")
        allocations[key] = tensor.size
    return sum(allocations.values())


@dataclass(frozen=True)
class LayerwisePolicy:
    num_shared_buffers: int
    num_prefetch_layers: int
    independent_layers: tuple[int, ...]
    excluded_owners: tuple[str, ...] = ()


@dataclass(frozen=True)
class LayerwiseReuseLayout:
    policy: LayerwisePolicy
    num_blocks: int
    # Each slot is one physical region. A logical buffer can contain several
    # regions for components that the same physical layer uses together.
    buffer_slots: tuple[tuple[str, ...], ...]
    resident_layer_names: frozenset[str]
    predecessors: dict[int, tuple[int, ...]]
    layer_buffers: dict[int, int]

    @property
    def has_layer_reuse(self) -> bool:
        return any(len(slot) > 1 for slot in self.buffer_slots)

    @property
    def rotating_layer_names(self) -> frozenset[str]:
        return frozenset(name for slot in self.buffer_slots if len(slot) > 1 for name in slot)


@dataclass
class LayerwiseKVCacheConfig(KVCacheConfig):
    layerwise_reuse: LayerwiseReuseLayout | None = None


def get_layerwise_layout(config: KVCacheConfig | None) -> LayerwiseReuseLayout | None:
    return getattr(config, "layerwise_reuse", None)


def serialize_layerwise_config(config: KVCacheConfig) -> KVCacheConfig:
    """Declare the placement result at the dataclass IPC boundary."""
    layout = get_layerwise_layout(config)
    if layout is None or isinstance(config, LayerwiseKVCacheConfig):
        return config
    result = LayerwiseKVCacheConfig(
        **{field.name: getattr(config, field.name) for field in fields(KVCacheConfig)},
        layerwise_reuse=layout,
    )
    result.__dict__.update(config.__dict__)
    return result
