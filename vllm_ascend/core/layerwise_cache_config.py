# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Serializable slot assignments; geometry remains in cache descriptors."""

from dataclasses import dataclass, fields

from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheTensor


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
    # Slots are independently scoped to their original storage domain/bucket.
    buffer_slots: tuple[tuple[str, ...], ...]
    resident_layer_names: frozenset[str]
    predecessors: dict[int, tuple[int, ...]]

    @property
    def has_layer_reuse(self) -> bool:
        # Disjoint group block IDs can share a slot without an overwrite
        # predecessor. Placement still needs to materialize that shared slot.
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
