# SPDX-License-Identifier: Apache-2.0
"""Logical layerwise KV placement. Physical storage belongs to the allocator."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, replace

import regex as re
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheTensor


@dataclass(frozen=True)
class CacheBucket:
    block_stride: int
    resident_slots: tuple[tuple[str, ...], ...]
    reuse_slots: tuple[tuple[str, ...], ...]


@dataclass(frozen=True)
class CachePlacement:
    resident_names: frozenset[str]
    # Buffer numbers are local to a block-stride bucket. Each group starts
    # round-robin allocation at buffer zero after removing resident caches.
    buffer_by_name: dict[str, int]
    # Owners with a temporal alias in the same scheduler block-ID lifetime.
    rotating_names: frozenset[str]
    predecessors: dict[int, tuple[int, ...]]
    prefetch_layers: int
    buckets: tuple[CacheBucket, ...]


@dataclass
class LayerwiseKVCacheConfig(KVCacheConfig):
    reuse_config: dict[str, bool] | None = None
    placement: CachePlacement | None = None


def _relative_cache_name(name: str) -> str:
    match = re.search(r"(?:^|\.)layers\.\d+\.(.+)$", name)
    return match.group(1) if match else name


def plan_cache_placement(
    config: KVCacheConfig,
    reuse_config: Mapping[str, bool],
    layer_indices: Mapping[str, int],
    independent_layers: set[int],
    excluded_names: set[str],
    num_buffers: int,
    prefetch_layers: int,
) -> CachePlacement:
    """Split resident caches, bucket by stride, and rotate within each group."""
    if num_buffers < 1 or prefetch_layers < 1:
        raise ValueError("Layerwise buffer count and prefetch distance must be positive")
    groups = {name: i for i, group in enumerate(config.kv_cache_groups) for name in group.layer_names}
    resident = {
        name
        for name, layer in layer_indices.items()
        if layer in independent_layers
        or name in excluded_names
        or not reuse_config.get(_relative_cache_name(name), False)
    }
    seen = set()
    resident_tensors = []
    rotating_tensors = []
    for tensor in config.kv_cache_tensors:
        resident_layers = []
        rotating_layers = []
        for name in tensor.layers:
            if name in seen or name not in groups:
                raise ValueError(f"Duplicate or unknown cache descriptor: {name}")
            seen.add(name)
            (resident_layers if name in resident else rotating_layers).append(name)
        if resident_layers:
            resident_tensors.append(replace(tensor, layers=resident_layers))
        if rotating_layers:
            rotating_tensors.append(replace(tensor, layers=rotating_layers))
    if seen != groups.keys() or seen != layer_indices.keys():
        raise ValueError("Layerwise descriptors must cover every cache owner")

    # Both partitions use the same stride buckets. Different scheduler groups
    # can overlay slots because they own disjoint active physical block IDs.
    resident_groups: dict[int, dict[int, list[str]]] = {}
    rotating_groups: dict[int, dict[int, list[str]]] = {}
    for tensors, partition in ((resident_tensors, resident_groups), (rotating_tensors, rotating_groups)):
        for tensor in tensors:
            if tensor.block_stride <= 0:
                raise ValueError("Layerwise cache block strides must be positive")
            bucket = partition.setdefault(tensor.block_stride, {})
            for name in tensor.layers:
                bucket.setdefault(groups[name], []).append(name)

    buffers = {}
    rotating = set()
    predecessors: dict[int, set[int]] = defaultdict(set)
    buckets = []
    for stride in dict.fromkeys((*resident_groups, *rotating_groups)):
        resident_slots: list[list[str]] = []
        for names in resident_groups.get(stride, {}).values():
            names.sort(key=layer_indices.__getitem__)
            for slot, name in enumerate(names):
                if slot == len(resident_slots):
                    resident_slots.append([])
                resident_slots[slot].append(name)

        reuse_slots: list[list[str]] = []
        for names in rotating_groups.get(stride, {}).values():
            names.sort(key=layer_indices.__getitem__)
            by_buffer: dict[int, list[str]] = defaultdict(list)
            for index, name in enumerate(names):
                buffer = index % num_buffers
                buffers[name] = buffer
                if buffer == len(reuse_slots):
                    reuse_slots.append([])
                reuse_slots[buffer].append(name)
                by_buffer[buffer].append(name)
            for members in by_buffer.values():
                if len(members) < 2:
                    continue
                rotating.update(members)
                for previous, current in zip(members, members[1:]):
                    source, target = layer_indices[previous], layer_indices[current]
                    if source == target:
                        raise ValueError(f"Simultaneous caches cannot reuse one buffer: {previous}, {current}")
                    predecessors[target].add(source)
        buckets.append(CacheBucket(stride, tuple(map(tuple, resident_slots)), tuple(map(tuple, reuse_slots))))
    return CachePlacement(
        frozenset(resident),
        buffers,
        frozenset(rotating),
        {layer: tuple(sorted(sources)) for layer, sources in predecessors.items()},
        prefetch_layers,
        tuple(buckets),
    )


def build_layerwise_cache_tensors(
    config: LayerwiseKVCacheConfig, *, alignment: int = 1, excluded_names: frozenset[str] = frozenset()
) -> tuple[list[KVCacheTensor], int]:
    """Place resident slots followed by reuse slots in each stride bucket."""
    assert config.placement is not None
    templates = {name: tensor for tensor in config.kv_cache_tensors for name in tensor.layers}
    tensors = []
    cursor = 0
    for bucket in config.placement.buckets:
        for resident, slots in ((True, bucket.resident_slots), (False, bucket.reuse_slots)):
            run = None
            for slot in slots:
                names = [name for name in slot if name not in excluded_names]
                if not names:
                    continue
                cursor = (cursor + alignment - 1) // alignment * alignment
                slot_size = bucket.block_stride * config.num_blocks
                if (
                    resident
                    and len(names) == 1
                    and run is not None
                    and templates[run.layers[0]] is templates[names[0]]
                    and cursor == run.offset + len(run.layers) * slot_size
                ):
                    run.layer_stride = slot_size
                    run.layers.extend(names)
                else:
                    tensor = replace(templates[names[0]], layers=names, offset=cursor, layer_stride=0)
                    tensors.append(tensor)
                    run = tensor if resident and len(names) == 1 else None
                cursor += slot_size
    size = (cursor + alignment - 1) // alignment * alignment
    return [replace(tensor, size=size) for tensor in tensors], size
