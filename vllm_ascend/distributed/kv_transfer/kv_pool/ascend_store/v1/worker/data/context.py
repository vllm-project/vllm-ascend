"""Bind shared identities, local geometry and optional physical-layer layouts once."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import torch

from ...backend import LayerwiseAccessKind
from ...backend.naming import WireKeyBuilder
from ...infrastructure.mapping import (
    BlockHashMapping,
    LayerTable,
    LocalTable,
    PartitionId,
    RemoteTable,
    build_gva_remote,
    build_key_range_remote,
    build_mappings,
    local_lengths,
)
from ...infrastructure.topology import KVPoolGroupTopology, KVPoolTopology
from .selection import CachePartition


@dataclass(frozen=True, slots=True)
class TransferContext:
    """Shared bound facts; operations own their inputs, parameters and progress."""

    topology: KVPoolTopology
    groups: dict[int, KVPoolGroupTopology]
    block_hash_mappings: dict[int, BlockHashMapping]
    key_parts: dict[int, dict[CachePartition, tuple[str, str]]]
    partition_order: dict[CachePartition, int]
    local: dict[int, LocalTable]
    sizes: dict[int, dict[PartitionId, tuple[int, ...]]]
    object_sizes: dict[CachePartition, int]
    state_group_ids: frozenset[int]
    layers: dict[int, LayerTable]
    remote: dict[int, RemoteTable]
    layer_ids_by_name: dict[str, int]
    read_at_compute_start: bool


def bind_context(
    topology: KVPoolTopology,
    kv_caches: Mapping[str, torch.Tensor | Sequence[torch.Tensor]],
    kv_cache_layout: str,
    *,
    layerwise_access: LayerwiseAccessKind | None = None,
    key_builder: WireKeyBuilder | None = None,
) -> TransferContext:
    groups = {group.group_id: group for group in topology.transfer_groups}
    key_parts, local, sizes, layers, remote = {}, {}, {}, {}, {}
    layer_ids_by_name: dict[str, int] = {}
    for group_id, group in groups.items():
        group_keys, group_local, group_layers = build_mappings(
            topology,
            kv_caches,
            group_id=group_id,
            kv_cache_layout=kv_cache_layout,
            key_builder=key_builder,
            layerwise=layerwise_access is not None,
        )
        if not any(len(segments["bases"]) for segments in group_local.values()):
            raise ValueError("KV cache group has no registered memory entries")
        key_parts[group_id] = {(group_id, partition): parts for partition, parts in group_keys.items()}
        local[group_id] = group_local
        sizes[group_id] = {
            partition: tuple(np.broadcast_to(local_lengths(segments), segments["bases"].shape).tolist())
            for partition, segments in group_local.items()
        }
        if group_layers is not None:
            layers[group_id] = group_layers
            remote[group_id] = (
                build_gva_remote(topology, group_local, group_layers)
                if layerwise_access is LayerwiseAccessKind.GVA
                else build_key_range_remote(group_local)
            )
        for layer in group.layers:
            for name in layer.layer_names:
                previous = layer_ids_by_name.setdefault(name, layer.physical_layer_id)
                if previous != layer.physical_layer_id:
                    raise ValueError(f"Layer {name!r} has conflicting physical layer IDs")
    if layerwise_access is not None:
        occupied: list[tuple[int, int, int]] = []
        for group_id, partitions in local.items():
            for partition, segments in partitions.items():
                for layer_id, span in layers[group_id][partition[0]].items():
                    occupied.extend(
                        (int(address), int(address) + size, layer_id)
                        for address, size in zip(segments["bases"][span], sizes[group_id][partition][span], strict=True)
                    )
        previous_end, previous_layer = 0, -1
        for start, end, layer_id in sorted(occupied):
            if start < previous_end and layer_id != previous_layer:
                raise ValueError("Prototype Layerwise does not support shared layer buffers")
            if end > previous_end:
                previous_end, previous_layer = end, layer_id
    return TransferContext(
        topology=topology,
        groups=groups,
        block_hash_mappings={
            group_id: BlockHashMapping(group.block_size, topology.config.hash_block_size)
            for group_id, group in groups.items()
        },
        key_parts=key_parts,
        partition_order={
            coordinate: index
            for index, coordinate in enumerate(coordinate for parts in key_parts.values() for coordinate in parts)
        },
        object_sizes={
            (group_id, partition): (
                remote[group_id][partition[0]]["size_bytes"] if group_id in remote else sum(lengths)
            )
            for group_id, partitions in sizes.items()
            for partition, lengths in partitions.items()
        },
        local=local,
        sizes=sizes,
        state_group_ids=topology.config.align_state_group_ids.intersection(groups),
        layers=layers,
        remote=remote,
        layer_ids_by_name=layer_ids_by_name,
        read_at_compute_start=layerwise_access is LayerwiseAccessKind.KEY_RANGE and len(groups) > 1,
    )
