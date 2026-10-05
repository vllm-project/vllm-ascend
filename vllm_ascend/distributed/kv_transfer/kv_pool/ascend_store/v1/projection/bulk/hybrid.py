"""Registered group facts for coordinated multi-group and align-state Bulk."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import PoolKey

from ...topology import KVPoolGroupTopology, KVPoolTopology
from ..reachability import HybridReachability
from .common import (
    BulkRangeBatch,
    ByteArray,
    KeyAxes,
    bind_contiguous_layout,
    contiguous_ranges,
    key_prefixes,
    local_prefix,
    lookup_prefixes,
)


@dataclass(frozen=True, slots=True)
class HybridStoreAxis:
    """One Backend key axis paired with the local entries it publishes."""

    key_prefix: str
    entry_slice: tuple[int, int]
    physical_layer_ids: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class HybridBulkGroupProjection:
    """Registration-bound identity and memory facts for one original group id."""

    dense_index: int
    group_id: int
    block_size: int
    hash_block_size: int
    physical_layer_ids: tuple[int, ...]
    uses_align_state: bool
    minimum_block_id: int
    local_key_prefixes: tuple[str, ...]
    lookup_key_prefixes: tuple[str, ...]
    store_axes: tuple[HybridStoreAxis, ...]
    base_addresses: ByteArray
    block_lengths: ByteArray
    block_strides: ByteArray
    object_size: int


@dataclass(frozen=True, slots=True)
class HybridCheckpointPolicy:
    """Groups that own exact state sources and their attention companions."""

    state_group_ids: tuple[int, ...]
    companion_group_ids: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class HybridBulkProjection:
    """Coordinated Bulk facts for heterogeneous or align-state cache groups."""

    groups: tuple[HybridBulkGroupProjection, ...]
    group_ids: tuple[int, ...]
    fine_grained_lookup: bool
    reachability: HybridReachability
    checkpoint_policy: HybridCheckpointPolicy | None

    def group(self, group_id: int) -> HybridBulkGroupProjection:
        for group in self.groups:
            if group.group_id == group_id:
                return group
        raise KeyError(group_id)


def bind_hybrid_bulk_projection(
    topology: KVPoolTopology,
    max_model_len: int,
    base_addresses: Mapping[int, Sequence[int]],
    block_lengths: Mapping[int, Sequence[int]],
    block_strides: Mapping[int, Sequence[int]],
    layer_entry_offsets: Mapping[int, Sequence[int]],
    *,
    use_eagle: bool,
    retention_interval: int | None,
) -> HybridBulkProjection:
    transfer_groups = topology.transfer_groups
    if topology.tp_partition.tp_mismatch:
        raise ValueError("Hybrid Bulk projection cannot be composed with TP mismatch")
    if len(transfer_groups) == 1 and not transfer_groups[0].uses_align_state:
        raise ValueError("Hybrid Bulk projection requires coordinated groups or align-state")

    bound_groups = tuple(
        _bind_group(
            topology,
            dense_index,
            group,
            base_addresses,
            block_lengths,
            block_strides,
            layer_entry_offsets,
        )
        for dense_index, group in enumerate(transfer_groups)
    )
    state_group_ids = tuple(group.group_id for group in bound_groups if group.uses_align_state)
    checkpoint_policy = (
        HybridCheckpointPolicy(
            state_group_ids,
            tuple(group.group_id for group in bound_groups if not group.uses_align_state),
        )
        if state_group_ids
        else None
    )
    fine_grained_lookup = any(
        group.uses_align_state and group.block_size > topology.hash_block_size for group in bound_groups
    )
    return HybridBulkProjection(
        groups=bound_groups,
        group_ids=tuple(group.group_id for group in bound_groups),
        fine_grained_lookup=fine_grained_lookup,
        reachability=HybridReachability(
            transfer_groups,
            scheduler_block_size=topology.cache_transfer_granularity,
            hash_block_size=topology.hash_block_size,
            max_model_len=max_model_len,
            use_eagle=use_eagle,
            retention_interval=retention_interval,
        ),
        checkpoint_policy=checkpoint_policy,
    )


def hybrid_load_keys(group: HybridBulkGroupProjection, hashes) -> KeyAxes:
    return key_prefixes(group.local_key_prefixes, hashes)


def hybrid_store_keys(group: HybridBulkGroupProjection, hashes) -> KeyAxes:
    return key_prefixes(tuple(axis.key_prefix for axis in group.store_axes), hashes)


def hybrid_lookup_keys(group: HybridBulkGroupProjection, hashes) -> KeyAxes:
    return key_prefixes(group.lookup_key_prefixes, hashes)


def hybrid_bulk_ranges(
    group: HybridBulkGroupProjection,
    block_ids,
    token_counts,
    selected_objects=None,
    *,
    store: bool,
) -> BulkRangeBatch:
    entry_slices = tuple(axis.entry_slice for axis in group.store_axes) if store else ((0, len(group.base_addresses)),)
    return contiguous_ranges(
        block_ids,
        token_counts,
        selected_objects=selected_objects,
        bases=group.base_addresses,
        lengths=group.block_lengths,
        strides=group.block_strides,
        block_size=group.block_size,
        entry_slices=entry_slices,
        full_extent=group.uses_align_state,
    )


def _bind_group(
    topology: KVPoolTopology,
    dense_index: int,
    group: KVPoolGroupTopology,
    base_addresses: Mapping[int, Sequence[int]],
    block_lengths: Mapping[int, Sequence[int]],
    block_strides: Mapping[int, Sequence[int]],
    layer_entry_offsets: Mapping[int, Sequence[int]],
) -> HybridBulkGroupProjection:
    try:
        bases, lengths, strides = bind_contiguous_layout(
            group,
            base_addresses[group.group_id],
            block_lengths[group.group_id],
            block_strides[group.group_id],
            layer_entry_offsets[group.group_id],
        )
        layer_offsets = layer_entry_offsets[group.group_id]
    except KeyError as error:
        raise RuntimeError(f"KV cache group {group.group_id} has not registered local memory") from error

    physical_layers = tuple(layer.physical_layer_id for layer in group.layers)
    return HybridBulkGroupProjection(
        dense_index=dense_index,
        group_id=group.group_id,
        block_size=group.block_size,
        hash_block_size=topology.hash_block_size,
        physical_layer_ids=physical_layers,
        uses_align_state=group.uses_align_state,
        minimum_block_id=1 if group.uses_align_state else 0,
        local_key_prefixes=(local_prefix(group),),
        lookup_key_prefixes=lookup_prefixes(topology, group),
        store_axes=_bind_store_axes(
            topology,
            group,
            physical_layers,
            layer_offsets,
            len(bases),
        ),
        base_addresses=bases,
        block_lengths=lengths,
        block_strides=strides,
        object_size=int(lengths.sum()),
    )


def _bind_store_axes(
    topology: KVPoolTopology,
    group: KVPoolGroupTopology,
    physical_layers: tuple[int, ...],
    layer_offsets: Sequence[int],
    entry_count: int,
) -> tuple[HybridStoreAxis, ...]:
    partitions = topology.consumer_pipeline_partitions
    if partitions is None or len(partitions) <= 1:
        return (
            HybridStoreAxis(
                local_prefix(group),
                (0, entry_count),
                physical_layers,
            ),
        )

    layer_bounds = {
        layer_id: (int(layer_offsets[index]), int(layer_offsets[index + 1]))
        for index, layer_id in enumerate(physical_layers)
    }
    axes: list[HybridStoreAxis] = []
    first_layer = 0
    for pp_rank, layer_count in enumerate(partitions):
        last_layer = first_layer + layer_count
        selected_layers = tuple(layer_id for layer_id in physical_layers if first_layer <= layer_id < last_layer)
        first_layer = last_layer
        if not selected_layers:
            continue
        axes.append(
            HybridStoreAxis(
                PoolKey(replace(group.key_metadata, pp_rank=pp_rank), "").to_string(),
                (layer_bounds[selected_layers[0]][0], layer_bounds[selected_layers[-1]][1]),
                selected_layers,
            )
        )
    if not axes or sum(axis.entry_slice[1] - axis.entry_slice[0] for axis in axes) != entry_count:
        raise ValueError(
            f"Consumer pipeline partitions do not cover the registered entries for cache group {group.group_id}"
        )
    return tuple(axes)


__all__ = (
    "HybridBulkGroupProjection",
    "HybridBulkProjection",
    "HybridCheckpointPolicy",
    "HybridStoreAxis",
    "bind_hybrid_bulk_projection",
    "hybrid_bulk_ranges",
    "hybrid_load_keys",
    "hybrid_lookup_keys",
    "hybrid_store_keys",
)
