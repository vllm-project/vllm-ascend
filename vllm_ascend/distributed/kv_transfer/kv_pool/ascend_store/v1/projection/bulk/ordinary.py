"""Static key and contiguous-memory facts for ordinary single-group Bulk."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from ...topology import KVPoolTopology
from ..reachability import UnitaryReachability
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
class OrdinaryBulkProjection:
    """Registration-bound formula inputs for one ordinary Bulk cache group."""

    group_id: int
    block_size: int
    hash_block_size: int
    physical_layer_ids: tuple[int, ...]
    local_key_prefixes: tuple[str, ...]
    lookup_key_prefixes: tuple[str, ...]
    base_addresses: ByteArray
    block_lengths: ByteArray
    block_strides: ByteArray
    object_size: int
    reachability: UnitaryReachability


def bind_ordinary_bulk_projection(
    topology: KVPoolTopology,
    max_model_len: int,
    base_addresses: Mapping[int, Sequence[int]],
    block_lengths: Mapping[int, Sequence[int]],
    block_strides: Mapping[int, Sequence[int]],
    layer_entry_offsets: Mapping[int, Sequence[int]],
) -> OrdinaryBulkProjection:
    group = topology.transfer_groups[0]
    try:
        bases, lengths, strides = bind_contiguous_layout(
            group,
            base_addresses[group.group_id],
            block_lengths[group.group_id],
            block_strides[group.group_id],
            layer_entry_offsets[group.group_id],
        )
    except KeyError as error:
        raise RuntimeError(f"KV cache group {group.group_id} has not registered local memory") from error
    return OrdinaryBulkProjection(
        group_id=group.group_id,
        block_size=group.block_size,
        hash_block_size=topology.hash_block_size,
        physical_layer_ids=tuple(layer.physical_layer_id for layer in group.layers),
        local_key_prefixes=(local_prefix(group),),
        lookup_key_prefixes=lookup_prefixes(topology, group),
        base_addresses=bases,
        block_lengths=lengths,
        block_strides=strides,
        object_size=int(lengths.sum()),
        reachability=UnitaryReachability(group.group_id, max_model_len, topology.cache_transfer_granularity),
    )


def ordinary_load_keys(projection: OrdinaryBulkProjection, hashes) -> KeyAxes:
    return key_prefixes(projection.local_key_prefixes, hashes)


def ordinary_store_keys(projection: OrdinaryBulkProjection, hashes) -> KeyAxes:
    return key_prefixes(projection.local_key_prefixes, hashes)


def ordinary_lookup_keys(projection: OrdinaryBulkProjection, hashes) -> KeyAxes:
    return key_prefixes(projection.lookup_key_prefixes, hashes)


def ordinary_bulk_ranges(
    projection: OrdinaryBulkProjection,
    block_ids,
    token_counts,
    selected_objects=None,
) -> BulkRangeBatch:
    return contiguous_ranges(
        block_ids,
        token_counts,
        selected_objects=selected_objects,
        bases=projection.base_addresses,
        lengths=projection.block_lengths,
        strides=projection.block_strides,
        block_size=projection.block_size,
        entry_slices=((0, len(projection.base_addresses)),),
    )
