"""Producer-stage key and memory projection for Consumer-PP Bulk Store."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import PoolKey

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
class ConsumerPipelineAxis:
    """One producer PP key axis bound atomically to its local layer entries."""

    pp_rank: int
    key_prefix: str
    entry_slice: tuple[int, int]
    physical_layer_ids: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class ConsumerPipelineBulkProjection:
    """Registration-bound formula inputs for Consumer-PP Bulk Store."""

    group_id: int
    block_size: int
    hash_block_size: int
    physical_layer_ids: tuple[int, ...]
    local_key_prefixes: tuple[str, ...]
    lookup_key_prefixes: tuple[str, ...]
    store_axes: tuple[ConsumerPipelineAxis, ...]
    base_addresses: ByteArray
    block_lengths: ByteArray
    block_strides: ByteArray
    object_size: int
    reachability: UnitaryReachability


def bind_consumer_pipeline_bulk_projection(
    topology: KVPoolTopology,
    max_model_len: int,
    base_addresses: Mapping[int, Sequence[int]],
    block_lengths: Mapping[int, Sequence[int]],
    block_strides: Mapping[int, Sequence[int]],
    layer_entry_offsets: Mapping[int, Sequence[int]],
) -> ConsumerPipelineBulkProjection:
    group = topology.transfer_groups[0]
    partitions = topology.consumer_pipeline_partitions
    if partitions is None or len(partitions) <= 1:
        raise ValueError("Consumer pipeline Bulk projection requires multiple producer partitions")
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
    layer_bounds = {
        layer_id: (int(layer_offsets[index]), int(layer_offsets[index + 1]))
        for index, layer_id in enumerate(physical_layers)
    }
    partition_bounds: list[tuple[int, int]] = []
    first_layer = 0
    for layer_count in partitions:
        last_layer = first_layer + layer_count
        partition_bounds.append((first_layer, last_layer))
        first_layer = last_layer

    axes: list[ConsumerPipelineAxis] = []
    for pp_rank, (partition_start, partition_end) in enumerate(partition_bounds):
        selected_layers = tuple(layer_id for layer_id in physical_layers if partition_start <= layer_id < partition_end)
        if not selected_layers:
            continue
        axes.append(
            ConsumerPipelineAxis(
                pp_rank=pp_rank,
                key_prefix=PoolKey(replace(group.key_metadata, pp_rank=pp_rank), "").to_string(),
                entry_slice=(layer_bounds[selected_layers[0]][0], layer_bounds[selected_layers[-1]][1]),
                physical_layer_ids=selected_layers,
            )
        )
    if not axes or sum(axis.entry_slice[1] - axis.entry_slice[0] for axis in axes) != len(bases):
        raise ValueError("Consumer pipeline partitions do not cover the registered KV entries")

    return ConsumerPipelineBulkProjection(
        group_id=group.group_id,
        block_size=group.block_size,
        hash_block_size=topology.hash_block_size,
        physical_layer_ids=physical_layers,
        local_key_prefixes=(local_prefix(group),),
        lookup_key_prefixes=lookup_prefixes(topology, group),
        store_axes=tuple(axes),
        base_addresses=bases,
        block_lengths=lengths,
        block_strides=strides,
        object_size=int(lengths.sum()),
        reachability=UnitaryReachability(group.group_id, max_model_len, topology.cache_transfer_granularity),
    )


def consumer_pipeline_load_keys(projection: ConsumerPipelineBulkProjection, hashes) -> KeyAxes:
    return key_prefixes(projection.local_key_prefixes, hashes)


def consumer_pipeline_store_keys(projection: ConsumerPipelineBulkProjection, hashes) -> KeyAxes:
    return key_prefixes(tuple(axis.key_prefix for axis in projection.store_axes), hashes)


def consumer_pipeline_lookup_keys(projection: ConsumerPipelineBulkProjection, hashes) -> KeyAxes:
    return key_prefixes(projection.lookup_key_prefixes, hashes)


def consumer_pipeline_load_ranges(
    projection: ConsumerPipelineBulkProjection,
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


def consumer_pipeline_store_ranges(
    projection: ConsumerPipelineBulkProjection,
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
        entry_slices=tuple(axis.entry_slice for axis in projection.store_axes),
    )
