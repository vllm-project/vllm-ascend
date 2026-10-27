"""Memcache wire placement: global layer reservation followed by shared context slots."""

from functools import partial

import numpy as np

from ...infrastructure.configuration.topology import CPPartitionSpec, KVPoolTopology, PPPartitionSpec
from ...infrastructure.mapping.local_layout import LocalLayout
from ...infrastructure.mapping.object_layout import RemoteObjectLayout, RemoteObjectLayoutFactory
from ...infrastructure.mapping.object_layout.context_partitioning import bind_context_object_span
from ...infrastructure.mapping.object_layout.layer_partitioning import bind_layer_object_span
from ...infrastructure.mapping.object_layout.offsets import build_segment_offsets


def create_object_layout(topology: KVPoolTopology) -> RemoteObjectLayoutFactory:
    """Resolve fixed wire parameters before acquiring Backend resources."""
    return partial(
        build_memcache_object_layouts,
        pp_partition=topology.pp_partition,
        cp_partition=topology.cp_partition,
        stage_layer_start=topology.stage_layer_start,
        stage_layer_count=topology.stage_layer_count,
        object_shard_index=topology.object_shard_index,
    )


def build_memcache_object_layouts(
    local: LocalLayout,
    *,
    pp_partition: PPPartitionSpec,
    cp_partition: CPPartitionSpec,
    stage_layer_start: int = 0,
    stage_layer_count: int = 0,
    object_shard_index: int = 0,
) -> tuple[RemoteObjectLayout, ...]:
    """Place local segments without changing their order or allocating remote storage."""
    result = []
    for source, offsets in zip(local.partitions, build_segment_offsets(local.partitions), strict=True):
        if pp_partition.global_layer_count is not None:
            boundaries = source.layer_segment_index_boundaries
            layer_lengths = [int(offsets[end] - offsets[start]) for start, end in zip(boundaries, boundaries[1:])]
            if len(set(layer_lengths)) != 1:
                raise ValueError("Global layer spans require uniform per-layer payloads")
        shard_size, layer_offset = bind_layer_object_span(
            pp_partition,
            source.payload_bytes,
            len(source.physical_layer_ids),
            stage_layer_start=stage_layer_start,
            stage_layer_count=stage_layer_count,
        )
        object_size, base_offset = bind_context_object_span(
            cp_partition, shard_size, layer_offset, shard_index=object_shard_index
        )
        if base_offset:
            offsets = offsets + np.uint64(base_offset)
            offsets.flags.writeable = False
        result.append(RemoteObjectLayout(object_size, offsets))
    return tuple(result)
