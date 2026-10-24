"""Bind Entries once, then select the local bytes belonging to each PP/head object.

Registration order and wire traversal are preserved. Head and PP slicing share
read-only arrays; neither operation copies KV bytes or chooses Store writers.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

import numpy as np

from . import contiguous, head_slicing, layer_partitioning
from .types import LocalLayout, LocalMemorySegments, ObjectCoordinate, ObjectSourceSegments

if TYPE_CHECKING:
    import torch

    from ...configuration.topology import KVPoolTopology


def bind_local_layout(
    topology: KVPoolTopology,
    entry_addresses_by_group: Mapping[int, Sequence[int]],
    entry_block_payload_bytes_by_group: Mapping[int, Sequence[int]],
    entry_block_strides_bytes_by_group: Mapping[int, Sequence[int]],
    layer_entry_boundaries_by_group: Mapping[int, Sequence[int]],
    *,
    group_id: int | None = None,
    kv_cache_layout: str = "NHD",
    kv_caches: Mapping[str, torch.Tensor | Sequence[torch.Tensor]] | None = None,
) -> LocalLayout:
    groups = topology.transfer_groups
    if group_id is None:
        if len(groups) != 1:
            raise ValueError("Select an original group_id when binding multiple transferable cache groups")
        group = groups[0]
    else:
        selected = next((group for group in groups if group.group_id == group_id), None)
        if selected is None:
            raise KeyError(f"Unknown transferable KV cache group {group_id}")
        group = selected
    entry_addresses = entry_addresses_by_group[group.group_id]
    entry_block_payload_bytes = entry_block_payload_bytes_by_group[group.group_id]
    entry_block_strides_bytes = entry_block_strides_bytes_by_group[group.group_id]
    layer_entry_boundaries = layer_entry_boundaries_by_group[group.group_id]

    # ========================================
    # Bind registration and target PP spans
    # ========================================

    physical_layer_ids = tuple(layer.physical_layer_id for layer in group.layers)
    if group.block_size <= 0:
        raise ValueError("Group block size must be positive")
    addresses = np.array(entry_addresses, dtype=np.uint64)
    payload_bytes = np.array(entry_block_payload_bytes, dtype=np.uint64)
    strides_bytes = np.array(entry_block_strides_bytes, dtype=np.uint64)
    if not len(addresses):
        raise ValueError("KV cache group has no registered memory entries")
    for array in (addresses, payload_bytes, strides_bytes):
        array.flags.writeable = False
    local_group_block_bytes = int(payload_bytes.sum())
    target_pp_layer_counts = topology.consumer_pipeline_partitions or ()
    pp_partitions = (
        layer_partitioning.partition_layers(physical_layer_ids, layer_entry_boundaries, target_pp_layer_counts)
        if len(target_pp_layer_counts) > 1
        else ()
    )
    metadata = group.key_metadata
    if len(pp_partitions) == 1 and pp_partitions[0].pp_rank == metadata.pp_rank:
        pp_partitions = ()

    # ========================================
    # Bind whole Entries or head slices once
    # ========================================

    head_slices_per_rank = 1
    entry_segment_boundaries = None
    segments_by_head_slice: tuple[LocalMemorySegments, ...]
    if topology.tp_partition.tp_mismatch:
        head_slices_per_rank = topology.tp_partition.key_slices_per_rank
        head_major = kv_cache_layout in ("HND", "LBHNC")
        entry_layouts, whole_entries_contiguous = head_slicing.inspect_entries(
            group, payload_bytes, strides_bytes, head_slices_per_rank, head_major, kv_caches
        )
        if whole_entries_contiguous:
            segments_by_head_slice = (contiguous.bind_segments(addresses, payload_bytes, strides_bytes),)
        else:
            entry_segment_boundaries = [0]
            geometry = head_slicing.compile_geometries(
                entry_layouts, head_slices_per_rank, head_major, entry_segment_boundaries
            )
            segments_by_head_slice = head_slicing.materialize_segments(
                addresses, strides_bytes, entry_layouts, head_slices_per_rank, geometry
            )
    else:
        segments_by_head_slice = (contiguous.bind_segments(addresses, payload_bytes, strides_bytes),)
    layer_segment_boundaries = (
        tuple(layer_entry_boundaries)
        if entry_segment_boundaries is None
        else tuple(entry_segment_boundaries[index] for index in layer_entry_boundaries)
    )

    # ========================================
    # Assign local segments to numeric objects
    # ========================================

    first_head_rank = metadata.head_or_tp_rank * head_slices_per_rank
    objects = []
    if not pp_partitions:
        for head_index, segments in enumerate(segments_by_head_slice):
            objects.append(
                ObjectSourceSegments(
                    ObjectCoordinate(metadata.pp_rank, metadata.dcp_rank, first_head_rank + head_index),
                    physical_layer_ids,
                    segments,
                    layer_segment_boundaries,
                    local_group_block_bytes // head_slices_per_rank,
                )
            )
    else:
        for partition in pp_partitions:
            pp_segments = (
                segments_by_head_slice
                if len(pp_partitions) == 1
                else layer_partitioning.slice_layer_segments(
                    segments_by_head_slice, partition, entry_segment_boundaries
                )
            )
            object_bytes = (
                int(payload_bytes[partition.entry_start_index : partition.entry_end_index].sum())
                // head_slices_per_rank
            )
            if len(pp_partitions) == 1:
                boundaries = layer_segment_boundaries
            else:
                segment_start = layer_segment_boundaries[partition.local_layer_start_index]
                boundaries = tuple(
                    boundary - segment_start
                    for boundary in layer_segment_boundaries[
                        partition.local_layer_start_index : partition.local_layer_end_index + 1
                    ]
                )
            for head_index, segments in enumerate(pp_segments):
                objects.append(
                    ObjectSourceSegments(
                        ObjectCoordinate(partition.pp_rank, metadata.dcp_rank, first_head_rank + head_index),
                        partition.physical_layer_ids,
                        segments,
                        boundaries,
                        object_bytes,
                    )
                )
    return LocalLayout(
        group.group_id,
        group.block_size,
        topology.hash_block_size,
        physical_layer_ids,
        local_group_block_bytes,
        tuple(objects),
        tuple(obj.physical_layer_ids for obj in objects),
    )
