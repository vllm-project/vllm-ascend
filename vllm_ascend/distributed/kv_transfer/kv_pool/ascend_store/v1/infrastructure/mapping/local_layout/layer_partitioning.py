"""Select target PP layer spans from already-bound local memory segments.

Example: local physical layers (4, 5), target PP layer counts (5, 1).

    local layer index           0                   1
    physical layer          L4 (PP0)            L5 (PP1)
    Entries                  K4, V4              K5, V5
    Entry indices            [0:2]               [2:4]
    segment indices          [0:5]               [5:10]

    layer_entry_boundaries   = [0, 2, 4]
    entry_segment_boundaries = [0, 3, 5, 8, 10]

    PP1 -> local layers[1:2] -> Entries[2:4] -> segments[5:10]
                                  |                 |
                        via layer boundaries   via Entry boundaries

These are array indices, not byte offsets; Entry/segment counts may vary.
Contiguous binding uses Entry indices directly; entry_segment_boundaries is None.
Target PP describes stored objects, independently of execution PP. Segment bases
already include head-slice offsets: PP selection shares columns without rebinding.
"""

from collections.abc import Sequence
from dataclasses import dataclass

from .types import LocalMemorySegments


@dataclass(frozen=True, slots=True)
class PPLayerPartition:
    """One target PP rank's locally present layers and half-open array spans.

    physical_layer_ids are absolute IDs; local_layer_* index the local layer
    list, and entry_* index the registration Entry list. None are byte offsets.
    """

    pp_rank: int
    physical_layer_ids: tuple[int, ...]
    entry_start_index: int
    entry_end_index: int
    local_layer_start_index: int
    local_layer_end_index: int


def partition_layers(
    physical_layer_ids: tuple[int, ...], layer_entry_boundaries: Sequence[int], target_pp_layer_counts: Sequence[int]
) -> tuple[PPLayerPartition, ...]:
    """Require sorted absolute layer IDs covered by nonnegative target PP counts.

    Target ranks with no local layers are omitted.
    """
    result = []
    layer_cursor = 0
    pp_layer_end = 0
    for pp_rank, layer_count in enumerate(target_pp_layer_counts):
        if layer_count < 0:
            raise ValueError("Target PP layer counts must be nonnegative")
        pp_layer_end += layer_count
        start = layer_cursor
        while layer_cursor < len(physical_layer_ids) and physical_layer_ids[layer_cursor] < pp_layer_end:
            layer_cursor += 1
        if start == layer_cursor:
            continue
        result.append(
            PPLayerPartition(
                pp_rank,
                physical_layer_ids[start:layer_cursor],
                layer_entry_boundaries[start],
                layer_entry_boundaries[layer_cursor],
                start,
                layer_cursor,
            )
        )
    if not result or layer_cursor != len(physical_layer_ids):
        raise ValueError("Target PP layer spans do not cover the registered entries")
    return tuple(result)


def slice_layer_segments(
    segments_by_head_slice: tuple[LocalMemorySegments, ...],
    pp_partition: PPLayerPartition,
    entry_segment_boundaries: list[int] | None,
) -> tuple[LocalMemorySegments, ...]:
    """Share read-only NumPy columns for this PP span across all head slices."""
    first = segments_by_head_slice[0]
    start, end = pp_partition.entry_start_index, pp_partition.entry_end_index
    if entry_segment_boundaries is not None and len(entry_segment_boundaries) > 1:
        start, end = entry_segment_boundaries[start], entry_segment_boundaries[end]
    strides, lengths = first.group_block_strides_bytes, first.segment_lengths_bytes
    strides = strides if len(strides) == 1 else strides[start:end]
    lengths = lengths if len(lengths) == 1 else lengths[start:end]
    return tuple(
        LocalMemorySegments(segments.block_zero_segment_addresses[start:end], strides, lengths)
        for segments in segments_by_head_slice
    )
