"""Compile consecutive segment boundaries without choosing a Backend object format."""

from collections.abc import Sequence

import numpy as np

from ..local_layout.types import ObjectSourceSegments
from ..types import UInt64Array


def build_segment_offsets(sources: Sequence[ObjectSourceSegments]) -> tuple[UInt64Array, ...]:
    """Return each source's read-only segment boundaries, including its payload end.

    Equal head slices share the same lengths array and therefore the same compiled
    offsets. No memory is copied or rearranged; these are positions, not KV bytes.
    """
    offsets_by_layout: dict[tuple[int, int], UInt64Array] = {}
    result = []
    for source in sources:
        segments = source.memory_segments
        segment_count = len(segments.block_zero_addresses)
        layout_key = (id(segments.segment_lengths_bytes), segment_count)
        offsets = offsets_by_layout.get(layout_key)
        if offsets is None:
            offsets = np.empty(segment_count + 1, dtype=np.uint64)
            offsets[0] = 0
            np.cumsum(np.broadcast_to(segments.segment_lengths_bytes, (segment_count,)), out=offsets[1:])
            offsets.flags.writeable = False
            offsets_by_layout[layout_key] = offsets
        if int(offsets[-1]) != source.payload_bytes:
            raise ValueError("Local segments must cover the complete payload size")
        result.append(offsets)
    return tuple(result)
