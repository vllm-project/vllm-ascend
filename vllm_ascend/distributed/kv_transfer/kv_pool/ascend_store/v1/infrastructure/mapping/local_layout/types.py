"""Local byte geometry and numeric object coordinates, independent of Key encoding."""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from ..types import UInt64Array


@dataclass(frozen=True, slots=True)
class ObjectPartitionCoordinate:
    """An object's fixed PP, DCP and head/TP identity, not an axis selector."""

    pp_rank: int
    dcp_rank: int
    head_rank: int


@dataclass(frozen=True, slots=True)
class LocalMemorySegments:
    """Read-only local payload intervals in serialization order, measured in bytes.

    Block b's segment j begins at block_zero_segment_addresses[j] + b * group_block_strides_bytes[j].
    Strides and lengths may be singletons shared by all columns. This describes
    locations only: it neither owns memory nor proves readiness.
    """

    block_zero_segment_addresses: UInt64Array
    group_block_strides_bytes: UInt64Array
    segment_lengths_bytes: UInt64Array

    def addresses(self, local_block_ids: Sequence[int] | UInt64Array) -> UInt64Array:
        """Evaluate [input block, bound segment] addresses without selecting or reordering.

        Allocation bounds remain caller-owned. Arithmetic retains uint64 wrapping,
        matching the registered local-address formula used by both transfer routes.
        """
        ids = np.asarray(local_block_ids, dtype=np.uint64)
        if ids.ndim != 1:
            raise ValueError("Local block IDs must be one-dimensional")
        return ids[:, None] * self.group_block_strides_bytes + self.block_zero_segment_addresses


@dataclass(frozen=True, slots=True)
class LocalPartitionLayout:
    """Local sources assigned to one bound partition of the remote object family.

    LocalLayout contains one such entry per PP/head partition. Layer boundaries
    index its segment columns; concrete hashes and block IDs arrive at runtime.
    """

    coordinate: ObjectPartitionCoordinate
    physical_layer_ids: tuple[int, ...]
    local_segments: LocalMemorySegments
    layer_segment_index_boundaries: tuple[int, ...]
    payload_bytes: int


@dataclass(frozen=True, slots=True)
class LocalLayout:
    """One original cache group's geometry on this Worker, ordered by PP then local head."""

    group_id: int
    group_block_size: int
    hash_block_size: int
    physical_layer_ids: tuple[int, ...]
    local_group_block_bytes: int
    partitions: tuple[LocalPartitionLayout, ...]
    physical_layer_ids_by_partition: tuple[tuple[int, ...], ...]
