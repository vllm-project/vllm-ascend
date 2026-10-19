"""Local byte geometry and numeric object coordinates, independent of Key encoding."""

from dataclasses import dataclass

from ..types import UInt64Array


@dataclass(frozen=True, slots=True)
class ObjectCoordinate:
    """An object's PP, DCP and head/TP position, not its serialized name."""

    pp_rank: int
    dcp_rank: int
    head_rank: int


@dataclass(frozen=True, slots=True)
class LocalMemorySegments:
    """Read-only local payload intervals in serialization order, measured in bytes.

    Block b's segment j begins at block_zero_addresses[j] + b * group_block_strides_bytes[j].
    Strides and lengths may be singletons shared by all columns. This describes
    locations only: it neither owns memory nor proves readiness.
    """

    block_zero_addresses: UInt64Array
    group_block_strides_bytes: UInt64Array
    segment_lengths_bytes: UInt64Array


@dataclass(frozen=True, slots=True)
class ObjectSourceSegments:
    """Local sources assigned to one target key axis, not a local storage object.

    LocalLayout contains one such entry per PP/head partition. Layer boundaries
    index its segment columns; concrete hashes and block IDs arrive at runtime.
    """

    coordinate: ObjectCoordinate
    physical_layer_ids: tuple[int, ...]
    memory_segments: LocalMemorySegments
    layer_segment_boundaries: tuple[int, ...]
    payload_bytes: int


@dataclass(frozen=True, slots=True)
class LocalLayout:
    """One original cache group's bound geometry, ordered by target PP then local head."""

    group_id: int
    group_block_size: int
    hash_block_size: int
    physical_layer_ids: tuple[int, ...]
    local_group_block_bytes: int
    objects: tuple[ObjectSourceSegments, ...]
    physical_layer_ids_by_object: tuple[tuple[int, ...], ...]
