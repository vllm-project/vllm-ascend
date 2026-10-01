"""Bound transfer plans and request-local rows used by Backend execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from .representation import KVChunk, PhysicalCoordinate, RemoteKVObject

ByteValues: TypeAlias = NDArray[np.uint64]
MAX_BYTE_VALUE = int(np.iinfo(np.uint64).max)


@dataclass(frozen=True, slots=True, eq=False)
class ContiguousLayoutPlan:
    """Static affine geometry for one remote representation."""

    physical_layer_ids: tuple[int, ...]
    coordinate_index: int
    object_size: int
    order_index: int
    bases: ByteValues
    block_strides: ByteValues
    bytes_per_token: ByteValues
    offsets: ByteValues
    token_capacity: int
    maximum_block_id: int = field(init=False)

    def __post_init__(self) -> None:
        _validate_layout_identity(self.physical_layer_ids, self.coordinate_index, self.object_size, self.order_index)
        _validate_arrays(self.bases, self.block_strides, self.bytes_per_token, self.offsets)
        if self.token_capacity <= 0 or np.any(self.bytes_per_token == 0):
            raise ValueError("Contiguous layout requires positive token capacity and byte sizes")
        if np.any(self.offsets >= self.object_size):
            raise ValueError("Contiguous layout offsets must fall inside the remote object")
        remaining = np.uint64(self.object_size) - self.offsets
        if not np.all(np.uint64(self.token_capacity) <= remaining // self.bytes_per_token):
            raise ValueError("Contiguous layout does not fit inside the remote object")
        object.__setattr__(self, "maximum_block_id", _maximum_block_id(self.bases, self.block_strides))


@dataclass(frozen=True, slots=True, eq=False)
class StridedLayoutPlan:
    """Static token-sliced geometry for one effective-rank representation."""

    physical_layer_ids: tuple[int, ...]
    coordinate_index: int
    object_size: int
    order_index: int
    bases: ByteValues
    block_strides: ByteValues
    sizes: ByteValues
    offsets: ByteValues
    token_indices: ByteValues
    token_capacity: int
    maximum_block_id: int = field(init=False)

    def __post_init__(self) -> None:
        _validate_layout_identity(self.physical_layer_ids, self.coordinate_index, self.object_size, self.order_index)
        _validate_arrays(self.bases, self.block_strides, self.sizes, self.offsets, self.token_indices)
        if self.token_capacity <= 0 or np.any(self.sizes == 0) or np.any(self.token_indices >= self.token_capacity):
            raise ValueError("Strided layout requires positive slices inside its token capacity")
        if np.any(self.offsets >= self.object_size) or not np.all(
            self.sizes <= np.uint64(self.object_size) - self.offsets
        ):
            raise ValueError("Strided layout does not fit inside the remote object")
        object.__setattr__(self, "maximum_block_id", _maximum_block_id(self.bases, self.block_strides))


TransferLayoutPlan: TypeAlias = ContiguousLayoutPlan | StridedLayoutPlan


@dataclass(frozen=True, slots=True)
class SubmissionPlan:
    """The static layouts submitted together at one execution fence."""

    physical_layer_id: int | None
    layout_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.layout_indices:
            raise ValueError("A submission plan must contain at least one layout")
        if len(set(self.layout_indices)) != len(self.layout_indices):
            raise ValueError("A submission plan cannot contain duplicate layouts")


@dataclass(frozen=True, slots=True, eq=False)
class BoundGroupPlan:
    """A fully validated execution plan for one original KV cache group."""

    group_id: int
    coordinates: tuple[PhysicalCoordinate, ...]
    key_prefixes: tuple[str, ...]
    layouts: tuple[TransferLayoutPlan, ...]
    submissions: tuple[SubmissionPlan, ...]
    block_capacity: int
    maximum_block_id: int = field(init=False)
    maximum_token_count: int = field(init=False)
    object_sizes: tuple[int, ...] = field(init=False)
    layer_ids: tuple[int, ...] = field(init=False)
    _submission_by_layer: MappingProxyType = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.coordinates or len(self.coordinates) != len(self.key_prefixes):
            raise ValueError("A bound group plan must align every coordinate and key prefix")
        if not self.layouts or not self.submissions or self.block_capacity <= 0:
            raise ValueError("A bound group plan requires layouts and submissions")
        if tuple(sorted(layout.order_index for layout in self.layouts)) != tuple(range(len(self.layouts))):
            raise ValueError("Layout order indices must form one dense canonical enumeration")
        if any(layout.coordinate_index >= len(self.coordinates) for layout in self.layouts):
            raise ValueError("A layout references an unknown physical coordinate")

        covered = tuple(index for submission in self.submissions for index in submission.layout_indices)
        if tuple(sorted(covered)) != tuple(range(len(self.layouts))):
            raise ValueError("Submission plans must cover every layout exactly once")
        for submission in self.submissions:
            if any(index < 0 or index >= len(self.layouts) for index in submission.layout_indices):
                raise ValueError("A submission references an unknown layout")

        object_sizes: list[int | None] = [None] * len(self.coordinates)
        for layout in self.layouts:
            current = object_sizes[layout.coordinate_index]
            if current is not None and current != layout.object_size:
                raise ValueError("One remote coordinate cannot have inconsistent object sizes")
            object_sizes[layout.coordinate_index] = layout.object_size
        if any(size is None for size in object_sizes):
            raise ValueError("Every remote coordinate must be used by at least one layout")

        submissions_by_layer = {
            submission.physical_layer_id: submission
            for submission in self.submissions
            if submission.physical_layer_id is not None
        }
        if len(submissions_by_layer) != sum(item.physical_layer_id is not None for item in self.submissions):
            raise ValueError("A bound group plan cannot contain duplicate layer submissions")

        object.__setattr__(self, "maximum_block_id", min(layout.maximum_block_id for layout in self.layouts))
        object.__setattr__(self, "maximum_token_count", min(layout.token_capacity for layout in self.layouts))
        object.__setattr__(self, "object_sizes", tuple(int(size) for size in object_sizes if size is not None))
        object.__setattr__(self, "layer_ids", tuple(submissions_by_layer))
        object.__setattr__(self, "_submission_by_layer", MappingProxyType(submissions_by_layer))

    def submission_for_layer(self, physical_layer_id: int) -> SubmissionPlan | None:
        return self._submission_by_layer.get(physical_layer_id)

    @property
    def bulk_submission(self) -> SubmissionPlan:
        if len(self.submissions) != 1 or self.submissions[0].physical_layer_id is not None:
            raise RuntimeError("This group plan is not a bulk submission plan")
        return self.submissions[0]


@dataclass(frozen=True, slots=True, eq=False)
class TransferRows:
    """One request-local Chunk axis bound once to one immutable group plan."""

    plan: BoundGroupPlan
    chunks: tuple[KVChunk, ...]
    block_ids: ByteValues
    memory_token_counts: ByteValues
    keys_by_coordinate: tuple[tuple[str, ...], ...]
    remote_objects_by_coordinate: tuple[tuple[RemoteKVObject, ...], ...]

    def __post_init__(self) -> None:
        row_count = len(self.chunks)
        for values in (self.block_ids, self.memory_token_counts):
            if values.ndim != 1 or values.dtype != np.uint64 or len(values) != row_count:
                raise ValueError("Transfer rows must align Chunk metadata as uint64 arrays")
            values.flags.writeable = False
        if any(chunk.group_id != self.plan.group_id for chunk in self.chunks):
            raise ValueError(f"Transfer rows contain chunks outside cache group {self.plan.group_id}")
        if np.any(self.memory_token_counts == 0):
            raise ValueError("Transfer rows require positive local token counts")
        if int(self.block_ids.max(initial=0)) > self.plan.maximum_block_id:
            raise ValueError("KV block address exceeds the registered transfer plan")
        if int(self.block_ids.max(initial=0)) >= self.plan.block_capacity:
            raise ValueError("KV block ID exceeds the registered cache capacity")
        if int(self.memory_token_counts.max(initial=0)) > self.plan.maximum_token_count:
            raise ValueError("KV transfer exceeds the registered token capacity")
        if len(self.keys_by_coordinate) != len(self.plan.coordinates):
            raise ValueError("Transfer rows must provide one Key axis per physical coordinate")
        if any(len(keys) != row_count for keys in self.keys_by_coordinate):
            raise ValueError("Every transfer Key axis must align the Chunk axis")
        if len(self.remote_objects_by_coordinate) != len(self.plan.coordinates):
            raise ValueError("Transfer rows must provide one remote-object axis per physical coordinate")
        for coordinate_index, objects in enumerate(self.remote_objects_by_coordinate):
            if len(objects) != row_count:
                raise ValueError("Every remote-object axis must align the Chunk axis")
            coordinate = self.plan.coordinates[coordinate_index]
            keys = self.keys_by_coordinate[coordinate_index]
            if any(
                item.chunk is not chunk or item.coordinate != coordinate or item.key != key
                for item, chunk, key in zip(objects, self.chunks, keys, strict=True)
            ):
                raise ValueError("Remote objects must preserve the bound Chunk and Key axes")

    @property
    def row_count(self) -> int:
        return len(self.chunks)

    @property
    def remote_objects(self) -> tuple[RemoteKVObject, ...]:
        return tuple(item for coordinate_objects in self.remote_objects_by_coordinate for item in coordinate_objects)


@dataclass(frozen=True, slots=True, eq=False)
class TransferSource:
    """Compact provenance for one Backend result row."""

    rows: TransferRows
    row_index: int
    physical_layer_ids: tuple[int, ...]
    key: str

    @property
    def group_id(self) -> int:
        return self.rows.plan.group_id

    @property
    def chunk(self) -> KVChunk:
        return self.rows.chunks[self.row_index]

    @property
    def block_id(self) -> int:
        return int(self.rows.block_ids[self.row_index])


def _validate_layout_identity(
    physical_layer_ids: tuple[int, ...], coordinate_index: int, object_size: int, order_index: int
) -> None:
    if not physical_layer_ids or tuple(sorted(set(physical_layer_ids))) != physical_layer_ids:
        raise ValueError("Transfer layout physical layers must be nonempty, unique and ordered")
    if coordinate_index < 0 or order_index < 0 or not 0 < object_size <= MAX_BYTE_VALUE:
        raise ValueError("Transfer layout identity contains an invalid coordinate, order or object size")


def _validate_arrays(*arrays: ByteValues) -> None:
    length = len(arrays[0])
    if not length:
        raise ValueError("Registered transfer geometry requires memory segments")
    for values in arrays:
        if values.ndim != 1 or values.dtype != np.uint64 or len(values) != length:
            raise ValueError("Registered transfer geometry must align uint64 arrays")
        values.flags.writeable = False


def _maximum_block_id(bases: ByteValues, strides: ByteValues) -> int:
    nonzero = strides != 0
    limits = (np.uint64(MAX_BYTE_VALUE) - bases[nonzero]) // strides[nonzero]
    return int(limits.min(initial=np.uint64(MAX_BYTE_VALUE)))
