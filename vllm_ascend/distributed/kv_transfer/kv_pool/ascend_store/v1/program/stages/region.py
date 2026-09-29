"""Project local Block assignments into physical transfer regions."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from ..spec.topology import KVPoolGroupTopology, KVPoolTopology
from ..values.representation import (
    KVBlockAssignmentBatch,
    KVMemoryGeometry,
    KVMemorySegment,
    KVMemoryView,
    KVRegion,
    KVTransferLayout,
    LocalKVRegion,
    PhysicalCoordinate,
    RemoteObjectLayout,
    TransferLayoutBatch,
)


class TransferRegionProjection(Protocol):
    """Project local Block assignments and their local-to-remote transfer relation."""

    def bind_memory(self, memory_geometry: KVMemoryGeometry) -> None: ...

    def project(self, assignments: KVBlockAssignmentBatch) -> TransferLayoutBatch: ...


class ContiguousRegionProjection:
    """Project each assigned KV chunk onto contiguous memory segments."""

    def __init__(self, topology: KVPoolTopology) -> None:
        self._groups = _transfer_groups(topology)
        self._segments_by_group: dict[int, tuple[KVMemorySegment, ...]] | None = None

    def bind_memory(self, memory_geometry: KVMemoryGeometry) -> None:
        if self._segments_by_group is not None:
            raise RuntimeError("Contiguous region projection is already bound")
        self._segments_by_group = _select_group_memory(self._groups, memory_geometry)

    def project(self, assignments: KVBlockAssignmentBatch) -> TransferLayoutBatch:
        if self._segments_by_group is None:
            raise RuntimeError("KV transfer region projection is unavailable before cache registration")
        group = self._groups[assignments.group_id]
        segments = self._segments_by_group[assignments.group_id]
        remote_object_size = sum(segment.block_length for segment in segments)
        remote_offsets = _contiguous_offsets(tuple(segment.block_length for segment in segments))
        layouts = []
        for assignment in assignments.assignments:
            chunk = assignment.chunk
            addresses = tuple(segment.base_address + assignment.block_id * segment.block_stride for segment in segments)
            sizes = tuple(segment.bytes_per_token * assignment.memory_token_count for segment in segments)
            layouts.append(
                KVTransferLayout(
                    LocalKVRegion(
                        KVRegion(chunk, tuple(layer.physical_layer_id for layer in group.layers)),
                        KVMemoryView(assignment.block_id, addresses, sizes),
                    ),
                    RemoteObjectLayout(PhysicalCoordinate(), remote_object_size, remote_offsets),
                )
            )
        return TransferLayoutBatch(assignments.group_id, tuple(layouts))


@dataclass(frozen=True, slots=True)
class _LayerRegionTemplate:
    physical_layer_id: int
    segments: tuple[KVMemorySegment, ...]
    remote_object_size: int
    remote_offsets: tuple[int, ...]


class LayerwiseRegionProjection:
    """Project every assigned chunk into physical-layer transfer regions."""

    def __init__(self, topology: KVPoolTopology) -> None:
        self._groups = _transfer_groups(topology)
        self._regions_by_group: dict[int, tuple[_LayerRegionTemplate, ...]] | None = None

    def bind_memory(self, memory_geometry: KVMemoryGeometry) -> None:
        if self._regions_by_group is not None:
            raise RuntimeError("Layerwise region projection is already bound")
        segments_by_group = _select_group_memory(self._groups, memory_geometry)
        self._regions_by_group = {
            group_id: _compile_layer_regions(group, segments_by_group[group_id])
            for group_id, group in self._groups.items()
        }

    def project(self, assignments: KVBlockAssignmentBatch) -> TransferLayoutBatch:
        if self._regions_by_group is None:
            raise RuntimeError("Layerwise transfer region projection is unavailable before cache registration")
        layouts = []
        for assignment in assignments.assignments:
            chunk = assignment.chunk
            for template in self._regions_by_group[assignments.group_id]:
                addresses = tuple(
                    segment.base_address + assignment.block_id * segment.block_stride for segment in template.segments
                )
                sizes = tuple(segment.bytes_per_token * assignment.memory_token_count for segment in template.segments)
                layouts.append(
                    KVTransferLayout(
                        LocalKVRegion(
                            KVRegion(chunk, (template.physical_layer_id,)),
                            KVMemoryView(assignment.block_id, addresses, sizes),
                        ),
                        RemoteObjectLayout(PhysicalCoordinate(), template.remote_object_size, template.remote_offsets),
                    )
                )
        return TransferLayoutBatch(assignments.group_id, tuple(layouts))


@dataclass(frozen=True, slots=True)
class _StridedRepresentation:
    coordinate: PhysicalCoordinate
    slice_index: int


@dataclass(frozen=True, slots=True)
class _StridedMemorySegment:
    base_address: int
    block_stride: int
    bytes_per_token: int
    slice_size: int
    head_offset: int


@dataclass(frozen=True, slots=True)
class _BoundStridedRepresentation:
    coordinate: PhysicalCoordinate
    segments: tuple[_StridedMemorySegment, ...]


class StridedRegionProjection:
    """Project each assigned KV chunk into fixed effective-rank transfer regions."""

    def __init__(self, topology: KVPoolTopology) -> None:
        if len(topology.transfer_group_ids) != 1:
            raise ValueError("AscendStore v1 TP mismatch requires one transferable KV cache group")
        self._groups = _transfer_groups(topology)
        self._group_id = topology.transfer_group_ids[0]
        self._representations = tuple(
            _StridedRepresentation(
                PhysicalCoordinate(
                    effective_tp_rank=topology.tp_rank * topology.tp_partition.key_slices_per_rank + slice_index
                ),
                slice_index,
            )
            for slice_index in range(topology.tp_partition.key_slices_per_rank)
        )
        self._bound_representations: tuple[_BoundStridedRepresentation, ...] | None = None

    def bind_memory(self, memory_geometry: KVMemoryGeometry) -> None:
        if self._bound_representations is not None:
            raise RuntimeError("Strided region projection is already bound")
        segments = _select_group_memory(self._groups, memory_geometry)[self._group_id]
        if not segments:
            raise RuntimeError(f"KV cache group {self._group_id} registered no local memory segments")
        self._bound_representations = tuple(
            _BoundStridedRepresentation(
                representation.coordinate,
                tuple(
                    _compile_strided_segment(segment, len(self._representations), representation.slice_index)
                    for segment in segments
                ),
            )
            for representation in self._representations
        )

    def project(self, assignments: KVBlockAssignmentBatch) -> TransferLayoutBatch:
        if self._bound_representations is None:
            raise RuntimeError("KV transfer region projection is unavailable before cache registration")
        if assignments.group_id != self._group_id:
            raise ValueError(
                f"Strided region projection was bound for group {self._group_id}, received {assignments.group_id}"
            )
        group = self._groups[assignments.group_id]
        layouts = []
        for assignment in assignments.assignments:
            chunk = assignment.chunk
            region = KVRegion(chunk, tuple(layer.physical_layer_id for layer in group.layers))
            for representation in self._bound_representations:
                addresses = []
                sizes = []
                remote_offsets = []
                remote_object_size = sum(group.block_size * segment.slice_size for segment in representation.segments)
                segment_offset = 0
                for segment in representation.segments:
                    block_address = segment.base_address + assignment.block_id * segment.block_stride
                    for token_index in range(assignment.memory_token_count):
                        addresses.append(block_address + token_index * segment.bytes_per_token + segment.head_offset)
                        sizes.append(segment.slice_size)
                        remote_offsets.append(segment_offset + token_index * segment.slice_size)
                    segment_offset += group.block_size * segment.slice_size
                layouts.append(
                    KVTransferLayout(
                        LocalKVRegion(
                            region,
                            KVMemoryView(assignment.block_id, tuple(addresses), tuple(sizes)),
                        ),
                        RemoteObjectLayout(representation.coordinate, remote_object_size, tuple(remote_offsets)),
                    )
                )
        return TransferLayoutBatch(assignments.group_id, tuple(layouts))


def _transfer_groups(topology: KVPoolTopology) -> dict[int, KVPoolGroupTopology]:
    groups_by_id = {group.group_id: group for group in topology.groups}
    try:
        return {group_id: groups_by_id[group_id] for group_id in topology.transfer_group_ids}
    except KeyError as error:
        raise ValueError(f"Unknown transferable KV cache group {error.args[0]}") from error


def _select_group_memory(
    groups: dict[int, KVPoolGroupTopology],
    memory_geometry: KVMemoryGeometry,
) -> dict[int, tuple[KVMemorySegment, ...]]:
    segments_by_group = {}
    for group_id, group in groups.items():
        try:
            segments = memory_geometry[group_id]
        except KeyError as error:
            raise RuntimeError(f"KV cache group {group_id} has not registered local memory") from error
        physical_layer_ids = tuple(layer_id for layer_id, _ in _group_segments_by_layer(segments))
        expected_layer_ids = tuple(layer.physical_layer_id for layer in group.layers)
        if physical_layer_ids != expected_layer_ids:
            raise ValueError(
                f"KV cache group {group_id} registered physical layers {physical_layer_ids}, "
                f"expected {expected_layer_ids}"
            )
        segments_by_group[group_id] = segments
    return segments_by_group


def _group_segments_by_layer(
    segments: tuple[KVMemorySegment, ...],
) -> tuple[tuple[int, tuple[KVMemorySegment, ...]], ...]:
    grouped: list[tuple[int, list[KVMemorySegment]]] = []
    for segment in segments:
        if grouped and grouped[-1][0] == segment.physical_layer_id:
            grouped[-1][1].append(segment)
        else:
            grouped.append((segment.physical_layer_id, [segment]))
    return tuple((layer_id, tuple(layer_segments)) for layer_id, layer_segments in grouped)


def _compile_layer_regions(
    group: KVPoolGroupTopology,
    segments: tuple[KVMemorySegment, ...],
) -> tuple[_LayerRegionTemplate, ...]:
    remote_offset = 0
    offsets_by_segment = []
    for segment in segments:
        offsets_by_segment.append(remote_offset)
        remote_offset += segment.block_length
    regions = []
    segment_start = 0
    for physical_layer_id, layer_segments in _group_segments_by_layer(segments):
        segment_end = segment_start + len(layer_segments)
        regions.append(
            _LayerRegionTemplate(
                physical_layer_id,
                layer_segments,
                remote_offset,
                tuple(offsets_by_segment[segment_start:segment_end]),
            )
        )
        segment_start = segment_end
    physical_layer_ids = tuple(region.physical_layer_id for region in regions)
    expected_layer_ids = tuple(layer.physical_layer_id for layer in group.layers)
    if physical_layer_ids != expected_layer_ids:
        raise ValueError(
            f"KV cache group {group.group_id} compiled physical layers {physical_layer_ids}, "
            f"expected {expected_layer_ids}"
        )
    return tuple(regions)


def _contiguous_offsets(sizes: Sequence[int]) -> tuple[int, ...]:
    offsets = []
    current_offset = 0
    for size in sizes:
        offsets.append(current_offset)
        current_offset += size
    return tuple(offsets)


def _compile_strided_segment(segment: KVMemorySegment, slice_count: int, slice_index: int) -> _StridedMemorySegment:
    slice_size, remainder = divmod(segment.bytes_per_token, slice_count)
    if remainder:
        raise ValueError(
            f"KV memory segment has {segment.bytes_per_token} bytes per token, "
            f"which cannot be divided into {slice_count} Strided slices"
        )
    return _StridedMemorySegment(
        segment.base_address,
        segment.block_stride,
        segment.bytes_per_token,
        slice_size,
        slice_index * slice_size,
    )
