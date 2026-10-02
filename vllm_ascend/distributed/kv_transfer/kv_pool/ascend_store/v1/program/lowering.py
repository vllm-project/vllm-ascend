"""Lower registered KV geometry into immutable Backend execution plans."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import PoolKey, block_hash_to_str

from .spec.schedule import KVPoolSchedule
from .spec.topology import KVPoolGroupTopology, KVPoolTopology
from .values.representation import KVBlockAssignmentBatch, KVMemoryGeometry, KVMemorySegment, PhysicalCoordinate
from .values.selection import TransferSpan, TransferWork
from .values.transfer import (
    BoundGroupPlan,
    ContiguousLayoutPlan,
    StridedLayoutPlan,
    SubmissionPlan,
    TransferRows,
)


@dataclass(frozen=True, slots=True)
class BoundTransferPlans:
    """All direction-specific physical plans fixed by cache registration."""

    load: tuple[BoundGroupPlan, ...]
    store: tuple[BoundGroupPlan, ...]

    def __post_init__(self) -> None:
        load_ids = tuple(plan.group_id for plan in self.load)
        store_ids = tuple(plan.group_id for plan in self.store)
        if load_ids != store_ids or len(set(load_ids)) != len(load_ids):
            raise ValueError("Load and Store plans must align unique cache groups")


def lower_transfer_plans(
    topology: KVPoolTopology,
    schedule: KVPoolSchedule,
    memory_geometry: KVMemoryGeometry,
    block_capacity: int,
) -> BoundTransferPlans:
    """Validate topology and memory once, then retain only executable facts."""

    segments_by_group = _select_group_memory(topology, memory_geometry)
    if schedule.requires_layerwise_backend:
        plans = tuple(
            _lower_layerwise(group, segments_by_group[group.group_id], block_capacity)
            for group in topology.transfer_groups
        )
        return BoundTransferPlans(plans, plans)
    if topology.tp_partition.tp_mismatch:
        plans = tuple(
            _lower_strided(group, segments_by_group[group.group_id], topology, block_capacity)
            for group in topology.transfer_groups
        )
        return BoundTransferPlans(plans, plans)

    load = tuple(
        _lower_contiguous(group, segments_by_group[group.group_id], block_capacity)
        for group in topology.transfer_groups
    )
    if topology.consumer_pipeline_partitions is None or len(topology.consumer_pipeline_partitions) <= 1:
        return BoundTransferPlans(load, load)
    store = tuple(
        _lower_pipeline_partitions(
            group,
            segments_by_group[group.group_id],
            topology.consumer_pipeline_partitions,
            block_capacity,
        )
        for group in topology.transfer_groups
    )
    return BoundTransferPlans(load, store)


def bind_transfer_rows(
    plan: BoundGroupPlan,
    assignments: KVBlockAssignmentBatch,
) -> TransferRows:
    """Bind the dynamic Chunk axis once without expanding it over static layouts."""

    if assignments.group_id != plan.group_id:
        raise ValueError(f"KV block assignment group {assignments.group_id} does not match bound plan {plan.group_id}")
    chunks = tuple(item.chunk for item in assignments.assignments)
    block_ids = np.asarray([item.block_id for item in assignments.assignments], dtype=np.uint64)
    token_counts = np.asarray([item.memory_token_count for item in assignments.assignments], dtype=np.uint64)
    keys_by_coordinate = tuple(
        tuple(prefix + block_hash_to_str(chunk.content_hash) for chunk in chunks) for prefix in plan.key_prefixes
    )
    remote_objects = tuple(
        tuple(_remote_object(chunk, key, coordinate) for chunk, key in zip(chunks, keys, strict=True))
        for coordinate, keys in zip(plan.coordinates, keys_by_coordinate, strict=True)
    )
    return TransferRows(plan, chunks, block_ids, token_counts, keys_by_coordinate, remote_objects)


def enumerate_transfer_work(
    rows_by_group: tuple[TransferRows, ...],
    rank_offset: int = 0,
) -> tuple[TransferWork, ...]:
    """Enumerate Backend order once, retaining compact row spans for every fence."""

    layer_ids = tuple(
        dict.fromkeys(
            submission.physical_layer_id
            for rows in rows_by_group
            for submission in rows.plan.submissions
            if submission.physical_layer_id is not None
        )
    )
    if layer_ids:
        return _enumerate_layerwise_work(rows_by_group, layer_ids, rank_offset)

    return (TransferWork(None, _enumerate_bulk_spans(rows_by_group, rank_offset)),)


def _enumerate_layerwise_work(
    rows_by_group: tuple[TransferRows, ...],
    layer_ids: tuple[int, ...],
    rank_offset: int,
) -> tuple[TransferWork, ...]:
    """Rotate row-major traversal using at most two spans per group and layer."""

    total_count = sum(rows.row_count * len(rows.plan.layouts) for rows in rows_by_group)
    offset = 0 if total_count == 0 else rank_offset % total_count
    if offset == 0:
        return tuple(
            TransferWork(
                layer_id,
                tuple(
                    TransferSpan(rows, submission.layout_indices, range(rows.row_count))
                    for rows in rows_by_group
                    if rows.row_count and (submission := rows.plan.submission_for_layer(layer_id)) is not None
                ),
            )
            for layer_id in layer_ids
        )

    group_starts = []
    current = 0
    for rows in rows_by_group:
        group_starts.append(current)
        current += rows.row_count * len(rows.plan.layouts)

    result = []
    for layer_id in layer_ids:
        ordered_spans: list[tuple[int, TransferSpan]] = []
        for rows, group_start in zip(rows_by_group, group_starts, strict=True):
            submission = rows.plan.submission_for_layer(layer_id)
            if submission is None or rows.row_count == 0:
                continue
            layout_count = len(rows.plan.layouts)
            for layout_index in submission.layout_indices:
                first_position = group_start + layout_index
                split = min(
                    rows.row_count,
                    max(0, (offset - first_position + layout_count - 1) // layout_count),
                )
                if split < rows.row_count:
                    ordered_spans.append(
                        (
                            first_position + split * layout_count - offset,
                            TransferSpan(rows, (layout_index,), range(split, rows.row_count)),
                        )
                    )
                if split:
                    ordered_spans.append(
                        (
                            first_position + total_count - offset,
                            TransferSpan(rows, (layout_index,), range(split)),
                        )
                    )
        ordered_spans.sort(key=lambda item: item[0])
        result.append(TransferWork(layer_id, tuple(span for _, span in ordered_spans)))
    return tuple(result)


def _enumerate_bulk_spans(
    rows_by_group: tuple[TransferRows, ...],
    rank_offset: int,
) -> tuple[TransferSpan, ...]:
    total_count = sum(rows.row_count * len(rows.plan.layouts) for rows in rows_by_group)
    if total_count == 0:
        return ()
    offset = rank_offset % total_count
    group_starts = []
    current = 0
    for rows in rows_by_group:
        group_starts.append(current)
        current += rows.row_count * len(rows.plan.layouts)
    cut_group = max(index for index, start in enumerate(group_starts) if start <= offset)
    cut_rows = rows_by_group[cut_group]
    local_offset = offset - group_starts[cut_group]

    spans = list(_grid_interval(cut_rows, local_offset, cut_rows.row_count * len(cut_rows.plan.layouts)))
    for rows in rows_by_group[cut_group + 1 :]:
        spans.extend(_grid_interval(rows, 0, rows.row_count * len(rows.plan.layouts)))
    for rows in rows_by_group[:cut_group]:
        spans.extend(_grid_interval(rows, 0, rows.row_count * len(rows.plan.layouts)))
    spans.extend(_grid_interval(cut_rows, 0, local_offset))
    return tuple(spans)


def _grid_interval(rows: TransferRows, start: int, end: int) -> tuple[TransferSpan, ...]:
    """Describe a row-major layout interval without materializing its cells."""

    if start >= end:
        return ()
    layout_count = len(rows.plan.layouts)
    first_row, first_layout = divmod(start, layout_count)
    last_row, last_layout = divmod(end, layout_count)
    if first_row == last_row:
        return (TransferSpan(rows, tuple(range(first_layout, last_layout)), range(first_row, first_row + 1)),)

    spans = []
    if first_layout:
        spans.append(TransferSpan(rows, tuple(range(first_layout, layout_count)), range(first_row, first_row + 1)))
        first_row += 1
    if first_row < last_row:
        spans.append(TransferSpan(rows, tuple(range(layout_count)), range(first_row, last_row)))
    if last_layout:
        spans.append(TransferSpan(rows, tuple(range(last_layout)), range(last_row, last_row + 1)))
    return tuple(spans)


def _remote_object(chunk, key: str, coordinate: PhysicalCoordinate):
    # Local import avoids making the semantic representation module depend on lowering.
    from .values.representation import RemoteKVObject

    return RemoteKVObject(chunk, key, coordinate)


def _lower_contiguous(
    group: KVPoolGroupTopology,
    segments: tuple[KVMemorySegment, ...],
    block_capacity: int,
) -> BoundGroupPlan:
    coordinate = PhysicalCoordinate()
    layout = _contiguous_layout(
        group,
        segments,
        tuple(layer.physical_layer_id for layer in group.layers),
        coordinate_index=0,
        order_index=0,
    )
    return _group_plan(group, (coordinate,), (layout,), (SubmissionPlan(None, (0,)),), block_capacity)


def _lower_layerwise(
    group: KVPoolGroupTopology,
    segments: tuple[KVMemorySegment, ...],
    block_capacity: int,
) -> BoundGroupPlan:
    coordinate = PhysicalCoordinate()
    total_size = sum(segment.block_length for segment in segments)
    offsets = _contiguous_offsets(tuple(segment.block_length for segment in segments))
    layouts = []
    submissions = []
    segment_start = 0
    for order_index, (physical_layer_id, layer_segments) in enumerate(_group_segments_by_layer(segments)):
        segment_end = segment_start + len(layer_segments)
        layouts.append(
            _contiguous_layout(
                group,
                layer_segments,
                (physical_layer_id,),
                coordinate_index=0,
                order_index=order_index,
                object_size=total_size,
                offsets=offsets[segment_start:segment_end],
            )
        )
        submissions.append(SubmissionPlan(physical_layer_id, (order_index,)))
        segment_start = segment_end
    return _group_plan(group, (coordinate,), tuple(layouts), tuple(submissions), block_capacity)


def _lower_strided(
    group: KVPoolGroupTopology,
    segments: tuple[KVMemorySegment, ...],
    topology: KVPoolTopology,
    block_capacity: int,
) -> BoundGroupPlan:
    slice_count = topology.tp_partition.key_slices_per_rank
    coordinates = tuple(
        PhysicalCoordinate(effective_tp_rank=topology.tp_rank * slice_count + slice_index)
        for slice_index in range(slice_count)
    )
    physical_layer_ids = tuple(layer.physical_layer_id for layer in group.layers)
    layouts = []
    for slice_index, _coordinate in enumerate(coordinates):
        bases: list[int] = []
        strides: list[int] = []
        sizes: list[int] = []
        offsets: list[int] = []
        token_indices: list[int] = []
        object_offset = 0
        for segment in segments:
            slice_size, remainder = divmod(segment.bytes_per_token, slice_count)
            if remainder:
                raise ValueError(
                    f"KV memory segment has {segment.bytes_per_token} bytes per token, "
                    f"which cannot be divided into {slice_count} Strided slices"
                )
            for token_index in range(group.block_size):
                bases.append(segment.base_address + slice_index * slice_size + token_index * segment.bytes_per_token)
                strides.append(segment.block_stride)
                sizes.append(slice_size)
                offsets.append(object_offset + token_index * slice_size)
                token_indices.append(token_index)
            object_offset += group.block_size * slice_size
        layouts.append(
            StridedLayoutPlan(
                physical_layer_ids,
                slice_index,
                object_offset,
                slice_index,
                _bytes(bases),
                _bytes(strides),
                _bytes(sizes),
                _bytes(offsets),
                _bytes(token_indices),
                group.block_size,
            )
        )
    return _group_plan(
        group,
        coordinates,
        tuple(layouts),
        (SubmissionPlan(None, tuple(range(len(layouts)))),),
        block_capacity,
    )


def _lower_pipeline_partitions(
    group: KVPoolGroupTopology,
    segments: tuple[KVMemorySegment, ...],
    partitions: tuple[int, ...],
    block_capacity: int,
) -> BoundGroupPlan:
    layers = _group_segments_by_layer(segments)
    coordinates: list[PhysicalCoordinate] = []
    layouts: list[ContiguousLayoutPlan] = []
    first_physical_layer = 0
    for pipeline_rank, layer_count in enumerate(partitions):
        last_physical_layer = first_physical_layer + layer_count
        selected_layers = tuple(
            layer
            for layer in layers
            if layer[0] >= first_physical_layer
            and (pipeline_rank == len(partitions) - 1 or layer[0] < last_physical_layer)
        )
        first_physical_layer = last_physical_layer
        if not selected_layers:
            continue
        partition_segments = tuple(segment for _, layer_segments in selected_layers for segment in layer_segments)
        coordinate_index = len(coordinates)
        coordinates.append(PhysicalCoordinate(consumer_pp_slice=pipeline_rank))
        layouts.append(
            _contiguous_layout(
                group,
                partition_segments,
                tuple(layer_id for layer_id, _ in selected_layers),
                coordinate_index=coordinate_index,
                order_index=coordinate_index,
            )
        )
    covered = {layer_id for layout in layouts for layer_id in layout.physical_layer_ids}
    expected = {layer.physical_layer_id for layer in group.layers}
    if covered != expected:
        raise ValueError(f"Pipeline partitions do not cover physical layers {sorted(expected)}")
    return _group_plan(
        group,
        tuple(coordinates),
        tuple(layouts),
        (SubmissionPlan(None, tuple(range(len(layouts)))),),
        block_capacity,
    )


def _contiguous_layout(
    group: KVPoolGroupTopology,
    segments: tuple[KVMemorySegment, ...],
    physical_layer_ids: tuple[int, ...],
    *,
    coordinate_index: int,
    order_index: int,
    object_size: int | None = None,
    offsets: tuple[int, ...] | None = None,
) -> ContiguousLayoutPlan:
    segment_sizes = tuple(segment.block_length for segment in segments)
    return ContiguousLayoutPlan(
        physical_layer_ids,
        coordinate_index,
        sum(segment_sizes) if object_size is None else object_size,
        order_index,
        _bytes(segment.base_address for segment in segments),
        _bytes(segment.block_stride for segment in segments),
        _bytes(segment.bytes_per_token for segment in segments),
        _bytes(_contiguous_offsets(segment_sizes) if offsets is None else offsets),
        group.block_size,
    )


def _group_plan(
    group: KVPoolGroupTopology,
    coordinates: tuple[PhysicalCoordinate, ...],
    layouts,
    submissions: tuple[SubmissionPlan, ...],
    block_capacity: int,
) -> BoundGroupPlan:
    prefixes = tuple(_key_prefix(group, coordinate) for coordinate in coordinates)
    return BoundGroupPlan(group.group_id, coordinates, prefixes, layouts, submissions, block_capacity)


def _key_prefix(group: KVPoolGroupTopology, coordinate: PhysicalCoordinate) -> str:
    metadata = group.key_metadata
    head_rank = metadata.head_or_tp_rank
    pp_rank = metadata.pp_rank
    dcp_rank = metadata.dcp_rank
    if coordinate.head_rank is not None:
        head_rank = coordinate.head_rank
    if coordinate.effective_tp_rank is not None:
        head_rank = coordinate.effective_tp_rank
    if coordinate.pp_rank is not None:
        pp_rank = coordinate.pp_rank
    if coordinate.consumer_pp_slice is not None:
        pp_rank = coordinate.consumer_pp_slice
    if coordinate.dcp_rank is not None:
        dcp_rank = coordinate.dcp_rank
    projected = replace(metadata, head_or_tp_rank=head_rank, pp_rank=pp_rank, dcp_rank=dcp_rank)
    return PoolKey(projected, "").to_string()


def _select_group_memory(
    topology: KVPoolTopology,
    memory_geometry: KVMemoryGeometry,
) -> dict[int, tuple[KVMemorySegment, ...]]:
    selected = {}
    for group in topology.transfer_groups:
        try:
            segments = memory_geometry[group.group_id]
        except KeyError as error:
            raise RuntimeError(f"KV cache group {group.group_id} has not registered local memory") from error
        actual = tuple(layer_id for layer_id, _ in _group_segments_by_layer(segments))
        expected = tuple(layer.physical_layer_id for layer in group.layers)
        if actual != expected:
            raise ValueError(
                f"KV cache group {group.group_id} registered physical layers {actual}, expected {expected}"
            )
        selected[group.group_id] = segments
    return selected


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


def _contiguous_offsets(sizes: tuple[int, ...]) -> tuple[int, ...]:
    offsets = []
    offset = 0
    for size in sizes:
        offsets.append(offset)
        offset += size
    return tuple(offsets)


def _bytes(values) -> np.ndarray:
    result = np.fromiter(values, dtype=np.uint64)
    result.flags.writeable = False
    return result
