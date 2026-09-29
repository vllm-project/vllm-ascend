"""Partition physical transfer regions for the configured remote pipeline layout."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Protocol

from ..values.representation import (
    KVMemoryGeometry,
    KVMemorySegment,
    KVMemoryView,
    KVRegion,
    KVTransferLayout,
    LocalKVRegion,
    RemoteObjectLayout,
    TransferLayoutBatch,
)


class RegionPartition(Protocol):
    """Partition Store layouts before they are bound to remote object identities."""

    def bind_memory(self, memory_geometry: KVMemoryGeometry) -> None: ...

    def project(self, batches: tuple[TransferLayoutBatch, ...]) -> tuple[TransferLayoutBatch, ...]: ...


class IdentityRegionPartition:
    """Preserve transfer regions when no remote pipeline partition applies."""

    @staticmethod
    def bind_memory(memory_geometry: KVMemoryGeometry) -> None:
        del memory_geometry

    @staticmethod
    def project(batches: tuple[TransferLayoutBatch, ...]) -> tuple[TransferLayoutBatch, ...]:
        return batches


@dataclass(frozen=True, slots=True)
class _PipelinePartition:
    pipeline_rank: int
    physical_layer_ids: tuple[int, ...]
    segment_start: int
    segment_end: int
    remote_object_size: int
    remote_offsets: tuple[int, ...]


class PipelineRegionPartition:
    """Split Store regions into the producer pipeline partitions expected remotely."""

    def __init__(self, partitions: tuple[int, ...]) -> None:
        self._partitions = partitions
        self._partitions_by_group: dict[int, tuple[_PipelinePartition, ...]] | None = None

    def bind_memory(self, memory_geometry: KVMemoryGeometry) -> None:
        if self._partitions_by_group is not None:
            raise RuntimeError("Pipeline region partition is already bound")
        partitions_by_group = {}
        for group_id, segments in memory_geometry.items():
            layers = _group_segments_by_layer(segments)
            partitions_by_group[group_id] = self._compile_group_partitions(layers)
        self._partitions_by_group = partitions_by_group

    def project(self, batches: tuple[TransferLayoutBatch, ...]) -> tuple[TransferLayoutBatch, ...]:
        return tuple(self._project_batch(batch) for batch in batches)

    def _compile_group_partitions(
        self,
        layers: tuple[tuple[int, tuple[KVMemorySegment, ...]], ...],
    ) -> tuple[_PipelinePartition, ...]:
        partitions = []
        first_physical_layer = 0
        for pipeline_rank, layer_count in enumerate(self._partitions):
            last_physical_layer = first_physical_layer + layer_count
            selected_layers = tuple(
                layer
                for layer in layers
                if layer[0] >= first_physical_layer
                and (pipeline_rank == len(self._partitions) - 1 or layer[0] < last_physical_layer)
            )
            first_physical_layer = last_physical_layer
            if not selected_layers:
                continue
            segment_start = sum(
                len(layer_segments) for layer_id, layer_segments in layers if layer_id < selected_layers[0][0]
            )
            segment_end = segment_start + sum(len(layer_segments) for _, layer_segments in selected_layers)
            segment_sizes = tuple(
                segment.block_length for _, layer_segments in selected_layers for segment in layer_segments
            )
            partitions.append(
                _PipelinePartition(
                    pipeline_rank,
                    tuple(layer_id for layer_id, _ in selected_layers),
                    segment_start,
                    segment_end,
                    sum(segment_sizes),
                    _contiguous_offsets(segment_sizes),
                )
            )
        assigned_layer_ids = {layer_id for partition in partitions for layer_id in partition.physical_layer_ids}
        registered_layer_ids = {layer_id for layer_id, _ in layers}
        if assigned_layer_ids != registered_layer_ids:
            raise ValueError(f"Pipeline partitions do not cover physical layers {sorted(registered_layer_ids)}")
        return tuple(partitions)

    def _project_batch(self, batch: TransferLayoutBatch) -> TransferLayoutBatch:
        if self._partitions_by_group is None:
            raise RuntimeError("Pipeline region partition is unavailable before cache registration")
        partitions = self._partitions_by_group[batch.group_id]
        projected = []
        for transfer_layout in batch.layouts:
            local_region = transfer_layout.local_region
            memory = local_region.memory
            if partitions and len(memory.addresses) != partitions[-1].segment_end:
                raise ValueError("Pipeline region partition received misaligned memory segments")
            for partition in partitions:
                projected.append(
                    KVTransferLayout(
                        LocalKVRegion(
                            KVRegion(local_region.region.chunk, partition.physical_layer_ids),
                            KVMemoryView(
                                memory.block_id,
                                memory.addresses[partition.segment_start : partition.segment_end],
                                memory.sizes[partition.segment_start : partition.segment_end],
                            ),
                        ),
                        RemoteObjectLayout(
                            replace(
                                transfer_layout.remote_layout.coordinate,
                                consumer_pp_slice=partition.pipeline_rank,
                            ),
                            partition.remote_object_size,
                            partition.remote_offsets,
                        ),
                    )
                )
        return TransferLayoutBatch(batch.group_id, tuple(projected))


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
    current_offset = 0
    for size in sizes:
        offsets.append(current_offset)
        current_offset += size
    return tuple(offsets)
