"""Project Store work through ownership, consumer, and persistence policies."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from typing import Protocol

from ..representation import (
    BindingBatch,
    KVBinding,
    KVBlockAssignment,
    KVBlockAssignmentBatch,
    KVMemoryGeometry,
    KVMemorySegment,
    KVMemoryView,
    KVRegion,
    RemoteKVObject,
)
from ..spec.topology import KVPoolGroupTopology, KVPoolTopology
from .remote import replace_key_rank

ObjectPresenceObserver = Callable[[list[str]], tuple[int, ...]]


@dataclass(frozen=True, slots=True)
class _StoreOwnership:
    tp_rank: int
    pcp_rank: int
    pcp_size: int
    replica_count: int

    @classmethod
    def compile(cls, topology: KVPoolTopology, group: KVPoolGroupTopology) -> _StoreOwnership:
        replica_count = topology.put_step
        if topology.tp_partition.tp_mismatch or topology.dcp_size > 1 or group.uses_align_state:
            replica_count = 1
        return cls(topology.tp_rank, topology.pcp_rank, topology.pcp_size, replica_count)

    def select(self, assignments: tuple[KVBlockAssignment, ...]) -> tuple[KVBlockAssignment, ...]:
        shard_rank = self.pcp_rank * self.replica_count + self.tp_rank % self.replica_count
        shard_count = self.pcp_size * self.replica_count
        if shard_count <= 1:
            return assignments
        return tuple(item for index, item in enumerate(assignments) if index % shard_count == shard_rank)


class StoreOwnershipProjection:
    """Project Store assignments onto the subset owned by this participant."""

    def __init__(self, topology: KVPoolTopology) -> None:
        groups_by_id = {group.group_id: group for group in topology.groups}
        self.group_ids = topology.transfer_group_ids
        self._ownership = {
            group_id: _StoreOwnership.compile(topology, groups_by_id[group_id]) for group_id in self.group_ids
        }

    def project(self, batches: tuple[KVBlockAssignmentBatch, ...]) -> tuple[KVBlockAssignmentBatch, ...]:
        group_ids = tuple(batch.group_id for batch in batches)
        if group_ids != self.group_ids:
            raise ValueError(f"KV block assignment groups {group_ids} do not match compiled groups {self.group_ids}")
        return tuple(
            KVBlockAssignmentBatch(batch.group_id, self._ownership[batch.group_id].select(batch.assignments))
            for batch in batches
        )


class ConsumerProjection(Protocol):
    """Project Store bindings into the configured consumer representation."""

    def bind_memory(self, memory_geometry: KVMemoryGeometry) -> None: ...

    def project(self, batches: tuple[BindingBatch, ...]) -> tuple[BindingBatch, ...]: ...


class IdentityConsumerProjection:
    """Preserve Store bindings when no consumer pipeline partition applies."""

    @staticmethod
    def bind_memory(memory_geometry: KVMemoryGeometry) -> None:
        del memory_geometry

    @staticmethod
    def project(batches: tuple[BindingBatch, ...]) -> tuple[BindingBatch, ...]:
        return batches


@dataclass(frozen=True, slots=True)
class _ConsumerPartition:
    physical_layer_ids: tuple[int, ...]
    segment_start: int
    segment_end: int
    remote_object_size: int
    remote_offsets: tuple[int, ...]


class PipelinePartitionConsumerProjection:
    """Split Store bindings into the producer pipeline partitions expected remotely."""

    def __init__(self, partitions: tuple[int, ...]) -> None:
        self._partitions = partitions
        self._partitions_by_group: dict[int, tuple[_ConsumerPartition, ...]] | None = None

    def bind_memory(self, memory_geometry: KVMemoryGeometry) -> None:
        if self._partitions_by_group is not None:
            raise RuntimeError("Consumer pipeline projection is already bound")
        partitions_by_group = {}
        for group_id, segments in memory_geometry.items():
            layers = _group_segments_by_layer(segments)
            if sum(self._partitions) != len(layers):
                raise ValueError(
                    f"Consumer PP partitions cover {sum(self._partitions)} layers, "
                    f"but KV cache group {group_id} registered {len(layers)} physical layers"
                )
            group_partitions = []
            layer_start = 0
            segment_start = 0
            for layer_count in self._partitions:
                selected_layers = layers[layer_start : layer_start + layer_count]
                segment_end = segment_start + sum(len(layer_segments) for _, layer_segments in selected_layers)
                segment_sizes = tuple(
                    segment.block_length for _, layer_segments in selected_layers for segment in layer_segments
                )
                group_partitions.append(
                    _ConsumerPartition(
                        tuple(layer_id for layer_id, _ in selected_layers),
                        segment_start,
                        segment_end,
                        sum(segment_sizes),
                        _contiguous_offsets(segment_sizes),
                    )
                )
                layer_start += layer_count
                segment_start = segment_end
            partitions_by_group[group_id] = tuple(group_partitions)
        self._partitions_by_group = partitions_by_group

    def project(self, batches: tuple[BindingBatch, ...]) -> tuple[BindingBatch, ...]:
        return tuple(self._project_batch(batch) for batch in batches)

    def _project_batch(self, batch: BindingBatch) -> BindingBatch:
        if self._partitions_by_group is None:
            raise RuntimeError("Consumer pipeline projection is unavailable before cache registration")
        partitions = self._partitions_by_group[batch.group_id]
        projected = []
        for binding in batch.bindings:
            memory = binding.memory
            if partitions and len(memory.addresses) != partitions[-1].segment_end:
                raise ValueError("Consumer PP projection received misaligned memory segments")
            for index, partition in enumerate(partitions):
                region = KVRegion(binding.region.chunk, partition.physical_layer_ids)
                remote_object = RemoteKVObject(
                    region.chunk,
                    replace_key_rank(binding.remote_object.key, "pp_rank", index),
                    replace(binding.remote_object.coordinate, consumer_pp_slice=index),
                )
                projected.append(
                    KVBinding(
                        region,
                        remote_object,
                        partition.remote_object_size,
                        partition.remote_offsets,
                        KVMemoryView(
                            memory.block_id,
                            memory.addresses[partition.segment_start : partition.segment_end],
                            memory.sizes[partition.segment_start : partition.segment_end],
                        ),
                    )
                )
        return BindingBatch(batch.group_id, tuple(projected))


class MissingFilter(Protocol):
    """Select remote keys that still require a Backend write."""

    def select_missing_keys(self, keys: Sequence[str], observe_presence: ObjectPresenceObserver) -> tuple[str, ...]: ...


class IdentityMissingFilter:
    """Preserve every Store key when the Backend overwrites existing keys safely."""

    @staticmethod
    def select_missing_keys(keys: Sequence[str], observe_presence: ObjectPresenceObserver) -> tuple[str, ...]:
        del observe_presence
        return tuple(keys)


class BackendExistenceMissingFilter:
    """Remove Store keys that already exist in the Backend."""

    @staticmethod
    def select_missing_keys(keys: Sequence[str], observe_presence: ObjectPresenceObserver) -> tuple[str, ...]:
        presence = observe_presence(list(keys))
        if len(presence) != len(keys):
            raise RuntimeError(f"Store exists returned {len(presence)} results for {len(keys)} keys")
        if any(value not in (0, 1) for value in presence):
            raise RuntimeError("Store exists returned states other than 0 or 1")
        return tuple(key for key, value in zip(keys, presence, strict=True) if value != 1)


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
