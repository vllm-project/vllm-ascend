"""Project logical KV selections into request-local Backend and memory batches."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from functools import partial
from typing import Protocol

from vllm.utils.math_utils import cdiv

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import (
    ChunkedTokenDatabase,
    get_block_hashes,
)

from .coordinates import TokenRange
from .elements import (
    BindingBatch,
    KVBinding,
    KVChunk,
    LocalKVSlice,
    PhysicalCoordinate,
    RemoteKVObject,
    RemoteObjectBatch,
)
from .reachability import GroupSelection, KVSelection
from .topology import KVCacheGroupTopology, KVTopology


@dataclass(frozen=True, slots=True)
class KVProjectionGroup:
    """Instance-lifetime projection facts for one transferable cache group."""

    group_id: int
    block_size: int
    minimum_contiguous_block_id: int
    base_key_rank_values: tuple[str, str, str]


@dataclass(frozen=True, slots=True)
class ChunkProjectionBatch:
    """Content-identified chunks projected for one cache group and invocation."""

    group: KVProjectionGroup
    logical_block_count: int
    chunks: tuple[KVChunk, ...]
    base_keys: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ChunkAllocation:
    """One projected chunk assigned to a Worker-local Block ID."""

    chunk: KVChunk
    base_key: str
    block_id: int


@dataclass(frozen=True, slots=True)
class ChunkAllocationBatch:
    """Chunk allocations for one cache group and invocation."""

    group: KVProjectionGroup
    allocations: tuple[ChunkAllocation, ...]


@dataclass(frozen=True, slots=True)
class _LookupRepresentation:
    coordinate: PhysicalCoordinate
    rank_values: tuple[str, str, str]


@dataclass(frozen=True, slots=True)
class _LookupKeyTemplate:
    fragments: tuple[str, str, str, str]
    rank_order: tuple[int, int, int]

    @classmethod
    def compile(cls, base_key: str) -> _LookupKeyTemplate:
        spans = sorted(
            (
                (*_required_rank_span(base_key, "dcp"), 0),
                (*_required_rank_span(base_key, "head_or_tp_rank"), 1),
                (*_required_rank_span(base_key, "pp_rank"), 2),
            )
        )
        first, second, third = spans
        fragments = (
            base_key[: first[0]],
            base_key[first[1] : second[0]],
            base_key[second[1] : third[0]],
            base_key[third[1] :],
        )
        return cls(fragments, (first[2], second[2], third[2]))

    def render(self, rank_values: tuple[str, str, str]) -> str:
        first, second, third = self.rank_order
        return "".join(
            (
                self.fragments[0],
                rank_values[first],
                self.fragments[1],
                rank_values[second],
                self.fragments[2],
                rank_values[third],
                self.fragments[3],
            )
        )


@dataclass(frozen=True, slots=True)
class _StridedRepresentation:
    coordinate: PhysicalCoordinate
    effective_rank: int
    slice_index: int


@dataclass(frozen=True, slots=True)
class _RegisteredMemorySegment:
    base_address: int
    block_length: int
    block_stride: int
    bytes_per_token: int


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
    effective_rank: int
    segments: tuple[_StridedMemorySegment, ...]


@dataclass(frozen=True, slots=True)
class _StoreOwnership:
    tp_rank: int
    pcp_rank: int
    pcp_size: int
    replica_count: int

    @classmethod
    def compile(cls, topology: KVTopology, group: KVCacheGroupTopology) -> _StoreOwnership:
        replica_count = topology.put_step
        if topology.tp_partition.tp_mismatch or topology.dcp_size > 1 or group.uses_align_state:
            replica_count = 1
        return cls(topology.tp_rank, topology.pcp_rank, topology.pcp_size, replica_count)

    def select(self, allocations: tuple[ChunkAllocation, ...]) -> tuple[ChunkAllocation, ...]:
        shard_rank = self.pcp_rank * self.replica_count + self.tp_rank % self.replica_count
        shard_count = self.pcp_size * self.replica_count
        if shard_count <= 1:
            return allocations
        return tuple(allocation for index, allocation in enumerate(allocations) if index % shard_count == shard_rank)


class BindingProjection(Protocol):
    """Project allocated chunks into one fixed object-memory representation."""

    def compile_memory_mapping(self, groups: dict[int, KVProjectionGroup]) -> None: ...

    def assign_local_blocks(
        self, projection_batch: ChunkProjectionBatch, block_ids: Sequence[int]
    ) -> ChunkAllocationBatch: ...

    def bind_representations(self, allocation_batch: ChunkAllocationBatch) -> BindingBatch: ...


class ContiguousBindingProjection:
    """Bind each logical KV chunk to contiguous segments in one local block."""

    def __init__(self, token_database: ChunkedTokenDatabase) -> None:
        self._token_database = token_database
        self._segments_by_group: dict[int, tuple[_RegisteredMemorySegment, ...]] | None = None

    def compile_memory_mapping(self, groups: dict[int, KVProjectionGroup]) -> None:
        if self._segments_by_group is not None:
            raise RuntimeError("Contiguous binding projection is already bound")
        self._segments_by_group = {
            group_id: _compile_memory_segments(self._token_database, group) for group_id, group in groups.items()
        }

    def assign_local_blocks(
        self, projection_batch: ChunkProjectionBatch, block_ids: Sequence[int]
    ) -> ChunkAllocationBatch:
        allocations = _allocate_group(
            projection_batch,
            block_ids,
            minimum_block_id=projection_batch.group.minimum_contiguous_block_id,
        )
        return ChunkAllocationBatch(projection_batch.group, allocations)

    def bind_representations(
        self,
        allocation_batch: ChunkAllocationBatch,
    ) -> BindingBatch:
        if self._segments_by_group is None:
            raise RuntimeError("KV memory projection is unavailable before cache registration")
        bindings = []
        segments = self._segments_by_group[allocation_batch.group.group_id]
        for allocation in allocation_batch.allocations:
            chunk = allocation.chunk
            token_count = chunk.token_range.end_token - chunk.token_range.start_token
            addresses = tuple(segment.base_address + allocation.block_id * segment.block_stride for segment in segments)
            sizes = tuple(segment.bytes_per_token * token_count for segment in segments)
            remote_object = RemoteKVObject(chunk, allocation.base_key)
            local_slice = LocalKVSlice(chunk, allocation.block_id, addresses, sizes)
            bindings.append(KVBinding(remote_object, local_slice))
        return BindingBatch(allocation_batch.group.group_id, tuple(bindings))


class StridedBindingProjection:
    """Split each logical KV chunk into fixed effective-rank memory slices."""

    def __init__(self, token_database: ChunkedTokenDatabase, topology: KVTopology) -> None:
        if len(topology.transfer_group_ids) != 1:
            raise ValueError("AscendStore v1 TP mismatch requires one transferable KV cache group")
        self._token_database = token_database
        self._group_id = topology.transfer_group_ids[0]
        representations = []
        for slice_index in range(topology.tp_partition.key_slices_per_rank):
            effective_rank = topology.tp_rank * topology.tp_partition.key_slices_per_rank + slice_index
            representations.append(
                _StridedRepresentation(
                    PhysicalCoordinate(effective_tp_rank=effective_rank),
                    effective_rank,
                    slice_index,
                )
            )
        self._representations = tuple(representations)
        self._group: KVProjectionGroup | None = None
        self._bound_representations: tuple[_BoundStridedRepresentation, ...] | None = None

    def compile_memory_mapping(self, groups: dict[int, KVProjectionGroup]) -> None:
        if self._bound_representations is not None:
            raise RuntimeError("Strided binding projection is already bound")
        group = groups[self._group_id]
        segments = _compile_memory_segments(self._token_database, group)
        if not segments:
            raise RuntimeError(f"KV cache group {group.group_id} registered no local memory segments")
        self._group = group
        self._bound_representations = tuple(
            _BoundStridedRepresentation(
                representation.coordinate,
                representation.effective_rank,
                tuple(
                    _compile_strided_segment(
                        segment,
                        len(self._representations),
                        representation.slice_index,
                    )
                    for segment in segments
                ),
            )
            for representation in self._representations
        )

    def assign_local_blocks(
        self, projection_batch: ChunkProjectionBatch, block_ids: Sequence[int]
    ) -> ChunkAllocationBatch:
        allocations = _allocate_group(projection_batch, block_ids, minimum_block_id=0)
        return ChunkAllocationBatch(projection_batch.group, allocations)

    def bind_representations(
        self,
        allocation_batch: ChunkAllocationBatch,
    ) -> BindingBatch:
        if self._group is None or self._bound_representations is None:
            raise RuntimeError("KV memory projection is unavailable before cache registration")
        if allocation_batch.group != self._group:
            raise ValueError(
                f"Strided binding projection was bound for group {self._group.group_id}, "
                f"received {allocation_batch.group.group_id}"
            )
        bindings = []
        for allocation in allocation_batch.allocations:
            chunk = allocation.chunk
            token_count = chunk.token_range.end_token - chunk.token_range.start_token
            for representation in self._bound_representations:
                addresses = []
                sizes = []
                for segment in representation.segments:
                    block_address = segment.base_address + allocation.block_id * segment.block_stride
                    for token_index in range(token_count):
                        addresses.append(block_address + token_index * segment.bytes_per_token + segment.head_offset)
                        sizes.append(segment.slice_size)
                remote_object = RemoteKVObject(
                    chunk,
                    _replace_rank(allocation.base_key, "head_or_tp_rank", representation.effective_rank),
                    representation.coordinate,
                )
                local_slice = LocalKVSlice(
                    chunk,
                    allocation.block_id,
                    tuple(addresses),
                    tuple(sizes),
                    representation.coordinate,
                )
                bindings.append(KVBinding(remote_object, local_slice))
        return BindingBatch(allocation_batch.group.group_id, tuple(bindings))


class ConsumerProjection(Protocol):
    """Project Store bindings into the configured consumer representation."""

    def compile_memory_mapping(self) -> None: ...

    def project(self, batches: tuple[BindingBatch, ...]) -> tuple[BindingBatch, ...]: ...


class IdentityConsumerProjection:
    """Preserve Store bindings when no consumer pipeline partition applies."""

    @staticmethod
    def compile_memory_mapping() -> None:
        return

    @staticmethod
    def project(batches: tuple[BindingBatch, ...]) -> tuple[BindingBatch, ...]:
        return batches


class PipelinePartitionConsumerProjection:
    """Split Store bindings into the producer pipeline partitions expected remotely."""

    def __init__(self, token_database: ChunkedTokenDatabase, partitions: tuple[int, ...]) -> None:
        self._token_database = token_database
        self._partitions = partitions
        self._segment_ranges_by_group: dict[int, tuple[tuple[int, int], ...]] | None = None

    def compile_memory_mapping(self) -> None:
        if self._segment_ranges_by_group is not None:
            raise RuntimeError("Consumer pipeline projection is already bound")
        segment_ranges_by_group = {}
        group_num_layers = self._token_database.group_num_layers.get("kv", {})
        for group_id, addresses in self._token_database.group_kv_caches_base_addr.items():
            num_layers = group_num_layers.get(group_id)
            if num_layers is None or num_layers <= 0:
                raise ValueError(f"KV cache group {group_id} has no registered layer count")
            if len(addresses) % num_layers != 0:
                raise ValueError(
                    f"KV cache group {group_id} has {len(addresses)} memory segments for {num_layers} layers"
                )
            if sum(self._partitions) != num_layers:
                raise ValueError(
                    f"Consumer PP partitions cover {sum(self._partitions)} layers, "
                    f"but KV cache group {group_id} registered {num_layers}"
                )
            segments_per_layer = len(addresses) // num_layers
            segment_ranges = []
            start = 0
            for layer_count in self._partitions:
                end = start + layer_count * segments_per_layer
                segment_ranges.append((start, end))
                start = end
            segment_ranges_by_group[group_id] = tuple(segment_ranges)
        self._segment_ranges_by_group = segment_ranges_by_group

    def project(self, batches: tuple[BindingBatch, ...]) -> tuple[BindingBatch, ...]:
        return tuple(self._project_batch(batch) for batch in batches)

    def _project_batch(self, batch: BindingBatch) -> BindingBatch:
        if self._segment_ranges_by_group is None:
            raise RuntimeError("Consumer pipeline projection is unavailable before cache registration")
        segment_ranges = self._segment_ranges_by_group[batch.group_id]
        projected = []
        for binding in batch.bindings:
            addresses = binding.local_slice.addresses
            sizes = binding.local_slice.sizes
            if len(addresses) != len(sizes) or segment_ranges and len(addresses) != segment_ranges[-1][1]:
                raise ValueError("Consumer PP projection received misaligned memory segments")
            for index, (start, end) in enumerate(segment_ranges):
                coordinate = replace(binding.remote_object.coordinate, consumer_pp_slice=index)
                remote_object = RemoteKVObject(
                    binding.remote_object.chunk,
                    _replace_rank(binding.remote_object.key, "pp_rank", index),
                    coordinate,
                )
                local_slice = LocalKVSlice(
                    binding.local_slice.chunk,
                    binding.local_slice.block_id,
                    addresses[start:end],
                    sizes[start:end],
                    coordinate,
                )
                projected.append(KVBinding(remote_object, local_slice))
        return BindingBatch(batch.group_id, tuple(projected))


class KVProjection:
    """Apply fixed object, allocation and binding projections to request-local values."""

    def __init__(
        self,
        token_database: ChunkedTokenDatabase,
        topology: KVTopology,
        binding_projection: BindingProjection,
    ) -> None:
        self._token_database = token_database
        self._binding_projection = binding_projection
        groups_by_id = {group.group_id: group for group in topology.kv_cache_groups}
        try:
            group_topologies = tuple(groups_by_id[group_id] for group_id in topology.transfer_group_ids)
        except KeyError as error:
            raise ValueError(f"Unknown transferable KV cache group {error.args[0]}") from error
        self.group_ids = tuple(group.group_id for group in group_topologies)
        self._groups = {
            group.group_id: KVProjectionGroup(
                group.group_id,
                group.block_size,
                1 if group.uses_align_state else 0,
                (
                    str(group.key_metadata.dcp_rank),
                    str(group.key_metadata.head_or_tp_rank),
                    str(group.key_metadata.pp_rank),
                ),
            )
            for group in group_topologies
        }
        self._lookup_representations = tuple(
            _LookupRepresentation(
                PhysicalCoordinate(pp_rank=pp_rank, dcp_rank=dcp_rank, head_rank=head_rank),
                (str(dcp_rank), str(head_rank), str(pp_rank)),
            )
            for pp_rank in range(topology.pp_size)
            for dcp_rank in range(topology.dcp_size)
            for head_rank in range(topology.tp_partition.key_rank_count)
        )
        self._store_ownership_by_group = {
            group.group_id: _StoreOwnership.compile(topology, group) for group in group_topologies
        }

    def compile_memory_mapping(self) -> None:
        """Compile address formulas after Worker-local KV buffers have been registered."""

        self._binding_projection.compile_memory_mapping(self._groups)

    def project_chunks(self, selection: KVSelection) -> tuple[ChunkProjectionBatch, ...]:
        """Project a logical selection into content-identified base objects."""

        selection_group_ids = tuple(group.group_id for group in selection.groups)
        if selection_group_ids != self.group_ids:
            raise ValueError(f"KV selection groups {selection_group_ids} do not match compiled groups {self.group_ids}")
        return tuple(self._project_chunks(selection, group) for group in selection.groups)

    def project_remote_objects(
        self, projection_batches: tuple[ChunkProjectionBatch, ...]
    ) -> tuple[RemoteObjectBatch, ...]:
        """Expand base objects into every remote representation required by Lookup."""

        batches = []
        for projection_batch in projection_batches:
            if len(self._lookup_representations) == 1:
                representation = self._lookup_representations[0]
                if representation.rank_values == projection_batch.group.base_key_rank_values:
                    remote_objects = tuple(
                        RemoteKVObject(chunk, base_key, representation.coordinate)
                        for chunk, base_key in zip(
                            projection_batch.chunks,
                            projection_batch.base_keys,
                            strict=True,
                        )
                    )
                    batches.append(
                        RemoteObjectBatch(
                            projection_batch.group.group_id,
                            projection_batch.chunks,
                            remote_objects,
                        )
                    )
                    continue
            key_templates = tuple(_LookupKeyTemplate.compile(base_key) for base_key in projection_batch.base_keys)
            remote_objects = tuple(
                RemoteKVObject(chunk, template.render(representation.rank_values), representation.coordinate)
                for representation in self._lookup_representations
                for chunk, template in zip(projection_batch.chunks, key_templates, strict=True)
            )
            batches.append(
                RemoteObjectBatch(
                    projection_batch.group.group_id,
                    projection_batch.chunks,
                    remote_objects,
                )
            )
        return tuple(batches)

    def assign_local_blocks(
        self,
        projection_batches: tuple[ChunkProjectionBatch, ...],
        block_ids_by_group: tuple[tuple[int, ...], ...],
    ) -> tuple[ChunkAllocationBatch, ...]:
        """Bind projected objects to request-local vLLM Block IDs."""

        allocation_batches = []
        for projection_batch in projection_batches:
            group_id = projection_batch.group.group_id
            allocation_batches.append(
                self._binding_projection.assign_local_blocks(projection_batch, block_ids_by_group[group_id])
            )
        return tuple(allocation_batches)

    def select_owned_allocations(
        self, allocation_batches: tuple[ChunkAllocationBatch, ...]
    ) -> tuple[ChunkAllocationBatch, ...]:
        """Retain only the local replicas owned by this Store participant."""

        return tuple(
            ChunkAllocationBatch(
                batch.group,
                self._store_ownership_by_group[batch.group.group_id].select(batch.allocations),
            )
            for batch in allocation_batches
        )

    def bind_representations(self, allocation_batches: tuple[ChunkAllocationBatch, ...]) -> tuple[BindingBatch, ...]:
        """Project allocated objects into concrete Backend and local-memory bindings."""

        return tuple(self._binding_projection.bind_representations(batch) for batch in allocation_batches)

    def _project_chunks(self, selection: KVSelection, group_selection: GroupSelection) -> ChunkProjectionBatch:
        group = self._groups[group_selection.group_id]
        hashes = list(selection.block_hashes)
        aligned_start = selection.token_range.start_token // group.block_size * group.block_size
        logical_block_count = min(
            len(get_block_hashes(hashes, group.block_size, self._token_database.hash_block_size)),
            cdiv(selection.token_range.end_token, group.block_size),
        )
        key_records = tuple(
            self._token_database.process_token_key_strings(
                selection.token_range.end_token,
                hashes,
                mask_num=aligned_start,
                kv_cache_group_id=group.group_id,
                chunk_filter=partial(group_selection.includes, block_size=group.block_size),
            )
        )
        chunks = tuple(
            KVChunk(group.group_id, start // group.block_size, TokenRange(start, end), content_hash)
            for start, end, _, content_hash in key_records
        )
        return ChunkProjectionBatch(group, logical_block_count, chunks, tuple(key for _, _, key, _ in key_records))


def _compile_memory_segments(
    token_database: ChunkedTokenDatabase,
    group: KVProjectionGroup,
) -> tuple[_RegisteredMemorySegment, ...]:
    group_id = group.group_id
    try:
        addresses = token_database.group_kv_caches_base_addr[group_id]
        block_lengths = token_database.group_block_len[group_id]
    except KeyError as error:
        raise RuntimeError(f"KV cache group {group_id} has not registered local memory") from error
    block_strides = token_database.group_block_stride.get(group_id)
    if len(addresses) != len(block_lengths) or block_strides is not None and len(addresses) != len(block_strides):
        raise ValueError(f"KV cache group {group_id} registered misaligned memory geometry")
    return tuple(
        _RegisteredMemorySegment(
            base_address,
            block_lengths[index],
            block_strides[index] if block_strides else block_lengths[index],
            block_lengths[index] // group.block_size,
        )
        for index, base_address in enumerate(addresses)
    )


def _allocate_group(
    projection_batch: ChunkProjectionBatch,
    block_ids: Sequence[int],
    *,
    minimum_block_id: int,
) -> tuple[ChunkAllocation, ...]:
    block_offset = max(projection_batch.logical_block_count - len(block_ids), 0)
    allocations = []
    for chunk, base_key in zip(projection_batch.chunks, projection_batch.base_keys, strict=True):
        local_block_index = chunk.block_index - block_offset
        if 0 <= local_block_index < len(block_ids):
            block_id = block_ids[local_block_index]
            if block_id >= minimum_block_id:
                allocations.append(ChunkAllocation(chunk, base_key, block_id))
    return tuple(allocations)


def _compile_strided_segment(
    segment: _RegisteredMemorySegment,
    slice_count: int,
    slice_index: int,
) -> _StridedMemorySegment:
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


def _required_rank_span(key: str, field: str) -> tuple[int, int]:
    marker = f"@{field}:"
    value_start = key.index(marker) + len(marker)
    value_end = key.index("@", value_start)
    return value_start, value_end


def _replace_rank(key: str, field: str, rank: int) -> str:
    marker = f"@{field}:"
    marker_start = key.find(marker)
    if marker_start < 0:
        return key
    value_start = marker_start + len(marker)
    value_end = key.find("@", value_start)
    if value_end < 0:
        value_end = len(key)
    return f"{key[:value_start]}{rank}{key[value_end:]}"
