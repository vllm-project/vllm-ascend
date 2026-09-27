"""Build executable Load tasks from Worker-local cache layout."""

from __future__ import annotations

from collections.abc import Set
from dataclasses import dataclass
from typing import Protocol

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import ChunkedTokenDatabase

from ...protocol.transfer import LoadRequest
from ..projection import (
    EffectiveTPKeyProjector,
    KVObjectProjection,
    StridedKVMemoryProjector,
    bind_local_blocks,
)


def _circular_shift(values: list, offset: int) -> list:
    if not values or offset == 0:
        return values
    return values[offset:] + values[:offset]


@dataclass(frozen=True, slots=True)
class LoadChunk:
    """One Backend key and its aligned local destination segments."""

    group_id: int
    backend_key: str
    addresses: tuple[int, ...]
    sizes: tuple[int, ...]
    block_id: int


@dataclass(frozen=True, slots=True)
class LoadTask:
    """A fully resolved Load operation ready for execution."""

    request_id: str
    chunks: tuple[LoadChunk, ...]


class LoadTaskBuilder(Protocol):
    """Build one Backend-ready task from an approved Load request."""

    def build(self, request: LoadRequest, projections: tuple[KVObjectProjection, ...]) -> LoadTask: ...


class ContiguousLoadTaskBuilder:
    """Map each cached chunk to the contiguous segments of local KV blocks."""

    def __init__(
        self,
        token_database: ChunkedTokenDatabase,
        tp_rank: int,
        align_state_group_ids: Set[int] = frozenset(),
    ) -> None:
        self.token_database = token_database
        self.tp_rank = tp_rank
        self._align_state_group_ids = align_state_group_ids

    def build(self, request: LoadRequest, projections: tuple[KVObjectProjection, ...]) -> LoadTask:
        chunks = []
        for projection in projections:
            group_id = projection.group_id
            group_block_ids = request.block_ids_by_group[group_id]
            allocated_objects = bind_local_blocks(
                projection,
                group_block_ids,
                skip_null_blocks=group_id in self._align_state_group_ids,
            )
            for allocated_object in allocated_objects:
                kv_object = allocated_object.object
                address, size, block_id = self.token_database.prepare_value(
                    kv_object.token_range.start_token,
                    kv_object.token_range.end_token,
                    list(group_block_ids),
                    kv_cache_group_id=group_id,
                    block_id=allocated_object.block_id,
                )
                chunks.append(LoadChunk(group_id, kv_object.backend_key, tuple(address), tuple(size), block_id))
        chunks = _circular_shift(chunks, self.tp_rank % len(chunks)) if chunks else []
        return LoadTask(request.request_id, tuple(chunks))


class StridedLoadTaskBuilder:
    """Map each cached chunk to the KV head slices owned by the local TP rank."""

    def __init__(
        self,
        key_projector: EffectiveTPKeyProjector,
        memory_projector: StridedKVMemoryProjector,
    ) -> None:
        self._key_projector = key_projector
        self._memory_projector = memory_projector

    def build(self, request: LoadRequest, projections: tuple[KVObjectProjection, ...]) -> LoadTask:
        if len(projections) != 1:
            raise ValueError("Strided Load requires one cache-group projection")
        projection = projections[0]
        group_id = projection.group_id
        block_ids = request.block_ids_by_group[group_id]
        chunks = []
        for allocated_object in bind_local_blocks(projection, block_ids):
            kv_object = allocated_object.object
            keys = self._key_projector.project(kv_object.backend_key)
            memory_slices = self._memory_projector.project(
                allocated_object.block_id,
                kv_object.token_range.end_token - kv_object.token_range.start_token,
            )
            for key, memory_slice in zip(keys, memory_slices, strict=True):
                chunks.append(
                    LoadChunk(
                        group_id,
                        key,
                        memory_slice.addresses,
                        memory_slice.sizes,
                        allocated_object.block_id,
                    )
                )
        chunks = _circular_shift(chunks, self._key_projector.tp_rank % len(chunks)) if chunks else []
        return LoadTask(request.request_id, tuple(chunks))
