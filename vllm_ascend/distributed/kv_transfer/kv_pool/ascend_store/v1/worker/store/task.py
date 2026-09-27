"""Build executable Store tasks from Worker-local cache layout."""

from __future__ import annotations

from collections.abc import Set
from dataclasses import dataclass
from typing import Protocol

import torch

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import ChunkedTokenDatabase

from ...protocol.transfer import StoreRequest
from ..projection import (
    EffectiveTPKeyProjector,
    KVObjectProjection,
    StridedKVMemoryProjector,
    bind_local_blocks,
)


@dataclass(frozen=True, slots=True)
class StoreChunk:
    """One Backend key and its aligned local source segments."""

    group_id: int
    backend_key: str
    addresses: tuple[int, ...]
    sizes: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class StoreTask:
    """A fully resolved Store operation ready for asynchronous execution."""

    request_id: str
    source_ready_event: torch.npu.Event
    chunks: tuple[StoreChunk, ...]


class StoreTaskBuilder(Protocol):
    """Build one Backend-ready task from an approved Store request."""

    def build(
        self,
        request: StoreRequest,
        source_ready_event: torch.npu.Event,
        projections: tuple[KVObjectProjection, ...],
    ) -> StoreTask: ...


class ContiguousStoreTaskBuilder:
    """Map Store chunks to the contiguous segments of local KV blocks."""

    def __init__(
        self,
        token_database: ChunkedTokenDatabase,
        tp_rank: int,
        pcp_rank: int,
        pcp_size: int,
        dcp_size: int,
        put_step: int,
        kv_role: str,
        align_state_group_ids: Set[int] = frozenset(),
    ) -> None:
        self.token_database = token_database
        self.tp_rank = tp_rank
        self.pcp_rank = pcp_rank
        self.pcp_size = pcp_size
        self.dcp_size = dcp_size
        self.put_step = put_step
        self.kv_role = kv_role
        self._align_state_group_ids = align_state_group_ids

    def build(
        self,
        request: StoreRequest,
        source_ready_event: torch.npu.Event,
        projections: tuple[KVObjectProjection, ...],
    ) -> StoreTask:
        chunks = []
        for projection in projections:
            group_id = projection.group_id
            group_block_ids = request.block_ids_by_group[group_id]
            uses_align_state = group_id in self._align_state_group_ids
            allocated_objects = bind_local_blocks(
                projection,
                group_block_ids,
                skip_null_blocks=uses_align_state,
            )
            if not allocated_objects:
                continue

            tp_replicas = self.put_step if self.dcp_size <= 1 and not uses_align_state else 1
            shard_rank = self.pcp_rank * tp_replicas + self.tp_rank % tp_replicas
            shard_size = self.pcp_size * tp_replicas
            keys = []
            addresses = []
            sizes = []
            for candidate_index, allocated_object in enumerate(allocated_objects):
                if shard_size > 1 and candidate_index % shard_size != shard_rank:
                    continue
                kv_object = allocated_object.object
                address, size, _ = self.token_database.prepare_value(
                    kv_object.token_range.start_token,
                    kv_object.token_range.end_token,
                    list(group_block_ids),
                    kv_cache_group_id=group_id,
                    block_id=allocated_object.block_id,
                )
                keys.append(kv_object.backend_key)
                addresses.append(address)
                sizes.append(size)

            if self.kv_role == "kv_consumer":
                keys, addresses, sizes = self.token_database.decode_adaptor_prefill_pp(
                    keys,
                    addresses,
                    sizes,
                    kv_cache_group_id=group_id,
                )
            chunks.extend(
                StoreChunk(group_id, key, tuple(address), tuple(size))
                for key, address, size in zip(keys, addresses, sizes)
            )
        return StoreTask(request.request_id, source_ready_event, tuple(chunks))


class StridedStoreTaskBuilder:
    """Map Store chunks to the effective-TP head slices owned by this rank."""

    def __init__(
        self,
        pcp_rank: int,
        pcp_size: int,
        key_projector: EffectiveTPKeyProjector,
        memory_projector: StridedKVMemoryProjector,
    ) -> None:
        self.pcp_rank = pcp_rank
        self.pcp_size = pcp_size
        self._key_projector = key_projector
        self._memory_projector = memory_projector

    def build(
        self,
        request: StoreRequest,
        source_ready_event: torch.npu.Event,
        projections: tuple[KVObjectProjection, ...],
    ) -> StoreTask:
        if len(projections) != 1:
            raise ValueError("Strided Store requires one cache-group projection")
        projection = projections[0]
        group_id = projection.group_id
        allocated_objects = bind_local_blocks(projection, request.block_ids_by_group[group_id])
        chunks = []
        for candidate_index, allocated_object in enumerate(allocated_objects):
            if self.pcp_size > 1 and candidate_index % self.pcp_size != self.pcp_rank:
                continue
            kv_object = allocated_object.object
            keys = self._key_projector.project(kv_object.backend_key)
            memory_slices = self._memory_projector.project(
                allocated_object.block_id,
                kv_object.token_range.end_token - kv_object.token_range.start_token,
            )
            for key, memory_slice in zip(keys, memory_slices, strict=True):
                chunks.append(StoreChunk(group_id, key, memory_slice.addresses, memory_slice.sizes))
        return StoreTask(request.request_id, source_ready_event, tuple(chunks))
