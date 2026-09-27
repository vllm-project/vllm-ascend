"""Build executable Store tasks from Worker-local cache layout."""

from __future__ import annotations

from collections.abc import Set
from dataclasses import dataclass
from typing import Protocol

import torch

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import ChunkedTokenDatabase

from ...protocol.transfer import StoreRequest
from ..projection import (
    ContiguousKVBindingProjector,
    KVBinding,
    KVMemorySlice,
    KVObjectProjection,
    StridedKVBindingProjector,
    bind_local_blocks,
)


@dataclass(frozen=True, slots=True)
class StoreTask:
    """A fully resolved Store operation ready for asynchronous execution."""

    request_id: str
    source_ready_event: torch.npu.Event
    bindings: tuple[KVBinding, ...]


class StoreTaskBuilder(Protocol):
    """Build one Backend-ready task from an approved Store request."""

    def build(
        self,
        request: StoreRequest,
        source_ready_event: torch.npu.Event,
        projections: tuple[KVObjectProjection, ...],
    ) -> StoreTask: ...


class ContiguousStoreTaskBuilder:
    """Select owned objects and compile their contiguous Store bindings."""

    def __init__(
        self,
        token_database: ChunkedTokenDatabase,
        binding_projector: ContiguousKVBindingProjector,
        tp_rank: int,
        pcp_rank: int,
        pcp_size: int,
        dcp_size: int,
        put_step: int,
        kv_role: str,
        align_state_group_ids: Set[int] = frozenset(),
    ) -> None:
        self.token_database = token_database
        self._binding_projector = binding_projector
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
        bindings = []
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
            owned_objects = tuple(
                allocated_object
                for candidate_index, allocated_object in enumerate(allocated_objects)
                if shard_size <= 1 or candidate_index % shard_size == shard_rank
            )
            group_bindings = self._binding_projector.project(group_id, group_block_ids, owned_objects)
            if self.kv_role == "kv_consumer":
                group_bindings = self._adapt_consumer_pp(group_bindings)
            bindings.extend(group_bindings)
        return StoreTask(request.request_id, source_ready_event, tuple(bindings))

    def _adapt_consumer_pp(self, bindings: tuple[KVBinding, ...]) -> tuple[KVBinding, ...]:
        adapted_bindings = []
        for binding in bindings:
            keys, addresses, sizes = self.token_database.decode_adaptor_prefill_pp(
                [binding.backend_key],
                [list(binding.memory_slice.addresses)],
                [list(binding.memory_slice.sizes)],
                kv_cache_group_id=binding.group_id,
            )
            adapted_bindings.extend(
                KVBinding(
                    binding.group_id,
                    binding.base_object,
                    key,
                    binding.block_id,
                    KVMemorySlice(tuple(address), tuple(size)),
                )
                for key, address, size in zip(keys, addresses, sizes)
            )
        return tuple(adapted_bindings)


class StridedStoreTaskBuilder:
    """Select owned objects and compile their effective-TP Store bindings."""

    def __init__(
        self,
        pcp_rank: int,
        pcp_size: int,
        binding_projector: StridedKVBindingProjector,
    ) -> None:
        self.pcp_rank = pcp_rank
        self.pcp_size = pcp_size
        self._binding_projector = binding_projector

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
        owned_objects = tuple(
            allocated_object
            for candidate_index, allocated_object in enumerate(allocated_objects)
            if self.pcp_size <= 1 or candidate_index % self.pcp_size == self.pcp_rank
        )
        bindings = self._binding_projector.project(group_id, owned_objects)
        return StoreTask(request.request_id, source_ready_event, bindings)
