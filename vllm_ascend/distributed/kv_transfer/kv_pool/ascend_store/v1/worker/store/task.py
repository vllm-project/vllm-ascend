"""Build executable Store tasks from Worker-local cache layout."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import torch
from vllm.v1.core.kv_cache_utils import BlockHash

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import ChunkedTokenDatabase

from ...protocol.transfer import StoreRequest
from ..coordinator import ChunkSelection
from ..layout import StridedKVPartitioner


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
        selections: Sequence[ChunkSelection],
    ) -> StoreTask: ...


class ContiguousStoreTaskBuilder:
    """Map Store chunks to the contiguous segments of local KV blocks."""

    def __init__(
        self,
        token_database: ChunkedTokenDatabase,
        group_block_sizes: dict[int, int],
        tp_rank: int,
        pcp_rank: int,
        pcp_size: int,
        dcp_size: int,
        put_step: int,
        kv_role: str,
    ) -> None:
        self.token_database = token_database
        self.group_block_sizes = group_block_sizes
        self.tp_rank = tp_rank
        self.pcp_rank = pcp_rank
        self.pcp_size = pcp_size
        self.dcp_size = dcp_size
        self.put_step = put_step
        self.kv_role = kv_role

    def build(
        self,
        request: StoreRequest,
        source_ready_event: torch.npu.Event,
        selections: Sequence[ChunkSelection],
    ) -> StoreTask:
        chunks = []
        for selection in selections:
            group_id = selection.group_id
            candidates = self._select_group_chunks(request, selection)
            if not candidates:
                continue

            keys = []
            addresses = []
            sizes = []
            group_block_ids = list(request.block_ids_by_group[group_id])
            for start, end, key, _, block_id in candidates:
                address, size, _ = self.token_database.prepare_value(
                    start,
                    end,
                    group_block_ids,
                    kv_cache_group_id=group_id,
                    block_id=block_id,
                )
                keys.append(key)
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

    def _select_group_chunks(
        self,
        request: StoreRequest,
        selection: ChunkSelection,
    ) -> list[tuple[int, int, str, BlockHash | str, int]]:
        if selection.chunk_mask is not None and not any(selection.chunk_mask):
            return []

        group_id = selection.group_id
        group_block_size = self.group_block_sizes[group_id]

        tp_replicas = self.put_step if self.dcp_size <= 1 else 1
        chunks = self.token_database.process_token_key_strings_with_block_ids(
            request.store_end_token,
            list(request.block_hashes),
            list(request.block_ids_by_group[group_id]),
            kv_cache_group_id=group_id,
            chunk_filter=lambda start: selection.includes(start, group_block_size),
            shard_rank=self.pcp_rank * tp_replicas + self.tp_rank % tp_replicas,
            shard_size=self.pcp_size * tp_replicas,
        )
        return list(chunks)


class StridedStoreTaskBuilder:
    """Map Store chunks to the effective-TP head slices owned by this rank."""

    def __init__(
        self,
        token_database: ChunkedTokenDatabase,
        block_size: int,
        pcp_rank: int,
        pcp_size: int,
        kv_partitioner: StridedKVPartitioner,
    ) -> None:
        self.token_database = token_database
        self.block_size = block_size
        self.pcp_rank = pcp_rank
        self.pcp_size = pcp_size
        self.kv_partitioner = kv_partitioner

    def build(
        self,
        request: StoreRequest,
        source_ready_event: torch.npu.Event,
        selections: Sequence[ChunkSelection],
    ) -> StoreTask:
        if len(selections) != 1:
            raise ValueError("Strided Store requires one cache-group selection")
        selection = selections[0]
        group_id = selection.group_id
        token_chunks = self.token_database.process_token_key_strings_with_block_ids(
            request.store_end_token,
            list(request.block_hashes),
            list(request.block_ids_by_group[group_id]),
            kv_cache_group_id=group_id,
            chunk_filter=lambda start: selection.includes(start, self.block_size),
            shard_rank=self.pcp_rank,
            shard_size=self.pcp_size,
        )
        chunks = []
        for start, end, base_key, _, block_id in token_chunks:
            for key, addresses, sizes in self.kv_partitioner.partition(base_key, block_id, end - start):
                chunks.append(StoreChunk(group_id, key, addresses, sizes))
        return StoreTask(request.request_id, source_ready_event, tuple(chunks))
