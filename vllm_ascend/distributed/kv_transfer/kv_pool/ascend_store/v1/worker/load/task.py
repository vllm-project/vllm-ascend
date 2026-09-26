"""Build executable Load tasks from Worker-local cache layout."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from functools import partial
from typing import Protocol

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import ChunkedTokenDatabase

from ...protocol.transfer import LoadRequest
from ..coordinator import ChunkSelection
from ..layout import StridedKVPartitioner


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

    def build(self, request: LoadRequest, load_end_token: int, selections: Sequence[ChunkSelection]) -> LoadTask: ...


class ContiguousLoadTaskBuilder:
    """Map each cached chunk to the contiguous segments of local KV blocks."""

    def __init__(self, token_database: ChunkedTokenDatabase, group_block_sizes: dict[int, int], tp_rank: int) -> None:
        self.token_database = token_database
        self.group_block_sizes = group_block_sizes
        self.tp_rank = tp_rank

    def build(self, request: LoadRequest, load_end_token: int, selections: Sequence[ChunkSelection]) -> LoadTask:
        block_hashes = list(request.block_hashes)
        chunks = []
        for selection in selections:
            group_id = selection.group_id
            group_block_size = self.group_block_sizes[group_id]
            group_block_ids = list(request.block_ids_by_group[group_id])
            local_cache_boundary = request.local_cached_tokens // group_block_size * group_block_size

            token_chunks = self.token_database.process_token_key_strings_with_block_ids(
                load_end_token,
                block_hashes,
                group_block_ids,
                local_cache_boundary,
                kv_cache_group_id=group_id,
                chunk_filter=partial(selection.includes, block_size=group_block_size),
            )
            for start, end, key, _, block_id in token_chunks:
                address, size, block_id = self.token_database.prepare_value(
                    start,
                    end,
                    group_block_ids,
                    kv_cache_group_id=group_id,
                    block_id=block_id,
                )
                chunks.append(LoadChunk(group_id, key, tuple(address), tuple(size), block_id))
        chunks = _circular_shift(chunks, self.tp_rank % len(chunks)) if chunks else []
        return LoadTask(request.request_id, tuple(chunks))


class StridedLoadTaskBuilder:
    """Map each cached chunk to the KV head slices owned by the local TP rank."""

    def __init__(
        self,
        token_database: ChunkedTokenDatabase,
        block_size: int,
        cache_transfer_granularity: int,
        kv_partitioner: StridedKVPartitioner,
    ) -> None:
        self.token_database = token_database
        self.block_size = block_size
        self.cache_transfer_granularity = cache_transfer_granularity
        self.kv_partitioner = kv_partitioner

    def build(self, request: LoadRequest, load_end_token: int, selections: Sequence[ChunkSelection]) -> LoadTask:
        if len(selections) != 1:
            raise ValueError("Strided Load requires one cache-group selection")
        selection = selections[0]
        group_id = selection.group_id
        block_ids = list(request.block_ids_by_group[group_id])
        local_cache_boundary = request.local_cached_tokens // self.block_size * self.block_size
        token_chunks = self.token_database.process_token_key_strings_with_block_ids(
            load_end_token,
            list(request.block_hashes),
            block_ids,
            local_cache_boundary,
            kv_cache_group_id=group_id,
            chunk_filter=lambda start: selection.includes(start, self.block_size),
        )
        chunks = []
        for start, end, base_key, _, block_id in token_chunks:
            for key, addresses, sizes in self.kv_partitioner.partition(base_key, block_id, end - start):
                chunks.append(LoadChunk(group_id, key, addresses, sizes, block_id))
        chunks = _circular_shift(chunks, self.kv_partitioner.tp_rank % len(chunks)) if chunks else []
        return LoadTask(request.request_id, tuple(chunks))


def resolve_load_end_token(request: LoadRequest, cache_transfer_granularity: int) -> int:
    if request.kv_pool_cached_tokens % cache_transfer_granularity != 0 and (
        request.kv_pool_cached_tokens == request.transfer_end_token - 1
    ):
        return request.kv_pool_cached_tokens + 1
    return request.kv_pool_cached_tokens
