"""Build executable classic Load tasks from Worker-local cache layout."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import ChunkedTokenDatabase

from ...metadata import LoadRequest
from ..layout import StridedKVPartitioner


def _circular_shift(values: list, offset: int) -> list:
    if not values or offset == 0:
        return values
    return values[offset:] + values[:offset]


@dataclass(frozen=True, slots=True)
class LoadChunk:
    """One Backend key and its aligned local destination segments."""

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

    def build(self, request: LoadRequest) -> LoadTask: ...


class ContiguousLoadTaskBuilder:
    """Map each cached chunk to the contiguous segments of local KV blocks."""

    def __init__(
        self,
        token_database: ChunkedTokenDatabase,
        block_size: int,
        cache_transfer_granularity: int,
        tp_rank: int,
    ) -> None:
        self.token_database = token_database
        self.block_size = block_size
        self.cache_transfer_granularity = cache_transfer_granularity
        self.tp_rank = tp_rank

    def build(self, request: LoadRequest) -> LoadTask:
        token_len = _resolve_load_token_len(request, self.cache_transfer_granularity)
        block_hashes = list(request.block_hashes)
        block_ids_for_request = list(request.block_ids)
        load_masks = self.token_database.load_mask(block_hashes, token_len)
        mask_num = request.vllm_cached_tokens // self.block_size * self.block_size
        token_chunks = self.token_database.process_token_key_strings_with_block_ids(
            token_len,
            block_hashes,
            block_ids_for_request,
            mask_num,
            chunk_filter=lambda start, masks=load_masks: self.token_database.mask_allows_chunk(masks, 0, start),
        )
        chunks = []
        for start, end, key, _, block_id in token_chunks:
            address, size, block_id = self.token_database.prepare_value(
                start, end, block_ids_for_request, block_id=block_id
            )
            chunks.append(LoadChunk(key, tuple(address), tuple(size), block_id))
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

    def build(self, request: LoadRequest) -> LoadTask:
        token_len = _resolve_load_token_len(request, self.cache_transfer_granularity)
        block_ids = list(request.block_ids)
        mask_num = request.vllm_cached_tokens // self.block_size * self.block_size
        token_chunks = self.token_database.process_token_key_strings_with_block_ids(
            token_len,
            list(request.block_hashes),
            block_ids,
            mask_num,
        )
        chunks = []
        for start, end, base_key, _, block_id in token_chunks:
            for key, addresses, sizes in self.kv_partitioner.partition(base_key, block_id, end - start):
                chunks.append(LoadChunk(key, addresses, sizes, block_id))
        chunks = _circular_shift(chunks, self.kv_partitioner.tp_rank % len(chunks)) if chunks else []
        return LoadTask(request.request_id, tuple(chunks))


def _resolve_load_token_len(request: LoadRequest, cache_transfer_granularity: int) -> int:
    if request.kvpool_cached_tokens % cache_transfer_granularity != 0 and (
        request.kvpool_cached_tokens == request.transfer_end_token - 1
    ):
        return request.kvpool_cached_tokens + 1
    return request.kvpool_cached_tokens
