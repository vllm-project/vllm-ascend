"""Build executable classic Load tasks from Worker-local cache layout."""

from __future__ import annotations

from dataclasses import dataclass

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import ChunkedTokenDatabase

from ...metadata import LoadRequest


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


class LoadTaskBuilder:
    """Resolve Load requests into Backend-ready Worker-local tasks."""

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
        token_len = request.transfer_end_token
        if request.kvpool_cached_tokens % self.cache_transfer_granularity != 0 and (
            request.kvpool_cached_tokens == token_len - 1
        ):
            token_len = request.kvpool_cached_tokens + 1
        else:
            token_len = request.kvpool_cached_tokens
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
