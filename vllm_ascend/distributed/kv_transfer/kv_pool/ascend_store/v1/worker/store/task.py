"""Build executable classic Store tasks from Worker-local cache layout."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from vllm.logger import logger
from vllm.v1.core.kv_cache_utils import BlockHash

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import ChunkedTokenDatabase

from ...metadata import StoreRequest


@dataclass(frozen=True, slots=True)
class StoreChunk:
    """One Backend key and its aligned local source segments."""

    backend_key: str
    addresses: tuple[int, ...]
    sizes: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class StoreTask:
    """A fully resolved Store operation ready for asynchronous execution."""

    request_id: str
    source_ready_event: torch.npu.Event
    chunks: tuple[StoreChunk, ...]


class StoreTaskBuilder:
    """Resolve Store requests into Backend-neutral Worker-local tasks."""

    def __init__(
        self,
        token_database: ChunkedTokenDatabase,
        block_size: int,
        tp_rank: int,
        pcp_rank: int,
        pcp_size: int,
        dcp_size: int,
        put_step: int,
        kv_role: str,
    ) -> None:
        self.token_database = token_database
        self.block_size = block_size
        self.tp_rank = tp_rank
        self.pcp_rank = pcp_rank
        self.pcp_size = pcp_size
        self.dcp_size = dcp_size
        self.put_step = put_step
        self.kv_role = kv_role

    def build(self, request: StoreRequest, source_ready_event: torch.npu.Event) -> StoreTask:
        candidates = self._select_store_chunks(request)
        if not candidates:
            return StoreTask(request.request_id, source_ready_event, ())

        keys = []
        addresses = []
        sizes = []
        block_ids = list(request.block_ids)
        for start, end, key, _, block_id in candidates:
            address, size, _ = self.token_database.prepare_value(start, end, block_ids, block_id=block_id)
            keys.append(key)
            addresses.append(address)
            sizes.append(size)

        if self.kv_role == "kv_consumer":
            keys, addresses, sizes = self.token_database.decode_adaptor_prefill_pp(keys, addresses, sizes)
        chunks = tuple(
            StoreChunk(key, tuple(address), tuple(size)) for key, address, size in zip(keys, addresses, sizes)
        )
        return StoreTask(request.request_id, source_ready_event, chunks)

    def _select_store_chunks(self, request: StoreRequest) -> list[tuple[int, int, str, BlockHash | str, int]]:
        token_len = request.save_end_token
        try:
            store_masks = self.token_database.store_mask(token_len, request.num_prompt_tokens)
        except AssertionError as error:
            logger.debug("Skip AscendStore store mask for unaligned request %s: %s", request.request_id, error)
            store_masks = None

        group_store_mask = list(store_masks[0]) if store_masks is not None else None
        if group_store_mask is not None and not any(group_store_mask):
            return []

        def chunk_filter(start: int) -> bool:
            block_index = start // self.block_size
            return group_store_mask is None or (block_index < len(group_store_mask) and group_store_mask[block_index])

        tp_replicas = self.put_step if self.dcp_size <= 1 else 1
        chunks = self.token_database.process_token_key_strings_with_block_ids(
            token_len,
            list(request.block_hashes),
            list(request.block_ids),
            chunk_filter=chunk_filter,
            shard_rank=self.pcp_rank * tp_replicas + self.tp_rank % tp_replicas,
            shard_size=self.pcp_size * tp_replicas,
        )
        return list(chunks)
