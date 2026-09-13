"""Subprocess-only handlers for TP-asymmetric dense KV transfer."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any, Protocol

from vllm.distributed.kv_events import BlockStored
from vllm.logger import logger
from vllm.v1.core.kv_cache_utils import maybe_convert_block_hash

from ..kv_transfer import KVCacheStoreRecvingThread, KVCacheStoreSendingThread, record_failed_blocks
from ..metadata import ChunkedTokenDatabase, ReqMeta


class _TPMismatchBackend(Protocol):
    def put(self, keys: list[str], addrs: list[list[int]], sizes: list[list[int]]) -> Any: ...

    def get(self, keys: list[str], addrs: list[list[int]], sizes: list[list[int]]) -> list[int] | None: ...


class TPMismatchTransfer:
    """Execute TP-mismatch I/O from child-restored KV layout metadata."""

    def __init__(
        self,
        backend: _TPMismatchBackend,
        token_database: ChunkedTokenDatabase,
        tp_rank: int,
        num_sub_keys: int,
        *,
        enable_kv_events: bool = False,
        record_operation: Callable[[str, float, int], None] | None = None,
    ) -> None:
        if len(token_database.block_size) != 1:
            raise ValueError("TP mismatch requires exactly one KV cache group")
        if num_sub_keys <= 0:
            raise ValueError("TP mismatch num_sub_keys must be positive")

        block_size = token_database.block_size[0]
        if block_size <= 0:
            raise ValueError("TP mismatch block_size must be positive")
        addresses = token_database.group_kv_caches_base_addr.get(0, [])
        block_lengths = token_database.group_block_len.get(0, [])
        block_strides = token_database.group_block_stride.get(0, [])
        if not addresses or not (len(addresses) == len(block_lengths) == len(block_strides)):
            raise ValueError("TP mismatch requires a complete registered KV layout")

        per_token_sizes = []
        for block_len in block_lengths:
            if block_len <= 0 or block_len % block_size:
                raise ValueError("TP mismatch KV block lengths must be divisible by block_size")
            per_token_sizes.append(block_len // block_size)
        if len(set(per_token_sizes)) != 1:
            raise ValueError("TP mismatch requires uniform per-token KV entry sizes")
        per_token_bytes = per_token_sizes[0]
        if per_token_bytes % num_sub_keys:
            raise ValueError("TP mismatch per-token KV bytes must be divisible by num_sub_keys")

        self.backend = backend
        self.token_database = token_database
        self.tp_rank = tp_rank
        self.num_sub_keys = num_sub_keys
        self.block_size = block_size
        self.sub_size_bytes = per_token_bytes // num_sub_keys
        self.enable_kv_events = enable_kv_events
        self.record_operation = record_operation

    @staticmethod
    def _replace_key_field(key: str, field: str, value: int) -> str:
        marker = f"@{field}:"
        start = key.find(marker)
        if start < 0:
            return key
        value_start = start + len(marker)
        value_end = key.find("@", value_start)
        if value_end < 0:
            value_end = len(key)
        return f"{key[:value_start]}{value}{key[value_end:]}"

    def _build_strided_addrs(self, block_id: int, token_count: int, sub_idx: int) -> tuple[list[int], list[int]]:
        head_offset_bytes = sub_idx * self.sub_size_bytes
        addrs: list[int] = []
        sizes: list[int] = []
        for base_addr, entry_block_len, entry_block_stride in zip(
            self.token_database.group_kv_caches_base_addr[0],
            self.token_database.group_block_len[0],
            self.token_database.group_block_stride[0],
            strict=True,
        ):
            entry_per_token_bytes = entry_block_len // self.block_size
            block_base = base_addr + block_id * entry_block_stride
            for token_index in range(token_count):
                addrs.append(block_base + token_index * entry_per_token_bytes + head_offset_bytes)
                sizes.append(self.sub_size_bytes)
        return addrs, sizes

    def _build_keys_and_addrs(
        self,
        block_hashes: list,
        block_ids: list[int],
        token_len: int,
        mask_num: int = 0,
    ) -> tuple[list[str], list[list[int]], list[list[int]], list[int]]:
        keys: list[str] = []
        addrs: list[list[int]] = []
        sizes: list[list[int]] = []
        key_block_ids: list[int] = []
        chunks = self.token_database.process_token_key_strings_with_block_ids(
            token_len,
            block_hashes,
            block_ids,
            mask_num=mask_num,
        )
        for start, end, base_key, _block_hash, block_id in chunks:
            token_count = end - start
            for sub_idx in range(self.num_sub_keys):
                effective_rank = self.tp_rank * self.num_sub_keys + sub_idx
                sub_addrs, sub_sizes = self._build_strided_addrs(block_id, token_count, sub_idx)
                keys.append(self._replace_key_field(base_key, "head_or_tp_rank", effective_rank))
                addrs.append(sub_addrs)
                sizes.append(sub_sizes)
                key_block_ids.append(block_id)
        return keys, addrs, sizes, key_block_ids

    def store(self, request: ReqMeta, lookup: Callable[[list[str]], list[bool]]) -> list[BlockStored]:
        keys, addrs, sizes, _ = self._build_keys_and_addrs(
            request.block_hashes,
            request.block_ids_by_group[0],
            request.token_len_chunk,
        )
        if not keys:
            return []
        exists = lookup(keys)
        missing_indices = [index for index, present in enumerate(exists) if not present]
        if not missing_indices:
            return []
        keys = [keys[index] for index in missing_indices]
        addrs = [addrs[index] for index in missing_indices]
        sizes = [sizes[index] for index in missing_indices]
        if request.current_event is not None:
            request.current_event.synchronize()
        logger.debug(
            "KV transfer process TP mismatch put req=%s keys=%d sample_keys=%s",
            request.req_id,
            len(keys),
            keys[:3],
        )
        self.backend.put(keys, addrs, sizes)
        return self._build_store_events(request) if self.enable_kv_events else []

    def load(
        self,
        block_hashes: list,
        block_ids: list[int],
        token_len: int,
        mask_num: int,
    ) -> set[int]:
        keys, addrs, sizes, key_block_ids = self._build_keys_and_addrs(
            block_hashes,
            block_ids,
            token_len,
            mask_num,
        )
        if not keys:
            return set()
        offset = self.tp_rank % len(keys)
        keys = keys[offset:] + keys[:offset]
        addrs = addrs[offset:] + addrs[:offset]
        sizes = sizes[offset:] + sizes[:offset]
        key_block_ids = key_block_ids[offset:] + key_block_ids[:offset]
        logger.debug("KV transfer process TP mismatch get keys=%d sample_keys=%s", len(keys), keys[:3])
        start = time.perf_counter()
        result = self.backend.get(keys, addrs, sizes)
        if self.record_operation is not None:
            self.record_operation("load_get", time.perf_counter() - start, len(keys))
        if result is None:
            result = [1] * len(key_block_ids)
        return record_failed_blocks(key_block_ids, result)

    def _build_store_events(self, request: ReqMeta) -> list[BlockStored]:
        block_size = (
            request.original_block_size[0]
            if isinstance(request.original_block_size, list)
            else request.original_block_size
        )
        events: list[BlockStored] = []
        parent_hash = None
        for index, (start, end, _base_key) in enumerate(
            self.token_database.process_tokens(request.token_len_chunk, request.block_hashes)
        ):
            if index >= len(request.block_hashes):
                break
            block_hash = maybe_convert_block_hash(request.block_hashes[index])
            token_ids = request.token_ids[start:end] if request.token_ids is not None else None
            events.append(
                BlockStored(
                    block_hashes=[block_hash],
                    parent_block_hash=parent_hash,
                    token_ids=token_ids,
                    block_size=block_size,
                    lora_id=None,
                    medium="cpu",
                    lora_name=None,
                )
            )
            parent_hash = block_hash
        return events


class KVCacheStoreSendingTPMismatchHandler(KVCacheStoreSendingThread):
    """Keep TP-mismatch request bookkeeping local to the child sender."""

    def __init__(self, *args: Any, transfer: TPMismatchTransfer, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.transfer = transfer

    def _handle_request(self, request: ReqMeta) -> None:
        req_id = request.req_id
        tracked_request = self.is_stored_request(req_id)
        try:
            if not tracked_request:
                return
            stored_events = self.transfer.store(request, self.lookup)
            if stored_events:
                self.update_kv_event(stored_events)
        except Exception:
            logger.exception("Failed to store KV cache for TP-mismatch request %s", req_id)
        finally:
            remaining = self.dec_stored_request(req_id) if tracked_request else None
            if tracked_request and remaining == 0:
                self.delete_finished_stored_request(req_id)
                self.set_finished_request(req_id)
            if request.event_id is not None:
                with self.completed_events_lock:
                    self.completed_events[request.event_id] = 1
            self.request_queue.task_done()


class KVCacheStoreRecvingTPMismatchHandler(KVCacheStoreRecvingThread):
    """Keep TP-mismatch completion and invalid-block state in the child receiver."""

    def __init__(self, *args: Any, transfer: TPMismatchTransfer, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.transfer = transfer

    def _handle_request(self, request: ReqMeta) -> None:
        try:
            load_spec = request.load_spec
            if load_spec is None:
                logger.error("KV pool async recv request %s has no load spec; skip load.", request.req_id)
                self.set_finished_request(request.req_id)
                return
            mask_num = load_spec.vllm_cached_tokens // self.transfer.block_size * self.transfer.block_size
            failed_blocks = self.transfer.load(
                request.block_hashes,
                request.block_ids_by_group[0],
                load_spec.token_len,
                mask_num,
            )
            with self._invalid_block_ids_lock:
                self._invalid_block_ids.update(failed_blocks)
            self.set_finished_request(request.req_id)
        finally:
            self.request_queue.task_done()


__all__ = [
    "KVCacheStoreRecvingTPMismatchHandler",
    "KVCacheStoreSendingTPMismatchHandler",
    "TPMismatchTransfer",
]
