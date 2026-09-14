from __future__ import annotations

import ctypes
import json
import queue
import threading
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from vllm.distributed.kv_events import BlockStored
from vllm.logger import logger
from vllm.v1.core.kv_cache_utils import maybe_convert_block_hash

from vllm_ascend import envs
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.base import (
    Backend,
    require_aligned_batch_results,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise_transfer import LayerBatchBuilder

# isort: off
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import (
    ChunkedTokenDatabase,
    LayerBatchReqMeta,
    LayerLoadTask,
    LayerSaveTask,
    LayerwisePreparation,
    LayerRangeReqMeta,
    LayerTransferTask,
    ReqMeta,
    SharedBlockData,
    get_block_hashes,
)

# isort: on
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.mooncake_session_tracker import (
    MooncakeSessionTracker,
)

_KVPOOL_RANGE_DEBUG_PREFIX = "[KVPOOL_RANGE_DEBUG]"


def _build_range_debug_payload(
    direction: str,
    layer_id: int,
    sizes: list[list[int]],
    object_offsets: list[list[int]],
    results: list[int],
) -> dict[str, Any]:
    nested_sizes = [[int(size) for size in key_sizes] for key_sizes in sizes]
    return {
        "event": "range",
        "direction": direction,
        "layer_id": int(layer_id),
        "key_count": len(results),
        "requested_bytes": [sum(key_sizes) for key_sizes in nested_sizes],
        "sizes": nested_sizes,
        "object_offsets": [[int(offset) for offset in key_offsets] for key_offsets in object_offsets],
        "results": [int(result) for result in results],
    }


def _emit_range_debug_event(
    direction: str,
    layer_id: int,
    sizes: list[list[int]],
    object_offsets: list[list[int]],
    results: list[int],
) -> None:
    try:
        if not envs.VLLM_ASCEND_KVPOOL_RANGE_DEBUG:
            return
        payload = _build_range_debug_payload(direction, layer_id, sizes, object_offsets, results)
        logger.info("%s %s", _KVPOOL_RANGE_DEBUG_PREFIX, json.dumps(payload, separators=(",", ":")))
    except Exception:
        # Debug instrumentation must never affect the transfer path.
        pass


def _emit_commit_debug_event(layer_id: int, key_count: int, results: list[int]) -> None:
    try:
        if not envs.VLLM_ASCEND_KVPOOL_RANGE_DEBUG:
            return
        payload = {
            "event": "commit",
            "layer_id": int(layer_id),
            "key_count": int(key_count),
            "results": [int(result) for result in results],
        }
        logger.info("%s %s", _KVPOOL_RANGE_DEBUG_PREFIX, json.dumps(payload, separators=(",", ":")))
    except Exception:
        pass


def _circular_shift(lst: list, offset: int) -> list:
    if not lst or offset == 0:
        return lst
    return lst[offset:] + lst[:offset]


@dataclass(frozen=True)
class _LayerRevokeTask:
    keys: tuple[str, ...]


class KVTransferThread(threading.Thread):
    def __init__(
        self,
        m_store: Backend,
        token_database: ChunkedTokenDatabase,
        block_size: int | list[int],
        tp_rank: int,
        tp_size: int = 1,
        dcp_size: int = 1,
        ready_event: threading.Event | None = None,
        name: str = "KVTransferThread",
    ):
        super().__init__(daemon=True, name=name)
        self.m_store = m_store
        self.ready_event = ready_event or threading.Event()
        self.block_size = block_size
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self.dcp_size = dcp_size
        self.token_database = token_database
        self.num_addrs_per_block = len(token_database.group_block_len[0])
        self.done_task_lock = threading.Lock()
        self.request_queue: queue.Queue[Any] = queue.Queue()
        self.stored_requests: defaultdict[str, int] = defaultdict(int)
        self.finished_requests: set[str] = set()
        self.kv_event_lock = threading.Lock()
        self.kv_events: list[BlockStored] = []
        self._fatal_error: BaseException | None = None

    def _get_block_size(self, kv_cache_group_id: int = 0) -> int:
        if isinstance(self.block_size, list):
            if kv_cache_group_id >= len(self.block_size):
                return self.block_size[0]
            return self.block_size[kv_cache_group_id]
        return self.block_size

    def add_request(self, request: Any) -> None:
        self.raise_if_failed()
        self.request_queue.put(request)

    def get_and_clear_finished_requests(
        self,
        req_ids: set[str] | None = None,
    ) -> set[str]:
        """
        Get and clear the requests that have been completed.
        Returns:
            A set of request IDs that have been completed.
        """
        with self.done_task_lock:
            if req_ids is None:
                finished_requests = self.finished_requests.copy()
                self.finished_requests.clear()
            else:
                finished_requests = self.finished_requests & req_ids
                self.finished_requests -= finished_requests
        return finished_requests

    def discard_finished_requests(self, req_ids: set[str]) -> None:
        with self.done_task_lock:
            self.finished_requests -= req_ids

    def raise_if_failed(self) -> None:
        if self._fatal_error is not None:
            raise RuntimeError(f"{self.name} failed during asynchronous transfer") from self._fatal_error

    def set_finished_request(self, req_id):
        with self.done_task_lock:
            self.finished_requests.add(req_id)

    def add_stored_request(self, req_id: str):
        with self.done_task_lock:
            self.stored_requests[req_id] += 1

    def dec_stored_request(self, req_id: str):
        with self.done_task_lock:
            if req_id in self.stored_requests:
                self.stored_requests[req_id] -= 1
                return self.stored_requests[req_id]
            return None

    def try_finish_and_delete_stored_request(self, req_id: str) -> bool:
        with self.done_task_lock:
            if req_id in self.stored_requests and self.stored_requests[req_id] == 0:
                del self.stored_requests[req_id]
                return True
            return False

    @staticmethod
    def _split_transfer_packets(
        gvas: np.ndarray,
        addrs: np.ndarray,
        sizes: np.ndarray,
        max_transfer_bytes: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if max_transfer_bytes <= 0:
            return gvas, addrs, sizes

        split_counts: np.ndarray = (sizes + max_transfer_bytes - 1) // max_transfer_bytes
        total_splits = int(split_counts.sum())
        if total_splits == sizes.shape[0]:
            return gvas, addrs, sizes

        split_indices: np.ndarray = np.arange(int(split_counts.max()), dtype=np.int64)
        split_mask = split_indices[:, None] < split_counts[None, :]
        entry_indices = np.broadcast_to(
            np.arange(sizes.shape[0], dtype=np.int64),
            split_mask.shape,
        )[split_mask]
        transfer_offsets = np.broadcast_to(
            split_indices[:, None] * max_transfer_bytes,
            split_mask.shape,
        )[split_mask]

        split_gvas = gvas[entry_indices] + transfer_offsets
        split_addrs = addrs[entry_indices] + transfer_offsets
        split_sizes = np.minimum(
            max_transfer_bytes,
            sizes[entry_indices] - transfer_offsets,
        )
        return split_gvas, split_addrs, split_sizes

    def _batch_copy_with_limits(
        self,
        gvas: np.ndarray,
        addrs: np.ndarray,
        sizes: np.ndarray,
        direction: int,
        max_transfer_blocks: int,
        max_transfer_bytes: int,
    ) -> int:
        if len(gvas) == 0:
            return 0

        # direction: 0/SMEMB_COPY_L2G = save (write), 1/SMEMB_COPY_G2L = load (read)
        dir_name = "save(L2G)" if direction == 0 else "load(G2L)" if direction == 1 else f"dir{direction}"
        logger.debug(
            "[KVPOOL] batch_copy %s gvas=%d total_bytes=%d",
            dir_name,
            len(gvas),
            int(sizes.sum()) if len(sizes) else 0,
        )

        max_transfer_addrs = 0
        if max_transfer_blocks > 0:
            max_transfer_addrs = max_transfer_blocks * self.num_addrs_per_block
        if max_transfer_addrs <= 0:
            max_transfer_addrs = len(gvas)

        assert self.m_store.store is not None
        for start in range(0, len(gvas), max_transfer_addrs):
            end = start + max_transfer_addrs
            split_gvas, split_addrs, split_sizes = self._split_transfer_packets(
                gvas[start:end],
                addrs[start:end],
                sizes[start:end],
                max_transfer_bytes,
            )
            logger.debug(
                "[KVPOOL] batch_copy %s split_gvas=%s split_sizes=%s",
                dir_name,
                split_gvas.tolist(),
                split_sizes.tolist(),
            )
            res = self.m_store.store.batch_copy(
                split_gvas.tolist(),
                split_addrs.tolist(),
                split_sizes.tolist(),
                direction,
            )
            if res != 0:
                logger.error("[KVPOOL] batch_copy %s FAILED res=%d", dir_name, res)
                return res
        return 0

    @staticmethod
    def _range_transfer_batches(
        keys: list[str],
        all_buffers: list[list[int]],
        all_sizes: list[list[int]],
        all_offsets: list[list[int]],
        max_transfer_blocks: int,
        max_transfer_bytes: int,
    ) -> list[tuple[list[str], list[list[int]], list[list[int]], list[list[int]]]]:
        """Split Mooncake range calls by key count and per-segment byte size."""
        row_count = len(keys)
        if not (len(all_buffers) == len(all_sizes) == len(all_offsets) == row_count):
            raise ValueError("Mooncake range metadata must contain one buffer/size/offset row per key")

        normalized_buffers: list[list[int]] = []
        normalized_sizes: list[list[int]] = []
        normalized_offsets: list[list[int]] = []
        for buffers, sizes, offsets in zip(all_buffers, all_sizes, all_offsets, strict=True):
            if not (len(buffers) == len(sizes) == len(offsets)):
                raise ValueError("Mooncake range rows must align buffers, sizes, and offsets")
            split_buffers: list[int] = []
            split_sizes: list[int] = []
            split_offsets: list[int] = []
            for buffer, size, offset in zip(buffers, sizes, offsets, strict=True):
                if size < 0:
                    raise ValueError(f"Mooncake range size must be non-negative, got {size}")
                if size == 0:
                    continue
                packet_size = max_transfer_bytes if max_transfer_bytes > 0 else size
                for packet_offset in range(0, size, packet_size):
                    split_buffers.append(buffer + packet_offset)
                    split_sizes.append(min(packet_size, size - packet_offset))
                    split_offsets.append(offset + packet_offset)
            normalized_buffers.append(split_buffers)
            normalized_sizes.append(split_sizes)
            normalized_offsets.append(split_offsets)

        rows_per_batch = max_transfer_blocks if max_transfer_blocks > 0 else max(1, row_count)
        return [
            (
                keys[start:end],
                normalized_buffers[start:end],
                normalized_sizes[start:end],
                normalized_offsets[start:end],
            )
            for start in range(0, row_count, rows_per_batch)
            for end in [min(start + rows_per_batch, row_count)]
        ]

    def _set_os_thread_name(self) -> None:
        try:
            libc = ctypes.CDLL("libc.so.6")
            # Linux task comm is limited to 15 visible bytes plus NUL.
            libc.prctl(15, self.name[:15].encode(), 0, 0, 0)
        except Exception:
            pass

    def run(self):
        """Run the thread to handle KV cache transfer requests."""
        self._set_os_thread_name()
        self.m_store.set_device()
        self.ready_event.set()
        while True:
            request_data = None
            try:
                request_data = self.request_queue.get()
                if request_data is None:
                    logger.warning("Received a None request. This indicates queue shutdown or invalid request.")
                    self.request_queue.task_done()
                    continue
                self._handle_request(request_data)
            except Exception as e:
                self._fatal_error = e
                logger.error(
                    "Error in KVCacheTransferThread(%s). type=%s, error=%s. Check thread state and request processing.",
                    self.name,
                    type(e).__name__,
                    e,
                )
                return

    def _handle_request(self, req_meta: Any):
        pass

    def _handle_request_exception(self, request_data: Any):
        """Allow subclasses to complete queue/request bookkeeping on errors."""
        pass

    def lookup(
        self,
        keys: list[str],
    ) -> list[bool]:
        """
        Check the existence of all keys from the cache engine.
        :return: A bool list where True means the key exists in store.
        """
        try:
            res = self.m_store.exists(keys)  # type: ignore[assignment]
            exists_list = [False] * len(keys)
            for index, value in enumerate(res):  # type: ignore[arg-type]
                exists_list[index] = value == 1
            return exists_list
        except Exception as e:
            logger.error(
                "Remote connection failed in lookup. type=%s, error=%s. Check network and remote store.",
                type(e).__name__,
                e,
            )
            return [False] * len(keys)

    def _get_missing_indices(self, keys: list[str], require_exists_check: bool = False) -> list[int]:
        """Filter existing keys unless the backend can do so during put.

        Callers that need the exact newly stored key set, such as KV event
        publishers, can force connector-side filtering.
        """
        if not require_exists_check and not self.m_store.requires_exists_before_put:
            return list(range(len(keys)))
        exists_states = self.lookup(keys)
        return [index for index, exists in enumerate(exists_states) if not exists]

    def update_kv_event(self, event: list[BlockStored]):
        with self.kv_event_lock:
            self.kv_events.extend(event)

    def get_kv_events(self) -> list[BlockStored]:
        with self.kv_event_lock:
            events = self.kv_events.copy()
            self.kv_events.clear()
        return events

    @staticmethod
    def _skip_null_blocks(req_meta: ReqMeta, group_id: int, cache_role: str = "kv") -> bool:
        if cache_role != "kv":
            return False
        skip_flags = req_meta.skip_null_blocks_by_group
        return group_id < len(skip_flags) and skip_flags[group_id] if skip_flags else False

    def _prepare_value(
        self,
        start: int,
        end: int,
        block_ids: list[int],
        kv_cache_group_id: int = 0,
        cache_role: str = "kv",
        block_id: int | None = None,
    ):
        try:
            return self.token_database.prepare_value(
                start,
                end,
                block_ids,
                kv_cache_group_id=kv_cache_group_id,
                cache_role=cache_role,
                block_id=block_id,
            )
        except TypeError:
            return self.token_database.prepare_value(start, end, block_ids)

    def _decode_adaptor_prefill_pp(
        self,
        keys: list[str],
        addrs: list[list[int]],
        sizes: list[list[int]],
        kv_cache_group_id: int = 0,
        cache_role: str = "kv",
    ):
        try:
            return self.token_database.decode_adaptor_prefill_pp(
                keys,
                addrs,
                sizes,
                kv_cache_group_id=kv_cache_group_id,
                cache_role=cache_role,
            )
        except TypeError:
            return self.token_database.decode_adaptor_prefill_pp(keys, addrs, sizes)


class KVCacheStoreSendingThread(KVTransferThread):
    def __init__(
        self,
        m_store: Backend,
        token_database: ChunkedTokenDatabase,
        block_size: int | list[int],
        tp_rank: int,
        tp_size: int = 1,
        dcp_size: int = 1,
        put_step: int = 1,
        kv_role: str = "kv_producer",
        ready_event: threading.Event | None = None,
        group_uses_align_state: list[bool] | None = None,
        enable_kv_event: bool = False,
        worker: Any = None,
    ):
        super().__init__(
            m_store, token_database, block_size, tp_rank, tp_size, dcp_size, ready_event, name="KVCacheSendingThread"
        )
        self.put_step = put_step
        self.kv_role = kv_role
        self.group_uses_align_state = group_uses_align_state or []
        self.enable_kv_event = enable_kv_event
        self.completed_events_lock = threading.Lock()
        self.completed_events: dict[int, int] = {}
        self.worker = worker

    def is_stored_request(self, req_id: str) -> bool:
        with self.done_task_lock:
            return req_id in self.stored_requests

    def get_stored_request_count(self, req_id: str) -> int | None:
        with self.done_task_lock:
            return self.stored_requests.get(req_id)

    def delete_finished_stored_request(self, req_id: str):
        with self.done_task_lock:
            self.stored_requests.pop(req_id, None)

    def get_completed_events(self):
        if not self.completed_events:
            return None
        with self.completed_events_lock:
            completed_events = self.completed_events.copy()
            self.completed_events.clear()
        return completed_events

    def _handle_request_exception(self, request_data: Any):
        req_id = getattr(request_data, "req_id", None)
        if req_id is not None:
            with self.done_task_lock:
                tracked_request = req_id in self.stored_requests
            if tracked_request:
                self.dec_stored_request(req_id)
        self.request_queue.task_done()

    def _handle_request(self, req_meta: ReqMeta):
        if self.worker is not None and getattr(self.worker, "tp_mismatch", False):
            req_id = req_meta.req_id
            try:
                self.worker._store_kv_tp_mismatch(req_meta)
            except Exception:
                logger.exception("Failed to store KV cache for TP-mismatch request %s", req_id)
            finally:
                remaining = self.get_stored_request_count(req_id)
                if remaining == 0:
                    self.delete_finished_stored_request(req_id)
                    self.set_finished_request(req_id)
                if req_meta.event_id is not None:
                    with self.completed_events_lock:
                        self.completed_events[req_meta.event_id] = 1
                self.request_queue.task_done()
            return

        req_id = req_meta.req_id
        tracked_request = False
        try:
            with self.done_task_lock:
                tracked_request = req_id in self.stored_requests
            if not tracked_request:
                return
            self._handle_stored_request(req_meta)
        except Exception:
            logger.exception("Failed to store KV cache for request %s", req_id)
        finally:
            remaining = self.dec_stored_request(req_id) if tracked_request else None
            if tracked_request and remaining == 0:
                self.delete_finished_stored_request(req_id)
                self.set_finished_request(req_id)
            if req_meta.event_id is not None:
                with self.completed_events_lock:
                    self.completed_events[req_meta.event_id] = 1
            self.request_queue.task_done()

    def _handle_stored_request(self, req_meta: ReqMeta):
        """Store missing KV chunks for one request."""
        token_len = req_meta.token_len_chunk
        req_id = req_meta.req_id
        current_event = req_meta.current_event
        try:
            store_masks = self.token_database.store_mask(token_len, req_meta.num_prompt_tokens)
        except AssertionError as exc:
            logger.debug("Skip AscendStore store mask for unaligned request %s: %s", req_id, exc)
            store_masks = None
        load_spec = req_meta.load_spec
        skip_start = load_spec.vllm_cached_tokens if load_spec is not None else 0
        skip_end = (
            (
                load_spec.kvpool_store_skip_tokens
                if load_spec.kvpool_store_skip_tokens is not None
                else load_spec.kvpool_cached_tokens
            )
            if load_spec is not None
            else 0
        )

        def should_skip(start: int, end: int) -> bool:
            return skip_end > skip_start and start >= skip_start and end <= skip_end

        for group_id in req_meta.kv_cache_group_ids or [0]:
            group_block_size = self._get_block_size(group_id)

            group_store_mask = (
                list(store_masks[group_id]) if store_masks is not None and group_id < len(store_masks) else None
            )
            if group_store_mask is not None:
                skipped_chunks = 0
                for chunk_id, allowed in enumerate(group_store_mask):
                    start = chunk_id * group_block_size
                    if allowed and should_skip(start, start + group_block_size):
                        group_store_mask[chunk_id] = False
                        skipped_chunks += 1
                if skipped_chunks:
                    logger.debug(
                        "KV pool put skipped %d pooled chunks for request %s group %d",
                        skipped_chunks,
                        req_id,
                        group_id,
                    )
                if not group_store_mask or not any(group_store_mask):
                    continue

            starts: list[int] = []
            ends: list[int] = []
            keys: list[str] = []
            block_hashes = []
            key_block_ids: list[int] = []
            block_ids = req_meta.block_ids_by_group[group_id]
            skip_null_blocks = self._skip_null_blocks(req_meta, group_id)
            align_state_group = group_id < len(self.group_uses_align_state) and self.group_uses_align_state[group_id]

            def chunk_filter(
                start: int,
                group_block_size=group_block_size,
                group_store_mask=group_store_mask,
            ) -> bool:
                block_idx = start // group_block_size
                mask_allows = group_store_mask is None or (
                    block_idx < len(group_store_mask) and group_store_mask[block_idx]
                )
                chunk_start = block_idx * group_block_size
                return mask_allows and not should_skip(chunk_start, chunk_start + group_block_size)

            pre_shard = self.dcp_size <= 1 and not align_state_group
            iterator = self.token_database.process_token_key_strings_with_block_ids(
                token_len,
                req_meta.block_hashes,
                block_ids,
                kv_cache_group_id=group_id,
                skip_null_blocks=skip_null_blocks,
                chunk_filter=chunk_filter,
                shard_rank=self.tp_rank % self.put_step if pre_shard else None,
                shard_size=self.put_step if pre_shard else None,
            )
            for start, end, key, block_hash, block_id in iterator:
                starts.append(start)
                ends.append(end)
                keys.append(key)
                if self.enable_kv_event:
                    block_hashes.append(block_hash)
                key_block_ids.append(block_id)

            if not keys:
                continue
            missing_indices = self._get_missing_indices(keys, require_exists_check=self.enable_kv_event)
            if not missing_indices:
                continue
            starts = [starts[index] for index in missing_indices]
            ends = [ends[index] for index in missing_indices]
            keys = [keys[index] for index in missing_indices]
            if self.enable_kv_event:
                block_hashes = [block_hashes[index] for index in missing_indices]
            key_block_ids = [key_block_ids[index] for index in missing_indices]

            logger.debug(
                "Storing KV cache for %d out of %d blocks for request %s in group %d",
                len(keys),
                token_len // group_block_size,
                req_id,
                group_id,
            )
            addrs = []
            sizes = []
            stored_events: list[BlockStored] = []
            all_hashes = []
            if self.enable_kv_event:
                group_block_hashes = get_block_hashes(
                    req_meta.block_hashes,
                    group_block_size,
                    getattr(self.token_database, "hash_block_size", group_block_size),
                )
                all_hashes = [maybe_convert_block_hash(bh) for bh in group_block_hashes]
            logger.debug(
                "KV pool put request=%s group=%d token_len=%d keys=%d sample_keys=%s",
                req_id,
                group_id,
                token_len,
                len(keys),
                keys[:3],
            )
            for index, start in enumerate(starts):
                addr, size, _ = self._prepare_value(
                    start,
                    ends[index],
                    block_ids,
                    kv_cache_group_id=group_id,
                    block_id=key_block_ids[index],
                )
                addrs.append(addr)
                sizes.append(size)
                if self.enable_kv_event:
                    token_ids = req_meta.token_ids[start : ends[index]] if req_meta.token_ids is not None else None
                    block_size = (
                        req_meta.original_block_size[group_id]
                        if isinstance(req_meta.original_block_size, list)
                        else req_meta.original_block_size
                    )
                    if block_size is not None:
                        block_idx = start // group_block_size
                        if block_idx >= len(all_hashes):
                            continue
                        current_hash = all_hashes[block_idx]
                        parent_hash = all_hashes[block_idx - 1] if block_idx > 0 else None
                        stored_event = BlockStored(
                            block_hashes=[current_hash],
                            parent_block_hash=parent_hash,
                            token_ids=token_ids,
                            block_size=block_size,
                            lora_id=None,
                            medium="cpu",
                            lora_name=None,
                        )
                        stored_events.append(stored_event)
                        logger.debug("Added kv cache event '%s' to kv cache events queue", stored_event)

            if self.kv_role == "kv_consumer":
                keys, addrs, sizes = self._decode_adaptor_prefill_pp(
                    keys,
                    addrs,
                    sizes,
                    kv_cache_group_id=group_id,
                )
            if current_event is not None:
                current_event.synchronize()
            self.m_store.put(keys, addrs, sizes)
            if self.enable_kv_event and stored_events:
                self.update_kv_event(stored_events)


class KVCacheStoreRecvingThread(KVTransferThread):
    def __init__(
        self,
        m_store: Backend,
        token_database: ChunkedTokenDatabase,
        block_size: int | list[int],
        tp_rank: int,
        tp_size: int = 1,
        dcp_size: int = 1,
        ready_event: threading.Event | None = None,
        invalid_block_ids: set[int] | None = None,
        invalid_block_ids_lock: threading.Lock | None = None,
        worker: Any = None,
        record_operation: Callable[[str, float, int], None] | None = None,
    ):
        super().__init__(
            m_store,
            token_database,
            block_size,
            tp_rank,
            tp_size,
            dcp_size,
            ready_event,
            name="KVCacheStoreRecvingThread",
        )
        self._invalid_block_ids = invalid_block_ids if invalid_block_ids is not None else set()
        self._invalid_block_ids_lock = invalid_block_ids_lock or threading.Lock()
        self.worker = worker
        self._record_operation_cb = record_operation

    def _handle_request(self, req_meta: ReqMeta):
        try:
            load_spec = req_meta.load_spec
            req_id = req_meta.req_id
            if load_spec is None:
                logger.error("KV pool async recv request %s has no load spec; skip load.", req_id)
                self.set_finished_request(req_id)
                return

            token_len = load_spec.token_len
            if self.worker is not None and getattr(self.worker, "tp_mismatch", False):
                group_block_size = self._get_block_size(0)
                mask_num = load_spec.vllm_cached_tokens // group_block_size * group_block_size
                self.worker._load_kv_tp_mismatch(
                    req_meta.block_hashes,
                    req_meta.block_ids_by_group[0],
                    token_len,
                    mask_num,
                )
                self.set_finished_request(req_id)
                return

            addr_list = []
            size_list = []
            key_list = []
            block_id_list: list[int] = []
            group_ids = req_meta.kv_cache_group_ids or [0]
            load_masks = self.token_database.load_mask(req_meta.block_hashes, token_len)
            for group_id in group_ids:
                block_ids = req_meta.block_ids_by_group[group_id]
                group_block_size = self._get_block_size(group_id)
                mask_num = load_spec.vllm_cached_tokens // group_block_size * group_block_size

                def chunk_filter(start: int, group_id=group_id) -> bool:
                    return self.token_database.mask_allows_chunk(load_masks, group_id, start)

                token_iter = self.token_database.process_token_key_strings_with_block_ids(
                    token_len,
                    req_meta.block_hashes,
                    block_ids,
                    mask_num,
                    kv_cache_group_id=group_id,
                    skip_null_blocks=self._skip_null_blocks(req_meta, group_id),
                    chunk_filter=chunk_filter,
                )
                for start, end, key, _block_hash, block_id in token_iter:
                    addr, size, block_id = self._prepare_value(
                        start,
                        end,
                        block_ids,
                        kv_cache_group_id=group_id,
                        block_id=block_id,
                    )
                    key_list.append(key)
                    addr_list.append(addr)
                    size_list.append(size)
                    block_id_list.append(block_id)
            if not key_list:
                self.set_finished_request(req_id)
                return
            key_list_c = key_list[self.tp_rank % len(key_list) :] + key_list[: self.tp_rank % len(key_list)]
            addr_list_c = addr_list[self.tp_rank % len(addr_list) :] + addr_list[: self.tp_rank % len(addr_list)]
            size_list_c = size_list[self.tp_rank % len(size_list) :] + size_list[: self.tp_rank % len(size_list)]
            block_id_list_c = (
                block_id_list[self.tp_rank % len(block_id_list) :] + block_id_list[: self.tp_rank % len(block_id_list)]
            )
            logger.debug(
                "KV pool async recv calls backend get request=%s token_len=%d groups=%s keys=%d sample_keys=%s",
                req_id,
                token_len,
                req_meta.kv_cache_group_ids or [0],
                len(key_list_c),
                key_list_c[:3],
            )
            load_get_start = time.perf_counter() if self._record_operation_cb is not None else 0.0
            ret = self.m_store.get(key_list_c, addr_list_c, size_list_c)
            if self._record_operation_cb is not None:
                self._record_operation_cb(
                    "load_get",
                    time.perf_counter() - load_get_start,
                    len(key_list_c),
                )
            if ret is not None and any(r != 0 for r in ret):
                missing_block_ids = record_failed_blocks(
                    block_id_list_c,
                    ret,
                )
                if len(req_meta.block_ids_by_group) == 1:
                    with self._invalid_block_ids_lock:
                        self._invalid_block_ids.update(missing_block_ids)
                elif missing_block_ids:
                    logger.error(
                        "KV load failed for hybrid request %s. "
                        "Skip invalid-block fallback to avoid scheduler crash. "
                        "failed_blocks=%s",
                        req_id,
                        missing_block_ids,
                    )
            elif ret is None:
                missing_block_ids = record_failed_blocks(
                    block_id_list_c,
                    [1] * len(block_id_list_c),
                )
                if len(req_meta.block_ids_by_group) == 1:
                    with self._invalid_block_ids_lock:
                        self._invalid_block_ids.update(missing_block_ids)
                elif missing_block_ids:
                    logger.error(
                        "KV load failed for hybrid request %s. "
                        "Skip invalid-block fallback to avoid scheduler crash. "
                        "failed_blocks=%s",
                        req_id,
                        missing_block_ids,
                    )
            logger.debug(
                "KV pool async recv backend get returned request=%s token_len=%d groups=%s keys=%d",
                req_id,
                token_len,
                req_meta.kv_cache_group_ids or [0],
                len(key_list_c),
            )
            self.set_finished_request(req_id)
        finally:
            self.request_queue.task_done()


class KVCacheStoreKeyLayerSendingThread(KVTransferThread):
    def __init__(
        self,
        m_store: Backend,
        token_database: ChunkedTokenDatabase,
        block_size: int,
        tp_rank: int,
        tp_size: int,
        dcp_size: int,
        put_step: int,
        ready_event: threading.Event,
        num_layers: int,
        layer_save_finished_events: list[threading.Event],
        sync_save_events: list[torch.npu.Event],
    ):
        super().__init__(
            m_store,
            token_database,
            block_size,
            tp_rank,
            tp_size,
            dcp_size,
            ready_event,
            name="KVCacheStoreKeyLayerSendingThread",
        )
        self.final_layer_id = num_layers - 1
        self.put_step = put_step
        self.layer_save_finished_events = layer_save_finished_events
        self.sync_save_events = sync_save_events

    def build_cached_process_tokens(self, task: LayerTransferTask) -> dict[int, list[tuple[int, int, list]]] | None:
        """Pre-compute process_tokens results for all layers (Key path).

        Returns a dict mapping block_range index to a list of
        (start, end, key_all_layers) tuples, where key_all_layers is the
        result of key.split_layers().
        """
        if not task.block_ranges:
            return None

        group_block_size = self._get_block_size(0)
        cache: dict[int, list[tuple[int, int, list]]] = {}

        for br_idx, block_range in enumerate(task.block_ranges):
            request = block_range.request
            mask_num = request.save_start_token // group_block_size * group_block_size
            entries = []
            for start, end, key in self.token_database.process_tokens(
                request.save_end_token,
                request.block_hashes,
                mask_num,
            ):
                block_index = start // group_block_size
                if block_index < block_range.start_block or block_index >= block_range.end_block:
                    continue
                key_all = key.split_layers(self.final_layer_id + 1)
                entries.append((start, end, key_all))
            cache[br_idx] = entries

        return cache

    def _handle_request(  # type: ignore[override]
        self, transfer_tasks: list[LayerTransferTask]
    ):
        if len(transfer_tasks) == 0:
            self.request_queue.task_done()
            return
        if len(transfer_tasks) > 1:
            raise ValueError(f"Expected at most one layer transfer task, got {len(transfer_tasks)}")

        transfer_task = transfer_tasks[0]
        layer_id = transfer_task.layer_id
        key_list = []
        addr_list = []
        size_list = []
        req_ids = []
        is_last_chunks = []

        # Reuse pre-computed process_tokens results if available
        cached_tokens = transfer_task.cached_process_tokens

        for br_idx, block_range in enumerate(transfer_task.block_ranges):
            request = block_range.request
            req_ids.append(request.req_id)
            is_last_chunks.append(request.is_last_chunk)
            starts = []
            ends = []
            keys = []
            group_block_size = self._get_block_size(0)

            if cached_tokens is not None:
                # Fast path: reuse cached (start, end, key_all) tuples
                for start, end, key_all in cached_tokens[br_idx]:
                    block_index = start // group_block_size
                    if block_index < block_range.start_block or block_index >= block_range.end_block:
                        continue
                    starts.append(start)
                    ends.append(end)
                    keys.append(key_all[layer_id])
            else:
                mask_num = request.save_start_token // group_block_size * group_block_size
                for start, end, key in self.token_database.process_tokens(
                    request.save_end_token,
                    request.block_hashes,
                    mask_num,
                ):
                    block_index = start // group_block_size
                    if block_index < block_range.start_block or block_index >= block_range.end_block:
                        continue
                    starts.append(start)
                    ends.append(end)
                    keys.append(key.split_layers(self.final_layer_id + 1)[layer_id])

            if not self.dcp_size > 1:
                starts = starts[self.tp_rank % self.put_step :: self.put_step]
                ends = ends[self.tp_rank % self.put_step :: self.put_step]
                keys = keys[self.tp_rank % self.put_step :: self.put_step]

            for index, key in enumerate(keys):
                key_list.append(key.to_string())
                addr, size, _ = self.token_database.prepare_value_layer(
                    starts[index],
                    ends[index],
                    request.block_ids,
                    layer_id,
                )
                addr_list.append(addr)
                size_list.append(size)

        for req_id in req_ids:
            self.dec_stored_request(req_id)

        if key_list:
            missing_indices = self._get_missing_indices(key_list)
            keys_to_put = [key_list[index] for index in missing_indices]
            addrs_to_put = [addr_list[index] for index in missing_indices]
            sizes_to_put = [size_list[index] for index in missing_indices]
            if keys_to_put:
                self.sync_save_events[layer_id].synchronize()
                self.m_store.put(keys_to_put, addrs_to_put, sizes_to_put)

        if layer_id == self.final_layer_id:
            for req_id, is_last_chunk in zip(req_ids, is_last_chunks):
                if is_last_chunk and self.try_finish_and_delete_stored_request(req_id):
                    self.set_finished_request(req_id)

        assert not self.layer_save_finished_events[layer_id].is_set(), f"thread: {layer_id} save failed "
        logger.debug("Key-based layer save event set: layer %d", layer_id)
        self.layer_save_finished_events[layer_id].set()
        transfer_tasks.clear()
        self.request_queue.task_done()


class KVCacheStoreKeyLayerRecvingThread(KVTransferThread):
    def __init__(
        self,
        m_store: Backend,
        token_database: ChunkedTokenDatabase,
        block_size: int,
        tp_rank: int,
        tp_size: int,
        dcp_size: int,
        ready_event: threading.Event,
        get_event: threading.Event,
        layer_load_finished_events: list[threading.Event],
        layer_save_finished_events: list[threading.Event],
        num_layers: int,
    ):
        super().__init__(
            m_store,
            token_database,
            block_size,
            tp_rank,
            tp_size,
            dcp_size,
            ready_event,
            name="KVCacheStoreKeyLayerRecvingThread",
        )
        self.get_event = get_event
        self.layer_load_finished_events = layer_load_finished_events
        self.layer_save_finished_events = layer_save_finished_events
        self.final_layer_id = num_layers - 1

    def _wait_for_save(self, layer_id: int) -> None:
        while not self.layer_save_finished_events[layer_id].wait(timeout=10):
            logger.info("Layerwise %d save wait timed out, keep waiting before load", layer_id)
        logger.debug("Key-based layer save event cleared: layer %d", layer_id)
        self.layer_save_finished_events[layer_id].clear()

    def _handle_request(  # type: ignore[override]
        self, data: LayerLoadTask
    ):
        wait_for_save = data.wait_for_save_layer
        layer_id = data.layer_id
        if wait_for_save is not None:
            self._wait_for_save(wait_for_save)

        if data.attention_start_gate is not None:
            while not data.attention_start_gate.wait(timeout=10):
                logger.info("Layerwise %d load waits for attention compute start", layer_id)

        key_list = []
        addr_list = []
        size_list = []
        req_ids = []
        is_last_chunks = []
        if len(data.transfer_tasks) > 1:
            raise ValueError(f"Expected at most one layer transfer task, got {len(data.transfer_tasks)}")
        if data.transfer_tasks:
            transfer_task = data.transfer_tasks[0]
            for block_range in transfer_task.block_ranges:
                request = block_range.request
                req_ids.append(request.req_id)
                is_last_chunks.append(request.is_last_chunk)
                for block_index in range(block_range.start_block, block_range.end_block):
                    if block_index >= len(request.block_hashes):
                        continue
                    block_hash = request.block_hashes[block_index]
                    chunk_hash = block_hash if isinstance(block_hash, str) else block_hash.hex()
                    key = self.token_database._make_key_by_hash(
                        chunk_hash,
                    ).split_layers(self.final_layer_id + 1)[layer_id]
                    group_block_size = self._get_block_size(0)
                    start = block_index * group_block_size
                    end = start + group_block_size
                    addr, size, _ = self.token_database.prepare_value_layer(
                        start,
                        end,
                        request.block_ids,
                        layer_id,
                    )
                    key_list.append(key.to_string())
                    addr_list.append(addr)
                    size_list.append(size)

        if key_list:
            shift = (self.tp_rank * len(key_list)) // self.tp_size
            key_list_c = _circular_shift(key_list, shift)
            addr_list_c = _circular_shift(addr_list, shift)
            size_list_c = _circular_shift(size_list, shift)
            self.m_store.get(key_list_c, addr_list_c, size_list_c)

        if layer_id == self.final_layer_id:
            for req_id, is_last_chunk in zip(req_ids, is_last_chunks):
                if is_last_chunk:
                    self.set_finished_request(req_id)

        assert not self.layer_load_finished_events[layer_id].is_set(), f"thread: {layer_id} load failed "
        logger.debug("Key-based layer load event set: layer %d", layer_id)
        self.layer_load_finished_events[layer_id].set()
        data.transfer_tasks.clear()
        self.request_queue.task_done()
        self.get_event.set()


class KVCacheStoreLayerSendingThread(KVTransferThread):
    def __init__(
        self,
        m_store: Backend,
        token_database: ChunkedTokenDatabase,
        block_size: int | list[int],
        tp_rank: int,
        tp_size: int,
        dcp_size: int,
        page_size_bytes: int,
        ready_event: threading.Event,
        num_layers: int,
        layer_save_finished_events: list[threading.Event],
        sync_save_events: list[torch.npu.Event],
        max_transfer_blocks: int = 0,
        max_transfer_bytes: int = 0,
        group_builders: list[LayerBatchBuilder] | None = None,
        put_started_keys: set[str] | None = None,
        put_started_keys_lock: threading.Lock | None = None,
        session_tracker: MooncakeSessionTracker | None = None,
        sync_attn_events: list[torch.npu.Event] | None = None,
        layer_attn_recorded_events: list[threading.Event] | None = None,
    ):
        super().__init__(
            m_store,
            token_database,
            block_size,
            tp_rank,
            tp_size,
            dcp_size,
            ready_event,
            name="KVCacheStoreLayerSendingThread",
        )
        self.final_layer_id = num_layers - 1
        self.layer_save_finished_events = layer_save_finished_events
        self.sync_save_events = sync_save_events
        self.sync_attn_events = sync_attn_events
        self.layer_attn_recorded_events = layer_attn_recorded_events
        self.max_transfer_blocks = max_transfer_blocks
        self.max_transfer_bytes = max_transfer_bytes
        self.write_results: dict[str, int] = {}
        self._put_started_keys = put_started_keys if put_started_keys is not None else set()
        self._put_started_keys_lock = put_started_keys_lock or threading.Lock()
        self._session_tracker = session_tracker
        self._active_put_keys: set[str] | None = None
        self.group_builders: list[LayerBatchBuilder] | None = group_builders
        if group_builders is not None:
            self.layer_batch_builder = group_builders[0]
        else:
            self.layer_batch_builder = LayerBatchBuilder(
                token_database,
                page_size_bytes,
                num_layers,
                group_id=0,
            )

    def delete_finished_stored_request(self, req_id: str):
        with self.done_task_lock:
            if req_id in self.stored_requests:
                del self.stored_requests[req_id]

    def build_shared_data(self, task: LayerTransferTask) -> SharedBlockData | None:
        """Pre-compute block rows shared by every layer in the task."""
        if self.group_builders is not None:
            builder = self.group_builders[task.group_id]
        else:
            builder = self.layer_batch_builder
        return builder.build_shared(task, is_save=True)

    def add_revoke_request(self, keys: list[str]) -> None:
        deduplicated_keys = tuple(dict.fromkeys(keys))
        if deduplicated_keys:
            self.request_queue.put(_LayerRevokeTask(deduplicated_keys))

    def _remove_started_keys(self, keys: list[str]) -> None:
        with self._put_started_keys_lock:
            self._put_started_keys.difference_update(keys)

    def _revoke_range_keys(self, keys: list[str]) -> None:
        keys = list(dict.fromkeys(keys))
        if not keys:
            return
        try:
            results = require_aligned_batch_results("batch_revoke", keys, self.m_store.batch_revoke(keys))
            if any(result != 0 for result in results):
                logger.error("Mooncake layerwise revoke failed keys=%s results=%s", keys, results)
        except Exception as exc:
            logger.error("Mooncake layerwise revoke raised keys=%s error=%s", keys, exc)
        finally:
            self._remove_started_keys(keys)
            if self._session_tracker is not None:
                self._session_tracker.revoke_put_keys(keys)

    def _handle_range_request(self, req_meta: LayerRangeReqMeta) -> None:
        layer_id = req_meta.layer_id
        if self._active_put_keys is None or layer_id == 0:
            self._active_put_keys = set(req_meta.keys)
        assert self._active_put_keys is not None
        active_indices = [index for index, key in enumerate(req_meta.keys) if key in self._active_put_keys]
        active_keys = [req_meta.keys[index] for index in active_indices]
        if active_keys:
            self.sync_save_events[layer_id].synchronize()
            active_buffers = [req_meta.all_buffers[index] for index in active_indices]
            active_sizes = [req_meta.all_sizes[index] for index in active_indices]
            active_offsets = [req_meta.all_offsets[index] for index in active_indices]
            results: list[int] = []
            for keys, buffers, sizes, offsets in self._range_transfer_batches(
                active_keys,
                active_buffers,
                active_sizes,
                active_offsets,
                self.max_transfer_blocks,
                self.max_transfer_bytes,
            ):
                results.extend(
                    require_aligned_batch_results(
                        "batch_copy_put",
                        keys,
                        self.m_store.batch_copy_put(keys, buffers, sizes, offsets),
                    )
                )
            _emit_range_debug_event("save", layer_id, active_sizes, active_offsets, results)
            failed_keys = [key for key, result in zip(active_keys, results, strict=True) if result < 0]
            if failed_keys:
                self._revoke_range_keys(failed_keys)
                self._active_put_keys.difference_update(failed_keys)

        if layer_id != self.final_layer_id:
            return
        active_keys = [key for key in req_meta.keys if key in self._active_put_keys]
        if active_keys:
            try:
                commit_results = require_aligned_batch_results(
                    "batch_commit", active_keys, self.m_store.batch_commit(active_keys)
                )
                _emit_commit_debug_event(layer_id, len(active_keys), commit_results)
            except Exception:
                self._revoke_range_keys(active_keys)
                raise
            failed_keys = [key for key, result in zip(active_keys, commit_results, strict=True) if result != 0]
            if failed_keys:
                self._revoke_range_keys(failed_keys)
            committed_keys = [key for key, result in zip(active_keys, commit_results, strict=True) if result == 0]
            if self._session_tracker is not None:
                self._session_tracker.commit_put_keys(committed_keys)
            self._remove_started_keys(active_keys)
        self._active_put_keys = None

    def _handle_range_layer_tasks(self, transfer_tasks: list[LayerTransferTask]) -> None:
        layer_id = transfer_tasks[0].layer_id if transfer_tasks else 0
        shared: SharedBlockData | None = None
        try:
            if len(transfer_tasks) != 1:
                raise ValueError(f"Expected one Mooncake range task, got {len(transfer_tasks)}")
            task = transfer_tasks[0]
            shared = task.shared_block_data
            if shared is None:
                raise RuntimeError("Mooncake range save requires shared block metadata")
            builder = self.group_builders[task.group_id] if self.group_builders else self.layer_batch_builder
            req_meta = builder.build_addrs(shared, task.layer_id)
            if not isinstance(req_meta, LayerRangeReqMeta):
                raise TypeError(f"Expected Mooncake range metadata, got {type(req_meta).__name__}")
            self._handle_range_request(req_meta)
            for req_id in req_meta.req_ids:
                self.dec_stored_request(req_id)
                if self.try_finish_and_delete_stored_request(req_id):
                    self.set_finished_request(req_id)
        except Exception as exc:
            self._fatal_error = exc
            if self._active_put_keys is not None:
                keys_to_revoke = sorted(self._active_put_keys)
            elif shared is not None and shared.block_keys is not None:
                keys_to_revoke = list(dict.fromkeys(shared.block_keys))
            else:
                keys_to_revoke = []
            self._revoke_range_keys(keys_to_revoke)
            self._active_put_keys = set()
            raise
        finally:
            if not self.layer_save_finished_events[layer_id].is_set():
                self.layer_save_finished_events[layer_id].set()
            transfer_tasks.clear()
            self.request_queue.task_done()

    def _wait_attention_done(self, physical_layer: int) -> None:
        # slot_free also requires the compute stream to be past this layer's
        # attention. The threading flag guards against the npu event being a
        # no-op when synchronize() runs before record().
        if self.layer_attn_recorded_events is None or self.sync_attn_events is None:
            return
        while not self.layer_attn_recorded_events[physical_layer].wait(timeout=10):
            logger.info("Layerwise %d attention not recorded, keep waiting before slot_free", physical_layer)
        self.sync_attn_events[physical_layer].synchronize()

    def _handle_request(  # type: ignore[override]
        self, request: list[LayerTransferTask] | _LayerRevokeTask | LayerSaveTask | LayerwisePreparation
    ):
        if isinstance(request, LayerwisePreparation):
            request.ensure_ready()
            self.request_queue.task_done()
            return
        if isinstance(request, _LayerRevokeTask):
            try:
                self._revoke_range_keys(list(request.keys))
            finally:
                self.request_queue.task_done()
            return
        if isinstance(request, LayerSaveTask):
            physical_layer = request.layer_id
            transfer_tasks = request.transfer_tasks
        else:
            transfer_tasks = request
            physical_layer = transfer_tasks[0].layer_id if transfer_tasks else 0
        # Layerwise threads only run when the worker-side
        # protocol gate is on; every store call below sits on the Backend ABC.
        if len(transfer_tasks) == 0:
            if isinstance(request, LayerSaveTask):
                self._wait_attention_done(physical_layer)
                self.layer_save_finished_events[physical_layer].set()
            self.request_queue.task_done()
            return
        if transfer_tasks[0].use_key_major_ranges:
            self._handle_range_layer_tasks(transfer_tasks)
            return
        physical_layer = transfer_tasks[0].layer_id
        preparation = transfer_tasks[0].preparation
        if preparation is not None:
            preparation.ensure_ready()
        has_any_save = False
        all_gvas = []
        all_addrs = []
        all_sizes = []
        all_req_ids = []
        all_save_keys: list[str] = []
        write_finish_keys: list[str] = []
        for task in transfer_tasks:
            shared = task.shared_block_data
            if shared is None:
                continue
            has_any_save = True
            builder = self.group_builders[task.group_id] if self.group_builders else self.layer_batch_builder
            req_meta = builder.build_addrs(shared, task.layer_idx_in_group)
            if not isinstance(req_meta, LayerBatchReqMeta):
                raise TypeError(f"Expected GVA layer metadata, got {type(req_meta).__name__}")
            for req_id in req_meta.req_ids:
                all_req_ids.append(req_id)
            all_save_keys.extend(shared.save_keys)
            write_finish_keys.extend(task.write_finish_keys)
            all_gvas.append(req_meta.gvas_array)
            all_addrs.append(req_meta.addr_array)
            all_sizes.append(req_meta.size_array)
        if has_any_save:
            self.sync_save_events[physical_layer].synchronize()
            gvas_array = np.concatenate(all_gvas) if len(all_gvas) > 1 else all_gvas[0]
            addr_array = np.concatenate(all_addrs) if len(all_addrs) > 1 else all_addrs[0]
            size_array = np.concatenate(all_sizes) if len(all_sizes) > 1 else all_sizes[0]
            res = self._batch_copy_with_limits(
                gvas_array,
                addr_array,
                size_array,
                0,
                self.max_transfer_blocks,
                self.max_transfer_bytes,
            )
            if res != 0:
                raise RuntimeError(f"Layerwise {physical_layer} save batch_copy failed with return code {res}")
            if all_save_keys:
                save_keys = list(dict.fromkeys(all_save_keys))
                for key in save_keys:
                    self.write_results[key] = self.write_results.get(key, 0) or res
            if write_finish_keys:
                finish_keys = list(dict.fromkeys(write_finish_keys))
                results = [self.write_results.pop(key) for key in finish_keys]
                finish_results = self.m_store.batch_write_finish(finish_keys, results)
                if len(finish_results) != len(finish_keys) or any(result != 0 for result in finish_results):
                    raise RuntimeError(
                        f"Layerwise save batch_write_finish failed: "
                        f"expected={len(finish_keys)}, results={finish_results}"
                    )
        self._wait_attention_done(physical_layer)
        finished_req_ids = set().union(*(task.finished_req_ids for task in transfer_tasks))
        for req_id in all_req_ids:
            self.dec_stored_request(req_id)
        last_chunk_req_ids = {
            block_range.request.req_id
            for task in transfer_tasks
            for block_range in task.block_ranges
            if block_range.request.is_last_chunk
        }
        for req_id in finished_req_ids & last_chunk_req_ids if preparation is not None else all_req_ids:
            if self.try_finish_and_delete_stored_request(req_id):
                self.set_finished_request(req_id)
        if not has_any_save:
            assert not self.layer_save_finished_events[physical_layer].is_set(), (
                f"thread: {physical_layer} save failed "
            )
            logger.debug("Layer save event set: layer %d", physical_layer)
            self.layer_save_finished_events[physical_layer].set()
            transfer_tasks.clear()
            self.request_queue.task_done()
            return
        assert not self.layer_save_finished_events[physical_layer].is_set(), f"thread: {physical_layer} save failed "
        logger.debug("Layer save event set: layer %d", physical_layer)
        self.layer_save_finished_events[physical_layer].set()
        transfer_tasks.clear()
        self.request_queue.task_done()


class KVCacheStoreLayerRecvingThread(KVTransferThread):
    def __init__(
        self,
        m_store: Backend,
        token_database: ChunkedTokenDatabase,
        block_size: int | list[int],
        tp_rank: int,
        tp_size: int,
        dcp_size: int,
        page_size_bytes: int,
        ready_event: threading.Event,
        get_event: threading.Event,
        layer_load_finished_events: list[threading.Event],
        layer_save_finished_events: list[threading.Event],
        sync_save_events: list[torch.npu.Event],
        num_layers: int,
        h2d_stagger_us: int = 0,
        max_transfer_blocks: int = 0,
        max_transfer_bytes: int = 0,
        group_builders: list[LayerBatchBuilder] | None = None,
        external_slot_release_waiter: Callable[[int], None] | None = None,
        save_failure_checker: Callable[[], None] | None = None,
        invalid_block_ids: set[int] | None = None,
        invalid_block_ids_lock: threading.Lock | None = None,
        load_abort_event: threading.Event | None = None,
        release_load_leases: Callable[[set[str]], None] | None = None,
    ):
        super().__init__(
            m_store,
            token_database,
            block_size,
            tp_rank,
            tp_size,
            dcp_size,
            ready_event,
            name="KVCacheStoreLayerRecvingThread",
        )
        self.get_event = get_event
        self.layer_load_finished_events = layer_load_finished_events
        self.layer_save_finished_events = layer_save_finished_events
        self.sync_save_events = sync_save_events
        self.final_layer_id = num_layers - 1
        self.release_load_leases = release_load_leases
        self.h2d_stagger_us = h2d_stagger_us
        self.max_transfer_blocks = max_transfer_blocks
        self.max_transfer_bytes = max_transfer_bytes
        self.external_slot_release_waiter = external_slot_release_waiter
        self.save_failure_checker = save_failure_checker
        self._invalid_block_ids = invalid_block_ids if invalid_block_ids is not None else set()
        self._invalid_block_ids_lock = invalid_block_ids_lock or threading.Lock()
        self._load_abort_event = load_abort_event or threading.Event()
        self._active_load_indices: set[int] | None = None
        self.group_builders: list[LayerBatchBuilder] | None = group_builders
        if group_builders is not None:
            self.layer_batch_builder = group_builders[0]
        else:
            self.layer_batch_builder = LayerBatchBuilder(
                token_database,
                page_size_bytes,
                num_layers,
                group_id=0,
            )

    def build_shared_data(self, task: LayerTransferTask) -> SharedBlockData | None:
        """Pre-compute block rows shared by every layer in the task."""
        if self.group_builders is not None:
            builder = self.group_builders[task.group_id]
        else:
            builder = self.layer_batch_builder
        return builder.build_shared(task, is_save=False)

    def _mark_invalid_transfer_task_blocks(self, transfer_tasks: list[LayerTransferTask]) -> None:
        block_ids: set[int] = set()
        for task in transfer_tasks:
            for block_range in task.block_ranges:
                request_block_ids = (
                    block_range.request.block_ids_by_group[task.group_id]
                    if task.group_id < len(block_range.request.block_ids_by_group)
                    else block_range.request.block_ids
                )
                start = max(0, block_range.start_block)
                end = min(block_range.end_block, len(request_block_ids))
                block_ids.update(request_block_ids[start:end])
                partial_index = block_range.partial_block_index
                if partial_index is not None and 0 <= partial_index < len(request_block_ids):
                    block_ids.add(request_block_ids[partial_index])
        with self._invalid_block_ids_lock:
            self._invalid_block_ids.update(block_ids)

    def _mark_invalid_range_indices(self, req_meta: LayerRangeReqMeta, indices: list[int]) -> None:
        with self._invalid_block_ids_lock:
            self._invalid_block_ids.update(req_meta.block_ids[index] for index in indices)

    def _handle_range_request(self, req_meta: LayerRangeReqMeta, shared: SharedBlockData) -> None:
        layer_id = req_meta.layer_id
        if self._active_load_indices is None or layer_id == 0:
            self._active_load_indices = set(range(len(req_meta.keys)))
        assert self._active_load_indices is not None
        active_indices = [
            index
            for index in range(len(req_meta.keys))
            if not self._load_abort_event.is_set() and index in self._active_load_indices
        ]
        active_keys = [req_meta.keys[index] for index in active_indices]
        if active_keys:
            self._stagger_h2d_submit(layer_id)
            active_buffers = [req_meta.all_buffers[index] for index in active_indices]
            active_sizes = [req_meta.all_sizes[index] for index in active_indices]
            active_offsets = [req_meta.all_offsets[index] for index in active_indices]
            results: list[int] = []
            for keys, buffers, sizes, offsets in self._range_transfer_batches(
                active_keys,
                active_buffers,
                active_sizes,
                active_offsets,
                self.max_transfer_blocks,
                self.max_transfer_bytes,
            ):
                results.extend(
                    require_aligned_batch_results(
                        "batch_copy_get",
                        keys,
                        self.m_store.batch_copy_get(keys, buffers, sizes, offsets),
                    )
                )
            _emit_range_debug_event("load", layer_id, active_sizes, active_offsets, results)
            failed_indices = [index for index, result in zip(active_indices, results, strict=True) if result < 0]
            if failed_indices:
                self._mark_invalid_range_indices(req_meta, failed_indices)
                self._active_load_indices.difference_update(failed_indices)

        if layer_id == self.final_layer_id:
            for req_id, is_last_chunk in zip(req_meta.req_ids, shared.is_last_chunks, strict=True):
                if is_last_chunk:
                    self.set_finished_request(req_id)
            self._active_load_indices = None

    def _handle_range_layer_task(self, data: LayerLoadTask) -> None:
        layer_id = data.layer_id
        try:
            wait_for_save = data.wait_for_save_layer
            if wait_for_save is not None:
                while not self.layer_save_finished_events[wait_for_save].wait(timeout=10):
                    if self.save_failure_checker is not None:
                        self.save_failure_checker()
                    logger.info("Layerwise %d save wait timed out, keep waiting before load", wait_for_save)
                if self.save_failure_checker is not None:
                    self.save_failure_checker()
                self.sync_save_events[wait_for_save].synchronize()
                self.layer_save_finished_events[wait_for_save].clear()

            if len(data.transfer_tasks) != 1:
                raise ValueError(f"Expected one Mooncake range task, got {len(data.transfer_tasks)}")
            task = data.transfer_tasks[0]
            shared = task.shared_block_data
            if shared is None:
                raise RuntimeError("Mooncake range load requires shared block metadata")
            builder = self.group_builders[task.group_id] if self.group_builders else self.layer_batch_builder
            req_meta = builder.build_addrs(shared, task.layer_id)
            if not isinstance(req_meta, LayerRangeReqMeta):
                raise TypeError(f"Expected Mooncake range metadata, got {type(req_meta).__name__}")
            if data.attention_start_gate is not None:
                while not data.attention_start_gate.wait(timeout=10):
                    logger.info("Layerwise %d load waits for attention compute start", layer_id)
            if self.external_slot_release_waiter is not None:
                self.external_slot_release_waiter(layer_id)
            self._handle_range_request(req_meta, shared)
        except Exception as exc:
            self._fatal_error = exc
            if self._active_load_indices is not None:
                self._active_load_indices.clear()
            self._mark_invalid_transfer_task_blocks(data.transfer_tasks)
            self._load_abort_event.set()
            raise
        finally:
            if not self.layer_load_finished_events[layer_id].is_set():
                self.layer_load_finished_events[layer_id].set()
            self.request_queue.task_done()
            self.get_event.set()

    def _get_h2d_stagger_delay_us(self, layer_id: int) -> int:
        if self.h2d_stagger_us <= 0:
            return 0
        slot = (self.tp_rank + layer_id) % self.tp_size
        return slot * self.h2d_stagger_us

    def _stagger_h2d_submit(self, layer_id: int) -> None:
        delay_us = self._get_h2d_stagger_delay_us(layer_id)
        if delay_us <= 0:
            return
        deadline = time.perf_counter() + delay_us / 1_000_000
        while time.perf_counter() < deadline:
            pass

    def _handle_request(  # type: ignore[override]
        self, data: LayerLoadTask | LayerwisePreparation
    ):
        if isinstance(data, LayerwisePreparation):
            data.ensure_ready()
            self.request_queue.task_done()
            return
        if data.preparation is not None:
            data.preparation.ensure_ready()
        if data.transfer_tasks and data.transfer_tasks[0].use_key_major_ranges:
            self._handle_range_layer_task(data)
            return
        # Layerwise threads only run when the worker-side
        # protocol gate is on; every store call below sits on the Backend ABC.
        wait_for_save = data.wait_for_save_layer
        transfer_tasks = data.transfer_tasks
        layer_id = data.layer_id
        attention_start_gate = data.attention_start_gate

        if wait_for_save is not None:
            while not self.layer_save_finished_events[wait_for_save].wait(timeout=10):
                if self.save_failure_checker is not None:
                    self.save_failure_checker()
                logger.info("Layerwise %d save wait timed out, keep waiting before load", wait_for_save)
            if self.save_failure_checker is not None:
                self.save_failure_checker()
            # Non-saving TP ranks have no D2H task to synchronize the event.
            # Their CPU save-finished signal only means the event was recorded;
            # wait for the NPU work before reusing the local HBM buffer.
            self.sync_save_events[wait_for_save].synchronize()
            logger.debug("Layer save event cleared: layer %d", wait_for_save)
            self.layer_save_finished_events[wait_for_save].clear()

        if len(transfer_tasks) == 0:
            if self.external_slot_release_waiter is not None:
                self.external_slot_release_waiter(layer_id)
            assert not self.layer_load_finished_events[layer_id].is_set()
            logger.debug("Layer load event set: layer %d", layer_id)
            self.layer_load_finished_events[layer_id].set()
            self.request_queue.task_done()
            return

        # Build req_meta for all tasks first; if all are None, early return.
        task_metas: list[tuple[LayerTransferTask, LayerBatchReqMeta]] = []
        for task in transfer_tasks:
            shared = task.shared_block_data
            builder = self.group_builders[task.group_id] if self.group_builders else self.layer_batch_builder
            if shared is not None:
                built_meta = builder.build_addrs(shared, task.layer_idx_in_group)
                if not isinstance(built_meta, LayerBatchReqMeta):
                    raise TypeError(f"Expected GVA layer metadata, got {type(built_meta).__name__}")
                req_meta: LayerBatchReqMeta | None = built_meta
            else:
                candidate_meta = builder.build(task, is_save=False)
                if candidate_meta is not None and not isinstance(candidate_meta, LayerBatchReqMeta):
                    raise TypeError(f"Expected GVA layer metadata, got {type(candidate_meta).__name__}")
                req_meta = candidate_meta
            if req_meta is not None:
                task_metas.append((task, req_meta))

        if not task_metas:
            if self.external_slot_release_waiter is not None:
                self.external_slot_release_waiter(layer_id)
            assert not self.layer_load_finished_events[layer_id].is_set()
            logger.debug("Layer load event set: layer %d", layer_id)
            self.layer_load_finished_events[layer_id].set()
            self.request_queue.task_done()
            return

        if attention_start_gate is not None:
            while not attention_start_gate.wait(timeout=10):
                logger.info("Layerwise %d load waits for attention compute start", layer_id)

        all_load_keys: list[str] = []
        all_req_ids: set[str] = set()
        last_chunk_req_ids: set[str] = set()
        all_gvas = []
        all_addrs = []
        all_sizes = []
        for task, req_meta in task_metas:
            if req_meta.load_keys:
                all_load_keys.extend(req_meta.load_keys)
            for req_id, is_last_chunk in zip(req_meta.req_ids, req_meta.is_last_chunks):
                all_req_ids.add(req_id)
                if is_last_chunk:
                    last_chunk_req_ids.add(req_id)
            all_gvas.append(req_meta.gvas_array)
            all_addrs.append(req_meta.addr_array)
            all_sizes.append(req_meta.size_array)

        self._stagger_h2d_submit(layer_id)
        gvas_array = np.concatenate(all_gvas) if len(all_gvas) > 1 else all_gvas[0]
        addr_array = np.concatenate(all_addrs) if len(all_addrs) > 1 else all_addrs[0]
        size_array = np.concatenate(all_sizes) if len(all_sizes) > 1 else all_sizes[0]
        if self.external_slot_release_waiter is not None:
            self.external_slot_release_waiter(layer_id)
        res = self._batch_copy_with_limits(
            gvas_array,
            addr_array,
            size_array,
            1,
            self.max_transfer_blocks,
            self.max_transfer_bytes,
        )
        if layer_id <= 2 or res != 0:
            logger.debug(
                "load_thread: layer=%d groups=%d blocks=%d res=%d",
                layer_id,
                len(all_gvas),
                len(gvas_array),
                res,
            )
        if res != 0:
            raise RuntimeError(f"Layerwise {layer_id} load batch_copy failed with return code {res}")

        finished_req_ids = set().union(*(task.finished_req_ids for task, _ in task_metas))
        if self.release_load_leases is not None:
            self.release_load_leases(finished_req_ids)
        elif layer_id == self.final_layer_id and all_load_keys:
            unique_load_keys = list(dict.fromkeys(all_load_keys))
            self.m_store.batch_remove_lease(unique_load_keys)
            logger.debug(
                "[KVPOOL] load_thread released %d leases after final layer %d",
                len(unique_load_keys),
                layer_id,
            )
        if finished_req_ids or layer_id == self.final_layer_id:
            for req_id in finished_req_ids if self.release_load_leases is not None else all_req_ids:
                if req_id in last_chunk_req_ids:
                    self.set_finished_request(req_id)
        assert not self.layer_load_finished_events[layer_id].is_set(), f"thread: {layer_id} load failed "
        logger.debug("Layer load event set: layer %d", layer_id)
        self.layer_load_finished_events[layer_id].set()
        # transfer_tasks aliases KVPoolWorker.layer_load_tasks[layer_id]. Do
        # not mutate the worker-owned list from this asynchronous thread. The
        # worker replaces all per-layer lists at the beginning of every step.
        self.request_queue.task_done()
        self.get_event.set()


def record_failed_blocks(
    block_ids: list[int],
    ret_codes: list[int],
) -> set[int]:
    failed_blocks: set[int] = set()
    for block_id, code in zip(block_ids, ret_codes):
        if code != 0:
            failed_blocks.add(block_id)
    if failed_blocks:
        logger.error(
            "Failed to load blocks. failed_count=%d, failed_blocks=%s. Check block availability and memory state.",
            len(failed_blocks),
            failed_blocks,
        )
    return failed_blocks
