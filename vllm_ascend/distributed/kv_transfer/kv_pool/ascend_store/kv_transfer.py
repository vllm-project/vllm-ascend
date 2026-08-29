from __future__ import annotations

import ctypes
import queue
import threading
from collections import defaultdict
from typing import Any

import torch
from vllm.distributed.kv_events import BlockStored
from vllm.logger import logger
from vllm.v1.core.kv_cache_utils import maybe_convert_block_hash

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.base import Backend

# isort: off
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import (
    ChunkedTokenDatabase,
    LayerLoadTask,
    LayerTransferTask,
    ReqMeta,
    get_block_hashes,
)
# isort: on


def _circular_shift(lst: list, offset: int) -> list:
    if not lst or offset == 0:
        return lst
    return lst[offset:] + lst[:offset]


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
            exists_states = self.lookup(keys)
            missing_indices = [index for index, exists in enumerate(exists_states) if not exists]
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
            ret = self.m_store.get(key_list_c, addr_list_c, size_list_c)
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
            exists_states = self.lookup(key_list)
            missing_indices = [index for index, exists in enumerate(exists_states) if not exists]
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
