import queue
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from collections.abc import Callable
from typing import Any

import torch
from vllm.distributed.kv_events import BlockStored
from vllm.logger import logger
from vllm.v1.core.kv_cache_utils import maybe_convert_block_hash

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.backend import Backend

# isort: off
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.config_data import (
    ChunkedTokenDatabase,
    LayerMultiBlockReqMeta,
    ReqMeta,
)
# isort: on


class KVTransferThread(threading.Thread):
    def __init__(
        self,
        m_store: Backend,
        token_database: ChunkedTokenDatabase,
        block_size: int | list[int],
        tp_rank: int,
        dcp_size: int,
        ready_event: threading.Event,
        name: str,
    ):
        super().__init__(daemon=True, name=name)
        self.m_store = m_store
        self.ready_event = ready_event
        self.block_size = block_size
        self.tp_rank = tp_rank
        self.dcp_size = dcp_size
        self.token_database = token_database
        self.done_task_lock = threading.Lock()
        self.request_queue: queue.Queue[Any] = queue.Queue()
        # TODO(jianzs): make this configurable
        self.executor = ThreadPoolExecutor(max_workers=32)
        self.finished_requests: set[str] = set()
        self.kv_event_lock = threading.Lock()
        self.kv_events: list[BlockStored] = []

    def _get_block_size(self, kv_cache_group_id: int = 0) -> int:
        if isinstance(self.block_size, list):
            if kv_cache_group_id >= len(self.block_size):
                return self.block_size[0]
            return self.block_size[kv_cache_group_id]
        return self.block_size

    def add_request(
        self,
        request: ReqMeta | LayerMultiBlockReqMeta,
    ) -> torch.Tensor:
        import time as _kvtrace_time
        if not hasattr(self, "_kvtrace_enqueue_times"):
            self._kvtrace_enqueue_times = {}
        self._kvtrace_enqueue_times[request.req_id] = _kvtrace_time.perf_counter()
        self.request_queue.put(request)

    def get_and_clear_finished_requests(self) -> set[str]:
        """
        Get and clear the requests that have been completed.
        Returns:
            A set of request IDs that have been completed.
        """
        with self.done_task_lock:
            finished_requests = self.finished_requests.copy()
            self.finished_requests.clear()
        return finished_requests

    def set_finished_request(self, req_id):
        with self.done_task_lock:
            self.finished_requests.add(req_id)

    def run(self):
        """Run the thread to handle KV cache transfer requests."""
        self.m_store.set_device()
        self.ready_event.set()
        while True:
            try:
                request_data = self.request_queue.get()
                if request_data is None:
                    logger.warning("Received a None request!")
                    self.request_queue.task_done()
                    continue
                self._handle_request(request_data)
            except Exception as e:
                logger.error("Error in KVCacheTransferThread: %s", e)

    def _handle_request(self, req_meta: Any):
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
            logger.error("Remote connection failed in contains: %s", e)
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

    def _process_tokens_with_block_ids(
        self,
        token_len: int,
        block_hashes,
        block_ids: list[int],
        mask_num: int = 0,
        kv_cache_group_id: int = 0,
        skip_null_blocks: bool = False,
        cache_role: str = "kv",
        chunk_filter: Callable[[int], bool] | None = None,
    ):
        process_with_block_ids = getattr(self.token_database, "process_tokens_with_block_ids", None)
        if process_with_block_ids is not None:
            return process_with_block_ids(
                token_len,
                block_hashes,
                block_ids,
                mask_num,
                kv_cache_group_id=kv_cache_group_id,
                skip_null_blocks=skip_null_blocks,
                cache_role=cache_role,
                chunk_filter=chunk_filter,
            )

        def iter_with_legacy_process_tokens():
            try:
                token_iter = self.token_database.process_tokens(token_len, block_hashes, mask_num)
            except TypeError:
                token_iter = self.token_database.process_tokens(token_len, block_hashes)
            group_block_size = self._get_block_size(kv_cache_group_id)
            for start, end, key in token_iter:
                if chunk_filter is not None and not chunk_filter(start):
                    continue
                block_idx = start // group_block_size
                if block_idx >= len(block_ids):
                    continue
                block_id = block_ids[block_idx]
                if skip_null_blocks and cache_role == "kv" and block_id <= 0:
                    continue
                yield start, end, key, block_id

        return iter_with_legacy_process_tokens()

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

    def _chunk_mask_allows(
        self,
        masks: tuple[list[bool], ...] | None,
        kv_cache_group_id: int,
        start: int,
    ) -> bool:
        mask_allows_chunk = getattr(self.token_database, "mask_allows_chunk", None)
        if mask_allows_chunk is not None:
            return mask_allows_chunk(masks, kv_cache_group_id, start)
        if masks is None or kv_cache_group_id >= len(masks):
            return True
        mask = masks[kv_cache_group_id]
        chunk_idx = start // self._get_block_size(kv_cache_group_id)
        return chunk_idx < len(mask) and mask[chunk_idx]

    def _store_mask(self, req_meta: ReqMeta) -> tuple[list[bool], ...] | None:
        store_mask = getattr(self.token_database, "store_mask", None)
        if store_mask is None:
            return None
        try:
            return store_mask(req_meta.token_len_chunk, req_meta.num_prompt_tokens)
        except AssertionError as exc:
            logger.debug("Skip AscendStore store mask for unaligned request %s: %s", req_meta.req_id, exc)
            return None

    def _load_mask(
        self,
        req_meta: ReqMeta,
        token_len: int,
    ) -> tuple[list[bool], ...] | None:
        load_mask = getattr(self.token_database, "load_mask", None)
        if load_mask is None:
            return None
        return load_mask(req_meta.block_hashes, token_len)


class KVCacheStoreSendingThread(KVTransferThread):
    def __init__(
        self,
        m_store: Backend,
        token_database: ChunkedTokenDatabase,
        block_size: int | list[int],
        tp_rank: int,
        dcp_size: int,
        put_step: int,
        kv_role: str,
        ready_event: threading.Event,
        enable_kv_event: bool = False,
    ):
        super().__init__(
            m_store, token_database, block_size, tp_rank, dcp_size, ready_event, name="KVCacheSendingThread"
        )
        self.put_step = put_step
        self.kv_role = kv_role
        self.stored_requests = defaultdict[str, int](int)
        self.enable_kv_event = enable_kv_event
        import os as _os
        self._skip_existed_group = _os.getenv("VLLM_ASCEND_SKIP_STORE_EXISTED_GROUP", "0") == "1"
        self._put_key_string_fast_path = _os.getenv(
            "VLLM_ASCEND_PUT_KEY_STRING_FAST_PATH", "1"
        ) != "0"
        self._put_sparse_store_mask = _os.getenv("VLLM_ASCEND_PUT_SPARSE_STORE_MASK", "0") == "1"
        self._put_pre_shard_key_build = _os.getenv("VLLM_ASCEND_PUT_PRE_SHARD_KEY_BUILD", "0") == "1"
        self._key_build_cache = {}  # (block_hash_prefix, group_id) -> (starts, ends, keys, block_hashes)
        self._key_build_cache_max = 10000
        self._per_request_save_wait = _os.getenv("VLLM_ASCEND_PER_REQUEST_SAVE_WAIT", "0") == "1"
        self._stored_request_done_events: dict[str, threading.Event] = {}

    def add_stored_request(self, req_id: str):
        with self.done_task_lock:
            self.stored_requests[req_id] += 1

    def dec_stored_request(self, req_id: str):
        with self.done_task_lock:
            if req_id in self.stored_requests:
                self.stored_requests[req_id] -= 1
                return self.stored_requests[req_id]
            return None

    def prepare_stored_request_done_event(self, req_id: str):
        if not self._per_request_save_wait:
            return None
        with self.done_task_lock:
            event = self._stored_request_done_events.get(req_id)
            if event is None:
                event = threading.Event()
                self._stored_request_done_events[req_id] = event
            return event

    def _notify_stored_request_done(self, req_id: str, remaining: int | None = 0):
        if not self._per_request_save_wait:
            return
        if remaining is not None and remaining > 0:
            return
        with self.done_task_lock:
            event = self._stored_request_done_events.pop(req_id, None)
        if event is not None:
            event.set()

    def delete_finished_stored_request(self, req_id: str):
        with self.done_task_lock:
            if req_id in self.stored_requests:
                del self.stored_requests[req_id]
            event = self._stored_request_done_events.pop(req_id, None)
        if event is not None:
            event.set()

    def _handle_request(self, req_meta: ReqMeta):
        import time as _kvtrace_time
        _kvtrace_t_enq = getattr(self, "_kvtrace_enqueue_times", {}).pop(req_meta.req_id, None)
        _kvtrace_t_start = _kvtrace_time.perf_counter()
        _kvtrace_qwait_ms = ((_kvtrace_t_start - _kvtrace_t_enq) * 1000) if _kvtrace_t_enq else -1
        _kvtrace_qsize = self.request_queue.qsize()
        token_len = req_meta.token_len_chunk
        req_id = req_meta.req_id
        current_event = req_meta.current_event
        if req_id not in self.stored_requests:
            self._notify_stored_request_done(req_id)
            self.request_queue.task_done()
            return

        _kvtrace_t_sm = _kvtrace_time.perf_counter()
        store_masks = self._store_mask(req_meta)
        _kvtrace_store_mask_ms = (_kvtrace_time.perf_counter() - _kvtrace_t_sm) * 1000
        for group_id in req_meta.kv_cache_group_ids or [0]:
            # 6C: store_mask 前移
            group_store_mask = None
            if store_masks is not None and group_id < len(store_masks):
                group_store_mask = store_masks[group_id]
                if not group_store_mask or not any(group_store_mask):
                    if logger.isEnabledFor(20):
                        logger.info(
                            "KVTRACE req=%s stage=kvpool_put_skip_group group=%d "
                            "reason=no_store_mask", req_id, group_id
                        )
                    continue
            _kvtrace_t_group = _kvtrace_time.perf_counter()
            starts = []
            ends = []
            keys = []
            block_hashes = []
            key_block_ids = []
            block_ids = req_meta.block_ids_by_group[group_id]
            group_block_size = self._get_block_size(group_id)
            chunk_filter = lambda start, group_id=group_id: self._chunk_mask_allows(store_masks, group_id, start)

            _kvtrace_pt_count = 0
            _kvtrace_mask_ms = 0.0
            _kvtrace_str_app_ms = 0.0
            _kvtrace_cached = False
            _kvtrace_sparse_mask = False
            _kvtrace_pre_sharded = False

            # Phase 6B': check key_build cache
            if self._skip_existed_group and req_meta.block_hashes:
                _cache_key = (hash(tuple(req_meta.block_hashes[:8])), group_id)
                _cached_data = self._key_build_cache.get(_cache_key)
                if _cached_data is not None:
                    _kvtrace_cached = True
                    _raw_starts, _raw_ends, _raw_keys, _raw_block_hashes = _cached_data
                    for _ci, _cs in enumerate(_raw_starts):
                        _kvtrace_pt_count += 1
                        _kvtrace_t_m = _kvtrace_time.perf_counter()
                        if not self._chunk_mask_allows(store_masks, group_id, _cs):
                            _kvtrace_mask_ms += (_kvtrace_time.perf_counter() - _kvtrace_t_m) * 1000
                            continue
                        _kvtrace_mask_ms += (_kvtrace_time.perf_counter() - _kvtrace_t_m) * 1000
                        starts.append(_cs)
                        ends.append(_raw_ends[_ci])
                        keys.append(_raw_keys[_ci])
                        block_hashes.append(_raw_block_hashes[_ci])
                        key_block_ids.append(block_ids[_cs // group_block_size])
            if not _kvtrace_cached:
                if self._put_key_string_fast_path:
                    if self._put_sparse_store_mask and group_store_mask is not None:
                        _kvtrace_sparse_mask = True
                        _kvtrace_pre_sharded = (
                            self._put_pre_shard_key_build
                            and not self.dcp_size > 1
                            and not req_meta.disable_tp_key_sharding
                        )
                        iterator = self.token_database.process_token_key_strings_with_block_ids_sparse_store_mask(
                            token_len,
                            req_meta.block_hashes,
                            block_ids,
                            group_store_mask,
                            kv_cache_group_id=group_id,
                            skip_null_blocks=self._skip_null_blocks(req_meta, group_id),
                            shard_rank=(self.tp_rank % self.put_step) if _kvtrace_pre_sharded else None,
                            shard_size=self.put_step if _kvtrace_pre_sharded else None,
                        )
                    else:
                        iterator = self.token_database.process_token_key_strings_with_block_ids(
                            token_len,
                            req_meta.block_hashes,
                            block_ids,
                            kv_cache_group_id=group_id,
                            skip_null_blocks=self._skip_null_blocks(req_meta, group_id),
                            chunk_filter=chunk_filter,
                        )
                    for start, end, key_string, chunk_hash, block_id in iterator:
                        _kvtrace_pt_count += 1
                        if not _kvtrace_sparse_mask:
                            _kvtrace_t_m = _kvtrace_time.perf_counter()
                            if not self._chunk_mask_allows(store_masks, group_id, start):
                                _kvtrace_mask_ms += (_kvtrace_time.perf_counter() - _kvtrace_t_m) * 1000
                                continue
                            _kvtrace_mask_ms += (_kvtrace_time.perf_counter() - _kvtrace_t_m) * 1000
                        _kvtrace_t_sa = _kvtrace_time.perf_counter()
                        starts.append(start)
                        ends.append(end)
                        keys.append(key_string)
                        if self.enable_kv_event:
                            block_hashes.append(chunk_hash)
                        key_block_ids.append(block_id)
                        _kvtrace_str_app_ms += (_kvtrace_time.perf_counter() - _kvtrace_t_sa) * 1000
                else:
                    for start, end, key, block_id in self._process_tokens_with_block_ids(
                        token_len,
                        req_meta.block_hashes,
                        block_ids,
                        kv_cache_group_id=group_id,
                        skip_null_blocks=self._skip_null_blocks(req_meta, group_id),
                        chunk_filter=chunk_filter,
                    ):
                        _kvtrace_pt_count += 1
                        _kvtrace_t_m = _kvtrace_time.perf_counter()
                        if not self._chunk_mask_allows(store_masks, group_id, start):
                            _kvtrace_mask_ms += (_kvtrace_time.perf_counter() - _kvtrace_t_m) * 1000
                            continue
                        _kvtrace_mask_ms += (_kvtrace_time.perf_counter() - _kvtrace_t_m) * 1000
                        _kvtrace_t_sa = _kvtrace_time.perf_counter()
                        starts.append(start)
                        ends.append(end)
                        keys.append(key.to_string())
                        if self.enable_kv_event:
                            assert key.chunk_hash_bytes is not None
                            block_hashes.append(key.chunk_hash_bytes)
                        key_block_ids.append(block_id)
                        _kvtrace_str_app_ms += (_kvtrace_time.perf_counter() - _kvtrace_t_sa) * 1000
                # Cache key_build result (pre-sharding, post-mask)
                if (
                    self._skip_existed_group
                    and not _kvtrace_pre_sharded
                    and req_meta.block_hashes
                    and len(self._key_build_cache) < self._key_build_cache_max
                ):
                    self._key_build_cache[_cache_key] = (list(starts), list(ends), list(keys), list(block_hashes) if self.enable_kv_event else [])

            if not _kvtrace_pre_sharded and not self.dcp_size > 1 and not req_meta.disable_tp_key_sharding:
                starts = starts[self.tp_rank % self.put_step :: self.put_step]
                ends = ends[self.tp_rank % self.put_step :: self.put_step]
                keys = keys[self.tp_rank % self.put_step :: self.put_step]
                block_hashes = block_hashes[self.tp_rank % self.put_step :: self.put_step]
                key_block_ids = key_block_ids[self.tp_rank % self.put_step :: self.put_step]

            _kvtrace_keybuild_ms = (_kvtrace_time.perf_counter() - _kvtrace_t_group) * 1000
            _kvtrace_pt_iter_ms = _kvtrace_keybuild_ms - _kvtrace_mask_ms - _kvtrace_str_app_ms - _kvtrace_store_mask_ms
            logger.info(
                "KVTRACE stage=kvpool_put_kb_detail group=%d "
                "pt_count=%d iter_ms=%.3f mask_ms=%.3f str_app_ms=%.3f "
                "store_mask_ms=%.3f total_ms=%.3f fast_path=%s sparse_mask=%s pre_shard=%s",
                group_id, _kvtrace_pt_count,
                _kvtrace_pt_iter_ms, _kvtrace_mask_ms, _kvtrace_str_app_ms,
                _kvtrace_store_mask_ms, _kvtrace_keybuild_ms,
                self._put_key_string_fast_path, _kvtrace_sparse_mask, _kvtrace_pre_sharded
            )

            if not keys:
                logger.info(
                    "KVTRACE req=%s stage=kvpool_put_group group=%d "
                    "key_build_ms=%.3f exist_ms=0.000 prepare_ms=0.000 "
                    "event_sync_ms=0.000 put_ms=0.000 total_ms=%.3f keys=0",
                    req_id, group_id, _kvtrace_keybuild_ms, _kvtrace_keybuild_ms
                )
                continue

            _kvtrace_t0 = _kvtrace_time.perf_counter()
            exists_states = self.lookup(keys)
            _kvtrace_exist_ms = (_kvtrace_time.perf_counter() - _kvtrace_t0) * 1000
            missing_indices = [index for index, exists in enumerate(exists_states) if not exists]
            logger.info(
                "KVTRACE req=%s stage=kvpool_put_exist elapsed_ms=%.3f "
                "group=%d total_keys=%d missing_keys=%d",
                req_id, _kvtrace_exist_ms, group_id,
                len(keys), len(missing_indices)
            )

            if not missing_indices:
                logger.info(
                    "KVTRACE req=%s stage=kvpool_put_skip_group group=%d "
                    "reason=all_exist keys=%d key_build_ms=%.3f exist_ms=%.3f cached=%s",
                    req_id, group_id, len(keys),
                    _kvtrace_keybuild_ms, _kvtrace_exist_ms, _kvtrace_cached
                )
                continue

            starts = [starts[index] for index in missing_indices]
            ends = [ends[index] for index in missing_indices]
            keys = [keys[index] for index in missing_indices]
            if self.enable_kv_event:
                block_hashes = [block_hashes[index] for index in missing_indices]
            key_block_ids = [key_block_ids[index] for index in missing_indices]

            logger.info(
                "Storing KV cache for %d out of %d blocks (missing_count=%d) for request %s in group %d",
                len(keys),
                token_len // group_block_size,
                len(missing_indices),
                req_id,
                group_id,
            )
            logger.debug(
                "KV pool put request=%s group=%d token_len=%d keys=%d sample_keys=%s",
                req_id,
                group_id,
                token_len,
                len(keys),
                keys[:3],
            )

            _kvtrace_t_prep = _kvtrace_time.perf_counter()
            addrs = []
            sizes = []
            stored_events: list[BlockStored] = []
            prev_key = None
            new_block_hashes = [maybe_convert_block_hash(bh) for bh in block_hashes] if self.enable_kv_event else []
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

                # Create KV event
                if self.enable_kv_event:
                    token_ids = req_meta.token_ids[start : ends[index]] if req_meta.token_ids is not None else None
                    block_size = (
                        req_meta.original_block_size[group_id]
                        if isinstance(req_meta.original_block_size, list)
                        else req_meta.original_block_size
                    )
                    if block_size is not None:
                        stored_event = BlockStored(
                            block_hashes=[new_block_hashes[index]],
                            parent_block_hash=prev_key,
                            token_ids=token_ids,
                            block_size=block_size,
                            lora_id=None,
                            medium="cpu",
                            lora_name=None,
                        )
                        stored_events.append(stored_event)
                        prev_key = new_block_hashes[index]
                        logger.debug("Added kv cache event '%s' to kv cache events queue", stored_event)

            if self.kv_role == "kv_consumer":
                keys, addrs, sizes = self._decode_adaptor_prefill_pp(
                    keys,
                    addrs,
                    sizes,
                    kv_cache_group_id=group_id,
                )

            _kvtrace_prep_ms = (_kvtrace_time.perf_counter() - _kvtrace_t_prep) * 1000
            _kvtrace_t0_evt = _kvtrace_time.perf_counter()
            if current_event is not None:
                current_event.synchronize()
            _kvtrace_evt_ms = (_kvtrace_time.perf_counter() - _kvtrace_t0_evt) * 1000
            logger.info(
                "KVTRACE req=%s stage=kvpool_put_event_sync elapsed_ms=%.3f "
                "has_event=%s",
                req_id, _kvtrace_evt_ms, current_event is not None
            )
            _kvtrace_t0_put = _kvtrace_time.perf_counter()
            self.m_store.put(keys, addrs, sizes)
            _kvtrace_put_ms = (_kvtrace_time.perf_counter() - _kvtrace_t0_put) * 1000
            logger.info(
                "KVTRACE req=%s stage=kvpool_put_backend elapsed_ms=%.3f "
                "group=%d keys=%d role=%s tp_rank=%d put_step=%d",
                req_id, _kvtrace_put_ms, group_id, len(keys),
                self.kv_role, self.tp_rank, self.put_step
            )

            # TODO Query specific replica info to update the event
            if self.enable_kv_event and stored_events is not None:
                self.update_kv_event(stored_events)

            _kvtrace_group_total_ms = (_kvtrace_time.perf_counter() - _kvtrace_t_group) * 1000
            logger.info(
                "KVTRACE req=%s stage=kvpool_put_group group=%d "
                "key_build_ms=%.3f exist_ms=%.3f prepare_ms=%.3f "
                "event_sync_ms=%.3f put_ms=%.3f total_ms=%.3f keys=%d",
                req_id, group_id,
                _kvtrace_keybuild_ms, _kvtrace_exist_ms,
                _kvtrace_prep_ms, _kvtrace_evt_ms, _kvtrace_put_ms,
                _kvtrace_group_total_ms, len(keys)
            )

        _kvtrace_handle_ms = (_kvtrace_time.perf_counter() - _kvtrace_t_start) * 1000
        logger.info(
            "KVTRACE req=%s stage=kvpool_send_handle_done elapsed_ms=%.3f "
            "queue_wait_ms=%.3f queue_size=%d group=%d",
            req_id, _kvtrace_handle_ms, _kvtrace_qwait_ms,
            _kvtrace_qsize, group_id
        )
        remaining_jobs = self.dec_stored_request(req_id)
        self._notify_stored_request_done(req_id, remaining_jobs)
        self.request_queue.task_done()


class KVCacheStoreRecvingThread(KVTransferThread):
    def __init__(
        self,
        m_store: Backend,
        token_database: ChunkedTokenDatabase,
        block_size: int | list[int],
        tp_rank: int,
        dcp_size: int,
        ready_event: threading.Event,
    ):
        super().__init__(
            m_store, token_database, block_size, tp_rank, dcp_size, ready_event, name="KVCacheStoreRecvingThread"
        )

    def _handle_request(self, req_meta: ReqMeta):
        token_len = req_meta.load_spec.token_len  # type: ignore[union-attr]
        req_id = req_meta.req_id
        addr_list = []
        size_list = []
        key_list = []
        load_masks = self._load_mask(req_meta, token_len)
        for group_id in req_meta.kv_cache_group_ids or [0]:
            block_ids = req_meta.block_ids_by_group[group_id]
            group_block_size = self._get_block_size(group_id)
            mask_num = (
                req_meta.load_spec.vllm_cached_tokens  # type: ignore[union-attr]
                // group_block_size
                * group_block_size
            )
            chunk_filter = lambda start, group_id=group_id: self._chunk_mask_allows(load_masks, group_id, start)
            for start, end, key, block_id in self._process_tokens_with_block_ids(
                token_len,
                req_meta.block_hashes,
                block_ids,
                mask_num,
                kv_cache_group_id=group_id,
                skip_null_blocks=self._skip_null_blocks(req_meta, group_id),
                chunk_filter=chunk_filter,
            ):
                if not self._chunk_mask_allows(load_masks, group_id, start):
                    continue
                addr, size, _ = self._prepare_value(
                    start,
                    end,
                    block_ids,
                    kv_cache_group_id=group_id,
                    block_id=block_id,
                )
                key_list.append(key.to_string())
                addr_list.append(addr)
                size_list.append(size)
        if not key_list:
            self.set_finished_request(req_id)
            self.request_queue.task_done()
            return
        key_list_c = key_list[self.tp_rank % len(key_list) :] + key_list[: self.tp_rank % len(key_list)]
        addr_list_c = addr_list[self.tp_rank % len(addr_list) :] + addr_list[: self.tp_rank % len(addr_list)]
        size_list_c = size_list[self.tp_rank % len(size_list) :] + size_list[: self.tp_rank % len(size_list)]
        logger.debug(
            "KV pool async recv calls backend get request=%s token_len=%d groups=%s keys=%d sample_keys=%s",
            req_id,
            token_len,
            req_meta.kv_cache_group_ids or [0],
            len(key_list_c),
            key_list_c[:3],
        )
        import time as _kvtrace_time
        _kvtrace_t0 = _kvtrace_time.perf_counter()
        self.m_store.get(key_list_c, addr_list_c, size_list_c)
        _kvtrace_get_ms = (_kvtrace_time.perf_counter() - _kvtrace_t0) * 1000
        logger.info(
            "KVTRACE req=%s stage=kvpool_get_backend elapsed_ms=%.3f "
            "mode=async keys=%d",
            req_id, _kvtrace_get_ms, len(key_list_c)
        )
        self.set_finished_request(req_id)
        self.request_queue.task_done()


class KVCacheStoreLayerSendingThread(KVTransferThread):
    def __init__(
        self,
        m_store: Backend,
        token_database: ChunkedTokenDatabase,
        block_size: int | list[int],
        tp_rank: int,
        dcp_size: int,
        put_step: int,
        ready_event: threading.Event,
        num_layers: int,
        enable_kv_event: bool = False,
    ):
        super().__init__(
            m_store, token_database, block_size, tp_rank, dcp_size, ready_event, name="KVCacheStoreLayerSendingThread"
        )
        self.final_layer_id = num_layers - 1
        self.put_step = put_step
        self.enable_kv_event = enable_kv_event

    def add_request(  # type: ignore[override]
        self, req_meta: ReqMeta
    ) -> torch.Tensor:
        self.request_queue.put(req_meta)

    def _handle_request(  # type: ignore[override]
        self, req_meta: LayerMultiBlockReqMeta
    ):
        starts = req_meta.starts
        ends = req_meta.ends
        keys = req_meta.keys
        layer_id = req_meta.layer_id
        current_event = req_meta.current_event
        total_block = len(keys)
        is_last_chunk = req_meta.is_last_chunk
        if not self.dcp_size > 1:
            starts = starts[self.tp_rank % self.put_step :: self.put_step]
            ends = ends[self.tp_rank % self.put_step :: self.put_step]
            keys = keys[self.tp_rank % self.put_step :: self.put_step]

        if not keys:
            if is_last_chunk:
                self.set_finished_request(req_meta.req_id)
            return

        key_list = []
        for key in keys:
            key_list.append(key.to_string())

        exists_states = self.lookup(key_list)
        missing_indices = [index for index, exists in enumerate(exists_states) if not exists]

        if not missing_indices:
            if is_last_chunk and layer_id == self.final_layer_id:
                self.set_finished_request(req_meta.req_id)
            return

        starts = [starts[index] for index in missing_indices]
        ends = [ends[index] for index in missing_indices]
        key_list = [key_list[index] for index in missing_indices]

        addr_list = []
        size_list = []
        for index, key in enumerate(key_list):
            addr, size = self.token_database.prepare_value_layer(
                starts[index], ends[index], req_meta.block_ids_by_group[0], layer_id
            )
            addr_list.append(addr)
            size_list.append(size)

        if current_event is not None:
            current_event.synchronize()
        self.m_store.put(key_list, addr_list, size_list)

        if layer_id == self.final_layer_id and is_last_chunk:
            self.set_finished_request(req_meta.req_id)
        self.request_queue.task_done()

        logger.info(
            "Storing KV cache for %d out of %d blocks (missing_count=%d) for request %s",
            len(key_list),
            total_block,
            len(missing_indices),
            req_meta.req_id,
        )


class KVCacheStoreLayerRecvingThread(KVTransferThread):
    def __init__(
        self,
        m_store: Backend,
        token_database: ChunkedTokenDatabase,
        block_size: int | list[int],
        tp_rank: int,
        dcp_size: int,
        ready_event: threading.Event,
        get_event: threading.Event,
    ):
        super().__init__(
            m_store, token_database, block_size, tp_rank, dcp_size, ready_event, name="KVCacheStoreLayerRecvingThread"
        )
        self.get_event = get_event

    def add_request(  # type: ignore[override]
        self, req_meta: LayerMultiBlockReqMeta
    ) -> torch.Tensor:
        self.request_queue.put(req_meta)

    def _handle_request(  # type: ignore[override]
        self, req_meta: LayerMultiBlockReqMeta
    ):
        addr_list = []
        size_list = []
        key_list = []
        for index, key in enumerate(req_meta.keys):
            addr, size = self.token_database.prepare_value_layer(
                req_meta.starts[index], req_meta.ends[index], req_meta.block_ids_by_group[0], req_meta.layer_id
            )
            key_list.append(key.to_string())
            addr_list.append(addr)
            size_list.append(size)
        key_list_c = key_list[self.tp_rank % len(key_list) :] + key_list[: self.tp_rank % len(key_list)]
        addr_list_c = addr_list[self.tp_rank % len(addr_list) :] + addr_list[: self.tp_rank % len(addr_list)]
        size_list_c = size_list[self.tp_rank % len(size_list) :] + size_list[: self.tp_rank % len(size_list)]
        self.m_store.get(key_list_c, addr_list_c, size_list_c)

        self.request_queue.task_done()
        self.get_event.set()
