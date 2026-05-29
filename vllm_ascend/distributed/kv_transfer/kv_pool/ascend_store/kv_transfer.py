import queue
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
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
    get_block_hashes,
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
            )

        def iter_with_legacy_process_tokens():
            try:
                token_iter = self.token_database.process_tokens(token_len, block_hashes, mask_num)
            except TypeError:
                token_iter = self.token_database.process_tokens(token_len, block_hashes)
            group_block_size = self._get_block_size(kv_cache_group_id)
            for start, end, key in token_iter:
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
    ):
        try:
            return self.token_database.prepare_value(
                start,
                end,
                block_ids,
                kv_cache_group_id=kv_cache_group_id,
                cache_role=cache_role,
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

    def add_stored_request(self, req_id: str):
        with self.done_task_lock:
            self.stored_requests[req_id] += 1

    def dec_stored_request(self, req_id: str):
        with self.done_task_lock:
            if req_id in self.stored_requests:
                self.stored_requests[req_id] -= 1

    def delete_finished_stored_request(self, req_id: str):
        with self.done_task_lock:
            if req_id in self.stored_requests:
                del self.stored_requests[req_id]

    def _handle_request(self, req_meta: ReqMeta):
        token_len = req_meta.token_len_chunk
        req_id = req_meta.req_id
        current_event = req_meta.current_event
        if req_id not in self.stored_requests:
            self.request_queue.task_done()
            return

        for group_id in req_meta.kv_cache_group_ids or [0]:
            starts = []
            ends = []
            keys = []
            block_hashes = []
            block_ids = req_meta.block_ids_by_group[group_id]
            group_block_size = self._get_block_size(group_id)
            group_block_hashes = get_block_hashes(
                req_meta.block_hashes,
                group_block_size,
                getattr(self.token_database, "hash_block_size", group_block_size),
            )

            for start, end, key, _ in self._process_tokens_with_block_ids(
                token_len,
                req_meta.block_hashes,
                block_ids,
                kv_cache_group_id=group_id,
                skip_null_blocks=self._skip_null_blocks(req_meta, group_id),
            ):
                starts.append(start)
                ends.append(end)
                keys.append(key.to_string())
                block_hashes.append(group_block_hashes[start // group_block_size])

            if not self.dcp_size > 1 and not req_meta.disable_tp_key_sharding:
                starts = starts[self.tp_rank % self.put_step :: self.put_step]
                ends = ends[self.tp_rank % self.put_step :: self.put_step]
                keys = keys[self.tp_rank % self.put_step :: self.put_step]
                block_hashes = block_hashes[self.tp_rank % self.put_step :: self.put_step]

            if not keys:
                continue

            exists_states = self.lookup(keys)
            missing_indices = [index for index, exists in enumerate(exists_states) if not exists]

            if not missing_indices:
                continue

            starts = [starts[index] for index in missing_indices]
            ends = [ends[index] for index in missing_indices]
            keys = [keys[index] for index in missing_indices]
            block_hashes = [block_hashes[index] for index in missing_indices]

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

            addrs = []
            sizes = []
            stored_events: list[BlockStored] = []
            prev_key = None
            new_block_hashes = [maybe_convert_block_hash(bh) for bh in block_hashes]
            for index, start in enumerate(starts):
                addr, size, _ = self._prepare_value(
                    start,
                    ends[index],
                    block_ids,
                    kv_cache_group_id=group_id,
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

            if current_event is not None:
                current_event.synchronize()
            self.m_store.put(keys, addrs, sizes)

            # TODO Query specific replica info to update the event
            if self.enable_kv_event and stored_events is not None:
                self.update_kv_event(stored_events)

        self.dec_stored_request(req_id)
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
        for group_id in req_meta.kv_cache_group_ids or [0]:
            block_ids = req_meta.block_ids_by_group[group_id]
            group_block_size = self._get_block_size(group_id)
            mask_num = (
                req_meta.load_spec.vllm_cached_tokens  # type: ignore[union-attr]
                // group_block_size
                * group_block_size
            )
            for start, end, key, _ in self._process_tokens_with_block_ids(
                token_len,
                req_meta.block_hashes,
                block_ids,
                mask_num,
                kv_cache_group_id=group_id,
                skip_null_blocks=self._skip_null_blocks(req_meta, group_id),
            ):
                addr, size, _ = self._prepare_value(
                    start,
                    end,
                    block_ids,
                    kv_cache_group_id=group_id,
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
        self.m_store.get(key_list_c, addr_list_c, size_list_c)
        logger.debug(
            "KV pool async recv backend get returned request=%s token_len=%d groups=%s keys=%d",
            req_id,
            token_len,
            req_meta.kv_cache_group_ids or [0],
            len(key_list_c),
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
        self.layerwise_event_starts: dict[str, set[int]] = defaultdict(set)
        self.stored_requests: dict[str, int] = defaultdict(int)
        self.done_task_lock = threading.Lock()

    def add_stored_request(self, req_id: str):
        with self.done_task_lock:
            self.stored_requests[req_id] += 1

    def dec_stored_request(self, req_id: str):
        with self.done_task_lock:
            if req_id in self.stored_requests:
                self.stored_requests[req_id] -= 1

    def delete_finished_stored_request(self, req_id: str):
        with self.done_task_lock:
            if req_id in self.stored_requests:
                del self.stored_requests[req_id]
        self.layerwise_event_starts.pop(req_id, None)

    def _record_layerwise_event_starts(self, req_meta: LayerMultiBlockReqMeta, starts: list[int]) -> None:
        if self.enable_kv_event:
            self.layerwise_event_starts[req_meta.req_id].update(starts)

    def _build_stored_events(self, req_meta: LayerMultiBlockReqMeta) -> list[BlockStored]:
        if not self.enable_kv_event or req_meta.layer_id != self.final_layer_id:
            return []
        block_size = (
            req_meta.original_block_size[req_meta.kv_cache_group_id]
            if isinstance(req_meta.original_block_size, list)
            else req_meta.original_block_size
        )
        stored_events: list[BlockStored] = []
        prev_key = None
        group_block_size = self._get_block_size(req_meta.kv_cache_group_id)
        new_block_hashes = [maybe_convert_block_hash(bh) for bh in req_meta.block_hashes]
        for start in sorted(self.layerwise_event_starts.pop(req_meta.req_id, set())):
            block_idx = start // group_block_size
            if block_idx >= len(new_block_hashes):
                continue
            block_hash = new_block_hashes[block_idx]
            end = min(start + group_block_size, len(req_meta.token_ids or []))
            token_ids = req_meta.token_ids[start:end] if req_meta.token_ids is not None else None
            stored_event = BlockStored(
                block_hashes=[block_hash],
                parent_block_hash=prev_key,
                token_ids=token_ids,
                block_size=block_size,
                lora_id=None,
                medium="cpu",
                lora_name=None,
            )
            stored_events.append(stored_event)
            prev_key = block_hash
            logger.debug("Added layerwise kv cache event '%s' to kv cache events queue", stored_event)
        return stored_events

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
            if layer_id == self.final_layer_id:
                stored_events = self._build_stored_events(req_meta)
                if stored_events:
                    self.update_kv_event(stored_events)
            if is_last_chunk:
                self.dec_stored_request(req_meta.req_id)
                self.set_finished_request(req_meta.req_id)
            self.request_queue.task_done()
            return

        key_list = []
        for key in keys:
            key_list.append(key.to_string())

        exists_states = self.lookup(key_list)
        missing_indices = [index for index, exists in enumerate(exists_states) if not exists]

        if not missing_indices:
            if layer_id == self.final_layer_id:
                stored_events = self._build_stored_events(req_meta)
                if stored_events:
                    self.update_kv_event(stored_events)
            if is_last_chunk:
                self.dec_stored_request(req_meta.req_id)
                self.set_finished_request(req_meta.req_id)
            self.request_queue.task_done()
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
        self._record_layerwise_event_starts(req_meta, starts)
        stored_events = self._build_stored_events(req_meta)
        if stored_events:
            self.update_kv_event(stored_events)

        if layer_id == self.final_layer_id and is_last_chunk:
            self.layerwise_event_starts.pop(req_meta.req_id, None)
            self.dec_stored_request(req_meta.req_id)
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
