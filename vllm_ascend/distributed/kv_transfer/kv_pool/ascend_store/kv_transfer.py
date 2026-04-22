import queue
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import Any, List

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


def _circular_shift(lst: list, offset: int) -> list:
    if not lst or offset == 0:
        return lst
    return lst[offset:] + lst[:offset]


class KVTransferThread(threading.Thread):
    def __init__(
        self,
        m_store: Backend,
        token_database: ChunkedTokenDatabase,
        block_size: int,
        tp_rank: int,
        tp_size: int,
        dcp_size: int,
        ready_event: threading.Event,
        name: str,
    ):
        super().__init__(daemon=True, name=name)
        self.m_store = m_store
        self.ready_event = ready_event
        self.block_size = block_size
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self.dcp_size = dcp_size
        self.token_database = token_database
        self.done_task_lock = threading.Lock()
        self.request_queue: queue.Queue[Any] = queue.Queue()
        # TODO(jianzs): make this configurable
        self.executor = ThreadPoolExecutor(max_workers=32)
        self.finished_requests: set[str] = set()
        self.kv_event_lock = threading.Lock()
        self.kv_events: list[BlockStored] = []

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
            request_data = self.request_queue.get()
            if request_data is None:
                logger.warning("Received a None request!")
                self.request_queue.task_done()
                continue
            try:
                self._handle_request(request_data)
            except Exception as e:
                logger.error("Error in KVCacheTransferThread: %s", e)
                self.request_queue.task_done()

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
            logger.error(f"Remote connection failed in contains: {e}")
            return [False] * len(keys)

    def update_kv_event(self, event: list[BlockStored]):
        with self.kv_event_lock:
            self.kv_events.extend(event)

    def get_kv_events(self) -> list[BlockStored]:
        with self.kv_event_lock:
            events = self.kv_events.copy()
            self.kv_events.clear()
        return events


class KVCacheStoreSendingThread(KVTransferThread):
    def __init__(
        self,
        m_store: Backend,
        token_database: ChunkedTokenDatabase,
        block_size: int,
        tp_rank: int,
        tp_size: int,
        dcp_size: int,
        put_step: int,
        kv_role: str,
        ready_event: threading.Event,
        enable_kv_event: bool = False,
    ):
        super().__init__(
            m_store, token_database, block_size, tp_rank, tp_size, dcp_size, ready_event, name="KVCacheSendingThread"
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

    def try_finish_and_delete_stored_request(self, req_id: str) -> bool:
        with self.done_task_lock:
            if req_id in self.stored_requests and self.stored_requests[req_id] == 0:
                del self.stored_requests[req_id]
                return True
            return False

    def _handle_request(self, req_meta: ReqMeta):
        token_len = req_meta.token_len_chunk
        block_ids = req_meta.block_ids
        req_id = req_meta.req_id
        current_event = req_meta.current_event
        starts = []
        ends = []
        keys = []
        block_hashes = []
        if req_id not in self.stored_requests:
            self.request_queue.task_done()
            return

        for index, (start, end, key) in enumerate(self.token_database.process_tokens(token_len, req_meta.block_hashes)):
            starts.append(start)
            ends.append(end)
            keys.append(key.to_string())
            block_hashes.append(req_meta.block_hashes[index])

        if not self.dcp_size > 1:
            starts = starts[self.tp_rank % self.put_step :: self.put_step]
            ends = ends[self.tp_rank % self.put_step :: self.put_step]
            keys = keys[self.tp_rank % self.put_step :: self.put_step]
            block_hashes = block_hashes[self.tp_rank % self.put_step :: self.put_step]

        if not keys:
            self.dec_stored_request(req_id)
            self.request_queue.task_done()
            return

        exists_states = self.lookup(keys)
        missing_indices = [index for index, exists in enumerate(exists_states) if not exists]

        if not missing_indices:
            self.dec_stored_request(req_id)
            self.request_queue.task_done()
            return

        starts = [starts[index] for index in missing_indices]
        ends = [ends[index] for index in missing_indices]
        keys = [keys[index] for index in missing_indices]
        block_hashes = [block_hashes[index] for index in missing_indices]

        logger.debug(
            "Storing KV cache for %d out of %d blocks (missing_count=%d) for request %s",
            len(keys),
            token_len // self.block_size,
            len(missing_indices),
            req_id,
        )

        if keys:
            """
            Note: Due to a bug in ADXL, calling current_event.synchronize() may occasionally hang.
            This issue will be fixed in CANN version 8.5.rc1.
            You can manually build the master branch of the project at https://gitcode.com/cann/hixl
            to resolve this issue before the 8.5.RC1 release.
            """
            addrs = []
            sizes = []
            stored_events: list[BlockStored] = []
            prev_key = None
            new_block_hashes = [maybe_convert_block_hash(bh) for bh in block_hashes]
            for index, start in enumerate(starts):
                addr, size, _ = self.token_database.prepare_value(start, ends[index], block_ids)
                addrs.append(addr)
                sizes.append(size)

                # Create KV event
                if self.enable_kv_event:
                    token_ids = req_meta.token_ids[start : ends[index]] if req_meta.token_ids is not None else None
                    stored_event = BlockStored(
                        block_hashes=[new_block_hashes[index]],
                        parent_block_hash=prev_key,
                        token_ids=token_ids,
                        block_size=req_meta.original_block_size,
                        lora_id=None,
                        medium="cpu",
                        lora_name=None,
                    )
                    stored_events.append(stored_event)
                    prev_key = new_block_hashes[index]
                    logger.debug(f"Added kv cache event '{stored_event}' to kv cache events queue")

            if self.kv_role == "kv_consumer":
                keys, addrs, sizes = self.token_database.decode_adaptor_prefill_pp(keys, addrs, sizes)

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
        block_size: int,
        tp_rank: int,
        tp_size: int,
        dcp_size: int,
        ready_event: threading.Event,
    ):
        super().__init__(
            m_store, token_database, block_size, tp_rank, tp_size, dcp_size, ready_event, name="KVCacheStoreRecvingThread"
        )

    def _handle_request(self, req_meta: ReqMeta):
        token_len = req_meta.load_spec.token_len  # type: ignore[union-attr]
        req_id = req_meta.req_id
        mask_num = (
            req_meta.load_spec.vllm_cached_tokens  # type: ignore[union-attr]
            // self.block_size
            * self.block_size
        )
        addr_list = []
        size_list = []
        key_list = []
        for start, end, key in self.token_database.process_tokens(token_len, req_meta.block_hashes, mask_num):
            addr, size, _ = self.token_database.prepare_value(start, end, req_meta.block_ids)
            key_list.append(key.to_string())
            addr_list.append(addr)
            size_list.append(size)
        key_list_c = _circular_shift(key_list, (self.tp_rank * len(key_list)) // self.tp_size)
        addr_list_c = _circular_shift(addr_list, (self.tp_rank * len(addr_list)) // self.tp_size)
        size_list_c = _circular_shift(size_list, (self.tp_rank * len(size_list)) // self.tp_size)
        self.m_store.get(key_list_c, addr_list_c, size_list_c)
        self.set_finished_request(req_id)
        self.request_queue.task_done()


class KVCacheStoreLayerSendingThread(KVTransferThread):
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
        layer_save_finished_events: List[threading.Event],
        sync_save_events: List[torch.npu.Event],
        enable_kv_event: bool = False,
        layer_transfer_finished_events = None,
    ):
        super().__init__(
            m_store, token_database, block_size, tp_rank, tp_size, dcp_size, ready_event, name="KVCacheStoreLayerSendingThread"
        )
        self.final_layer_id = num_layers - 1
        self.put_step = put_step
        self.enable_kv_event = enable_kv_event
        self.layer_save_finished_events = layer_save_finished_events
        self.sync_save_events = sync_save_events
        self.stored_requests = defaultdict[str, int](int)
        self.layer_transfer_finished_events = layer_transfer_finished_events

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

    def try_finish_and_delete_stored_request(self, req_id: str) -> bool:
        with self.done_task_lock:
            if req_id in self.stored_requests and self.stored_requests[req_id] == 0:
                del self.stored_requests[req_id]
                return True
            return False

    def add_request(  # type: ignore[override]
        self, req_meta: ReqMeta
    ) -> torch.Tensor:
        self.request_queue.put(req_meta)

    def _handle_request(  # type: ignore[override]
        self, req_metas: LayerMultiBlockReqMeta
    ):
        if len(req_metas) == 0:
            self.request_queue.task_done()
            return
        addr_list = []
        gvas_list = []
        size_list = []
        layer_id = req_metas[0].layer_id
        for req_meta in req_metas:
            is_last_chunk = req_meta.is_last_chunk
            if req_meta.addr_list is not None and req_meta.size_list is not None:
                addr_list.extend(req_meta.addr_list[self.tp_rank % self.put_step::self.put_step])
                size_list.extend(req_meta.size_list[self.tp_rank % self.put_step::self.put_step])
            if req_meta.gvas_list is not None:
                gvas_list.extend(req_meta.gvas_list[self.tp_rank % self.put_step::self.put_step])
            if layer_id == self.final_layer_id and is_last_chunk:
                self.set_finished_request(req_meta.req_id)
            self.dec_stored_request(req_meta.req_id)
        self.sync_save_events[layer_id].synchronize()
        res = self.m_store.store.batch_copy(gvas_list, addr_list, size_list, 0)
        # wait for KV transfer (PD)
        # if self.layer_transfer_finished_events is not None:
        #     is_finish = self.layer_transfer_finished_events[layer_id].wait(timeout=10)  # try---cache
        #     if not is_finish:
        #         logger.info(f"Layerwise {layer_id} transfer failed")
        #     self.layer_transfer_finished_events[layer_id].clear()
        if res != 0:
            logger.error("Layerwise %d save batch_copy failed with return code %d", layer_id, res)
        assert not self.layer_save_finished_events[layer_id].is_set(), f"thread: {layer_id} save failed "
        self.layer_save_finished_events[layer_id].set()
        req_metas.clear()

        self.request_queue.task_done()


class KVCacheStoreLayerRecvingThread(KVTransferThread):
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
        layer_load_finished_events: List[threading.Event],
        layer_save_finished_events: List[threading.Event],
    ):
        super().__init__(
            m_store, token_database, block_size, tp_rank, tp_size, dcp_size, ready_event, name="KVCacheStoreLayerRecvingThread"
        )
        self.get_event = get_event
        self.layer_load_finished_events = layer_load_finished_events
        self.layer_save_finished_events = layer_save_finished_events

    def add_request(  # type: ignore[override]
        self, req_meta: LayerMultiBlockReqMeta
    ) -> torch.Tensor:
        self.request_queue.put(req_meta)

    def _handle_request(  # type: ignore[override]
        self, data: LayerMultiBlockReqMeta
    ):
        wait_for_save, req_metas, layer_id = data

        if wait_for_save is not None:
            is_finish = self.layer_save_finished_events[wait_for_save].wait(timeout=10)
            if not is_finish:
                logger.info("Layerwise %d save wait timed out", wait_for_save)
            self.layer_save_finished_events[wait_for_save].clear()

        if len(req_metas) == 0:
            assert not self.layer_load_finished_events[layer_id].is_set()
            self.layer_load_finished_events[layer_id].set()
            self.request_queue.task_done()
            return

        addr_list = []
        gvas_list = []
        size_list = []
        layer_id = req_metas[0].layer_id
        for req_meta in req_metas:
            if req_meta.addr_list is not None and req_meta.size_list is not None:
                addr_list.extend(req_meta.addr_list)
                size_list.extend(req_meta.size_list)
            if req_meta.gvas_list is not None:
                gvas_list.extend(req_meta.gvas_list)

        gvas_list_c = _circular_shift(gvas_list, (self.tp_rank * len(gvas_list)) // self.tp_size)
        addr_list_c = _circular_shift(addr_list, (self.tp_rank * len(addr_list)) // self.tp_size)
        size_list_c = _circular_shift(size_list, (self.tp_rank * len(size_list)) // self.tp_size)
        res = self.m_store.store.batch_copy(gvas_list_c, addr_list_c, size_list_c, 1)
        if res != 0:
            logger.error("Layerwise %d load batch_copy failed with return code %d", layer_id, res)
        assert not self.layer_load_finished_events[layer_id].is_set(), f"thread: {layer_id} load failed "
        self.layer_load_finished_events[layer_id].set()
        req_metas.clear()
        self.request_queue.task_done()
        self.get_event.set()
