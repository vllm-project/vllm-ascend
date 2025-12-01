import queue
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Optional

import torch
from vllm.utils import logger
from vllm.v1.core.kv_cache_utils import BlockHash

from vllm_ascend.distributed.kvpool.backend.backend import Backend

# isort: off
from vllm_ascend.distributed.kvpool.config_data import (ChunkedTokenDatabase,
                                                        LasyerMultiBlockReqMeta
                                                        )
# isort: on


class KVTransferThread(threading.Thread):

    def __init__(self, m_store: Backend, token_database: ChunkedTokenDatabase,
                 tp_rank: int, dcp_size: int, ready_event: threading.Event,
                 name: str):
        super().__init__(daemon=True, name=name)
        self.m_store = m_store
        self.ready_event = ready_event
        self.tp_rank = tp_rank
        self.dcp_size = dcp_size
        self.token_database = token_database
        self.done_task_lock = threading.Lock()
        self.request_queue: queue.Queue[Any] = queue.Queue()
        # TODO(jianzs): make this configurable
        self.executor = ThreadPoolExecutor(max_workers=32)
        self.finished_requests: set[str] = set()

    def add_request(
        self,
        req_id: str,
        token_len: int,
        block_ids: list[int],
        block_hashes: list[BlockHash],
        mask_num: int = 0,
        is_last_chunk: Optional[bool] = None,
    ) -> torch.Tensor:
        req = ({
            "req_id": req_id,
            "token_len": token_len,
            "block_ids": block_ids,
            "block_hashes": block_hashes,
            "mask_num": mask_num,
            "is_last_chunk": is_last_chunk,
        })
        self.request_queue.put(req)

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
                logger.error(f"Error in KVCacheTransferThread: {e}")

    def _handle_request(self, req_meta: dict[str, Any]):
        pass


class KVCacheStoreSendingThread(KVTransferThread):

    def __init__(self, m_store: Backend, token_database: ChunkedTokenDatabase,
                 tp_rank: int, dcp_size: int, put_step: int,
                 ready_event: threading.Event):
        super().__init__(m_store,
                         token_database,
                         tp_rank,
                         dcp_size,
                         ready_event,
                         name="KVCacheSendingThread")
        self.put_step = put_step

    def _handle_request(self, req_meta: dict[str, Any]):
        token_len = req_meta["token_len"]
        mask_num = req_meta["mask_num"]
        block_ids = req_meta["block_ids"]
        block_hashes = req_meta["block_hashes"]
        req_id = req_meta["req_id"]
        is_last_chunk = req_meta["is_last_chunk"]
        addr_list = []
        size_list = []
        key_list = []
        for start, end, key in self.token_database.process_tokens(
                token_len, block_hashes, mask_num):
            addr, size, _ = self.token_database.prepare_value(
                start, end, block_ids)
            key_list.append(key.to_string())
            addr_list.append(addr)
            size_list.append(size)
        if self.dcp_size > 1:
            self.m_store.put(key_list, addr_list, size_list)
        else:
            key_list_tp = key_list[self.tp_rank % self.put_step::self.put_step]
            addr_list_tp = addr_list[self.tp_rank %
                                     self.put_step::self.put_step]
            size_list_tp = size_list[self.tp_rank %
                                     self.put_step::self.put_step]
            if key_list_tp:
                self.m_store.put(key_list_tp, addr_list_tp, size_list_tp)
        if is_last_chunk:
            self.set_finished_request(req_id)
        self.request_queue.task_done()


class KVCacheStoreRecvingThread(KVTransferThread):

    def __init__(self, m_store: Backend, token_database: ChunkedTokenDatabase,
                 tp_rank: int, dcp_size: int, ready_event: threading.Event):
        super().__init__(m_store,
                         token_database,
                         tp_rank,
                         dcp_size,
                         ready_event,
                         name="KVCacheStoreRecvingThread")

    def _handle_request(self, req_meta: dict[str, Any]):
        token_len = req_meta["token_len"]
        mask_num = req_meta["mask_num"]
        block_ids = req_meta["block_ids"]
        req_id = req_meta["req_id"]
        block_hashes = req_meta["block_hashes"]
        addr_list = []
        size_list = []
        key_list = []
        for start, end, key in self.token_database.process_tokens(
                token_len, block_hashes, mask_num):
            addr, size, _ = self.token_database.prepare_value(
                start, end, block_ids)
            key_list.append(key.to_string())
            addr_list.append(addr)
            size_list.append(size)
        key_list_c = key_list[self.tp_rank %
                              len(key_list):] + key_list[:self.tp_rank %
                                                         len(key_list)]
        addr_list_c = addr_list[self.tp_rank %
                                len(addr_list):] + addr_list[:self.tp_rank %
                                                             len(addr_list)]
        size_list_c = size_list[self.tp_rank %
                                len(size_list):] + size_list[:self.tp_rank %
                                                             len(size_list)]
        self.m_store.get(key_list_c, addr_list_c, size_list_c)
        self.set_finished_request(req_id)
        self.request_queue.task_done()


class KVCacheStoreLayerSendingThread(KVTransferThread):

    def __init__(self, m_store: Backend, token_database: ChunkedTokenDatabase,
                 tp_rank: int, dcp_size: int, put_step: int,
                 ready_event: threading.Event, num_layers: int):
        super().__init__(m_store,
                         token_database,
                         tp_rank,
                         dcp_size,
                         ready_event,
                         name="KVCacheStoreLayerSendingThread")
        self.final_layer_id = num_layers - 1
        self.put_step = put_step

    def add_request(  # type: ignore[override]
            self, req_meta: LasyerMultiBlockReqMeta) -> torch.Tensor:
        self.request_queue.put(req_meta)

    def _handle_request(  # type: ignore[override]
            self, req_meta: LasyerMultiBlockReqMeta):
        addr_list = []
        size_list = []
        key_list = []
        for index, key in enumerate(req_meta.keys):
            addr, size = self.token_database.prepare_value_layer(
                req_meta.starts[index], req_meta.ends[index],
                req_meta.block_ids, req_meta.layer_id)
            key_list.append(key.to_string())
            addr_list.append(addr)
            size_list.append(size)
        if self.dcp_size > 1:
            self.m_store.put(key_list, addr_list, size_list)
        else:
            key_list_tp = key_list[self.tp_rank % self.put_step::self.put_step]
            addr_list_tp = addr_list[self.tp_rank %
                                     self.put_step::self.put_step]
            size_list_tp = size_list[self.tp_rank %
                                     self.put_step::self.put_step]
            if key_list_tp:
                self.m_store.put(key_list_tp, addr_list_tp, size_list_tp)
        if req_meta.layer_id == self.final_layer_id and req_meta.is_last_chunk:
            self.set_finished_request(req_meta.req_id)
        self.request_queue.task_done()


class KVCacheStoreLayerRecvingThread(KVTransferThread):

    def __init__(self, m_store: Backend, token_database: ChunkedTokenDatabase,
                 tp_rank: int, dcp_size: int, ready_event: threading.Event,
                 get_event: threading.Event):
        super().__init__(m_store,
                         token_database,
                         tp_rank,
                         dcp_size,
                         ready_event,
                         name="KVCacheStoreLayerRecvingThread")
        self.get_event = get_event

    def add_request(  # type: ignore[override]
            self, req_meta: LasyerMultiBlockReqMeta) -> torch.Tensor:
        self.request_queue.put(req_meta)

    def _handle_request(  # type: ignore[override]
            self, req_meta: LasyerMultiBlockReqMeta):
        addr_list = []
        size_list = []
        key_list = []
        for index, key in enumerate(req_meta.keys):
            addr, size = self.token_database.prepare_value_layer(
                req_meta.starts[index], req_meta.ends[index],
                req_meta.block_ids, req_meta.layer_id)
            key_list.append(key.to_string())
            addr_list.append(addr)
            size_list.append(size)
        key_list_c = key_list[self.tp_rank %
                              len(key_list):] + key_list[:self.tp_rank %
                                                         len(key_list)]
        addr_list_c = addr_list[self.tp_rank %
                                len(addr_list):] + addr_list[:self.tp_rank %
                                                             len(addr_list)]
        size_list_c = size_list[self.tp_rank %
                                len(size_list):] + size_list[:self.tp_rank %
                                                             len(size_list)]
        self.m_store.get(key_list_c, addr_list_c, size_list_c)

        self.request_queue.task_done()
        self.get_event.set()
