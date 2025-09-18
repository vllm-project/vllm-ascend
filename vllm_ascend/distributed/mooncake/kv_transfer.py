import queue
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Optional

import torch
from vllm.utils import logger

from vllm_ascend.distributed.mooncake.config_data import (
    ChunkedTokenDatabase, LasyerMultiBlockReqMeta)
from vllm_ascend.distributed.mooncake.mooncake_store import Mooncakestore


class KVTransferThread(threading.Thread):

    def __init__(self, tp_rank: int, tp_size: int, m_store: Mooncakestore,
                 local_kv_caches_base_addr: list[int],
                 token_database: ChunkedTokenDatabase, block_len: list[int],
                 block_size: int, ready_event: threading.Event, name: str):
        super().__init__(daemon=True, name=name)
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self.m_store = m_store
        self.ready_event = ready_event
        self.kv_caches_base_addr = local_kv_caches_base_addr
        self.block_len = block_len
        self.token_database = token_database
        self.block_size = block_size
        self.done_task_lock = threading.Lock()
        # TODO(jianzs): find a better way to detect MLA.
        self.use_mla = len(block_len) == 2

        self.request_queue: queue.Queue[Any] = queue.Queue()
        # TODO(jianzs): make this configurable
        self.executor = ThreadPoolExecutor(max_workers=32)
        self.finished_requests: set[str] = set()

    def prepare_value(self, start: int, end: int, block_ids: list[int]):
        addr_list = []
        size_list = []
        block_id = block_ids[start // self.block_size]
        for index, base_addr in enumerate(self.kv_caches_base_addr):
            block_len = (self.block_len[index % 2]
                         if self.use_mla else self.block_len[0])

            addr = base_addr + block_id * block_len
            length = int(block_len / self.block_size * (end - start))
            addr_list.append(addr)
            size_list.append(length)
        return addr_list, size_list, block_id

    def prepare_value_layer(self, start: int, end: int, block_ids: list[int],
                            layer_id: int):
        block_id = block_ids[start // self.block_size]
        if self.use_mla:
            addr_k = self.kv_caches_base_addr[layer_id *
                                              2] + block_id * self.block_len[0]
            addr_v = self.kv_caches_base_addr[layer_id * 2 +
                                              1] + block_id * self.block_len[1]
            length_k = int(self.block_len[0] / self.block_size * (end - start))
            length_v = int(self.block_len[1] / self.block_size * (end - start))
            size_list = [length_k, length_v]
        else:
            addr_k = self.kv_caches_base_addr[layer_id *
                                              2] + block_id * self.block_len[0]
            addr_v = self.kv_caches_base_addr[layer_id * 2 +
                                              1] + block_id * self.block_len[0]
            length = int(self.block_len[0] / self.block_size * (end - start))
            size_list = [length, length]
        addr_list = [addr_k, addr_v]
        return addr_list, size_list

    def add_request(
        self,
        req_id: str,
        tokens: torch.Tensor,
        block_ids: list[int],
        mask: Optional[torch.Tensor] = None,
        is_last_chunk: Optional[bool] = None,
    ) -> torch.Tensor:
        req = ({
            "req_id": req_id,
            "tokens": tokens,
            "block_ids": block_ids,
            "mask": mask,
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

    def __init__(self, tp_rank: int, tp_size: int, m_store: Mooncakestore,
                 local_kv_caches_base_addr: list[int],
                 token_database: ChunkedTokenDatabase, block_len: list[int],
                 block_size: int, ready_event: threading.Event):
        super().__init__(tp_rank,
                         tp_size,
                         m_store,
                         local_kv_caches_base_addr,
                         token_database,
                         block_len,
                         block_size,
                         ready_event,
                         name="KVCacheSendingThread")

    def _handle_request(self, req_meta: dict[str, Any]):
        tokens = req_meta["tokens"]
        mask = req_meta["mask"]
        block_ids = req_meta["block_ids"]
        req_id = req_meta["req_id"]
        is_last_chunk = req_meta["is_last_chunk"]
        torch.npu.current_stream().synchronize()
        for start, end, key in self.token_database.process_tokens(
                tokens, mask):
            addr, size, _ = self.prepare_value(start, end, block_ids)
            self.m_store.put(key, addr, size)
        if is_last_chunk:
            self.set_finished_request(req_id)
        self.request_queue.task_done()


class KVCacheStoreRecvingThread(KVTransferThread):

    def __init__(self, tp_rank: int, tp_size: int, m_store: Mooncakestore,
                 local_kv_caches_base_addr: list[int],
                 token_database: ChunkedTokenDatabase, block_len: list[int],
                 block_size: int, ready_event: threading.Event):
        super().__init__(tp_rank,
                         tp_size,
                         m_store,
                         local_kv_caches_base_addr,
                         token_database,
                         block_len,
                         block_size,
                         ready_event,
                         name="KVCacheStoreRecvingThread")

    def _handle_request(self, req_meta: dict[str, Any]):
        tokens = req_meta["tokens"]
        mask = req_meta["mask"]
        block_ids = req_meta["block_ids"]
        req_id = req_meta["req_id"]
        for start, end, key in self.token_database.process_tokens(
                tokens, mask):
            addr, size, _ = self.prepare_value(start, end, block_ids)
            self.m_store.get(key, addr, size)
        self.set_finished_request(req_id)
        self.request_queue.task_done()


class KVCacheStoreLayerSendingThread(KVTransferThread):

    def __init__(self, tp_rank: int, tp_size: int, m_store: Mooncakestore,
                 local_kv_caches_base_addr: list[int],
                 token_database: ChunkedTokenDatabase, block_len: list[int],
                 block_size: int, ready_event: threading.Event,
                 num_layers: int):
        super().__init__(tp_rank,
                         tp_size,
                         m_store,
                         local_kv_caches_base_addr,
                         token_database,
                         block_len,
                         block_size,
                         ready_event,
                         name="KVCacheStoreLayerSendingThread")
        self.final_layer_id = num_layers - 1

    def add_request(  # type: ignore[override]
            self, req_meta: LasyerMultiBlockReqMeta) -> torch.Tensor:
        self.request_queue.put(req_meta)

    def _handle_request(  # type: ignore[override]
            self, req_meta: LasyerMultiBlockReqMeta):
        torch.npu.current_stream().synchronize()
        for index, key in enumerate(req_meta.keys):
            addr, size = self.prepare_value_layer(req_meta.starts[index],
                                                  req_meta.ends[index],
                                                  req_meta.block_ids,
                                                  req_meta.layer_id)
            self.m_store.put(key, addr, size)
        if req_meta.layer_id == self.final_layer_id:
            self.set_finished_request(req_meta.req_id)
        self.request_queue.task_done()


class KVCacheStoreLayerRecvingThread(KVTransferThread):

    def __init__(self, tp_rank: int, tp_size: int, m_store: Mooncakestore,
                 local_kv_caches_base_addr: list[int],
                 token_database: ChunkedTokenDatabase, block_len: list[int],
                 block_size: int, ready_event: threading.Event,
                 get_event: threading.Event):
        super().__init__(tp_rank,
                         tp_size,
                         m_store,
                         local_kv_caches_base_addr,
                         token_database,
                         block_len,
                         block_size,
                         ready_event,
                         name="KVCacheStoreLayerRecvingThread")
        self.get_event = get_event

    def add_request(  # type: ignore[override]
            self, req_meta: LasyerMultiBlockReqMeta) -> torch.Tensor:
        self.request_queue.put(req_meta)

    def _handle_request(  # type: ignore[override]
            self, req_meta: LasyerMultiBlockReqMeta):
        for index, key in enumerate(req_meta.keys):
            addr, size = self.prepare_value_layer(req_meta.starts[index],
                                                  req_meta.ends[index],
                                                  req_meta.block_ids,
                                                  req_meta.layer_id)
            self.m_store.get(key, addr, size)
        self.request_queue.task_done()
        self.get_event.set()
