import queue
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import torch
from vllm.utils import logger

from vllm_ascend.distributed.mooncake_user.backend import (Backend,
                                                           MooncakeBackend)
from vllm_ascend.distributed.mooncake_user.user_config_data import (
    LasyerMultiBlockReqMeta, MooncakeEngineMetadata, MooncakeUserKey)


def get_start_end(
    num_tokens: int,
    block_size: int,
):
    starts = list(range(0, num_tokens, block_size))
    ends = [min(s + block_size, num_tokens) for s in starts]
    return starts, ends


class KVTransferThread(threading.Thread):

    def __init__(self, tp_rank: int, tp_size: int, m_store: Backend,
                 local_kv_caches_base_addr: list[int],
                 metadata: MooncakeEngineMetadata, block_len: list[int],
                 block_size: int, ready_event: threading.Event, name: str):
        super().__init__(daemon=True, name=name)
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self.m_store = m_store
        self.ready_event = ready_event
        self.kv_caches_base_addr = local_kv_caches_base_addr
        self.block_len = block_len
        self.metadata = metadata
        self.block_size = block_size
        self.done_task_lock = threading.Lock()
        # TODO(jianzs): find a better way to detect MLA.
        # self.use_mla = len(block_len) == 2

        self.request_queue: queue.Queue[Any] = queue.Queue()
        # TODO(jianzs): make this configurable
        self.executor = ThreadPoolExecutor(max_workers=32)
        self.finished_requests: set[str] = set()

        if isinstance(
                self.m_store,
                MooncakeBackend) and self.m_store.config.use_ascend_direct:
            self.use_ascend_direct = True
        else:
            self.use_ascend_direct = False

    def prepare_value(self, start: int, end: int, block_ids: list[int]):
        addr_list = []
        size_list = []
        block_id = block_ids[start // self.block_size]
        for index, base_addr in enumerate(self.kv_caches_base_addr):
            block_len = self.block_len[0]

            addr = base_addr + block_id * block_len
            length = int(block_len / self.block_size * (end - start))
            addr_list.append(addr)
            size_list.append(length)
        return addr_list, size_list, block_id

    def prepare_value_layer(self, start: int, end: int, block_ids: list[int],
                            layer_id: int):
        block_id = block_ids[start // self.block_size]

        # logger.info(f"prepare_value_layer, start: {start}, end: {end}, block_id: {block_id}, layer_id: {layer_id}")

        addr_k = self.kv_caches_base_addr[layer_id *
                                          2] + block_id * self.block_len[0]
        addr_v = self.kv_caches_base_addr[layer_id * 2 +
                                          1] + block_id * self.block_len[0]
        length = int(self.block_len[0] / self.block_size * (end - start))

        size_list = [length, length]
        addr_list = [addr_k, addr_v]
        # logger.info(f"after prepare value layer, size_list: {size_list}, addr_list: {addr_list}")
        return addr_list, size_list

    def add_request(
        self,
        req_id: str,
        uid: int,
        tokens: torch.Tensor,
        block_ids: list[int],
    ) -> torch.Tensor:
        req = ({
            "req_id": req_id,
            "uid": uid,
            "tokens": tokens,
            "block_ids": block_ids,
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
                logger.error(f"Error in KVCacheTransferThread: {e}",
                             exc_info=True)

    def _handle_request(self, req_meta: dict[str, Any]):
        pass


class KVCacheStoreSendingThread(KVTransferThread):

    def __init__(self, tp_rank: int, tp_size: int, m_store: Backend,
                 local_kv_caches_base_addr: list[int],
                 metadata: MooncakeEngineMetadata, block_len: list[int],
                 block_size: int, ready_event: threading.Event):
        super().__init__(tp_rank,
                         tp_size,
                         m_store,
                         local_kv_caches_base_addr,
                         metadata,
                         block_len,
                         block_size,
                         ready_event,
                         name="KVCacheSendingThread")

    def _handle_request(self, req_meta: dict[str, Any]):
        tokens = req_meta["tokens"]
        block_ids = req_meta["block_ids"]
        req_id = req_meta["req_id"]
        uid = req_meta["uid"]
        if self.use_ascend_direct:
            addr_list = []
            size_list = []
            key_list = []
            blockIds = []

            starts, ends = get_start_end(len(tokens), self.block_size)
            key = MooncakeUserKey(uid=uid,
                                  model_name=self.metadata.model_name,
                                  world_size=self.metadata.world_size,
                                  worker_id=self.metadata.worker_id,
                                  value_type="kv_cache")
            key_list += [key.to_string()]

            for start, end in zip(starts, ends):
                addr, size, block_id = self.prepare_value(
                    start, end, block_ids)
                # addr_list.append(addr)
                # size_list.append(size)
                addr_list += addr
                size_list += size
                blockIds.append(block_id)
            torch.npu.current_stream().synchronize()
            self.m_store.put_batch(key_list, [addr_list], [size_list],
                                   blockIds)

        self.set_finished_request(req_id)
        self.request_queue.task_done()


class KVCacheStoreRecvingThread(KVTransferThread):

    def __init__(self, tp_rank: int, tp_size: int, m_store: Backend,
                 local_kv_caches_base_addr: list[int],
                 metadata: MooncakeEngineMetadata, block_len: list[int],
                 block_size: int, ready_event: threading.Event):
        super().__init__(tp_rank,
                         tp_size,
                         m_store,
                         local_kv_caches_base_addr,
                         metadata,
                         block_len,
                         block_size,
                         ready_event,
                         name="KVCacheStoreRecvingThread")

    def _handle_request(self, req_meta: dict[str, Any]):
        tokens = req_meta["tokens"]
        block_ids = req_meta["block_ids"]
        req_id = req_meta["req_id"]
        uid = req_meta["uid"]
        if self.use_ascend_direct:
            addr_list = []
            size_list = []
            key_list = []
            blockIds = []

            starts, ends = get_start_end(len(tokens), self.block_size)
            key = MooncakeUserKey(uid=uid,
                                  model_name=self.metadata.model_name,
                                  world_size=self.metadata.world_size,
                                  worker_id=self.metadata.worker_id,
                                  value_type="kv_cache")
            key_list += [key.to_string()]

            for start, end in zip(starts, ends):
                addr, size, block_id = self.prepare_value(
                    start, end, block_ids)
                # addr_list.append(addr)
                # size_list.append(size)
                addr_list += addr
                size_list += size
                blockIds.append(block_id)
            self.m_store.get_batch(key_list, [addr_list], [size_list],
                                   blockIds)

        self.set_finished_request(req_id)
        self.request_queue.task_done()


class KVCacheStoreLayerSendingThread(KVTransferThread):

    def __init__(self, tp_rank: int, tp_size: int, m_store: Backend,
                 local_kv_caches_base_addr: list[int],
                 metadata: MooncakeEngineMetadata, block_len: list[int],
                 block_size: int, ready_event: threading.Event,
                 num_layers: int):
        super().__init__(tp_rank,
                         tp_size,
                         m_store,
                         local_kv_caches_base_addr,
                         metadata,
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
        addrs = []
        sizes = []
        for start, end in zip(req_meta.starts, req_meta.ends):
            addr, size = self.prepare_value_layer(start, end,
                                                  req_meta.block_ids,
                                                  req_meta.layer_id)
            addrs += addr
            sizes += size

        self.m_store.put_batch([req_meta.key.to_string()], [addrs], [sizes],
                               None)

        if req_meta.layer_id == self.final_layer_id:
            self.set_finished_request(req_meta.req_id)
        self.request_queue.task_done()

        logger.info(f"kvcache send size: {sum(sizes)/1048576}")


class KVCacheStoreLayerRecvingThread(KVTransferThread):

    def __init__(self, tp_rank: int, tp_size: int, m_store: Backend,
                 local_kv_caches_base_addr: list[int],
                 metadata: MooncakeEngineMetadata, block_len: list[int],
                 block_size: int, ready_event: threading.Event,
                 get_event: threading.Event):
        super().__init__(tp_rank,
                         tp_size,
                         m_store,
                         local_kv_caches_base_addr,
                         metadata,
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
        addrs = []
        sizes = []
        for start, end in zip(req_meta.starts, req_meta.ends):
            addr, size = self.prepare_value_layer(start, end,
                                                  req_meta.block_ids,
                                                  req_meta.layer_id)
            addrs += addr
            sizes += size

        self.m_store.get_batch([req_meta.key.to_string()], [addrs], [sizes],
                               None)
        self.request_queue.task_done()
        self.get_event.set()

        logger.info(f"kvcache rcv size: {sum(sizes)/1048576}")
