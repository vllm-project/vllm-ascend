# SPDX-License-Identifier: Apache-2.0
import contextlib
import hashlib
import math
import queue
import random
import struct
import threading
import time
from collections import defaultdict
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, List, Optional, Tuple

import httpx
import msgspec
import numpy as np
import numpy.typing as npt
import torch
import zmq
from mooncake.engine import TransferEngine  # type: ignore
from vllm import envs
from vllm.config import VllmConfig, get_current_vllm_config
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1, KVConnectorMetadata, KVConnectorRole)
from vllm.distributed.parallel_state import (get_tensor_model_parallel_rank,
                                             get_tp_group, get_world_group)
from vllm.utils import get_ip, logger, make_zmq_path, make_zmq_socket
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.request import RequestStatus

import vllm_ascend.envs as envs_ascend
from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.distributed.utils import (align_memory,
                                           kv_alltoall_and_rearrange)

if TYPE_CHECKING:
    from vllm.attention.backends.abstract import AttentionMetadata
    from vllm.forward_context import ForwardContext
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.request import Request

GET_META_MSG = b"get_meta_msg"
DONE_RECVING_MSG = b"done_recving_msg"


class MooncakeAgentMetadata(msgspec.Struct, omit_defaults=True, dict=True):
    engine_id: str
    te_rpc_port: int
    kv_caches_base_addr: list[int]
    num_blocks: int


@dataclass
class ReqMeta:
    local_block_ids: list[int]
    # Not None if layer-wise is disabled
    remote_block_ids: Optional[list[int]]
    remote_host: Optional[str]
    remote_port: Optional[int]
    remote_engine_id: Optional[str]
    # Not None if layer-wise is enabled
    metaserver: Optional[str]
    remote_tp_size: Optional[int]


class DecodeMooncakeAgentMetadata(msgspec.Struct,
                                  omit_defaults=True,
                                  dict=True):
    req_id: str
    block_ids: list[int]
    host: str
    port: int
    engine_id: str
    te_rpc_port: int
    kv_caches_base_addr: list[int]
    num_blocks: int


class KVCacheTaskTracker:

    def __init__(self,
                 target_count: int = 1,
                 on_done: Callable[[str], None] = lambda x: None,
                 on_timeout: Callable[[set[str]], Any] = lambda x: None):
        super().__init__()
        self.target_count = target_count
        self.done_task_lock = threading.Lock()
        self.done_task_counts: defaultdict[str, int] = defaultdict(int)
        self.finished_requests: set[str] = set()
        # Only used in prefill node. Tracks requests whose kv blocks freeing is
        # intentionally delayed. Each entry is a tuple of (request_id,
        # timestamp). If a request remains in this queue for too long, it will
        # be force-freed.
        # Notice: In layer-wise mode, the transfer may complete before it is
        # added to delayed_free_requests when prefill node finishes forwarding.
        # Therefore we need to track requests that are removed before being added.
        self.delayed_free_requests: dict[str, float] = {}
        self.removed_delayed_free_requests: set[str] = set()
        self.on_done = on_done
        self.on_timeout = on_timeout

    def update_done_task_count(self, request_id: str):
        self.done_task_counts[request_id] += 1
        if self.done_task_counts[request_id] == self.target_count:
            with self.done_task_lock:
                self.finished_requests.add(request_id)
            self.done_task_counts.pop(request_id)
            self.on_done(request_id)

    def get_and_clear_finished_requests(self) -> set[str]:
        """
        Get and clear the requests that have been completed.
        Returns:
            A set of request IDs that have been completed.
        """
        with self.done_task_lock:
            finished_requests = self.finished_requests.copy()
            expired_requests = self._retrieve_expired_requests()
            finished_requests.update(expired_requests)
            self.finished_requests.clear()
        self.on_timeout(expired_requests)
        return finished_requests

    def add_delayed_request(self, request_id: str, delay_start_time: float):
        """Add a delayed free request, where delay_start_time is monotonic increasing."""
        with self.done_task_lock:
            if request_id in self.removed_delayed_free_requests:
                self.removed_delayed_free_requests.remove(request_id)
            else:
                self.delayed_free_requests[request_id] = delay_start_time

    def _retrieve_expired_requests(self):
        """Retrieve all expired delayed requests."""
        expired_requests: set[str] = set()
        # Free delayed requests if they exceed the timeout
        current_time = time.time()
        while self.delayed_free_requests:
            request_id, delay_start_time = next(
                iter(self.delayed_free_requests.items()))
            if (current_time - delay_start_time
                    > envs.VLLM_NIXL_ABORT_REQUEST_TIMEOUT):
                self.delayed_free_requests.pop(request_id)
                expired_requests.add(request_id)
                logger.info("Force freed request: %s", request_id)
            else:
                break
        return expired_requests

    def remove_delayed_request(self, request_id: str):
        """Remove all delayed free requests matching the given request_id."""
        with self.done_task_lock:
            if self.delayed_free_requests.pop(request_id, None) is None:
                self.removed_delayed_free_requests.add(request_id)


class KVCacheSendingLayerThread(threading.Thread):

    def __init__(self, tp_rank: int, tp_size: int, decode_tp_size: int,
                 local_engine_id: str, side_channel_host: str,
                 side_channel_port: int, metadata: MooncakeAgentMetadata,
                 ready_event: threading.Event, total_layers: int,
                 engine: TransferEngine, local_kv_base_addr: list[int],
                 block_len: list[int], use_mla: bool,
                 first_kv_cache: torch.Tensor):
        super().__init__(daemon=True, name="KVCacheSendingLayerThread")
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self.decode_tp_size = decode_tp_size
        self.local_engine_id = local_engine_id
        self.side_channel_host = side_channel_host
        self.side_channel_port = side_channel_port
        self.task_tracker = KVCacheTaskTracker(total_layers,
                                               on_done=self._post_transfer,
                                               on_timeout=self._abort_requests)
        self.send_layer_thread = SendingLayerThread(
            self.task_tracker, total_layers, engine, local_kv_base_addr,
            block_len, use_mla, self.tp_rank, first_kv_cache)
        self.ready_decode = dict[str, DecodeMooncakeAgentMetadata]()
        self.pending_decode = dict[str,
                                   list[tuple[list[int], int, torch.Tensor,
                                              torch.Tensor]]]()
        self.total_layers = total_layers
        self.lock = threading.Lock()
        self.ready_event = ready_event

    def get_and_clear_finished_requests(self) -> set[str]:
        """
        Get and clear the requests that have been completed.
        Returns:
            A set of request IDs that have been completed.
        """
        # vllm won't call us if all inference is done, so we can't do step 9 here
        return self.task_tracker.get_and_clear_finished_requests()

    def add_delayed_request(self, request_id: str, delay_start_time: float):
        return self.task_tracker.add_delayed_request(request_id,
                                                     delay_start_time)

    def run(self):
        """Run the thread to handle KV cache transfer requests."""
        self.send_layer_thread.start()
        handshake_port = self.side_channel_port + self.tp_rank
        path = make_zmq_path("tcp", self.side_channel_host, handshake_port)
        logger.info("Starting listening on path: %s", path)
        with zmq_ctx(zmq.ROUTER, path) as sock:  # type: ignore
            self.ready_event.set()
            decoder = msgspec.msgpack.Decoder(type=DecodeMooncakeAgentMetadata)
            while True:
                try:
                    frames = sock.recv_multipart()
                    if len(frames) < 2:
                        logger.error("Invalid message format: %s", frames)
                        continue

                    identity = frames[0]
                    payload = [f for f in frames[1:] if f != b""]
                    if len(payload) != 1:
                        logger.error("Invalid message format: %s", frames)
                        continue

                    metadata = decoder.decode(payload[0])
                    request_id = metadata.req_id
                    logger.debug(
                        f"Prefiller has received that request {request_id} from the decoder."
                    )
                    sock.send_multipart((identity, b"", b"ACK"))
                    self.task_tracker.remove_delayed_request(request_id)
                    with self.lock:
                        self.ready_decode[request_id] = metadata
                        pending = self.pending_decode.pop(request_id, [])
                    for local_block_ids, layer_index, key, value in pending:
                        self.send_layer_thread.send_queue.put(
                            (metadata, request_id, local_block_ids,
                             layer_index, key, value))
                except Exception as e:
                    logger.error("Failed to decode message: %s", e)

    def _post_transfer(self, request_id: str):
        with self.lock:
            decoder_meta = self.ready_decode.pop(request_id)
        path = make_zmq_path("tcp", decoder_meta.host, decoder_meta.port)
        msg_encoder = msgspec.msgpack.Encoder()
        encoded_data = msg_encoder.encode(request_id)
        with zmq_ctx(zmq.REQ, path) as sock:  # type: ignore
            ensure_zmq_send(sock, encoded_data)
            ack = sock.recv()
            if ack != b"ACK":
                raise ValueError(f"Unexpected ACK response: {ack}")

    def add_request(self, request_id: str, local_block_ids: list[int],
                    layer_index: int, key: torch.Tensor, value: torch.Tensor):
        # add request to send layer thread
        with self.lock:
            if request_id in self.ready_decode:
                self.send_layer_thread.send_queue.put(
                    (self.ready_decode[request_id], request_id,
                     local_block_ids, layer_index, key, value))
            else:
                self.pending_decode.setdefault(request_id, []).append(
                    (local_block_ids, layer_index, key, value))

    def _abort_requests(self, request_ids: set[str]):
        with self.lock:
            for request_id in request_ids:
                self.pending_decode.pop(request_id, None)


class SendingLayerThread(threading.Thread):

    def __init__(self, task_tracker: KVCacheTaskTracker, total_layers: int,
                 engine: TransferEngine, local_kv_base_addr: list[int],
                 block_len: list[int], use_mla: bool, tp_rank: int,
                 first_kv_cache: torch.Tensor):
        super().__init__(daemon=True, name="KVCacheRecvingPrefillerByeThread")
        self.send_queue = queue.Queue[tuple[DecodeMooncakeAgentMetadata, str,
                                            list[int], int, torch.Tensor,
                                            torch.Tensor]]()
        self.completion_event: Optional[threading.Event] = None
        self.completion_event_count: int
        self.task_tracker = task_tracker
        self.total_layers = total_layers
        self.local_kv_base_addr = local_kv_base_addr
        self.block_len = block_len
        self.use_mla = use_mla
        self.engine = engine
        self.tp_rank = tp_rank
        self.pd_tp_ratio = get_ascend_config().pd_tp_ratio
        self.num_head_replica = get_ascend_config().num_head_replica
        self.pd_head_ratio = get_ascend_config().pd_head_ratio
        vllm_config = get_current_vllm_config()
        max_model_len = vllm_config.scheduler_config.max_model_len
        first_kv_cache = first_kv_cache[:max_model_len]
        alignment = 2 * 1024 * 1024
        self.k_buffer = torch.zeros(
            first_kv_cache.numel() + alignment,
            dtype=first_kv_cache.dtype,
            device=first_kv_cache.device)  # 【4,1,128】-》【1000， 128】
        self.k_buffer = align_memory(self.k_buffer,
                                     alignment)[:first_kv_cache.numel()].view(
                                         -1, first_kv_cache.shape[-1])
        self.v_buffer = torch.zeros(first_kv_cache.numel() + alignment,
                                    dtype=first_kv_cache.dtype,
                                    device=first_kv_cache.device)
        self.v_buffer = align_memory(self.v_buffer,
                                     alignment)[:first_kv_cache.numel()].view(
                                         -1, first_kv_cache.shape[-1])

        for tensor in (self.k_buffer, self.v_buffer):
            assert tensor.data_ptr(
            ) % alignment == 0, "The address of the registered kv cache should be aligned to 2M"
            ret_value = self.engine.register_memory(tensor.data_ptr(),
                                                    tensor.numel())
            logger.info(
                f"Sendinglayerthread register_memory {tensor.data_ptr()} {tensor.numel()} {ret_value=}"
            )
            if ret_value != 0:
                raise RuntimeError("Mooncake memory registration failed. ")

    def run(self):
        """Run the thread to handle KV cache receiving for prefiller bye messages."""
        # send kv cache for request in send_queue
        local_rank = get_world_group().local_rank
        device = torch.device(f"npu:{local_rank}")
        torch.npu.set_device(device)
        while True:
            request = self.send_queue.get()
            self._handle_request(request)

    def _handle_request(self, request: tuple[DecodeMooncakeAgentMetadata, str,
                                             list[int], int, torch.Tensor,
                                             torch.Tensor]):
        # send kv layer to remote
        req_meta, request_id, local_block_ids, layer_index, key, value = request

        try:
            logger.debug(
                f"Starting to transfer KV cache for request {request_id}.")
            self._transfer_kv_cache(req_meta, local_block_ids, layer_index,
                                    key, value)
            logger.debug(
                f"Finished transferring KV cache for request {request_id}.")
        except Exception as e:
            logger.error("Failed to transfer KV cache for request "
                         f"{request_id}: {e}")
        finally:
            self.task_tracker.update_done_task_count(request_id)
            self.send_queue.task_done()

    def _transfer_kv_cache(self, req_meta: DecodeMooncakeAgentMetadata,
                           local_block_ids: list[int], layer_index: int, key,
                           value):
        # send kv layer to remote
        if len(local_block_ids) == 0:
            return

        remote_host = req_meta.host
        remote_te_port = req_meta.te_rpc_port
        remote_kv_base_addrs = req_meta.kv_caches_base_addr

        remote_block_ids = req_meta.block_ids
        if self.num_head_replica >= 1 and self.tp_rank % self.num_head_replica != 0:
            pass
        elif self.pd_head_ratio == 1:
            layer_local_kv_base_addr = [
                self.local_kv_base_addr[i]
                for i in [2 * layer_index, 2 * layer_index + 1]
            ]
            layer_remote_kv_base_addr = [
                remote_kv_base_addrs[i]
                for i in [2 * layer_index, 2 * layer_index + 1]
            ]
            grouped_remote_block_ids, grouped_local_block_ids = \
                group_concurrent_contiguous(remote_block_ids, local_block_ids)

            session_id = f"{remote_host}:{remote_te_port}"
            src_list, dst_list, length_list = [], [], []
            for k, (src_layer_base_addr, dst_layer_base_addr) in enumerate(
                    zip(layer_local_kv_base_addr, layer_remote_kv_base_addr)):
                block_len = self.block_len[
                    k % 2] if self.use_mla else self.block_len[0]
                for group_remote_block_id, group_local_block_id in zip(
                        grouped_remote_block_ids, grouped_local_block_ids):
                    src = src_layer_base_addr + group_local_block_id[
                        0] * block_len
                    dst = dst_layer_base_addr + group_remote_block_id[
                        0] * block_len
                    length = len(group_local_block_id) * block_len
                    src_list.append(src)
                    dst_list.append(dst)
                    length_list.append(length)
            torch.npu.synchronize()
            ret = self.engine.batch_transfer_sync_write(
                session_id, src_list, dst_list, length_list)

            if ret < 0:
                logger.error("Mooncake transfer failed for request %s",
                             req_meta.req_id)
                raise RuntimeError(f"Mooncake transfer failed, ret: {ret}")
        else:
            key = key.view(-1, key.shape[-1])
            value = value.view(-1, key.shape[-1])
            self.k_buffer[:key.shape[0]].copy_(key)  # [:4, 128] ->
            self.v_buffer[:value.shape[0]].copy_(value)

            layer_local_kv_base_addr = [
                self.k_buffer.data_ptr(),
                self.v_buffer.data_ptr()
            ]

            layer_remote_kv_base_addr = [
                remote_kv_base_addrs[i]
                for i in [2 * layer_index, 2 * layer_index + 1]
            ]

            grouped_remote_block_ids, _ = group_concurrent_contiguous(
                remote_block_ids)

            session_id = f"{remote_host}:{remote_te_port}"
            src_list, dst_list, length_list = [], [], []
            for k, (src_layer_base_addr, dst_layer_base_addr) in enumerate(
                    zip(layer_local_kv_base_addr, layer_remote_kv_base_addr)):
                src_layer_addr = src_layer_base_addr
                for group_remote_block_id in grouped_remote_block_ids:
                    block_len = self.block_len[0]
                    remote_block_len = self.block_len[0] * self.pd_head_ratio
                    src_list.append(src_layer_addr)

                    if src_layer_addr + len(
                            group_remote_block_id
                    ) * block_len > src_layer_base_addr + key.numel(
                    ) * key.element_size():
                        length = src_layer_base_addr + key.numel(
                        ) * key.element_size() - src_layer_addr
                    else:
                        length = len(group_remote_block_id) * block_len
                    length_list.append(length)

                    dst_list.append(dst_layer_base_addr +
                                    group_remote_block_id[0] *
                                    remote_block_len + length *
                                    ((self.tp_rank // self.num_head_replica) %
                                     self.pd_head_ratio))
                    src_layer_addr += length
            torch.npu.synchronize()
            ret = self.engine.batch_transfer_sync_write(
                session_id, src_list, dst_list, length_list)
            if ret < 0:
                logger.error("Mooncake transfer failed for request %s",
                             req_meta.req_id)
                raise RuntimeError(f"Mooncake transfer failed, ret: {ret}")
        if self.completion_event is not None:
            self.completion_event_count -= 1
            if self.completion_event_count == 0:
                self.completion_event.set()
                self.completion_event = None

    def add_event(self, event: threading.Event, count: int) -> None:
        self.completion_event = event
        self.completion_event_count = count


class KVCacheRecvingLayerThread(threading.Thread):

    def __init__(self, tp_rank: int, side_channel_port: int, tp_size: int,
                 local_engine_id: str, ready_event: threading.Event):
        super().__init__(daemon=True, name="KVCacheRecvingLayerThread")
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self.local_engine_id = local_engine_id
        self.side_channel_host = get_ip()
        self.side_channel_port = side_channel_port
        self.lock = threading.Lock()
        self.done_requests = set[str]()
        self.ready_event = ready_event

    def get_and_clear_finished_requests(self) -> set[str]:
        """
        Get and clear the requests that have been completed.
        Returns:
            A set of request IDs that have been completed.
        """
        with self.lock:
            finished_requests = self.done_requests
            self.done_requests = set()
        return finished_requests

    def run(self):
        """Run the thread to handle KV cache transfer requests."""
        #TODO layerwise step9
        # with zmq_ctx(zmq.ROUTER, path) as sock:  # type: ignore
        #    while True:
        #       recv_msg from prefill request send finish=
        # Listen for new requests for metadata.
        # NOTE(rob): we need each rank to have a unique port. This hack to keeps
        # us moving. We will switch when moving to etcd or where we have a
        # single ZMQ socket in the scheduler.
        handshake_port = self.side_channel_port + self.tp_rank
        path = make_zmq_path("tcp", self.side_channel_host, handshake_port)
        logger.info("Starting listening on path: %s", path)
        with zmq_ctx(zmq.ROUTER, path) as sock:  # type: ignore
            self.ready_event.set()
            decoder = msgspec.msgpack.Decoder(type=str)
            while True:
                try:
                    frames = sock.recv_multipart()
                    if len(frames) < 2:
                        logger.error("Invalid message format: %s", frames)
                        continue

                    identity = frames[0]
                    payload = [f for f in frames[1:] if f != b""]
                    if len(payload) != 1:
                        logger.error("Invalid message format: %s", frames)
                        continue

                    request_id = decoder.decode(payload[0])
                    with self.lock:
                        self.done_requests.add(request_id)
                    sock.send_multipart((identity, b"", b"ACK"))
                except Exception as e:
                    logger.error("Failed to decode message: %s", e)


class MooncakeLayerwiseConnectorMetadata(KVConnectorMetadata):

    def __init__(self):
        self.requests: dict[str, ReqMeta] = {}
        self.requests_to_send: dict[str, float] = {}

    def add_new_req(self,
                    request_id: str,
                    local_block_ids: list[int],
                    kv_transfer_params: dict[str, Any],
                    metaserver=None):
        self.requests[request_id] = ReqMeta(
            local_block_ids=local_block_ids,
            remote_block_ids=kv_transfer_params.get("remote_block_ids", None),
            remote_engine_id=kv_transfer_params["remote_engine_id"],
            remote_host=kv_transfer_params["remote_host"],
            remote_port=kv_transfer_params["remote_port"],
            metaserver=metaserver,
            remote_tp_size=kv_transfer_params.get("remote_tp_size", None),
        )


class MooncakeLayerwiseConnector(KVConnectorBase_V1):

    def __init__(self, vllm_config: VllmConfig, role: KVConnectorRole):
        assert vllm_config.kv_transfer_config is not None
        self.engine_id = vllm_config.kv_transfer_config.engine_id

        if role == KVConnectorRole.SCHEDULER:
            self.connector_scheduler: Optional[MooncakeLayerwiseConnectorScheduler] = \
                MooncakeLayerwiseConnectorScheduler(vllm_config, str(self.engine_id))
            self.connector_worker: Optional[
                MooncakeLayerwiseConnectorWorker] = None
        elif role == KVConnectorRole.WORKER:
            self.connector_scheduler = None
            self.connector_worker = MooncakeLayerwiseConnectorWorker(
                vllm_config, str(self.engine_id))

    ############################################################
    # Scheduler Side Methods
    ############################################################

    def get_num_new_matched_tokens(
            self, request: "Request",
            num_computed_tokens: int) -> tuple[int, bool]:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.get_num_new_matched_tokens(
            request, num_computed_tokens)

    def update_state_after_alloc(self, request: "Request",
                                 blocks: "KVCacheBlocks",
                                 num_external_tokens: int):
        assert self.connector_scheduler is not None
        return self.connector_scheduler.update_state_after_alloc(
            request, blocks, num_external_tokens)

    def build_connector_meta(
        self,
        scheduler_output: SchedulerOutput,
    ) -> KVConnectorMetadata:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.build_connector_meta(scheduler_output)

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, Optional[dict[str, Any]]]:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.request_finished(request, block_ids)

    def get_finished_count(self) -> Optional[int]:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.get_finished_count()

    ############################################################
    # Worker Side Methods
    ############################################################
    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        assert self.connector_worker is not None
        self.connector_worker.register_kv_caches(kv_caches)

    def get_finished(self,
                     finished_req_ids: set[str]) -> tuple[set[str], set[str]]:
        """Get the finished recving and sending requests."""
        assert self.connector_worker is not None
        return self.connector_worker.get_finished()

    def start_load_kv(self, forward_context: "ForwardContext",
                      **kwargs) -> None:
        assert self.connector_worker is not None
        assert isinstance(self._connector_metadata,
                          MooncakeLayerwiseConnectorMetadata)
        self.connector_worker.start_load_kv(self._connector_metadata)

    def wait_for_layer_load(self, layer_name: str) -> None:
        """MooncakeLayerwiseConnector does not do layerwise saving."""
        assert self.connector_worker is not None
        assert isinstance(self._connector_metadata,
                          MooncakeLayerwiseConnectorMetadata)
        self.connector_worker.wait_for_layer_load(layer_name)

    def save_kv_layer(self, layer_name: str, kv_layer: torch.Tensor,
                      attn_metadata: "AttentionMetadata", **kwargs) -> None:
        """MooncakeLayerwiseConnector does not save explicitly."""
        assert self.connector_worker is not None
        assert isinstance(self._connector_metadata,
                          MooncakeLayerwiseConnectorMetadata)
        self.connector_worker.save_kv_layer(layer_name, kv_layer,
                                            attn_metadata,
                                            self._connector_metadata)

    def wait_for_save(self):
        """MooncakeLayerwiseConnector does not save explicitly."""
        pass


class MooncakeLayerwiseConnectorScheduler:
    """Implementation of Scheduler side methods"""

    def __init__(self, vllm_config: VllmConfig, engine_id: str):
        self.vllm_config = vllm_config
        self.block_size = vllm_config.cache_config.block_size
        self.engine_id = engine_id
        logger.info("Initializing Mooncake Scheduler %s", engine_id)

        self.side_channel_host = get_ip()
        self.max_device_id = vllm_config.parallel_config.tensor_parallel_size * \
                             vllm_config.parallel_config.data_parallel_size

        # Handshake base port
        self.side_channel_port = (
            vllm_config.kv_transfer_config.kv_port +
            vllm_config.parallel_config.data_parallel_rank_local *
            vllm_config.parallel_config.tensor_parallel_size)

        # Requests that need to start recv.
        # New requests are added by update_state_after_alloc in
        # the scheduler. Used to make metadata passed to Worker.
        self._reqs_need_recv: dict[str, tuple[Request, list[int]]] = {}
        self._reqs_need_send: dict[str, float] = {}
        self._reqs_need_send_layerwise: dict[str, tuple[str, int,
                                                        list[int]]] = {}

    def get_num_new_matched_tokens(
            self, request: "Request",
            num_computed_tokens: int) -> tuple[int, bool]:
        """
        For remote prefill, pull all prompt blocks from remote
        asynchronously relative to engine execution.

        Args:
            request (Request): the request object.
            num_computed_tokens (int): the number of locally
                computed tokens for this request
        Returns:
            * the number of tokens that can be loaded from the
              external KV cache beyond what is already computed.
            * true if the external KV cache tokens will be loaded
              asynchronously (between scheduler steps).
        """

        params = request.kv_transfer_params
        logger.debug(
            "MooncakeLayerwiseConnector get_num_new_matched_tokens: "
            "num_computed_tokens=%s, kv_transfer_params=%s",
            num_computed_tokens, params)

        if params is not None and params.get("do_remote_prefill"):
            assert num_computed_tokens == 0, "Currently only support " \
                                             "prefill with num_computed_tokens == 0."
            # Assume that the request's KV cache is already fully prefilled and
            # can be fetched entirely from the prefill node.
            count = len(request.prompt_token_ids)
            if count > 0:
                return count, True

        # No remote prefill for this request.
        return 0, False

    def update_state_after_alloc(self, request: "Request",
                                 blocks: "KVCacheBlocks",
                                 num_external_tokens: int):

        params = request.kv_transfer_params
        logger.debug(
            "MooncakeLayerwiseConnector update_state_after_alloc: "
            "num_external_tokens=%s, kv_transfer_params=%s",
            num_external_tokens, params)

        if params is not None and params.get("do_remote_prefill"):
            if all(p in params for p in ("remote_engine_id", "remote_host",
                                         "remote_port")):
                local_block_ids = (blocks.get_unhashed_block_ids()
                                   if num_external_tokens > 0 else [])
                # Get unhashed blocks to pull from remote.
                self._reqs_need_recv[request.request_id] = (request,
                                                            local_block_ids)
            else:
                logger.warning(
                    "Got invalid KVTransferParams: %s. This "
                    "request will not utilize KVTransfer", params)
            params["do_remote_prefill"] = False

        # Layerwise prefiller add request need send
        if params is not None and params.get("do_remote_decode"):
            local_block_ids = (blocks.get_block_ids()[0])
            self._reqs_need_send_layerwise[request.request_id] = (
                params["metaserver"], len(request.all_token_ids),
                local_block_ids)

    def build_connector_meta(
        self,
        scheduler_output: SchedulerOutput,
    ) -> KVConnectorMetadata:
        meta = MooncakeLayerwiseConnectorMetadata()

        # Loop through scheduled reqs and convert to ReqMeta.
        for req_id, (req, block_ids) in self._reqs_need_recv.items():
            assert req.kv_transfer_params is not None
            # For the case where there are no remote blocks to pull
            # (block_ids is empty), we don't need to schedule
            # an async read on the worker side.
            meta.add_new_req(
                request_id=req_id,
                local_block_ids=block_ids,
                kv_transfer_params=req.kv_transfer_params,
            )

        # Clear the list once workers start the transfers
        self._reqs_need_recv.clear()

        cached_reqs = scheduler_output.scheduled_cached_reqs
        new_reqs = scheduler_output.scheduled_new_reqs
        for req_id, new_blocks in zip(cached_reqs.req_ids,
                                      cached_reqs.new_block_ids):
            if req_id in self._reqs_need_send_layerwise and new_blocks is not None:
                metaserver, total_tokens, block_ids = self._reqs_need_send_layerwise[
                    req_id]
                block_ids.extend(new_blocks[0])

        computed_tokens = dict(
            list(zip(cached_reqs.req_ids, cached_reqs.num_computed_tokens)) +
            [(x.req_id, x.num_computed_tokens) for x in new_reqs])
        for req_id, scheduled_tokens in scheduler_output.num_scheduled_tokens.items(
        ):
            if req_id in self._reqs_need_send_layerwise:
                metaserver, total_tokens, block_ids = self._reqs_need_send_layerwise[
                    req_id]
                current_tokens = computed_tokens.get(req_id,
                                                     0) + scheduled_tokens
                if current_tokens == total_tokens:
                    meta.add_new_req(
                        request_id=req_id,
                        local_block_ids=block_ids,
                        kv_transfer_params=defaultdict(lambda: None),
                        metaserver=metaserver)
                    self._reqs_need_send_layerwise.pop(req_id)

        meta.requests_to_send = self._reqs_need_send
        self._reqs_need_send = {}

        return meta

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, Optional[dict[str, Any]]]:
        """
        Once a request is finished, determine whether request blocks
        should be freed now or will be sent asynchronously and freed later.
        """

        params = request.kv_transfer_params
        logger.debug(
            "MooncakeLayerwiseConnector request_finished, request_status=%s, "
            "kv_transfer_params=%s", request.status, params)

        if (params is None or not params.get("do_remote_decode")
                or request.status != RequestStatus.FINISHED_LENGTH_CAPPED):
            return False, None

        computed_block_ids = block_ids
        delay_free_blocks = len(computed_block_ids) > 0
        if delay_free_blocks:
            logger.info("Delaying free of %d blocks for request %s",
                        len(computed_block_ids), request.request_id)
            self._reqs_need_send[request.request_id] = time.time()

        return delay_free_blocks, dict(
            do_remote_prefill=True,
            do_remote_decode=False,
            remote_engine_id=self.engine_id,
            remote_host=self.side_channel_host,
            remote_port=self.side_channel_port,
            remote_block_ids=computed_block_ids,
        )

    def get_finished_count(self) -> Optional[int]:
        prefill_parallel_config: dict[
            str,
            Any] = self.vllm_config.kv_transfer_config.get_from_extra_config(
                "prefill", {})

        assert "tp_size" in prefill_parallel_config.keys()
        self._prefill_tp_size = prefill_parallel_config["tp_size"]
        decode_parallel_config: dict[
            str,
            Any] = self.vllm_config.kv_transfer_config.get_from_extra_config(
                "decode", {})
        assert "tp_size" in decode_parallel_config.keys()
        self._decode_tp_size = decode_parallel_config["tp_size"]

        if self.vllm_config.model_config.use_mla:
            return self._decode_tp_size
        else:
            # TODO support mha and gqa
            return None


class MooncakeLayerwiseConnectorWorker:
    """Implementation of Worker side methods"""

    def __init__(self, vllm_config: VllmConfig, engine_id: str):
        self._get_prefill_decode_size(vllm_config)
        if self._prefill_tp_size < self._decode_tp_size:
            raise ValueError(
                f"prefill_tp_size: {self._prefill_tp_size} must be greater than"
                f" or equal to the decode_tp_size: {self._decode_tp_size}")

        if TransferEngine is None:
            raise RuntimeError("mooncake is not available")
        logger.info("Initializing Mooncake work %s", engine_id)
        self.engine = TransferEngine()

        # Metadata.
        self.completion_event: threading.Event
        self.vllm_config = vllm_config
        self.engine_id = engine_id
        self.tp_rank = get_tensor_model_parallel_rank()
        self.tp_size = vllm_config.parallel_config.tensor_parallel_size
        self.tp_group = get_tp_group()
        self.dp_rank = vllm_config.parallel_config.data_parallel_rank_local
        self.dp_size = vllm_config.parallel_config.data_parallel_size_local
        self.kv_caches: dict[str, torch.Tensor] = {}
        self.side_channel_host = get_ip()
        self.max_device_id = self.tp_size * self.dp_size
        self.kv_role = vllm_config.kv_transfer_config.kv_role
        self.total_layers = vllm_config.model_config.get_num_layers(
            vllm_config.parallel_config)

        self.executor = ThreadPoolExecutor(1)
        self.metaserver_client = httpx.Client(
            limits=httpx.Limits(max_connections=100000),
            timeout=None) if self.tp_rank == 0 else None

        # Handshake base port
        self.side_channel_port = (
            vllm_config.kv_transfer_config.kv_port +
            vllm_config.parallel_config.data_parallel_rank_local *
            vllm_config.parallel_config.tensor_parallel_size)
        self.handshake_port = self.side_channel_port + self.tp_rank
        self.sockets: dict = {}

        # get tp device id
        # TODO(kw): https://github.com/vllm-project/vllm-ascend/pull/940
        # introducing some changes
        device_ids_str = envs_ascend.PHYSICAL_DEVICES
        if device_ids_str is None:
            device_ids = list(
                range(self.dp_rank * self.tp_size,
                      (self.dp_rank + 1) * self.tp_size))
        else:
            device_ids = list(map(int, device_ids_str.split(',')))
            start_index = self.dp_rank * self.tp_size
            end_index = start_index + self.tp_size
            if len(device_ids) < end_index:
                raise ValueError(
                    f"Not enough physical devices available for DP rank {self.dp_rank}. "
                    f"Expected at least {end_index} devices, but found {len(device_ids)} "
                    "in PHYSICAL_DEVICES.")
            device_ids = device_ids[start_index:end_index]
        assert len(device_ids) > self.tp_rank  # type: ignore
        self.device_id = device_ids[self.tp_rank]  # type: ignore

        if vllm_config.kv_transfer_config.get_from_extra_config(
                'use_ascend_direct', False):
            hostname = self.side_channel_host
        else:
            hostname = f"{self.side_channel_host}:0:npu_{self.device_id}"
        self._initialize(hostname=hostname, device_name=None)
        self.te_rpc_port = self.engine.get_rpc_port()

        # Background thread for sending or receiving KV caches.
        self.kv_send_layer_thread: Optional[KVCacheSendingLayerThread] = None
        self.kv_recv_layer_thread: Optional[KVCacheRecvingLayerThread] = None

        self.vllm_config = vllm_config
        self.block_size = vllm_config.cache_config.block_size
        self.kv_caches_base_addr: list[int] = []

        self.pd_tp_ratio = get_ascend_config().pd_tp_ratio
        self.pd_head_ratio = get_ascend_config().pd_head_ratio

        self.first_kv_cache = None

    def _get_prefill_decode_size(self, vllm_config: VllmConfig):
        # get prefill tp and dp size from extra config
        prefill_parallel_config: dict[
            str, Any] = vllm_config.kv_transfer_config.get_from_extra_config(
                "prefill", {})

        assert "tp_size" in prefill_parallel_config.keys()
        self._prefill_tp_size = prefill_parallel_config["tp_size"]

        assert "dp_size" in prefill_parallel_config.keys()
        self._prefill_dp_size = prefill_parallel_config["dp_size"]

        # get decode tp and dp size from extra config
        decode_parallel_config: dict[
            str, Any] = vllm_config.kv_transfer_config.get_from_extra_config(
                "decode", {})
        assert "tp_size" in decode_parallel_config.keys()
        self._decode_tp_size = decode_parallel_config["tp_size"]
        assert "dp_size" in decode_parallel_config.keys()
        self._decode_dp_size = decode_parallel_config["dp_size"]

    def _initialize(
        self,
        hostname: str,
        device_name: Optional[str],
    ) -> None:
        """Initialize the mooncake instance."""
        device_name = device_name if device_name is not None else ""
        ret_value = self.engine.initialize(hostname, "P2PHANDSHAKE", "ascend",
                                           device_name)
        if ret_value != 0:
            raise RuntimeError(
                f"Mooncake initialization failed with ret_value: {ret_value}")

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        """Register the KV Cache data."""

        _, first_kv_cache_tuple = next(iter(kv_caches.items()))
        first_kv_cache = first_kv_cache_tuple[0]
        self.first_kv_cache = first_kv_cache

        # TODO(tms): Find a more robust way to detect and handle MLA
        self.use_mla = first_kv_cache_tuple[0].size(
            -1) != first_kv_cache_tuple[1].size(-1)
        if self.use_mla:
            # MLA case.[num_block, block_size, 1, hidden_dim]
            self.num_blocks = first_kv_cache.shape[0]
            block_rank = 3  # [block_size, latent_dim]
            block_shape_norm = first_kv_cache_tuple[0].shape[-block_rank:]
            block_shape_pe = first_kv_cache_tuple[1].shape[-block_rank:]
            self.block_len = [
                first_kv_cache[0].element_size() * math.prod(block_shape_norm),
                first_kv_cache[1].element_size() * math.prod(block_shape_pe)
            ]
            logger.info(
                "num_blocks: %s, block_shape_norm: %s, block_shape_pe: %s",
                self.num_blocks, block_shape_norm, block_shape_pe)
        else:
            # [num_block, block_size, num_head, hidden_dim]
            self.num_blocks = first_kv_cache.shape[0]
            kv_elem_size = first_kv_cache.element_size()
            block_rank = 3  # [block_size, kv_heads, head_dim]
            block_shape = first_kv_cache.shape[-block_rank:]
            self.block_len = [kv_elem_size * math.prod(block_shape)]
            logger.info("num_blocks: %s, block_shape: %s", self.num_blocks,
                        block_shape)

        logger.info("Registering KV_Caches. use_mla: %s, shape %s",
                    self.use_mla, first_kv_cache.shape)

        self.kv_caches = kv_caches
        kv_caches_base_addr = []
        for cache_or_caches in kv_caches.values():
            # Normalize to always be a list of caches
            if self.use_mla:
                for i, cache in enumerate(cache_or_caches, 0):
                    base_addr = cache.data_ptr()
                    region_len = self.num_blocks * self.block_len[i % 2]
                    kv_caches_base_addr.append(base_addr)
                    self._register(base_addr, region_len)
            else:
                cache_list = [cache_or_caches
                              ] if self.use_mla else cache_or_caches
                for cache in cache_list:
                    base_addr = cache.data_ptr()
                    region_len = self.num_blocks * self.block_len[0]
                    kv_caches_base_addr.append(base_addr)
                    self._register(base_addr, region_len)
        self.kv_caches_base_addr = kv_caches_base_addr

        # After KV Caches registered, start the sending or receiving thread.
        metadata = MooncakeAgentMetadata(
            engine_id=self.engine_id,
            te_rpc_port=self.te_rpc_port,
            kv_caches_base_addr=kv_caches_base_addr,
            num_blocks=self.num_blocks,
        )

        ready_event = threading.Event()
        if self.kv_role == 'kv_producer':
            self.kv_send_layer_thread = KVCacheSendingLayerThread(
                self.tp_rank, self.tp_size, self._decode_tp_size,
                self.engine_id, self.side_channel_host, self.side_channel_port,
                metadata, ready_event, self.total_layers, self.engine,
                kv_caches_base_addr, self.block_len, self.use_mla,
                self.first_kv_cache)
            self.kv_send_layer_thread.start()
        else:
            self.kv_recv_layer_thread = KVCacheRecvingLayerThread(
                self.tp_rank, self.side_channel_port, self.tp_size,
                self.engine_id, ready_event)
            self.kv_recv_layer_thread.start()
        ready_event.wait()

    def _register(self, ptr, length):
        logger.info(
            "Registering KV cache: ptr=0x%x, length=%d, num_blocks=%d, "
            "block_lens=%s", ptr, length, self.num_blocks, self.block_len)
        ret_value = self.engine.register_memory(ptr, length)
        if ret_value != 0:
            raise RuntimeError("Mooncake memory registration failed.")

    def _access_metaserver(self, url, message):
        self.metaserver_client.post(url, json=message)

    def get_finished(self) -> tuple[set[str], set[str]]:
        done_sending = (
            self.kv_send_layer_thread.
            get_and_clear_finished_requests(  # type: ignore[union-attr]
            ) if self.kv_role == 'kv_producer' else set())
        done_recving = (
            self.kv_recv_layer_thread.
            get_and_clear_finished_requests(  # type: ignore[union-attr]
            ) if self.kv_role == 'kv_consumer' else set())
        if self.tp_rank == 0:
            logger.debug(
                "Number of completed KV cache send requests: %d, receive "
                "requests: %d", len(done_sending), len(done_recving))
        return done_sending, done_recving

    def start_load_kv(self, metadata: MooncakeLayerwiseConnectorMetadata):
        """Start loading KV blocks from remote engine."""
        self.current_layer = 0
        if self.vllm_config.kv_transfer_config.is_kv_producer:
            for req_id, meta in metadata.requests.items():
                logger.debug(
                    f"Send request: {req_id} to proxy metaserver: {meta.metaserver}"
                )
                if self.tp_rank == 0:
                    # All parameters here should appear in the returned dict of
                    # request_finished in the scheduler side except "request_id".
                    kv_transfer_params = dict(
                        request_id=req_id,
                        do_remote_prefill=True,
                        do_remote_decode=False,
                        remote_engine_id=self.engine_id,
                        remote_host=self.side_channel_host,
                        remote_port=self.side_channel_port)

                    future = self.executor.submit(
                        self._access_metaserver,
                        url=meta.metaserver,
                        message=kv_transfer_params,
                    )

                    def handle_exception(future):
                        if future.exception():
                            logger.error(
                                f"Access metaserver fail: {future.exception()}"
                            )

                    future.add_done_callback(handle_exception)
        else:
            for req_id, meta in metadata.requests.items():
                for offset in range(self.pd_tp_ratio):
                    path = make_zmq_path(
                        "tcp", meta.remote_host, meta.remote_port +
                        self.tp_rank * self.pd_tp_ratio + offset)
                    logger.info(
                        f"Notify the prefiller: {path} that request: {req_id} from decoder is ready."
                    )
                    msg_encoder = msgspec.msgpack.Encoder()
                    docode_metadata = DecodeMooncakeAgentMetadata(
                        req_id=req_id,
                        block_ids=meta.local_block_ids,
                        port=self.handshake_port,
                        host=self.side_channel_host,
                        engine_id=self.engine_id,
                        te_rpc_port=self.te_rpc_port,
                        kv_caches_base_addr=self.kv_caches_base_addr,
                        num_blocks=self.num_blocks)
                    encoded_data = msg_encoder.encode(docode_metadata)
                    size_in_bytes = len(encoded_data)
                    logger.debug(
                        "Size of encoded Mooncake agent metadata: %d bytes",
                        size_in_bytes)
                    with zmq_ctx(zmq.REQ, path) as sock:  # type: ignore
                        ensure_zmq_send(sock, encoded_data)
                        ack = sock.recv()
                        if ack != b"ACK":
                            raise ValueError(
                                f"Unexpected ACK from prefill node: {ack}")

        if self.kv_send_layer_thread is not None:
            for req_id, delay_start_time in metadata.requests_to_send.items():
                if self.tp_rank in self._get_remote_tp_ranks_for_req(req_id):
                    self.kv_send_layer_thread.add_delayed_request(
                        req_id, delay_start_time)

    def save_kv_layer(self, layer_name: str, kv_layer: Tuple[torch.Tensor,
                                                             torch.Tensor],
                      attn_metadata: "AttentionMetadata",
                      connector_metadata: MooncakeLayerwiseConnectorMetadata,
                      **kwargs) -> None:
        """MooncakeLayerwiseConnector does not save explicitly."""
        if self.kv_role == 'kv_producer':
            if self.pd_head_ratio != 1:
                if self.current_layer != 0:
                    self.completion_event.wait()
                self.completion_event = threading.Event()
                if self.kv_send_layer_thread is not None:
                    self.kv_send_layer_thread.send_layer_thread.add_event(
                        self.completion_event,
                        len(connector_metadata.requests.keys()))

                def sort_kv_cache(input_kv: list[list[int]]):
                    return torch.cat([
                        torch.chunk(tensor, self.pd_head_ratio, dim=0)[x]
                        for x in range(self.pd_head_ratio)
                        for tensor in input_kv
                    ])

                total_block_ids = [
                    request.local_block_ids
                    for request in connector_metadata.requests.values()
                ]
                keys = [
                    kv_layer[0][block_ids].reshape(
                        -1, *kv_layer[0].shape[2:]).clone()
                    for block_ids in total_block_ids
                ]
                values = [
                    kv_layer[1][block_ids].reshape(
                        -1, *kv_layer[1].shape[2:]).clone()
                    for block_ids in total_block_ids
                ]
                key_block_size = keys[0].size(0) // len(total_block_ids[0])
                value_block_size = values[0].size(0) // len(total_block_ids[0])
                keys = sort_kv_cache(keys)  # [req1_key, req2_key]
                values = sort_kv_cache(values)
                (keys,
                 values) = kv_alltoall_and_rearrange(self.pd_head_ratio, keys,
                                                     values)
                key_start_id = 0
                value_start_id = 0
            else:
                key = None
                value = None
            for req_id, request in connector_metadata.requests.items():
                logger.info(f"Add request {req_id} to kv send layer thread. ")
                if self.pd_head_ratio != 1:
                    key_block_num = len(
                        request.local_block_ids) * key_block_size
                    value_block_num = len(
                        request.local_block_ids) * value_block_size
                    key = keys[key_start_id:key_start_id +
                               key_block_num]  #.clone().contiguous()
                    value = values[value_start_id:value_start_id +
                                   value_block_num]  #.clone().contiguous()
                    key_start_id += key_block_num
                    value_start_id += value_block_num
                if self.kv_send_layer_thread is not None:
                    self.kv_send_layer_thread.add_request(
                        request_id=req_id,
                        local_block_ids=request.local_block_ids,
                        layer_index=self.current_layer,
                        key=key,
                        value=value)
            self.current_layer += 1

    def wait_for_layer_load(self, layer_name: str) -> None:
        pass

    def _get_remote_tp_rank(self, req_id: str) -> int:
        return self._get_remote_tp_ranks_for_req(req_id)[self.tp_rank]

    def _get_remote_tp_ranks_for_req(self, req_id: str) -> list[int]:
        if self._prefill_tp_size == self._decode_tp_size:
            return list(range(self._prefill_tp_size))

        seed = string_to_int64_hash(req_id)
        rand = random.Random(seed)
        sampled_nums = rand.sample(range(self._prefill_tp_size),
                                   self._decode_tp_size)
        return sampled_nums


@contextlib.contextmanager
def zmq_ctx(socket_type: Any,
            addr: str) -> Iterator[zmq.Socket]:  # type: ignore
    """Context manager for a ZMQ socket"""

    if socket_type not in (zmq.ROUTER, zmq.REQ, zmq.DEALER):  # type: ignore
        raise ValueError(f"Unexpected socket type: {socket_type}")

    ctx: Optional[zmq.Context] = None  # type: ignore
    try:
        ctx = zmq.Context()  # type: ignore
        yield make_zmq_socket(ctx=ctx,
                              path=addr,
                              socket_type=socket_type,
                              bind=socket_type == zmq.ROUTER)  # type: ignore
    finally:
        if ctx is not None:
            ctx.destroy(linger=0)


def group_concurrent_contiguous(
    src: List[int],
    dst: List[int] = []
) -> Tuple[List[npt.NDArray[np.int64]], List[npt.NDArray[np.int64]]]:
    """Vectorised NumPy implementation."""
    if not dst:
        src_only_indices: npt.NDArray[np.int64] = np.array(src, dtype=np.int64)

        if src_only_indices.size == 0:
            return [], []

        brk = np.where((np.diff(src_only_indices) != 1))[0] + 1
        src_groups = np.split(src_only_indices, brk)
        src_groups = [g.tolist() for g in src_groups]

        return src_groups, []

    else:
        src_indices: npt.NDArray[np.int64] = np.array(src, dtype=np.int64)
        dst_indices: npt.NDArray[np.int64] = np.array(dst, dtype=np.int64)

        if src_indices.size == 0:
            return [], []

        brk = np.where((np.diff(src_indices) != 1)
                       | (np.diff(dst_indices) != 1))[0] + 1
        src_groups = np.split(src_indices, brk)
        dst_groups = np.split(dst_indices, brk)

        src_groups = [g.tolist() for g in src_groups]
        dst_groups = [g.tolist() for g in dst_groups]

        return src_groups, dst_groups


def string_to_int64_hash(input_str):
    """
    Hash the string using SHA-256 and convert it into an int64 integer.
    """
    hashed_bytes = hashlib.sha256(input_str.encode("utf-8")).digest()
    trunked_bytes = hashed_bytes[:8]
    uint64_value = struct.unpack("<Q", trunked_bytes)[0]
    return uint64_value


def ensure_zmq_send(
        socket: zmq.Socket,  # type: ignore
        data: bytes,
        max_retries: int = 3):
    retries_left = max_retries
    while True:
        try:
            socket.send(data)
            return
        except zmq.ZMQError as e:  # type: ignore
            retries_left -= 1
            if retries_left > 0:
                logger.warning(
                    f"Send failed: {e}, retrying... ({retries_left} "
                    "attempts left)")
                time.sleep(0.1)
            else:
                logger.error(f"Send failed after all retries: {e}")
                raise RuntimeError(f"Failed to send data after {max_retries} "
                                   f"retries: {e}")


def ensure_zmq_recv(
        socket: zmq.Socket,  # type: ignore
        poller: zmq.Poller,  # type: ignore
        timeout: float = 1.0,
        max_retries: int = 3) -> bytes:
    retries_left = max_retries
    while True:
        try:
            if dict(poller.poll(int(timeout * 1000))):  # milliseconds
                data = socket.recv()
                return data
            else:
                raise zmq.ZMQError("Receive timeout")  # type: ignore
        except zmq.ZMQError as e:  # type: ignore
            retries_left -= 1
            if retries_left > 0:
                logger.warning(f"Receive failed: {e}, retrying... "
                               f"({retries_left} attempts left)")
                time.sleep(0.1)
            else:
                logger.error(f"Receive failed after all retries: {e}")
                raise RuntimeError(
                    f"Failed to receive data after {max_retries} "
                    f"retries: {e}")
