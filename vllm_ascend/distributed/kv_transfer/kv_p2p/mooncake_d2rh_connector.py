# SPDX-License-Identifier: Apache-2.0
import contextlib
import copy
import hashlib
import logging
import math
import os
import queue
import random
import struct
import threading
import time
from collections import OrderedDict, defaultdict, deque
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypedDict

import msgspec
import numpy as np
import numpy.typing as npt
import torch
import torch_npu
import zmq
from mooncake.engine import TransferEngine  # type: ignore
from vllm import envs
from vllm.config import VllmConfig
from vllm.distributed import get_pcp_group
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorHandshakeMetadata,
    KVConnectorMetadata,
    KVConnectorRole,
)
from vllm.distributed.parallel_state import (
    get_decode_context_model_parallel_rank,
    get_decode_context_model_parallel_world_size,
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    get_tp_group,
)
from vllm.distributed.utils import get_pp_indices
from vllm.logger import logger
from vllm.utils.network_utils import get_ip, make_zmq_path, make_zmq_socket
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.request import RequestStatus

from vllm_ascend import envs as ascend_envs
from vllm_ascend.ascend_config import get_ascend_config, init_ascend_config
from vllm_ascend.distributed.kv_transfer.utils.mooncake_transfer_engine import global_te
from vllm_ascend.distributed.kv_transfer.utils.utils import get_transfer_timeout_value
from vllm_ascend.utils import enable_custom_op, is_vl_model

# isort: off
if TYPE_CHECKING:
    from vllm.v1.attention.backend import AttentionMetadata  # type: ignore
    from vllm.forward_context import ForwardContext
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.request import Request
# isort: on

GET_META_MSG = b"get_meta_msg"
DONE_RECVING_MSG = b"done_recving_msg"
READY_SCHEDULER = b"ready_scheduler"
STAGING_FULL = b"staging_full"
START_PULL = b"START_PULL"

# ZMQ ports for D2RH (hop1) and scheduler ready signaling (hop1 done).
# Layout matches side_channel_port + device_index used by KV handshake:
#   port = BASE + dp_rank * tp_size * pp_size * pcp_size + (pp_rank + pcp_rank) * tp_size + tp_rank
# TP=1 / DP0 / PP0 / PCP0 → D2RH=8100, READY=8200 (same as legacy hardcoded values).
D2RH_ZMQ_PORT_BASE = 8100
SCHEDULER_READY_ZMQ_PORT_BASE = 8200


class RemotePortInfo(TypedDict):
    num: int
    host: str


class MooncakeAgentMetadata(msgspec.Struct, omit_defaults=True, dict=True):
    engine_id: str
    te_rpc_port: int
    kv_caches_base_addr: list[int]
    num_blocks: int
    local_ip: str = ""


@dataclass
class ReqMeta:
    local_block_ids: list[int]
    num_external_tokens: int
    remote_block_ids: list[int]
    remote_host: str
    remote_port: int
    remote_engine_id: str
    remote_request_id: str
    remote_pcp_size: int
    remote_dcp_size: int
    remote_ptp_size: int | None
    remote_multi_nodes_meta_mapping: dict[str, dict[str, Any]]
    num_prompt_blocks: int


@dataclass
class SizedDict(OrderedDict):
    def __init__(self, max_size=16000, *args, **kwargs):
        self.max_size = max_size
        super().__init__(*args, **kwargs)

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        if len(self) > self.max_size:
            self.popitem(last=False)

    def __getitem__(self, key):
        try:
            return super().__getitem__(key)
        except KeyError:
            value: dict[int, list[int]] = {}
            self[key] = value
            return value


class KVCacheTaskTracker:
    def __init__(self):
        super().__init__()

        self.done_task_lock = threading.Lock()
        self.finished_requests: set[str] = set()
        # Only used in prefill node. Tracks requests whose kv blocks freeing is
        # intentionally delayed. Each entry is a tuple of (request_id,
        # timestamp). If a request remains in this queue for too long, it will
        # be force-freed.
        self.delayed_free_requests: OrderedDict[str, float] = OrderedDict()
        self.reqs_to_process: set[str] = set()

    def add_req_to_process(self, request_id: str):
        self.reqs_to_process.add(request_id)

    def add_not_transfer_request(self, request_id: str):
        with self.done_task_lock:
            self.finished_requests.add(request_id)
            self.reqs_to_process.discard(request_id)

    def update_done_task_count(self, request_id: str):
        with self.done_task_lock:
            if request_id in self.reqs_to_process:
                self.finished_requests.add(request_id)
                self.reqs_to_process.discard(request_id)
                self.delayed_free_requests.pop(request_id, None)
            else:
                logger.error(
                    "MooncakeConnector finish req not in reqs to process."
                    "If it is a P node, this request may have been force freed."
                )

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
        return finished_requests

    def add_delayed_request(self, request_id: str, delay_start_time: float):
        """Add a delayed free request."""
        with self.done_task_lock:
            if request_id in self.reqs_to_process:
                self.delayed_free_requests[request_id] = delay_start_time

    def _retrieve_expired_requests(self):
        """Retrieve all expired delayed requests."""
        expired_requests: set[str] = set()
        # Free delayed requests if they exceed the timeout
        current_time = time.time()
        while self.delayed_free_requests:
            request_id = next(iter(self.delayed_free_requests))
            delay_start_time = self.delayed_free_requests[request_id]
            if current_time - delay_start_time > envs.VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT:
                self.delayed_free_requests.popitem(last=False)
                self.reqs_to_process.discard(request_id)
                expired_requests.add(request_id)
                logger.info("Force freed request: %s", request_id)
            else:
                break
        return expired_requests


class KVCacheSendingThread(threading.Thread):
    def __init__(
        self,
        vllm_config: VllmConfig,
        tp_rank: int,
        prefill_tp_size: int,
        local_engine_id: str,
        side_channel_host: str,
        side_channel_port: int,
        metadata: MooncakeAgentMetadata,
        ready_event: threading.Event,
        kv_caches: dict[str, Any],
        pcp_rank: int,
    ):
        super().__init__(daemon=True, name="KVCacheSendingThread")
        self.tp_rank = tp_rank
        self.prefill_tp_size = prefill_tp_size
        self.pp_rank = get_pp_group().rank_in_group
        self.pp_size = vllm_config.parallel_config.pipeline_parallel_size
        self.tp_size = get_tensor_model_parallel_world_size()
        self.local_engine_id = local_engine_id
        self.side_channel_host = side_channel_host
        self.side_channel_port = side_channel_port
        self.metadata = metadata
        self.ready_event = ready_event
        self.kv_caches = kv_caches
        self.pcp_rank = pcp_rank
        self.port_send_num: dict[str, int] = {}

        self.task_tracker = KVCacheTaskTracker()

    def get_and_clear_finished_requests(self) -> set[str]:
        """
        Get and clear the requests that have been completed.
        Returns:
            A set of request IDs that have been completed.
        """
        return self.task_tracker.get_and_clear_finished_requests()

    def add_not_transfer_request(self, request_id: str):
        self.task_tracker.add_not_transfer_request(request_id)

    def add_delayed_request(self, request_id: str, delay_start_time: float):
        return self.task_tracker.add_delayed_request(request_id, delay_start_time)

    def run(self):
        """Run the thread to handle KV cache transfer requests."""
        try:
            # Listen for new requests for metadata. NOTE(rob): we need each rank
            # to have a unique port. This hack to keeps us moving. We will
            # switch when moving to etcd or where we have a single ZMQ socket in
            # the scheduler.
            device_index = self.pp_rank * self.tp_size + self.tp_rank + self.pcp_rank * self.prefill_tp_size
            handshake_port = self.side_channel_port + device_index
            path = make_zmq_path("tcp", self.side_channel_host, handshake_port)
            logger.info("Starting listening on path: %s", path)
            with zmq_ctx(zmq.ROUTER, path) as sock:  # type: ignore
                self.ready_event.set()
                self.run_busy_loop(sock)
        except Exception as e:
            logger.error("Mooncake KVCacheSendingThread exception: %s", e, exc_info=True)

    def run_busy_loop(self, sock: zmq.Socket):  # type: ignore
        encoder = msgspec.msgpack.Encoder()
        encoded_data = encoder.encode(self.metadata)
        size_in_bytes = len(encoded_data)
        logger.debug("Size of encoded MooncakeAgentMetadata: %s bytes", str(size_in_bytes))

        decoder = msgspec.msgpack.Decoder(type=tuple)
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

                msg = decoder.decode(payload[0])
                if msg[0] == GET_META_MSG:
                    sock.send_multipart((identity, b"", encoded_data))
                elif msg[0] == DONE_RECVING_MSG:
                    logger.debug("Got DONE_RECVING_MSG for request %s", msg[1])
                    request_id = msg[1]
                    remote_port_send_num = msg[2]
                    if remote_port_send_num:
                        if request_id not in self.port_send_num:
                            self.port_send_num[request_id] = 0
                        self.port_send_num[request_id] += 1
                        device_index = self.pp_rank * self.tp_size + self.tp_rank + self.pcp_rank * self.prefill_tp_size
                        handshake_port = self.side_channel_port + device_index
                        if self.port_send_num[request_id] >= remote_port_send_num[handshake_port]["num"]:
                            self.task_tracker.update_done_task_count(request_id)
                            del self.port_send_num[request_id]
                    else:
                        self.task_tracker.update_done_task_count(request_id)
                    # Acknowledge the request completion.
                    while True:
                        try:
                            # Send ACK to the sender.
                            sock.send_multipart((identity, b"", b"ACK"), flags=zmq.NOBLOCK)  # type: ignore
                            break
                        except zmq.Again:  # type: ignore
                            # If the socket is not ready, retry sending.
                            logger.debug("Socket not ready, retrying to send ACK for request %s", msg[1])
                            time.sleep(0.01)
                else:
                    logger.error("Connection listener got unexpected message %s", msg)
            except Exception as e:
                logger.error("Connection listener got exception %s: %s", type(e), e)


class cpu_kvcache_manager:
    def __init__(self, num_blocks):
        self.free_queue = deque(range(num_blocks))
        self.used_set = set()

    def alloc_blocks(self, req_need_blocks):
        now_block = len(self.free_queue)
        if req_need_blocks > len(self.free_queue):
            logger.info(f"Dont have enough blocks {req_need_blocks=} {now_block=}")
            return None
        else:
            local_block_ids = [self.free_queue.popleft() for _ in range(req_need_blocks)]
            self.used_set.update(local_block_ids)
            logger.info(f"Alloc blocks{local_block_ids}")
            return local_block_ids

    def free_blocks(self, req_block_ids):
        flat_block_ids = [block_id for sublist in req_block_ids for block_id in sublist]
        self.free_queue.extend(flat_block_ids)
        self.used_set.difference_update(flat_block_ids)


class HostListeningThread(threading.Thread):
    def __init__(
        self,
        all_requests: set[str],
        decode_tp_size: int,
        scheduler_ready_port: int,
    ):
        super().__init__(daemon=True, name="HostListeningThread")
        self.port_send_num: dict[str, int] = {}
        self.all_requests = all_requests
        self.decode_tp_size = max(decode_tp_size, 1)
        self.scheduler_ready_port = scheduler_ready_port
        self.ready_count: dict[str, int] = defaultdict(int)

        self.task_tracker = KVCacheTaskTracker()
        self.ready_request = set()
        self.ready_lock = threading.Lock()
        self.host_ip = get_ip()
        self.encoder = msgspec.msgpack.Encoder()
        self.decoder = msgspec.msgpack.Decoder(MooncakeAgentMetadata)
        self.remote_sockets_lock = threading.Lock()
        self.remote_sockets: dict[  # type: ignore
            str, deque[zmq.Socket]
        ] = defaultdict(  # type: ignore
            deque
        )
        self.remote_poller = zmq.Poller()  # type: ignore
        self.timeout = 1.0  # seconds

    def get_and_clear_finished_requests(self) -> set[str]:
        """
        Get and clear the requests that have been completed.
        Returns:
            A set of request IDs that have been completed.
        """
        return self.task_tracker.get_and_clear_finished_requests()

    def run(self):
        """Run the thread to handle KV cache transfer requests."""
        try:
            # Listen for new requests for metadata. NOTE(rob): we need each rank
            # to have a unique port. This hack to keeps us moving. We will
            # switch when moving to etcd or where we have a single ZMQ socket in
            # the scheduler.
            handshake_port = self.scheduler_ready_port
            path = make_zmq_path("tcp", self.host_ip, handshake_port)
            logger.info("Starting scheduler ready listener on path: %s", path)
            with zmq_ctx(zmq.ROUTER, path) as sock:  # type: ignore
                self.run_busy_loop(sock)
        except Exception as e:
            logger.error("Mooncake KVCacheSendingThread exception: %s", e, exc_info=True)

    def run_busy_loop(self, sock: zmq.Socket):  # type: ignore
        # encoder = msgspec.msgpack.Encoder()
        # encoded_data = encoder.encode(self.metadata)
        # size_in_bytes = len(encoded_data)
        # logger.debug("Size of encoded MooncakeAgentMetadata: %s bytes", str(size_in_bytes))

        decoder = msgspec.msgpack.Decoder(type=tuple)
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

                msg = decoder.decode(payload[0])
                if msg[0] == READY_SCHEDULER:
                    logger.debug("Got READY_SCHEDULER for request %s", msg[1])
                    request_id = msg[1]
                    with self.ready_lock:
                        self.ready_count[request_id] += 1
                        if self.ready_count[request_id] >= self.decode_tp_size:
                            self.ready_request.add(request_id)
                            del self.ready_count[request_id]
                    # Acknowledge the request completion.
                    while True:
                        try:
                            # Send ACK to the sender.
                            sock.send_multipart((identity, b"", b"ACK"), flags=zmq.NOBLOCK)  # type: ignore
                            break
                        except zmq.Again:  # type: ignore
                            # If the socket is not ready, retry sending.
                            logger.debug("Socket not ready, retrying to send ACK for request %s", msg[1])
                            time.sleep(0.01)
                elif msg[0] == STAGING_FULL:
                    request_id = msg[1]
                    logger.info(
                        "CPU staging full for request %s, remove from all_requests to allow retry",
                        request_id,
                    )
                    with self.ready_lock:
                        self.all_requests.discard(request_id)
                        self.ready_count.pop(request_id, None)
                    while True:
                        try:
                            sock.send_multipart((identity, b"", b"ACK"), flags=zmq.NOBLOCK)  # type: ignore
                            break
                        except zmq.Again:  # type: ignore
                            logger.debug("Socket not ready, retrying to send ACK for STAGING_FULL %s", msg[1])
                            time.sleep(0.01)
                else:
                    logger.error("Connection listener got unexpected message %s", msg)
            except Exception as e:
                logger.error("Connection listener got exception %s: %s", type(e), e)


class D2RHThread(threading.Thread):
    def __init__(
        self,
        cpu_ptrs: list[int],
        block_len: list[int],
        engine,
        cpu_kvcache_manager,
        remote_local_block_map,
        vllm_config,
        d2rh_handshake_port: int,
        scheduler_ready_port: int,
        tp_rank: int = 0,
    ):
        super().__init__(daemon=True, name=f"D2RHThread-TP{tp_rank}")
        self.kv_caches_base_addr: dict[str, dict[int, list[int]]] = SizedDict()
        self.remote_te_port: dict[str, dict[int, int]] = SizedDict()
        self.request_queue: queue.Queue[Any] = queue.Queue()
        self.remote_sockets_lock = threading.Lock()
        self.remote_sockets: dict[  # type: ignore
            str, deque[zmq.Socket]
        ] = defaultdict(  # type: ignore
            deque
        )
        self.timeout = 1.0
        self.remote_poller = zmq.Poller()
        self.encoder = msgspec.msgpack.Encoder()
        self.decoder = msgspec.msgpack.Decoder(MooncakeAgentMetadata)
        self.cpu_ptrs = cpu_ptrs
        self.block_len = block_len
        self.engine = engine
        self.host_ip = get_ip()
        self.cpu_kvcache_manager = cpu_kvcache_manager
        self.remote_local_block_map = remote_local_block_map
        self.vllm_config = vllm_config
        self.d2rh_handshake_port = d2rh_handshake_port
        self.scheduler_ready_port = scheduler_ready_port
        self.tp_rank = tp_rank

    def add_request(
        self,
        request_id,
        remote_request_id,
        remote_host,
        remote_engine_id,
        remote_handshake_ports,
        remote_port_base,
        remote_multi_nodes_meta_mapping,
        grouped_remote_block_ids,
        grouped_local_block_ids,
    ):
        self.request_queue.put(
            {
                "request_id": request_id,
                "remote_request_id": remote_request_id,
                "remote_host": remote_host,
                "remote_engine_id": remote_engine_id,
                "remote_handshake_ports": remote_handshake_ports,
                "remote_port_base": remote_port_base,
                "remote_multi_nodes_meta_mapping": remote_multi_nodes_meta_mapping,
                "grouped_remote_block_ids": grouped_remote_block_ids,
                "grouped_local_block_ids": grouped_local_block_ids,
            }
        )

    # def run(self):
    #     """Run the thread to handle KV cache transfer requests."""
    #     try:
    #         # Listen for new requests for metadata. NOTE(rob): we need each rank
    #         # to have a unique port. This hack to keeps us moving. We will
    #         # switch when moving to etcd or where we have a single ZMQ socket in
    #         # the scheduler.
    #         handshake_port = 8100
    #         path = make_zmq_path("tcp", self.host_ip, handshake_port)
    #         logger.info("Starting listening on path: %s", path)
    #         with zmq_ctx(zmq.ROUTER, path) as sock:  # type: ignore
    #             self.run_busy_loop(sock)
    #     except Exception as e:
    #         logger.error("Mooncake KVCacheSendingThread exception: %s", e, exc_info=True)
    #     while True:
    #         # try:
    #         request_data = self.request_queue.get()
    #         if request_data is None:
    #             logger.warning("Received a None request!")
    #             self.request_queue.task_done()
    #             continue
    #         self._handle_request(request_data)
    #         # except Exception as e:
    #         #     logger.error(f"Error in KVCacheTransferThread: {e}")

    def run(self):
        """Run the thread to handle KV cache transfer requests."""
        try:
            handshake_port = self.d2rh_handshake_port
            path = make_zmq_path("tcp", self.host_ip, handshake_port)
            logger.info(
                "Starting D2RH listener on tp_rank=%s path: %s",
                self.tp_rank,
                path,
            )

            def zmq_listener_worker():
                try:
                    with zmq_ctx(zmq.ROUTER, path) as sock:  # type: ignore
                        self.run_busy_loop(sock)
                except Exception as e:
                    logger.error("ZMQ background listener worker crashed: %s", e, exc_info=True)

            bg_zmq_thread = threading.Thread(
                target=zmq_listener_worker,
                name=f"D2RH-ZMQListener-TP{self.tp_rank}",
                daemon=True,
            )
            bg_zmq_thread.start()
            logger.info("ZMQ background listener thread started successfully.")

        except Exception as e:
            logger.error("Failed to initialize ZMQ path in D2RHThread: %s", e, exc_info=True)
            return

        logger.info("D2RHThread queue consumer loop started.")
        while True:
            try:
                request_data = self.request_queue.get()
                if request_data is None:
                    logger.warning("Received an unexpected None request in worker loop! Skipping...")
                    continue
                self._handle_request(request_data)

            except Exception as e:
                logger.error("Error occurred while processing request in D2RHThread loop: %s", e, exc_info=True)
            finally:
                self.request_queue.task_done()

    def run_busy_loop(self, sock: zmq.Socket):  # type: ignore
        # encoder = msgspec.msgpack.Encoder()
        # encoded_data = encoder.encode(self.metadata)
        # size_in_bytes = len(encoded_data)
        # logger.debug("Size of encoded MooncakeAgentMetadata: %s bytes", str(size_in_bytes))

        decoder = msgspec.msgpack.Decoder(type=tuple)
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

                msg = decoder.decode(payload[0])
                if msg[0] == START_PULL:
                    logger.info("Got START_PULL for request %s", msg[1])
                    request_id = msg[1]
                    pull_ack = b"ACK"
                    try:
                        params = msg[2]
                        remote_host = params["remote_host"]
                        remote_engine_id = params["remote_engine_id"]
                        remote_block_ids = params.get("remote_block_ids") or []
                        remote_request_id = params.get("remote_request_id", request_id)
                        remote_handshake_ports = params.get("remote_handshake_ports")
                        if not remote_handshake_ports:
                            remote_port_base = params["remote_port"]
                            remote_handshake_port = params.get("remote_handshake_port", remote_port_base)
                            remote_handshake_ports = [remote_handshake_port]
                        else:
                            remote_port_base = params["remote_port"]
                        cpu_block_ids = self.cpu_kvcache_manager.alloc_blocks(len(remote_block_ids))
                        if cpu_block_ids is not None:
                            block_map = dict(zip(remote_block_ids, cpu_block_ids))
                            self.remote_local_block_map[request_id] = block_map
                            if remote_request_id != request_id:
                                self.remote_local_block_map[remote_request_id] = block_map
                            logger.info(
                                "[===] CPU staging blocks allocated: %s -> %s",
                                remote_block_ids,
                                cpu_block_ids,
                            )
                            self.add_request(
                                request_id=request_id,
                                remote_request_id=remote_request_id,
                                remote_host=remote_host,
                                remote_engine_id=remote_engine_id,
                                remote_handshake_ports=remote_handshake_ports,
                                remote_port_base=remote_port_base,
                                remote_multi_nodes_meta_mapping=params.get("remote_multi_nodes_meta_mapping"),
                                grouped_remote_block_ids=remote_block_ids,
                                grouped_local_block_ids=cpu_block_ids,
                            )
                        else:
                            logger.warning(
                                "CPU staging full for request %s, need %d blocks",
                                request_id,
                                len(remote_block_ids),
                            )
                            pull_ack = STAGING_FULL
                    except Exception as e:
                        logger.error(
                            "Failed to handle START_PULL for request %s: %s",
                            request_id,
                            e,
                            exc_info=True,
                        )
                        pull_ack = STAGING_FULL

                    while True:
                        try:
                            sock.send_multipart((identity, b"", pull_ack), flags=zmq.NOBLOCK)  # type: ignore
                            break
                        except zmq.Again:  # type: ignore
                            logger.debug("Socket not ready, retrying START_PULL response for %s", request_id)
                            time.sleep(0.01)
                else:
                    logger.error("Connection listener got unexpected message %s", msg)
            except Exception as e:
                logger.error("Connection listener got exception %s: %s", type(e), e)

    def _handle_request(self, req_meta: dict[str, Any]):
        self._transfer_kv_cache(req_meta)

    def _transfer_kv_cache(self, req_meta: dict[str, Any]):
        request_id = req_meta["request_id"]
        remote_request_id = req_meta.get("remote_request_id", request_id)
        remote_engine_id = req_meta["remote_engine_id"]
        remote_block_ids = req_meta["grouped_remote_block_ids"]
        cpu_block_ids = req_meta["grouped_local_block_ids"]
        remote_handshake_ports = req_meta.get("remote_handshake_ports") or [req_meta.get("remote_port_base", 0)]
        remote_port_base = req_meta.get("remote_port_base", remote_handshake_ports[0])
        remote_multi_nodes_meta_mapping = req_meta.get("remote_multi_nodes_meta_mapping")
        default_remote_host = req_meta["remote_host"]

        if remote_block_ids and isinstance(remote_block_ids[0], int):
            grouped_remote_block_ids, grouped_cpu_block_ids = group_concurrent_contiguous(
                remote_block_ids, cpu_block_ids
            )
        else:
            grouped_remote_block_ids, grouped_cpu_block_ids = remote_block_ids, cpu_block_ids

        tp_num_need_pulls = len(remote_handshake_ports)
        transfer_ok = True
        block_length = len(self.block_len)

        for inner_offset, remote_handshake_port in enumerate(remote_handshake_ports):
            remote_host, port_engine_id = resolve_remote_host_for_handshake_port(
                remote_port_base,
                remote_handshake_port,
                default_remote_host,
                remote_engine_id,
                remote_multi_nodes_meta_mapping,
            )
            if (
                port_engine_id not in self.kv_caches_base_addr
                or remote_handshake_port not in self.kv_caches_base_addr[port_engine_id]
            ):
                self._get_remote_metadata(remote_host, remote_handshake_port)
            remote_kv_caches_base_addrs = self.kv_caches_base_addr[port_engine_id][remote_handshake_port]
            te_rpc_port = self.remote_te_port[port_engine_id][remote_handshake_port]
            session_id = f"{remote_host}:{te_rpc_port}"
            local_list, peer_list, length_list = [], [], []
            # logger.info(
            #     "Prefill KV base addrs port=%s inner_offset=%s: %s",
            #     remote_handshake_port,
            #     inner_offset,
            #     remote_kv_caches_base_addrs,
            # )
            for layer_idx, remote_layer_base in enumerate(remote_kv_caches_base_addrs):
                if layer_idx >= len(self.cpu_ptrs):
                    logger.error(
                        "CPU staging layer count %d < prefill layer count %d",
                        len(self.cpu_ptrs),
                        len(remote_kv_caches_base_addrs),
                    )
                    break
                cpu_layer_base = self.cpu_ptrs[layer_idx]
                block_len = self.block_len[layer_idx % block_length]
                inner_block_len = block_len // tp_num_need_pulls
                for remote_block_id, cpu_block_id in zip(grouped_remote_block_ids, grouped_cpu_block_ids):
                    local_list.append(cpu_layer_base + cpu_block_id[0] * block_len + inner_offset * inner_block_len)
                    peer_list.append(remote_layer_base + remote_block_id[0] * block_len)
                    length_list.append(inner_block_len * len(remote_block_id))
            # logger.info(
            #     "D2RH transfer %s local=%s peer=%s lengths=%s blocks=%s->%s",
            #     session_id,
            #     local_list,
            #     peer_list,
            #     length_list,
            #     grouped_remote_block_ids,
            #     grouped_cpu_block_ids,
            # )
            ret = self.engine.batch_transfer_sync_read(session_id, local_list, peer_list, length_list)
            if ret < 0:
                logger.error(
                    "D2RH transfer failed for request %s, session %s, port %s",
                    remote_request_id,
                    session_id,
                    remote_handshake_port,
                )
                transfer_ok = False
                break
            self._send_done_recv_signal(remote_request_id, remote_host, remote_handshake_port)

        if transfer_ok:
            self.send_pull_done(request_id)

    def _send_done_recv_signal(
        self,
        request_id: str,
        remote_host: str,
        remote_handshake_port: int,
    ) -> None:
        """Tell prefill that KV has been pulled into CPU staging."""
        logger.debug(
            "D2RH sending done recving signal for request %s to %s:%d",
            request_id,
            remote_host,
            remote_handshake_port,
        )
        sock: zmq.Socket | None = None  # type: ignore
        try:
            sock = self._get_remote_socket(remote_host, remote_handshake_port)
            data_bytes = self.encoder.encode((DONE_RECVING_MSG, request_id, {}))
            ensure_zmq_send(sock, data_bytes, f"{remote_host}:{remote_handshake_port}")
            resp = ensure_zmq_recv(
                sock, self.remote_poller, f"{remote_host}:{remote_handshake_port}", timeout=self.timeout
            )
            logger.debug("D2RH received response for request %s: %s", request_id, resp.decode("utf-8"))
            if resp != b"ACK":
                logger.error(
                    "D2RH failed to receive ACK for request %s from %s:%d",
                    request_id,
                    remote_host,
                    remote_handshake_port,
                )
                raise RuntimeError(f"Failed to receive ACK, resp: {resp.decode('utf-8')}")
        except RuntimeError as e:
            if isinstance(sock, zmq.Socket):  # type: ignore
                sock.close()
                sock = None
                logger.warning(
                    "D2RH unexpected error occurred in socket, %s, closing the original channel",
                    e,
                )
        finally:
            if sock is not None:
                self._return_remote_socket(sock, remote_host, remote_handshake_port)
                logger.debug("Returned socket to pool for %s:%d", remote_host, remote_handshake_port)

    def send_pull_done(self, request_id):
        sock: zmq.Socket | None = None  # type: ignore
        try:
            port = self.scheduler_ready_port
            sock = self._get_remote_socket(self.host_ip, port)
            data_bytes = self.encoder.encode((READY_SCHEDULER, request_id))
            ensure_zmq_send(sock, data_bytes, f"{self.host_ip}:{port}")
            resp = ensure_zmq_recv(sock, self.remote_poller, f"{self.host_ip}:{port}", timeout=self.timeout)
            logger.debug(f"Received response for request {request_id}: {resp.decode('utf-8')}")
            if resp != b"ACK":
                logger.error("Failed to receive ACK for request %s from %s:%d", request_id, self.host_ip, port)
                raise RuntimeError(f"Failed to receive ACK, resp: {resp.decode('utf-8')}")
        except RuntimeError as e:
            if isinstance(sock, zmq.Socket):  # type: ignore
                sock.close()
                sock = None
                logger.warning(f"Unexpected error occurred in socket, {e}, closing the original channel")
        finally:
            if sock is not None:
                self._return_remote_socket(sock, self.host_ip, port)
                logger.debug("Returned socket to pool for %s:%d", self.host_ip, port)

    def _get_remote_metadata(self, remote_host: str, remote_handshake_port: int) -> None:
        """Get the metadata from the remote host."""
        sock: zmq.Socket | None = None  # type: ignore
        try:
            sock = self._get_remote_socket(remote_host, remote_handshake_port)
            ensure_zmq_send(sock, self.encoder.encode((GET_META_MSG, "")), f"{remote_host}:{remote_handshake_port}")
            metadata_bytes = ensure_zmq_recv(sock, self.remote_poller, f"{remote_host}:{remote_handshake_port}")
            agent_meta = self.decoder.decode(metadata_bytes)
            engine_id = agent_meta.engine_id
            # assert engine_id != self.local_engine_id, (
            #     f"Conflict engine id {engine_id} with local engine id {self.local_engine_id}."
            # )
            self.kv_caches_base_addr[engine_id][remote_handshake_port] = agent_meta.kv_caches_base_addr
            self.remote_te_port[engine_id][remote_handshake_port] = agent_meta.te_rpc_port
            logger.info(f"[===] base_add te_rpc_port {agent_meta.kv_caches_base_addr} {agent_meta.te_rpc_port}")
            logger.info(f"[===] _get_remote_metadata {self.kv_caches_base_addr} {self.remote_te_port}")
        finally:
            if sock is not None:
                self._return_remote_socket(sock, remote_host, remote_handshake_port)
                logger.debug("Returned socket to pool for %s:%d", remote_host, remote_handshake_port)

    def _get_remote_socket(self, remote_host: str, remote_handshake_port: int) -> zmq.Socket:  # type: ignore
        """Get a socket to the remote host."""
        remote_path = make_zmq_path("tcp", remote_host, remote_handshake_port)
        with self.remote_sockets_lock:
            if self.remote_sockets[remote_path]:
                return self.remote_sockets[remote_path].popleft()

            ctx = zmq.Context()  # type: ignore
            sock = make_zmq_socket(
                ctx=ctx,
                path=remote_path,
                socket_type=zmq.REQ,  # type: ignore
                bind=False,
            )
            sock.setsockopt(
                zmq.SNDTIMEO,  # type: ignore
                int(self.timeout * 1000),
            )
            self.remote_poller.register(sock, zmq.POLLIN)  # type: ignore
            return sock

    def _return_remote_socket(
        self,
        sock: zmq.Socket,  # type: ignore
        remote_host: str,
        remote_handshake_port: int,
    ) -> None:
        """Return the remote socket to the pool."""
        remote_path = make_zmq_path("tcp", remote_host, remote_handshake_port)
        with self.remote_sockets_lock:
            self.remote_sockets[remote_path].append(sock)


class KVCacheRecvingThread(threading.Thread):
    def __init__(
        self,
        tp_rank: int,
        tp_size: int,
        _prefill_pp_size: int,
        engine: TransferEngine,
        local_engine_id: str,
        local_handshake_port: int,
        side_channel_port: int,
        local_kv_caches_base_addr: list[int],
        block_len: list[int],
        ready_event: threading.Event,
        vllm_config: VllmConfig,
        kv_caches: dict[str, Any],
        te_rpc_port,
        cpu_ptr,
        cpu_kv_caches_base_addr,
        cpu_te_rpc_port,
        cpu_kvcache_manager,
        use_cpu_staging: bool = False,
        remote_local_block_map: dict[str, dict[int, int]] = {},
        prefill_pp_layer_partition: str | None = None,
    ):
        super().__init__(daemon=True, name="KVCacheRecvingThread")
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self._prefill_pp_size = _prefill_pp_size
        self.local_engine_id = local_engine_id
        self.local_handshake_port = local_handshake_port
        self.side_channel_port = side_channel_port
        self.engine = engine
        self.ready_event = ready_event
        self.te_rpc_port = te_rpc_port
        self.cpu_ptrs = cpu_ptr
        self.cpu_kv_caches_base_addr = cpu_kv_caches_base_addr
        self.cpu_te_rpc_port = cpu_te_rpc_port
        self.cpu_kvcache_manager = cpu_kvcache_manager
        self.use_cpu_staging = use_cpu_staging
        self.remote_local_block_map = remote_local_block_map

        self.kv_caches = kv_caches
        self.kv_caches_base_addr: dict[str, dict[int, list[int]]] = SizedDict()
        self.kv_caches_base_addr[local_engine_id][local_handshake_port] = local_kv_caches_base_addr
        self.remote_te_port: dict[str, dict[int, int]] = SizedDict()
        self.block_len = block_len
        # TODO(jianzs): find a better way to detect MLA.
        self.use_mla = len(block_len) == 2

        self.request_queue: queue.Queue[Any] = queue.Queue()
        self.executor = ThreadPoolExecutor(max_workers=32)

        self.task_tracker = KVCacheTaskTracker()

        self.encoder = msgspec.msgpack.Encoder()
        self.decoder = msgspec.msgpack.Decoder(MooncakeAgentMetadata)
        self.remote_sockets_lock = threading.Lock()
        self.remote_sockets: dict[  # type: ignore
            str, deque[zmq.Socket]
        ] = defaultdict(  # type: ignore
            deque
        )
        self.remote_poller = zmq.Poller()  # type: ignore
        self.timeout = 1.0  # seconds

        self.vllm_config = vllm_config
        self.model_config = self.vllm_config.model_config
        self.block_size = self.vllm_config.cache_config.block_size
        self.num_layers = self.model_config.hf_text_config.num_hidden_layers
        self.pp_layer_indices = {
            rank: get_prefill_pp_indices(self.num_layers, rank, self._prefill_pp_size, prefill_pp_layer_partition)
            for rank in range(self._prefill_pp_size)
        }
        if not is_vl_model(vllm_config):
            if self.use_mla:
                self.k_head_dim = self.model_config.hf_text_config.kv_lora_rank
                self.v_head_dim = self.model_config.hf_text_config.qk_rope_head_dim
                self.num_kv_heads = 1
            else:
                self.k_head_dim = self.model_config.hf_text_config.head_dim
                self.v_head_dim = self.model_config.hf_text_config.head_dim
                self.num_kv_heads = max(self.model_config.hf_text_config.num_key_value_heads // self.tp_size, 1)
        self.proc_not_transfer_request: dict[str, bool] = {}

    def add_request(
        self,
        request_id: str,
        remote_request_id: str,
        local_block_ids: list[int],
        remote_block_ids: list[int],
        remote_engine_id: str,
        remote_host: str,
        remote_handshake_port: int,
        offset: int,
        tp_num_need_pulls: int,
        remote_port_send_num: dict[int, RemotePortInfo] | None = None,
        all_task_done: bool = False,
    ):
        """Add a new request to the queue for processing."""
        if remote_port_send_num is None:
            remote_port_send_num = {}
        logger.debug(f"Adding request {request_id} to the queue.")
        self.request_queue.put(
            {
                "request_id": request_id,
                "local_block_ids": local_block_ids,
                "remote_block_ids": remote_block_ids,
                "remote_engine_id": remote_engine_id,
                "remote_request_id": remote_request_id,
                "remote_host": remote_host,
                "remote_handshake_port": remote_handshake_port,
                "offset": offset,
                "tp_num_need_pulls": tp_num_need_pulls,
                "remote_port_send_num": remote_port_send_num,
                "all_task_done": all_task_done,
            }
        )

    def get_and_clear_finished_requests(self) -> set[str]:
        """
        Get and clear the requests that have been completed.
        Returns:
            A set of request IDs that have been completed.
        """
        return self.task_tracker.get_and_clear_finished_requests()

    def get_and_clear_invalid_block_ids(self) -> set[int]:
        return set()

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
        request_id = req_meta["request_id"]
        remote_request_id = req_meta["remote_request_id"]
        remote_host = req_meta["remote_host"]
        remote_handshake_port = req_meta["remote_handshake_port"]
        remote_port_send_num = req_meta["remote_port_send_num"]
        all_task_done = req_meta["all_task_done"]

        try:
            logger.debug(f"Starting to transfer KV cache for request {remote_request_id}.")
            self._transfer_kv_cache(req_meta)
            logger.debug(f"Finished transferring KV cache for request {remote_request_id}.")
        except Exception as e:
            logger.error(f"Failed to transfer KV cache for request {remote_request_id}: {e}", exc_info=True)
        finally:
            if not self.use_cpu_staging:
                self._send_done_signal_to_free_remote_port(remote_request_id, remote_host, remote_port_send_num)
            if all_task_done:
                self.task_tracker.update_done_task_count(request_id)
                if request_id in self.proc_not_transfer_request:
                    del self.proc_not_transfer_request[request_id]
            self.request_queue.task_done()
            if not self.use_cpu_staging:
                self._send_done_recv_signal(remote_request_id, remote_host, remote_handshake_port, remote_port_send_num)
            elif len(req_meta.get("local_block_ids", [])) == 0:
                # Full prefix cache hit: no hop2 pull, but P still needs the release signal.
                self._send_done_recv_signal(remote_request_id, remote_host, remote_handshake_port, remote_port_send_num)

    def _send_done_signal_to_free_remote_port(
        self, request_id: str, remote_host: str, remote_port_send_num: dict[int, RemotePortInfo]
    ):
        if self.side_channel_port != self.local_handshake_port or not remote_port_send_num:
            return
        if request_id not in self.proc_not_transfer_request:
            self.proc_not_transfer_request[request_id] = True
        if self.proc_not_transfer_request[request_id]:
            for remote_port in remote_port_send_num:
                if remote_port_send_num[remote_port]["num"] == 0:
                    remote_host_ = remote_port_send_num[remote_port]["host"]
                    self._send_done_recv_signal(request_id, remote_host_, remote_port, remote_port_send_num)
            self.proc_not_transfer_request[request_id] = False

    def _transfer_kv_cache(self, req_meta: dict[str, Any]):
        """Handle a KV cache transfer request."""
        remote_request_id = req_meta["remote_request_id"]
        remote_block_ids = req_meta["remote_block_ids"]
        local_block_ids = req_meta["local_block_ids"]
        remote_engine_id = req_meta["remote_engine_id"]
        remote_host = req_meta["remote_host"]
        remote_handshake_port = req_meta["remote_handshake_port"]
        offset = req_meta["offset"]
        tp_num_need_pulls = req_meta["tp_num_need_pulls"]

        # Full prefix cache hit: do not need to read remote blocks, just notify
        # P worker that we have the blocks we need.
        num_local_blocks = len(local_block_ids)
        if num_local_blocks == 0:
            return

        num_remote_blocks = len(remote_block_ids)
        assert num_local_blocks <= num_remote_blocks
        if num_local_blocks < num_remote_blocks:
            remote_block_ids = remote_block_ids[-num_local_blocks:]

        # Check if we have the remote metadata cached.
        if (
            remote_engine_id not in self.kv_caches_base_addr
            or remote_handshake_port not in self.kv_caches_base_addr[remote_engine_id]
        ):
            self._get_remote_metadata(remote_host, remote_handshake_port)

        if tp_num_need_pulls == 1:
            grouped_remote_block_ids, grouped_local_block_ids = group_concurrent_contiguous(
                remote_block_ids, local_block_ids
            )
            logger.info(
                f"[===]{grouped_remote_block_ids=}{grouped_local_block_ids=}{remote_block_ids=}{local_block_ids=}"
            )
        else:
            remote_block_ids = list(map(lambda x: [x], remote_block_ids))
            local_block_ids = list(map(lambda x: [x], local_block_ids))
            grouped_remote_block_ids, grouped_local_block_ids = remote_block_ids, local_block_ids
        num_transfer_groups = len(grouped_remote_block_ids)
        # tp_num_need_pulls: number of KV caches each Decode node needs to pull from each PP stage
        # Due to GQA, different KV heads are distributed across different ranks, so there are offsets
        # indicating which KV head to pull
        global_offset = offset  # Global offset of request across all ranks
        prefill_pp_rank = offset // tp_num_need_pulls  # PP rank where current request resides
        inner_offset = offset % tp_num_need_pulls  # Offset within each PP stage

        first_layer_index, end_layer_index = self.pp_layer_indices[prefill_pp_rank]
        # support MTP layer kv transfer
        if self.vllm_config.speculative_config is not None:
            # all MTP layer use the same kv cache layer, so only need to transfer once
            if prefill_pp_rank == self._prefill_pp_size - 1:
                end_layer_index = end_layer_index + 1
        num_cache_per_layer = len(list(self.kv_caches.values())[0])  # Number of KV caches per layer
        local_kv_caches_base_addrs = self.kv_caches_base_addr[self.local_engine_id][self.local_handshake_port][
            first_layer_index * num_cache_per_layer : end_layer_index * num_cache_per_layer
        ]
        logger.debug(f"transfer kv cache first_layer_index:{first_layer_index} , end_layer_index:{end_layer_index}")
        num_blocks = len(local_block_ids)
        local_host = get_ip()
        cpu_block_map = self.remote_local_block_map.get(remote_request_id) or self.remote_local_block_map.get(
            req_meta["request_id"]
        )
        if cpu_block_map is None:
            raise RuntimeError(
                f"CPU staging block map missing for request {remote_request_id}. "
                "Ensure D2RHThread finished before decode pull."
            )
        cpu_kv_caches_base_addrs = self.cpu_kv_caches_base_addr[
            first_layer_index * num_cache_per_layer : end_layer_index * num_cache_per_layer
        ]
        session_id = f"{local_host}:{self.cpu_te_rpc_port}"

        req_start_time = time.perf_counter()
        local_list, peer_list, length_list = [], [], []
        block_length = len(self.block_len)
        for k, (npu_layer_base, cpu_layer_base) in enumerate(zip(local_kv_caches_base_addrs, cpu_kv_caches_base_addrs)):
            block_len = self.block_len[k % block_length]
            inner_block_len = block_len // tp_num_need_pulls
            for remote_block_id, npu_block_id in zip(grouped_remote_block_ids, grouped_local_block_ids):
                cpu_block_id = cpu_block_map[remote_block_id[0]]
                local_list.append(npu_layer_base + npu_block_id[0] * block_len + inner_offset * inner_block_len)
                peer_list.append(cpu_layer_base + cpu_block_id * block_len + inner_offset * inner_block_len)
                length_list.append(inner_block_len * len(npu_block_id))
                logger.info(f"[===] {cpu_block_id=} {grouped_remote_block_ids=}")
        ret = self.engine.batch_transfer_sync_read(session_id, local_list, peer_list, length_list)
        if req_meta.get("all_task_done") and self.cpu_kvcache_manager is not None:
            # cpu_blocks_to_free = list({cpu_block_map[r[0]] for r in grouped_remote_block_ids})
            # cpu_blocks_to_free = list({
            #     cpu_block_map[p]
            #     for r in grouped_remote_block_ids
            #     # 先在 if 里面通过 (p := r[0]) 赋值，再在后面 print(p)
            #     if (p := r[0]) is not None and print(f"DEBUG r[0]: {p}") is None
            # })
            cpu_blocks_to_free = list(cpu_block_map.values())
            self.cpu_kvcache_manager.free_blocks([cpu_blocks_to_free])
            self.remote_local_block_map.pop(remote_request_id, None)
            self.remote_local_block_map.pop(req_meta["request_id"], None)
        if ret < 0:
            logger.error("Mooncake transfer failed for request %s", req_meta["remote_request_id"])
            raise RuntimeError(f"Mooncake transfer failed, ret: {ret}")

        req_end_time = time.perf_counter()
        req_transfer_elapsed = (req_end_time - req_start_time) * 1000
        logger.info(
            "KV cache transfer for request %s took %.2f ms (%d groups,"
            " %d blocks). local_ip %s local_device_id %s remote_session_id %s",
            remote_request_id,
            req_transfer_elapsed,
            num_transfer_groups,
            num_blocks,
            get_ip(),
            self.tp_rank,
            session_id,
        )

        # Determine if the current position is the offset position at the end of
        # the KV transmission.
        is_kv_transfer_end = global_offset == tp_num_need_pulls * self._prefill_pp_size - 1
        need_cat_cache = tp_num_need_pulls > 1 and is_kv_transfer_end
        need_nz_cache = get_ascend_config().enable_kv_nz and is_kv_transfer_end
        use_fused_op = ascend_envs.VLLM_ASCEND_FUSION_OP_TRANSPOSE_KV_CACHE_BY_BLOCK
        if need_nz_cache or need_cat_cache:
            # use fused op to reformat kv cache, we keep original implementation to provide ability to disable it.
            if use_fused_op and enable_custom_op():
                if need_cat_cache:
                    # the fused op only support cat GQA/MHA kv cache by head
                    self.reformat_kv_cache_with_fused_op(grouped_local_block_ids, tp_num_need_pulls)
                if need_nz_cache:
                    # maybe use fused op to reformat kv nz too in the future.
                    self.reformat_kv_cache(grouped_local_block_ids, tp_num_need_pulls, False, need_nz_cache)
            else:
                self.reformat_kv_cache(grouped_local_block_ids, tp_num_need_pulls, need_cat_cache, need_nz_cache)

    def reformat_kv_cache_with_fused_op(self, block_ids: list[list[int]], tp_num_need_pulls: int):
        # Get necessary parameters
        k_cache = list(self.kv_caches.values())[0][0]
        device = k_cache.device
        head_dim = self.model_config.hf_text_config.head_dim
        block_size = self.vllm_config.cache_config.block_size
        num_kv_head = max(self.model_config.hf_text_config.num_key_value_heads // self.tp_size, 1)
        layers = len(self.kv_caches)
        flat_block_ids = [item for sublist in block_ids for item in sublist]
        block_ids_tensor = torch.tensor(flat_block_ids, dtype=torch.int64, device=device)

        k_caches = []
        v_caches = []
        for _, (k_cache_layer, v_cache_layer) in self.kv_caches.items():
            k_caches.append(k_cache_layer)
            v_caches.append(v_cache_layer)

        torch.ops._C_ascend.transpose_kv_cache_by_block(
            k_caches, v_caches, block_ids_tensor, block_size, num_kv_head, head_dim, tp_num_need_pulls, layers
        )

    def reformat_kv_cache(
        self,
        block_ids: list[list[int]],
        tp_num_need_pulls: int,
        need_cat_cache: bool = False,
        need_nz_cache: bool = False,
    ):
        # Get necessary parameters
        k_cache = list(self.kv_caches.values())[0][0]
        dtype = k_cache.dtype
        device = k_cache.device

        flat_block_ids = [item for sublist in block_ids for item in sublist]
        block_ids_tensor = torch.tensor(flat_block_ids, dtype=torch.int32, device=device)
        num_blocks = len(flat_block_ids)
        num_tokens = num_blocks * self.block_size

        # Create device tensors for copy operations
        block_table = block_ids_tensor.view(1, -1)
        block_len_tensor = torch.tensor([num_tokens], dtype=torch.int32, device=device)
        seq_start_tensor = torch.tensor([0], dtype=torch.int32, device=device)

        # Initialize buffers
        k_buffer = torch.empty((num_tokens, self.num_kv_heads, self.k_head_dim), dtype=dtype, device=device)
        v_buffer = torch.empty((num_tokens, self.num_kv_heads, self.v_head_dim), dtype=dtype, device=device)

        # Create slot mapping for reshape operations
        block_offsets = torch.arange(0, self.block_size, dtype=torch.int32, device=device)
        slot_mapping = (
            block_offsets.reshape((1, self.block_size)) + block_ids_tensor.reshape((num_blocks, 1)) * self.block_size
        ).flatten()

        # FIXME: Right now, if we skip synchronization at this point, the system
        # will crash in GQA scenarios. However, we still haven't identified the
        # root cause.
        torch.npu.synchronize()

        # Process each layer in the KV cache
        for _, (k_cache_layer, v_cache_layer) in self.kv_caches.items():
            # Load cache data into buffers
            torch_npu.atb.npu_paged_cache_load(
                k_cache_layer,
                v_cache_layer,
                block_table,
                block_len_tensor,
                seq_starts=seq_start_tensor,
                key=k_buffer,
                value=v_buffer,
            )
            if need_cat_cache:
                self._cat_kv_cache(
                    k_cache_layer,
                    v_cache_layer,
                    k_buffer,
                    v_buffer,
                    tp_num_need_pulls,
                    num_blocks,
                    num_tokens,
                    slot_mapping,
                )
            if need_nz_cache:
                self._nz_kv_cache(k_cache_layer, v_cache_layer, k_buffer, v_buffer, slot_mapping)
        # Clean up buffers
        del k_buffer, v_buffer

    def _cat_kv_cache(
        self, k_cache_layer, v_cache_layer, k_buffer, v_buffer, tp_num_need_pulls, num_blocks, num_tokens, slot_mapping
    ):
        def _transpose_kv_cache_between_head(buffer: torch.Tensor) -> torch.Tensor:
            buffer = buffer.view(num_blocks, tp_num_need_pulls, self.block_size, -1)
            buffer.transpose_(1, 2)
            return buffer.contiguous().view(num_tokens, self.num_kv_heads, -1)

        # Transpose KV cache
        k_buffer = _transpose_kv_cache_between_head(k_buffer)
        v_buffer = _transpose_kv_cache_between_head(v_buffer)

        # Reshape and cache the processed buffers
        torch_npu._npu_reshape_and_cache(
            key=k_buffer, value=v_buffer, key_cache=k_cache_layer, value_cache=v_cache_layer, slot_indices=slot_mapping
        )

    def _nz_kv_cache(self, k_cache_layer, v_cache_layer, k_buffer, v_buffer, slot_mapping):
        nz_fmt_last_dim = 16
        k_cache_layer = k_cache_layer.view(
            -1, self.k_head_dim * self.num_kv_heads // nz_fmt_last_dim, self.block_size, nz_fmt_last_dim
        )
        v_cache_layer = v_cache_layer.view(
            -1, self.v_head_dim * self.num_kv_heads // nz_fmt_last_dim, self.block_size, nz_fmt_last_dim
        )
        torch_npu.npu_scatter_pa_kv_cache(k_buffer, v_buffer, k_cache_layer, v_cache_layer, slot_mapping)

    def _get_remote_metadata(self, remote_host: str, remote_handshake_port: int) -> None:
        """Get the metadata from the remote host."""
        sock: zmq.Socket | None = None  # type: ignore
        try:
            sock = self._get_remote_socket(remote_host, remote_handshake_port)
            ensure_zmq_send(sock, self.encoder.encode((GET_META_MSG, "")), f"{remote_host}:{remote_handshake_port}")
            metadata_bytes = ensure_zmq_recv(sock, self.remote_poller, f"{remote_host}:{remote_handshake_port}")
            agent_meta = self.decoder.decode(metadata_bytes)
            engine_id = agent_meta.engine_id
            assert engine_id != self.local_engine_id, (
                f"Conflict engine id {engine_id} with local engine id {self.local_engine_id}."
            )
            self.kv_caches_base_addr[engine_id][remote_handshake_port] = agent_meta.kv_caches_base_addr
            self.remote_te_port[engine_id][remote_handshake_port] = agent_meta.te_rpc_port
        finally:
            if sock is not None:
                self._return_remote_socket(sock, remote_host, remote_handshake_port)
                logger.debug("Returned socket to pool for %s:%d", remote_host, remote_handshake_port)

    def _send_done_recv_signal(
        self,
        request_id: str,
        remote_host: str,
        remote_handshake_port: int,
        remote_port_send_num: dict[int, RemotePortInfo],
    ):
        logger.debug(
            "Sending done recving signal for request %s to %s:%d", request_id, remote_host, remote_handshake_port
        )
        sock: zmq.Socket | None = None  # type: ignore
        try:
            sock = self._get_remote_socket(remote_host, remote_handshake_port)
            data_bytes = self.encoder.encode((DONE_RECVING_MSG, request_id, remote_port_send_num))
            ensure_zmq_send(sock, data_bytes, f"{remote_host}:{remote_handshake_port}")
            resp = ensure_zmq_recv(
                sock, self.remote_poller, f"{remote_host}:{remote_handshake_port}", timeout=self.timeout
            )
            logger.debug(f"Received response for request {request_id}: {resp.decode('utf-8')}")
            if resp != b"ACK":
                logger.error(
                    "Failed to receive ACK for request %s from %s:%d", request_id, remote_host, remote_handshake_port
                )
                raise RuntimeError(f"Failed to receive ACK, resp: {resp.decode('utf-8')}")
        except RuntimeError as e:
            if isinstance(sock, zmq.Socket):  # type: ignore
                sock.close()
                sock = None
                logger.warning(f"Unexpected error occurred in socket, {e}, closing the original channel")
        finally:
            if sock is not None:
                self._return_remote_socket(sock, remote_host, remote_handshake_port)
                logger.debug("Returned socket to pool for %s:%d", remote_host, remote_handshake_port)

    def _get_remote_socket(self, remote_host: str, remote_handshake_port: int) -> zmq.Socket:  # type: ignore
        """Get a socket to the remote host."""
        remote_path = make_zmq_path("tcp", remote_host, remote_handshake_port)
        with self.remote_sockets_lock:
            if self.remote_sockets[remote_path]:
                return self.remote_sockets[remote_path].popleft()

            ctx = zmq.Context()  # type: ignore
            sock = make_zmq_socket(
                ctx=ctx,
                path=remote_path,
                socket_type=zmq.REQ,  # type: ignore
                bind=False,
            )
            sock.setsockopt(
                zmq.SNDTIMEO,  # type: ignore
                int(self.timeout * 1000),
            )
            self.remote_poller.register(sock, zmq.POLLIN)  # type: ignore
            return sock

    def _return_remote_socket(
        self,
        sock: zmq.Socket,  # type: ignore
        remote_host: str,
        remote_handshake_port: int,
    ) -> None:
        """Return the remote socket to the pool."""
        remote_path = make_zmq_path("tcp", remote_host, remote_handshake_port)
        with self.remote_sockets_lock:
            self.remote_sockets[remote_path].append(sock)


class MooncakeConnectorMetadata(KVConnectorMetadata):
    def __init__(self):
        self.requests: dict[str, ReqMeta] = {}
        self.requests_to_send: dict[str, float] = {}
        self.reqs_in_batch: set[str] = set()

    def add_new_req(
        self,
        request_id: str,
        local_block_ids: list[int],
        num_external_tokens: int,
        kv_transfer_params: dict[str, Any],
    ):
        self.requests[request_id] = ReqMeta(
            local_block_ids=local_block_ids,
            num_external_tokens=num_external_tokens,
            remote_block_ids=kv_transfer_params["remote_block_ids"],
            remote_engine_id=kv_transfer_params["remote_engine_id"],
            remote_request_id=kv_transfer_params["remote_request_id"],
            remote_host=kv_transfer_params["remote_host"],
            remote_port=kv_transfer_params["remote_port"],
            remote_pcp_size=kv_transfer_params.get("remote_pcp_size", 1),
            remote_dcp_size=kv_transfer_params.get("remote_dcp_size", 1),
            remote_ptp_size=kv_transfer_params.get("remote_ptp_size"),
            remote_multi_nodes_meta_mapping=kv_transfer_params.get("remote_multi_nodes_meta_mapping", {}),
            num_prompt_blocks=kv_transfer_params.get("num_prompt_blocks", 0),
        )


class MooncakeConnector(KVConnectorBase_V1):
    def __init__(self, vllm_config: VllmConfig, role: KVConnectorRole, kv_cache_config: KVCacheConfig | None = None):
        assert vllm_config.kv_transfer_config is not None
        self.engine_id = vllm_config.kv_transfer_config.engine_id
        self._connector_metadata = MooncakeConnectorMetadata()

        if role == KVConnectorRole.SCHEDULER:
            self.connector_scheduler: MooncakeConnectorScheduler | None = MooncakeConnectorScheduler(
                vllm_config, str(self.engine_id)
            )
            self.connector_worker: MooncakeConnectorWorker | None = None
        elif role == KVConnectorRole.WORKER:
            self.connector_scheduler = None
            self.connector_worker = MooncakeConnectorWorker(vllm_config, str(self.engine_id))

    ############################################################
    # Scheduler Side Methods
    ############################################################

    def get_num_new_matched_tokens(self, request: "Request", num_computed_tokens: int) -> tuple[int, bool]:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.get_num_new_matched_tokens(request, num_computed_tokens)

    def update_state_after_alloc(self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int):
        assert self.connector_scheduler is not None
        return self.connector_scheduler.update_state_after_alloc(request, blocks, num_external_tokens)

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
    ) -> tuple[bool, dict[str, Any] | None]:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.request_finished(request, block_ids)

    ############################################################
    # Worker Side Methods
    ############################################################
    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        assert self.connector_worker is not None
        self.connector_worker.register_kv_caches(kv_caches)

    def get_finished(self, finished_req_ids: set[str]) -> tuple[set[str], set[str]]:
        """Get the finished recving and sending requests."""
        assert self.connector_worker is not None
        return self.connector_worker.get_finished()

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs) -> None:
        assert self.connector_worker is not None
        assert isinstance(self._connector_metadata, MooncakeConnectorMetadata)
        self.connector_worker.start_load_kv(self._connector_metadata)

    def wait_for_layer_load(self, layer_name: str) -> None:
        """MooncakeConnector does not do layerwise saving."""
        pass

    def save_kv_layer(
        self, layer_name: str, kv_layer: torch.Tensor, attn_metadata: "AttentionMetadata", **kwargs
    ) -> None:
        """MooncakeConnector does not save explicitly."""
        pass

    def wait_for_save(self):
        """MooncakeConnector does not save explicitly."""
        pass

    def get_handshake_metadata(self) -> KVConnectorHandshakeMetadata | None:
        """
        Get the KVConnector handshake metadata for this connector.
        This metadata is used for out-of-band connector handshake
        between P/D workers.

        Returns:
            KVConnectorHandshakeMetadata: the handshake metadata.
            None if no handshake metadata is available.
        """
        assert self.connector_worker is not None
        return self.connector_worker.xfer_handshake_metadata

    def set_xfer_handshake_metadata(self, metadata: dict[int, KVConnectorHandshakeMetadata]) -> None:
        """
        Set the KV connector handshake metadata for this connector.

        Args:
            metadata (dict): the handshake metadata to set.
        """
        assert self.connector_scheduler is not None
        self.connector_scheduler.set_xfer_handshake_metadata(metadata)


class MooncakeConnectorScheduler:
    """Implementation of Scheduler side methods"""

    def __init__(self, vllm_config: VllmConfig, engine_id: str):
        self.vllm_config = vllm_config
        init_ascend_config(vllm_config)
        self.ascend_config = get_ascend_config()
        self.block_size = vllm_config.cache_config.block_size
        self.engine_id = engine_id
        self.local_ip = get_ip()
        logger.info("Initializing Mooncake Scheduler %s", engine_id)

        self.side_channel_host = get_ip()
        self.pcp_size = vllm_config.parallel_config.prefill_context_parallel_size
        self.dcp_size = vllm_config.parallel_config.decode_context_parallel_size
        self.tp_size = vllm_config.parallel_config.tensor_parallel_size
        self.max_device_id = (
            vllm_config.parallel_config.tensor_parallel_size
            * vllm_config.parallel_config.data_parallel_size
            * self.pcp_size
            * vllm_config.parallel_config.pipeline_parallel_size
        )

        # Handshake base port
        self.side_channel_port = (
            vllm_config.kv_transfer_config.kv_port
            + vllm_config.parallel_config.data_parallel_rank
            * vllm_config.parallel_config.tensor_parallel_size
            * vllm_config.parallel_config.pipeline_parallel_size
            * self.pcp_size
        )
        # Requests that need to start recv.
        # New requests are added by update_state_after_alloc in
        # the scheduler. Used to make metadata passed to Worker.
        self._reqs_need_recv: dict[str, tuple[Request, list[int], int]] = {}
        self._reqs_need_send: dict[str, float] = {}
        self._reqs_in_batch: set[str] = set()

        self.kv_role = vllm_config.kv_transfer_config.kv_role
        prefill_parallel_config: dict[str, Any] = vllm_config.kv_transfer_config.get_from_extra_config("prefill", {})
        decode_parallel_config: dict[str, Any] = vllm_config.kv_transfer_config.get_from_extra_config("decode", {})
        self._prefill_tp_size = prefill_parallel_config["tp_size"]
        self._prefill_pp_size = prefill_parallel_config.get("pp_size", 1)
        self._decode_tp_size = decode_parallel_config["tp_size"]
        self.num_key_value_heads = vllm_config.model_config.hf_text_config.num_key_value_heads
        self.is_deepseek_mla = vllm_config.model_config.is_deepseek_mla
        self.use_sparse = False

        if self.kv_role == "kv_consumer":
            self.all_requests: set[str] = set()
            self.listeningthread = HostListeningThread(
                self.all_requests,
                self._decode_tp_size,
                get_scheduler_ready_zmq_port(vllm_config),
            )
            self.listeningthread.start()

        # master-slave meta information for cross-nodes
        self.multi_nodes_meta_mapping: dict[str, dict[str, Any]] = {}
        self.encoder = msgspec.msgpack.Encoder()
        self.decoder = msgspec.msgpack.Decoder(MooncakeAgentMetadata)
        self.remote_sockets: dict[  # type: ignore
            str, deque[zmq.Socket]
        ] = defaultdict(  # type: ignore
            deque
        )
        self.remote_poller = zmq.Poller()  # type: ignore
        self.timeout = 1.0  # seconds
        self.remote_sockets_lock = threading.Lock()
        self.local_host = get_ip()

    def _build_start_pull_params(
        self,
        request_id: str,
        params: dict[str, Any],
        decode_tp_rank: int,
    ) -> dict[str, Any]:
        prefill_tp_size = params.get("remote_ptp_size") or self._prefill_tp_size
        remote_request_id = params.get("remote_request_id", request_id)
        remote_ranks_per_decode = get_remote_ranks_for_req(
            remote_request_id,
            prefill_tp_size,
            self._decode_tp_size,
            self._prefill_pp_size,
            self.num_key_value_heads,
            self.is_deepseek_mla,
            self.use_sparse,
        )
        p_ranks = remote_ranks_per_decode[decode_tp_rank]
        base_port = params["remote_port"]
        remote_handshake_ports = [base_port + p_rank for p_rank in p_ranks]
        remote_host, remote_engine_id = resolve_remote_host_for_handshake_port(
            base_port,
            remote_handshake_ports[0],
            params["remote_host"],
            params["remote_engine_id"],
            params.get("remote_multi_nodes_meta_mapping"),
        )
        pull_params = copy.copy(params)
        pull_params["remote_host"] = remote_host
        pull_params["remote_engine_id"] = remote_engine_id
        pull_params["remote_handshake_ports"] = remote_handshake_ports
        pull_params["decode_tp_rank"] = decode_tp_rank
        return pull_params

    def _send_start_pull(self, request_id: str, params: dict[str, Any], d2rh_port: int) -> bytes:
        sock: zmq.Socket | None = None  # type: ignore
        try:
            sock = self._get_remote_socket(self.local_host, d2rh_port)
            data_bytes = self.encoder.encode((START_PULL, request_id, params))
            ensure_zmq_send(sock, data_bytes, f"{self.local_host}:{d2rh_port}")
            return ensure_zmq_recv(
                sock,
                self.remote_poller,
                f"{self.local_host}:{d2rh_port}",
                timeout=self.timeout,
            )
        finally:
            if sock is not None:
                self._return_remote_socket(sock, self.local_host, d2rh_port)

    def _get_remote_socket(self, remote_host: str, remote_handshake_port: int) -> zmq.Socket:  # type: ignore
        """Get a socket to the remote host."""
        remote_path = make_zmq_path("tcp", remote_host, remote_handshake_port)
        with self.remote_sockets_lock:
            if self.remote_sockets[remote_path]:
                return self.remote_sockets[remote_path].popleft()

            ctx = zmq.Context()  # type: ignore
            sock = make_zmq_socket(
                ctx=ctx,
                path=remote_path,
                socket_type=zmq.REQ,  # type: ignore
                bind=False,
            )
            sock.setsockopt(
                zmq.SNDTIMEO,  # type: ignore
                int(self.timeout * 1000),
            )
            self.remote_poller.register(sock, zmq.POLLIN)  # type: ignore
            return sock

    def _return_remote_socket(
        self,
        sock: zmq.Socket,  # type: ignore
        remote_host: str,
        remote_handshake_port: int,
    ) -> None:
        """Return the remote socket to the pool."""
        remote_path = make_zmq_path("tcp", remote_host, remote_handshake_port)
        with self.remote_sockets_lock:
            self.remote_sockets[remote_path].append(sock)

    def get_num_new_matched_tokens(self, request: "Request", num_computed_tokens: int) -> tuple[int, bool]:
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
            "MooncakeConnector get_num_new_matched_tokens: num_computed_tokens=%s, kv_transfer_params=%s",
            num_computed_tokens,
            params,
        )

        if params is not None and params.get("do_remote_prefill"):
            if request.request_id not in self.all_requests:
                try:
                    got_staging_full = False
                    for decode_tp_rank in range(self._decode_tp_size):
                        d2rh_port = get_d2rh_zmq_port(self.vllm_config, decode_tp_rank)
                        pull_params = self._build_start_pull_params(request.request_id, params, decode_tp_rank)
                        resp = self._send_start_pull(request.request_id, pull_params, d2rh_port)
                        logger.info(
                            "START_PULL request=%s decode_tp_rank=%d d2rh_port=%d remote_handshake_ports=%s resp=%s",
                            request.request_id,
                            decode_tp_rank,
                            d2rh_port,
                            pull_params.get("remote_handshake_ports"),
                            resp.decode("utf-8"),
                        )
                        if resp == STAGING_FULL:
                            got_staging_full = True
                            break
                        if resp != b"ACK":
                            logger.error(
                                "Failed to receive ACK for request %s from %s:%d",
                                request.request_id,
                                self.local_host,
                                d2rh_port,
                            )
                            raise RuntimeError(f"Failed to receive ACK, resp: {resp.decode('utf-8')}")
                    if got_staging_full:
                        with self.listeningthread.ready_lock:
                            self.all_requests.discard(request.request_id)
                            self.listeningthread.ready_count.pop(request.request_id, None)
                        return None, False
                    self.all_requests.add(request.request_id)
                except RuntimeError as e:
                    logger.warning(
                        "Unexpected error during START_PULL for request %s: %s",
                        request.request_id,
                        e,
                    )
            with self.listeningthread.ready_lock:
                if request.request_id in self.listeningthread.ready_request:
                    self.all_requests.remove(request.request_id)
                    assert num_computed_tokens % self.block_size == 0
                    # Note: We use the full token count as transmit data here.
                    count = max(len(request.prompt_token_ids) - num_computed_tokens, 0)
                    return count, count > 0
                else:
                    return None, False

        # No remote prefill for this request.
        return 0, False

    def update_state_after_alloc(self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int):
        params = request.kv_transfer_params
        logger.debug(
            "MooncakeConnector update_state_after_alloc: num_external_tokens=%s, kv_transfer_params=%s",
            num_external_tokens,
            params,
        )

        if params is not None and (params.get("do_remote_prefill", False) or params.get("do_remote_decode", False)):
            self._reqs_in_batch.add(request.request_id)
        if params is not None and params.get("do_remote_prefill"):
            if params.get("remote_block_ids"):
                if all(p in params for p in ("remote_engine_id", "remote_host", "remote_port", "remote_request_id")):
                    local_block_ids = blocks.get_unhashed_block_ids() if num_external_tokens > 0 else []
                    # Get unhashed blocks to pull from remote.
                    self._reqs_need_recv[request.request_id] = (request, local_block_ids, num_external_tokens)
                else:
                    logger.warning("Got invalid KVTransferParams: %s. This request will not utilize KVTransfer", params)
            else:
                assert num_external_tokens == 0
            # Only trigger 1 KV transfer per request.
            params["do_remote_prefill"] = False

    def build_connector_meta(
        self,
        scheduler_output: SchedulerOutput,
    ) -> KVConnectorMetadata:
        meta = MooncakeConnectorMetadata()

        # Loop through scheduled reqs and convert to ReqMeta.
        for req_id, (req, block_ids, num_external_tokens) in self._reqs_need_recv.items():
            assert req.kv_transfer_params is not None
            # For the case where there are no remote blocks to pull
            # (block_ids is empty), we don't need to schedule
            # an async read on the worker side.
            meta.add_new_req(
                request_id=req_id,
                local_block_ids=block_ids,
                num_external_tokens=num_external_tokens,
                kv_transfer_params=req.kv_transfer_params,
            )

        # Clear the list once workers start the transfers
        self._reqs_need_recv.clear()
        meta.requests_to_send = self._reqs_need_send
        self._reqs_need_send = {}
        meta.reqs_in_batch = self._reqs_in_batch
        self._reqs_in_batch = set()

        return meta

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, dict[str, Any] | None]:
        """
        Once a request is finished, determine whether request blocks
        should be freed now or will be sent asynchronously and freed later.
        """

        params = request.kv_transfer_params
        logger.debug(
            "MooncakeConnector request_finished, request_status=%s, kv_transfer_params=%s", request.status, params
        )

        if (
            params is None
            or not params.get("do_remote_decode")
            or request.status != RequestStatus.FINISHED_LENGTH_CAPPED
        ):
            return False, None

        computed_block_ids = block_ids
        delay_free_blocks = len(computed_block_ids) > 0
        if delay_free_blocks:
            logger.info("Delaying free of %d blocks for request %s", len(computed_block_ids), request.request_id)
            self._reqs_need_send[request.request_id] = time.time()

        num_prompt_blocks = math.ceil(len(request.prompt_token_ids) / self.block_size)

        return delay_free_blocks, dict(
            do_remote_prefill=True,
            do_remote_decode=False,
            remote_block_ids=computed_block_ids,
            remote_engine_id=self.engine_id,
            remote_request_id=request.request_id,
            remote_host=self.side_channel_host,
            remote_port=self.side_channel_port,
            remote_pcp_size=self.pcp_size,
            remote_dcp_size=self.dcp_size,
            remote_ptp_size=self.tp_size,
            last_token_id=request.output_token_ids[-1],
            remote_multi_nodes_meta_mapping=self.multi_nodes_meta_mapping,
            num_prompt_blocks=num_prompt_blocks,
        )

    def set_xfer_handshake_metadata(self, metadata: dict[int, KVConnectorHandshakeMetadata]) -> None:
        """
        Set the KV connector handshake metadata for this connector.

        Args:
            metadata (dict): the handshake metadata to set.
        """
        for local_rank, rank_metadata in metadata.items():
            self.multi_nodes_meta_mapping[str(local_rank)] = {
                "host": rank_metadata.local_ip,
                "engine_id": rank_metadata.engine_id,
            }


class MooncakeConnectorWorker:
    """Implementation of Worker side methods"""

    def __init__(self, vllm_config: VllmConfig, engine_id: str):
        self._get_prefill_decode_size(vllm_config)
        os.environ["ASCEND_TRANSFER_TIMEOUT"] = str(get_transfer_timeout_value())
        if self._prefill_tp_size < self._decode_tp_size:
            raise ValueError(
                f"prefill_tp_size: {self._prefill_tp_size} must be greater than"
                f" or equal to the decode_tp_size: {self._decode_tp_size}"
            )

        # Metadata.
        self.vllm_config = vllm_config
        self.ascend_config = get_ascend_config()
        self.engine_id = engine_id
        self.tp_rank = get_tensor_model_parallel_rank()
        self.tp_size = vllm_config.parallel_config.tensor_parallel_size
        self.tp_group = get_tp_group()
        self.pp_rank = get_pp_group().rank_in_group
        self.dp_rank = vllm_config.parallel_config.data_parallel_rank_local
        self.dp_size = vllm_config.parallel_config.data_parallel_size_local
        self.pp_size = vllm_config.parallel_config.pipeline_parallel_size
        self.kv_caches: dict[str, torch.Tensor] = {}
        self.side_channel_host = get_ip()
        self.pcp_size = get_pcp_group().world_size
        # Assert that pp_size and pcp_size cannot both be greater than 1
        assert not (self.pp_size > 1 and self.pcp_size > 1), "pp and pcp cannot open in same time"
        self.pcp_rank = get_pcp_group().rank_in_group if self.pcp_size > 1 else 0
        self.dcp_size = get_decode_context_model_parallel_world_size()
        self.dcp_rank = get_decode_context_model_parallel_rank() if self.dcp_size > 1 else 0

        self.max_device_id = self.tp_size * self.dp_size * self.pcp_size * self.pp_size
        self.kv_role = vllm_config.kv_transfer_config.kv_role
        self.num_key_value_heads = self.vllm_config.model_config.hf_text_config.num_key_value_heads

        # Handshake base port
        self.side_channel_port = (
            vllm_config.kv_transfer_config.kv_port
            + vllm_config.parallel_config.data_parallel_rank
            * vllm_config.parallel_config.tensor_parallel_size
            * vllm_config.parallel_config.pipeline_parallel_size
            * self.pcp_size
        )
        device_index = (self.pp_rank + self.pcp_rank) * self.tp_size + self.tp_rank
        self.handshake_port = self.side_channel_port + device_index
        self.sockets: dict = {}
        self.engine = global_te.get_transfer_engine(self.side_channel_host, device_name=None)
        self.te_rpc_port = self.engine.get_rpc_port()

        # Background thread for sending or receiving KV caches.
        self.kv_send_thread: KVCacheSendingThread | None = None
        self.kv_recv_thread: KVCacheRecvingThread | None = None

        # Handshake metadata of this worker
        self.xfer_handshake_metadata: MooncakeAgentMetadata | None = None

        # kv_transfer variables
        self.vllm_config = vllm_config
        self.block_size = vllm_config.cache_config.block_size
        if self.vllm_config.model_config.is_deepseek_mla:
            self.tp_num_need_pulls = 1
        else:
            num_d_block_heads = max(1, self.num_key_value_heads // self.tp_size)
            num_p_block_heads = max(1, self.num_key_value_heads // self._prefill_tp_size)
            self.tp_num_need_pulls = num_d_block_heads // num_p_block_heads
        self.local_remote_block_port_mapping: dict[str, list[list[int]] | None] = {}
        self.remote_port_send_num: dict[str, dict[int, RemotePortInfo]] = {}
        self.remote_local_block_map: dict[str, dict[int, int]] = {}

    def _get_prefill_decode_size(self, vllm_config: VllmConfig):
        # get prefill tp and dp size from extra config
        prefill_parallel_config: dict[str, Any] = vllm_config.kv_transfer_config.get_from_extra_config("prefill", {})

        assert "tp_size" in prefill_parallel_config
        self._prefill_tp_size = prefill_parallel_config["tp_size"]

        assert "dp_size" in prefill_parallel_config
        self._prefill_dp_size = prefill_parallel_config["dp_size"]
        # get prefill pp size from extra config
        self._prefill_pp_size = prefill_parallel_config.get("pp_size", 1)
        # get decode tp and dp size from extra config
        decode_parallel_config: dict[str, Any] = vllm_config.kv_transfer_config.get_from_extra_config("decode", {})
        assert "tp_size" in decode_parallel_config
        self._decode_tp_size = decode_parallel_config["tp_size"]
        assert "dp_size" in decode_parallel_config
        self._decode_dp_size = decode_parallel_config["dp_size"]
        # get prefill pp size from extra config
        self._decode_pp_size = decode_parallel_config.get("pp_size", 1)
        assert self._decode_pp_size == 1, "decode pp size must be 1"
        self._prefill_pp_layer_partition = prefill_parallel_config.get("pp_layer_partition")

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        """Register the KV Cache data."""

        _, first_kv_cache_tuple = next(iter(kv_caches.items()))
        first_kv_cache = first_kv_cache_tuple[0]

        # TODO(tms): Find a more robust way to detect and handle MLA
        self.use_mla = (
            first_kv_cache_tuple[0].size(-1) != first_kv_cache_tuple[1].size(-1) and len(first_kv_cache_tuple) == 2
        )
        self.use_sparse = len(first_kv_cache_tuple) == 3

        self.num_blocks = first_kv_cache.shape[0]
        logger.info("num_blocks: %s", self.num_blocks)
        self.block_len = []
        if self.use_mla or self.use_sparse:
            block_rank = 3  # [block_size, latent_dim]
            for i in range(len(first_kv_cache_tuple)):
                block_shape = first_kv_cache_tuple[i].shape[-block_rank:]
                logger.info("block_shape: %s", block_shape)
                self.block_len.append(first_kv_cache[i].element_size() * math.prod(block_shape))
        else:
            # eager:[num_block, block_size, num_head, hidden_dim]
            block_rank = (
                len(first_kv_cache.shape) - 1
            )  # [block_size, kv_heads, head_dim] or [block_size, kv_heads*head_dim]
            block_shape = first_kv_cache.shape[-block_rank:]
            logger.info("block_shape: %s", block_shape)
            self.block_len = [first_kv_cache.element_size() * math.prod(block_shape)]

        logger.info(
            "Registering KV_Caches. use_mla: %s, use_sparse: %s, shape %s",
            self.use_mla,
            self.use_sparse,
            first_kv_cache.shape,
        )

        self.kv_caches = kv_caches
        kv_caches_base_addr = []
        ptrs = []
        lengths = []
        length = len(self.block_len)
        for cache_or_caches in kv_caches.values():
            # Normalize to always be a list of caches
            for i, cache in enumerate(cache_or_caches, 0):
                base_addr = cache.data_ptr()
                region_len = self.num_blocks * self.block_len[i % length]
                kv_caches_base_addr.append(base_addr)
                ptrs.append(base_addr)
                lengths.append(region_len)

        cpu_kv_caches_base_addr: list[int] = []
        cpu_kvcache_manager_instance: cpu_kvcache_manager | None = None
        cpu_te_rpc_port = self.te_rpc_port
        self.cpu_caches_hold: list[torch.Tensor] = []
        self.d2rh_thread: D2RHThread | None = None

        if self.kv_role == "kv_consumer":
            cpu_kvcache_manager_instance = cpu_kvcache_manager(self.num_blocks)
            for cache_or_caches in kv_caches.values():
                for i, cache in enumerate(cache_or_caches):
                    cpu_cache = torch.empty(
                        cache.shape,
                        dtype=cache.dtype,
                        device="cpu",
                        pin_memory=True,
                    )
                    self.cpu_caches_hold.append(cpu_cache)
                    cpu_kv_caches_base_addr.append(cpu_cache.data_ptr())
                    ptrs.append(cpu_cache.data_ptr())
                    lengths.append(self.num_blocks * self.block_len[i % length])

            d2rh_port = get_d2rh_zmq_port(self.vllm_config, self.tp_rank, self.pp_rank, self.pcp_rank)
            scheduler_ready_port = get_scheduler_ready_zmq_port(self.vllm_config)
            self.d2rh_thread = D2RHThread(
                cpu_ptrs=cpu_kv_caches_base_addr,
                block_len=self.block_len,
                engine=self.engine,
                cpu_kvcache_manager=cpu_kvcache_manager_instance,
                remote_local_block_map=self.remote_local_block_map,
                vllm_config=self.vllm_config,
                d2rh_handshake_port=d2rh_port,
                scheduler_ready_port=scheduler_ready_port,
                tp_rank=self.tp_rank,
            )
            self.d2rh_thread.start()
            cpu_te_rpc_port = self.te_rpc_port

        global_te.register_buffer(ptrs, lengths)
        # After KV Caches registered, start the sending or receiving thread.
        metadata = MooncakeAgentMetadata(
            engine_id=self.engine_id,
            te_rpc_port=self.te_rpc_port,
            kv_caches_base_addr=kv_caches_base_addr,
            num_blocks=self.num_blocks,
            local_ip=get_ip(),
        )
        self.xfer_handshake_metadata = metadata

        ready_event = threading.Event()
        if self.kv_role == "kv_producer":
            self.kv_send_thread = KVCacheSendingThread(
                self.vllm_config,
                self.tp_rank,
                self._prefill_tp_size,
                self.engine_id,
                self.side_channel_host,
                self.side_channel_port,
                metadata,
                ready_event,
                self.kv_caches,
                self.pcp_rank,
            )
            self.kv_send_thread.start()
        else:
            self.kv_recv_thread = KVCacheRecvingThread(
                self.tp_rank,
                self.tp_size,
                self._prefill_pp_size,
                self.engine,
                self.engine_id,
                self.handshake_port,
                self.side_channel_port,
                kv_caches_base_addr,
                self.block_len,
                ready_event,
                self.vllm_config,
                self.kv_caches,
                self.te_rpc_port,
                cpu_kv_caches_base_addr,
                cpu_kv_caches_base_addr,
                cpu_te_rpc_port,
                cpu_kvcache_manager_instance,
                use_cpu_staging=True,
                prefill_pp_layer_partition=self._prefill_pp_layer_partition,
                remote_local_block_map=self.remote_local_block_map,
            )
            self.kv_recv_thread.start()

        start_wait_time = time.time()
        thread = self.kv_send_thread if self.kv_role == "kv_producer" else self.kv_recv_thread
        assert thread is not None
        while not ready_event.is_set():
            if not thread.is_alive():
                raise RuntimeError("KV Cache sending/receiving thread failed to start.")
            if time.time() - start_wait_time > 5 * 60:
                raise RuntimeError("Timeout waiting for KV Cache thread to be ready.")
            time.sleep(3)

    def get_finished(self) -> tuple[set[str], set[str]]:
        done_sending = (
            self.kv_send_thread.get_and_clear_finished_requests(  # type: ignore[union-attr]
            )
            if self.kv_role == "kv_producer"
            else set()
        )
        done_recving = (
            self.kv_recv_thread.get_and_clear_finished_requests(  # type: ignore[union-attr]
            )
            if self.kv_role == "kv_consumer"
            else set()
        )
        if self.tp_rank == 0:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "Number of completed KV cache send requests: %d, receive requests: %d",
                    len(done_sending),
                    len(done_recving),
                )
        return done_sending, done_recving

    def get_block_ids_with_load_errors(self) -> set[int]:
        if self.kv_role == "kv_consumer" and self.kv_recv_thread is not None:
            return self.kv_recv_thread.get_and_clear_invalid_block_ids()
        return set()

    def _get_kv_split_metadata(
        self,
        req_id: str,
        meta: ReqMeta,
    ) -> tuple[list[list[int]], list[list[int]], list[list[int]]]:
        """
        In cp/dcp scenario, kv_cache may be split, so we need to pull multiple blocks from multiple remote P node.
        Use this function to calculate remote port and remote block number of each remote P node that we need to pull.
        """
        prefill_tp_size = meta.remote_ptp_size if getattr(meta, "remote_ptp_size", None) else self._prefill_tp_size
        if meta.remote_pcp_size * meta.remote_dcp_size * self.pcp_size * self.dcp_size == 1:
            chosen_rank_list = self._get_remote_rank(req_id, prefill_tp_size)
            remote_handshake_port_list = [[x + meta.remote_port for x in chosen_rank_list]]
            local_block_ids_list, remote_block_ids_list = [meta.local_block_ids], [meta.remote_block_ids]
            return remote_handshake_port_list, local_block_ids_list, remote_block_ids_list

        def context_parallel_parameters_check():
            assert (meta.remote_pcp_size * meta.remote_dcp_size) % (self.pcp_size * self.dcp_size) == 0
            if not (self.use_mla or self.use_sparse):
                p_node_heads_per_rank = math.ceil(self.num_key_value_heads / prefill_tp_size)
                d_node_heads_per_rank = math.ceil(self.num_key_value_heads / self.tp_size)
                assert d_node_heads_per_rank % p_node_heads_per_rank == 0

        def get_kv_head_groups(tp_size):
            if self.use_mla or self.use_sparse:
                kv_head_groups = []
                kv_head_ids = [0]
                kv_head_groups.append(tuple(kv_head_ids))
                return kv_head_groups
            if self.num_key_value_heads // tp_size >= 1:
                kv_head_groups = []
                for tp_rank in range(tp_size):
                    kv_head_ids = [
                        head_idx + tp_rank * (self.num_key_value_heads // tp_size)
                        for head_idx in range(self.num_key_value_heads // tp_size)
                    ]
                    kv_head_groups.append(tuple(kv_head_ids))
                return kv_head_groups
            if tp_size // self.num_key_value_heads > 1:
                kv_head_groups = []
                for kv_head_ids_ in range(self.num_key_value_heads):
                    kv_head_groups.append(tuple([kv_head_ids_]))
                return kv_head_groups

        def get_cp_group_meta(tp_size, pcp_size, dcp_size, port_base):
            # key is kv_head_group, value is cp_groups and which cp_groups to select
            cp_group_meta: dict = {}
            kv_head_groups = get_kv_head_groups(tp_size)
            dcp_repeat_num = tp_size // len(kv_head_groups) // dcp_size

            for kv_head_group_idx, kv_head_group in enumerate(kv_head_groups):
                if kv_head_group not in cp_group_meta:
                    cp_group_meta[kv_head_group] = {}
                    cp_group_meta[kv_head_group]["cp_groups"] = []
                    cp_group_meta[kv_head_group]["select_cp_groups_id"] = 0
                kv_head_group_offset = tp_size // len(kv_head_groups) * kv_head_group_idx
                for dcp_repeat_idx in range(dcp_repeat_num):
                    # len(cp_group) == pcp_size * dcp_size
                    cp_group = []
                    dcp_repeat_offset = dcp_size * dcp_repeat_idx
                    for pcp_rank in range(pcp_size):
                        pcp_rank_offset = tp_size * pcp_rank
                        for dcp_rank in range(dcp_size):
                            cp_group.append(
                                dcp_rank + port_base + pcp_rank_offset + dcp_repeat_offset + kv_head_group_offset
                            )
                    cp_group_meta[kv_head_group]["cp_groups"].append(cp_group)

            return cp_group_meta

        def get_local_remote_block_port_mappings():
            context_parallel_parameters_check()
            p_node_cp_group_meta = get_cp_group_meta(
                prefill_tp_size, meta.remote_pcp_size, meta.remote_dcp_size, meta.remote_port
            )
            d_node_cp_group_meta = get_cp_group_meta(self.tp_size, self.pcp_size, self.dcp_size, self.side_channel_port)
            local_remote_block_port_mappings: dict[int, list[list[int]]] = {}
            for d_node_head_key in d_node_cp_group_meta:
                for p_node_head_key in p_node_cp_group_meta:
                    if not set(p_node_head_key).issubset(set(d_node_head_key)):
                        continue
                    d_node_head_group = d_node_cp_group_meta[d_node_head_key]
                    p_node_head_group = p_node_cp_group_meta[p_node_head_key]
                    for d_cp_group in d_node_head_group["cp_groups"]:
                        select_cp_groups_id = p_node_head_group["select_cp_groups_id"]
                        p_cp_groups = p_node_head_group["cp_groups"]
                        p_cp_group = p_cp_groups[select_cp_groups_id]
                        p_node_head_group["select_cp_groups_id"] = (
                            select_cp_groups_id + 1 if select_cp_groups_id + 1 < len(p_cp_groups) else 0
                        )
                        for d_idx, d_port in enumerate(d_cp_group):
                            if d_port not in local_remote_block_port_mappings:
                                local_remote_block_port_mappings[d_port] = []
                            p_port_remote_list = []
                            for p_idx, p_port in enumerate(p_cp_group):
                                if p_idx % len(d_cp_group) == d_idx:
                                    p_port_remote_list.append(p_port)
                            local_remote_block_port_mappings[d_port].append(p_port_remote_list)

            logger.info(
                "p_node_cp_group_meta is:: %s. d_node_cp_group_meta is:: %s. "
                "local_remote_block_port_mappings is:: %s. ",
                p_node_cp_group_meta,
                d_node_cp_group_meta,
                local_remote_block_port_mappings,
            )

            return local_remote_block_port_mappings

        def get_remote_port_send_num(
            local_remote_block_port_mappings: dict[int, list[list[int]]],
        ) -> dict[int, RemotePortInfo]:
            remote_port_send_num: dict[int, RemotePortInfo] = {}
            for port in range(prefill_tp_size * meta.remote_pcp_size):
                remote_host_info = meta.remote_multi_nodes_meta_mapping.get(str(port), None)
                if remote_host_info is None:
                    remote_host = meta.remote_host
                else:
                    remote_host = remote_host_info["host"]
                remote_port_send_num[meta.remote_port + port] = {"num": 0, "host": remote_host}

            for remote_port_head_list in local_remote_block_port_mappings.values():
                for remote_port_list in remote_port_head_list:
                    for remote_port in remote_port_list:
                        remote_port_send_num[remote_port]["num"] += 1
            return remote_port_send_num

        if meta.remote_engine_id not in self.local_remote_block_port_mapping:
            self.local_remote_block_port_mapping[meta.remote_engine_id] = None

        if self.local_remote_block_port_mapping[meta.remote_engine_id] is None:
            local_remote_block_port_mappings = get_local_remote_block_port_mappings()
            self.local_remote_block_port_mapping[meta.remote_engine_id] = local_remote_block_port_mappings[
                self.handshake_port
            ]
            self.remote_port_send_num[meta.remote_engine_id] = get_remote_port_send_num(
                local_remote_block_port_mappings
            )

        local_remote_block_port_mapping = copy.deepcopy(self.local_remote_block_port_mapping[meta.remote_engine_id])

        num_external_blocks = math.ceil(meta.num_external_tokens / self.block_size)

        assert math.ceil(num_external_blocks / (self.pcp_size * self.dcp_size)) == len(meta.local_block_ids), (
            f"num_external_blocks({num_external_blocks}), cp_size({self.pcp_size * self.dcp_size}), "
            f"local_block_ids_len ({len(meta.local_block_ids)})"
        )
        assert meta.num_prompt_blocks >= num_external_blocks, (
            f"meta.num_prompt_blocks({meta.num_prompt_blocks}), num_external_blocks({num_external_blocks})"
        )

        remote_cp_size = meta.remote_pcp_size * meta.remote_dcp_size
        remote_block_nums_all = [meta.num_prompt_blocks // remote_cp_size] * remote_cp_size
        num_remain_blocks = meta.num_prompt_blocks % remote_cp_size
        for i in range(num_remain_blocks):
            remote_block_nums_all[i] += 1
        last_block_location = (num_remain_blocks + remote_cp_size - 1) % remote_cp_size

        # Considering prefix cache, the remote_block_nums_all should be revised
        num_prefix_cached_blocks = meta.num_prompt_blocks - num_external_blocks
        remote_block_nums_all = [num - num_prefix_cached_blocks // remote_cp_size for num in remote_block_nums_all]
        num_remain_blocks = num_prefix_cached_blocks % remote_cp_size
        for i in range(num_remain_blocks):
            remote_block_nums_all[i] -= 1

        # make sure the last block (which may be unfull) of P nodes is put to the last block of D node
        remote_block_nums: list[int] = []
        final_block_idx: int | None = None
        local_cp_rank = self.dcp_rank + self.pcp_rank * self.dcp_size
        local_cp_size = self.dcp_size * self.pcp_size
        for cp_rank, block_num in enumerate(remote_block_nums_all):
            if cp_rank % local_cp_size == local_cp_rank:
                if last_block_location == cp_rank:
                    final_block_idx = len(remote_block_nums)
                remote_block_nums.append(block_num)

        assert local_remote_block_port_mapping is not None
        if final_block_idx is not None:
            final_block_num = remote_block_nums.pop(final_block_idx)
            remote_block_nums.append(final_block_num)
            for mapping in local_remote_block_port_mapping:
                final_block_port = mapping.pop(final_block_idx)
                mapping.append(final_block_port)

        remote_handshake_port_list, local_block_ids_list, remote_block_ids_list = [], [], []
        for idx in range(len(local_remote_block_port_mapping[0])):
            mapping_list = []
            for mapping in local_remote_block_port_mapping:
                mapping_list.append(mapping[idx])
            remote_handshake_port_list.append(mapping_list)

        # the local_block_ids_list and remote_block_ids_list are related with remote_handshake_port_list
        # such as: local_block_ids_list[[1],[2],[5],[6]], remote_block_ids_list[[1],[1],[1],[1]],
        # remote_handshake_port_list[[30000],[30001],[30004],[30005]]
        # D rank will get remote block 1 in port 30004 and save it in local block 5
        local_block_offset = 0
        for remote_kv_id in range(len(remote_handshake_port_list)):
            num_blocks_to_pull = remote_block_nums[remote_kv_id]
            remote_block_ids_list.append(meta.remote_block_ids[:num_blocks_to_pull])
            local_block_ids_list.append(
                meta.local_block_ids[local_block_offset : local_block_offset + num_blocks_to_pull]
            )
            local_block_offset += num_blocks_to_pull

        tp_num_need_pulls = self._get_tp_num_need_pulls(prefill_tp_size)
        assert tp_num_need_pulls == len(remote_handshake_port_list[0]), (
            f"tp_num_need_pulls: {tp_num_need_pulls}, remote_handshake_port_list: {remote_handshake_port_list}"
        )

        return remote_handshake_port_list, local_block_ids_list, remote_block_ids_list

    def start_load_kv(self, metadata: MooncakeConnectorMetadata):
        """Start loading KV blocks from remote engine."""
        for req_id, meta in metadata.requests.items():
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "start_load_kv for request %s from remote engine %s. "
                    "Num local_block_ids: %s. Num remote_block_ids: %s. ",
                    req_id,
                    meta.remote_engine_id,
                    len(meta.local_block_ids),
                    len(meta.remote_block_ids),
                )

            prefill_tp_size = meta.remote_ptp_size if getattr(meta, "remote_ptp_size", None) else self._prefill_tp_size
            tp_num_need_pulls = self._get_tp_num_need_pulls(prefill_tp_size)
            remote_req_id = meta.remote_request_id

            if meta.remote_pcp_size * meta.remote_dcp_size > 1:
                remote_handshake_port_list, local_block_ids_list, remote_block_ids_list = self._get_kv_split_metadata(
                    req_id, meta
                )

                for pcp_dcp_rank in range(len(remote_handshake_port_list)):
                    for i in range(tp_num_need_pulls):
                        assert self.kv_recv_thread is not None
                        remote_host, remote_engine_id = self._get_remote_host_info_by_port(
                            meta.remote_port,
                            remote_handshake_port_list[pcp_dcp_rank][i],
                            meta.remote_host,
                            meta.remote_engine_id,
                            meta.remote_multi_nodes_meta_mapping,
                        )
                        self.kv_recv_thread.add_request(
                            request_id=req_id,
                            remote_request_id=remote_req_id,
                            local_block_ids=local_block_ids_list[pcp_dcp_rank],
                            remote_block_ids=remote_block_ids_list[pcp_dcp_rank],
                            remote_engine_id=remote_engine_id,
                            remote_host=remote_host,
                            remote_handshake_port=remote_handshake_port_list[pcp_dcp_rank][i],
                            offset=i,
                            tp_num_need_pulls=tp_num_need_pulls,
                            remote_port_send_num=self.remote_port_send_num[meta.remote_engine_id],
                            all_task_done=(
                                pcp_dcp_rank == len(remote_handshake_port_list) - 1 and i == tp_num_need_pulls - 1
                            ),
                        )
            else:  # TODO: support prefill context parallel and pipeline parallel open at the same time
                chosen_rank_list = self._get_remote_rank(remote_req_id, prefill_tp_size)
                remote_handshake_port_list = [[x + meta.remote_port] for x in chosen_rank_list]
                for i in range(tp_num_need_pulls * self._prefill_pp_size):
                    assert self.kv_recv_thread is not None
                    remote_host, remote_engine_id = self._get_remote_host_info_by_port(
                        meta.remote_port,
                        remote_handshake_port_list[i][0],
                        meta.remote_host,
                        meta.remote_engine_id,
                        meta.remote_multi_nodes_meta_mapping,
                    )
                    self.kv_recv_thread.add_request(
                        request_id=req_id,
                        remote_request_id=remote_req_id,
                        local_block_ids=meta.local_block_ids,
                        remote_block_ids=meta.remote_block_ids,
                        remote_engine_id=remote_engine_id,
                        remote_host=remote_host,
                        remote_handshake_port=remote_handshake_port_list[i][0],
                        offset=i,
                        tp_num_need_pulls=tp_num_need_pulls,
                        all_task_done=(i == tp_num_need_pulls * self._prefill_pp_size - 1),
                    )

        for req_id in metadata.reqs_in_batch:
            if self.kv_send_thread is not None:
                self.kv_send_thread.task_tracker.add_req_to_process(req_id)
            if self.kv_recv_thread is not None:
                self.kv_recv_thread.task_tracker.add_req_to_process(req_id)

        if self.kv_send_thread is not None and self.pcp_size * self.dcp_size == 1:
            for req_id, delay_start_time in metadata.requests_to_send.items():
                if self.tp_rank in self._prefill_get_remote_rank(req_id):
                    self.kv_send_thread.add_delayed_request(req_id, delay_start_time)
                else:
                    self.kv_send_thread.add_not_transfer_request(req_id)

        if self.kv_send_thread is not None and self.pcp_size * self.dcp_size > 1:
            for req_id, delay_start_time in metadata.requests_to_send.items():
                self.kv_send_thread.add_delayed_request(req_id, delay_start_time)

    def _get_tp_num_need_pulls(self, prefill_tp_size: int) -> int:
        if prefill_tp_size is None:
            prefill_tp_size = self._prefill_tp_size

        if prefill_tp_size == self._prefill_tp_size:
            return self.tp_num_need_pulls

        if self.vllm_config.model_config.is_deepseek_mla:
            tp_num_need_pulls = 1
        else:
            num_d_block_heads = max(1, self.num_key_value_heads // self.tp_size)
            num_p_block_heads = max(1, self.num_key_value_heads // prefill_tp_size)
            tp_num_need_pulls = num_d_block_heads // num_p_block_heads
        return tp_num_need_pulls

    def _get_remote_host_info_by_port(
        self,
        base_port: int,
        remote_handshake_port: int,
        remote_host: str,
        remote_engine_id: str,
        remote_multi_nodes_meta_mapping: dict,
    ):
        rank = str(remote_handshake_port - base_port)
        if remote_multi_nodes_meta_mapping is None or remote_multi_nodes_meta_mapping.get(rank) is None:
            return remote_host, remote_engine_id
        info = remote_multi_nodes_meta_mapping[rank]
        return info.get("host", remote_host), info.get("engine_id", remote_engine_id)

    def _prefill_get_remote_rank(self, req_id: str) -> list[int]:
        return sum(self._get_remote_ranks_for_req(req_id), [])

    def _get_remote_rank(self, req_id: str, prefill_tp_size: int | None = None) -> list[int]:
        return self._get_remote_ranks_for_req(req_id, prefill_tp_size)[self.tp_rank]

    def _get_remote_ranks_for_req(self, req_id: str, prefill_tp_size: int | None = None) -> list[list[int]]:
        if prefill_tp_size is None:
            prefill_tp_size = self._prefill_tp_size
        return get_remote_ranks_for_req(
            req_id,
            prefill_tp_size,
            self._decode_tp_size,
            self._prefill_pp_size,
            self.num_key_value_heads,
            self.vllm_config.model_config.is_deepseek_mla,
            self.use_sparse,
        )

    def _get_remote_tp_ranks(
        self, tp_ori_data: np.ndarray, rand_group_index: list[int], num_groups: int, prefill_tp_size: int
    ) -> list[list[int]]:
        return get_remote_tp_ranks(
            tp_ori_data,
            rand_group_index,
            num_groups,
            prefill_tp_size,
            self._decode_tp_size,
            self.num_key_value_heads,
            self.vllm_config.model_config.is_deepseek_mla,
            self.use_sparse,
        )


@contextlib.contextmanager
def zmq_ctx(socket_type: Any, addr: str) -> Iterator[zmq.Socket]:  # type: ignore
    """Context manager for a ZMQ socket"""

    if socket_type not in (zmq.ROUTER, zmq.REQ, zmq.DEALER):  # type: ignore
        raise ValueError(f"Unexpected socket type: {socket_type}")

    ctx: zmq.Context | None = None  # type: ignore
    try:
        ctx = zmq.Context()  # type: ignore
        yield make_zmq_socket(ctx=ctx, path=addr, socket_type=socket_type, bind=socket_type == zmq.ROUTER)  # type: ignore
    finally:
        if ctx is not None:
            ctx.destroy(linger=0)


def group_concurrent_contiguous(
    src: list[int], dst: list[int]
) -> tuple[list[npt.NDArray[np.int64]], list[npt.NDArray[np.int64]]]:
    """Vectorised NumPy implementation."""
    src_indices: npt.NDArray[np.int64] = np.array(src, dtype=np.int64)
    dst_indices: npt.NDArray[np.int64] = np.array(dst, dtype=np.int64)

    if src_indices.size == 0:
        return [], []

    brk = np.where((np.diff(src_indices) != 1) | (np.diff(dst_indices) != 1))[0] + 1
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
    path: str,
    max_retries: int = 3,
):
    retries_left = max_retries
    while True:
        try:
            socket.send(data)
            return
        except zmq.ZMQError as e:  # type: ignore
            retries_left -= 1
            if retries_left > 0:
                logger.warning("Send failed: %s, retrying... (%s attempts left)", e, retries_left)
                time.sleep(0.1)
            else:
                logger.error("Send failed after all retries: %s", e)
                raise RuntimeError(f"Failed to send data to {path} after {max_retries} retries: {e}")


def ensure_zmq_recv(
    socket: zmq.Socket,  # type: ignore
    poller: zmq.Poller,  # type: ignore
    path: str,
    timeout: float = 1.0,
    max_retries: int = 3,
) -> bytes:
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
                logger.warning("Receive failed: %s, retrying... (%s attempts left)", e, retries_left)
                time.sleep(0.1)
            else:
                logger.error("Receive failed from %s after all retries: %s", path, e)
                raise RuntimeError(f"Failed to receive data after {max_retries} retries: {e}")


# decode node should know pp_partition_layer in prefill node,
# it is configured in kv_transfer_config by partition_list_str,
# default using vllm layer split algorithm.
def get_prefill_pp_indices(
    num_hidden_layers: int, pp_rank: int, pp_size: int, partition_list_str: str | None = None
) -> tuple[int, int]:
    if partition_list_str is None:
        return get_pp_indices(num_hidden_layers, pp_rank, pp_size)
    else:
        try:
            partitions = [int(layer) for layer in partition_list_str.split(",")]
        except ValueError as err:
            raise ValueError("Invalid partition string: {}".format(partition_list_str)) from err
        if len(partitions) != pp_size:
            raise ValueError(f"{len(partitions)=} does not match {pp_size=}.")
        if sum(partitions) != num_hidden_layers:
            raise ValueError(f"{sum(partitions)=} does not match {num_hidden_layers=}.")
        start_layer = sum(partitions[:pp_rank])
        end_layer = start_layer + partitions[pp_rank]
        return (start_layer, end_layer)


def get_parallel_device_index(
    tp_rank: int,
    tp_size: int,
    pp_rank: int = 0,
    pcp_rank: int = 0,
) -> int:
    return (pp_rank + pcp_rank) * tp_size + tp_rank


def get_dp_port_offset(
    dp_rank: int,
    tp_size: int,
    pp_size: int,
    pcp_size: int,
) -> int:
    return dp_rank * tp_size * pp_size * pcp_size


def get_d2rh_zmq_port(
    vllm_config: VllmConfig,
    tp_rank: int,
    pp_rank: int = 0,
    pcp_rank: int = 0,
) -> int:
    parallel_config = vllm_config.parallel_config
    tp_size = parallel_config.tensor_parallel_size
    pp_size = parallel_config.pipeline_parallel_size
    pcp_size = parallel_config.prefill_context_parallel_size
    dp_rank = parallel_config.data_parallel_rank
    device_index = get_parallel_device_index(tp_rank, tp_size, pp_rank, pcp_rank)
    dp_offset = get_dp_port_offset(dp_rank, tp_size, pp_size, pcp_size)
    return D2RH_ZMQ_PORT_BASE + dp_offset + device_index


def get_scheduler_ready_zmq_port(vllm_config: VllmConfig) -> int:
    parallel_config = vllm_config.parallel_config
    tp_size = parallel_config.tensor_parallel_size
    pp_size = parallel_config.pipeline_parallel_size
    pcp_size = parallel_config.prefill_context_parallel_size
    dp_rank = parallel_config.data_parallel_rank
    dp_offset = get_dp_port_offset(dp_rank, tp_size, pp_size, pcp_size)
    return SCHEDULER_READY_ZMQ_PORT_BASE + dp_offset


def compute_tp_num_need_pulls(
    num_key_value_heads: int,
    decode_tp_size: int,
    prefill_tp_size: int,
    is_deepseek_mla: bool,
) -> int:
    if is_deepseek_mla:
        return 1
    num_d_block_heads = max(1, num_key_value_heads // decode_tp_size)
    num_p_block_heads = max(1, num_key_value_heads // prefill_tp_size)
    return num_d_block_heads // num_p_block_heads


def get_remote_tp_ranks(
    tp_ori_data: np.ndarray,
    rand_group_index: list[int],
    num_groups: int,
    prefill_tp_size: int,
    decode_tp_size: int,
    num_key_value_heads: int,
    is_deepseek_mla: bool,
    use_sparse: bool,
) -> list[list[int]]:
    tp_num_need_pulls = compute_tp_num_need_pulls(num_key_value_heads, decode_tp_size, prefill_tp_size, is_deepseek_mla)
    tp_sampled_nums: list[list[int]] = []
    if prefill_tp_size > num_key_value_heads or is_deepseek_mla or use_sparse:
        tp_ori_data = tp_ori_data.reshape(-1, num_groups)
        chosen_group = tp_ori_data[:, rand_group_index]
        flattened = chosen_group.reshape(-1).tolist()
        tp_sampled_nums = [flattened[i : i + tp_num_need_pulls] for i in range(0, len(flattened), tp_num_need_pulls)]
    else:
        group_size = prefill_tp_size // decode_tp_size
        for i in range(decode_tp_size):
            slice_data = tp_ori_data[i * group_size : (i + 1) * group_size]
            tp_sampled_nums.append(slice_data.tolist())
    return tp_sampled_nums


def get_remote_ranks_for_req(
    req_id: str,
    prefill_tp_size: int,
    decode_tp_size: int,
    prefill_pp_size: int,
    num_key_value_heads: int,
    is_deepseek_mla: bool,
    use_sparse: bool,
) -> list[list[int]]:
    if prefill_tp_size == decode_tp_size:
        return [[tp + pp * prefill_tp_size for pp in range(prefill_pp_size)] for tp in range(prefill_tp_size)]

    if is_deepseek_mla or use_sparse:
        num_kv_head = 1
    else:
        num_kv_head = num_key_value_heads
    ori_data = np.arange(prefill_tp_size * prefill_pp_size)
    seed = string_to_int64_hash(req_id)
    rand = random.Random(seed)
    ori_data = ori_data.reshape(prefill_pp_size, -1)
    num_groups = max(1, len(ori_data[0]) // num_kv_head)
    rand_group_index = rand.sample(range(num_groups), max(decode_tp_size // num_kv_head, 1))
    all_results = [
        get_remote_tp_ranks(
            ori_data[pp_index],
            rand_group_index,
            num_groups,
            prefill_tp_size,
            decode_tp_size,
            num_key_value_heads,
            is_deepseek_mla,
            use_sparse,
        )
        for pp_index in range(prefill_pp_size)
    ]
    sampled_nums: list[list[int]] = []
    for group_index in range(len(all_results[0])):
        group: list[int] = []
        for pp_index in range(prefill_pp_size):
            group.extend(all_results[pp_index][group_index])
        sampled_nums.append(group)
    return sampled_nums


def resolve_remote_host_for_handshake_port(
    base_port: int,
    remote_handshake_port: int,
    remote_host: str,
    remote_engine_id: str,
    remote_multi_nodes_meta_mapping: dict[str, dict[str, Any]] | None,
) -> tuple[str, str]:
    rank = str(remote_handshake_port - base_port)
    if remote_multi_nodes_meta_mapping is None or remote_multi_nodes_meta_mapping.get(rank) is None:
        return remote_host, remote_engine_id
    info = remote_multi_nodes_meta_mapping[rank]
    return info.get("host", remote_host), info.get("engine_id", remote_engine_id)
