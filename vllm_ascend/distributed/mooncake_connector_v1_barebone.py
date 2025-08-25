# SPDX-License-Identifier: Apache-2.0
import contextlib
import hashlib
import json
import math
import os
import queue
import random
import socket
import struct
import threading
import time
from collections import defaultdict, deque
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, List, Optional, Tuple

import msgspec
import numpy as np
import numpy.typing as npt
import torch
import torch_npu
import zmq
from mooncake.engine import TransferEngine
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1, KVConnectorMetadata, KVConnectorRole)
from vllm.distributed.parallel_state import (get_tensor_model_parallel_rank,
                                             get_tp_group)
from vllm.utils import get_ip, logger, make_zmq_path, make_zmq_socket
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.request import RequestStatus

import vllm_ascend.envs as envs_ascend

if TYPE_CHECKING:
    from vllm.attention.backends.abstract import AttentionMetadata
    from vllm.forward_context import ForwardContext
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.request import Request

GET_META_MSG = b"get_meta_msg"
DONE_RECVING_MSG = b"done_recving_msg"
KV_CACHE_DTYPE = torch.bfloat16


class MooncakeAgentMetadata(msgspec.Struct, omit_defaults=True, dict=True):
    engine_id: str
    te_rpc_port: int
    kv_caches_base_addr: list[int]
    num_blocks: int


@dataclass
class ReqMeta:
    local_block_ids: list[int]
    remote_block_ids: list[int]
    remote_host: str
    remote_port: int
    remote_engine_id: str


class KVCacheTaskTracker:
    def __init__(self, tp_rank: int, local_engine_id: str, target_count: int):
        super().__init__()
        self.tp_rank = tp_rank
        self.local_engine_id = local_engine_id
        self.target_count = target_count

        self.done_task_lock = threading.Lock()
        self.done_task_counts: defaultdict[str, set[int]] = defaultdict(set)
        self.finished_requests: set[str] = set()

        self.socket_path = \
            f"ipc:///tmp/vllm_mooncake_connector_{self.local_engine_id}.ipc"
        if tp_rank == 0:
            self.listener = threading.Thread(
                target=self._listen_for_completion_signals,
                daemon=True,
                name="KVCacheTaskTrackerListenerThread")
            self.listener.start()
            self.socket = None
        else:
            self.listener = None
            self.socket = make_zmq_socket(
                ctx=zmq.Context(),
                path=self.socket_path,
                socket_type=zmq.PUSH,
                bind=False)
            logger.info("Connecting to transfer socket at %s", self.socket_path)

    def _listen_for_completion_signals(self):
        socket = make_zmq_socket(ctx=zmq.Context(), path=self.socket_path,
                                 socket_type=zmq.PULL, bind=True)
        logger.info("Listening for completion signals on %s", self.socket_path)

        while True:
            try:
                done_request_id, tp_rank = socket.recv_pyobj()
                logger.debug("Received completion notification for request: "
                             f"{done_request_id} from tp rank {tp_rank}")
                self._increment_task_count(done_request_id, tp_rank)
            except Exception as e:
                logger.error(f"Error in run_busy_loop: {e}")

    def update_done_task_count(self, request_id: str, tp_rank: int):
        if self.tp_rank == 0:
            self._increment_task_count(request_id, tp_rank)
        else:
            self.socket.send_pyobj((request_id, tp_rank))
            logger.debug("Sent done signal for request %s to tp 0", request_id)

    def _increment_task_count(self, request_id: str, tp_rank: int):
        with self.done_task_lock:
            if tp_rank in self.done_task_counts[request_id]:
                logger.warning(
                    f"Received duplicate done signal for request {request_id} "
                    f"from tp rank {tp_rank}. Ignoring.")
                return

            self.done_task_counts[request_id].add(tp_rank)
            if len(self.done_task_counts[request_id]) == self.target_count:
                self.finished_requests.add(request_id)
                self.done_task_counts.pop(request_id)
                logger.info("All transfers completed for request: "
                            f"{request_id}. Total ranks: "
                            f"{self.target_count}.")

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

         
class KVCacheSendingThread(threading.Thread):

    def __init__(self, tp_rank: int, decode_tp_size: int, local_engine_id: str,
                 side_channel_host: str, side_channel_port: int,
                 metadata: MooncakeAgentMetadata,
                 ready_event: threading.Event,
                 kv_caches: dict[str, Any]):
        super().__init__(daemon=True, name="KVCacheSendingThread")
        self.tp_rank = tp_rank
        self.decode_tp_size = decode_tp_size
        self.local_engine_id = local_engine_id
        self.side_channel_host = side_channel_host
        self.side_channel_port = side_channel_port
        self.metadata = metadata
        self.ready_event = ready_event
        self.kv_caches = kv_caches

        self.task_tracker = KVCacheTaskTracker(
            self.tp_rank,  self.local_engine_id, self.decode_tp_size)

    def get_and_clear_finished_requests(self) -> set[str]:
        """
        Get and clear the requests that have been completed.
        Returns:
            A set of request IDs that have been completed.
        """
        return self.task_tracker.get_and_clear_finished_requests()

    def run(self):
        """Run the thread to handle KV cache transfer requests."""

        encoder = msgspec.msgpack.Encoder()
        encoded_data = encoder.encode(self.metadata)
        size_in_bytes = len(encoded_data)
        logger.debug("Size of encoded MooncakeAgentMetadata: %s bytes",
                     str(size_in_bytes))

        # Listen for new requests for metadata.
        # NOTE(rob): we need each rank to have a unique port. This hack to keeps
        # us moving. We will switch when moving to etcd or where we have a
        # single ZMQ socket in the scheduler.
        handshake_port = self.side_channel_port + self.tp_rank
        path = make_zmq_path("tcp", self.side_channel_host, handshake_port)
        logger.info("Starting listening on path: %s", path)
        with zmq_ctx(zmq.ROUTER, path) as sock:
            self.ready_event.set()
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
                        logger.debug("Got DONE_RECVING_MSG for request %s",
                                     msg[1])
                        request_id, decode_tp_rank = msg[1], msg[2]
                        self.task_tracker.update_done_task_count(request_id,
                                                                 decode_tp_rank)
                        # Acknowledge the request completion.
                        while True:
                            try:
                                # Send ACK to the sender.
                                sock.send_multipart((identity, b"", b"ACK"),
                                                    flags=zmq.NOBLOCK)
                                break
                            except zmq.Again:
                                # If the socket is not ready, retry sending.
                                logger.debug(
                                    "Socket not ready, retrying to send ACK for "
                                    "request %s", msg[1])
                                time.sleep(0.01)
                    else:
                        logger.error(
                            "Connection listener got unexpected message %s",
                            msg)
                except Exception as e:
                    logger.error("Connection listener got exception %s: %s",
                                 type(e), e)


class KVCacheRecvingThread(threading.Thread):

    def __init__(self, tp_rank: int, tp_size: int, engine: TransferEngine,
                 local_engine_id: str, local_handshake_port: int,
                 local_kv_caches_base_addr: list[int], block_len: list[int],
                 ready_event: threading.Event, vllm_config: VllmConfig, kv_caches: dict[str, Any]):
        super().__init__(daemon=True, name="KVCacheRecvingThread")
        self.tp_rank = tp_rank
        self.tp_size = tp_size

        self.local_engine_id = local_engine_id
        self.local_handshake_port = local_handshake_port
        self.engine = engine
        self.ready_event = ready_event

        self.kv_caches_base_addr: dict[str, dict[int, list[int]]] = \
            defaultdict(dict)
        self.kv_caches_base_addr[local_engine_id][local_handshake_port] = \
            local_kv_caches_base_addr
        self.remote_te_port: dict[str, dict[int, int]] = \
            defaultdict(dict)
        self.block_len = block_len

        self.request_queue = queue.Queue()
        # TODO(jianzs): make this configurable
        max_workers = getattr(envs_ascend, 'MAX_TRANSFER_WORKERS', 32)
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

        self.task_tracker = KVCacheTaskTracker(
            self.tp_rank, self.local_engine_id, self.tp_size)

        self.encoder = msgspec.msgpack.Encoder()
        self.decoder = msgspec.msgpack.Decoder(MooncakeAgentMetadata)
        self.remote_sockets_lock = threading.Lock()
        self.remote_sockets: dict[str, deque[zmq.Socket]] = defaultdict(deque)
        self.remote_poller = zmq.Poller()
        self.timeout = 1.0  # seconds

        self.vllm_config = vllm_config
        self.model_config = self.vllm_config.model_config
        self.num_key_value_heads = self.model_config.hf_config.num_key_value_heads
        self.kv_caches = kv_caches

    def add_request(self, request_id: str, local_block_ids: list[int],
                    remote_block_ids: list[int], remote_engine_id: str,
                    remote_host: str, remote_handshake_port: int,
                    offset: int, num_need_pulls: int):
        """Add a new request to the queue for processing."""
        logger.debug(f"Adding request {request_id} to the queue.")
        self.request_queue.put({
            "request_id": request_id,
            "local_block_ids": local_block_ids,
            "remote_block_ids": remote_block_ids,
            "remote_engine_id": remote_engine_id,
            "remote_host": remote_host,
            "remote_handshake_port": remote_handshake_port,
            "offset": offset,
            "num_need_pulls": num_need_pulls
            })

    def get_and_clear_finished_requests(self) -> set[str]:
        """
        Get and clear the requests that have been completed.
        Returns:
            A set of request IDs that have been completed.
        """
        return self.task_tracker.get_and_clear_finished_requests()

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
        remote_host = req_meta["remote_host"]
        remote_handshake_port = req_meta["remote_handshake_port"]
        offset = req_meta["offset"]
        num_need_pulls = req_meta["num_need_pulls"]

        try:
            logger.debug(
                f"Starting to transfer KV cache for request {request_id}.")
            self._transfer_kv_cache(req_meta)
            logger.debug(
                f"Finished transferring KV cache for request {request_id}.")
        except Exception as e:
            logger.error("Failed to transfer KV cache for request "
                         f"{request_id}: {e}")
        finally:
            # Always send the done signal to the remote host to ensure proper
            # resource cleanup. Failing to do so may cause a memory leak on the
            # remote host.
            if offset == num_need_pulls - 1:
                self._send_done_recv_signal(request_id, remote_host, remote_handshake_port)
                self.task_tracker.update_done_task_count(request_id, self.tp_rank)
            self.request_queue.task_done()

    def _transfer_kv_cache(self, req_meta: dict[str, Any]):
        """Handle a KV cache transfer request."""
        request_id = req_meta["request_id"]
        remote_block_ids = req_meta["remote_block_ids"]
        local_block_ids = req_meta["local_block_ids"]
        remote_engine_id = req_meta["remote_engine_id"]
        remote_host = req_meta["remote_host"]
        remote_handshake_port = req_meta["remote_handshake_port"]
        offset = req_meta["offset"]
        self.num_need_pulls = req_meta["num_need_pulls"]

        # Full prefix cache hit: do not need to read remote blocks, just notify
        # P worker that we have the blocks we need.
        if len(local_block_ids) == 0:
            return

        # Check if we have the remote metadata cached.
        if remote_engine_id not in self.kv_caches_base_addr or \
            remote_handshake_port not in self.kv_caches_base_addr[remote_engine_id]:
            self._get_remote_metadata(remote_host, remote_handshake_port)

        if self.num_need_pulls == 1:
            grouped_remote_block_ids, grouped_local_block_ids = \
                    group_concurrent_contiguous(remote_block_ids, local_block_ids)
        else:
            remote_block_ids = list(map(lambda x: [x], remote_block_ids))
            local_block_ids = list(map(lambda x: [x], local_block_ids))
            grouped_remote_block_ids, grouped_local_block_ids = remote_block_ids, local_block_ids
        num_transfer_groups = len(grouped_remote_block_ids)

        remote_kv_caches_base_addrs = \
            self.kv_caches_base_addr[remote_engine_id][remote_handshake_port]
        local_kv_caches_base_addrs = \
            self.kv_caches_base_addr[self.local_engine_id][self.local_handshake_port]
        remote_transfer_port = self.remote_te_port[remote_engine_id][remote_handshake_port]
        num_blocks = len(local_block_ids)
        session_id = f"{remote_host}:{remote_transfer_port}"

        req_start_time = time.perf_counter()
        src_list, dst_list, length_list = [], [], []
        for k, (src_layer_base_addr, dst_layer_base_addr) in enumerate(
                zip(local_kv_caches_base_addrs, remote_kv_caches_base_addrs)):
            block_len = self.block_len[k % 2]
            inner_block_len = block_len // self.num_need_pulls
            for remote_block_id, local_block_id in zip(grouped_remote_block_ids, grouped_local_block_ids):
                src = src_layer_base_addr + local_block_id[0] * block_len + offset * inner_block_len
                dst = dst_layer_base_addr + remote_block_id[0] * inner_block_len
                length = inner_block_len * len(local_block_id)
                src_list.append(src)
                dst_list.append(dst)
                length_list.append(length)

        ret = self.engine.batch_transfer_sync_read(session_id, src_list, dst_list,
                                                length_list)
        if ret < 0:
            logger.error(
                "Mooncake transfer failed for request %s: ", req_meta["request_id"])
            raise RuntimeError(f"Mooncake transfer failed, ret: {ret}")

        req_end_time = time.perf_counter()
        req_transfer_elapsed = (req_end_time - req_start_time) * 1000
        logger.info("KV cache transfer for request %s took %.2f ms (%d groups,"
                    " %d blocks). local_ip %s local_device_id %s remote_session_id %s", request_id, req_transfer_elapsed,
                    num_transfer_groups, num_blocks, get_ip(), self.tp_rank, session_id)
        if self.num_need_pulls > 1 and offset == self.num_need_pulls -1:
            self._cat_kv_cache(local_block_ids)

    def _cat_kv_cache(self, block_ids: list[list[int]]):
        # get necessary params from config
        head_dim = self.model_config.hf_config.head_dim
        block_size = self.vllm_config.cache_config.block_size
        num_kv_head = self.model_config.hf_config.num_key_value_heads

        dtype = list(self.kv_caches.values())[0][0].dtype
        block_ids = [item for sublist in block_ids for item in sublist]
        block_ids_tensor = torch.tensor(block_ids, dtype=torch.int32)
        num_blocks = len(block_ids)
        block_len = num_blocks * block_size
        block_table = block_ids_tensor.view(1, -1).npu()
        block_len_tensor = torch.tensor([num_blocks * block_size], dtype=torch.int32).npu()
        seq_start_locate = torch.tensor([0], dtype=torch.int32).npu()

        k_buffer = torch.empty(block_len, num_kv_head, head_dim, dtype=dtype).npu()
        v_buffer = torch.empty(block_len, num_kv_head, head_dim, dtype=dtype).npu()

        block_offsets = torch.arange(0, block_size, dtype=torch.int32)
        slot_mapping = block_offsets.reshape(
            (1, block_size)) + block_ids_tensor.reshape(
            (num_blocks, 1)) * block_size
        slot_mapping = slot_mapping.flatten().npu()

        for kv_cache_layer in self.kv_caches.values():
            k_cache_layer, v_cache_layer = kv_cache_layer
            torch_npu.atb.npu_paged_cache_load(
                k_cache_layer,
                v_cache_layer,
                block_table,
                block_len_tensor,
                seq_starts=seq_start_locate,
                key=k_buffer,
                value=v_buffer,
            )

            k_buffer = k_buffer.view(num_blocks, self.num_need_pulls, block_size, -1)
            k_buffer.transpose_(1, 2)
            k_buffer = k_buffer.contiguous().view(block_len, num_kv_head, -1)
            v_buffer = v_buffer.view(num_blocks, self.num_need_pulls, block_size, -1)
            v_buffer.transpose_(1, 2)
            v_buffer = v_buffer.contiguous().view(block_len, num_kv_head, -1)

            torch_npu._npu_reshape_and_cache(
                key=k_buffer,
                value=v_buffer,
                key_cache=k_cache_layer,
                value_cache=v_cache_layer,
                slot_indices=slot_mapping,
            )
        del k_buffer
        del v_buffer

    def _get_remote_metadata(self, remote_host: str,
                             remote_handshake_port: int) -> None:
        """Get the metadata from the remote host."""
        sock: Optional[zmq.Socket] = None
        try:
            sock = self._get_remote_socket(remote_host, remote_handshake_port)
            ensure_zmq_send(sock, self.encoder.encode((GET_META_MSG, "")))
            metadata_bytes = ensure_zmq_recv(sock, self.remote_poller)
            agent_meta = self.decoder.decode(metadata_bytes)
            engine_id = agent_meta.engine_id
            assert engine_id != self.local_engine_id, (
                f"Conflict engine id {engine_id} with local engine id "
                f"{self.local_engine_id}.")
            self.kv_caches_base_addr[engine_id][remote_handshake_port] = \
                agent_meta.kv_caches_base_addr
            self.remote_te_port[engine_id][remote_handshake_port] = \
                agent_meta.te_rpc_port
        finally:
            if sock is not None:
                self._return_remote_socket(sock, remote_host,
                                           remote_handshake_port)
                logger.debug("Returned socket to pool for %s:%d",
                             remote_host, remote_handshake_port)

    def _send_done_recv_signal(self, request_id: str, remote_host: str,
                               remote_handshake_port: int):
        logger.debug("Sending done recving signal for request %s to %s:%d",
                     request_id, remote_host, remote_handshake_port)
        sock: Optional[zmq.Socket] = None
        try:
            sock = self._get_remote_socket(remote_host, remote_handshake_port)
            data_bytes = self.encoder.encode((DONE_RECVING_MSG, request_id,
                                            self.tp_rank))
            ensure_zmq_send(sock, data_bytes)
            resp = ensure_zmq_recv(sock, self.remote_poller,
                                   timeout=self.timeout)
            logger.debug(f"Received response for request {request_id}: {resp}")
            if resp != b"ACK":
                logger.error("Failed to receive ACK for request %s from %s:%d",
                            request_id, remote_host, remote_handshake_port)
                raise RuntimeError(f"Failed to receive ACK, resp: {resp}")
        finally:
            if sock is not None:
                self._return_remote_socket(sock, remote_host,
                                           remote_handshake_port)
                logger.debug("Returned socket to pool for %s:%d",
                             remote_host, remote_handshake_port)

    def _get_remote_socket(self, remote_host: str,
                    remote_handshake_port: int) -> zmq.Socket:
        """Get a socket to the remote host."""
        remote_path = make_zmq_path("tcp", remote_host, remote_handshake_port)
        with self.remote_sockets_lock:
            if self.remote_sockets[remote_path]:
                return self.remote_sockets[remote_path].popleft()

            ctx = zmq.Context()
            sock = make_zmq_socket(ctx=ctx,
                                   path=remote_path,
                                   socket_type=zmq.REQ,
                                   bind=False)
            sock.setsockopt(zmq.SNDTIMEO, int(self.timeout * 1000))
            self.remote_poller.register(sock, zmq.POLLIN)
            return sock

    def _return_remote_socket(self, sock: zmq.Socket, remote_host: str,
                              remote_handshake_port: int) -> None:
        """Return the remote socket to the pool."""
        remote_path = make_zmq_path("tcp", remote_host, remote_handshake_port)
        with self.remote_sockets_lock:
            self.remote_sockets[remote_path].append(sock)


class MooncakeConnectorMetadata(KVConnectorMetadata):

    def __init__(self):
        self.requests: dict[str, ReqMeta] = {}

    def add_new_req(
        self,
        request_id: str,
        local_block_ids: list[int],
        kv_transfer_params: dict[str, Any],
    ):
        self.requests[request_id] = ReqMeta(
            local_block_ids=local_block_ids,
            remote_block_ids=kv_transfer_params["remote_block_ids"],
            remote_engine_id=kv_transfer_params["remote_engine_id"],
            remote_host=kv_transfer_params["remote_host"],
            remote_port=kv_transfer_params["remote_port"],
        )


class MooncakeConnectorV1_barebone(KVConnectorBase_V1):

    def __init__(self, vllm_config: VllmConfig, role: KVConnectorRole):
        assert vllm_config.kv_transfer_config is not None
        self.engine_id = vllm_config.kv_transfer_config.engine_id

        if role == KVConnectorRole.SCHEDULER:
            self.connector_scheduler: Optional[MooncakeConnectorScheduler] = \
                MooncakeConnectorScheduler(vllm_config, str(self.engine_id))
            self.connector_worker: Optional[MooncakeConnectorWorker] = None
        elif role == KVConnectorRole.WORKER:
            self.connector_scheduler = None
            self.connector_worker = MooncakeConnectorWorker(
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
        assert isinstance(self._connector_metadata, MooncakeConnectorMetadata)
        self.connector_worker.start_load_kv(self._connector_metadata)

    def wait_for_layer_load(self, layer_name: str) -> None:
        """MooncakeConnector does not do layerwise saving."""
        pass

    def save_kv_layer(self, layer_name: str, kv_layer: torch.Tensor,
                      attn_metadata: "AttentionMetadata", **kwargs) -> None:
        """MooncakeConnector does not save explicitly."""
        pass

    def wait_for_save(self):
        """MooncakeConnector does not save explicitly."""
        pass


class MooncakeConnectorScheduler:
    """Implementation of Scheduler side methods"""

    def __init__(self, vllm_config: VllmConfig, engine_id: str):
        self.vllm_config = vllm_config
        self.block_size = vllm_config.cache_config.block_size
        self.engine_id = engine_id
        self.local_ip = get_ip()
        logger.info("Initializing Mooncake Scheduler %s", engine_id)

        self.side_channel_host = get_local_ip_by_remote()
        self.max_device_id = vllm_config.parallel_config.tensor_parallel_size * \
                             vllm_config.parallel_config.data_parallel_size

        # Handshake base port
        self.side_channel_port = (
            envs_ascend.VLLM_BASE_PORT + 
            vllm_config.parallel_config.data_parallel_rank_local *
            vllm_config.parallel_config.tensor_parallel_size)

        # Requests that need to start recv.
        # New requests are added by update_state_after_alloc in
        # the scheduler. Used to make metadata passed to Worker.
        self._reqs_need_recv: dict[str, tuple[Request, list[int]]] = {}

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
            "MooncakeConnector get_num_new_matched_tokens: "
            "num_computed_tokens=%s, kv_transfer_params=%s",
            num_computed_tokens, params)

        if params is not None and params.get("do_remote_prefill"):
            assert num_computed_tokens == 0, "Currently only support " \
                "prefill with num_computed_tokens == 0."
            # Assume that the request's KV cache is already fully prefilled and
            # can be fetched entirely from the prefill node.
            count = max(len(request.prompt_token_ids) - 1, 0)
            if count > 0:
                return count, True

        # No remote prefill for this request.
        return 0, False

    def update_state_after_alloc(self, request: "Request",
                                 blocks: "KVCacheBlocks",
                                 num_external_tokens: int):

        params = request.kv_transfer_params
        logger.debug(
            "MooncakeConnector update_state_after_alloc: "
            "num_external_tokens=%s, kv_transfer_params=%s",
            num_external_tokens, params)

        if params is not None and params.get("do_remote_prefill"):
            if params.get("remote_block_ids"):
                if all(p in params for p in ("remote_engine_id", "remote_host",
                                             "remote_port")):
                    local_block_ids = (blocks.get_unhashed_block_ids()
                                       if num_external_tokens > 0 else [])
                    # Get unhashed blocks to pull from remote.
                    self._reqs_need_recv[request.request_id] = (
                        request, local_block_ids)
                else:
                    logger.warning(
                        "Got invalid KVTransferParams: %s. This "
                        "request will not utilize KVTransfer", params)
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
            "MooncakeConnector request_finished, request_status=%s, "
            "kv_transfer_params=%s", request.status, params)

        if (params is None or not params.get("do_remote_decode")
                or request.status != RequestStatus.FINISHED_LENGTH_CAPPED):
            return False, None

        computed_block_ids = block_ids
        delay_free_blocks = len(computed_block_ids) > 0
        if delay_free_blocks:
            logger.info("Delaying free of %d blocks for request %s",
                        len(computed_block_ids), request.request_id)
        return delay_free_blocks, dict(
            do_remote_prefill=True,
            do_remote_decode=False,
            remote_block_ids=computed_block_ids,
            remote_engine_id=self.engine_id,
            remote_host=self.side_channel_host,
            remote_port=self.side_channel_port,
            last_token_id=request.output_token_ids[-1],
        )


class MooncakeConnectorWorker:
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
        self.engine_id = engine_id
        self.tp_rank = get_tensor_model_parallel_rank()
        self.tp_size = vllm_config.parallel_config.tensor_parallel_size
        self.tp_group = get_tp_group()
        self.dp_rank = vllm_config.parallel_config.data_parallel_rank_local
        self.dp_size = vllm_config.parallel_config.data_parallel_size_local
        self.kv_caches: dict[str, torch.Tensor] = {}
        self.side_channel_host = get_local_ip_by_remote()
        self.max_device_id = self.tp_size * self.dp_size
        self.kv_role = vllm_config.kv_transfer_config.kv_role

        # Handshake base port
        self.side_channel_port = (
            envs_ascend.VLLM_BASE_PORT +
            vllm_config.parallel_config.data_parallel_rank_local *
            vllm_config.parallel_config.tensor_parallel_size)
        self.handshake_port = self.side_channel_port + self.tp_rank
        self.sockets: dict = {}

        # get tp device id 
        # TODO(kw): https://github.com/vllm-project/vllm-ascend/pull/940 
        # introducing some changes
        device_ids = os.getenv("ASCEND_RT_VISIBLE_DEVICES", None)
        if device_ids is None:
            device_ids = list(
                range(self.dp_rank * self.tp_size,
                      (self.dp_rank + 1) * self.tp_size))
        else:
            device_ids = list(map(int, device_ids.split(',')))
        assert len(device_ids) > self.tp_rank
        self.device_id = device_ids[self.tp_rank]

        self._initialize(
            hostname=self.side_channel_host + ':' + "0" + ':' + 'npu_' \
                + str(self.device_id),
            device_name=None)
        self.te_rpc_port = self.engine.get_rpc_port()

        # Background thread for sending or receiving KV caches.
        self.kv_send_thread: Optional[KVCacheSendingThread] = None
        self.kv_recv_thread: Optional[KVCacheRecvingThread] = None

        # kv_transfer variables
        self.vllm_config = vllm_config
        self.block_size = vllm_config.cache_config.block_size
        self.num_key_value_heads = self.vllm_config.model_config.hf_config.num_key_value_heads
        num_d_block_heads = max(1, self.num_key_value_heads // self.tp_size)
        num_p_block_heads = max(1, self.num_key_value_heads // self._prefill_tp_size)
        if self.vllm_config.model_config.is_deepseek_mla:
            self.num_need_pulls = 1
        else:
            self.num_need_pulls = num_d_block_heads // num_p_block_heads

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
        ret_value = self.engine.initialize(hostname, "P2PHANDSHAKE",
                                           envs_ascend.MOONCAKE_CONNECTOR_PROTOCOL,
                                           device_name)
        if ret_value != 0:
            raise RuntimeError(
                f"Mooncake initialization failed with ret_value: {ret_value}")

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        """Register the KV Cache data."""

        _, first_kv_cache_tuple = next(iter(kv_caches.items()))
        first_kv_cache = first_kv_cache_tuple[0]

        # TODO(tms): Find a more robust way to detect and handle MLA
        self.num_blocks = first_kv_cache.shape[0]
        block_rank = 3
        kv_elem_size = first_kv_cache.element_size()
        # MLA case.[num_block, block_size, 1, hidden_dims],
        # block_shape_1 -> nope, block_shape_2 -> rope
        # non-MLA case.[num_block, block_size, num_heads, head_dims]
        # block_shape_1 -> k, block_shape_1 -> v
        block_shape_1 = first_kv_cache_tuple[0].shape[-block_rank:]
        block_shape_2 = first_kv_cache_tuple[1].shape[-block_rank:]
        self.block_len = [
            kv_elem_size * math.prod(block_shape_1),
            kv_elem_size * math.prod(block_shape_2)
        ]
        logger.info(
            "num_blocks: %s, block_shape_1: %s, block_shape_2: %s",
            self.num_blocks, block_shape_1, block_shape_1)

        self.kv_caches = kv_caches
        kv_caches_base_addr = []
        region_len_1 = self.num_blocks * self.block_len[0]
        region_len_2 = self.num_blocks * self.block_len[1]
        for cache_or_caches in kv_caches.values():
            # Normalize to always be a list of caches
            assert len(cache_or_caches) == 2, "per-layer kv_caches must equal to 2"
            base_addr_1 = cache_or_caches[0].data_ptr()
            base_addr_2 = cache_or_caches[1].data_ptr()
            kv_caches_base_addr.append(base_addr_1)
            self._register(base_addr_1, region_len_1)
            kv_caches_base_addr.append(base_addr_2)
            self._register(base_addr_2, region_len_2)

        # After KV Caches registered, start the sending or receiving thread.
        metadata = MooncakeAgentMetadata(
            engine_id=self.engine_id,
            te_rpc_port=self.te_rpc_port,
            kv_caches_base_addr=kv_caches_base_addr,
            num_blocks=self.num_blocks,
        )

        ready_event = threading.Event()
        if self.kv_role == 'kv_producer':
            self.kv_send_thread = KVCacheSendingThread(
                self.tp_rank, self._decode_tp_size, self.engine_id,
                self.side_channel_host, self.side_channel_port, metadata,
                ready_event, self.kv_caches)
            self.kv_send_thread.start()
        else:
            self.kv_recv_thread = KVCacheRecvingThread(
                self.tp_rank, self.tp_size, self.engine, self.engine_id,
                self.handshake_port, kv_caches_base_addr, self.block_len,
                ready_event, self.vllm_config, self.kv_caches)
            self.kv_recv_thread.start()
        ready_event.wait()

    def _register(self, ptr, length):
        logger.info(
            "Registering KV cache: ptr=0x%x, length=%d, num_blocks=%d, "
            "block_lens=%s", ptr, length, self.num_blocks, self.block_len)
        ret_value = self.engine.register_memory(ptr, length)
        if ret_value != 0:
            raise RuntimeError("Mooncake memory registration failed.")

    def get_finished(self) -> tuple[set[str], set[str]]:
        done_sending = (self.kv_send_thread.get_and_clear_finished_requests()
                        if self.kv_role == 'kv_producer' else set())
        done_recving = (self.kv_recv_thread.get_and_clear_finished_requests()
                        if self.kv_role == 'kv_consumer' else set())
        if self.tp_rank == 0:
            logger.debug(
                "Number of completed KV cache send requests: %d, receive "
                "requests: %d", len(done_sending), len(done_recving))
        return done_sending, done_recving

    def start_load_kv(self, metadata: MooncakeConnectorMetadata):
        """Start loading KV blocks from remote engine."""
        for req_id, meta in metadata.requests.items():
            logger.debug(
                "start_load_kv for request %s from remote engine %s. "
                "Num local_block_ids: %s. Num remote_block_ids: %s. ", req_id,
                meta.remote_engine_id, len(meta.local_block_ids),
                len(meta.remote_block_ids))

            choosen_rank_list = self._get_remote_tp_rank(req_id)
            remote_handshake_port_list = [x + meta.remote_port for x in choosen_rank_list]
            for i in range(self.num_need_pulls):
                self.kv_recv_thread.add_request(
                    request_id=req_id,
                    local_block_ids=meta.local_block_ids,
                    remote_block_ids=meta.remote_block_ids,
                    remote_engine_id=meta.remote_engine_id,
                    remote_host=meta.remote_host,
                    remote_handshake_port=remote_handshake_port_list[i],
                    offset=i,
                    num_need_pulls=self.num_need_pulls
                )

    def _get_remote_tp_rank(self, req_id: str) -> List[int]:
        return self._get_remote_tp_ranks_for_req(req_id)[self.tp_rank]

    def _get_remote_tp_ranks_for_req(self, req_id: str) -> List[List[int]]:
        if self._prefill_tp_size == self._decode_tp_size:
            result = list(map(lambda x: [x], range(self._prefill_tp_size)))
            return result

        seed = string_to_int64_hash(req_id)
        rand = random.Random(seed)
        sampled_nums = []
        ori_data = np.arange(self._prefill_tp_size)

        # random split prefill tp list
        if self._prefill_tp_size > self.num_key_value_heads:
            # TODO: can be modified
            num_groups = len(ori_data) // self.num_key_value_heads
            ori_data = ori_data.reshape(-1, num_groups)
            rand_group_index = rand.sample(range(num_groups), \
                max(self.tp_size // self.num_key_value_heads, 1)) # random choose a group

            choosen_group = ori_data[:, [rand_group_index]]
            flattened = choosen_group.reshape(-1).tolist()
            sampled_nums = [flattened[i:i + self.num_need_pulls] for i in range(0, len(flattened), self.num_need_pulls)]

        # non-random split
        else:
            group_size = self._prefill_tp_size // self.tp_size
            for i in range(self.tp_size):
                slice = ori_data[i*group_size : (i+1)*group_size]
                sampled_nums.append(slice)
        return sampled_nums


@contextlib.contextmanager
def zmq_ctx(socket_type: Any, addr: str) -> Iterator[zmq.Socket]:
    """Context manager for a ZMQ socket"""

    if socket_type not in (zmq.ROUTER, zmq.REQ, zmq.DEALER):
        raise ValueError(f"Unexpected socket type: {socket_type}")

    ctx: Optional[zmq.Context] = None
    try:
        ctx = zmq.Context()
        yield make_zmq_socket(ctx=ctx,
                              path=addr,
                              socket_type=socket_type,
                              bind=socket_type == zmq.ROUTER)
    finally:
        if ctx is not None:
            ctx.destroy(linger=0)


def get_local_ip_by_remote() -> str:
    # try ipv4
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))  # Doesn't need to be reachable
        return s.getsockname()[0]
    except Exception:
        logger.error("get_local_ip_by_remote Error.")

    try:
        hostname = socket.gethostname()
        ip = socket.gethostbyname(hostname)
        if ip and ip != "127.0.0.1" and ip != "0.0.0.0":
            return ip
    except Exception:
        logger.error("get_local_ip_by_remote Error 2.")

    # try ipv6
    try:
        s = socket.socket(socket.AF_INET6, socket.SOCK_DGRAM)
        # Google's public DNS server, see
        # https://developers.google.com/speed/public-dns/docs/using#addresses
        s.connect(
            ("2001:4860:4860::8888", 80
             ))  # Doesn't need to be reachable  # Doesn't need to be reachable
        return s.getsockname()[0]
    except Exception:
        raise ValueError("Can not get local ip")


def group_concurrent_contiguous(
    src: List[int], dst: List[int]
) -> Tuple[List[npt.NDArray[np.int64]], List[npt.NDArray[np.int64]]]:
    """Vectorised NumPy implementation."""
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


def ensure_zmq_send(socket: zmq.Socket, data: bytes, max_retries: int = 3):
    retries_left = max_retries
    while True:
        try:
            socket.send(data)
            return
        except zmq.ZMQError as e:
            retries_left -= 1
            if retries_left > 0:
                logger.warning(f"Send failed: {e}, retrying... ({retries_left} "
                               "attempts left)")
                time.sleep(0.1)
            else:
                logger.error(f"Send failed after all retries: {e}")
                raise RuntimeError(f"Failed to send data after {max_retries} "
                                   f"retries: {e}")

           
def ensure_zmq_recv(socket: zmq.Socket, poller: zmq.Poller,
                    timeout: float = 1.0, max_retries: int = 3) -> bytes:
    retries_left = max_retries
    while True:
        try:
            if dict(poller.poll(int(timeout * 1000))):  # milliseconds
                data = socket.recv()
                return data
            else:
                raise zmq.ZMQError("Receive timeout")
        except zmq.ZMQError as e:
            retries_left -= 1
            if retries_left > 0:
                logger.warning(f"Receive failed: {e}, retrying... "
                               f"({retries_left} attempts left)")
                time.sleep(0.1)
            else:
                logger.error(f"Receive failed after all retries: {e}")
                raise RuntimeError(f"Failed to receive data after {max_retries} "
                                   f"retries: {e}")
