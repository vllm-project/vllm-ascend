# SPDX-License-Identifier: Apache-2.0
import contextlib
import hashlib
import math
import queue
import random
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
import zmq
from mooncake.engine import TransferEngine  # type: ignore
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

    def __init__(self):
        super().__init__()

        self.done_task_lock = threading.Lock()
        self.finished_requests: set[str] = set()
        # Only used in prefill node. Tracks requests whose kv blocks freeing is
        # intentionally delayed. Each entry is a tuple of (request_id,
        # timestamp). If a request remains in this queue for too long, it will
        # be force-freed.
        self.delayed_free_requests: deque[Tuple[str, float]] = deque()

    def update_done_task_count(self, request_id: str):
        with self.done_task_lock:
            self.finished_requests.add(request_id)
            self._remove_delayed_requests(request_id)

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
            self.delayed_free_requests.append((request_id, delay_start_time))

    def _retrieve_expired_requests(self):
        """Retrieve all expired delayed requests."""
        expired_requests: set[str] = set()
        # Free delayed requests if they exceed the timeout
        current_time = time.time()
        while self.delayed_free_requests:
            request_id, delay_start_time = self.delayed_free_requests[0]
            if (current_time - delay_start_time
                    > envs_ascend.VLLM_ASCEND_KVCACHE_DELAY_FREE_TIMEOUT):
                self.delayed_free_requests.popleft()
                expired_requests.add(request_id)
                logger.info("Force freed request: %s", request_id)
            else:
                break
        return expired_requests

    def _remove_delayed_requests(self, request_id: str):
        """Remove all delayed free requests matching the given request_id."""
        self.delayed_free_requests = deque(
            (r, t) for r, t in self.delayed_free_requests if r != request_id)


class KVCacheSendingThread(threading.Thread):

    def __init__(self, tp_rank: int, decode_tp_size: int, local_engine_id: str,
                 side_channel_host: str, side_channel_port: int,
                 metadata: MooncakeAgentMetadata,
                 ready_event: threading.Event):
        super().__init__(daemon=True, name="KVCacheSendingThread")
        self.tp_rank = tp_rank
        self.decode_tp_size = decode_tp_size
        self.local_engine_id = local_engine_id
        self.side_channel_host = side_channel_host
        self.side_channel_port = side_channel_port
        self.metadata = metadata
        self.ready_event = ready_event

        self.task_tracker = KVCacheTaskTracker()

    def get_and_clear_finished_requests(self) -> set[str]:
        """
        Get and clear the requests that have been completed.
        Returns:
            A set of request IDs that have been completed.
        """
        return self.task_tracker.get_and_clear_finished_requests()

    def add_delayed_request(self, request_id: str, delay_start_time: float):
        return self.task_tracker.add_delayed_request(request_id,
                                                     delay_start_time)

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
        with zmq_ctx(zmq.ROUTER, path) as sock:  # type: ignore
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
                        request_id = msg[1]
                        self.task_tracker.update_done_task_count(request_id)
                        # Acknowledge the request completion.
                        while True:
                            try:
                                # Send ACK to the sender.
                                sock.send_multipart(
                                    (identity, b"", b"ACK"),
                                    flags=zmq.NOBLOCK)  # type: ignore
                                break
                            except zmq.Again:  # type: ignore
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
                 ready_event: threading.Event):
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
        # TODO(jianzs): find a better way to detect MLA.
        self.use_mla = len(block_len) == 2

        self.request_queue: queue.Queue[Any] = queue.Queue()
        # TODO(jianzs): make this configurable
        self.executor = ThreadPoolExecutor(max_workers=32)

        self.task_tracker = KVCacheTaskTracker()

        self.encoder = msgspec.msgpack.Encoder()
        self.decoder = msgspec.msgpack.Decoder(MooncakeAgentMetadata)
        self.remote_sockets_lock = threading.Lock()
        self.remote_sockets: dict[  # type: ignore
            str, deque[zmq.Socket]] = defaultdict(  # type: ignore
                deque)
        self.remote_poller = zmq.Poller()  # type: ignore
        self.timeout = 1.0  # seconds

    def add_request(self, request_id: str, local_block_ids: list[int],
                    remote_block_ids: list[int], remote_engine_id: str,
                    remote_host: str, remote_handshake_port: int):
        """Add a new request to the queue for processing."""
        logger.debug(f"Adding request {request_id} to the queue.")
        self.request_queue.put({
            "request_id": request_id,
            "local_block_ids": local_block_ids,
            "remote_block_ids": remote_block_ids,
            "remote_engine_id": remote_engine_id,
            "remote_host": remote_host,
            "remote_handshake_port": remote_handshake_port,
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
            self.task_tracker.update_done_task_count(request_id)
            # Always send the done signal to the remote host to ensure proper
            # resource cleanup. Failing to do so may cause a memory leak on the
            # remote host.
            self._send_done_recv_signal(request_id, remote_host,
                                        remote_handshake_port)
            self.request_queue.task_done()

    def _transfer_kv_cache(self, req_meta: dict[str, Any]):
        """Handle a KV cache transfer request."""
        request_id = req_meta["request_id"]
        remote_block_ids = req_meta["remote_block_ids"]
        local_block_ids = req_meta["local_block_ids"]
        remote_engine_id = req_meta["remote_engine_id"]
        remote_host = req_meta["remote_host"]
        remote_handshake_port = req_meta["remote_handshake_port"]

        # Full prefix cache hit: do not need to read remote blocks, just notify
        # P worker that we have the blocks we need.
        if len(local_block_ids) == 0:
            return

        # Check if we have the remote metadata cached.
        if remote_engine_id not in self.kv_caches_base_addr or \
                remote_handshake_port not in self.kv_caches_base_addr[remote_engine_id]:
            self._get_remote_metadata(remote_host, remote_handshake_port)

        grouped_remote_block_ids, grouped_local_block_ids = \
            group_concurrent_contiguous(remote_block_ids, local_block_ids)
        remote_kv_caches_base_addrs = \
            self.kv_caches_base_addr[remote_engine_id][remote_handshake_port]
        local_kv_caches_base_addrs = \
            self.kv_caches_base_addr[self.local_engine_id][self.local_handshake_port]

        req_start_time = time.perf_counter()
        num_transfer_groups = len(grouped_remote_block_ids)
        num_blocks = len(local_block_ids)

        remote_transfer_port = self.remote_te_port[remote_engine_id][
            remote_handshake_port]
        session_id = f"{remote_host}:{remote_transfer_port}"
        src_list, dst_list, length_list = [], [], []
        for k, (src_layer_base_addr, dst_layer_base_addr) in enumerate(
                zip(local_kv_caches_base_addrs, remote_kv_caches_base_addrs)):
            block_len = (self.block_len[k % 2]
                         if self.use_mla else self.block_len[0])
            for i, remote_block_id in enumerate(grouped_remote_block_ids):
                local_block_ids = grouped_local_block_ids[i]
                src = src_layer_base_addr + local_block_ids[0] * block_len
                dst = dst_layer_base_addr + remote_block_id[0] * block_len
                length = len(local_block_ids) * block_len
                src_list.append(src)
                dst_list.append(dst)
                length_list.append(length)
        ret = self.engine.batch_transfer_sync_read(session_id, src_list,
                                                   dst_list, length_list)
        if ret < 0:
            logger.error("Mooncake transfer failed for request %s",
                         req_meta["request_id"])
            raise RuntimeError(f"Mooncake transfer failed, ret: {ret}")

        req_end_time = time.perf_counter()
        req_transfer_elapsed = (req_end_time - req_start_time) * 1000
        logger.info(
            "KV cache transfer for request %s took %.2f ms (%d groups,"
            " %d blocks).", request_id, req_transfer_elapsed,
            num_transfer_groups, num_blocks)

    def _get_remote_metadata(self, remote_host: str,
                             remote_handshake_port: int) -> None:
        """Get the metadata from the remote host."""
        sock: Optional[zmq.Socket] = None  # type: ignore
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
                logger.debug("Returned socket to pool for %s:%d", remote_host,
                             remote_handshake_port)

    def _send_done_recv_signal(self, request_id: str, remote_host: str,
                               remote_handshake_port: int):
        logger.debug("Sending done recving signal for request %s to %s:%d",
                     request_id, remote_host, remote_handshake_port)
        sock: Optional[zmq.Socket] = None  # type: ignore
        try:
            sock = self._get_remote_socket(remote_host, remote_handshake_port)
            data_bytes = self.encoder.encode((DONE_RECVING_MSG, request_id))
            ensure_zmq_send(sock, data_bytes)
            resp = ensure_zmq_recv(sock,
                                   self.remote_poller,
                                   timeout=self.timeout)
            logger.debug(
                f"Received response for request {request_id}: {resp.decode('utf-8')}"
            )
            if resp != b"ACK":
                logger.error("Failed to receive ACK for request %s from %s:%d",
                             request_id, remote_host, remote_handshake_port)
                raise RuntimeError(
                    f"Failed to receive ACK, resp: {resp.decode('utf-8')}")
        finally:
            if sock is not None:
                self._return_remote_socket(sock, remote_host,
                                           remote_handshake_port)
                logger.debug("Returned socket to pool for %s:%d", remote_host,
                             remote_handshake_port)

    def _get_remote_socket(
            self, remote_host: str,
            remote_handshake_port: int) -> zmq.Socket:  # type: ignore
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
                bind=False)
            sock.setsockopt(
                zmq.SNDTIMEO,  # type: ignore
                int(self.timeout * 1000))
            self.remote_poller.register(sock, zmq.POLLIN)  # type: ignore
            return sock

    def _return_remote_socket(
            self,
            sock: zmq.Socket,  # type: ignore
            remote_host: str,
            remote_handshake_port: int) -> None:
        """Return the remote socket to the pool."""
        remote_path = make_zmq_path("tcp", remote_host, remote_handshake_port)
        with self.remote_sockets_lock:
            self.remote_sockets[remote_path].append(sock)


class MooncakeConnectorMetadata(KVConnectorMetadata):

    def __init__(self):
        self.requests: dict[str, ReqMeta] = {}
        self.requests_to_send: dict[str, float] = {}

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


class MooncakeConnector(KVConnectorBase_V1):

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
            self._reqs_need_send[request.request_id] = time.time()

        return delay_free_blocks, dict(
            do_remote_prefill=True,
            do_remote_decode=False,
            remote_block_ids=computed_block_ids,
            remote_engine_id=self.engine_id,
            remote_host=self.side_channel_host,
            remote_port=self.side_channel_port,
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
        self.kv_send_thread: Optional[KVCacheSendingThread] = None
        self.kv_recv_thread: Optional[KVCacheRecvingThread] = None

        self.vllm_config = vllm_config
        self.block_size = vllm_config.cache_config.block_size

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

        # After KV Caches registered, start the sending or receiving thread.
        metadata = MooncakeAgentMetadata(
            engine_id=self.engine_id,
            te_rpc_port=self.te_rpc_port,
            kv_caches_base_addr=kv_caches_base_addr,
            num_blocks=self.num_blocks,
        )

        ready_event = threading.Event()
        if self.kv_role == 'kv_producer':
            self.kv_send_thread = KVCacheSendingThread(self.tp_rank,
                                                       self._decode_tp_size,
                                                       self.engine_id,
                                                       self.side_channel_host,
                                                       self.side_channel_port,
                                                       metadata, ready_event)
            self.kv_send_thread.start()
        else:
            self.kv_recv_thread = KVCacheRecvingThread(
                self.tp_rank, self.tp_size, self.engine, self.engine_id,
                self.handshake_port, kv_caches_base_addr, self.block_len,
                ready_event)
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
        done_sending = (
            self.kv_send_thread.
            get_and_clear_finished_requests(  # type: ignore[union-attr]
            ) if self.kv_role == 'kv_producer' else set())
        done_recving = (
            self.kv_recv_thread.
            get_and_clear_finished_requests(  # type: ignore[union-attr]
            ) if self.kv_role == 'kv_consumer' else set())
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

            remote_handshake_port = meta.remote_port + \
                                    self._get_remote_tp_rank(req_id)
            self.kv_recv_thread.add_request(  # type: ignore[union-attr]
                request_id=req_id,
                local_block_ids=meta.local_block_ids,
                remote_block_ids=meta.remote_block_ids,
                remote_engine_id=meta.remote_engine_id,
                remote_host=meta.remote_host,
                remote_handshake_port=remote_handshake_port,
            )

        if self.kv_send_thread is not None:
            for req_id, delay_start_time in metadata.requests_to_send.items():
                if self.tp_rank in self._get_remote_tp_ranks_for_req(req_id):
                    self.kv_send_thread.add_delayed_request(
                        req_id, delay_start_time)

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
