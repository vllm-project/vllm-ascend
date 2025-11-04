# SPDX-License-Identifier: Apache-2.0
import contextlib
import copy
import hashlib
import math
import os
import queue
import struct
import threading
import time
from collections import defaultdict, deque
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
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1, KVConnectorMetadata, KVConnectorRole)
from vllm.distributed.parallel_state import (get_tensor_model_parallel_rank,
                                             get_tp_group, get_world_group)
from vllm.utils import get_ip, logger, make_zmq_path, make_zmq_socket
from vllm.v1.core.sched.output import SchedulerOutput

import vllm_ascend.envs as envs_ascend
from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.distributed.utils import (align_memory,
                                           get_transfer_timeout_value,
                                           kv_alltoall_and_rearrange)

if TYPE_CHECKING:
    from vllm.attention.backends.abstract import AttentionMetadata
    from vllm.forward_context import ForwardContext
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.request import Request

GET_META_MSG = b"get_meta_msg"
DONE_SENDING_MSG = b"done_sending_msg"


class MooncakeAgentMetadata(msgspec.Struct, omit_defaults=True, dict=True):
    te_rpc_port: int
    kv_caches_base_addr: list[int]


@dataclass
class ReqMeta:
    local_block_ids: list[int]
    token_ids: list[int]
    # Not None if layer-wise is disabled
    remote_block_ids: list[int]
    remote_engine_id: Optional[str]
    remote_host: Optional[str]
    remote_port: Optional[int]
    remote_te_rpc_port: Optional[int]
    remote_kv_caches_base_addr: Optional[list[int]]
    metaserver: Optional[str]


class KVCacheSendingLayerThread(threading.Thread):

    def __init__(self,
                 engine: TransferEngine,
                 total_layers: int,
                 ready_event: threading.Event,
                 tp_rank: int,
                 pd_head_ratio: int,
                 num_head_replica: int,
                 kv_cache_base_addr: list[int],
                 use_mla: bool,
                 block_len: list[int],
                 first_kv_cache: torch.Tensor,
                 callback_func: Callable[..., None] = lambda x: None):
        super().__init__(daemon=True, name="KVCacheSendingLayerThread")
        self.engine = engine
        self.tp_rank = tp_rank
        self.pd_head_ratio = pd_head_ratio
        self.num_head_replica = num_head_replica
        self.kv_caches_base_addr = kv_cache_base_addr
        self.total_layers = total_layers
        self.use_mla = use_mla
        self.block_len = block_len

        if self.pd_head_ratio > 1:
            # regesit kv buffer for tp inequal
            alignment = 2 * 1024 * 1024
            self.k_buffer = torch.zeros(first_kv_cache.numel() + alignment,
                                        dtype=first_kv_cache.dtype,
                                        device=first_kv_cache.device)
            self.k_buffer = align_memory(
                self.k_buffer, alignment)[:first_kv_cache.numel()].view(
                    -1, first_kv_cache.shape[-1])
            self.v_buffer = torch.zeros(first_kv_cache.numel() + alignment,
                                        dtype=first_kv_cache.dtype,
                                        device=first_kv_cache.device)
            self.v_buffer = align_memory(
                self.v_buffer, alignment)[:first_kv_cache.numel()].view(
                    -1, first_kv_cache.shape[-1])

            for tensor in (self.k_buffer, self.v_buffer):
                assert tensor.data_ptr(
                ) % alignment == 0, "The address of the registered kv cache should be aligned to 2M"
                ret_value = self.engine.register_memory(
                    tensor.data_ptr(), tensor.numel())
                logger.info(
                    f"Register memory for prefill when pd head ratio > 1 {tensor.data_ptr()} {tensor.numel()} {ret_value=}"
                )
                if ret_value != 0:
                    raise RuntimeError("Mooncake memory registration failed. ")

        self.send_queue = queue.Queue[Tuple[str, ReqMeta, int, torch.Tensor,
                                            torch.Tensor]]()

        self.ready_event = ready_event
        self.callback_func = callback_func

    def run(self):
        local_rank = get_world_group().local_rank
        device = torch.device(f"npu:{local_rank}")
        torch.npu.set_device(device)
        self.ready_event.set()
        while True:
            req_id, req_meta, layer_index, key, value = self.send_queue.get()
            self._handle_request(req_id, req_meta, layer_index, key, value)

    def _handle_request(self, req_id, req_meta, layer_index, key, value):
        try:
            logger.debug(
                f"Starting to transfer KV cache for request {req_id} {req_meta.remote_te_rpc_port=}."
            )
            self._transfer_kv_cache(req_id, req_meta, layer_index, key, value)
            logger.debug(
                f"Finished transferring KV cache for request {req_id} {req_meta.remote_te_rpc_port=}."
            )
        except Exception as e:
            logger.error("Failed to transfer KV cache for request "
                         f"{req_id}: {e}")

    def _transfer_kv_cache(self, req_id, req_meta, layer_index, key, value):
        # send kv layer to remote
        if len(req_meta.local_block_ids) == 0:
            return
        # not need to send kv cache
        if self.tp_rank % self.num_head_replica != 0:
            return

        remote_host = req_meta.remote_host
        remote_block_ids = req_meta.remote_block_ids
        remote_te_port = req_meta.remote_te_rpc_port
        remote_kv_base_addrs = req_meta.remote_kv_caches_base_addr
        local_kv_base_addr = self.kv_caches_base_addr
        local_block_ids = req_meta.local_block_ids

        if self.pd_head_ratio == 1:
            layer_local_kv_base_addr = [
                local_kv_base_addr[i]
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
                logger.error("Mooncake transfer failed for request %s", req_id)
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
                logger.error("Mooncake transfer failed for request %s", req_id)
                raise RuntimeError(f"Mooncake transfer failed, ret: {ret}")

        if layer_index == (self.total_layers - 1):
            self.callback_func(req_id, req_meta)


class KVCacheRecvingLayerThread(threading.Thread):

    def __init__(self, tp_rank: int, side_channel_port: int, tp_size: int,
                 pd_head_ratio: int, local_engine_id: str,
                 metadata: MooncakeAgentMetadata,
                 ready_event: threading.Event):
        super().__init__(daemon=True, name="KVCacheRecvingLayerThread")
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self.pd_head_ratio = pd_head_ratio
        self.local_engine_id = local_engine_id
        self.side_channel_host = get_ip()
        self.side_channel_port = side_channel_port
        self.lock = threading.Lock()
        self.done_requests = set[str]()
        self.task_tracker = dict[str, int]()
        self.ready_event = ready_event
        self.metadata = metadata

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

    def update_task(self, req_id):
        with self.lock:
            self.task_tracker[req_id] += 1
            if self.task_tracker[req_id] == self.pd_head_ratio:
                self.task_tracker.pop(req_id)
                self.done_requests.add(req_id)

    def run(self):
        """Run the thread to handle KV cache transfer requests."""
        handshake_port = self.side_channel_port + self.tp_rank
        path = make_zmq_path("tcp", self.side_channel_host, handshake_port)
        logger.info("Starting listening on path: %s", path)
        encoder = msgspec.msgpack.Encoder()
        encoded_data = encoder.encode(self.metadata)
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
                        logger.info("Got GET META INFO for request %s", msg[0])
                        sock.send_multipart((identity, b"", encoded_data))
                    elif msg[0] == DONE_SENDING_MSG:
                        logger.debug("Got DONE_RECVING_MSG for request %s",
                                     msg[1])
                        request_id = msg[1]
                        self.update_task(request_id)
                        sock.send_multipart((identity, b"", b"ACK"))
                    else:
                        logger.error(
                            "Connection listener got unexpected message %s",
                            msg)
                except Exception as e:
                    logger.error("Failed to decode message: %s", e)


class MooncakeLayerwiseConnectorMetadata(KVConnectorMetadata):

    def __init__(self):
        self.requests: dict[str, ReqMeta] = {}

    def add_new_req(self,
                    request_id: str,
                    local_block_ids: list[int],
                    kv_transfer_params: dict[str, Any],
                    token_ids: Optional[list[int]] = None):
        self.requests[request_id] = ReqMeta(
            token_ids=token_ids or [],
            local_block_ids=local_block_ids,
            remote_block_ids=kv_transfer_params.get("remote_block_ids", []),
            remote_engine_id=kv_transfer_params.get("remote_engine_id", None),
            remote_host=kv_transfer_params.get("remote_host", None),
            remote_port=kv_transfer_params.get("remote_port", None),
            remote_te_rpc_port=kv_transfer_params.get("remote_te_rpc_port",
                                                      None),
            remote_kv_caches_base_addr=kv_transfer_params.get(
                "remote_kv_caches_base_addr", None),
            metaserver=kv_transfer_params.get("metaserver", None),
        )


class MooncakeLayerwiseConnector(KVConnectorBase_V1):

    def __init__(self, vllm_config: VllmConfig, role: KVConnectorRole):
        assert vllm_config.kv_transfer_config is not None
        self.engine_id = vllm_config.kv_transfer_config.engine_id
        self._connector_metadata = MooncakeLayerwiseConnectorMetadata()

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
            vllm_config.parallel_config.data_parallel_rank *
            vllm_config.parallel_config.tensor_parallel_size)

        # Requests that need to start recv.
        # New requests are added by update_state_after_alloc in
        # the scheduler. Used to make metadata passed to Worker.
        self._reqs_need_recv: dict[str, tuple[Request, list[int],
                                              list[int]]] = {}
        self._reqs_need_send_layerwise: dict[str, tuple[
            int, list[int],
            Request]] = {}  # req_id, (len(prompt), local_block_ids, request)

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
            # Remote prefill: get all prompt blocks from remote.
            assert num_computed_tokens % self.block_size == 0
            # Note: We use the full token count as transmit data here.
            count = max(len(request.prompt_token_ids) - num_computed_tokens, 0)
            return count, count > 0

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
            local_block_ids = (blocks.get_unhashed_block_ids()
                               if num_external_tokens > 0 else [])
            # Get unhashed blocks to pull from remote.
            self._reqs_need_recv[request.request_id] = (
                request,
                [],  #request._all_token_ids,
                local_block_ids)

            params["do_remote_prefill"] = False

        # Layerwise prefiller add request need send
        if params is not None and params.get("do_remote_decode"):
            local_block_ids = (blocks.get_block_ids()[0])
            self._reqs_need_send_layerwise[request.request_id] = (len(
                request.all_token_ids), local_block_ids, request)

    def build_connector_meta(
        self,
        scheduler_output: SchedulerOutput,
    ) -> KVConnectorMetadata:
        meta = MooncakeLayerwiseConnectorMetadata()

        # Loop through scheduled reqs and convert to ReqMeta.
        for req_id, (req, token_ids,
                     block_ids) in self._reqs_need_recv.items():
            assert req.kv_transfer_params is not None
            # For the case where there are no remote blocks to pull
            # (block_ids is empty), we don't need to schedule
            # an async read on the worker side.
            meta.add_new_req(request_id=req_id,
                             local_block_ids=block_ids,
                             kv_transfer_params=req.kv_transfer_params,
                             token_ids=token_ids)

        # Clear the list once workers start the transfers
        self._reqs_need_recv.clear()

        cached_reqs = scheduler_output.scheduled_cached_reqs
        new_reqs = scheduler_output.scheduled_new_reqs
        for req_id, new_blocks in zip(cached_reqs.req_ids,
                                      cached_reqs.new_block_ids):
            if req_id in self._reqs_need_send_layerwise and new_blocks is not None:
                total_tokens, block_ids, req = self._reqs_need_send_layerwise[
                    req_id]
                block_ids.extend(new_blocks[0])

        computed_tokens = dict(
            list(zip(cached_reqs.req_ids, cached_reqs.num_computed_tokens)) +
            [(x.req_id, x.num_computed_tokens) for x in new_reqs])
        for req_id, scheduled_tokens in scheduler_output.num_scheduled_tokens.items(
        ):
            if req_id in self._reqs_need_send_layerwise:
                total_tokens, block_ids, req = self._reqs_need_send_layerwise[
                    req_id]
                current_tokens = computed_tokens.get(req_id,
                                                     0) + scheduled_tokens
                if current_tokens == total_tokens:
                    meta.add_new_req(request_id=req_id,
                                     local_block_ids=block_ids,
                                     kv_transfer_params=req.kv_transfer_params,
                                     token_ids=[])
                    self._reqs_need_send_layerwise.pop(req_id)
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
        # layer_wise push, not need delay_free_blocks
        return False, None


class MooncakeLayerwiseConnectorWorker:
    """Implementation of Worker side methods"""

    def __init__(self, vllm_config: VllmConfig, engine_id: str):
        self._get_prefill_decode_size(vllm_config)
        os.environ["ASCEND_TRANSFER_TIMEOUT"] = str(
            get_transfer_timeout_value())
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
        self.local_engine_id: str = " "
        self.engine_id = engine_id
        self.tp_rank = get_tensor_model_parallel_rank()
        self.tp_size = vllm_config.parallel_config.tensor_parallel_size
        self.tp_group = get_tp_group()
        self.dp_rank = vllm_config.parallel_config.data_parallel_rank
        self.dp_size = vllm_config.parallel_config.data_parallel_size_local
        self.kv_caches: dict[str, torch.Tensor] = {}
        self.side_channel_host = get_ip()
        self.max_device_id = self.tp_size * self.dp_size
        self.total_layers = vllm_config.model_config.get_num_layers(
            vllm_config.parallel_config)

        self.executor = ThreadPoolExecutor(32)
        self.metaserver_client = httpx.Client(
            limits=httpx.Limits(max_connections=100000),
            timeout=None) if self.tp_rank == 0 else None

        # Handshake base port
        self.side_channel_port = (
            vllm_config.kv_transfer_config.kv_port +
            vllm_config.parallel_config.data_parallel_rank *
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
                'use_ascend_direct', True):
            hostname = self.side_channel_host
        else:
            hostname = f"{self.side_channel_host}:0:npu_{self.device_id}"
        self._initialize(hostname=hostname, device_name=None)
        self.te_rpc_port = self.engine.get_rpc_port()

        # Background thread for sending or receiving KV caches.
        self.kv_recv_layer_thread: Optional[KVCacheRecvingLayerThread] = None
        self.kv_send_layer_thread: Optional[KVCacheSendingLayerThread] = None

        self.vllm_config = vllm_config
        self.block_size = vllm_config.cache_config.block_size
        self.kv_caches_base_addr: list[int] = []

        self.pd_tp_ratio = get_ascend_config().pd_tp_ratio
        self.pd_head_ratio = get_ascend_config().pd_head_ratio
        self.num_head_replica = get_ascend_config().num_head_replica

        self.first_kv_cache = None
        self.remote_poller = zmq.Poller()  # type: ignore
        self.decoder = msgspec.msgpack.Decoder(MooncakeAgentMetadata)
        self.encoder = msgspec.msgpack.Encoder()

        self.remote_kv_caches_base_addr: dict[str, dict[int, list[int]]] = \
            defaultdict(dict)
        self.remote_te_port: dict[str, dict[int, int]] = \
            defaultdict(dict)
        self.remote_sockets_lock = threading.Lock()
        self.remote_sockets: dict[  # type: ignore
            str, deque[zmq.Socket]] = defaultdict(  # type: ignore
                deque)
        self.remote_poller = zmq.Poller()  # type: ignore
        self.timeout = 1.0  # seconds

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
            te_rpc_port=self.te_rpc_port,
            kv_caches_base_addr=self.kv_caches_base_addr,
        )
        if self.vllm_config.kv_transfer_config.is_kv_producer:
            ready_event = threading.Event()
            self.kv_send_layer_thread = KVCacheSendingLayerThread(
                engine=self.engine,
                total_layers=self.total_layers,
                ready_event=ready_event,
                tp_rank=self.tp_rank,
                pd_head_ratio=self.pd_head_ratio,
                num_head_replica=self.num_head_replica,
                kv_cache_base_addr=self.kv_caches_base_addr,
                use_mla=self.use_mla,
                block_len=self.block_len,
                first_kv_cache=first_kv_cache,
                callback_func=self.send_done_send_signal)
            self.kv_send_layer_thread.start()
            ready_event.wait()

        if self.vllm_config.kv_transfer_config.is_kv_consumer:
            ready_event = threading.Event()
            self.kv_recv_layer_thread = KVCacheRecvingLayerThread(
                self.tp_rank, self.side_channel_port, self.tp_size,
                self.pd_head_ratio, self.engine_id, metadata, ready_event)
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
        success = False
        retry = 0
        while retry < 3 and success is False:
            retry += 1
            try:
                self.metaserver_client.post(url, json=message)
                success = True
            except Exception as e:
                logger.error(
                    f"Failed to connect to metaserver: {url}, retry {retry} time."
                )
                if retry == 3:
                    raise e

    def get_finished(self) -> tuple[set[str], set[str]]:
        done_recving = (
            self.kv_recv_layer_thread.
            get_and_clear_finished_requests(  # type: ignore[union-attr]
            ) if self.vllm_config.kv_transfer_config.is_kv_consumer else set())
        if len(done_recving) > 0:
            logger.info(
                "Number of completed KV cache recv requests: %d, receive "
                "requests: %d", 0, len(done_recving))
        return set(), done_recving

    def start_load_kv(self, metadata: MooncakeLayerwiseConnectorMetadata):
        """Start loading KV blocks from remote engine."""
        self.current_layer = 0
        if self.vllm_config.kv_transfer_config.is_kv_consumer:
            for req_id, meta in metadata.requests.items():
                if self.tp_rank % self.tp_size == 0:
                    logger.info(
                        f"Send request: {req_id} to proxy metaserver: {meta.metaserver}"
                    )
                    # All parameters here should appear in the returned dict of
                    # request_finished in the scheduler side except "request_id".
                    kv_transfer_params = dict(
                        token_ids=meta.token_ids,
                        request_id=req_id,
                        do_remote_prefill=False,
                        do_remote_decode=True,
                        remote_block_ids=meta.local_block_ids,
                        remote_engine_id=self.engine_id,
                        remote_host=self.side_channel_host,
                        remote_port=self.side_channel_port,
                    )
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
                assert self.kv_recv_layer_thread is not None
                with self.kv_recv_layer_thread.lock:
                    self.kv_recv_layer_thread.task_tracker[req_id] = 0

    def save_kv_layer(self, layer_name: str, kv_layer: Tuple[torch.Tensor,
                                                             torch.Tensor],
                      attn_metadata: "AttentionMetadata",
                      connector_metadata: MooncakeLayerwiseConnectorMetadata,
                      **kwargs) -> None:
        """MooncakeLayerwiseConnector does not save explicitly."""
        if self.vllm_config.kv_transfer_config.is_kv_producer and connector_metadata.requests.keys(
        ):
            # enable decode prefix cache
            for request in connector_metadata.requests.values():
                assert len(request.local_block_ids) >= len(
                    request.remote_block_ids
                ), "When prefix cache enabled, remote KVCacheBlocks num should not larger than local KVCacheBlocks num."
                request.local_block_ids = request.local_block_ids[
                    -len(request.remote_block_ids):]
            if self.pd_head_ratio != 1:

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
            for req_id, req_meta in connector_metadata.requests.items():
                logger.debug(
                    f"Add request {req_id} to kv send layer thread. {req_meta=}"
                )
                if self.pd_head_ratio != 1:
                    key_block_num = len(
                        req_meta.local_block_ids) * key_block_size
                    value_block_num = len(
                        req_meta.local_block_ids) * value_block_size
                    key = keys[key_start_id:key_start_id + key_block_num]
                    value = values[value_start_id:value_start_id +
                                   value_block_num]
                    key_start_id += key_block_num
                    value_start_id += value_block_num
                req_meta_update = self.update_decoder_info(req_id, req_meta)
                assert self.kv_send_layer_thread is not None
                self.kv_send_layer_thread.send_queue.put(
                    (req_id, req_meta_update, self.current_layer, key, value))
            self.current_layer += 1

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

    def update_decoder_info(self, req_id, req_meta):
        req_meta_update = copy.deepcopy(req_meta)
        if self.pd_tp_ratio > 1:
            req_meta_update.remote_port = req_meta_update.remote_port + self.tp_rank // self.pd_tp_ratio
        else:
            req_meta_update.remote_port = req_meta_update.remote_port + self.tp_rank
        if req_meta_update.remote_engine_id not in self.remote_kv_caches_base_addr or \
            req_meta_update.remote_port not in self.remote_kv_caches_base_addr[req_meta_update.remote_engine_id]:
            try:
                encoded_data = self.encoder.encode((GET_META_MSG, req_id))
                sock = self._get_remote_socket(req_meta_update.remote_host,
                                               req_meta_update.remote_port)
                ensure_zmq_send(sock, encoded_data)
                metadata_bytes = ensure_zmq_recv(sock, self.remote_poller)
                agent_meta = self.decoder.decode(metadata_bytes)
            except Exception as e:
                logger.error(
                    f"Query to port and kv base addr for request {req_id} from {req_meta_update.remote_host}:{req_meta_update.remote_port} fail with error: {e}"
                )
            assert req_meta_update.remote_engine_id != self.engine_id, (
                f"Conflict engine id {req_meta_update.remote_engine_id} with local engine id "
                f"{self.local_engine_id}.")
            self.remote_kv_caches_base_addr[req_meta_update.remote_engine_id][
                req_meta_update.remote_port] = agent_meta.kv_caches_base_addr
            self.remote_te_port[req_meta_update.remote_engine_id][
                req_meta_update.remote_port] = agent_meta.te_rpc_port
            logger.info(
                f"Query to port and kv base addr for request {req_id} from {req_meta_update.remote_host}:{req_meta_update.remote_port} success {agent_meta.kv_caches_base_addr=} {agent_meta.te_rpc_port=}"
            )
        req_meta_update.remote_te_rpc_port = self.remote_te_port[
            req_meta_update.remote_engine_id][req_meta_update.remote_port]
        req_meta_update.remote_kv_caches_base_addr = self.remote_kv_caches_base_addr[
            req_meta_update.remote_engine_id][req_meta_update.remote_port]
        return req_meta_update

    def send_done_send_signal(self, req_id, req_meta):
        logger.info("Sending done sending signal for request %s to %s:%d",
                    req_id, req_meta.remote_host, req_meta.remote_port)
        try:
            path = make_zmq_path("tcp", req_meta.remote_host,
                                 req_meta.remote_port)
            msg_encoder = msgspec.msgpack.Encoder()
            encoded_data = msg_encoder.encode((DONE_SENDING_MSG, req_id))
            with zmq_ctx(zmq.REQ, path) as sock:  # type: ignore
                ensure_zmq_send(sock, encoded_data)
                ack = sock.recv()
                if ack != b"ACK":
                    raise ValueError(f"Unexpected ACK response: {ack}")
        except Exception as e:
            logger.error(
                f"Sending done sending signal for request {req_id} to {req_meta.remote_host}:{req_meta.remote_port} fail with error: {e}"
            )

    def wait_for_layer_load(self, layer_name: str) -> None:
        pass


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
