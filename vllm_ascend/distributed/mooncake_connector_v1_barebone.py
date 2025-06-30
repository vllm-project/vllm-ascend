# SPDX-License-Identifier: Apache-2.0
import contextlib
import math
import threading
import time
import random
import socket
from enum import Enum
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional, List, Tuple
import os
import json
import msgspec
import torch
import zmq
import numpy as np
import numpy.typing as npt
import hashlib
import pickle
import struct

from concurrent.futures import Future

from vllm_ascend import envs
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1, KVConnectorMetadata, KVConnectorRole)
from vllm.distributed.parallel_state import (get_tensor_model_parallel_rank,
                                             get_tp_group, get_dp_group)
from vllm.utils import logger
from vllm.utils import make_zmq_path, make_zmq_socket, get_ip
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.request import RequestStatus
from mooncake.engine import TransferEngine

if TYPE_CHECKING:
    from vllm.attention.backends.abstract import AttentionMetadata
    from vllm.forward_context import ForwardContext
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.request import Request

GET_META_MSG = b"get_meta_msg"
DONE_RECVING_MSG = b"done_recving_msg"
UPDATE_ASYNC_REQ = False

global MOONCAKE_BASE_PORT
BASE_PORT = int(os.getenv("VLLM_BASE_PORT", "8790"))


class MooncakeAgentMetadata(msgspec.Struct, omit_defaults=True, dict=True):
    engine_id: str
    kv_caches_base_addr: list[int]
    num_blocks: int


class MooncakeRole(Enum):
    PROMPT = 1
    DECODER = 2
    MIX = 3


@dataclass
class ReqMeta:
    local_block_ids: list[int]
    remote_block_ids: list[int]
    remote_host: str
    remote_port: int
    remote_engine_id: str


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

        # 单边port用于TE上进行数据面的传输
        self.side_channel_port = (
            BASE_PORT + vllm_config.parallel_config.data_parallel_rank_local *
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
            assert num_computed_tokens % self.block_size == 0
            count = max(len(request.prompt_token_ids) - num_computed_tokens, 0)
            if count > 0:
                return count, True

        # No remote prefill for this request.
        return 0, False

    def update_state_after_alloc(self, request: "Request",
                                 blocks: "KVCacheBlocks",
                                 num_external_tokens: int):

        params = request.kv_transfer_params
        logger.info(
            "MooncakeConnector update_state_after_alloc: "
            "num_external_tokens=%s, kv_transfer_params=%s",
            num_external_tokens, params)

        if params is not None and params.get("do_remote_prefill"):
            if params.get("remote_block_ids"):
                if all(p in params for p in ("remote_engine_id", "remote_host",
                                             "remote_port")):
                    # Get unhashed blocks to pull from remote.
                    self._reqs_need_recv[request.request_id] = (
                        request, blocks.get_unhashed_block_ids())
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
        # If prompt < block_size, no xfer so free blocks immediately.
        delay_free_blocks = len(computed_block_ids) > 0

        return delay_free_blocks, dict(
            do_remote_prefill=True,
            do_remote_decode=False,
            remote_block_ids=computed_block_ids,
            remote_engine_id=self.engine_id,
            remote_host=self.side_channel_host,
            remote_port=self.side_channel_port,
        )


class MooncakeConnectorWorker:
    """Implementation of Worker side methods"""

    def __init__(self, vllm_config: VllmConfig, engine_id: str):
        self._get_prefill_decode_size(vllm_config)
        if self._prefill_tp_size < self._decode_tp_size:
            raise ValueError(
                f"prefill_tp_size: {self._prefill_tp_size} must be "
                f"greater than or equal to the decode_tp_size: {self._decode_tp_size}"
            )

        if TransferEngine is None:
            logger.error("mooncake is not available")
            raise RuntimeError("mooncake is not available")
        logger.info("Initializing Mooncake work %s", engine_id)
        self.engine = TransferEngine()

        # Metadata.
        self.engine_id = engine_id
        self.tp_rank = get_tensor_model_parallel_rank()
        self.tp_size = vllm_config.parallel_config.tensor_parallel_size
        self.tp_group = get_tp_group()
        self.dp_rank = vllm_config.parallel_config.data_parallel_rank_local
        self.dp_size = vllm_config.parallel_config.data_parallel_size_local  # here we use dp local size
        self.kv_caches: dict[str, torch.Tensor] = {}
        self.side_channel_host = get_local_ip_by_remote()
        self.max_device_id = self.tp_size * self.dp_size

        self.kv_role = vllm_config.kv_transfer_config.kv_role

        self.dp_group = get_dp_group()
        device_ids = os.getenv("ASCEND_RT_VISIBLE_DEVICES", None)

        logger.info(f"os getenv ASCEND_RT_VISIBLE_DEVICES: {device_ids}")
        # device_ids = device_ids.split(',')
        if device_ids is None:
            device_ids_list = list(
                range(self.dp_rank * self.tp_size,
                      (self.dp_rank + 1) * self.tp_size))
        else:
            device_ids_list = list(map(int, device_ids.split(',')))
        assert len(device_ids_list) > self.tp_rank
        # self.device_id = device_ids[self.tp_rank + self.dp_group_num*self.dp_size]
        # 0: [0,0,2,4] 1:[]

        # self.device_id = device_ids[self.dp_rank * self.tp_size + self.tp_rank]
        # self.device_id = (self.dp_rank * self.tp_size + self.tp_rank)
        # TODO just verify TP=1
        self.device_id = device_ids_list[self.tp_rank]

        # 单边port用于TE上进行数据面的传输
        # self.side_channel_port = (
        #     BASE_PORT +
        #     self.dp_rank * self.tp_size + self.tp_rank + self.max_device_id)

        self.side_channel_port = (
            BASE_PORT + vllm_config.parallel_config.data_parallel_rank_local *
            vllm_config.parallel_config.tensor_parallel_size)

        # get tp device id
        # self.device_id = (self.dp_rank * self.tp_size + self.tp_rank)
        # note: https://github.com/vllm-project/vllm-ascend/pull/940 introducing some changes

        logger.info(
            f"dp_rank {self.dp_rank} \n"
            f"tp_rank {self.tp_rank}\n "
            f"dp_size {self.dp_size}\n"
            f"max_device_id {self.max_device_id}\n"
            f"side_channel_port {self.side_channel_port+ self.tp_rank + self.max_device_id}\n"
            f"device_id {self.device_id}\n")

        #----------------------------------

        # device_ids = os.getenv("ASCEND_RT_VISIBLE_DEVICES", None)
        # device_ids = device_ids.split(',')
        # self.dp_rank = get_dp_group().rank_in_group
        # device_ids = device_ids[self.dp_rank*self.tp_size : (self.dp_rank+1)*self.tp_size]
        # self.device_id = device_ids[self.tp_rank]
        #----------------------------------

        self.ib_device = None
        self._initialize(
            hostname=self.side_channel_host + ':' +
            str(self.side_channel_port + self.tp_rank + self.max_device_id) +
            ':' + 'npu_' + str(self.device_id),
            device_name=self.ib_device,
        )

        # Map of engine_id -> agent_name.
        self._remote_engine: List = []

        # Map of engine_id -> kv_caches_base_addr
        self.kv_caches_base_addr: dict[str, dict[int, list[int]]] = {}

        self.kv_caches_base_addr_port: dict[int, list[int]] = {}

        self.num_layers = 0

        # # Complete transfer tracker. Used by the rank 0 to track finished
        # # transactions on ranks 1 to N-1.
        # # [req_id -> count]
        self._done_recving_count: defaultdict[str,
                                              int] = defaultdict(lambda: 0)
        self._done_sending_count: defaultdict[str,
                                              int] = defaultdict(lambda: 0)

        # Background thread for establishing new connections.
        self._message_listener_t: Optional[threading.Thread] = None

        self.vllm_config = vllm_config
        self.block_size = vllm_config.cache_config.block_size

        # create async thread pool
        self.futures: dict[str, list[Future]] = {}
        self.req_record: dict[str, tuple[str, int]] = {}
        self.sync_trans_req: list[str] = []

    def _get_prefill_decode_size(self, vllm_config: VllmConfig):
        # get prefill tp size from extra config
        prefill_parallel_config: dict[
            str, Any] = vllm_config.kv_transfer_config.get_from_extra_config(
                "prefill", {})

        assert "tp_size" in prefill_parallel_config.keys()
        self._prefill_tp_size = prefill_parallel_config["tp_size"]

        assert "dp_size" in prefill_parallel_config.keys()
        self._prefill_dp_size = prefill_parallel_config["dp_size"]

        # get decode tp size from extra config
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
        # get mooncake config json content
        assert envs.MOONCAKE_CONFIG_PATH, "Please set path of rank_table to env variable MOONCAKE_CONFIG_PATH"
        rank_table_path = envs.MOONCAKE_CONFIG_PATH
        with open(rank_table_path, "r", encoding="utf-8") as f:
            mooncake_config = json.load(f)

        device_name if device_name is not None else ""
        logger.info(
            "******self.engine.initialize now, para1: %s, para2: P2PHANDSHAKE, para3: %s, para4: %s",
            hostname, mooncake_config["protocol"], device_name)
        ret_value = self.engine.initialize(hostname, "P2PHANDSHAKE",
                                           mooncake_config["protocol"],
                                           device_name)

        if ret_value != 0:
            logger.error("Mooncake Transfer Engine initialization failed.")
            raise RuntimeError(
                "Mooncake Transfer Engine initialization failed.")

    def _message_listener(self, metadata: MooncakeAgentMetadata,
                          ready_event: threading.Event, tp_rank: int):
        """Background thread for getting new Mooncake handshakes."""

        encoder = msgspec.msgpack.Encoder()
        encoded_data = encoder.encode(metadata)
        size_in_bytes = len(encoded_data)
        logger.debug("Size of encoded MooncakeAgentMetadata: %s bytes",
                     str(size_in_bytes))

        # Listen for new requests for metadata.
        # NOTE(rob): we need each rank to have a unique port. This
        # hack to keeps us moving. We will switch when moving to etcd
        # or where we have a single ZMQ socket in the scheduler.
        #handshake_port = self.side_channel_port - self.max_device_id
        handshake_port = self.side_channel_port + tp_rank

        path = make_zmq_path("tcp", self.side_channel_host, handshake_port)

        logger.info("Starting listening on path: %s", path)
        with zmq_ctx(zmq.ROUTER, path) as sock:
            ready_event.set()
            while True:
                identity, _, msg = sock.recv_multipart()
                msg = pickle.loads(msg)
                if msg[0] == b"get_meta_msg":
                    sock.send_multipart((identity, b"", encoded_data))
                elif msg[0] == b"done_recving_msg":
                    logger.debug(
                        "Got notify from remote engine that load kv is done")
                    self._done_sending_count[msg[1]] += 1
                else:
                    logger.error(
                        "Connection listener got unexpected message %s", msg)
                # ready_event.set()

    def _message_req(self, host: str, port: int, msg: tuple[bytes, str]):
        """send a normal message with a remote instance."""

        start_time = time.perf_counter()
        # NOTE(rob): we need each rank to have a unique port. This is
        # a hack to keep us moving. We will switch when moving to etcd
        # or where we have a single ZMQ socket in the scheduler.
        print("self.tp_rank:", self.tp_rank)
        print("_message_req port:", port)
        path = make_zmq_path("tcp", host, port)
        logger.debug("Querying metadata on path: %s", path)
        with zmq_ctx(zmq.REQ, path) as sock:
            # Send msg to remote. It will recv a msg in shakehand case and other would not
            if msg[0] == GET_META_MSG:
                logger.debug("Sending query for metadata")
                data_bytes = pickle.dumps(msg)
                sock.send(data_bytes)
                metadata_bytes = sock.recv()
                decoder = msgspec.msgpack.Decoder(MooncakeAgentMetadata)
                metadata = decoder.decode(metadata_bytes)
                # Register Remote agent.
                self.add_remote_agent(metadata, port)
            elif msg[0] == DONE_RECVING_MSG:
                logger.debug("Sending notify to prefill that load is done")
                data_bytes = pickle.dumps(msg)
                sock.send(data_bytes)
            else:
                logger.warning("Connection listener got unexpected message %s",
                               msg)
                raise RuntimeError(f"Unexpected message: {msg}")

            end_time = time.perf_counter()
            logger.debug("send %s message took: %s", msg[0],
                         end_time - start_time)

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        """Register the KV Cache data."""

        _, first_kv_cache_tuple = next(iter(kv_caches.items()))
        first_kv_cache = first_kv_cache_tuple[0]

        # print("first_kv_cache.shape:",first_kv_cache.shape)

        # TODO(tms): Find a more robust way to detect and handle MLA
        self.use_mla = first_kv_cache_tuple[0].size(
            -1) != first_kv_cache_tuple[1].size(-1)
        if self.use_mla:
            # MLA case.
            print("**************we are in mla***************")
            self.num_blocks = first_kv_cache.shape[1]
            block_rank = 3  # [block_size, latent_dim]
            block_shape_norm = first_kv_cache_tuple[0].shape[-block_rank:]
            block_shape_pe = first_kv_cache_tuple[1].shape[-block_rank:]
            self.block_len = [
                first_kv_cache[0].element_size() * math.prod(block_shape_norm),
                first_kv_cache[1].element_size() * math.prod(block_shape_pe)
            ]
        else:
            # [2 (k and v), num_blocks, ...]
            self.num_blocks = first_kv_cache.shape[1]
            kv_elem_size = first_kv_cache.element_size()
            block_rank = 3  # [block_size, kv_heads, head_dim]
            block_shape = first_kv_cache.shape[-block_rank:]
            self.block_len = kv_elem_size * math.prod(block_shape)

        logger.info("Registering KV_Caches. use_mla: %s, shape %s",
                    self.use_mla, first_kv_cache.shape)
        if self.use_mla:
            logger.info(
                "num_blocks: %s, block_shape_norm: %s, block_shape_pe: %s",
                self.num_blocks, block_shape_norm, block_shape_pe)
        else:
            logger.info("num_blocks: %s, block_shape: %s", self.num_blocks,
                        block_shape)
        logger.info("Per layer kv cache size: %s", first_kv_cache.shape)
        self.kv_caches = kv_caches
        kv_caches_base_addr = []

        for cache_or_caches in kv_caches.values():
            # Normalize to always be a list of caches
            if self.use_mla:
                for i, cache in enumerate(cache_or_caches, 0):
                    print("cache.shape:", cache.shape)
                    kv_elem_size = cache.element_size()
                    base_addr = cache.data_ptr()
                    region_len = self.num_blocks * self.block_len[i % 2]
                    kv_caches_base_addr.append(base_addr)
                    self._register(base_addr, region_len)
            else:
                cache_list = [cache_or_caches
                              ] if self.use_mla else cache_or_caches
                for cache in cache_list:
                    base_addr = cache.data_ptr()
                    region_len = self.num_blocks * self.block_len
                    kv_caches_base_addr.append(base_addr)
                    self._register(base_addr, region_len)
        print("self.engine_id %s", self.engine_id)
        print("self.side_channel_port + self.tp_rank %s",
              self.side_channel_port + self.tp_rank)
        print("self.kv_caches_base_addr %s", self.kv_caches_base_addr)
        self.kv_caches_base_addr_port[self.side_channel_port +
                                      self.tp_rank] = kv_caches_base_addr
        self.kv_caches_base_addr[
            self.engine_id] = self.kv_caches_base_addr_port
        self.num_layers = len(self.kv_caches.keys())

        # After KV Caches registered, listen for new connections.
        metadata = MooncakeAgentMetadata(
            engine_id=self.engine_id,
            kv_caches_base_addr=self.kv_caches_base_addr[self.engine_id][
                self.side_channel_port + self.tp_rank],
            num_blocks=self.num_blocks,
        )
        ready_event = threading.Event()
        self._message_listener_t = threading.Thread(
            target=self._message_listener,
            args=(metadata, ready_event, self.tp_rank),
            daemon=True,
            name="message_listener")
        self._message_listener_t.start()
        ready_event.wait()

    def _register(self, ptr, length):
        if self.use_mla:
            logger.info(
                "*--- _register: length: %s, self.num_blocks: %s, self.block_lens: [%s,%s], ptr: %x",
                length, self.num_blocks, self.block_len[0], self.block_len[1],
                ptr)
        else:
            logger.info(
                "*--- _register: length: %s, self.num_blocks: %s, self.block_len: %s, ptr: %x",
                length, self.num_blocks, self.block_len, ptr)
        ret_value = self.engine.register_memory(ptr, length)
        if ret_value != 0:
            logger.error("Mooncake memory registration failed.")
            raise RuntimeError("Mooncake memory registration failed.")

    def add_remote_agent(self, agent_meta: MooncakeAgentMetadata, port: int):
        engine_id = agent_meta.engine_id
        assert engine_id != self.engine_id, "Conflict engine id found!"
        if engine_id in self._remote_engine:
            return

        self._remote_engine.append(engine_id)
        kv_caches_base_port = {}
        kv_caches_base_port[port] = agent_meta.kv_caches_base_addr
        self.kv_caches_base_addr[engine_id] = kv_caches_base_port

    def get_finished(self) -> tuple[set[str], set[str]]:

        # get the async transfer completed tasks (done recving)
        if UPDATE_ASYNC_REQ:
            for req in self.futures.keys():
                success_count = 0
                try:
                    # 获取每个任务的返回值
                    future = self.futures[req]
                    for trans_task in future:
                        ret = trans_task.result()
                        # 假设status为True时表示成功
                        if ret >= 0:
                            success_count += 1
                except Exception as e:
                    # 处理任务中的异常情况
                    logger.error(
                        "Mooncake Transfer Engine Return Error, error is %s",
                        e)
                    raise RuntimeError(
                        "Mooncake Transfer Engine Return Error.")

                if success_count == len(self.futures[req]):
                    self._done_recving_count[req] += 1
                    msg = (DONE_RECVING_MSG, req)
                    remote_host = self.req_record[req][0]
                    remote_handshake_port = self.req_record[req][1]
                    self._message_req(remote_host, remote_handshake_port, msg)
        else:
            for req in self.sync_trans_req:
                logger.info("******Get Finished Sync**********")
                self._done_recving_count[req] += 1
                msg = (DONE_RECVING_MSG, req)
                remote_host = self.req_record[req][0]
                remote_handshake_port = self.req_record[req][1]
                self._message_req(remote_host, remote_handshake_port, msg)
                self.sync_trans_req.remove(req)

        done_sending = set(self._done_sending_count.keys())
        done_recving = set(self._done_recving_count.keys())

        if self.tp_size == 1:
            return done_sending, done_recving

        # Rank 0: get finished from all other ranks.
        if self.tp_rank == 0:
            # Keep track of how many other ranks have finished.
            other_ranks_finished_ids: list[str] = []
            for i in range(1, self.tp_size):
                other_ranks_finished_ids.extend(
                    self.tp_group.recv_object(src=i))
            for req_id in other_ranks_finished_ids:
                if (req_id in self._done_recving_count):
                    self._done_recving_count[req_id] += 1
                else:
                    self._done_sending_count[req_id] += 1

            # Return ids that finished on all ranks to the scheduler.
            all_done_recving: set[str] = set()
            for req_id in list(self._done_recving_count.keys()):
                if self._done_recving_count[req_id] == self.tp_size:
                    # if self._done_recving_count[req_id] == self.tp_size or self._done_recving_count[req_id] == self.decode_tp_size:
                    del self._done_recving_count[req_id]
                    all_done_recving.add(req_id)

            all_done_sending: set[str] = set()
            for req_id in list(self._done_sending_count.keys()):
                if self._done_sending_count[req_id] == self._decode_tp_size:
                    del self._done_sending_count[req_id]
                    all_done_sending.add(req_id)
            if self.kv_role == 'kv_producer':
                return all_done_sending, set()
            else:
                return set(), all_done_recving

        # Ranks 1 to N-1: send finished ids to Rank 0.
        else:
            finished_req_ids = list(done_recving.union(done_sending))
            self.tp_group.send_object(finished_req_ids, dst=0)
            # Unused as only Rank 0 results are sent to scheduler.
            return done_sending, done_recving

    def start_load_kv(self, metadata: MooncakeConnectorMetadata):
        """
        Start loading KV blocks from remote engine.
        Args:
            metadata: dict of request_id -> MooncakeConnectorMetadata
        """
        for req_id, meta in metadata.requests.items():
            logger.info(
                "start_load_kv for request %s from remote engine %s. "
                "Num local_block_ids: %s. Num remote_block_ids: %s. ", req_id,
                meta.remote_engine_id, len(meta.local_block_ids),
                len(meta.remote_block_ids))

            self._read_blocks(
                request_id=req_id,
                dst_engine_id=meta.remote_engine_id,
                local_block_ids=meta.local_block_ids,
                remote_block_ids=meta.remote_block_ids,
                remote_host=meta.remote_host,
                remote_port=meta.remote_port,
            )

    def _read_blocks(
        self,
        local_block_ids: list[int],
        remote_block_ids: list[int],
        remote_host: str,
        remote_port: int,
        dst_engine_id: str,
        request_id: str,
    ):
        # get target tp rank
        remote_handshake_port = remote_port + self._get_remote_tp_rank(
            request_id)
        self.req_record[request_id] = (remote_host, remote_handshake_port)

        # NOTE(rob): this takes ~2s. We need to get this off the hotpath.
        if dst_engine_id not in self._remote_engine:
            #remote_handshake_port = remote_port - self.max_device_id
            msg = (b"get_meta_msg", "")
            self._message_req(remote_host, remote_handshake_port, msg)

        # NOTE(rob): having the staging blocks be on the READER side is
        # not going to work well (since we will have to call rearrange tensors).
        # after we detect the txn is complete (which means we cannot make the
        # read trxn async easily). If we want to make "READ" happen cleanly,
        # then we will need to have the staging blocks on the remote side.

        # NOTE(rob): according to nvidia the staging blocks are used to
        # saturate IB with heterogeneous TP sizes. We should remove the staging
        # blocks until we are ready.

        # Full prefix cache hit: do not need to read remote blocks,
        # just notify P worker that we have the blocks we need.
        num_local_blocks = len(local_block_ids)
        if num_local_blocks == 0:
            self._done_recving_count[request_id] += 1
            msg = (DONE_RECVING_MSG, request_id)
            self._message_req(remote_host, remote_handshake_port, msg)
            return

        # Partial prefix cache hit: just read uncomputed blocks.
        num_remote_blocks = len(remote_block_ids)
        assert num_local_blocks <= num_remote_blocks
        if num_local_blocks < num_remote_blocks:
            remote_block_ids = remote_block_ids[-num_local_blocks:]
        if UPDATE_ASYNC_REQ:
            from concurrent.futures import ThreadPoolExecutor
            grouped_remote_block_ids, grouped_local_block_ids = \
                group_concurrent_contiguous(remote_block_ids, local_block_ids)

            with ThreadPoolExecutor() as executor:

                # mooncake_session_id = f"{remote_host}:{remote_port}"  # our orinial
                # TODO only support kv from P to D
                remote_transfer_port = remote_handshake_port + self._prefill_dp_size * self._prefill_tp_size
                mooncake_session_id = f"{remote_host}:{remote_transfer_port}"

                for k, (src_layer_base_addr, dst_layer_base_addr) in enumerate(
                        zip(
                            self.kv_caches_base_addr[self.engine_id][
                                self.side_channel_port + self.tp_rank],
                            self.kv_caches_base_addr[dst_engine_id]
                            [remote_handshake_port])):
                    if self.use_mla:
                        block_len = self.block_len[k % 2]
                    else:
                        block_len = self.block_len
                    for i in range(len(grouped_remote_block_ids)):

                        src = src_layer_base_addr + grouped_local_block_ids[i][
                            0] * block_len
                        dst = dst_layer_base_addr + grouped_remote_block_ids[
                            i][0] * block_len
                        length = len(grouped_local_block_ids[i]) * block_len

                        logger.info(
                            "****--- executor.submit, mooncake_session_id: %s, src:%s, dst: %s, length:%s",
                            mooncake_session_id, src, dst, length)
                        future = executor.submit(self._transfer_sync,
                                                 mooncake_session_id, src, dst,
                                                 length)
                        print("****--- future: %s", future)
                        if request_id in self.futures:
                            self.futures[request_id].append(future)
                        else:
                            self.futures[request_id] = [future]
        else:
            self.sync_trans_req.append(request_id)
            grouped_remote_block_ids, grouped_local_block_ids = \
                group_concurrent_contiguous(remote_block_ids, local_block_ids)

            remote_transfer_port = remote_handshake_port + self._prefill_dp_size * self._prefill_tp_size
            mooncake_session_id = f"{remote_host}:{remote_transfer_port}"

            for k, (src_layer_base_addr, dst_layer_base_addr) in enumerate(
                    zip(
                        self.kv_caches_base_addr[self.engine_id][
                            self.side_channel_port + self.tp_rank],
                        self.kv_caches_base_addr[dst_engine_id]
                        [remote_handshake_port])):
                if self.use_mla:
                    block_len = self.block_len[k % 2]
                else:
                    block_len = self.block_len
                for i in range(len(grouped_remote_block_ids)):

                    src = src_layer_base_addr + grouped_local_block_ids[i][
                        0] * block_len
                    dst = dst_layer_base_addr + grouped_remote_block_ids[i][
                        0] * block_len
                    length = len(grouped_local_block_ids[i]) * block_len

                    logger.info(
                        "****--- sync transfer, mooncake_session_id: %s, src:%s, dst: %s, length:%s",
                        mooncake_session_id, src, dst, length)
                    ret = self._transfer_sync(mooncake_session_id, src, dst,
                                              length)
                    if ret > 0:
                        logger.info(
                            "Mooncake Transfer Engine Return Success, ret is %s",
                            ret)

    def _transfer_sync(self, session_id: str, buffer: int,
                       peer_buffer_address: int, length: int) -> int:
        """Synchronously transfer data to the specified address."""
        ret = self.engine.transfer_sync_read(session_id, buffer,
                                             peer_buffer_address, length)
        if ret < 0:
            logger.error("Mooncake Transfer Engine Return Error, ret is %s",
                         ret)
            raise RuntimeError("Mooncake Transfer Engine Return Error.")
        return ret

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
def zmq_ctx(socket_type: Any, addr: str) -> Iterator[zmq.Socket]:
    """Context manager for a ZMQ socket"""

    if socket_type not in (zmq.ROUTER, zmq.REQ):
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
