# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

import json
import math
import os
import pickle
import threading
import time
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any, Optional, Union

import zmq
from omni.accelerators.pd.utils import get_config_from_dict_or_env
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1, KVConnectorMetadata, KVConnectorRole)
from vllm.distributed.parallel_state import get_tensor_model_parallel_rank
from vllm.envs import VLLM_RPC_TIMEOUT
from vllm.logger import init_logger
from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.core.sched.output import SchedulerOutput

if TYPE_CHECKING:
    from vllm.config import VllmConfig, KVTransferConfig
    from vllm.attention.backends.abstract import AttentionMetadata
    from vllm.forward_context import ForwardContext
    from vllm.v1.request import Request

import mmap as pymmap
import queue
import random
import signal
import socket
import struct
import subprocess
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from ctypes import (CDLL, POINTER, Structure, c_bool, c_char_p, c_size_t,
                    c_uint32, c_uint64, c_void_p, create_string_buffer)
from dataclasses import dataclass

import torch
from vllm.distributed.parallel_state import (
    get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size,
    get_tp_group)
from vllm.utils import get_open_port, round_down
from vllm.v1.request import Request, RequestStatus

GET_META_MSG = b"get_meta_msg"

logger = init_logger(__name__)

thread_dump_path = os.environ.get("VLLM_THREAD_DUMP_PATH", "/tmp/vllm_thread_info")
BLOCK_RELEASE_DELAY = 30  # seconds, use to free blocks when the request is finished for a long time 

from omni.accelerators.pd.llmdatadist_manager import (LLMDataDistConfig,
                                                      LLMDataDistManager)

P_NODE_BIN = os.environ.get("P_NODE_BIN", "/data/yyx/omni_infer_cache/omniinfer/omni/accelerators/pd/H2H_pull_kv/p_node_server")
LIB_PATH = os.environ.get("KV_BRIDGE_SO", "/data/yyx/omni_infer_cache/omniinfer/omni/accelerators/pd/H2H_pull_kv/libkv_py_bridge.so")
VERBOSE = int(os.environ.get("VERBOSE", "1"))
READINESS_TIMEOUT = float(os.environ.get("READINESS_TIMEOUT", "20"))
READINESS_INTERVAL = float(os.environ.get("READINESS_INTERVAL", "0.2"))

BASE_PORT = 15000

NUM_P_NODES = 1
P_IP_LIST = ["7.150.8.246"]
NODE_SPECS = []
for node_id in range(NUM_P_NODES):
    NODE_SPECS.append(f"{P_IP_LIST[node_id]}:{BASE_PORT}:{node_id}")

class KVBlockC(Structure):
    _fields_ = [
        ("block_id", c_uint64),
        ("memory_ptr", c_void_p),
        ("memory_size", c_size_t),
    ]

@dataclass
class ReqMeta:
    local_block_ids: list[int]
    remote_block_ids: list[int]
    remote_host: str
    remote_cluster_id: str
    spec_token_ids: Optional[list[int]]
    remote_dp_rank: Optional[int]
    remote_request_id: Optional[str]

@dataclass
class ReqMetaPrefill:
    finish_time: float

class DatadistConnectorMetadata(KVConnectorMetadata):
    """Metadata for datadist connector."""

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
            remote_host=kv_transfer_params["remote_host_ip"],
            remote_cluster_id=kv_transfer_params["remote_cluster_id"],
            spec_token_ids=kv_transfer_params["spec_token_ids"],
            remote_dp_rank=kv_transfer_params.get("remote_dp_rank", 0),
            remote_request_id=kv_transfer_params.get("remote_request_id", None),
        )

class DatadistConnectorMetadataPrefill(KVConnectorMetadata):
    """Metadata for datadist connector."""

    def __init__(self):
        self.requests: dict[str, ReqMeta] = {}

    def add_new_req(
        self,
        request_id: str,
        finish_time: float,
    ):
        self.requests[request_id] = ReqMeta(
            finish_time=finish_time
        )


class LLMDataDistConnector(KVConnectorBase_V1):
    def __init__(self, vllm_config: VllmConfig, role: KVConnectorRole):
        if vllm_config.kv_transfer_config is None:
            raise RuntimeError("vllm_config.kv_transfer_config cannot be None")

        if vllm_config.model_config.is_deepseek_mla:
            vllm_config.kv_transfer_config.kv_parallel_size = 1
            logger.info("Set kv_parallel_size to 1 when use deepseek mla model.")

        self.datadist_config = LLMDataDistConfig(vllm_config, ignore_load_rank=True)
        self.cluster_id_start = self.datadist_config.cluster_id_start
        self.host_ip = self.datadist_config.local_group.host_ip
        # Introduce the environment variable VLLM_LLMDATADIST_ZMQ_PORT to resolve ZMQ connection conflicts during
        # multi-P deployments on the same machine.
        # This variable should not be set separately unless specifically required for this scenario.
        self.host_port = get_config_from_dict_or_env(vllm_config.kv_transfer_config, "kv_port",
                                                     "VLLM_LLMDATADIST_ZMQ_PORT", "5568", int)
        dp_rank = vllm_config.parallel_config.data_parallel_rank
        self.host_port += dp_rank
        self.is_prefill = vllm_config.kv_transfer_config.kv_role == "kv_producer"

        if role == KVConnectorRole.SCHEDULER:
            if self.is_prefill:
                self.connector_scheduler = PrefillConnectorScheduler(vllm_config, self.cluster_id_start, self.host_ip,
                                                                     str(self.host_port))
            else:
                self.connector_scheduler = DecodeConnectorScheduler(vllm_config)
            self.connector_worker = None
        elif role == KVConnectorRole.WORKER:
            if self.is_prefill:
                self.connector_worker = PrefillConnectorWorker(vllm_config, str(self.host_ip), str(self.host_port))
            else:
                self.connector_worker = DecodeConnectorWorker(vllm_config, str(self.host_ip), self.cluster_id_start)
            self.connector_scheduler = None

    ############################################################
    # Scheduler Side Methods
    ############################################################

    def get_num_new_matched_tokens(
            self, request: "Request",
            num_computed_tokens: int) -> tuple[int, bool]:
        if self.connector_scheduler is None:
            raise RuntimeError("self.connector_scheduler cannot be None")
        return self.connector_scheduler.get_num_new_matched_tokens(request, num_computed_tokens)

    def update_state_after_alloc(self, request: "Request",
                                 blocks: "KVCacheBlocks",
                                 num_external_tokens: int):
        if self.connector_scheduler is None:
            raise RuntimeError("self.connector_scheduler cannot be None")
        return self.connector_scheduler.update_state_after_alloc(request, blocks, num_external_tokens)

    def build_connector_meta(
            self,
            scheduler_output: SchedulerOutput,
    ) -> KVConnectorMetadata:
        if self.connector_scheduler is None:
            raise RuntimeError("self.connector_scheduler cannot be None")
        return self.connector_scheduler.build_connector_metadata(scheduler_output)

    def request_finished(
            self,
            request: "Request",
            block_ids: list[int],
            spec_token_ids: Optional[list[int]] = []
    ) -> tuple[bool, Optional[dict[str, Any]]]:
        if self.connector_scheduler is None:
            raise RuntimeError("self.connector_scheduler cannot be None")
        return self.connector_scheduler.request_finished(request, block_ids, spec_token_ids)

    ############################################################
    # Worker Side Methods
    ############################################################
    def register_kv_caches(self, kv_pool_mmap_path, data_type, block_len_dtype, start_offset=0):
        data_type = "bf16"
        # start_offset = 0
        if self.connector_worker is None:
            raise RuntimeError("self.connector_worker cannot be None")
        return self.connector_worker.register_kv_caches(kv_pool_mmap_path, data_type, block_len_dtype, start_offset)

    def get_finished(self,
                     finished_req_ids: set[str]) -> tuple[set[str], set[str]]:
        """Get the finished recving and sending requests."""
        if self.connector_worker is None:
            raise RuntimeError("self.connector_worker cannot be None")
        return self.connector_worker.get_finished(self._connector_metadata)

    def start_load_kv(self, forward_context: "ForwardContext",
                      **kwargs) -> None:
        if self.connector_worker is None:
            raise RuntimeError("self.connector_worker cannot be None")
        if not isinstance(self._connector_metadata, Union[DatadistConnectorMetadata, DatadistConnectorMetadataPrefill]):
            raise RuntimeError("self._connector_metadata must be an instance of DatadistConnectorMetadata or DatadistConnectorMetadataPrefill")
        self.connector_worker.start_load_kv(self._connector_metadata)

    def wait_for_layer_load(self, layer_name: str) -> None:
        """Connector does not do layerwise saving."""
        pass

    def save_kv_layer(self, layer_name: str, kv_layer: torch.Tensor,
                      attn_metadata: "AttentionMetadata", **kwargs) -> None:
        """Connector does not save explicitly."""
        pass

    def wait_for_save(self):
        """Connector does not save explicitly."""
        pass

class PrefillConnectorScheduler:
    """Implementation of Scheduler side methods"""

    def __init__(self, vllm_config, cluster_id_start: str, host_ip: str, host_port: str):
        self.vllm_config = vllm_config
        self.cluster_id_start = cluster_id_start
        self.host_ip = host_ip
        self.host_port = host_port
        logger.info("Initializing LLMDataDist Scheduler %s %s %s", cluster_id_start, host_ip, host_port)
        # initialize the dict to save requests finish time
        self.requests_finish_time = dict()

    def get_num_new_matched_tokens(
            self, request: "Request",
            num_computed_tokens: int) -> tuple[int, bool]:
        return 0, False

    def update_state_after_alloc(self, request: "Request",
                                 blocks: "KVCacheBlocks",
                                 num_external_tokens: int):
        pass

    def build_connector_metadata(
            self,
            scheduler_output: SchedulerOutput,
    ) -> KVConnectorMetadata:
        metadata = DatadistConnectorMetadataPrefill()
        # add requests finish time to metadata, to pass to worker connector
        metadata.requests = {req_id: ReqMetaPrefill(finish_time=finish_time)
                     for req_id, finish_time in self.requests_finish_time.items()}
        self.requests_finish_time.clear()
        return metadata

    def request_finished(
            self,
            request: "Request",
            block_ids: list[int],
            spec_token_ids: Optional[list[int]] = []
    ) -> tuple[bool, Optional[dict[str, Any]]]:
        """
        Once a request is finished, determine whether request blocks
        should be freed now or will be sent asynchronously and freed later.
        """
        if request.status != RequestStatus.FINISHED_LENGTH_CAPPED:
            return False, None

        delay_free_blocks = len(block_ids) > 0
        # record the finish time of the request
        if delay_free_blocks:
            self.requests_finish_time[request.request_id] = time.monotonic()

        return delay_free_blocks, dict(
            remote_block_ids=block_ids,
            remote_cluster_id=self.cluster_id_start,
            remote_host_ip=f"tcp://{self.host_ip}:{self.host_port}",
            spec_token_ids=spec_token_ids,
            remote_dp_rank=self.vllm_config.parallel_config.data_parallel_rank,
            remote_request_id=request.request_id
        )


class PrefillConnectorWorker:
    """Implementation of Worker side methods"""

    def __init__(self, vllm_config: "VllmConfig", host_ip: str, host_port: str):
        # Metadata.
        self.host_ip = host_ip
        self.host_port = host_port
        self.rank = get_tensor_model_parallel_rank()
        if self.rank == 0:
            self.ctx = zmq.Context()
            self.input_socket = self.ctx.socket(zmq.constants.PULL)
            self.input_socket.bind(f"tcp://{self.host_ip}:{self.host_port}")
            logger.info(f"ConnectWorker bind tcp://{self.host_ip}:{self.host_port}")
            self._transfer_lock = threading.Lock()
            self.receive_req_list = []
            thread_name = "prefill_connector_get_pulled_kv_req_list"
            self.thread = threading.Thread(target=self.get_pulled_kv_req_list, daemon=True, name=thread_name)
            self.thread.start()
            dump_thread_to_file(self.thread, thread_name, thread_dump_path)

        # check whether omni attention is enabled
        from omni.accelerators.cache import (OmniBiGroupDataDistManager,
                                             check_omni_attn_cmd_arg)
        use_omni_attn_mgr = check_omni_attn_cmd_arg(vllm_config.additional_config)
        if use_omni_attn_mgr:
            manager_cls = OmniBiGroupDataDistManager
            logger.warning(f"PrefillingConnector is using Omni datadist manager for KV transfer.")
            self.datadist_manager = manager_cls(vllm_config)
        else:
            manager_cls = LLMDataDistManager
            self.datadist_manager = manager_cls(vllm_config)

        # initialize the dict to save requests finish time
        self.requests_finish_time = dict()

    def register_kv_caches(self, kv_pool_mmap_path, data_type, block_len_dtype, start_offset):
        if self.rank == 0:
            self.start_p_server_kv_transfer(kv_pool_mmap_path, data_type, block_len_dtype, start_offset)
        else:
            pass

    def start_p_server_kv_transfer(self, kv_pool_mmap_path, data_type, block_len_dtype, start_offset):
        p_procs = start_p_node_servers(BASE_PORT, 1, [kv_pool_mmap_path], data_type, block_len_dtype, start_offset)
        attach_logs(p_procs)
        ok, not_ready = _wait_ports([("127.0.0.1", BASE_PORT + i) for i in range(1)])
        if not ok:
            for p in p_procs: 
                try: p.kill()
                except: pass
            raise RuntimeError(f"[ERROR] P not ready: {sorted(list(not_ready))}")
        logger.info("[READY] All P nodes are ready")

    def start_load_kv(self, metadata: DatadistConnectorMetadataPrefill):
        pass

    def get_finished(self, metadata: DatadistConnectorMetadataPrefill) -> tuple[set[str], set[str]]:
        """
        Get requests that are done sending or recving.
        """
        all_done_sending: set[str] = set()
        all_done_recving: set[str] = set()
        if self.rank == 0:
            # Update requests_finish_time with new finish times from metadata
            with self._transfer_lock:
                self.requests_finish_time.update(
                    {req_id: meta.finish_time for req_id, meta in metadata.requests.items()}
                )
                current_time = time.monotonic()
                # Identify requests whose finish time exceeds BLOCK_RELEASE_DELAY
                out_date_reqs = []
                for req_id, finish_time in self.requests_finish_time.items():
                    if current_time - finish_time > BLOCK_RELEASE_DELAY:
                        out_date_reqs.append(req_id)
                    else:
                        # Since the dict is ordered by finish_time, we can break early
                        break
                for req_id in out_date_reqs:
                    logger.warning(
                        f"Request {req_id} is out of date, finish time: {self.requests_finish_time[req_id]}. Freeing blocks now."
                    )
                    all_done_sending.add(req_id)
                    del self.requests_finish_time[req_id]

            if len(self.receive_req_list) == 0:
                return all_done_sending, all_done_recving

            with self._transfer_lock:
                for req_id in self.receive_req_list:
                    logger.debug(f"Get_finished: request {req_id}")
                    all_done_sending.add(req_id)
                    # if the request's kv has been received, remove it from requests_finish_time
                    if req_id in self.requests_finish_time:
                        del self.requests_finish_time[req_id]
                self.receive_req_list.clear()

        return all_done_sending, all_done_recving

    def get_pulled_kv_req_list(self):
        while True:
            try:
                if self.input_socket.poll(timeout=10) > 0:
                    message = self.input_socket.recv_string()
                    id_list = json.loads(message)  # Parse the received JSON string into a list
                    logger.debug("Received: %s", id_list)
                    with self._transfer_lock:
                        self.receive_req_list.extend(id_list)
            except Exception as e:
                logger.error("get pulled kv req list failed: %s", e)


class DecodeConnectorScheduler:
    """Implementation of Scheduler side methods"""
    def __init__(self, vllm_config: VllmConfig):
        self.vllm_config = vllm_config
        self.block_size = vllm_config.cache_config.block_size
        self._reqs_need_recv: dict[str, tuple[Request, list[int]]] = {}
        self.processed_request: set[str] = set()

        additional_config = vllm_config.additional_config
        if additional_config:
            self.async_pull_kv = additional_config.get("async_pull_kv", False)
        else:
            self.async_pull_kv = False

        if self.async_pull_kv:
            self.context = zmq.Context()
            self.pub = self.context.socket(zmq.PUB)
            self.pub.bind(f"ipc:///tmp/sched-pub-{vllm_config.parallel_config.data_parallel_rank_local}")

    def get_num_new_matched_tokens(
            self, request: "Request",
            num_computed_tokens: int) -> tuple[int, bool]:
        if request.request_id in self.processed_request:
            return 0, False
        self.processed_request.add(request.request_id)
        params = request.kv_transfer_params
        if params is None:
            return 0, False
        logger.debug(
            "DatadistConnector get_num_new_matched_tokens: "
            "num_computed_tokens=%s, kv_transfer_params=%s",
            num_computed_tokens, params)

        if num_computed_tokens % self.block_size != 0:
            raise RuntimeError("num_computed_tokens must be divisible by self.block_size")
        rounded_num_prompt_tokens = self._round_up(
            len(request.prompt_token_ids), self.block_size)
        count = max(rounded_num_prompt_tokens - num_computed_tokens, 0)
        return count, count > 0

    def _round_up(self, x: int, y: int) -> int:
        return ((x + y - 1) // y) * y

    def update_state_after_alloc(self, request: "Request",
                                 blocks: "KVCacheBlocks",
                                 num_external_tokens: int):
        logger.debug(f"Request id {request.request_id}: blocks length is {len(blocks.blocks)}")
        params = request.kv_transfer_params
        logger.debug(
            "DatadistConnector update_state_after_alloc: "
            "num_external_tokens=%s, kv_transfer_params=%s",
            num_external_tokens, params)

        if params is not None:
            if params.get("remote_block_ids"):
                if all(p in params for p in ("remote_cluster_id", "remote_host_ip")):
                    self._reqs_need_recv[request.request_id] = (
                        request, blocks.get_unhashed_block_ids())
                else:
                    logger.warning(
                        "Got invalid KVTransferParams: %s.", params)

    def build_connector_metadata(
        self,
        scheduler_output: SchedulerOutput,
    ) -> KVConnectorMetadata:
        metadata = DatadistConnectorMetadata()
        for req_id, (req, block_ids) in self._reqs_need_recv.items():
            if req.kv_transfer_params is None:
                logger.warning(f"For request {req_id}: kv_transfer_params now is None")
            else:
                metadata.add_new_req(
                    request_id=req_id,
                    local_block_ids=block_ids,
                    kv_transfer_params=req.kv_transfer_params,
                )
            req.kv_transfer_params = None
        self._reqs_need_recv.clear()

        if self.async_pull_kv:
            if scheduler_output is None:
                # Let go fast path
                if metadata.requests:
                    serialized_data = pickle.dumps(metadata)
                    self.pub.send(serialized_data)

        return metadata

    def request_finished(
            self,
            request: "Request",
            block_ids: list[int],
            spec_token_ids: Optional[list[int]] = []
    ) -> tuple[bool, Optional[dict[str, Any]]]:
        if request.request_id in self.processed_request:
            self.processed_request.remove(request.request_id)
        return False, None


class DecodeConnectorWorker:
    """Worker implementation for datadist."""

    def __init__(self, vllm_config: "VllmConfig", host_ip: str, cluster_id_start: int):
        
        self.lib = CDLL(LIB_PATH)

        # Connection pool
        self.lib.kv_init_conn_pool.argtypes = [POINTER(c_char_p), c_size_t]
        self.lib.kv_init_conn_pool.restype = c_bool
        self.lib.kv_shutdown_conn_pool.argtypes = []
        self.lib.kv_shutdown_conn_pool.restype = None

        # D mmap (note: start_offset added)
        self.lib.kv_dpool_init_from_mmap.argtypes = [c_char_p, c_char_p, c_size_t, c_size_t]
        self.lib.kv_dpool_init_from_mmap.restype = c_bool

        # D pool operations
        self.lib.kv_dpool_set_direct_id_mode.argtypes = [c_uint64]
        self.lib.kv_dpool_set_direct_id_mode.restype  = c_bool
        self.lib.kv_dpool_get_blocks.argtypes = [c_void_p, POINTER(c_uint64), c_size_t, POINTER(KVBlockC)]
        self.lib.kv_dpool_get_blocks.restype = c_size_t
        self.lib.kv_dpool_release.argtypes = [c_void_p, POINTER(c_uint64), c_size_t]
        self.lib.kv_dpool_release.restype = None

        # Decode helper
        self.lib.kv_decode_block_to_string.argtypes = [c_void_p, c_size_t, c_char_p, c_size_t]
        self.lib.kv_decode_block_to_string.restype = c_size_t

        # New: transfer by cluster (cluster_id + cluster_size + 1D p_block_ids)
        self.lib.kv_transfer_cluster_sequential.argtypes = [
            c_char_p,            # request_id
            c_uint32,            # cluster_id
            c_uint32,            # cluster_size
            POINTER(c_uint64),   # p_block_ids (length A)
            c_size_t,            # A
            POINTER(KVBlockC),   # d_blocks (length A)
            c_size_t             # total_blocks (==A)
        ]
        self.lib.kv_transfer_cluster_sequential.restype = c_bool

        self.vllm_config = vllm_config
        self.cluster_id_start = cluster_id_start
        self.dp_rank = vllm_config.parallel_config.data_parallel_rank_local
        additional_config = vllm_config.additional_config
        if additional_config:
            self.async_pull_kv = additional_config.get("async_pull_kv", False)
            self.multi_thread_pull_kv = additional_config.get("multi_thread_pull_kv", False)
            self.multi_rank_pull_kv = additional_config.get("multi_rank_pull_kv", False)
        else:
            self.async_pull_kv = False
            self.multi_thread_pull_kv = False
            self.multi_rank_pull_kv = False
        if self.multi_rank_pull_kv:
            self.multi_thread_pull_kv = True
        if vllm_config.parallel_config.tensor_parallel_size > 1 and self.multi_rank_pull_kv:
            raise ValueError("multi_rank_pull_kv are not supported when tp > 1.")

        from omni.accelerators.cache import (OmniBiGroupDataDistManager,
                                             check_omni_attn_cmd_arg)
        use_omni_attn_mgr = check_omni_attn_cmd_arg(vllm_config.additional_config)
        if use_omni_attn_mgr:
            manager_cls = OmniBiGroupDataDistManager
            logger.warning(f"DecodeConnector is using Omni datadist manager for KV transfer.")
            self.datadist_manager = manager_cls(vllm_config)
        else:
            manager_cls = LLMDataDistManager
            self.datadist_manager = manager_cls(vllm_config)
        self._recving_transfers: list = []
        self._done_recving_count: defaultdict[str, int] = defaultdict(lambda: 0)

        self._pull_kv_lock = threading.Lock()
        self.queues = {} # cluster_id -> queue.Queue
        self.threads = {} # cluster_id -> threading.Thread

        self._transfer_lock = threading.Lock()

        self.ctx = zmq.Context()
        self.zmq_socket_map = {}

        if self.async_pull_kv:
            # dp_rank = vllm_config.parallel_config.data_parallel_rank_local
            thread_name = f"async_pull_kv_{self.dp_rank}"
            self.thread_on_fast_path_req = threading.Thread(target=self.on_fast_path_req, daemon=True, name=thread_name)
            self.thread_on_fast_path_req.start()
            logger.warning(f"DecodeConnectorWorker initialized with self.async_pull_kv enabled.")

            # Write thread name and native_id to file
            dump_thread_to_file(self.thread_on_fast_path_req, thread_name, thread_dump_path)

        if self.multi_thread_pull_kv and self.vllm_config.parallel_config.tensor_parallel_size > 1:
            self.tp_sync_path = f"ipc:///tmp/tp-sync-dp{self.vllm_config.parallel_config.data_parallel_rank}"
            if get_tensor_model_parallel_rank() == 0:
                self.input_socket = self.ctx.socket(zmq.constants.PULL)
                self.input_socket.bind(self.tp_sync_path)
                logger.info(f"ConnectWorker bind {self.tp_sync_path}")

                self.tp_sync_req_dict = {}
                thread_name = f"decode_connector_sync_pulled_tp_kvcache_and_send_dp{self.vllm_config.parallel_config.data_parallel_rank}"
                self.sync_thread = threading.Thread(target=self.sync_pulled_tp_kvcache_and_send, daemon=True,
                                                    name=thread_name)
                self.sync_thread.start()
                dump_thread_to_file(self.sync_thread, thread_name, thread_dump_path)

        logger.info(" ***** Using single thread to pull kv.")
        max_concurrents = 1
        self.executor = ThreadPoolExecutor(max_workers=max_concurrents)

    def sync_pulled_tp_kvcache_and_send(self):
        while True:
            try:
                if self.input_socket.poll(timeout=10) > 0:
                    data = self.input_socket.recv_json()
                    request_id = data.get("request_id")
                    remote_request_id = data.get("remote_request_id")
                    remote_host_ip = data.get("remote_host_ip")
                    # if request_id not in dict, set to 0, else do nothing
                    self.tp_sync_req_dict.setdefault(request_id, 0)
                    self.tp_sync_req_dict[request_id] += 1
                    logger.debug(f"{request_id} finish pull kv {self.tp_sync_req_dict[request_id]} times.")
                    if self.tp_sync_req_dict[request_id] == self.vllm_config.parallel_config.tensor_parallel_size:
                        self.tp_sync_req_dict.pop(request_id)
                        self._send_pulled_kv_req_list(remote_host_ip, [remote_request_id])
                        with self._transfer_lock:
                            self._recving_transfers.append(request_id)
            except Exception as e:
                logger.error("Sync pulled kv when tp > 1 and send failed: %s", e)

    def on_fast_path_req(self):
        context = zmq.Context()
        sub = context.socket(zmq.SUB)
        sub.connect(f"ipc:///tmp/sched-pub-{self.vllm_config.parallel_config.data_parallel_rank_local}")
        sub.setsockopt_string(zmq.SUBSCRIBE, "")

        while True:
            serialized_data = sub.recv()
            metadata = pickle.loads(serialized_data)
            for req_id, meta in metadata.requests.items():
                if (len(meta.local_block_ids) > 0) and (len(meta.remote_block_ids) > 0):
                    self.start_load_kv(metadata)
                    logger.info(
                        "Received fast path request for request %s with "
                        "local_block_ids: %s, remote_block_ids: %s.",
                        req_id,
                        len(meta.local_block_ids),
                        len(meta.remote_block_ids)
                    )

    def worker(self, cluster_id):
        q = self.queues[cluster_id]
        time.sleep(0)
        while True:
            task = q.get()
            if task is None:
                continue
            try:
                self._read_blocks(**task)
            except Exception as e:
                logger.error("KV transfer task failed in thread %s: %s", cluster_id, e)
                self._send_pulled_kv_req_list(task['remote_host_ip'], [task['request_id']])
                raise RuntimeError(f"Failed to pull kv for request:{task['request_id']} from cluster:{cluster_id}.")
            q.task_done()

    def register_kv_caches(self, kv_pool_mmap_path, data_type, block_len_dtype, start_offset):
        self.start_build_connection_with_p_server(NODE_SPECS, kv_pool_mmap_path, data_type, block_len_dtype, start_offset)

    def start_build_connection_with_p_server(self, node_specs, kv_pool_mmap_path, data_type, block_len_dtype, start_offset):
        logger.warning(f"[Trying] D mmap initialize: kv_pool_mmap_path={kv_pool_mmap_path}, data_type={data_type}, block_len_dtype={block_len_dtype}, start_offset={start_offset}")
        if not self.lib.kv_dpool_init_from_mmap(kv_pool_mmap_path.encode('utf-8'), data_type.encode('utf-8'), c_size_t(block_len_dtype), c_size_t(start_offset)):
            raise RuntimeError("[ERROR] init D from mmap failed")
        logger.warning(f"[OK] D mmap initialize: kv_pool_mmap_path={kv_pool_mmap_path}, data_type={data_type}, block_len_dtype={block_len_dtype}, start_offset={start_offset}")

        if not self.lib.kv_dpool_set_direct_id_mode(c_uint64(0)):
            raise RuntimeError("[ERROR] setting direct-id mode failed")
        logger.warning(f"[OK] direct-id: id_base={0}, D logic index = (block_id - id_base)")

        if not self.init_conn_pool_with_retry(node_specs):
            raise RuntimeError("[ERROR] Failed to initialize connection pool to P nodes")
        logger.warning("[OK] ConnectionPool initialized and maintained by C++ background threads")


    def init_conn_pool_with_retry(self, node_specs,
                              max_wait_sec: float = None,   # type: ignore
                              retry_interval_sec: float = None,  # type: ignore
                              backoff: float = None) -> bool:    # type: ignore
        max_wait_sec = float(os.environ.get("CONN_POOL_MAX_WAIT_SEC", "600")) if max_wait_sec is None else float(max_wait_sec)
        retry_interval_sec = float(os.environ.get("CONN_POOL_RETRY_INTERVAL_SEC", "0.5")) if retry_interval_sec is None else float(retry_interval_sec)
        backoff = float(os.environ.get("CONN_POOL_BACKOFF", "1.5")) if backoff is None else float(backoff)

        deadline = time.time() + max_wait_sec
        attempt = 1
        while time.time() < deadline:
            arr = (c_char_p * len(node_specs))(*[s.encode("utf-8") for s in node_specs])
            if self.lib.kv_init_conn_pool(arr, c_size_t(len(node_specs))):
                print(f"[OK] ConnectionPool initialized (attempt {attempt})", flush=True)
                return True
            remaining = max(0.0, deadline - time.time())
            print(f"[RETRY] kv_init_conn_pool failed (attempt {attempt}), retry in {retry_interval_sec:.2f}s, remaining {remaining:.1f}s", flush=True)
            time.sleep(retry_interval_sec)
            retry_interval_sec = min(retry_interval_sec * backoff, 5.0)
            attempt += 1
        return False

    # Now go asynchronous pull_kv
    def start_load_kv(self, metadata: DatadistConnectorMetadata):
        logger.info(f" ***** start_load_kv: {len(metadata.requests)}")
        futures = []
        for req_id, meta in metadata.requests.items():
            # if the local_block_ids is empty, skip pulling kv for the request
            if len(meta.local_block_ids) == 0:
                logger.info(f" ***** Request {req_id} has 0 local blocks, skip load kv.")
                continue
            # If local_block_ids is a flat list of int, omni-attention is not used
            # and we can directly use the local_block_ids and remote_block_ids
            if isinstance(meta.local_block_ids[0], int):
                # local_block_ids (kv blocks in D) is more than remote_block_ids (kv blocks in P)
                # led by lookahead num, which is used by eagle and multi step
                if len(meta.remote_block_ids) < len(meta.local_block_ids):
                    meta.local_block_ids = meta.local_block_ids[:len(meta.remote_block_ids)]
                    logger.debug("look ahead token num is greater than 0")
                # If remote_block_ids is more than local_block_ids, we only need the last N remote blocks
                # where N is the number of local blocks
                elif len(meta.remote_block_ids) > len(meta.local_block_ids):
                    meta.remote_block_ids = meta.remote_block_ids[-len(meta.local_block_ids):]
                logger.info(
                    " ***** start_load_kv for request %s "
                    "Num local_block_ids: %s. Num remote_block_ids: %s.",
                    req_id,
                    len(meta.local_block_ids),
                    len(meta.remote_block_ids)
                )
            # If local_block_ids is a list of lists (e.g., [[], []]), omni-attention is used
            # local_block_ids[0] is a list of local block ids for uncompressed layers
            # local_block_ids[1] is a list of local block ids for compressed layers
            elif isinstance(meta.local_block_ids[0], list):
                # If local_block_ids[0] is a list of lists, we need to ensure that remote_block_ids
                # is a list of lists as well, where each sublist corresponds to the local_block
                meta.remote_block_ids = [meta.remote_block_ids] * len(meta.local_block_ids)
                # If local_block_ids[0] is empty, skip pulling kv for the request
                if len(meta.local_block_ids[0]) == 0:
                    logger.info(f" ***** Request {req_id} has 0 local blocks, skip load kv.")
                    continue
                # remote_block_ids in P is less than local_block_ids[0] in D, 
                # led by lookahead num, which is used by eagle and multi step
                elif len(meta.remote_block_ids[0]) < len(meta.local_block_ids[0]):
                    meta.local_block_ids[0] = meta.local_block_ids[0][:len(meta.remote_block_ids[0])]
                    logger.debug("look ahead token num is greater than 0")
                # If remote_block_ids in P is more than local_block_ids[0] in D, we only need the last N remote blocks
                elif len(meta.remote_block_ids[0]) > len(meta.local_block_ids[0]):
                    meta.remote_block_ids[0] = meta.remote_block_ids[0][-len(meta.local_block_ids[0]):]
                logger.info(
                    " ***** start_load_kv for request %s "
                    "Num local_block_ids: %s. Num remote_block_ids: %s.",
                    req_id,
                    len(meta.local_block_ids[0]),
                    len(meta.remote_block_ids[0])
                )
            # handle the unexpected case where local_block_ids is not a list of int or list of lists
            else:
                logger.error(f"Unexpected type for meta.local_block_ids[0]: {type(meta.local_block_ids[0])}")
                raise RuntimeError(f"Unexpected type for meta.local_block_ids[0]: {type(meta.local_block_ids[0])}")
            # cluster_ids = self.datadist_manager.get_real_remote_cluster_ids(meta)
            cluster_ids = [0]
            if self.multi_rank_pull_kv:
                # If multi_rank_pull_kv is enabled, each DP rank will pull kv from multiple P ranks
                # and the cluster_ids are obtained from registered_link_infos
                # If the local_block_ids is a flat list of int, we can directly use it
                # As multi_rank_pull_kv is designed to pull kv from two P ranks,
                # we split the local_block_ids and remote_block_ids into two parts
                if not isinstance(meta.local_block_ids[0], list):
                    block_thre = len(meta.local_block_ids) // 2
                # If the local_block_ids is a flat list of list, only split the blocks for uncompressed layers
                else:
                    block_thre = len(meta.local_block_ids[0]) // 2
                for idx_cluster, cluster_id in enumerate(cluster_ids):
                    if not isinstance(meta.local_block_ids[0], list):
                        if idx_cluster == 0:
                            local_blocks = meta.local_block_ids[:block_thre]
                            remote_blocks = meta.remote_block_ids[:block_thre]
                            len_local_blocks = len(local_blocks)
                        else:
                            local_blocks = meta.local_block_ids[block_thre:]
                            remote_blocks = meta.remote_block_ids[block_thre:]
                            len_local_blocks = len(local_blocks)
                    else:
                        if idx_cluster == 0:
                            # For uncompressed layers, split the local_block_ids[0] and remote_block_ids
                            # For compressed layers, only pull kv from the second P rank
                            local_blocks = [meta.local_block_ids[0][:block_thre], []]
                            # remote_blocks need to be split as well for getting kv blocks for compressed layers in P
                            remote_blocks = [meta.remote_block_ids[0][:block_thre], []]
                            len_local_blocks = len(local_blocks[0])
                        else:
                            local_blocks = [meta.local_block_ids[0][block_thre:], meta.local_block_ids[1]]
                            remote_blocks = [meta.remote_block_ids[0][block_thre:], meta.remote_block_ids[1]]
                            len_local_blocks = len(local_blocks[0])
                    if len_local_blocks > 0:
                        task = {
                            'request_id': req_id,
                            'remote_request_id': meta.remote_request_id,
                            'dst_cluster_id': cluster_id,
                            'local_block_ids': local_blocks,
                            'remote_block_ids': remote_blocks,
                            'remote_host_ip': meta.remote_host,
                        }
                        logger.warning(f"*********** dst cluster_id is {cluster_id}.")
                        self.queues[cluster_id].put(task)
            elif self.multi_thread_pull_kv:
                task = {
                    'request_id': req_id,
                    'remote_request_id': meta.remote_request_id,
                    'dst_cluster_id': cluster_ids[0],
                    'local_block_ids': meta.local_block_ids,
                    'remote_block_ids': meta.remote_block_ids,
                    'remote_host_ip': meta.remote_host,
                }

                self.queues[cluster_ids[0]].put(task)
            else:
                # Use ThreadPoolExecutor to handle the task
                future = self.executor.submit(
                    self._read_blocks,
                    local_block_ids=meta.local_block_ids,
                    remote_block_ids=meta.remote_block_ids,
                    dst_cluster_id=cluster_ids[0],
                    request_id=req_id,
                    remote_request_id=meta.remote_request_id,
                    remote_host_ip=meta.remote_host,
                )
                futures.append(future)

        if not self.multi_thread_pull_kv:
            for future in futures:
                future.add_done_callback(handle_exception)

    def _read_blocks(
        self,
        local_block_ids: list[int],
        remote_block_ids: list[int],
        dst_cluster_id: str,
        request_id: str,
        remote_request_id: str,
        remote_host_ip: str,
    ):
        start = time.time()

        local_block_ids = [x % 44 for x in local_block_ids]
        total_blocks = len(local_block_ids)
        DBlocksArray = KVBlockC * total_blocks
        d_blocks = DBlocksArray()
        d_ids_list = local_block_ids

        # get pointers/size of each D blocks
        DBlocksArray = KVBlockC * total_blocks
        d_blocks = DBlocksArray()
        filled = self.lib.kv_dpool_get_blocks(None, (c_uint64 * total_blocks)(*d_ids_list), c_size_t(total_blocks), d_blocks)
        if filled == 0:
            self.lib.kv_dpool_set_direct_id_mode(c_uint64(0))
            logger.warning(f"kv_dpool timeout, reset it to direct_id_mode")
            filled = self.lib.kv_dpool_get_blocks(None, (c_uint64 * total_blocks)(*d_ids_list), c_size_t(total_blocks), d_blocks)
        if filled != total_blocks:
            raise RuntimeError(f"[ERROR] d_get_blocks failed: {filled}/{total_blocks}")

        p_ids_arr = (c_uint64 * total_blocks)(*remote_block_ids)
        trans_ok = self.lib.kv_transfer_cluster_sequential(
                    c_char_p(request_id.encode("utf-8")),
                    c_uint32(int(dst_cluster_id)),
                    c_uint32(int(NUM_P_NODES)), # for 1p1d, should be CLUSTER_SIZE for xPyD
                    p_ids_arr,
                    c_size_t(total_blocks),
                    d_blocks,
                    c_size_t(total_blocks)
        )
        if not trans_ok:
            raise RuntimeError("[ERROR] KV transfer failed")

        if self.vllm_config.parallel_config.tensor_parallel_size == 1:
            # tp=1, send to prefill tp rank0 directly.
            self._send_pulled_kv_req_list(remote_host_ip, [remote_request_id])
            with self._transfer_lock:
                self._recving_transfers.append(request_id)
        else:
            if self.multi_thread_pull_kv:
                # tp>1, send to decode to rank0 firstly.
                self._send_pulled_kv_req_list(
                    self.tp_sync_path,
                    {
                        "request_id": request_id,
                        "remote_request_id": remote_request_id,
                        "remote_host_ip": remote_host_ip
                    }
                )
            else:
                torch.distributed.barrier(group=get_tp_group().cpu_group)
                if get_tensor_model_parallel_rank() == 0:
                    self._send_pulled_kv_req_list(remote_host_ip, [remote_request_id])
                with self._transfer_lock:
                    self._recving_transfers.append(request_id)
        logger.debug(f" ***** read block, req_id:{request_id}, local_block_ids:{local_block_ids}, remote_block_ids:{remote_block_ids}")
        cost = time.time() - start
        logger.info(f" ***** read block, req_id:{request_id}, cost:{cost:.6f}")


    def _send_pulled_kv_req_list(self, path, data):
        if path in self.zmq_socket_map:
            socket = self.zmq_socket_map[path]
        else:
            socket = self.ctx.socket(zmq.PUSH)
            socket.connect(path)
            self.zmq_socket_map[path] = socket
            logger.info(f"create new socket path:{path}")

        try:
            json_data = json.dumps(data)
            socket.send_string(json_data)
            logger.info(f"send string {json_data} path:{path}")
        except Exception as e:
            logger.error(f"Failed to send reqest_id {json_data} to prefill: {e}")

    def get_finished(self, metadata: DatadistConnectorMetadata) -> tuple[set[str], set[str]]:
        # for decode size, done_sending is no need
        all_done_sending: set[str] = set()
        with self._transfer_lock:
            all_done_recving = self._pop_done_transfers(self._recving_transfers)
        if len(all_done_recving) > 0:
            logger.debug(
                "Get_finished: %s requests done recving", len(all_done_recving))

        return all_done_sending, all_done_recving

    def _pop_done_transfers(self, transfers: list) -> set[str]:
        done_req_ids: set[str] = set()
        for req_id in transfers:
            done_req_ids.add(req_id)
        self._recving_transfers.clear()
        return done_req_ids

def handle_exception(future):
    if future.exception():
        logger.error(f"Exception occurred in future: {future.exception()}")
        raise future.exception()

def create_or_resize_file(path: str, size: int):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        f.truncate(size)

def mmap_rw(path: str, size: int):
    f = open(path, 'r+b')
    mm = pymmap.mmap(f.fileno(), size, access=pymmap.ACCESS_WRITE)
    return f, mm

def _wait_ports(host_ports, timeout_sec=READINESS_TIMEOUT, interval=READINESS_INTERVAL):
    remaining = set(host_ports)
    deadline = time.time() + timeout_sec
    def can_connect(h, p, to):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(to)
        try:
            s.connect((h, p))
            return True
        except Exception:
            return False
        finally:
            try: s.close()
            except Exception: pass
    while remaining and time.time() < deadline:
        ready = []
        for hp in list(remaining):
            if can_connect(hp[0], hp[1], interval):
                ready.append(hp)
        for hp in ready:
            remaining.discard(hp)
        if remaining:
            time.sleep(interval)
    return len(remaining) == 0, remaining

def start_p_node_servers(base_port: int, servers: int, mmap_paths, dtype: str, offset_dtype: int, start_offset: int):
    procs = []
    for nid in range(servers):
        port = base_port + nid
        args = [
            P_NODE_BIN,
            "--node_id", str(nid),
            "--port", str(port),
            "--mmap", mmap_paths[nid],
            "--dtype", dtype,
            "--offset_dtype", str(offset_dtype),
            "--start_offset", str(start_offset),
        ]
        p = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
        threading.Thread(target=lambda pref, pipe: [print(f'{pref} {l.rstrip()}', flush=True) for l in iter(pipe.readline, '')],
                         args=(f"[P{nid}] OUT", p.stdout), daemon=True).start()
        threading.Thread(target=lambda pref, pipe: [print(f'{pref} {l.rstrip()}', flush=True) for l in iter(pipe.readline, '')],
                         args=(f"[P{nid}] ERR", p.stderr), daemon=True).start()
        procs.append(p)
        time.sleep(0.05)
    return procs

def attach_logs(procs):
    def stream(prefix, pipe):
        for line in iter(pipe.readline, ''):
            print(f"{prefix} {line.rstrip()}", flush=True)
    for i, p in enumerate(procs):
        threading.Thread(target=stream, args=(f"[P{i}] OUT", p.stdout), daemon=True).start()
        threading.Thread(target=stream, args=(f"[P{i}] ERR", p.stderr), daemon=True).start()

def dump_thread_to_file(thread, thread_name: str, folder_path: str):

    timeout = 5  # seconds
    start_time = time.time()
    while not hasattr(thread, "native_id"):
        if time.time() - start_time > timeout:
            logger.error(f"Timeout waiting for thread {thread_name} to have native_id.")
            return
        time.sleep(0.005)

    # Ensure the folder exists
    if not os.path.exists(folder_path):
        try:
            os.makedirs(folder_path, exist_ok=True)
        except Exception as e:
            logger.error(f"Failed to create folder {folder_path}: {e}")
            return

    file_path = os.path.join(folder_path, thread_name)
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(str(thread.native_id))
    except Exception as e:
        logger.error(f"Failed to write thread info to {file_path}: {e}")
