# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

import json
import os
import pickle
import queue
import socket
import struct
import subprocess
import threading
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional, Tuple, List, Dict

import torch
import zmq
from concurrent.futures import ThreadPoolExecutor

from omni.accelerators.cache.omni_cache import BaseOmniCache

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1, KVConnectorMetadata, KVConnectorRole)
from vllm.distributed.parallel_state import (
    get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size,
    get_tp_group)
from vllm.envs import VLLM_RPC_TIMEOUT
from vllm.logger import init_logger
from vllm.utils import get_open_port

from omni.accelerators.pd.utils import get_config_from_dict_or_env
# from omni.accelerators.pd.llmdatadist_manager import LLMDataDistManager, LLMDataDistConfig

if TYPE_CHECKING:
    from vllm.config import KVTransferConfig
    from vllm.attention.backends.abstract import AttentionMetadata
    from vllm.forward_context import ForwardContext
    from vllm.v1.request import Request
from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.request import Request, RequestStatus
from vllm.model_executor.models.utils import extract_layer_index

GET_META_MSG = b"get_meta_msg"
logger = init_logger(__name__)

# seconds, used to free blocks after a delay once the request is finished
BLOCK_RELEASE_DELAY = 3000

BASE_DIR = os.path.dirname(__file__)
P_NODE_BIN = os.environ.get("P_NODE_BIN", os.path.join(BASE_DIR, "bin/p_node_server"))
D_AGENT_BIN = os.environ.get("D_AGENT_BIN", os.path.join(BASE_DIR, "bin/d_kv_agent"))

# Cluster/P-node configuration
P_NODES_ENV = os.environ.get("P_NODE_LIST", "7.150.13.67,7.150.14.143")
NUM_P_NODES = len(P_NODES_ENV.split(","))
CLUSTER_SIZE = int(os.environ.get("CLUSTER_SIZE", "1"))

BASE_PORT = int(os.environ.get("BASE_PORT", "15077"))
DIRECT_ID_BASE = int(os.environ.get("DIRECT_ID_BASE", "0"))
ZMQ_BASE_PORT = int(os.environ.get("ZMQ_BASE_PORT", "17555"))
VERBOSE = int(os.environ.get("VERBOSE", "1"))

# Normalize node list and specs
_P_NODE_LIST_RAW = [h.strip() for h in P_NODES_ENV.split(",") if h.strip()]
if not _P_NODE_LIST_RAW:
    _P_NODE_LIST_RAW = ["127.0.0.1"]
# If NUM_P_NODES is larger than the list length, cap it to avoid IndexError
NUM_P_NODES = min(NUM_P_NODES, len(_P_NODE_LIST_RAW))
P_NODE_LIST = _P_NODE_LIST_RAW[:NUM_P_NODES]
NODE_SPECS = ";".join([f"{P_NODE_LIST[i]}:{BASE_PORT}:{i}" for i in range(NUM_P_NODES)])

_print_lock = threading.Lock()

@dataclass
class ReqMeta:
    local_block_ids: List[List[int]]
    remote_block_ids: List[int]
    remote_host: str
    remote_cluster_id: str
    spec_token_ids: Optional[List[int]]
    remote_dp_rank: Optional[int]
    remote_tp_size: Optional[int]
    remote_request_id: Optional[str]


@dataclass
class ReqMetaPrefill:
    finish_time: float


class DatadistConnectorMetadata(KVConnectorMetadata):
    """Metadata for datadist connector (decode path)."""

    def __init__(self):
        self.requests: Dict[str, ReqMeta] = {}

    def add_new_req(
        self,
        request_id: str,
        local_block_ids: List[List[int]],
        kv_transfer_params: Dict[str, Any],
    ):
        self.requests[request_id] = ReqMeta(
            local_block_ids=local_block_ids,
            remote_block_ids=kv_transfer_params["remote_block_ids"],
            remote_host=kv_transfer_params["remote_host_ip"],
            remote_cluster_id=kv_transfer_params["remote_cluster_id"],
            spec_token_ids=kv_transfer_params["spec_token_ids"],
            remote_dp_rank=kv_transfer_params.get("remote_dp_rank", 0),
            remote_tp_size=kv_transfer_params.get("remote_tp_size", 16),
            remote_request_id=kv_transfer_params.get("remote_request_id"),
        )


class DatadistConnectorMetadataPrefill(KVConnectorMetadata):
    """Metadata for datadist connector (prefill path)."""

    def __init__(self):
        self.requests: Dict[str, ReqMetaPrefill] = {}

    def add_new_req(
        self,
        request_id: str,
        finish_time: float,
    ):
        self.requests[request_id] = ReqMetaPrefill(finish_time=finish_time)


class LLMDataDistConnector(KVConnectorBase_V1):
    def __init__(self, vllm_config: VllmConfig, role: KVConnectorRole):
        if vllm_config.kv_transfer_config is None:
            raise RuntimeError("vllm_config.kv_transfer_config cannot be None")

        if vllm_config.model_config.is_deepseek_mla:
            vllm_config.kv_transfer_config.kv_parallel_size = 1
            logger.info("Set kv_parallel_size to 1 when using deepseek MLA model.")

        # self.datadist_config = LLMDataDistConfig(vllm_config, ignore_load_rank=True)
        self.is_prefill = vllm_config.kv_transfer_config.kv_role == "kv_producer"
        if self.is_prefill:
            target_ip = self.get_local_ip()
            node_idx = self.get_ip_index(NODE_SPECS, target_ip)
            self.cluster_id_start = node_idx // CLUSTER_SIZE
            self.host_ip = _P_NODE_LIST_RAW[self.cluster_id_start * CLUSTER_SIZE]
            # Resolve ZMQ port conflicts in multi-P deployments on the same machine.
            self.host_port = get_config_from_dict_or_env(
                vllm_config.kv_transfer_config, "kv_port",
                "VLLM_LLMDATADIST_ZMQ_PORT", "5568", int)
            dp_rank = vllm_config.parallel_config.data_parallel_rank
            self.host_port += dp_rank
        else:
            # in decode instance, these twos are not used, just send some random thing to it
            self.host_ip = "127.0.0.1"
            self.cluster_id_start = 0

        if role == KVConnectorRole.SCHEDULER:
            if self.is_prefill:
                self.connector_scheduler = PrefillConnectorScheduler(
                    vllm_config, self.cluster_id_start, self.host_ip, str(self.host_port))
            else:
                self.connector_scheduler = DecodeConnectorScheduler(vllm_config)
            self.connector_worker = None
        elif role == KVConnectorRole.WORKER:
            if self.is_prefill:
                self.connector_worker = PrefillConnectorWorker(
                    vllm_config, str(self.host_ip), str(self.host_port))
            else:
                self.connector_worker = DecodeConnectorWorker(
                    vllm_config, str(self.host_ip), self.cluster_id_start)
            self.connector_scheduler = None

    def get_ip_index(self, node_specs: str, target_ip: str) -> int:
        for spec in node_specs.split(";"):
            ip, *_, idx = spec.split(":")
            if ip == target_ip:
                return int(idx)
        raise ValueError(f"{target_ip} not found in node specs")

    def get_local_ip(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect(('8.8.8.8', 80))
            ip = s.getsockname()[0]
        finally:
            s.close()
        return ip

    ############################################################
    # Scheduler Side Methods
    ############################################################

    def get_num_new_matched_tokens(
            self, request: "Request",
            num_computed_tokens: int) -> Tuple[int, bool]:
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
            block_ids: List[int],
            spec_token_ids: Optional[List[int]] = None
    ) -> Tuple[bool, Optional[dict]]:
        if self.connector_scheduler is None:
            raise RuntimeError("self.connector_scheduler cannot be None")
        return self.connector_scheduler.request_finished(request, block_ids, spec_token_ids or [])

    ############################################################
    # Worker Side Methods
    ############################################################
    def register_kv_caches(self, kv_pool_mmap_path, data_type, block_len_dtype, start_offset=0, omni_cache=None):
        data_type = 'bf16'
        if self.connector_worker is None:
            raise RuntimeError("self.connector_worker cannot be None")
        return self.connector_worker.register_kv_caches(kv_pool_mmap_path, data_type, block_len_dtype, start_offset, omni_cache)

    def get_finished(self,
                     finished_req_ids: set[str]) -> Tuple[set[str], set[str]]:
        """Get the finished recving and sending requests."""
        if self.connector_worker is None:
            raise RuntimeError("self.connector_worker cannot be None")
        # finished_req_ids is currently not used; we forward internal metadata
        return self.connector_worker.get_finished(self._connector_metadata)

    def start_load_kv(self, forward_context: "ForwardContext",
                      **kwargs) -> None:
        if self.connector_worker is None:
            raise RuntimeError("self.connector_worker cannot be None")
        if not isinstance(self._connector_metadata, (DatadistConnectorMetadata, DatadistConnectorMetadataPrefill)):
            raise RuntimeError("self._connector_metadata must be DatadistConnectorMetadata or DatadistConnectorMetadataPrefill")
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
    """Implementation of Scheduler side methods (prefill)."""

    def __init__(self, vllm_config, cluster_id_start: str, host_ip: str, host_port: str):
        self.vllm_config = vllm_config
        self.cluster_id_start = cluster_id_start
        self.host_ip = host_ip
        self.host_port = host_port
        logger.info("Initializing LLMDataDist Scheduler %s %s %s", cluster_id_start, host_ip, host_port)
        # initialize the dict to save requests finish time
        self.requests_finish_time: Dict[str, float] = {}

    def get_num_new_matched_tokens(
            self, request: "Request",
            num_computed_tokens: int) -> Tuple[int, bool]:
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
            block_ids: List[int],
            spec_token_ids: Optional[List[int]] = None
    ) -> Tuple[bool, Optional[dict]]:
        """
        Once a request is finished, determine whether request blocks
        should be freed now or will be sent asynchronously and freed later.
        """
        spec_token_ids = spec_token_ids or []
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
            remote_tp_size=self.vllm_config.parallel_config.tensor_parallel_size,
            remote_request_id=request.request_id
        )


class PrefillConnectorWorker:
    """Implementation of Worker side methods (prefill)."""

    def __init__(self, vllm_config: "VllmConfig", host_ip: str, host_port: str):
        # Metadata.
        self.host_ip = host_ip
        self.host_port = host_port
        self.vllm_config = vllm_config
        self.rank = get_tensor_model_parallel_rank()
        if self.rank == 0:
            self.ctx = zmq.Context()
            self.input_socket = self.ctx.socket(zmq.constants.PULL)
            self.input_socket.bind(f"tcp://{self.host_ip}:{self.host_port}")
            logger.info(f"ConnectWorker bind tcp://{self.host_ip}:{self.host_port}")
            self._transfer_lock = threading.Lock()
            self.receive_req_list: List[str] = []
            thread_name = "prefill_connector_get_pulled_kv_req_list"
            self.thread = threading.Thread(target=self.get_pulled_kv_req_list, daemon=True, name=thread_name)
            self.thread.start()

        # # check whether omni attention is enabled
        # from omni.accelerators.cache import OmniBiGroupDataDistManager, check_omni_attn_cmd_arg
        # use_omni_attn_mgr = check_omni_attn_cmd_arg(vllm_config.additional_config)
        # # if use_omni_attn_mgr or False:
        # #     manager_cls = OmniBiGroupDataDistManager
        # #     logger.warning("PrefillingConnector is using Omni datadist manager for KV transfer.")
        # #     self.datadist_manager = manager_cls(vllm_config)
        # # else:
        # manager_cls = LLMDataDistManager
        # self.datadist_manager = manager_cls(vllm_config)

        # initialize the dict to save requests finish time
        self.requests_finish_time: Dict[str, float] = {}

    def register_kv_caches(self, kv_pool_mmap_path, data_type, block_len_dtype, start_offset, omni_cache=None):
        self.start_p_server_kv_transfer(kv_pool_mmap_path, data_type, block_len_dtype, start_offset)

    def start_p_server_kv_transfer(self, kv_pool_mmap_path, data_type, block_len_dtype, start_offset):
        self.tp_rank_local = self.rank % (self.vllm_config.parallel_config.tensor_parallel_size // CLUSTER_SIZE)
        if self.tp_rank_local == 0:
            # Pass a list for mmap paths; start one P server
            p_procs = start_p_node_servers(BASE_PORT, 1, [kv_pool_mmap_path], data_type, block_len_dtype, start_offset)
            ok, not_ready = _wait_ports([("127.0.0.1", BASE_PORT)])
            if not ok:
                log_line("[ERROR]", f"P not ready: {sorted(list(not_ready))}")
                for p, th in p_procs:
                    stop_logged_process(p, th)
                raise RuntimeError(f"[ERROR] P not ready: {sorted(list(not_ready))}")
            logger.info("[READY] P node is ready")

    def start_load_kv(self, metadata: DatadistConnectorMetadataPrefill):
        pass

    def get_finished(self, metadata: DatadistConnectorMetadataPrefill) -> Tuple[set[str], set[str]]:
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
                out_date_reqs: List[str] = []
                for req_id, finish_time in list(self.requests_finish_time.items()):
                    if current_time - finish_time > BLOCK_RELEASE_DELAY:
                        out_date_reqs.append(req_id)
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
                # pyzmq Socket.poll timeout is in milliseconds; check every 1s
                if self.input_socket.poll(timeout=100) > 0:
                    message = self.input_socket.recv_string()
                    id_list = json.loads(message)  # Parse the received JSON string into a list
                    logger.debug("Received: %s", id_list)
                    with self._transfer_lock:
                        self.receive_req_list.extend(id_list)
            except Exception as e:
                logger.error("get pulled kv req list failed: %s", e)


class DecodeConnectorScheduler:
    """Implementation of Scheduler side methods (decode)."""

    def __init__(self, vllm_config: VllmConfig):
        self.vllm_config = vllm_config
        self.block_size = vllm_config.cache_config.block_size
        self._reqs_need_recv: Dict[str, Tuple[Request, List[int]]] = {}
        self.processed_request: set[str] = set()

        additional_config = vllm_config.additional_config or {}
        self.async_pull_kv = additional_config.get("async_pull_kv", False)

        if self.async_pull_kv:
            self.context = zmq.Context()
            self.pub = self.context.socket(zmq.PUB)
            self.pub.bind(f"ipc:///tmp/sched-pub-{vllm_config.parallel_config.data_parallel_rank_local}")

    def get_num_new_matched_tokens(
            self, request: "Request",
            num_computed_tokens: int) -> Tuple[int, bool]:
        if request.request_id in self.processed_request:
            return 0, False
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
                    logger.warning("Got invalid KVTransferParams: %s.", params)
                    
        self.processed_request.add(request.request_id)

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
            # Fast-path publish (scheduler_output may be None on fast path)
            if scheduler_output is None and metadata.requests:
                serialized_data = pickle.dumps(metadata)
                self.pub.send(serialized_data)

        return metadata

    def request_finished(
            self,
            request: "Request",
            block_ids: List[int],
            spec_token_ids: Optional[List[int]] = None
    ) -> Tuple[bool, Optional[dict]]:
        if request.request_id in self.processed_request:
            self.processed_request.remove(request.request_id)
        return False, None


class DecodeConnectorWorker:
    """Worker implementation for datadist (decode)."""

    def __init__(self, vllm_config: "VllmConfig", host_ip: str, cluster_id_start: int):
        self.vllm_config = vllm_config
        self.cluster_id_start = cluster_id_start
        self.dp_rank = vllm_config.parallel_config.data_parallel_rank_local
        additional_config = vllm_config.additional_config or {}
        self.async_pull_kv = additional_config.get("async_pull_kv", False)

        # from omni.accelerators.cache import OmniBiGroupDataDistManager, check_omni_attn_cmd_arg
        # use_omni_attn_mgr = check_omni_attn_cmd_arg(vllm_config.additional_config)
        # # if use_omni_attn_mgr or False:
        # #     manager_cls = OmniBiGroupDataDistManager
        # #     logger.warning("DecodeConnector is using Omni datadist manager for KV transfer.")
        # #     self.datadist_manager = manager_cls(vllm_config)
        # # else:
        # manager_cls = LLMDataDistManager
        # self.datadist_manager = manager_cls(vllm_config)

        self._recving_transfers: List[str] = []
        self._done_recving_count: defaultdict[str, int] = defaultdict(lambda: 0)

        self._pull_kv_lock = threading.Lock()
        self.queues: Dict[str, queue.Queue] = {}     # cluster_id -> Queue
        self.threads: Dict[str, threading.Thread] = {}  # cluster_id -> Thread

        self._transfer_lock = threading.Lock()

        self.ctx = zmq.Context()
        self.zmq_socket_map: Dict[str, zmq.Socket] = {}

        logger.info(" ***** Using single thread to pull kv.")
        max_concurrents = 1
        self.executor = ThreadPoolExecutor(max_workers=max_concurrents)

        self.omni_cache: BaseOmniCache = None

        if self.async_pull_kv:
            thread_name = f"async_pull_kv_{self.dp_rank}"
            self.thread_on_fast_path_req = threading.Thread(
                target=self.on_fast_path_req, daemon=True, name=thread_name)
            self.thread_on_fast_path_req.start()
            logger.warning("DecodeConnectorWorker initialized with self.async_pull_kv enabled.")

    def on_fast_path_req(self):
        context = zmq.Context()
        sub = context.socket(zmq.SUB)
        sub.connect(f"ipc:///tmp/sched-pub-{self.vllm_config.parallel_config.data_parallel_rank_local}")
        sub.setsockopt_string(zmq.SUBSCRIBE, "")

        while True:
            serialized_data = sub.recv()
            metadata: DatadistConnectorMetadata = pickle.loads(serialized_data)
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

    def register_kv_caches(self, kv_pool_mmap_path, data_type, block_len_dtype, start_offset, omni_cache):
        self.start_offset = start_offset
        self.block_len_dtype  = block_len_dtype
        self.agents: List[Tuple[subprocess.Popen, List[threading.Thread], str, int]] = []
        endpoint = f"tcp://127.0.0.1:{ZMQ_BASE_PORT + self.dp_rank}"
        args = [
            D_AGENT_BIN,
            "--nodes", NODE_SPECS,
            "--d_mmap", kv_pool_mmap_path,
            "--dtype", data_type,
            "--offset_dtype", str(block_len_dtype),
            "--start_offset", str(start_offset),
            "--direct_id_base", str(0),
            "--zmq_bind", endpoint,
            "--conns_per_node", "16"
        ]
        log_line(f"[D{self.dp_rank}]", " ".join(args))
        p, threads = spawn_logged_process(args, f"D{self.dp_rank}")
        self.agents.append((p, threads, endpoint, start_offset))

        # Build REQ sockets for agents
        self.sockets: List[zmq.Socket] = []
        for _, (_, _, agent_endpoint, _start_offset) in enumerate(self.agents):
            s = zmq.Context.instance().socket(zmq.REQ)
            s.connect(agent_endpoint)
            self.sockets.append(s)

        self.omni_cache = omni_cache

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
                # Adjust for lookahead tokens (eagle/multistep)
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
            # If local_block_ids is a list of lists (omni-attention case)
            elif isinstance(meta.local_block_ids[0], List):
                # If local_block_ids[0] is a list of lists, we need to ensure that remote_block_ids
                # is a list of lists as well, where each sublist corresponds to the local_block
                meta.remote_block_ids = [meta.remote_block_ids] * len(meta.local_block_ids)
                # If local_block_ids[0] is empty, skip pulling kv for the request
                if len(meta.local_block_ids[0]) == 0:
                    logger.info(f" ***** Request {req_id} has 0 local blocks, skip load kv.")
                    continue
                # remote_block_ids in P is less than local_block_ids[0] in D,
                # leaded by lookahead num, which is used by eagle and multi step
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
            # Use ThreadPoolExecutor to handle the task
            # TODO:now not support omni-attention case yet
            future = self.executor.submit(
                self._read_blocks,
                local_block_ids=meta.local_block_ids,
                remote_block_ids=meta.remote_block_ids if isinstance(meta.remote_block_ids[0], int) else meta.remote_block_ids[0],
                dst_cluster_id=meta.remote_cluster_id,
                request_id=req_id,
                remote_request_id=meta.remote_request_id,
                remote_tp_size=meta.remote_tp_size,
                remote_host_ip=meta.remote_host,
            )
            futures.append(future)

        for future in futures:
            future.add_done_callback(handle_exception)

    def _read_blocks(
        self,
        local_block_ids: List[List[int]],
        remote_block_ids: List[int],
        dst_cluster_id: str,
        request_id: str,
        remote_request_id: Optional[str],
        remote_host_ip: str,
        remote_tp_size: int,
    ):
        dst_cluster_id = int(dst_cluster_id) // remote_tp_size
        cluster_id = int(dst_cluster_id)
        start = time.time()

        # Create a temporary REQ socket per request to avoid concurrent reuse of the same REQ socket.
        tmp_sockets: list[zmq.Socket] = []
        try:
            for inst, (_proc, _threads, endpoint, _start_offset) in enumerate(self.agents):
                s = zmq.Context.instance().socket(zmq.REQ)
                s.setsockopt(zmq.RCVTIMEO, 5000)
                s.setsockopt(zmq.LINGER, 0)
                s.connect(endpoint)
                tmp_sockets.append(s)

            for inst, s in enumerate(tmp_sockets):
                cluster_id = int(dst_cluster_id)
                frames = [
                    b"pull_kv",
                    request_id.encode(),
                    str(cluster_id).encode(),
                    str(CLUSTER_SIZE).encode(),
                    pack_u64_le(remote_block_ids),
                    pack_u64_le(local_block_ids[1]),
                ]
                s.send_multipart(frames)

            # receove and verfy the request_id
            for inst, s in enumerate(tmp_sockets):
                self._recv_agent_reply(
                    s, inst,
                    expected_request_id=request_id,
                    timeout_ms=5000,
                    max_retries=0  # REQ-per-request normally no need to retry
                )
        finally:
            for s in tmp_sockets:
                try:
                    s.close()
                except:
                    pass

        # notify prefill side after success
        if self.vllm_config.parallel_config.tensor_parallel_size == 1:
            # tp=1, send to prefill tp rank0 directly.
            if remote_request_id is not None:
                self._send_pulled_kv_req_list(remote_host_ip, [remote_request_id])
            with self._transfer_lock:
                self._recving_transfers.append(request_id)
        else:
            torch.distributed.barrier(group=get_tp_group().cpu_group)
            if get_tensor_model_parallel_rank() == 0 and remote_request_id is not None:
                self._send_pulled_kv_req_list(remote_host_ip, [remote_request_id])
            with self._transfer_lock:
                self._recving_transfers.append(request_id)

        if self.omni_cache is None or self.omni_cache.device_cache is None:
            raise RuntimeError(f"Error! omni_cache is None or device_cache is None.")
        
        st = time.time()
        self.omni_cache.synchronize_h2d(local_block_ids)
        duration = time.time() - st
        logger.warning(f"<<< Time cost of decode synchronize_h2d is {duration*1000} ms")

        logger.debug(" ***** read block, req_id:%s, local_block_ids:%s, remote_block_ids:%s",
                     request_id, local_block_ids[1], remote_block_ids)
        cost = time.time() - start
        logger.warning(" ***** read block, req_id:%s, cost:%.6f", request_id, cost)

    def _recv_agent_reply(self,
                          sock: zmq.Socket,
                          inst: int,
                          expected_request_id: str | None = None,
                          timeout_ms: int = 5000,
                          max_retries: int = 0) -> bytes:

        sock.setsockopt(zmq.RCVTIMEO, timeout_ms)
        attempts = 0
        while True:
            attempts += 1
            try:
                parts = sock.recv_multipart()
            except zmq.Again:
                raise RuntimeError(f"[ERROR] agent#{inst} timeout waiting for reply")
                # logger.warning(f"[WARN] agent#{inst} timeout, retrying...")
                # continue
            except Exception as e:
                raise RuntimeError(f"[ERROR] agent#{inst} recv failed: {e}") from e

            if not parts:
                if attempts > max_retries:
                    raise RuntimeError(f"[ERROR] agent#{inst} empty reply")
                continue

            status = parts[0]
            req_id_part: bytes | None = None
            msg = b""

            if len(parts) == 2:
                # 兼容旧格式
                msg = parts[1]
            elif len(parts) >= 3:
                # 新格式 [status, request_id, message]
                req_id_part = parts[1]
                msg = parts[2]
            else:
                if attempts > max_retries:
                    raise RuntimeError(f"[ERROR] agent#{inst} invalid reply format ({len(parts)} frames)")
                continue

            if status != b"OK":
                text = msg.decode("utf-8", "ignore")
                rid = req_id_part.decode("utf-8", "ignore") if req_id_part else "?"
                raise RuntimeError(f"[ERROR] agent#{inst} error (req_id={rid}): {text}")

            # if has request_id，verify it
            if expected_request_id is not None and req_id_part is not None:
                got = req_id_part.decode("utf-8", "ignore")
                if got != expected_request_id:
                    if attempts <= max_retries:
                        continue
                    raise RuntimeError(
                        f"[ERROR] agent#{inst} reply request_id mismatch: expect={expected_request_id}, got={got}"
                    )

            return msg

    def _send_pulled_kv_req_list(self, path: str, data: List[str]):
        if path in self.zmq_socket_map:
            socket_ = self.zmq_socket_map[path]
        else:
            socket_ = self.ctx.socket(zmq.PUSH)
            socket_.connect(path)
            self.zmq_socket_map[path] = socket_
            logger.info("create new socket path:%s", path)

        try:
            json_data = json.dumps(data)
            socket_.send_string(json_data)
            logger.info("send string %s path:%s", json_data, path)
        except Exception as e:
            logger.error("Failed to send request_ids to prefill: %s", e)

    def get_finished(self, metadata: DatadistConnectorMetadata) -> Tuple[set[str], set[str]]:
        # for decode side, done_sending is not needed
        all_done_sending: set[str] = set()
        with self._transfer_lock:
            all_done_recving = self._pop_done_transfers(self._recving_transfers)
        if len(all_done_recving) > 0:
            logger.debug("Get_finished: %s requests done recving", len(all_done_recving))

        return all_done_sending, all_done_recving

    def _pop_done_transfers(self, transfers: List[str]) -> set[str]:
        done_req_ids: set[str] = set()
        for req_id in transfers:
            done_req_ids.add(req_id)
        transfers.clear()
        return done_req_ids


def handle_exception(future):
    if future.exception():
        logger.error("Exception occurred in future: %s", future.exception())
        # Re-raise on the caller thread if someone waits on the future elsewhere
        raise future.exception()


def log_line(prefix, line):
    if not VERBOSE:
        return
    with _print_lock:
        print(f"{prefix} {line.rstrip()}", flush=True)


def pump(pipe, prefix):
    try:
        for line in iter(pipe.readline, ''):
            if not line:
                break
            log_line(prefix, line)
    except Exception as e:
        log_line(prefix, f"[logger error] {e}")


def spawn_logged_process(args, tag):
    p = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
    th_out = threading.Thread(target=pump, args=(p.stdout, f"[{tag}] OUT"))
    th_err = threading.Thread(target=pump, args=(p.stderr, f"[{tag}] ERR"))
    th_out.start()
    th_err.start()
    return p, [th_out, th_err]

def stop_logged_process(proc, threads, timeout=5.0):
    try:
        if proc.poll() is None:
            proc.terminate()
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=timeout)
    finally:
        try:
            if proc.stdout:
                proc.stdout.close()
        except Exception:
            pass
        try:
            if proc.stderr:
                proc.stderr.close()
        except Exception:
            pass
        for t in threads:
            try:
                t.join(timeout=timeout)
            except Exception:
                pass


def _wait_ports(host_ports, timeout_sec=10, interval=0.1):
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
            try:
                s.close()
            except Exception:
                pass

    while remaining and time.time() < deadline:
        ready = [hp for hp in list(remaining) if can_connect(hp[0], hp[1], interval)]
        for hp in ready:
            remaining.discard(hp)
        if remaining:
            time.sleep(interval)
    return len(remaining) == 0, remaining


def start_p_node_servers(base_port: int, servers: int, mmap_paths: List[str], dtype: str, offset_dtype: int, start_offset: int):
    procs = []
    for nid in range(servers):
        port = base_port + nid
        args = [P_NODE_BIN, "--node_id", str(nid), "--port", str(port),
                "--mmap", mmap_paths[nid], "--dtype", dtype,
                "--offset_dtype", str(offset_dtype), "--start_offset", str(start_offset)]
        log_line(f"[P{nid}]", " ".join(args))
        p, threads = spawn_logged_process(args, f"P{nid}")
        procs.append((p, threads))
        time.sleep(0.05)
    return procs


def pack_u64_le(arr: List[int]) -> bytes:
    return b''.join(struct.pack('<Q', int(x)) for x in arr)