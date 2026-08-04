# SPDX-License-Identifier: Apache-2.0
"""HIXLConnector: direct HIXL LLM-DataDist P2P KV connector (no Mooncake).

Forked from MooncakeConnectorV1 with the byte-addressed Mooncake transfer
(repeated register_memory + batch_transfer_sync_read) replaced by block-indexed
HIXL transfer (register_blocks_cache + pull_blocks). Control plane (handshake
/ metadata / done-notify) still uses ZMQ, identical to Mooncake.
"""
import contextlib
import copy
import hashlib
import logging
import math
import queue
import random
import struct
import threading
import time
from collections import OrderedDict, defaultdict, deque
from collections.abc import Iterator, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypedDict

import msgspec
import numpy as np
import torch
import torch_npu
import zmq
from vllm import envs
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.utils import BlockIds
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorHandshakeMetadata,
    KVConnectorMetadata,
    KVConnectorRole,
    SupportsHMA,
)
from vllm.distributed import get_pcp_group
from vllm.distributed.parallel_state import (
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.distributed.utils import get_pp_indices
from vllm.logger import logger
from vllm.utils.network_utils import get_ip, make_zmq_path, make_zmq_socket
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.request import RequestStatus

from vllm_ascend.ascend_config import get_ascend_config, init_ascend_config
from vllm_ascend.distributed.kv_transfer.utils.hixl_datadist import (
    get_datadist,
    shutdown_datadist,
)
from vllm_ascend.distributed.utils import (
    get_decode_context_model_parallel_rank,
    get_decode_context_model_parallel_world_size,
)

# isort: off
if TYPE_CHECKING:
    from vllm.v1.attention.backend import AttentionMetadata  # type: ignore
    from vllm.forward_context import ForwardContext
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.request import Request
# isort: on

GET_META_MSG = b"get_meta_msg"
DONE_RECVING_MSG = b"done_recving_msg"

# A busy peer can otherwise keep a global executor worker forever when the
# number of peers is larger than max_workers. Yield after a small FIFO batch so
# other peers already waiting in the global executor queue can make progress.
MAX_REQUESTS_PER_PEER_HANDLER = 5


# ---------------------------------------------------------------------------
# torch dtype -> llm_datadist DataType
# ---------------------------------------------------------------------------
def _torch_dtype_to_llm_dtype(dtype: torch.dtype):
    from llm_datadist import DataType

    if dtype == torch.bfloat16:
        return DataType.DT_BF16
    if dtype == torch.float16:
        return DataType.DT_FLOAT16
    if dtype == torch.float32:
        return DataType.DT_FLOAT
    if dtype == torch.int8:
        return DataType.DT_INT8
    raise ValueError(f"Unsupported kv cache dtype for HIXLConnector: {dtype}")


# ---------------------------------------------------------------------------
# Prefill-PP layer segment helper (forked from Mooncake :3752).
# ---------------------------------------------------------------------------
def get_prefill_pp_indices(
    num_hidden_layers: int, pp_rank: int, pp_size: int, partition_list_str: str | None = None
) -> tuple[int, int]:
    if partition_list_str is None:
        return get_pp_indices(num_hidden_layers, pp_rank, pp_size)
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


# ---------------------------------------------------------------------------
# Metadata & dataclasses (engine-agnostic, forked from Mooncake)
# ---------------------------------------------------------------------------
class RemotePortInfo(TypedDict):
    num: int
    host: str


class HixlAgentMetadata(msgspec.Struct, omit_defaults=True, dict=True):
    """Replaces MooncakeAgentMetadata.

    Drops byte-addressed fields (te_rpc_port / kv_caches_base_addr /
    block_lens / block_strides); HIXL routes by cluster_id and addresses by
    block index, so no raw byte arithmetic is carried.
    """

    engine_id: str
    cluster_id: int  # local LLMDataDist cluster_id (D uses it as pull key)
    listen_ip: str  # llm listen ip (D->P link)
    listen_port: int  # llm listen port
    model_id: int = 0
    num_tensors_per_group: list[int] = []  # Cache.num_tensors per group (=layers*2)
    # Retained for block expansion / future reformat:
    kv_group2layeridx: dict[int, tuple[dict[str, Any], list[int]]] = {}
    block_size: int = 0
    num_blocks: int = 0
    block_size_scale: list[list[int]] = []
    local_ip: str = ""
    handshake_port: int = 0


@dataclass
class MambaCacheBundle:
    """conv/ssm registered as two independent Caches (CacheDesc's single-shape
    constraint can't hold heterogeneous state tensors: conv 2D vs ssm 3D).

    conv+ssm are as_strided views of the same raw_tensor (model_runner_v1.py),
    so they share one block table -> one block id. Pull reuses the same
    src/dst block ids for both sub-caches with tensor_num_per_layer=1 (each
    layer contributes exactly 1 conv / 1 ssm tensor). Forks Mooncake's per-
    layer (conv_addr, ssm_addr) byte geometry under block addressing.
    """
    conv: Any
    ssm: Any
    num_layers: int = 0  # == len(conv addrs) == len(ssm addrs)
    # Per-sub-cache model_id for BlocksCacheKey. conv and ssm are independent
    # registered caches and must NOT share a BlocksCacheKey: native
    # AddCacheIndices overwrites cache_key_to_id_[key] = cache_id (last-wins),
    # so a shared key silently re-points pulls to whichever cache registered
    # last (see HIXL pull_blocks LLM_FAILED root cause).
    conv_model_id: int = 0
    ssm_model_id: int = 0

    @property
    def subcaches(self) -> tuple:
        return (self.conv, self.ssm)


@dataclass
class ReqMeta:
    local_block_ids: BlockIds
    num_external_tokens: int
    num_computed_tokens: int
    remote_block_ids: BlockIds

    remote_host: str
    remote_port: int
    remote_engine_id: str
    remote_request_id: str
    remote_ptp_size: int
    num_prompt_blocks: int
    remote_block_size: int
    remote_pcp_size: int = 1
    remote_dcp_size: int = 1
    remote_multi_nodes_meta_mapping: dict[str, Any] | None = None


@dataclass(frozen=True)
class GroupPull:
    group_id: int
    remote_tp_offset: int
    num_group_pulls: int
    prefill_pp_rank: int = 0
    is_group_transfer_end: bool = False


@dataclass
class GroupTransferInfo:
    """Per-group transfer metadata (forked from Mooncake, byte-address fields
    dropped). State groups (Mamba) are not context-block aligned with attention
    KV, so they are kept intact during MTP extra-block clipping."""

    tokens_per_block: int = 0
    is_state_group: bool = False


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
    """Tracks finished / delayed-free requests. Forked unchanged from Mooncake."""

    def __init__(self):
        super().__init__()
        self.done_task_lock = threading.Lock()
        self.finished_requests: set[str] = set()
        # Only used in prefill node. Tracks requests whose kv blocks freeing is
        # intentionally delayed. Force-freed after VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT.
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
                logger.warning(
                    "HIXLConnector finish req not in reqs to process. request_id=%s. ",
                    request_id,
                )

    def get_and_clear_finished_requests(self) -> set[str]:
        with self.done_task_lock:
            finished_requests = self.finished_requests
            self.finished_requests = set()
            expired_requests = self._retrieve_expired_requests()
        finished_requests.update(expired_requests)
        return finished_requests

    def add_delayed_request(self, request_id: str, delay_start_time: float):
        with self.done_task_lock:
            if request_id in self.reqs_to_process:
                self.delayed_free_requests[request_id] = delay_start_time

    def _retrieve_expired_requests(self):
        expired_requests: set[str] = set()
        current_time = time.time()
        while self.delayed_free_requests:
            request_id = next(iter(self.delayed_free_requests))
            delay_start_time = self.delayed_free_requests[request_id]
            if current_time - delay_start_time > envs.VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT:
                self.delayed_free_requests.popitem(last=False)
                self.reqs_to_process.discard(request_id)
                expired_requests.add(request_id)
                logger.error(
                    "Force freed expired request: %s. Reason: exceeded timeout (%s seconds).",
                    request_id,
                    envs.VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT,
                )
            else:
                break
        return expired_requests


# ---------------------------------------------------------------------------
# KVCacheSendingThread (P side, only answers ZMQ handshake). Forked; only the
# metadata type changes (MooncakeAgentMetadata -> HixlAgentMetadata).
# ---------------------------------------------------------------------------
class KVCacheSendingThread(threading.Thread):
    def __init__(
        self,
        vllm_config: VllmConfig,
        tp_rank: int,
        prefill_tp_size: int,
        local_engine_id: str,
        side_channel_host: str,
        side_channel_port: int,
        metadata: HixlAgentMetadata,
        ready_event: threading.Event,
        kv_caches: dict[str, Any],
        pcp_rank: int,
        pcp_size: int = 1,
    ):
        super().__init__(daemon=True, name="HIXLKCacheSendingThread")
        self.tp_rank = tp_rank
        self.prefill_tp_size = prefill_tp_size
        self.pp_rank = get_pp_group().rank_in_group
        self.pcp_size = pcp_size
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
        return self.task_tracker.get_and_clear_finished_requests()

    def add_not_transfer_request(self, request_id: str):
        self.task_tracker.add_not_transfer_request(request_id)

    def add_delayed_request(self, request_id: str, delay_start_time: float):
        return self.task_tracker.add_delayed_request(request_id, delay_start_time)

    def run(self):
        try:
            device_index = (self.pp_rank * self.pcp_size + self.pcp_rank) * self.tp_size + self.tp_rank
            handshake_port = self.side_channel_port + device_index
            path = make_zmq_path("tcp", self.side_channel_host, handshake_port)
            logger.info(
                "HIXL KVCacheSendingThread listening on %s. tp_rank=%d pp_rank=%d",
                path, self.tp_rank, self.pp_rank,
            )
            with zmq_ctx(zmq.ROUTER, path) as sock:  # type: ignore
                self.ready_event.set()
                self.run_busy_loop(sock)
        except Exception as e:
            logger.exception(
                "HIXL KVCacheSendingThread exception. tp_rank=%d path=%s. Error: %s",
                self.tp_rank, path, e,
            )

    def run_busy_loop(self, sock: zmq.Socket):  # type: ignore
        encoder = msgspec.msgpack.Encoder()
        encoded_data = encoder.encode(self.metadata)
        decoder = msgspec.msgpack.Decoder(type=tuple)
        ack_poller = zmq.Poller()
        ack_poller.register(sock, zmq.POLLOUT)
        while True:
            try:
                frames = sock.recv_multipart()
                if len(frames) < 2:
                    logger.error("Invalid message in KVCacheSendingThread: %d frames", len(frames))
                    continue
                identity = frames[0]
                payload = [f for f in frames[1:] if f != b""]
                if len(payload) != 1:
                    logger.error("Invalid payload in KVCacheSendingThread: %d frames", len(payload))
                    continue
                msg = decoder.decode(payload[0])
                if msg[0] == GET_META_MSG:
                    sock.send_multipart((identity, b"", encoded_data))
                elif msg[0] == DONE_RECVING_MSG:
                    request_id = msg[1]
                    remote_port_send_num = msg[2]
                    if remote_port_send_num:
                        if request_id not in self.port_send_num:
                            self.port_send_num[request_id] = 0
                        self.port_send_num[request_id] += 1
                        device_index = (self.pp_rank * self.pcp_size + self.pcp_rank) * self.tp_size + self.tp_rank
                        handshake_port = self.side_channel_port + device_index
                        if self.port_send_num[request_id] >= remote_port_send_num[handshake_port]["num"]:
                            self.task_tracker.update_done_task_count(request_id)
                            del self.port_send_num[request_id]
                    else:
                        self.task_tracker.update_done_task_count(request_id)
                    while True:
                        if not dict(ack_poller.poll(timeout=1000)):
                            continue
                        try:
                            sock.send_multipart((identity, b"", b"ACK"), flags=zmq.NOBLOCK)  # type: ignore
                            break
                        except zmq.Again:  # type: ignore
                            continue
                else:
                    logger.error("Unexpected message type in KVCacheSendingThread: %s", msg[0])
            except Exception as e:
                logger.error("KVCacheSendingThread handler exception: %s", e)


# ---------------------------------------------------------------------------
# KVCacheRecvingThread (D side). _get_remote_metadata and
# _transfer_kv_cache_all_groups are rewritten for HIXL; the request-queue /
# peer-fairness / done-signal / socket-pool skeleton is forked from Mooncake.
# ---------------------------------------------------------------------------
class KVCacheRecvingThread(threading.Thread):
    def __init__(
        self,
        tp_rank: int,
        tp_size: int,
        hixl,  # HixlDataDist
        model_id: int,
        local_engine_id: str,
        local_handshake_port: int,
        side_channel_port: int,
        vllm_config: VllmConfig,
        kv_caches: dict[str, Any],
        kv_group2layeridx: dict[int, tuple[dict[str, Any], list[int]]],
        group_caches: dict[int, Any],  # kv_cache_group_id -> registered Cache
        group_model_ids: dict[int, int] | None = None,  # kv_cache_group_id -> model_id
        block_size_scale: list[list[int]] | None = None,
        ready_event: threading.Event | None = None,
        staging_tensors: dict[int, dict[str, list[torch.Tensor]]] | None = None,
        staging_caches: dict[int, Any] | None = None,
        is_hma_required: bool = False,
        prefill_tp_size: int = 1,
        num_key_value_heads: int = 0,
        num_blocks: int = 0,
        cluster_id: int = 0,
    ):
        super().__init__(daemon=True, name="HIXLKCacheRecvingThread")
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self.local_engine_id = local_engine_id
        self.local_handshake_port = local_handshake_port
        self.side_channel_port = side_channel_port
        self.hixl = hixl
        self.cache_manager = hixl.cache_manager
        self.model_id = model_id
        if ready_event is None:
            ready_event = threading.Event()
        self.ready_event = ready_event

        self.kv_caches = kv_caches
        self.kv_group2layeridx = kv_group2layeridx
        self.group_caches = group_caches  # kv_cache_group_id -> Cache
        self.group_model_ids = group_model_ids or {}  # kv_cache_group_id -> model_id
        self.block_size_scale = block_size_scale or []
        self.staging_tensors = staging_tensors or {}
        self.staging_caches = staging_caches or {}
        self.is_hma_required = is_hma_required
        self._prefill_tp_size = prefill_tp_size
        self.num_key_value_heads = num_key_value_heads
        self.num_blocks = num_blocks
        self.cluster_id = cluster_id
        # Reformat metadata keyed by request_id then shard index (No-CP: shard 0).
        # Populated by the last TP-offset pull task for each group; applied once
        # all pull tasks for the request finish (all_tasks_done).
        self.pending_reformat: defaultdict[str, dict[int, list[tuple[int, list[list[int]], int, list[int]]]]] = (
            defaultdict(dict)
        )
        self.pending_reformat_lock = threading.Lock()

        # remote_cluster_id[engine_id][handshake_port] = P cluster_id
        self.remote_cluster_id: dict[str, dict[int, int]] = SizedDict()
        self.remote_kv_group2layeridx: dict[str, dict[int, dict[int, tuple[dict[str, Any], list[int]]]]] = SizedDict()
        self.remote_metadata_lock = threading.Lock()

        self.request_queue: queue.Queue[Any] = queue.Queue()
        first_kv_cache = next(iter(self.kv_caches.values()), None)
        if first_kv_cache is None:
            self.executor = ThreadPoolExecutor(max_workers=32)
        else:
            # NPU device selection is thread-local. Executor workers do not
            # inherit the device selected by the model worker thread and would
            # otherwise use device 0 on their first NPU operation (pull_blocks /
            # reformat), corrupting multi-card TP>1 transfers.
            kv_cache_device = first_kv_cache[0].device
            self.executor = ThreadPoolExecutor(
                max_workers=32,
                initializer=torch.npu.set_device,
                initargs=(kv_cache_device,),
            )
        self.peer_request_queues: defaultdict[tuple[str, int], deque[dict[str, Any]]] = defaultdict(deque)
        self.active_peer_request_handlers: set[tuple[str, int]] = set()
        self.peer_request_queues_lock = threading.Lock()
        self.request_task_counts: defaultdict[str, int] = defaultdict(int)
        self.finished_request_markers: set[str] = set()
        self.request_task_counts_lock = threading.Lock()

        self.task_tracker = KVCacheTaskTracker()

        self.encoder = msgspec.msgpack.Encoder()
        self.decoder = msgspec.msgpack.Decoder(HixlAgentMetadata)
        self.remote_sockets_lock = threading.Lock()
        self.remote_sockets: dict[str, deque[zmq.Socket]] = defaultdict(deque)  # type: ignore
        self.timeout = 1.0
        self._group_kv_cache: dict[tuple, dict[str, Any]] = {}

        assert vllm_config is not None
        self.vllm_config: VllmConfig = vllm_config
        self.block_size = self.vllm_config.cache_config.block_size
        # G3: NZ layout (MLA D-node only, AscendConfig.enable_kv_nz). D cache is
        # physically NZ-ordered; pull_blocks writes ND into it, so a ND->NZ
        # reformat is needed after pull (fork Mooncake reformat_kv_cache:1244-1317).
        self.enable_kv_nz = bool(getattr(get_ascend_config(), "enable_kv_nz", False))

        self.proc_not_transfer_request: dict[str, bool] = {}
        self.proc_not_transfer_request_lock = threading.Lock()
        self.failed_recv_requests: set[str] = set()
        self.invalid_block_ids: set[int] = set()
        self.failed_recv_requests_lock = threading.Lock()

    def add_request(
        self,
        request_id: str,
        remote_request_id: str,
        local_block_ids: BlockIds,
        remote_block_ids: BlockIds,
        group_pulls: list[GroupPull],
        remote_engine_id: str,
        remote_host: str,
        remote_handshake_port: int,
        remote_port_send_num: dict[int, RemotePortInfo] | None = None,
        num_computed_tokens: int = 0,
        all_task_done: bool = False,
        shard_idx: int = 0,
    ):
        if remote_port_send_num is None:
            remote_port_send_num = {}
        trans_info = {
            "request_id": request_id,
            "local_block_ids": local_block_ids,
            "remote_block_ids": remote_block_ids,
            "group_pulls": group_pulls,
            "remote_engine_id": remote_engine_id,
            "remote_request_id": remote_request_id,
            "remote_host": remote_host,
            "remote_handshake_port": remote_handshake_port,
            "num_computed_tokens": num_computed_tokens,
            "remote_port_send_num": remote_port_send_num,
            "all_task_done": all_task_done,
            "shard_idx": shard_idx,
        }
        self.request_queue.put(trans_info)

    def get_and_clear_finished_requests(self) -> set[str]:
        return self.task_tracker.get_and_clear_finished_requests()

    def get_and_clear_invalid_block_ids(self) -> set[int]:
        with self.failed_recv_requests_lock:
            invalid_block_ids = self.invalid_block_ids
            self.invalid_block_ids = set()
        return invalid_block_ids

    def _is_failed_recv_request(self, request_id: str) -> bool:
        with self.failed_recv_requests_lock:
            return request_id in self.failed_recv_requests

    def _mark_failed_recv_request(self, request_id: str, local_block_ids: BlockIds) -> None:
        with self.failed_recv_requests_lock:
            self.failed_recv_requests.add(request_id)
            self.invalid_block_ids.update(local_block_ids[0])

    def _clear_failed_recv_request(self, request_id: str) -> None:
        with self.failed_recv_requests_lock:
            self.failed_recv_requests.discard(request_id)

    def run(self):
        self.ready_event.set()
        while True:
            try:
                request_data = self.request_queue.get()
                if request_data is None:
                    self.request_queue.task_done()
                    continue
                self._submit_request(request_data)
            except Exception as e:
                logger.error("Error in HIXL KVCacheRecvingThread. error=%s. ", e)

    def _submit_request(self, request_data: dict[str, Any]) -> None:
        peer_key = (request_data["remote_host"], request_data["remote_handshake_port"])
        self._mark_request_task_submitted(request_data)
        should_start_worker = False
        with self.peer_request_queues_lock:
            self.peer_request_queues[peer_key].append(request_data)
            if peer_key not in self.active_peer_request_handlers:
                self.active_peer_request_handlers.add(peer_key)
                should_start_worker = True
        if should_start_worker:
            self.executor.submit(self._handle_peer_requests, peer_key)

    def _handle_peer_requests(self, peer_key: tuple[str, int]) -> None:
        with self.peer_request_queues_lock:
            peer_queue = self.peer_request_queues.get(peer_key)
            if not peer_queue:
                self.peer_request_queues.pop(peer_key, None)
                self.active_peer_request_handlers.discard(peer_key)
                return
            batch: list[dict[str, Any]] = []
            while peer_queue and len(batch) < MAX_REQUESTS_PER_PEER_HANDLER:
                batch.append(peer_queue.popleft())
        for req_meta in batch:
            try:
                self._handle_request(req_meta)
            except Exception:
                logger.exception("Error handling HIXL KV transfer for peer %s:%d.", peer_key[0], peer_key[1])
        should_resubmit = False
        with self.peer_request_queues_lock:
            peer_queue = self.peer_request_queues.get(peer_key)
            if peer_queue:
                should_resubmit = True
            else:
                self.peer_request_queues.pop(peer_key, None)
                self.active_peer_request_handlers.discard(peer_key)
        if should_resubmit:
            self.executor.submit(self._handle_peer_requests, peer_key)

    def _mark_request_task_submitted(self, req_meta: dict[str, Any]) -> None:
        request_id = req_meta["request_id"]
        with self.request_task_counts_lock:
            self.request_task_counts[request_id] += 1
            if req_meta["all_task_done"]:
                self.finished_request_markers.add(request_id)

    def _mark_request_task_done(self, request_id: str, all_task_done: bool) -> bool:
        with self.request_task_counts_lock:
            pending_count = self.request_task_counts.get(request_id)
            if pending_count is None:
                return all_task_done
            pending_count -= 1
            if pending_count > 0:
                self.request_task_counts[request_id] = pending_count
                return False
            self.request_task_counts.pop(request_id, None)
            has_finished_marker = request_id in self.finished_request_markers
            self.finished_request_markers.discard(request_id)
            return has_finished_marker

    def _handle_request(self, req_meta: dict[str, Any]):
        request_id = req_meta["request_id"]
        remote_request_id = req_meta["remote_request_id"]
        remote_host = req_meta["remote_host"]
        remote_handshake_port = req_meta["remote_handshake_port"]
        remote_port_send_num = req_meta["remote_port_send_num"]
        all_task_done = req_meta["all_task_done"]
        transfer_failed = self._is_failed_recv_request(request_id)
        try:
            if transfer_failed:
                self._mark_failed_recv_request(request_id, req_meta["local_block_ids"])
                logger.warning("Skipping HIXL KV transfer for request. remote=%s. ", remote_request_id)
            else:
                try:
                    self._transfer_kv_cache_all_groups(req_meta)
                except Exception as e:
                    transfer_failed = True
                    self._mark_failed_recv_request(request_id, req_meta["local_block_ids"])
                    logger.exception("Failed HIXL KV transfer for request %s: %s", remote_request_id, e)
        finally:
            all_tasks_done = self._mark_request_task_done(request_id, all_task_done)
            if all_tasks_done:
                # Reformat must run BEFORE update_done_task_count so that
                # get_finished (gated by task_tracker) only surfaces the
                # request after staging -> D real cache is fully assembled.
                if transfer_failed or self._is_failed_recv_request(request_id):
                    with self.pending_reformat_lock:
                        self.pending_reformat.pop(request_id, None)
                else:
                    try:
                        self._reformat_pending_kv_caches(request_id)
                    except Exception as e:
                        transfer_failed = True
                        self._mark_failed_recv_request(request_id, req_meta["local_block_ids"])
                        with self.pending_reformat_lock:
                            self.pending_reformat.pop(request_id, None)
                        logger.exception(
                            "Failed to reformat HIXL KV cache after all pulls for request %s: %s",
                            remote_request_id, e,
                        )
                self.task_tracker.update_done_task_count(request_id)
                with self.proc_not_transfer_request_lock:
                    self.proc_not_transfer_request.pop(remote_request_id, None)
                self._clear_failed_recv_request(request_id)
            self.request_queue.task_done()
            # Bug 4 fix: free P ports with num==0 (mapped but not pulled by
            # this D rank) so P-side delayed_free doesn't wait for timeout
            # (fork Mooncake :751).
            self._send_done_signal_to_free_remote_port(remote_request_id, remote_port_send_num)
            # Always send the done signal to the remote host to ensure proper
            # resource cleanup. Failing to do so may cause a memory leak.
            self._send_done_recv_signal(remote_request_id, remote_host, remote_handshake_port, remote_port_send_num)

    def _get_remote_metadata(self, remote_host: str, remote_handshake_port: int) -> None:
        """Fetch HixlAgentMetadata over ZMQ and link the remote P cluster.

        Replaces Mooncake's byte-address bookkeeping: instead of caching
        kv_caches_base_addr / te_rpc_port, we cache the remote cluster_id and
        immediately ensure_linked to it (D3)."""
        sock: zmq.Socket | None = None  # type: ignore
        try:
            sock = self._get_remote_socket(remote_host, remote_handshake_port)
            ensure_zmq_send(sock, self.encoder.encode((GET_META_MSG, "")), f"{remote_host}:{remote_handshake_port}")
            metadata_bytes = ensure_zmq_recv(sock, f"{remote_host}:{remote_handshake_port}")
            agent_meta = self.decoder.decode(metadata_bytes)
            engine_id = agent_meta.engine_id
            assert engine_id != self.local_engine_id, (
                f"Conflict engine id {engine_id} with local {self.local_engine_id}."
            )
            if agent_meta.kv_group2layeridx != self.kv_group2layeridx:
                logger.warning(
                    "Remote kv_group2layeridx inconsistent. remote=%s local=%s",
                    agent_meta.kv_group2layeridx, self.kv_group2layeridx,
                )
            # D3: link the remote P cluster before any pull_blocks to it.
            self.hixl.ensure_linked(
                remote_cluster_id=agent_meta.cluster_id,
                remote_ip=agent_meta.listen_ip,
                remote_port=agent_meta.listen_port,
            )
            with self.remote_metadata_lock:
                self.remote_kv_group2layeridx[engine_id][remote_handshake_port] = agent_meta.kv_group2layeridx
                self.remote_cluster_id[engine_id][remote_handshake_port] = agent_meta.cluster_id
        except Exception:
            if isinstance(sock, zmq.Socket):  # type: ignore
                sock.close()
                sock = None
            raise
        finally:
            if sock is not None:
                self._return_remote_socket(sock, remote_host, remote_handshake_port)

    def _transfer_kv_cache_all_groups(self, req_meta: dict[str, Any]):
        """D-pull KV via HIXL pull_blocks (block-index addressed)."""
        remote_request_id = req_meta["remote_request_id"]
        local_block_ids: BlockIds = req_meta["local_block_ids"]
        remote_block_ids: BlockIds = req_meta["remote_block_ids"]
        group_pulls: list[GroupPull] = req_meta["group_pulls"]
        remote_engine_id = req_meta["remote_engine_id"]
        remote_host = req_meta["remote_host"]
        remote_handshake_port = req_meta["remote_handshake_port"]
        request_id = req_meta["request_id"]

        num_local_blocks = sum(len(group_block_ids) for group_block_ids in local_block_ids)
        if num_local_blocks == 0:
            return

        with self.remote_metadata_lock:
            has_remote_metadata = (
                remote_engine_id in self.remote_cluster_id
                and remote_handshake_port in self.remote_cluster_id[remote_engine_id]
            )
        if not has_remote_metadata:
            self._get_remote_metadata(remote_host, remote_handshake_port)

        from llm_datadist import BlocksCacheKey, LLMException

        ready_attention_group_reformat_block_ids: list[tuple[tuple, bool]] = []

        with self.remote_metadata_lock:
            remote_cluster_id = self.remote_cluster_id[remote_engine_id][remote_handshake_port]
        for group_pull in group_pulls:
            group_idx = group_pull.group_id
            group_spec, layer_indices = self.kv_group2layeridx[group_idx]
            kv_cache_group_id = group_spec.get("kv_cache_group_id", group_idx)
            tp_n = group_pull.num_group_pulls
            tp_offset = group_pull.remote_tp_offset
            src_blocks = list(remote_block_ids[kv_cache_group_id])
            dst_logical = list(local_block_ids[kv_cache_group_id])
            if not src_blocks or not dst_logical:
                continue

            is_state_group = group_spec.get("kv_cache_spec_type") == "MambaSpec"

            if is_state_group:
                # G2: mamba conv/ssm registered as two sub-caches (MambaCacheBundle).
                # conv+ssm share one block table -> reuse one src/dst block id for
                # both, with tensor_num_per_layer=1 (1 conv / 1 ssm per layer).
                # Forks Mooncake per-layer (conv_addr, ssm_addr) byte geometry
                # (MC:1142-1154) under block addressing.
                bundle = self.group_caches.get(kv_cache_group_id)
                if not isinstance(bundle, MambaCacheBundle):
                    logger.error(
                        "HIXL mamba group %s has no MambaCacheBundle; skip.",
                        kv_cache_group_id,
                    )
                    continue
                assert tp_n == 1, (
                    "mamba TP>1 head-shard via block API unsupported; "
                    "require prefill_tp==decode_tp (G2 R3)."
                )
                grouped_remote, grouped_local = group_concurrent_contiguous(
                    src_blocks, dst_logical,
                )
                # Each sub-cache (conv/ssm) has its own BlocksCacheKey on P;
                # pull with the matching model_id so the src resolves to the
                # right cache instead of whichever registered last.
                for sub_cache, sub_model_id in zip(
                    bundle.subcaches, (bundle.conv_model_id, bundle.ssm_model_id)
                ):
                    num_layers = sub_cache.cache_desc.num_tensors
                    for cr, cl in zip(grouped_remote, grouped_local):
                        self.cache_manager.pull_blocks(
                            BlocksCacheKey(remote_cluster_id, sub_model_id),
                            sub_cache,
                            src_blocks=cr,
                            dst_blocks=cl,
                            src_layer_range=range(num_layers),
                            dst_layer_range=range(num_layers),
                            tensor_num_per_layer=1,
                        )
                continue

            if tp_n > 1:
                staging_cache = self.staging_caches.get(kv_cache_group_id)
                if staging_cache is None:
                    logger.error(
                        "HIXL staging cache missing for group %s (tp_n=%s); skip.",
                        kv_cache_group_id, tp_n,
                    )
                    continue
                dst_cache = staging_cache
                # Each P-rank shard lands in staging block b*tp_n + tp_offset.
                dst_blocks = [b * tp_n + tp_offset for b in dst_logical]
                # staging blocks are non-contiguous across the shard set, so
                # pull one block at a time (mirrors Mooncake tp>1 path).
                grouped_remote = [[b] for b in src_blocks]
                grouped_local = [[b] for b in dst_blocks]
                reformat_local = [[b] for b in dst_logical]  # D real block ids
            else:
                dst_cache = self.group_caches.get(kv_cache_group_id)
                if dst_cache is None:
                    logger.error("No registered HIXL cache for group %s; skip.", kv_cache_group_id)
                    continue
                dst_blocks = dst_logical
                grouped_remote, grouped_local = group_concurrent_contiguous(src_blocks, dst_blocks)
                reformat_local = grouped_local

            num_layers = dst_cache.cache_desc.num_tensors // 2
            src_layer_range = range(num_layers)
            dst_layer_range = range(num_layers)
            for chunk_remote, chunk_local in zip(grouped_remote, grouped_local):
                try:
                    self.cache_manager.pull_blocks(
                        BlocksCacheKey(
                            remote_cluster_id,
                            self.group_model_ids[kv_cache_group_id],
                        ),
                        dst_cache,
                        src_blocks=chunk_remote,
                        dst_blocks=chunk_local,
                        src_layer_range=src_layer_range,
                        dst_layer_range=dst_layer_range,
                    )
                except LLMException as e:
                    logger.error(
                        "HIXL pull_blocks failed for request %s group %s: %s",
                        remote_request_id, group_idx, e,
                    )
                    raise
            # G3: NZ reformat. enable_kv_nz (MLA D-node only) stores D cache in
            # NZ physical layout; pull_blocks just wrote ND into it, so reformat
            # ND -> NZ via npu_paged_cache_load + npu_scatter_pa_kv_cache (fork
            # Mooncake reformat_kv_cache:1244-1317). TP=1 only: MLA NZ has
            # num_kv_heads==1 -> tp_n==1; TP>1+NZ needs a staging NZ scatter
            # branch (unsupported). Mamba groups (is_state_group) already
            # `continue`d above and are not NZ-ordered.
            if self.enable_kv_nz and tp_n == 1 and not is_state_group:
                group_kv = self._get_group_kv_caches(group_idx, layer_indices)
                self._reformat_kv_cache_nz(group_kv, dst_logical)

            if tp_n > 1 and not is_state_group:
                ready_attention_group_reformat_block_ids.append(
                    (
                        (group_idx, reformat_local, tp_n, layer_indices),
                        group_pull.is_group_transfer_end,
                    )
                )

        # Stash reformat metadata for the groups whose last shard just landed.
        ready = [
            reformat_group
            for reformat_group, is_end in ready_attention_group_reformat_block_ids
            if is_end
        ]
        if ready:
            # Bug 3 fix: stash per shard_idx (CP shards reformat independently),
            # not hardcoded 0 (which overwrote earlier shards' reformat meta).
            self._stash_pending_reformat(request_id, req_meta["shard_idx"], ready)

    def _get_group_kv_caches(self, group_idx: int, layer_indices: list[int] | None = None) -> dict[str, Any]:
        if layer_indices is None:
            _, layer_indices = self.kv_group2layeridx[group_idx]
        cache_key = (group_idx, tuple(layer_indices))
        cached = self._group_kv_cache.get(cache_key)
        if cached is not None:
            return cached
        layer_index_set = set(layer_indices)
        num_attn_module = 2 if self.vllm_config.model_config.hf_text_config.model_type == "longcat_flash" else 1
        from vllm.v1.worker.utils import extract_layer_index

        def layer_in_group(layer_name: str) -> bool:
            if "mtp" in layer_name:
                return any(layer_idx >= self.num_layers for layer_idx in layer_index_set)
            return extract_layer_index(layer_name, num_attn_module) in layer_index_set

        result = {
            layer_name: layer_cache for layer_name, layer_cache in self.kv_caches.items() if layer_in_group(layer_name)
        }
        self._group_kv_cache[cache_key] = result
        return result

    def _stash_pending_reformat(
        self,
        request_id: str,
        shard_idx: int,
        ready_attention_group_reformat_block_ids: list[tuple[int, list[list[int]], int, list[int]]],
    ) -> None:
        with self.pending_reformat_lock:
            self.pending_reformat[request_id][shard_idx] = ready_attention_group_reformat_block_ids

    def _reformat_pending_kv_caches(self, request_id: str) -> None:
        with self.pending_reformat_lock:
            shard_reformats = self.pending_reformat.pop(request_id, {})
        for shard_idx in sorted(shard_reformats):
            self._apply_kv_cache_reformat(shard_reformats[shard_idx])

    def _apply_kv_cache_reformat(
        self,
        ready_attention_group_reformat_block_ids: list[tuple[int, list[list[int]], int, list[int]]],
    ) -> None:
        """Transpose staging -> D real cache for groups with num_group_pulls>1.

        HIXL adaptation of Mooncake's in-place reformat_kv_cache_hybrid_linear_torch:
        the source is the staging Cache backing tensor (not the D cache itself,
        since pull_blocks writes whole staging blocks, not split sub-ranges).
        """
        if not ready_attention_group_reformat_block_ids:
            return
        gqa_reformat_groups = [
            (group_idx, grouped_local_block_ids, num_group_pulls, layer_indices)
            for (group_idx, grouped_local_block_ids, num_group_pulls, layer_indices) in ready_attention_group_reformat_block_ids
            if num_group_pulls > 1
        ]
        for group_idx, grouped_local_block_ids, num_group_pulls, layer_indices in gqa_reformat_groups:
            group_spec, _ = self.kv_group2layeridx[group_idx]
            kv_cache_group_id = group_spec.get("kv_cache_group_id", group_idx)
            staging_layers = self.staging_tensors.get(kv_cache_group_id)
            if not staging_layers:
                logger.warning("HIXL reformat: no staging tensors for group %s; skip.", kv_cache_group_id)
                continue
            group_kv = self._get_group_kv_caches(group_idx, layer_indices)
            self._reformat_staging_to_local(
                staging_layers, group_kv, grouped_local_block_ids, num_group_pulls
            )

    @torch.no_grad()
    def _reformat_staging_to_local(
        self,
        staging_layers: dict[str, list[torch.Tensor]],
        group_kv: dict[str, Any],
        grouped_local_block_ids: list[list[int]],
        tp_n: int,
    ) -> None:
        flat_local = [b for sub in grouped_local_block_ids for b in sub]
        if not flat_local or tp_n <= 1:
            return
        # staging block b*tp_n + i holds the i-th P-rank head shard of D block b.
        staging_block_ids = [b * tp_n + i for b in flat_local for i in range(tp_n)]
        num_blocks = len(flat_local)
        first_staging = next(iter(staging_layers.values()))[0]
        device = first_staging.device
        block_ids_tensor = torch.tensor(flat_local, dtype=torch.long, device=device)
        staging_ids_tensor = torch.tensor(staging_block_ids, dtype=torch.long, device=device)
        head_per_split = int(first_staging.shape[-2])
        dim = int(first_staging.shape[-1])
        num_d_heads = tp_n * head_per_split
        block_size = self.block_size

        def _transpose(staging: torch.Tensor, dst: torch.Tensor) -> None:
            # staging [N*tp_n, block_size, head_per_split, dim]
            #   -> view [N, tp_n, block_size, head_per_split, dim]
            #   -> transpose(1,2) -> [N, block_size, tp_n, head_per_split, dim]
            #   -> reshape [N, block_size, num_d_heads, dim]  (D real cache layout)
            #   -> index_copy_ into D real cache at flat_local.
            selected = staging.index_select(0, staging_ids_tensor)
            transposed = (
                selected.reshape(num_blocks, tp_n, block_size, head_per_split, dim)
                .transpose(1, 2)
                .contiguous()
                .reshape(num_blocks, block_size, num_d_heads, dim)
            )
            dst.index_copy_(0, block_ids_tensor, transposed)

        for layer_name, d_cache in group_kv.items():
            k_s, v_s = staging_layers[layer_name]
            if isinstance(d_cache, (list, tuple)):
                k_d, v_d = d_cache[0], d_cache[1]
            else:
                k_d = v_d = d_cache
            _transpose(k_s, k_d)
            _transpose(v_s, v_d)

    def _reformat_kv_cache_nz(
        self,
        group_kv: dict[str, Any],
        block_ids: list[int],
    ) -> None:
        """G3: ND -> NZ reformat of D real cache after pull (fork Mooncake
        reformat_kv_cache:1244-1317, NZ branch only).

        pull_blocks writes ND into the D cache; under enable_kv_nz the D cache
        is physically NZ-ordered (attention writes via npu_scatter_pa_kv_cache,
        mla_v1.py:1413). Load each layer's ND block range out of the D cache,
        then scatter it back into the D cache's NZ view. TP=1 only: MLA NZ has
        num_kv_heads==1 -> tp_n==1; TP>1+NZ needs a staging NZ scatter branch
        (left unsupported).
        """
        if not block_ids:
            return
        first_cache = next(iter(group_kv.values()))
        if isinstance(first_cache, (list, tuple)):
            k_ref, v_ref = first_cache[0], first_cache[1]
        else:
            k_ref = v_ref = first_cache
        dtype = k_ref.dtype
        device = k_ref.device
        num_kv_heads = int(k_ref.shape[-2])
        k_head_dim = int(k_ref.shape[-1])
        v_head_dim = int(v_ref.shape[-1])

        num_blocks = len(block_ids)
        num_tokens = num_blocks * self.block_size
        block_ids_tensor = torch.tensor(block_ids, dtype=torch.int32, device=device)
        block_table = block_ids_tensor.view(1, -1)
        block_len_tensor = torch.tensor([num_tokens], dtype=torch.int32, device=device)
        seq_start_tensor = torch.tensor([0], dtype=torch.int32, device=device)
        # slot_mapping = intra-block offset + block_id * block_size (MC:1273-1276).
        block_offsets = torch.arange(0, self.block_size, dtype=torch.int32, device=device)
        slot_mapping = (
            block_offsets.reshape((1, self.block_size))
            + block_ids_tensor.reshape((num_blocks, 1)) * self.block_size
        ).flatten()
        k_buffer = torch.empty((num_tokens, num_kv_heads, k_head_dim), dtype=dtype, device=device)
        v_buffer = torch.empty((num_tokens, num_kv_heads, v_head_dim), dtype=dtype, device=device)
        # FIXME: skipping sync crashes in GQA (MC:1278-1281); root cause unknown.
        torch.npu.synchronize()
        for d_cache in group_kv.values():
            if isinstance(d_cache, (list, tuple)):
                k_cache_layer, v_cache_layer = d_cache[0], d_cache[1]
            else:
                k_cache_layer = v_cache_layer = d_cache
            torch_npu.atb.npu_paged_cache_load(
                k_cache_layer,
                v_cache_layer,
                block_table,
                block_len_tensor,
                seq_starts=seq_start_tensor,
                key=k_buffer,
                value=v_buffer,
            )
            self._nz_kv_cache(
                k_cache_layer,
                v_cache_layer,
                k_buffer,
                v_buffer,
                slot_mapping,
                num_kv_heads,
                k_head_dim,
                v_head_dim,
            )

    def _nz_kv_cache(
        self,
        k_cache_layer,
        v_cache_layer,
        k_buffer,
        v_buffer,
        slot_mapping,
        num_kv_heads: int,
        k_head_dim: int,
        v_head_dim: int,
    ):
        # fork Mooncake :1347-1365. nz_fmt_last_dim=16 (MLA NZ, aligns
        # attention/mla_v1.py:1413; Mooncake uses 16 too).
        nz_fmt_last_dim = 16
        k_cache_layer = k_cache_layer.view(
            -1, k_head_dim * num_kv_heads // nz_fmt_last_dim,
            self.block_size, nz_fmt_last_dim,
        )
        v_cache_layer = v_cache_layer.view(
            -1, v_head_dim * num_kv_heads // nz_fmt_last_dim,
            self.block_size, nz_fmt_last_dim,
        )
        torch_npu.npu_scatter_pa_kv_cache(
            k_buffer, v_buffer, k_cache_layer, v_cache_layer, slot_mapping,
        )

    def _send_done_signal_to_free_remote_port(
        self, request_id: str, remote_port_send_num: dict[int, RemotePortInfo]
    ):
        """Bug 4 fix (fork Mooncake :757-772): free P ports with num==0
        (mapped into remote_port_send_num but not pulled by this D rank) so
        P-side delayed_free does not wait for timeout. Only device_index==0
        (side_channel_port == local_handshake_port) sends to dedup across
        D ranks that share the same P port set."""
        if self.side_channel_port != self.local_handshake_port or not remote_port_send_num:
            return
        with self.proc_not_transfer_request_lock:
            if request_id not in self.proc_not_transfer_request:
                self.proc_not_transfer_request[request_id] = True
            should_send = self.proc_not_transfer_request[request_id]
            if should_send:
                self.proc_not_transfer_request[request_id] = False
        if should_send:
            done_threads: list[threading.Thread] = []
            for remote_port in remote_port_send_num:
                if remote_port_send_num[remote_port]["num"] == 0:
                    remote_host_ = remote_port_send_num[remote_port]["host"]
                    t = threading.Thread(
                        target=self._send_done_recv_signal,
                        args=(request_id, remote_host_, remote_port, remote_port_send_num),
                    )
                    t.start()
                    done_threads.append(t)
            for t in done_threads:
                t.join()

    def _send_done_recv_signal(
        self,
        request_id: str,
        remote_host: str,
        remote_handshake_port: int,
        remote_port_send_num: dict[int, RemotePortInfo],
    ):
        sock: zmq.Socket | None = None  # type: ignore
        try:
            sock = self._get_remote_socket(remote_host, remote_handshake_port)
            data_bytes = msgspec.msgpack.encode((DONE_RECVING_MSG, request_id, remote_port_send_num))
            ensure_zmq_send(sock, data_bytes, f"{remote_host}:{remote_handshake_port}")
            resp = ensure_zmq_recv(sock, f"{remote_host}:{remote_handshake_port}")
            if resp != b"ACK":
                logger.error("Failed ACK for request %s from %s:%d", request_id, remote_host, remote_handshake_port)
                raise RuntimeError(f"Failed to receive ACK, resp: {resp.decode('utf-8')}")
        except RuntimeError as e:
            if isinstance(sock, zmq.Socket):  # type: ignore
                sock.close()
                sock = None
                logger.warning("Socket error in DONE send. error=%s. ", e)
        finally:
            if sock is not None:
                self._return_remote_socket(sock, remote_host, remote_handshake_port)

    def _get_remote_socket(self, remote_host: str, remote_handshake_port: int) -> zmq.Socket:  # type: ignore
        remote_path = make_zmq_path("tcp", remote_host, remote_handshake_port)
        with self.remote_sockets_lock:
            pool = self.remote_sockets[remote_path]
            if pool:
                return pool.popleft()
        ctx = zmq.Context()  # type: ignore
        sock = make_zmq_socket(ctx=ctx, path=remote_path, socket_type=zmq.REQ, bind=False)  # type: ignore
        sock.setsockopt(zmq.SNDTIMEO, int(self.timeout * 1000))  # type: ignore
        sock.setsockopt(zmq.RCVTIMEO, int(self.timeout * 1000))  # type: ignore
        return sock

    def _return_remote_socket(self, sock: zmq.Socket, remote_host: str, remote_handshake_port: int) -> None:  # type: ignore
        remote_path = make_zmq_path("tcp", remote_host, remote_handshake_port)
        with self.remote_sockets_lock:
            self.remote_sockets[remote_path].append(sock)


# ---------------------------------------------------------------------------
# ConnectorMetadata + dispatcher (forked; class rename only)
# ---------------------------------------------------------------------------
class HIXLConnectorMetadata(KVConnectorMetadata):
    def __init__(self):
        self.requests: dict[str, ReqMeta] = {}
        self.requests_to_send: dict[str, float] = {}
        self.reqs_in_batch: set[str] = set()

    def add_new_req(
        self,
        request_id: str,
        local_block_ids: BlockIds,
        num_external_tokens: int,
        kv_transfer_params: dict[str, Any],
    ):
        self.requests[request_id] = ReqMeta(
            local_block_ids=local_block_ids,
            num_external_tokens=num_external_tokens,
            num_computed_tokens=kv_transfer_params.get("num_computed_tokens", 0),
            remote_block_ids=kv_transfer_params["remote_block_ids"],
            remote_engine_id=kv_transfer_params["remote_engine_id"],
            remote_request_id=kv_transfer_params["remote_request_id"],
            remote_host=kv_transfer_params["remote_host"],
            remote_port=kv_transfer_params["remote_port"],
            remote_ptp_size=kv_transfer_params.get("remote_ptp_size", 1),
            num_prompt_blocks=kv_transfer_params.get("num_prompt_blocks", 0),
            remote_block_size=kv_transfer_params.get("remote_block_size", 0),
            remote_pcp_size=kv_transfer_params.get("remote_pcp_size", 1),
            remote_dcp_size=kv_transfer_params.get("remote_dcp_size", 1),
            remote_multi_nodes_meta_mapping=kv_transfer_params.get("remote_multi_nodes_meta_mapping", {}),
        )


class HIXLConnector(KVConnectorBase_V1, SupportsHMA):
    def __init__(self, vllm_config: VllmConfig, role: KVConnectorRole, kv_cache_config: KVCacheConfig | None = None):
        assert vllm_config.kv_transfer_config is not None
        self.engine_id = vllm_config.kv_transfer_config.engine_id
        self._connector_metadata = HIXLConnectorMetadata()
        if role == KVConnectorRole.SCHEDULER:
            self.connector_scheduler: HIXLConnectorScheduler | None = HIXLConnectorScheduler(
                vllm_config, str(self.engine_id), kv_cache_config
            )
            self.connector_worker: HIXLConnectorWorker | None = None
        elif role == KVConnectorRole.WORKER:
            self.connector_scheduler = None
            self.connector_worker = HIXLConnectorWorker(vllm_config, str(self.engine_id), kv_cache_config)

    # ---- Scheduler side ----
    def get_num_new_matched_tokens(self, request: "Request", num_computed_tokens: int) -> tuple[int, bool]:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.get_num_new_matched_tokens(request, num_computed_tokens)

    def update_state_after_alloc(self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int):
        assert self.connector_scheduler is not None
        return self.connector_scheduler.update_state_after_alloc(request, blocks, num_external_tokens)

    def build_connector_meta(self, scheduler_output: SchedulerOutput) -> KVConnectorMetadata:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.build_connector_meta(scheduler_output)

    def request_finished(self, request: "Request", block_ids: list[int]) -> tuple[bool, dict[str, Any] | None]:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.request_finished(request, (block_ids,))

    def request_finished_all_groups(
        self, request: "Request", block_ids: tuple[list[int], ...]
    ) -> tuple[bool, dict[str, Any] | None]:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.request_finished(request, block_ids)

    # ---- Worker side ----
    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        assert self.connector_worker is not None
        self.connector_worker.register_kv_caches(kv_caches)

    def get_finished(self, finished_req_ids: set[str]) -> tuple[set[str], set[str]]:
        assert self.connector_worker is not None
        return self.connector_worker.get_finished()

    def get_block_ids_with_load_errors(self) -> set[int]:
        assert self.connector_worker is not None
        return self.connector_worker.get_block_ids_with_load_errors()

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs) -> None:
        assert self.connector_worker is not None
        assert isinstance(self._connector_metadata, HIXLConnectorMetadata)
        self.connector_worker.start_load_kv(self._connector_metadata)

    def wait_for_layer_load(self, layer_name: str) -> None:
        pass

    def save_kv_layer(
        self, layer_name: str, kv_layer: torch.Tensor, attn_metadata: "AttentionMetadata", **kwargs
    ) -> None:
        pass

    def wait_for_save(self):
        pass

    def get_handshake_metadata(self) -> KVConnectorHandshakeMetadata | None:
        assert self.connector_worker is not None
        return self.connector_worker.xfer_handshake_metadata

    def set_xfer_handshake_metadata(
        self, metadata: Mapping[int | tuple[int, ...], KVConnectorHandshakeMetadata]
    ) -> None:
        assert self.connector_scheduler is not None
        self.connector_scheduler.set_xfer_handshake_metadata(metadata)

    def set_xfer_handshake_metadata_pp_aware(
        self, metadata: Mapping[int | tuple[int, ...], KVConnectorHandshakeMetadata]
    ) -> None:
        assert self.connector_scheduler is not None
        self.connector_scheduler.set_xfer_handshake_metadata_from_workers(metadata)

    def shutdown(self):
        if self.connector_worker is not None:
            self.connector_worker.shutdown()


class HIXLConnectorScheduler:
    def __init__(self, vllm_config: VllmConfig, engine_id: str, kv_cache_config: KVCacheConfig):
        self.vllm_config = vllm_config
        self.kv_cache_config = kv_cache_config
        init_ascend_config(vllm_config)
        self.ascend_config = get_ascend_config()
        self.block_size = vllm_config.cache_config.block_size
        self.engine_id = engine_id
        self.side_channel_host = get_ip()
        self.tp_size = vllm_config.parallel_config.tensor_parallel_size
        logger.info("Initializing HIXL Scheduler %s", engine_id)

        self.pcp_size = vllm_config.parallel_config.prefill_context_parallel_size
        self.dcp_size = vllm_config.parallel_config.decode_context_parallel_size
        self.side_channel_port = (
            vllm_config.kv_transfer_config.kv_port
            + vllm_config.parallel_config.data_parallel_rank
            * vllm_config.parallel_config.tensor_parallel_size
            * vllm_config.parallel_config.pipeline_parallel_size
            * self.pcp_size
        )
        self._reqs_need_recv: dict[str, tuple[Request, BlockIds, int]] = {}
        self._reqs_need_send: dict[str, float] = {}
        self._reqs_in_batch: set[str] = set()

        self.multi_nodes_meta_mapping: dict[str, dict[str, Any]] = {}
        self.kv_cache_groups = kv_cache_config.kv_cache_groups
        self.group_transfer_info: list[GroupTransferInfo] = [
            self._get_group_transfer_info(group) for group in self.kv_cache_groups
        ]
        self.use_compress = self._model_uses_compress()
        self.need_truncate = self.use_compress or any(
            info.is_state_group for info in self.group_transfer_info
        )

    def get_num_new_matched_tokens(self, request: "Request", num_computed_tokens: int) -> tuple[int, bool]:
        params = request.kv_transfer_params
        if params is not None and params.get("do_remote_prefill"):
            # Remote prefill: pull all prompt blocks from remote.
            token_ids = request.prompt_token_ids or []
            actual = self._state_prefill_token_count(len(token_ids))
            params["num_computed_tokens"] = num_computed_tokens
            count = max(actual - num_computed_tokens, 0)
            if count > 0:
                return count, True

        if params is not None and params.get("do_remote_decode") and self.need_truncate:
            self._truncate_request_for_prefill(request)

        return 0, False

    def _model_uses_compress(self) -> bool:
        hf_config = getattr(self.vllm_config.model_config, "hf_config", None)
        compress_ratios = getattr(hf_config, "compress_ratios", None)
        return isinstance(compress_ratios, (list, tuple, dict))

    def _state_prefill_token_count(self, num_prompt_tokens: int) -> int:
        """D-side only. Returns N-1 for Mamba models since the decoder always
        recomputes the last token and must start from h(N-1)."""
        if self.need_truncate and num_prompt_tokens > 1:
            return num_prompt_tokens - 1
        return num_prompt_tokens

    def _truncate_request_for_prefill(self, request: "Request") -> None:
        """P-side only: drop the last prompt token so the prefiller computes
        h(N-1) instead of h(N). The decoder recomputes the last token to derive
        h(N) correctly. Guarded by _p_side_truncated to avoid repeated
        truncation if the request is preempted and rescheduled."""
        params = request.kv_transfer_params
        if (
            params is not None
            and not params.get("_p_side_truncated")
            and request.num_prompt_tokens > 1
        ):
            if request.prompt_token_ids is not None:
                request.prompt_token_ids.pop()
            elif request.prompt_embeds is not None:
                request.prompt_embeds = request.prompt_embeds[:-1]
            else:
                return
            request._all_token_ids.pop()
            request.num_prompt_tokens -= 1
            request.max_tokens = 1
            params["_p_side_truncated"] = True

    def update_state_after_alloc(self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int):
        params = request.kv_transfer_params
        if params is not None and (params.get("do_remote_prefill", False) or params.get("do_remote_decode", False)):
            self._reqs_in_batch.add(request.request_id)
        if params is not None and params.get("do_remote_prefill"):
            if params.get("remote_block_ids"):
                if all(p in params for p in ("remote_engine_id", "remote_host", "remote_port", "remote_request_id")):
                    local_block_ids = (
                        blocks.get_unhashed_block_ids_all_groups() if num_external_tokens > 0 else []
                    )
                    self._reqs_need_recv[request.request_id] = (request, local_block_ids, num_external_tokens)
                else:
                    logger.warning("Got invalid KVTransferParams. params=%s. ", params)
            else:
                assert num_external_tokens == 0
            params["do_remote_prefill"] = False

    def build_connector_meta(self, scheduler_output: SchedulerOutput) -> KVConnectorMetadata:
        meta = HIXLConnectorMetadata()
        for req_id, (req, block_ids, num_external_tokens) in self._reqs_need_recv.items():
            assert req.kv_transfer_params is not None
            meta.add_new_req(
                request_id=req_id,
                local_block_ids=block_ids,
                num_external_tokens=num_external_tokens,
                kv_transfer_params=req.kv_transfer_params,
            )
        self._reqs_need_recv.clear()
        meta.requests_to_send = self._reqs_need_send
        self._reqs_need_send = {}
        meta.reqs_in_batch = self._reqs_in_batch
        self._reqs_in_batch = set()
        return meta

    def request_finished(
        self, request: "Request", block_ids: BlockIds
    ) -> tuple[bool, dict[str, Any] | None]:
        params = request.kv_transfer_params
        if (
            params is None
            or not params.get("do_remote_decode")
            or request.status != RequestStatus.FINISHED_LENGTH_CAPPED
        ):
            return False, None

        num_prompt_blocks = math.ceil(len(request.prompt_token_ids) / self.block_size)
        computed_block_ids = self._get_transfer_block_ids(
            block_ids, len(request.prompt_token_ids)
        )
        computed_block_lens = [len(bid_list) for bid_list in computed_block_ids]
        delay_free_blocks = sum(computed_block_lens) > 0
        if delay_free_blocks:
            logger.info("Delaying free of %d blocks for request %s", sum(computed_block_lens), request.request_id)
            self._reqs_need_send[request.request_id] = time.time()

        return delay_free_blocks, dict(
            do_remote_prefill=True,
            do_remote_decode=False,
            remote_block_ids=computed_block_ids,
            remote_engine_id=self.engine_id,
            remote_request_id=request.request_id,
            remote_host=self.side_channel_host,
            remote_port=self.side_channel_port,
            remote_ptp_size=self.tp_size,
            remote_pcp_size=self.pcp_size,
            remote_dcp_size=self.dcp_size,
            last_token_id=request.output_token_ids[-1],
            remote_multi_nodes_meta_mapping=self.multi_nodes_meta_mapping,
            num_prompt_blocks=num_prompt_blocks,
            remote_block_size=self.block_size,
        )

    def _get_group_unique_specs(self, group: Any) -> list[Any]:
        if not isinstance(group.kv_cache_spec, UniformTypeKVCacheSpecs):
            return [group.kv_cache_spec]
        specs: list[Any] = []
        for layer_name in group.layer_names:
            layer_spec = group.kv_cache_spec.kv_cache_specs[layer_name]
            if layer_spec not in specs:
                specs.append(layer_spec)
        return specs

    def _get_group_transfer_info(self, group: Any) -> GroupTransferInfo:
        specs = self._get_group_unique_specs(group)
        first_spec = specs[0] if specs else group.kv_cache_spec
        block_size = getattr(
            group.kv_cache_spec, "block_size",
            getattr(first_spec, "block_size", self.block_size),
        )
        is_state_group = any(type(spec).__name__ == "MambaSpec" for spec in specs)
        compress_ratio = 1
        for spec in specs:
            if hasattr(spec, "compress_ratio"):
                compress_ratio = spec.compress_ratio
        return GroupTransferInfo(
            tokens_per_block=block_size * max(1, int(compress_ratio)),
            is_state_group=is_state_group,
        )

    def _get_transfer_block_ids(self, block_ids: BlockIds, prompt_len: int) -> BlockIds:
        """Return blocks that contain prompt KV, dropping MTP extra blocks.

        State groups (Mamba) are not context-block aligned with attention KV,
        so keep them unchanged; only clip attention-like groups. Block-level
        clipping is native to HIXL's block addressing (no byte sub-range
        needed), so this forks Mooncake's _get_transfer_block_ids verbatim.
        """
        if len(block_ids) == 0:
            return block_ids
        assert len(block_ids) == len(self.group_transfer_info), (
            "Number of KV cache groups must match"
        )
        transfer_block_ids: list = []
        cp_size = max(1, self.pcp_size * self.dcp_size)
        for blocks, group_info in zip(block_ids, self.group_transfer_info):
            if group_info.is_state_group:
                transfer_block_ids.append(blocks)
            else:
                num_prompt_blocks = math.ceil(
                    prompt_len / (group_info.tokens_per_block * cp_size)
                )
                transfer_block_ids.append(blocks[:num_prompt_blocks])
        return tuple(transfer_block_ids)

    def _port_offset_from_handshake_metadata(
        self, rank_metadata: KVConnectorHandshakeMetadata, metadata_key: int | tuple[int, ...]
    ) -> int:
        kv_port = self.vllm_config.kv_transfer_config.kv_port
        handshake_port = getattr(rank_metadata, "handshake_port", 0)
        if handshake_port > 0:
            return handshake_port - kv_port
        if isinstance(metadata_key, int):
            return metadata_key
        raise ValueError(f"HIXL handshake metadata missing handshake_port for key {metadata_key}")

    def set_xfer_handshake_metadata_from_workers(
        self, metadata: Mapping[int | tuple[int, ...], KVConnectorHandshakeMetadata]
    ) -> None:
        if not metadata:
            return
        updated_mapping: dict[str, dict[str, Any]] = {}
        for metadata_key, rank_metadata in metadata.items():
            port_offset = self._port_offset_from_handshake_metadata(rank_metadata, metadata_key)
            updated_mapping[str(port_offset)] = {
                "host": rank_metadata.local_ip,
                "engine_id": rank_metadata.engine_id,
            }
        self.multi_nodes_meta_mapping.update(updated_mapping)
        logger.info(
            "HIXL set_xfer_handshake_metadata: worker_count=%d, mapping=%s",
            len(metadata), self.multi_nodes_meta_mapping,
        )

    def set_xfer_handshake_metadata(
        self, metadata: Mapping[int | tuple[int, ...], KVConnectorHandshakeMetadata]
    ) -> None:
        self.set_xfer_handshake_metadata_from_workers(metadata)


class HIXLConnectorWorker:
    def __init__(self, vllm_config: VllmConfig, engine_id: str, kv_cache_config: KVCacheConfig):
        self.vllm_config = vllm_config
        self.ascend_config = get_ascend_config()
        self.engine_id = engine_id
        self.tp_rank = get_tensor_model_parallel_rank()
        self.tp_size = vllm_config.parallel_config.tensor_parallel_size
        self.pp_rank = get_pp_group().rank_in_group
        self.pp_size = vllm_config.parallel_config.pipeline_parallel_size
        self.dp_rank = vllm_config.parallel_config.data_parallel_rank_local
        self.kv_caches: dict[str, torch.Tensor] = {}
        self.side_channel_host = get_ip()
        self.total_layers = vllm_config.model_config.get_total_num_hidden_layers()
        self.num_key_value_heads = vllm_config.model_config.hf_text_config.num_key_value_heads
        self.num_layers = vllm_config.model_config.hf_text_config.num_hidden_layers
        self.num_speculative_tokens = (
            vllm_config.speculative_config.num_speculative_tokens
            if vllm_config.speculative_config is not None
            else 0
        )
        self.num_draft_layers = 0
        if vllm_config.speculative_config is not None:
            if vllm_config.speculative_config.method == "mtp":
                self.num_draft_layers = 1
            elif (
                hasattr(vllm_config.speculative_config, "draft_model_config")
                and getattr(
                    getattr(vllm_config.speculative_config.draft_model_config, "hf_config", None),
                    "num_hidden_layers", None) is not None
            ):
                self.num_draft_layers = (
                    vllm_config.speculative_config.draft_model_config.hf_config.num_hidden_layers
                )

        self.pcp_size = get_pcp_group().world_size
        self.pcp_rank = get_pcp_group().rank_in_group if self.pcp_size > 1 else 0
        self.dcp_size = get_decode_context_model_parallel_world_size()
        self.dcp_rank = (
            get_decode_context_model_parallel_rank() if self.dcp_size > 1 else 0
        )
        assert not (self.pp_size > 1 and self.pcp_size > 1), (
            "HIXLConnector: pp and pcp cannot be enabled at the same time."
        )

        self.kv_cache_config = kv_cache_config
        self.num_blocks: int = kv_cache_config.num_blocks
        self.kv_group2layeridx: dict[int, tuple[dict[str, Any], list[int]]] = {}
        self._layer_specs = {
            layer: group.kv_cache_spec for group in kv_cache_config.kv_cache_groups for layer in group.layer_names
        }

        self._get_prefill_decode_size(vllm_config)
        self.pp_layer_indices = {
            rank: get_prefill_pp_indices(
                self.num_layers, rank, self._prefill_pp_size, self._prefill_pp_layer_partition
            )
            for rank in range(self._prefill_pp_size)
        }
        self._is_hma_required: bool = False  # finalized in register_kv_caches
        self.tp_num_need_pulls: int = 1

        self.side_channel_port = (
            vllm_config.kv_transfer_config.kv_port
            + vllm_config.parallel_config.data_parallel_rank
            * vllm_config.parallel_config.tensor_parallel_size
            * vllm_config.parallel_config.pipeline_parallel_size
            * self.pcp_size
        )
        device_index = (self.pp_rank * self.pcp_size + self.pcp_rank) * self.tp_size + self.tp_rank
        self.handshake_port = self.side_channel_port + device_index

        self.kv_role = vllm_config.kv_transfer_config.kv_role
        self.block_size = vllm_config.cache_config.block_size

        # HIXL engine handle (replaces Mooncake global_te).
        cluster_id, listen_ip, listen_port = self._compute_identity()
        extra = self._extra_options()
        self.hixl = get_datadist(
            kv_role=self.kv_role,
            cluster_id=cluster_id,
            listen_ip=listen_ip,
            listen_port=listen_port,
            device_id=self._current_npu_device_id(),
            link_timeout_ms=extra.get("link_timeout_ms", 5000),
            extra_options=extra,
        )
        self.cache_manager = self.hixl.cache_manager
        self.cluster_id = cluster_id
        self.listen_ip = listen_ip
        self.listen_port = listen_port
        self.model_id = extra.get("model_id", 0)
        self.group_caches: dict[int, Any] = {}  # kv_cache_group_id -> registered Cache
        # Per-cache model_id allocator. Each registered blocks cache MUST get a
        # unique BlocksCacheKey: native cache_key_to_id_ overwrites on duplicate
        # (last-wins), so a shared (cluster_id, model_id) across groups re-points
        # all pulls to the last-registered cache. P and D iterate kv_cache_groups
        # in the same order from the same config, so the assigned ids match.
        self._next_model_id: int = self.model_id
        self._group_model_ids: dict[int, int] = {}  # kv_cache_group_id -> model_id
        self.use_mla = vllm_config.model_config.is_deepseek_mla
        self.use_sparse = False
        self.local_remote_block_port_mapping: dict[str, Any] = {}
        self.remote_port_send_num: dict[str, Any] = {}

        self.kv_send_thread: KVCacheSendingThread | None = None
        self.kv_recv_thread: KVCacheRecvingThread | None = None
        self.xfer_handshake_metadata: HixlAgentMetadata | None = None

    def _extra_options(self) -> dict[str, Any]:
        kvtc = self.vllm_config.kv_transfer_config
        role_key = "prefill" if self.kv_role == "kv_producer" else "decode"
        role_cfg: dict[str, Any] = kvtc.get_from_extra_config(role_key, {})
        role_cfg["tp_size"] = role_cfg.get("tp_size", self.tp_size)
        hixl_cfg: dict[str, Any] = kvtc.get_from_extra_config("hixl", {})
        hixl_cfg.setdefault("prefill" if self.kv_role == "kv_producer" else "decode", role_cfg)
        hixl_cfg.setdefault("cluster_id_base", None)
        hixl_cfg.setdefault("listen_port_base", kvtc.kv_port + 1000)
        return hixl_cfg

    def _compute_identity(self) -> tuple[int, str, int]:
        """Per-rank unique cluster_id / listen_port (see design §5)."""
        extra = self._extra_options()
        cluster_id_base = extra["cluster_id_base"]
        assert cluster_id_base is not None, (
            "extra_config['hixl']['cluster_id_base'] is required (P/D must use disjoint bases)."
        )
        listen_port_base = extra.get("listen_port_base", self.vllm_config.kv_transfer_config.kv_port + 1000)
        device_index = (self.pp_rank * self.pcp_size + self.pcp_rank) * self.tp_size + self.tp_rank
        offset = self.dp_rank * (self.tp_size * self.pp_size * self.pcp_size) + device_index
        return int(cluster_id_base) + offset, self.side_channel_host, int(listen_port_base) + offset

    @staticmethod
    def _current_npu_device_id() -> int | None:
        try:
            cur = torch.npu.current_device()
            if isinstance(cur, int):
                return cur
            s = str(cur)
            return int(s.split(":")[-1]) if ":" in s else None
        except Exception:
            return None

    @staticmethod
    def _as_kv_cache_tuple(kv_cache_tuple: Any) -> list[torch.Tensor]:
        if isinstance(kv_cache_tuple, (list, tuple)):
            return list(kv_cache_tuple)
        return [kv_cache_tuple]

    def _build_kv_group2layeridx(self) -> dict[int, tuple[dict[str, Any], list[int]]]:
        from vllm.v1.worker.utils import extract_layer_index

        def to_msgpackable(value: Any) -> Any:
            if value is None or isinstance(value, (str, int, float, bool)):
                return value
            if isinstance(value, dict):
                return {str(k): to_msgpackable(v) for k, v in value.items()}
            if isinstance(value, (list, tuple)):
                return [to_msgpackable(item) for item in value]
            try:
                builtins_value = msgspec.to_builtins(value)
                if builtins_value is value:
                    return repr(value)
                return to_msgpackable(builtins_value)
            except TypeError:
                return repr(value)

        kv_group2layeridx: dict[int, tuple[dict[str, Any], list[int]]] = {}
        num_attn_module = 2 if self.vllm_config.model_config.hf_text_config.model_type == "longcat_flash" else 1
        next_mtp_layer_idx = self.total_layers
        transfer_group_id = 0
        for kv_cache_group_id, group_spec in enumerate(self.kv_cache_config.kv_cache_groups):
            layer_names = list(group_spec.layer_names)
            assigned_indices: set[int] = set()
            layer_indices: list[int] = []
            for name in layer_names:
                if "mtp" in name:
                    layer_idx = next_mtp_layer_idx
                    next_mtp_layer_idx += 1
                else:
                    layer_idx = extract_layer_index(name, num_attn_module)
                    if (assigned_indices and layer_idx < min(assigned_indices)) or (layer_idx in assigned_indices):
                        layer_idx = next_mtp_layer_idx
                        next_mtp_layer_idx += 1
                assigned_indices.add(layer_idx)
                layer_indices.append(layer_idx)
            spec = group_spec.kv_cache_spec
            if isinstance(spec, UniformTypeKVCacheSpecs):
                spec = {n: spec.kv_cache_specs[n] for n in layer_names}
            serialized_spec = to_msgpackable(spec)
            if not isinstance(serialized_spec, dict):
                serialized_spec = {"repr": serialized_spec}
            num_kv_heads = getattr(group_spec.kv_cache_spec, "num_kv_heads", None)
            if isinstance(num_kv_heads, int):
                serialized_spec["num_kv_heads"] = num_kv_heads
            kv_group2layeridx[transfer_group_id] = (
                {
                    "layer_names": layer_names,
                    "kv_cache_spec_type": type(group_spec.kv_cache_spec).__name__,
                    "kv_cache_spec": serialized_spec,
                    "kv_cache_group_id": kv_cache_group_id,
                },
                layer_indices,
            )
            transfer_group_id += 1
        return kv_group2layeridx

    def _get_prefill_decode_size(self, vllm_config: VllmConfig):
        """Prefill/decode parallel sizes from kv_transfer extra_config.

        Defaults to the local (decode) sizes when the disaggregated prefill
        config is absent, so a non-disaggregated setup degenerates to TP=1
        behavior (num_group_pulls==1, no staging).
        """
        prefill_parallel_config: dict[str, Any] = vllm_config.kv_transfer_config.get_from_extra_config(
            "prefill", {}
        )
        self._prefill_tp_size = int(prefill_parallel_config.get("tp_size", self.tp_size))
        self._prefill_dp_size = int(prefill_parallel_config.get("dp_size", self.dp_rank + 1))
        self._prefill_pp_size = int(prefill_parallel_config.get("pp_size", 1))
        self._prefill_pp_layer_partition = prefill_parallel_config.get("pp_layer_partition")
        self._decode_tp_size = self.tp_size
        # num_group_pulls = prefill_tp // decode_tp; TP>1 staging requires
        # prefill_tp >= decode_tp so each D rank reassembles >=1 P-rank shard.
        assert self._prefill_tp_size >= self._decode_tp_size, (
            f"prefill_tp_size({self._prefill_tp_size}) must be >= decode_tp_size"
            f"({self._decode_tp_size}); set extra_config['prefill']['tp_size']."
        )

    def _requires_group_aware_attention_transfer(self) -> bool:
        total_num_kv_heads = {
            self._get_attention_group_num_key_value_heads(group_spec)
            for group_spec, layer_indices in self.kv_group2layeridx.values()
            if layer_indices and group_spec["kv_cache_spec_type"] != "MambaSpec"
        }
        return len(total_num_kv_heads) > 1

    def _get_attention_group_num_need_pulls(self, group_spec: dict[str, Any], prefill_tp_size: int) -> int:
        return self._get_attention_group_num_need_pulls_for_decode_tp(
            group_spec, prefill_tp_size, self.tp_size
        )

    def _get_attention_group_num_need_pulls_for_decode_tp(
        self,
        group_spec: dict[str, Any],
        prefill_tp_size: int,
        decode_tp_size: int,
    ) -> int:
        num_key_value_heads = self._get_attention_group_num_key_value_heads(group_spec)
        num_d_block_heads = max(1, num_key_value_heads // decode_tp_size)
        num_p_block_heads = max(1, num_key_value_heads // prefill_tp_size)
        return num_d_block_heads // num_p_block_heads

    def _get_attention_group_num_key_value_heads(self, group_spec: dict[str, Any]) -> int:
        kv_cache_spec = group_spec.get("kv_cache_spec", {})
        if isinstance(kv_cache_spec, dict):
            for key in ("total_num_kv_heads", "num_kv_heads", "num_key_value_heads"):
                num_key_value_heads = kv_cache_spec.get(key)
                if isinstance(num_key_value_heads, int):
                    return num_key_value_heads
            for spec in kv_cache_spec.values():
                if not isinstance(spec, dict):
                    continue
                for key in ("total_num_kv_heads", "num_kv_heads", "num_key_value_heads"):
                    num_key_value_heads = spec.get(key)
                    if isinstance(num_key_value_heads, int):
                        return num_key_value_heads
        return self.num_key_value_heads

    def _get_attention_group_remote_rank(
        self,
        req_id: str,
        group_spec: dict[str, Any],
        prefill_tp_size: int,
    ) -> list[int]:
        num_key_value_heads = self._get_attention_group_num_key_value_heads(group_spec)
        num_group_pulls = self._get_attention_group_num_need_pulls(group_spec, prefill_tp_size)
        return self._get_remote_ranks_for_req(
            req_id,
            prefill_tp_size,
            num_key_value_heads=num_key_value_heads,
            tp_num_need_pulls=num_group_pulls,
            use_mla=num_key_value_heads == 1,
        )[self.tp_rank]

    def _get_tp_num_need_pulls(self, prefill_tp_size: int | None) -> int:
        if prefill_tp_size is None:
            prefill_tp_size = self._prefill_tp_size
        if prefill_tp_size == self._prefill_tp_size:
            return self.tp_num_need_pulls
        if self.vllm_config.model_config.is_deepseek_mla:
            return 1
        num_d_block_heads = max(1, self.num_key_value_heads // self.tp_size)
        num_p_block_heads = max(1, self.num_key_value_heads // prefill_tp_size)
        return num_d_block_heads // num_p_block_heads

    def _get_remote_rank(self, req_id: str, prefill_tp_size: int | None = None) -> list[int]:
        return self._get_remote_ranks_for_req(req_id, prefill_tp_size)[self.tp_rank]

    def _get_remote_tp_ranks(
        self,
        tp_ori_data: np.ndarray,
        rand_group_index: list[int],
        num_groups: int,
        prefill_tp_size: int,
        num_key_value_heads: int,
        tp_num_need_pulls: int,
        use_mla: bool,
    ) -> list[list[int]]:
        tp_sampled_nums: list[list[int]] = []
        if prefill_tp_size > num_key_value_heads or use_mla:
            tp_ori_data = tp_ori_data.reshape(-1, num_groups)
            chosen_group = tp_ori_data[:, [rand_group_index]]
            flattened = chosen_group.reshape(-1).tolist()
            tp_sampled_nums = [
                flattened[i : i + tp_num_need_pulls] for i in range(0, len(flattened), tp_num_need_pulls)
            ]
        else:
            group_size = prefill_tp_size // self._decode_tp_size
            for i in range(self._decode_tp_size):
                slice = tp_ori_data[i * group_size : (i + 1) * group_size]
                tp_sampled_nums.append(slice.tolist())
        return tp_sampled_nums

    def _get_remote_ranks_for_req(
        self,
        req_id: str,
        prefill_tp_size: int | None = None,
        num_key_value_heads: int | None = None,
        tp_num_need_pulls: int | None = None,
        use_mla: bool | None = None,
    ) -> list[list[int]]:
        if prefill_tp_size is None:
            prefill_tp_size = self._prefill_tp_size
        if num_key_value_heads is None:
            if self.vllm_config.model_config.is_deepseek_mla:
                num_key_value_heads = 1
            else:
                num_key_value_heads = self.num_key_value_heads
        if tp_num_need_pulls is None:
            tp_num_need_pulls = self._get_tp_num_need_pulls(prefill_tp_size)
        if use_mla is None:
            use_mla = self.vllm_config.model_config.is_deepseek_mla

        sampled_nums: list[list[int]] = []
        if prefill_tp_size == self._decode_tp_size:
            sampled_nums = list(
                map(
                    lambda tp: [tp + pp * prefill_tp_size for pp in range(self._prefill_pp_size)],
                    range(prefill_tp_size),
                )
            )
            return sampled_nums
        ori_data = np.arange(prefill_tp_size * self._prefill_pp_size)
        seed = string_to_int64_hash(req_id)
        rand = random.Random(seed)
        ori_data_2d = ori_data.reshape(self._prefill_pp_size, -1)
        num_groups = max(1, len(ori_data_2d[0]) // num_key_value_heads)
        rand_group_index = rand.sample(range(num_groups), max(self._decode_tp_size // num_key_value_heads, 1))
        all_results = [
            self._get_remote_tp_ranks(
                ori_data_2d[pp_index],
                rand_group_index,
                num_groups,
                prefill_tp_size,
                num_key_value_heads,
                tp_num_need_pulls,
                use_mla,
            )
            for pp_index in range(self._prefill_pp_size)
        ]
        for group_index in range(len(all_results[0])):
            group: list[int] = []
            for pp_index in range(self._prefill_pp_size):
                group.extend(all_results[pp_index][group_index])
            sampled_nums.append(group)
        return sampled_nums

    def _get_hybrid_remote_rank_group_pulls(
        self,
        req_id: str,
        prefill_tp_size: int,
    ) -> tuple[list[int], dict[int, list[GroupPull]]]:
        rank_group_pulls: OrderedDict[int, list[GroupPull]] = OrderedDict()

        def add_group_pull(remote_rank: int, group_pull: GroupPull) -> None:
            rank_group_pulls.setdefault(remote_rank, []).append(group_pull)

        for group_id, (group_spec, layer_indices) in self.kv_group2layeridx.items():
            if not layer_indices:
                continue

            if group_spec["kv_cache_spec_type"] == "MambaSpec":
                assert prefill_tp_size % self.tp_size == 0, (
                    f"Hybrid Mamba prefill tp size({prefill_tp_size}) must be divisible by "
                    f"decode tp size({self.tp_size})."
                )
                num_group_pulls = prefill_tp_size // self.tp_size
                for pp_rank in range(self._prefill_pp_size):
                    pp_rank_offset = pp_rank * prefill_tp_size
                    local_tp_offset = self.tp_rank * num_group_pulls
                    for remote_tp_offset in range(num_group_pulls):
                        remote_rank = pp_rank_offset + local_tp_offset + remote_tp_offset
                        add_group_pull(
                            remote_rank,
                            GroupPull(
                                group_id=group_id,
                                remote_tp_offset=remote_tp_offset,
                                num_group_pulls=num_group_pulls,
                                prefill_pp_rank=pp_rank,
                                is_group_transfer_end=remote_tp_offset == num_group_pulls - 1,
                            ),
                        )
                continue

            num_group_pulls = self._get_attention_group_num_need_pulls(group_spec, prefill_tp_size)
            chosen_rank_list = self._get_attention_group_remote_rank(req_id, group_spec, prefill_tp_size)
            assert len(chosen_rank_list) == num_group_pulls * self._prefill_pp_size, (
                f"chosen_rank_list({chosen_rank_list}) does not match num_group_pulls({num_group_pulls}) "
                f"and prefill pp size({self._prefill_pp_size})."
            )
            for rank_idx, remote_rank in enumerate(chosen_rank_list):
                prefill_pp_rank = rank_idx // num_group_pulls
                add_group_pull(
                    remote_rank,
                    GroupPull(
                        group_id=group_id,
                        remote_tp_offset=rank_idx % num_group_pulls,
                        num_group_pulls=num_group_pulls,
                        prefill_pp_rank=prefill_pp_rank,
                        is_group_transfer_end=rank_idx % num_group_pulls == num_group_pulls - 1,
                    ),
                )

        return list(rank_group_pulls), dict(rank_group_pulls)

    def _get_local_remote_cp_params(self, meta: ReqMeta):
        """Resolve CP geometry (fork Mooncake :2635-2660, address-agnostic).

        Returns (remote_block_size, local_cp_rank, local_cp_size,
        remote_cp_size, r_blk) where r_blk = Bd/Bp (>=1) is the D/P
        block-size ratio.
        """
        remote_block_size = meta.remote_block_size or self.block_size
        local_cp_rank = self.dcp_rank + self.pcp_rank * self.dcp_size
        local_cp_size = self.dcp_size * self.pcp_size
        remote_cp_size = meta.remote_pcp_size * meta.remote_dcp_size
        if remote_block_size != self.block_size:
            assert self.block_size % remote_block_size == 0 or remote_block_size % self.block_size == 0, (
                f"Block sizes of P ({remote_block_size}) and D ({self.block_size}) must be divisible by each other."
            )
            if local_cp_size > 1:
                assert self.block_size % remote_block_size == 0, (
                    f"D node DCP not support P node block_size({remote_block_size}) > D block_size({self.block_size})"
                )
                assert (remote_cp_size // local_cp_size) % (self.block_size // remote_block_size) == 0, (
                    f"remote_cp_size({remote_cp_size}) must be an integer multiple of"
                    f"r({self.block_size // remote_block_size}) * local_cp_size({local_cp_size})"
                )
        r_blk = self.block_size // remote_block_size if self.block_size > remote_block_size else 1
        return remote_block_size, local_cp_rank, local_cp_size, remote_cp_size, r_blk

    def _get_cp_shard_pulls(self, remote_handshake_port_list, prefill_tp_size, remote_base_port, remote_pcp_size):
        """CP case: group_pulls derived from port (fork Mooncake :3060-3109,
        address-agnostic). The port already encodes the random TP choice, so
        no table lookup is needed."""
        mamba_num = prefill_tp_size // self.tp_size
        attn_num = self._get_tp_num_need_pulls(prefill_tp_size)
        attn_gids = [
            g for g, (spec, li) in self.kv_group2layeridx.items()
            if li and spec["kv_cache_spec_type"] != "MambaSpec"
        ]
        mamba_gids = [
            g for g, (spec, li) in self.kv_group2layeridx.items()
            if li and spec["kv_cache_spec_type"] == "MambaSpec"
        ]
        num_shards = len(remote_handshake_port_list)
        result = []
        for shard_idx, ports in enumerate(remote_handshake_port_list):
            is_final = shard_idx == num_shards - 1
            shard_pulls = []
            for port_idx, port in enumerate(ports):
                pulls = []
                port_tp = (port - remote_base_port) % prefill_tp_size
                # PCP and PP are mutually exclusive; when PCP>1, pp_rank==0.
                pp_rank = 0 if remote_pcp_size > 1 else (port - remote_base_port) // prefill_tp_size
                if port_idx < attn_num:
                    pulls += [
                        GroupPull(
                            group_id=g,
                            remote_tp_offset=port_idx,
                            num_group_pulls=attn_num,
                            prefill_pp_rank=pp_rank,
                            is_group_transfer_end=port_idx == attn_num - 1,
                        )
                        for g in attn_gids
                    ]
                if is_final:
                    m_off = port_tp - self.tp_rank * mamba_num
                    if 0 <= m_off < mamba_num:
                        pulls += [
                            GroupPull(
                                group_id=g,
                                remote_tp_offset=m_off,
                                num_group_pulls=mamba_num,
                                prefill_pp_rank=pp_rank,
                                is_group_transfer_end=m_off == mamba_num - 1,
                            )
                            for g in mamba_gids
                        ]
                shard_pulls.append(pulls)
            result.append(shard_pulls)
        return result

    def _get_group_pulls_metadata(
        self,
        req_id: str,
        remote_handshake_port_list: list[list[int]],
        prefill_tp_size: int,
        remote_base_port: int,
        remote_pcp_size: int = 1,
        remote_dcp_size: int = 1,
    ) -> list[list[list[GroupPull]]]:
        cp_transfer = remote_pcp_size * remote_dcp_size > 1
        if cp_transfer:
            return self._get_cp_shard_pulls(
                remote_handshake_port_list, prefill_tp_size, remote_base_port, remote_pcp_size
            )
        if self._is_hma_required:
            # Non-CP: port = base + chosen_rank, one-to-one with table keys.
            _, rank_group_pulls = self._get_hybrid_remote_rank_group_pulls(req_id, prefill_tp_size)
            return [
                [rank_group_pulls[p - remote_base_port] for p in ports]
                for ports in remote_handshake_port_list
            ]

        tp_num_need_pulls = self._get_tp_num_need_pulls(prefill_tp_size)
        group_ids = [group_id for group_id, (_, layer_indices) in self.kv_group2layeridx.items() if layer_indices]

        def make_group_pulls(remote_tp_offset: int, prefill_pp_rank: int) -> list[GroupPull]:
            return [
                GroupPull(
                    group_id=group_id,
                    remote_tp_offset=remote_tp_offset,
                    num_group_pulls=tp_num_need_pulls,
                    prefill_pp_rank=prefill_pp_rank,
                    is_group_transfer_end=remote_tp_offset == tp_num_need_pulls - 1,
                )
                for group_id in group_ids
            ]

        group_pulls_list: list[list[list[GroupPull]]] = []
        for pcp_dcp_rank, remote_ports in enumerate(remote_handshake_port_list):
            if len(remote_ports) == 1:
                remote_tp_offsets = [pcp_dcp_rank % tp_num_need_pulls]
                prefill_pp_ranks = [
                    ((remote_ports[0] - remote_base_port) % (prefill_tp_size * self._prefill_pp_size))
                    // prefill_tp_size
                ]
            else:
                assert len(remote_ports) % tp_num_need_pulls == 0, (
                    f"tp_num_need_pulls: {tp_num_need_pulls}, remote_ports: {remote_ports}"
                )
                remote_tp_offsets = [rank_idx % tp_num_need_pulls for rank_idx in range(len(remote_ports))]
                prefill_pp_ranks = [
                    ((remote_port - remote_base_port) % (prefill_tp_size * self._prefill_pp_size)) // prefill_tp_size
                    for remote_port in remote_ports
                ]
            group_pulls_list.append(
                [
                    make_group_pulls(remote_tp_offset, prefill_pp_rank)
                    for remote_tp_offset, prefill_pp_rank in zip(remote_tp_offsets, prefill_pp_ranks)
                ]
            )
        return group_pulls_list

    def _get_kv_split_metadata(
        self,
        req_id: str,
        meta: ReqMeta,
    ) -> tuple[list[list[int]], list[BlockIds], list[BlockIds]]:
        prefill_tp_size: int = meta.remote_ptp_size if meta.remote_ptp_size is not None else self._prefill_tp_size

        if meta.remote_pcp_size * meta.remote_dcp_size * self.pcp_size * self.dcp_size != 1:
            return self._get_kv_split_metadata_cp(req_id, meta, prefill_tp_size)

        if self._is_hma_required:
            chosen_rank_list, _ = self._get_hybrid_remote_rank_group_pulls(req_id, prefill_tp_size)
        else:
            chosen_rank_list = self._get_remote_rank(req_id, prefill_tp_size)

        remote_handshake_port_list = [[x + meta.remote_port for x in chosen_rank_list]]
        local_block_ids: list[list[int]] = [[] for _ in meta.local_block_ids]
        remote_block_ids: list[list[int]] = [[] for _ in meta.remote_block_ids]
        for group_idx, (group_spec, layer_indices) in self.kv_group2layeridx.items():
            local_kernel_block_ids, remote_kernel_block_ids = self._get_kernel_block_ids(
                layer_indices, meta, group_idx, group_spec
            )
            kv_cache_group_id = self._get_kv_cache_group_id(group_idx, group_spec)
            local_block_ids[kv_cache_group_id] = local_kernel_block_ids
            remote_block_ids[kv_cache_group_id] = remote_kernel_block_ids
        local_block_ids_list = [tuple(local_block_ids) for _ in remote_handshake_port_list]  # type: ignore
        remote_block_ids_list = [tuple(remote_block_ids) for _ in remote_handshake_port_list]  # type: ignore
        return remote_handshake_port_list, local_block_ids_list, remote_block_ids_list

    def _get_kv_split_metadata_cp(
        self,
        req_id: str,
        meta: ReqMeta,
        prefill_tp_size: int,
    ) -> tuple[list[list[int]], list[BlockIds], list[BlockIds]]:
        def context_parallel_parameters_check():
            assert (meta.remote_pcp_size * meta.remote_dcp_size) % (self.pcp_size * self.dcp_size) == 0
            if not (self.use_mla or self.use_sparse):
                p_node_heads_per_rank = math.ceil(self.num_key_value_heads / prefill_tp_size)
                d_node_heads_per_rank = math.ceil(self.num_key_value_heads / self.tp_size)
                assert d_node_heads_per_rank % p_node_heads_per_rank == 0

        def get_kv_head_groups(tp_size):
            if self.use_mla or self.use_sparse:
                return [(0,)]
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
                return [(i,) for i in range(self.num_key_value_heads)]
            return [(0,)]

        def get_cp_group_meta(tp_size, pcp_size, dcp_size, port_base):
            cp_group_meta: dict = {}
            kv_head_groups = get_kv_head_groups(tp_size)
            dcp_repeat_num = tp_size // len(kv_head_groups) // dcp_size
            for kv_head_group_idx, kv_head_group in enumerate(kv_head_groups):
                if kv_head_group not in cp_group_meta:
                    cp_group_meta[kv_head_group] = {"cp_groups": [], "select_cp_groups_id": 0}
                kv_head_group_offset = tp_size // len(kv_head_groups) * kv_head_group_idx
                for dcp_repeat_idx in range(dcp_repeat_num):
                    cp_group = []
                    dcp_repeat_offset = dcp_size * dcp_repeat_idx
                    for pcp_rank in range(pcp_size):
                        pcp_rank_offset = tp_size * pcp_rank
                        for dcp_rank in range(dcp_size):
                            cp_group.append(
                                dcp_rank + port_base + pcp_rank_offset
                                + dcp_repeat_offset + kv_head_group_offset
                            )
                    cp_group_meta[kv_head_group]["cp_groups"].append(cp_group)
            return cp_group_meta

        def get_local_remote_block_port_mappings():
            context_parallel_parameters_check()
            p_node_cp_group_meta = get_cp_group_meta(
                prefill_tp_size, meta.remote_pcp_size, meta.remote_dcp_size, meta.remote_port
            )
            d_node_cp_group_meta = get_cp_group_meta(
                self.tp_size, self.pcp_size, self.dcp_size, self.side_channel_port
            )
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
                            select_cp_groups_id + 1
                            if select_cp_groups_id + 1 < len(p_cp_groups) else 0
                        )
                        for d_idx, d_port in enumerate(d_cp_group):
                            if d_port not in local_remote_block_port_mappings:
                                local_remote_block_port_mappings[d_port] = []
                            p_port_remote_list = []
                            for p_idx, p_port in enumerate(p_cp_group):
                                if (p_idx // r_blk) % len(d_cp_group) == d_idx:
                                    p_port_remote_list.append(p_port)
                            local_remote_block_port_mappings[d_port].append(p_port_remote_list)
            return local_remote_block_port_mappings

        def get_remote_port_send_num(local_remote_block_port_mappings):
            remote_port_send_num: dict[int, RemotePortInfo] = {}
            remote_ports: set[int] = set(
                range(meta.remote_port, meta.remote_port + prefill_tp_size * meta.remote_pcp_size)
            )
            kv_port = self.vllm_config.kv_transfer_config.kv_port
            for key, remote_host_info in (meta.remote_multi_nodes_meta_mapping or {}).items():
                remote_ports.add(int(remote_host_info.get("handshake_port", kv_port + int(key))))
            for remote_port_head_list in local_remote_block_port_mappings.values():
                for remote_port_list in remote_port_head_list:
                    for remote_port in remote_port_list:
                        remote_ports.add(remote_port)
            for remote_port in remote_ports:
                remote_host, _ = self._get_remote_host_info_by_port(
                    meta.remote_port, remote_port, meta.remote_host,
                    meta.remote_engine_id, meta.remote_multi_nodes_meta_mapping,
                )
                remote_port_send_num[remote_port] = {"num": 0, "host": remote_host}
            for remote_port_head_list in local_remote_block_port_mappings.values():
                for remote_port_list in remote_port_head_list:
                    for remote_port in remote_port_list:
                        remote_port_send_num[remote_port]["num"] += 1
            return remote_port_send_num

        def _set_hma_shared_port(remote_handshake_port_list):
            if self._is_hma_required and not (self.use_mla or self.use_sparse):
                remote_dcp = max(meta.remote_dcp_size, 1)
                group_span = prefill_tp_size // len(get_kv_head_groups(prefill_tp_size))
                n_replica = max(group_span // remote_dcp, 1)
                chosen_tp_list = self._get_remote_rank(req_id, prefill_tp_size)
                if n_replica > 1:
                    for shard_ports in remote_handshake_port_list:
                        for i in range(len(shard_ports)):
                            tp_off = (shard_ports[i] - meta.remote_port) % prefill_tp_size
                            pcp_seg = (shard_ports[i] - meta.remote_port) - tp_off
                            group_off = tp_off // group_span * group_span
                            dcp_part = (tp_off - group_off) % remote_dcp
                            replica = (chosen_tp_list[i % len(chosen_tp_list)] // remote_dcp) % n_replica
                            shard_ports[i] = (
                                meta.remote_port + pcp_seg + group_off + replica * remote_dcp + dcp_part
                            )
                k = prefill_tp_size // self.tp_size
                final_ports = remote_handshake_port_list[-1]
                pcp_seg = (final_ports[0] - meta.remote_port) // prefill_tp_size * prefill_tp_size
                for j in range(k):
                    p = meta.remote_port + pcp_seg + self.tp_rank * k + j
                    if p not in final_ports:
                        final_ports.append(p)
            return remote_handshake_port_list

        remote_block_size, local_cp_rank, local_cp_size, remote_cp_size, r_blk = (
            self._get_local_remote_cp_params(meta)
        )
        # Per attention group kernel-expansion params (local_scale, remote_scale,
        # kernel_size). scale>1 (MLA/compress) is expanded to kernel blocks below;
        # under scale==1 this degenerates to the original logical-slice geometry.
        group_kernel_params = self._get_group_kernel_params(remote_block_size)
        # r_blk>1 (Bd>Bp) requires MLA/compress (block_size_scale>1) so that
        # kernel_size = Bd/scale divides Bp; under scale==1 (no MLA) Mooncake's
        # _local_kernel_ids_for_shard yields kernels_per_p_block = Bp//Bd == 0
        # (no transfer). r_blk>1 is unsupported without MLA. scale>1 (use_mla)
        # now flows through kernel expansion; fail fast otherwise rather than
        # emit a wrong/empty shard.
        assert r_blk == 1 or self.use_mla, (
            "HIXL supports r_blk==1 (P/D same block_size) only; "
            "r_blk>1 (Bd>Bp) needs MLA/compress (block_size_scale>1)."
        )

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
        local_remote_block_port_mapping = list(
            self.local_remote_block_port_mapping[meta.remote_engine_id]
        )

        num_external_blocks = math.ceil(meta.num_external_tokens / self.block_size)
        num_external_blocks_p = math.ceil(meta.num_external_tokens / remote_block_size)
        kv_group_items = list(self.kv_group2layeridx.items())
        sequence_group_idx = next(
            (
                group_spec.get("kv_cache_group_id", group_idx)
                for group_idx, (group_spec, _) in kv_group_items
                if group_spec["kv_cache_spec_type"] != "MambaSpec"
            ),
            0,
        )
        assert math.ceil(num_external_blocks / (self.pcp_size * self.dcp_size)) == len(
            meta.local_block_ids[sequence_group_idx]
        ), (
            f"num_external_blocks({num_external_blocks}), cp_size({self.pcp_size * self.dcp_size}), "
            f"local_block_ids_len ({len(meta.local_block_ids[sequence_group_idx])})"
        )
        assert meta.num_prompt_blocks >= num_external_blocks_p, (
            f"meta.num_prompt_blocks({meta.num_prompt_blocks}), num_external_blocks({num_external_blocks})"
        )

        remote_block_nums_all = [meta.num_prompt_blocks // remote_cp_size] * remote_cp_size
        num_remain_blocks = meta.num_prompt_blocks % remote_cp_size
        for i in range(num_remain_blocks):
            remote_block_nums_all[i] += 1
        last_block_location = (num_remain_blocks + remote_cp_size - 1) % remote_cp_size

        # Considering prefix cache, the remote_block_nums_all should be revised
        num_prefix_cached_blocks = meta.num_prompt_blocks - num_external_blocks_p
        remote_block_nums_all = [num - num_prefix_cached_blocks // remote_cp_size for num in remote_block_nums_all]
        num_remain_blocks = num_prefix_cached_blocks % remote_cp_size
        for i in range(num_remain_blocks):
            remote_block_nums_all[i] -= 1

        remote_block_nums: list[int] = []
        shard_cp_ranks: list[int] = []
        final_block_idx: int | None = None
        for cp_rank, block_num in enumerate(remote_block_nums_all):
            if (cp_rank // r_blk) % local_cp_size == local_cp_rank:
                if last_block_location == cp_rank:
                    final_block_idx = len(remote_block_nums)
                remote_block_nums.append(block_num)
                shard_cp_ranks.append(cp_rank)

        assert local_remote_block_port_mapping is not None
        if final_block_idx is not None:
            final_block_num = remote_block_nums.pop(final_block_idx)
            shard_cp_ranks.append(shard_cp_ranks.pop(final_block_idx))
            remote_block_nums.append(final_block_num)
            for idx, mapping in enumerate(local_remote_block_port_mapping):
                mapping = mapping.copy()
                final_block_port = mapping.pop(final_block_idx)
                mapping.append(final_block_port)
                local_remote_block_port_mapping[idx] = mapping

        num_prefix_p_blocks = num_prefix_cached_blocks
        if r_blk > 1:
            assert num_prefix_p_blocks % r_blk == 0, (
                f"P0({num_prefix_p_blocks}) should be r_blk({r_blk}) integer multiple"
            )
        num_prefix_d_blocks = num_prefix_p_blocks // r_blk
        first_d = num_prefix_d_blocks + ((local_cp_rank - num_prefix_d_blocks) % local_cp_size)

        remote_handshake_port_list, local_block_ids_list, remote_block_ids_list = [], [], []
        for idx in range(len(local_remote_block_port_mapping[0])):
            mapping_list = []
            for mapping in local_remote_block_port_mapping:
                mapping_list.append(mapping[idx])
            remote_handshake_port_list.append(mapping_list)
        remote_handshake_port_list = _set_hma_shared_port(remote_handshake_port_list)

        for remote_kv_id in range(len(remote_handshake_port_list)):
            num_blocks_to_pull = remote_block_nums[remote_kv_id]
            shard_cp_rank = shard_cp_ranks[remote_kv_id]
            remote_first = (num_prefix_p_blocks - shard_cp_rank + remote_cp_size - 1) // remote_cp_size
            group_remote_block_ids: list[list[int]] = []
            group_local_block_ids: list[list[int]] = []
            is_final_shard = remote_kv_id == len(remote_handshake_port_list) - 1
            for group_idx, (group_spec, _) in kv_group_items:
                if group_spec["kv_cache_spec_type"] == "MambaSpec":
                    # Mamba state is not context-block sharded; transfer from
                    # the final PCP/DCP shard only (fork Mooncake :3010-3015).
                    group_remote_block_ids.append(list(meta.remote_block_ids[group_idx]) if is_final_shard else [])
                    group_local_block_ids.append(list(meta.local_block_ids[group_idx]) if is_final_shard else [])
                    continue
                # Attention: expand to kernel blocks (fork Mooncake :3016-3042).
                # Remote is sliced from remote_first (skips this rank's
                # prefix-cached blocks) then expanded; local kernels are located
                # directly from CP rank + block index via _local_kernel_ids_for_shard.
                # Under scale==1 + r_blk==1 this degenerates to the original
                # logical-slice behavior; scale>1 (MLA/compress) maps each logical
                # block to `scale` kernel (tensor) blocks.
                _, remote_scale, kernel_size = group_kernel_params[group_idx]
                remote_logical = list(
                    meta.remote_block_ids[group_idx][remote_first : remote_first + num_blocks_to_pull]
                )
                kernel_remote = self._expand_block_ids(remote_logical, remote_scale)
                kernel_local = self._local_kernel_ids_for_shard(
                    remote_first,
                    num_blocks_to_pull,
                    shard_cp_rank,
                    num_prefix_p_blocks,
                    first_d,
                    r_blk,
                    local_cp_size,
                    remote_cp_size,
                    remote_block_size,
                    kernel_size,
                    list(meta.local_block_ids[group_idx]),
                )
                num_kernel_blocks = min(len(kernel_remote), len(kernel_local))
                group_remote_block_ids.append(kernel_remote[:num_kernel_blocks])
                group_local_block_ids.append(kernel_local[:num_kernel_blocks])
            remote_block_ids_list.append(tuple(group_remote_block_ids))  # type: ignore
            local_block_ids_list.append(tuple(group_local_block_ids))  # type: ignore

        tp_num_need_pulls = self._get_tp_num_need_pulls(prefill_tp_size)
        if self._is_hma_required:
            assert len(remote_handshake_port_list[0]) >= tp_num_need_pulls, (
                f"tp_num_need_pulls: {tp_num_need_pulls}, remote_handshake_port_list: {remote_handshake_port_list}"
            )
        else:
            assert tp_num_need_pulls == len(remote_handshake_port_list[0]), (
                f"tp_num_need_pulls: {tp_num_need_pulls}, remote_handshake_port_list: {remote_handshake_port_list}"
            )
        return remote_handshake_port_list, local_block_ids_list, remote_block_ids_list

    def _get_remote_host_info_by_port(
        self,
        base_port: int,
        remote_handshake_port: int,
        remote_host: str,
        remote_engine_id: str,
        remote_multi_nodes_meta_mapping: dict,
    ):
        if remote_multi_nodes_meta_mapping is None:
            return remote_host, remote_engine_id
        kv_port = self.vllm_config.kv_transfer_config.kv_port
        rank = str(remote_handshake_port - kv_port)
        info = remote_multi_nodes_meta_mapping.get(rank)
        if info is None:
            rank = str(remote_handshake_port - base_port)
            info = remote_multi_nodes_meta_mapping.get(rank)
        if info is None:
            return remote_host, remote_engine_id
        return info.get("host", remote_host), info.get("engine_id", remote_engine_id)

    @staticmethod
    def _expand_block_ids(block_ids, scale):
        # Expand each logical block into its `scale` contiguous kernel blocks:
        # logical block b -> [b*scale, b*scale+1, ..., b*scale+scale-1].
        return [bid * scale + offset for bid in block_ids for offset in range(scale)]

    @staticmethod
    def _group_compress_ratio(group_spec):
        # Tokens per KV slot for this group (>1 for compressed specs); defaults to 1.
        compress_ratio = 1
        kv_cache_spec = group_spec.get("kv_cache_spec")
        if isinstance(kv_cache_spec, dict):
            for spec in kv_cache_spec.values():
                if isinstance(spec, dict) and isinstance(spec.get("compress_ratio"), int):
                    compress_ratio = max(1, spec["compress_ratio"])
                    break
        return compress_ratio

    @staticmethod
    def _get_kv_cache_group_id(group_idx: int, group_spec: dict[str, Any]) -> int:
        return group_spec.get("kv_cache_group_id", group_idx)

    def _get_kernel_block_ids(self, layer_indices, meta: ReqMeta, group_idx: int, group_spec):
        """No-CP per-group block ids at kernel granularity: (local, remote).

        HIXL adaptation: block_size_scale is indexed [group_idx][0] (one
        transfer group shares one shape, hixl:2546-2550, so group-internal
        scale is uniform). scale>1 (MLA/compress) expands each logical block
        into `scale` kernel blocks via _expand_block_ids. Mamba logical ids
        pass through unchanged.
        """
        kv_cache_group_id = self._get_kv_cache_group_id(group_idx, group_spec)
        if group_spec["kv_cache_spec_type"] == "MambaSpec":
            # align mode: the block table is position-indexed over max_len but
            # only 2+num_speculative_blocks state blocks are resident; earlier
            # blocks are nulled (MambaSpec :731-737). Pull just the final
            # resident state block — remote picks the live SSM block at
            # len - num_speculative_tokens - 1, local picks the freshly
            # allocated block 0 (fork Mooncake :867-869). all mode would pull
            # every block, but align is the Qwen3.6 default.
            remote_blocks = list(meta.remote_block_ids[kv_cache_group_id])
            local_blocks = list(meta.local_block_ids[kv_cache_group_id])
            if remote_blocks:
                transfer_block_idx = len(remote_blocks) - self.num_speculative_tokens - 1
                if transfer_block_idx < 0:
                    transfer_block_idx = 0
                remote_blocks = [remote_blocks[transfer_block_idx]]
            if local_blocks:
                local_blocks = [local_blocks[0]]
            return local_blocks, remote_blocks

        remote_block_size = meta.remote_block_size or self.block_size
        local_scale = self.block_size_scale[group_idx][0]
        kernel_size = self.block_size // local_scale
        assert remote_block_size % kernel_size == 0, (
            f"remote_block_size({remote_block_size}) not divisible by kernel_size({kernel_size})"
        )
        remote_scale = remote_block_size // kernel_size
        kernel_local = self._expand_block_ids(list(meta.local_block_ids[kv_cache_group_id]), local_scale)
        kernel_remote = self._expand_block_ids(list(meta.remote_block_ids[kv_cache_group_id]), remote_scale)
        remote_kernel_token_size = kernel_size * self._group_compress_ratio(group_spec)
        remote_start_idx = meta.num_computed_tokens // remote_kernel_token_size
        kernel_remote = kernel_remote[remote_start_idx:]
        num_kernel_blocks = min(len(kernel_remote), len(kernel_local))
        return kernel_local[:num_kernel_blocks], kernel_remote[:num_kernel_blocks]

    def _get_group_kernel_params(self, remote_block_size):
        # Per attention group kernel-expansion params: (local_scale, remote_scale, kernel_size).
        # The kernel size is shared by both sides, so remote_scale is derived locally from it
        # (no remote handshake scale needed). Mamba groups are not block-sharded and skipped.
        # fork Mooncake :2618-2633; block_size_scale indexed per-group (hixl) vs per-layer (MC).
        group_kernel_params: dict[int, tuple[int, int, int]] = {}
        for group_idx, (group_spec, layer_indices) in self.kv_group2layeridx.items():
            if group_spec["kv_cache_spec_type"] == "MambaSpec":
                continue
            local_scale = self.block_size_scale[group_idx][0]
            kernel_size = self.block_size // local_scale
            assert remote_block_size % kernel_size == 0, (
                f"remote_block_size({remote_block_size}) not divisible by kernel_size({kernel_size})"
            )
            remote_scale = remote_block_size // kernel_size
            group_kernel_params[group_idx] = (local_scale, remote_scale, kernel_size)
        return group_kernel_params

    def _local_kernel_ids_for_shard(
        self,
        shard_first_p_block,
        num_blocks_to_pull,
        shard_cp_rank,
        num_prefix_p_blocks,
        rank_first_d_block,
        block_size_ratio,
        local_cp_size,
        remote_cp_size,
        remote_block_size,
        kernel_size,
        local_block_ids,
    ):
        """Map this shard's pulled P-blocks straight to D-side kernel block ids.

        fork Mooncake :2505-2567 (block-level,寻址无关可直接 fork). The shard
        (CP rank ``shard_cp_rank``) pulls ``num_blocks_to_pull`` P-blocks starting
        at this rank's local index ``shard_first_p_block``. The destination
        kernel position is derived directly from the CP rank and the block
        index. Under r_blk==1 + scale==1 this degenerates to the original
        logical-slice behavior (kernels_per_d_block==1 -> kernel id == d_block).
        """
        # Number of kernel blocks contained in one D-block (Bd/kernel) and one P-block (Bp/kernel).
        kernels_per_d_block = self.block_size // kernel_size
        kernels_per_p_block = remote_block_size // kernel_size
        # Tokens addressable by this rank's D-blocks; a kernel beyond this has no destination.
        local_token_limit = len(local_block_ids) * self.block_size
        kernel_block_ids: list[int] = []
        for block_idx in range(num_blocks_to_pull):
            # P-blocks are round-robin interleaved across the remote CP ranks, so this rank's
            # block_idx-th pulled block maps to global prompt block (in P-units):
            #   global_p_block = (shard_first_p_block + block_idx) * Rcp + shard_cp_rank
            global_p_block = (shard_first_p_block + block_idx) * remote_cp_size + shard_cp_rank
            if remote_block_size > self.block_size:
                # Bp > Bd (only supported when D-side has no CP): one P-block spans multiple
                # D-blocks, so walk it kernel by kernel via the absolute token offset within
                # the external (post-prefix) zone: p_block_token_start = (p - P0) * Bp.
                p_block_token_start = (global_p_block - num_prefix_p_blocks) * remote_block_size
                for kernel_idx in range(kernels_per_p_block):
                    token_offset = p_block_token_start + kernel_idx * kernel_size
                    if token_offset >= local_token_limit:
                        # P-side tail block is partial; its trailing kernels have no D token.
                        break
                    # Locate the D-block holding this token, then the kernel slot inside it.
                    d_block = local_block_ids[token_offset // self.block_size]
                    kernel_in_d_block = (token_offset % self.block_size) // kernel_size
                    kernel_block_ids.append(d_block * kernels_per_d_block + kernel_in_d_block)
            else:
                # Bd >= Bp: the P-block falls entirely inside one D-block.
                # Global D-block d = p // r (r = Bd/Bp); its index within this rank's local
                # list is (d - rank_first_d_block) // Lcp. The P-block occupies a contiguous
                # run of kernels_per_p_block kernels starting at intra-block kernel offset
                # ((p % r) * Bp) / kernel.
                d_block_local_idx = (global_p_block // block_size_ratio - rank_first_d_block) // local_cp_size
                if d_block_local_idx >= len(local_block_ids):
                    # Pairs with the remote-side truncation when the P-side tail block is partial.
                    continue
                d_block = local_block_ids[d_block_local_idx]
                first_kernel_in_d_block = ((global_p_block % block_size_ratio) * remote_block_size) // kernel_size
                for kernel_idx in range(kernels_per_p_block):
                    kernel_block_ids.append(d_block * kernels_per_d_block + first_kernel_in_d_block + kernel_idx)
        return kernel_block_ids

    def _alloc_model_id(self) -> int:
        """Allocate a unique model_id for one registered blocks cache.

        Each blocks cache gets its own BlocksCacheKey so native cache_key_to_id_
        does not overwrite a sibling group's key (last-wins) and pull_blocks
        resolves the intended src cache. Deterministic on P and D since both
        walk kv_cache_groups in the same order.
        """
        mid = self._next_model_id
        self._next_model_id += 1
        return mid

    def _init_staging_caches(self) -> None:
        if self.kv_role != "kv_consumer":
            return
        from llm_datadist import CacheDesc, BlocksCacheKey, Placement

        self.staging_tensors: dict[int, dict[str, list[torch.Tensor]]] = {}
        self.staging_caches: dict[int, Any] = {}
        prefill_tp_size = self._prefill_tp_size
        for group_id, (group_spec, layer_indices) in self.kv_group2layeridx.items():
            if not layer_indices or group_spec["kv_cache_spec_type"] == "MambaSpec":
                continue
            tp_n = self._get_attention_group_num_need_pulls(group_spec, prefill_tp_size)
            if tp_n <= 1:
                continue
            kv_cache_group_id = self._get_kv_cache_group_id(group_id, group_spec)
            layer_names = group_spec["layer_names"]
            ref_k = self._as_kv_cache_tuple(self.kv_caches[layer_names[0]])[0]
            num_d_heads = int(ref_k.shape[-2])
            head_per_split = num_d_heads // tp_n
            dim = int(ref_k.shape[-1])
            staging_shape = [self.num_blocks * tp_n, self.block_size, head_per_split, dim]
            addrs: list[int] = []
            layer_staging: dict[str, list[torch.Tensor]] = {}
            for layer_name in layer_names:
                k_t = torch.zeros(staging_shape, dtype=ref_k.dtype, device=ref_k.device)
                v_t = torch.zeros(staging_shape, dtype=ref_k.dtype, device=ref_k.device)
                layer_staging[layer_name] = [k_t, v_t]
                addrs.append(int(k_t.data_ptr()))
                addrs.append(int(v_t.data_ptr()))
            self.staging_tensors[kv_cache_group_id] = layer_staging
            cache_desc = CacheDesc(
                num_tensors=len(addrs),
                shape=list(staging_shape),
                data_type=_torch_dtype_to_llm_dtype(ref_k.dtype),
                placement=Placement.DEVICE,
            )
            cache = self.cache_manager.register_blocks_cache(
                cache_desc,
                addrs,
                BlocksCacheKey(self.cluster_id, self._alloc_model_id()),
                remote_accessible=True,
            )
            self.staging_caches[kv_cache_group_id] = cache
            logger.info(
                "HIXL staging register: group=%s tp_n=%s shape=%s num_tensors=%s",
                kv_cache_group_id, tp_n, staging_shape, len(addrs),
            )

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        """Register KV via HIXL register_blocks_cache (one Cache per group).

        Replaces Mooncake's collect_storage_merged_register_regions +
        global_te.register_buffer. Both P and D register with
        remote_accessible=True: P so the decoder can find/resolve the src
        cache, D because llm_datadist's PullCacheByGet path (force-enabled by
        EnableRemoteCacheAccessible=1) requires the local dst cache to be
        remote_accessible too, else pull_blocks returns LLM_PARAM_INVALID."""
        from llm_datadist import CacheDesc, BlocksCacheKey, Placement

        self.kv_caches = kv_caches
        self.kv_group2layeridx = self._build_kv_group2layeridx()

        self._is_hma_required = self._requires_group_aware_attention_transfer() or any(
            spec["kv_cache_spec_type"] != "FullAttentionSpec"
            for spec, _ in self.kv_group2layeridx.values()
        )
        # Per-rank num_group_pulls for the non-HMA uniform path. HMA recomputes
        # this per group via _get_attention_group_num_need_pulls.
        if self.vllm_config.model_config.is_deepseek_mla:
            self.tp_num_need_pulls = 1
        else:
            num_d_block_heads = max(1, self.num_key_value_heads // self.tp_size)
            num_p_block_heads = max(1, self.num_key_value_heads // self._prefill_tp_size)
            self.tp_num_need_pulls = num_d_block_heads // num_p_block_heads

        num_tensors_per_group: list[int] = []
        block_size_scale: list[list[int]] = []
        layer_name_to_idx = {
            name: idx
            for _, (spec, idxs) in self.kv_group2layeridx.items()
            for name, idx in zip(spec["layer_names"], idxs)
        }

        for group_id, (group_spec, layer_indices) in self.kv_group2layeridx.items():
            kv_cache_group_id = group_spec.get("kv_cache_group_id", group_id)
            layer_names = group_spec["layer_names"]
            if group_spec["kv_cache_spec_type"] == "MambaSpec":
                # G2: conv/ssm registered as two independent Caches. CacheDesc's
                # single-shape constraint cannot hold heterogeneous state tensors
                # (conv 2D vs ssm 3D), and the old uniform-shape / *2 asserts
                # (:2648/:2655 below) only hold for attention K+V. conv+ssm share
                # one block table (same raw_tensor), so pull reuses one block id
                # for both sub-caches with tensor_num_per_layer=1. Forks Mooncake
                # per-layer (conv_addr, ssm_addr) byte geometry (MC:1131-1154).
                conv_addrs: list[int] = []
                ssm_addrs: list[int] = []
                conv_shape = ssm_shape = None
                conv_dtype = ssm_dtype = None
                for layer_name in layer_names:
                    states = self._as_kv_cache_tuple(self.kv_caches[layer_name])
                    assert len(states) == 2, (
                        f"mamba layer {layer_name} expects [conv, ssm]; got {len(states)}"
                    )
                    c, s = states[0], states[1]
                    conv_addrs.append(int(c.data_ptr()))
                    ssm_addrs.append(int(s.data_ptr()))
                    if conv_shape is None:
                        conv_shape, conv_dtype = tuple(c.shape), c.dtype
                        ssm_shape, ssm_dtype = tuple(s.shape), s.dtype
                    else:
                        assert tuple(c.shape) == conv_shape, (
                            f"mamba conv shape mismatch: {layer_name} "
                            f"{tuple(c.shape)} != {conv_shape}"
                        )
                        assert tuple(s.shape) == ssm_shape, (
                            f"mamba ssm shape mismatch: {layer_name} "
                            f"{tuple(s.shape)} != {ssm_shape}"
                        )
                # mamba has no compress: scale (tensor num_blocks / logical) == 1.
                conv_scale = conv_shape[0] // self.num_blocks
                ssm_scale = ssm_shape[0] // self.num_blocks
                assert conv_scale == 1 and ssm_scale == 1, (
                    f"mamba block_size_scale must be 1 (no compress); "
                    f"conv={conv_scale} ssm={ssm_scale}"
                )
                block_size_scale.append([conv_scale, ssm_scale])
                num_tensors_per_group.append(len(conv_addrs) + len(ssm_addrs))
                # conv and ssm are independent caches with different shapes; give
                # each a unique BlocksCacheKey so pull_blocks can target them
                # separately (a shared key is last-wins overwritten by the native
                # cache_key_to_id_ map).
                conv_model_id = self._alloc_model_id()
                ssm_model_id = self._alloc_model_id()
                conv_desc = CacheDesc(
                    num_tensors=len(conv_addrs),
                    shape=list(conv_shape),
                    data_type=_torch_dtype_to_llm_dtype(conv_dtype),
                    placement=Placement.DEVICE,
                )
                ssm_desc = CacheDesc(
                    num_tensors=len(ssm_addrs),
                    shape=list(ssm_shape),
                    data_type=_torch_dtype_to_llm_dtype(ssm_dtype),
                    placement=Placement.DEVICE,
                )
                conv_cache = self.cache_manager.register_blocks_cache(
                    conv_desc, conv_addrs,
                    BlocksCacheKey(self.cluster_id, conv_model_id),
                    remote_accessible=True,
                )
                ssm_cache = self.cache_manager.register_blocks_cache(
                    ssm_desc, ssm_addrs,
                    BlocksCacheKey(self.cluster_id, ssm_model_id),
                    remote_accessible=True,
                )
                self.group_caches[kv_cache_group_id] = MambaCacheBundle(
                    conv=conv_cache, ssm=ssm_cache, num_layers=len(conv_addrs),
                    conv_model_id=conv_model_id, ssm_model_id=ssm_model_id,
                )
                continue
            addrs: list[int] = []
            ref_shape = None
            ref_dtype = None
            for layer_name in layer_names:
                for single_kv_cache in self._as_kv_cache_tuple(kv_caches[layer_name]):
                    addrs.append(int(single_kv_cache.data_ptr()))
                    if ref_shape is None:
                        ref_shape = tuple(single_kv_cache.shape)
                        ref_dtype = single_kv_cache.dtype
                    else:
                        assert tuple(single_kv_cache.shape) == ref_shape, (
                            f"HIXL register_blocks_cache requires all tensors in a group "
                            f"share one shape; {layer_name} "
                            f"{tuple(single_kv_cache.shape)} != {ref_shape}"
                        )
            assert ref_shape is not None, f"No KV tensors found for group {group_id}"
            num_tensors = len(addrs)
            assert num_tensors == len(layer_indices) * 2, (
                f"num_tensors {num_tensors} != layers*2 {len(layer_indices) * 2}"
            )
            num_tensors_per_group.append(num_tensors)
            # block_size_scale[group]: tensor num_blocks / logical num_blocks.
            # scale>=1: scale==1 is standard FullAttention; scale>1 is MLA/compress
            # (DeepseekV4), where one logical block spans `scale` kernel (tensor)
            # blocks. Kernel-block expansion (_get_kernel_block_ids /
            # _local_kernel_ids_for_shard) handles scale>1 (fork Mooncake).
            scale = ref_shape[0] // self.num_blocks
            assert scale >= 1, (
                f"block_size_scale must be >= 1 (MLA/compress scale); got {scale}."
            )
            block_size_scale.append([scale])

            cache_desc = CacheDesc(
                num_tensors=num_tensors,
                shape=list(ref_shape),  # full single-tensor shape incl num_blocks
                data_type=_torch_dtype_to_llm_dtype(ref_dtype),
                placement=Placement.DEVICE,
            )
            # Both P and D register remote_accessible=True. P so the decoder
            # can resolve the src cache; D because llm_datadist's PullCacheByGet
            # path (force-enabled via EnableRemoteCacheAccessible=1) requires
            # the local dst cache to be remote_accessible too, else
            # pull_blocks returns LLM_PARAM_INVALID.
            remote_accessible = True
            attn_model_id = self._alloc_model_id()
            self._group_model_ids[kv_cache_group_id] = attn_model_id
            cache = self.cache_manager.register_blocks_cache(
                cache_desc,
                addrs,
                BlocksCacheKey(self.cluster_id, attn_model_id),
                remote_accessible=remote_accessible,
            )
            self.group_caches[kv_cache_group_id] = cache

        self.block_size_scale = block_size_scale

        # D-side staging caches for TP>1 head reassembly (no-op for producer).
        self.staging_tensors: dict[int, dict[str, list[torch.Tensor]]] = {}
        self.staging_caches: dict[int, Any] = {}
        self._init_staging_caches()

        metadata = HixlAgentMetadata(
            engine_id=self.engine_id,
            cluster_id=self.cluster_id,
            listen_ip=self.listen_ip,
            listen_port=self.listen_port,
            model_id=self.model_id,
            num_tensors_per_group=num_tensors_per_group,
            kv_group2layeridx=self.kv_group2layeridx,
            block_size=self.block_size,
            num_blocks=self.num_blocks,
            block_size_scale=block_size_scale,
            local_ip=get_ip(),
            handshake_port=self.handshake_port,
        )
        self.xfer_handshake_metadata = metadata

        ready_event = threading.Event()
        if self.kv_role == "kv_producer":
            self.kv_send_thread = KVCacheSendingThread(
                self.vllm_config,
                self.tp_rank,
                self.tp_size,
                self.engine_id,
                self.side_channel_host,
                self.side_channel_port,
                metadata,
                ready_event,
                self.kv_caches,
                self.pcp_rank,
                pcp_size=self.pcp_size,
            )
            self.kv_send_thread.start()
        else:
            self.kv_recv_thread = KVCacheRecvingThread(
                self.tp_rank,
                self.tp_size,
                self.hixl,
                self.model_id,
                self.engine_id,
                self.handshake_port,
                self.side_channel_port,
                self.vllm_config,
                self.kv_caches,
                self.kv_group2layeridx,
                self.group_caches,
                self._group_model_ids,
                block_size_scale,
                ready_event,
                staging_tensors=self.staging_tensors,
                staging_caches=self.staging_caches,
                is_hma_required=self._is_hma_required,
                prefill_tp_size=self._prefill_tp_size,
                num_key_value_heads=self.num_key_value_heads,
                num_blocks=self.num_blocks,
                cluster_id=self.cluster_id,
            )
            self.kv_recv_thread.start()

        start_wait_time = time.time()
        thread = self.kv_send_thread if self.kv_role == "kv_producer" else self.kv_recv_thread
        assert thread is not None
        while not ready_event.is_set():
            if not thread.is_alive():
                raise RuntimeError("HIXL KV Cache sending/receiving thread failed to start.")
            if time.time() - start_wait_time > 5 * 60:
                raise RuntimeError("Timeout waiting for HIXL KV Cache thread to be ready.")
            ready_event.wait(timeout=3)

    def get_finished(self) -> tuple[set[str], set[str]]:
        done_sending = (
            self.kv_send_thread.get_and_clear_finished_requests()
            if self.kv_role == "kv_producer"
            else set()
        )
        done_recving = (
            self.kv_recv_thread.get_and_clear_finished_requests()
            if self.kv_role == "kv_consumer"
            else set()
        )
        return done_sending, done_recving

    def get_block_ids_with_load_errors(self) -> set[int]:
        if self.kv_role == "kv_consumer" and self.kv_recv_thread is not None:
            return self.kv_recv_thread.get_and_clear_invalid_block_ids()
        return set()

    def start_load_kv(self, metadata: HIXLConnectorMetadata):
        for req_id in metadata.reqs_in_batch:
            if self.kv_send_thread is not None:
                self.kv_send_thread.task_tracker.add_req_to_process(req_id)
            if self.kv_recv_thread is not None:
                self.kv_recv_thread.task_tracker.add_req_to_process(req_id)

        for req_id, meta in metadata.requests.items():
            remote_req_id = meta.remote_request_id
            prefill_tp_size = (
                meta.remote_ptp_size if meta.remote_ptp_size is not None else self._prefill_tp_size
            )
            remote_handshake_port_list, local_block_ids_list, remote_block_ids_list = (
                self._get_kv_split_metadata(req_id, meta)
            )
            group_pulls_list = self._get_group_pulls_metadata(
                req_id, remote_handshake_port_list, prefill_tp_size, meta.remote_port,
                remote_pcp_size=meta.remote_pcp_size,
                remote_dcp_size=meta.remote_dcp_size,
            )
            assert self.kv_recv_thread is not None
            # Bug 1 fix: forward remote_port_send_num for CP multi-port so P-side
            # done counting waits for all D ranks pulling the same P port.
            cp_active = meta.remote_pcp_size * meta.remote_dcp_size > 1
            remote_port_send_num = (
                self.remote_port_send_num.get(meta.remote_engine_id)
                if cp_active else None
            )
            for pcp_dcp_rank, remote_ports in enumerate(remote_handshake_port_list):
                for remote_tp_offset, remote_handshake_port in enumerate(remote_ports):
                    remote_host, remote_engine_id = self._get_remote_host_info_by_port(
                        meta.remote_port,
                        remote_handshake_port,
                        meta.remote_host,
                        meta.remote_engine_id,
                        meta.remote_multi_nodes_meta_mapping,
                    )
                    self.kv_recv_thread.add_request(
                        request_id=req_id,
                        remote_request_id=remote_req_id,
                        local_block_ids=local_block_ids_list[pcp_dcp_rank],
                        remote_block_ids=remote_block_ids_list[pcp_dcp_rank],
                        group_pulls=group_pulls_list[pcp_dcp_rank][remote_tp_offset],
                        remote_engine_id=remote_engine_id,
                        remote_host=remote_host,
                        remote_handshake_port=remote_handshake_port,
                        remote_port_send_num=remote_port_send_num,
                        num_computed_tokens=meta.num_computed_tokens,
                        all_task_done=(
                            pcp_dcp_rank == len(remote_handshake_port_list) - 1
                            and remote_tp_offset == len(remote_ports) - 1
                        ),
                        # Bug 3 fix: per-shard reformat stash (CP shards).
                        shard_idx=pcp_dcp_rank,
                    )

        if self.kv_send_thread is not None:
            for req_id, delay_start_time in metadata.requests_to_send.items():
                self.kv_send_thread.add_delayed_request(req_id, delay_start_time)

    def shutdown(self):
        try:
            shutdown_datadist()
        except Exception as e:
            logger.warning("HIXL shutdown_datadist failed: %s", e)


# ---------------------------------------------------------------------------
# ZMQ helpers (forked from Mooncake, engine-agnostic)
# ---------------------------------------------------------------------------
@contextlib.contextmanager
def zmq_ctx(socket_type: Any, addr: str) -> Iterator[zmq.Socket]:  # type: ignore
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
) -> tuple[list[list[int]], list[list[int]]]:
    """Group block ids that are contiguous in both id space and memory."""
    if not src:
        return [], []
    src_groups: list[list[int]] = []
    dst_groups: list[list[int]] = []
    cur_src: list[int] = [src[0]]
    cur_dst: list[int] = [dst[0]]
    for i in range(1, len(src)):
        if src[i] == src[i - 1] + 1 and dst[i] == dst[i - 1] + 1:
            cur_src.append(src[i])
            cur_dst.append(dst[i])
        else:
            src_groups.append(cur_src)
            dst_groups.append(cur_dst)
            cur_src = [src[i]]
            cur_dst = [dst[i]]
    src_groups.append(cur_src)
    dst_groups.append(cur_dst)
    return src_groups, dst_groups


def string_to_int64_hash(input_str):
    hashed_bytes = hashlib.sha256(input_str.encode("utf-8")).digest()
    trunked_bytes = hashed_bytes[:8]
    uint64_value = struct.unpack("<Q", trunked_bytes)[0]
    return uint64_value


def ensure_zmq_send(socket: zmq.Socket, data: bytes, path: str, max_retries: int = 3):  # type: ignore
    retries_left = max_retries
    while True:
        try:
            socket.send(data)
            return
        except zmq.ZMQError as e:  # type: ignore
            retries_left -= 1
            if retries_left > 0:
                logger.warning("Send failed. error=%s, attempts_left=%d. ", e, retries_left)
                time.sleep(0.1)
            else:
                raise RuntimeError(f"Failed to send data to {path} after {max_retries} retries: {e}")


def ensure_zmq_recv(socket: zmq.Socket, path: str, max_retries: int = 3) -> bytes:  # type: ignore
    retries_left = max_retries
    while True:
        try:
            return socket.recv()
        except zmq.ZMQError as e:  # type: ignore
            retries_left -= 1
            if retries_left > 0:
                logger.warning("Receive failed. error=%s, attempts_left=%d. ", e, retries_left)
                time.sleep(0.1)
            else:
                raise RuntimeError(f"Failed to receive data after {max_retries} retries: {e}")
