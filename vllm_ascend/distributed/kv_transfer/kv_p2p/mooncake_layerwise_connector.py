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
from collections import OrderedDict, defaultdict, deque
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import httpx
import msgspec
import numpy as np
import numpy.typing as npt
import torch
import torch_npu
import zmq
from mooncake.engine import TransferEngine  # type: ignore
from vllm.config import VllmConfig
from vllm.distributed import get_pcp_group
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorBase_V1, KVConnectorMetadata, KVConnectorRole
from vllm.distributed.parallel_state import (
    get_decode_context_model_parallel_rank,
    get_tensor_model_parallel_rank,
    get_tp_group,
    get_world_group,
)
from vllm.logger import logger
from vllm.utils.math_utils import round_down
from vllm.utils.network_utils import get_ip, make_zmq_path, make_zmq_socket
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import KVCacheConfig

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_connector import GET_META_MSG
from vllm_ascend.distributed.kv_transfer.utils.mooncake_transfer_engine import global_te
from vllm_ascend.distributed.kv_transfer.utils.utils import (
    align_memory,
    context_parallel_parameters_check,
    get_cp_group,
    get_local_remote_block_port_mappings,
    get_transfer_mappings,
    get_transfer_timeout_value,
    kv_alltoall_and_rearrange,
    parallel_info,
)
from vllm_ascend.utils import npu_stream_switch

# isort: off
if TYPE_CHECKING:
    from vllm.v1.attention.backend import AttentionMetadata  # type: ignore
    from vllm.forward_context import ForwardContext
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.request import Request
# isort: on

DONE_SENDING_MSG = b"done_sending_msg"


class MooncakeAgentMetadata(msgspec.Struct, omit_defaults=True, dict=True):
    te_rpc_port: int
    kv_caches_base_addr: list[int]


@dataclass
class ReqMeta:
    local_block_ids: list[int]
    token_ids: list[int] | None
    # Not None if layer-wise is disabled
    remote_block_ids: list[int]
    remote_engine_id: str | None
    remote_host: str | None
    remote_port: int | None
    remote_te_rpc_port: int | None
    remote_kv_caches_base_addr: list[int] | None
    metaserver: str | None
    remote_tp_size: int | None
    remote_pcp_size: int | None
    remote_dcp_size: int | None
    chunk_finish: bool = False
    prompt_len: int = 0
    trans_count: int = 0
    remote_cache_tokens: int = 0
    local_computed_tokens: int = 0
    local_transed_tokens: int = 0


@dataclass
class SendTask:
    send_request: dict[str, ReqMeta] = field(default_factory=dict)
    # pd_head_ratio == 1 use
    wait_event: torch.npu.Event | None = None
    # pd_head_ratio > 1 use
    k_cache: torch.Tensor | None = None
    v_cache: torch.Tensor | None = None
    layer_idx: int = 0
    # trans block info
    rearrange_block_ids: list[int] | None = None
    num_blocks: int | None = None
    num_tokens: int | None = None
    block_table: torch.Tensor | None = None
    block_len_tensor: torch.Tensor | None = None
    seq_start_tensor: torch.Tensor | None = None


@dataclass
class TransferMeta:
    src: list[int]
    dst: list[int]
    length: list[int]
    req_ids: list[str]


@dataclass
class SendReqInfo:
    local_block_ids: list[int]
    local_transferred_tokens: int
    local_computed_tokens: int
    request: "Request"

    def extend_local_block_ids(self, new_block_ids: list[int]) -> None:
        """extend local block ids for this step"""
        self.local_block_ids.extend(new_block_ids)

    def update_computed_tokens(self, computed_tokens: int) -> None:
        """update local computen tokens for this step"""
        self.local_computed_tokens = computed_tokens

    def update_transferred_tokens(self, transferred_tokens: int) -> None:
        """update transferred tokens for this step"""
        self.local_transferred_tokens = transferred_tokens

    def unpack(self):
        return (
            self.local_block_ids,
            self.local_transferred_tokens,
            self.local_computed_tokens,
            self.request,
        )


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


class KVCacheSendingLayerThread(threading.Thread):
    def __init__(
        self,
        engine: TransferEngine,
        total_layers: int,
        ready_event: threading.Event,
        tp_rank: int,
        pd_head_ratio: int,
        num_head_replica: int,
        kv_cache_base_addr: list[int],
        use_mla: bool,
        block_len: list[int],
        k_buffer: torch.Tensor,
        v_buffer: torch.Tensor,
        resharding_stream: torch.npu.Stream,
        callback_func: Callable[..., None] = lambda x: None,
    ):
        super().__init__(daemon=True, name="KVCacheSendingLayerThread")
        self.engine = engine
        self.tp_rank = tp_rank
        self.pd_head_ratio = pd_head_ratio
        self.num_head_replica = num_head_replica
        self.kv_caches_base_addr = kv_cache_base_addr
        self.total_layers = total_layers
        self.use_mla = use_mla
        self.use_sparse = len(block_len) == 3
        self.block_len = block_len
        self.resharding_stream = resharding_stream
        self.current_layer = -1

        self.send_queue = queue.Queue[SendTask]()
        self.k_buffer = k_buffer
        self.v_buffer = v_buffer
        self.ready_event = ready_event
        self.callback_func = callback_func

    def run(self):
        local_rank = get_world_group().local_rank
        device = torch.device(f"npu:{local_rank}")
        torch.npu.set_device(device)
        self.ready_event.set()
        while True:
            send_task = self.send_queue.get()
            self._handle_request(send_task)

    def _handle_request(self, send_task: SendTask):
        try:
            self._transfer_kv_cache(send_task)
        except Exception as e:
            logger.error(f"Failed to transfer KV cache for layer idx {send_task.layer_idx}, {e}")

    def get_transfer_meta(self, send_task: SendTask, req_id: str, req_meta: ReqMeta):
        src_list: list[str] = []
        dst_list: list[str] = []
        length_list: list[int] = []

        layer_idx = send_task.layer_idx
        remote_block_ids = req_meta.remote_block_ids
        remote_kv_base_addrs = req_meta.remote_kv_caches_base_addr
        local_kv_base_addr = self.kv_caches_base_addr
        local_block_ids = req_meta.local_block_ids

        if self.pd_head_ratio == 1:
            if self.use_sparse:
                layer_local_kv_base_addr = [
                    local_kv_base_addr[i] for i in [3 * layer_idx, 3 * layer_idx + 1, 3 * layer_idx + 2]
                ]
                layer_remote_kv_base_addr = [
                    remote_kv_base_addrs[i]  # type:ignore
                    for i in [3 * layer_idx, 3 * layer_idx + 1, 3 * layer_idx + 2]
                ]
            else:
                layer_local_kv_base_addr = [local_kv_base_addr[i] for i in [2 * layer_idx, 2 * layer_idx + 1]]
                layer_remote_kv_base_addr = [
                    remote_kv_base_addrs[i]  # type:ignore
                    for i in [2 * layer_idx, 2 * layer_idx + 1]
                ]
            grouped_remote_block_ids, grouped_local_block_ids = group_concurrent_contiguous(
                remote_block_ids, local_block_ids
            )

            block_length = len(self.block_len)
            for k, (src_layer_base_addr, dst_layer_base_addr) in enumerate(
                zip(layer_local_kv_base_addr, layer_remote_kv_base_addr)
            ):
                block_len = self.block_len[k % block_length]
                for group_remote_block_id, group_local_block_id in zip(
                    grouped_remote_block_ids, grouped_local_block_ids
                ):
                    src = src_layer_base_addr + group_local_block_id[0] * block_len
                    dst = dst_layer_base_addr + group_remote_block_id[0] * block_len
                    length = len(group_local_block_id) * block_len
                    src_list.append(src)
                    dst_list.append(dst)
                    length_list.append(length)
        else:
            rearrange_block_ids = send_task.rearrange_block_ids
            rearrange_block_dict = {
                value: index
                for index, value in enumerate(rearrange_block_ids)  # type:ignore
            }
            layer_local_kv_base_addr = [self.k_buffer.data_ptr(), self.v_buffer.data_ptr()]

            layer_remote_kv_base_addr = [
                remote_kv_base_addrs[i]  # type:ignore
                for i in [2 * layer_idx, 2 * layer_idx + 1]
            ]

            src_list, dst_list, length_list = [], [], []
            for k, (src_layer_base_addr, dst_layer_base_addr) in enumerate(
                zip(layer_local_kv_base_addr, layer_remote_kv_base_addr)
            ):
                block_len = self.block_len[0]
                remote_block_len = self.block_len[0] * self.pd_head_ratio
                for remote_block_id, local_block_id in zip(remote_block_ids, local_block_ids):
                    src = src_layer_base_addr + rearrange_block_dict[local_block_id] * block_len
                    dst = (
                        dst_layer_base_addr
                        + remote_block_id * remote_block_len
                        + block_len * ((self.tp_rank // self.num_head_replica) % self.pd_head_ratio)
                    )
                    src_list.append(src)
                    dst_list.append(dst)
                    length_list.append(block_len)
        return (src_list, dst_list, length_list)

    def _transfer_kv_cache(self, send_task: SendTask):
        if self.pd_head_ratio > 1:
            with npu_stream_switch(self.resharding_stream):
                key = send_task.k_cache
                value = send_task.v_cache
                key = key.view(-1, key.shape[-1])  # type:ignore
                value = value.view(-1, key.shape[-1])  # type:ignore
                self.k_buffer[: key.shape[0]].copy_(key)  # [:4, 128] ->
                self.v_buffer[: value.shape[0]].copy_(value)

        # Merge transmission tasks of the same session
        session_meta: dict[str, TransferMeta] = {}
        for req_id, req_meta in send_task.send_request.items():
            session_id = f"{req_meta.remote_host}:{req_meta.remote_te_rpc_port}"
            if session_id not in session_meta:
                session_meta[session_id] = TransferMeta(src=[], dst=[], length=[], req_ids=[])

            (src_list, dst_list, length_list) = self.get_transfer_meta(send_task, req_id, req_meta)

            session_meta[session_id].src.extend(src_list)
            session_meta[session_id].dst.extend(dst_list)
            session_meta[session_id].length.extend(length_list)
            session_meta[session_id].req_ids.append(req_id)

        if self.pd_head_ratio == 1:
            """
            Note: Due to a bug in ADXL, calling current_event.synchronize() may occasionally hang.
            This issue will be fixed in CANN version 8.5.rc1.
            You can manually build the master branch of the project at https://gitcode.com/cann/hixl
            to resolve this issue before the 8.5.RC1 release.
            """
            send_task.wait_event.synchronize()  # type:ignore
        elif self.pd_head_ratio > 1:
            self.resharding_stream.synchronize()

        for session_id, transfer_meta in session_meta.items():
            if len(transfer_meta.src) > 0:
                ret = self.engine.batch_transfer_sync_write(
                    session_id, transfer_meta.src, transfer_meta.dst, transfer_meta.length
                )
                if ret < 0:
                    logger.error(
                        f"Mooncake transfer failed for send requests {transfer_meta.req_ids} kv cache to {session_id}"
                    )
                    if send_task.layer_idx == (self.total_layers - 1):
                        for req_id in transfer_meta.req_ids:
                            req_meta = send_task.send_request[req_id]
                            if req_meta.chunk_finish:
                                self.callback_func(
                                    req_id, req_meta
                                )  # TODO Send a signal indicating transmission failure
                else:
                    if send_task.layer_idx == (self.total_layers - 1):
                        for req_id in transfer_meta.req_ids:
                            req_meta = send_task.send_request[req_id]
                            if req_meta.chunk_finish:
                                self.callback_func(req_id, req_meta)


class KVCacheRecvingLayerThread(threading.Thread):
    def __init__(
        self,
        tp_rank: int,
        side_channel_port: int,
        tp_size: int,
        pd_head_ratio: int,
        local_engine_id: str,
        metadata: MooncakeAgentMetadata,
        ready_event: threading.Event,
    ):
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
        self.request_map = dict[str, str]()
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

    def update_task(self, req_id, trans_count):
        with self.lock:
            self.task_tracker[req_id] += 1
            if self.task_tracker[req_id] == trans_count:
                self.task_tracker.pop(req_id)
                self.done_requests.add(self.request_map[req_id])
                self.request_map.pop(req_id)

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
                        logger.debug("Got DONE_RECVING_MSG for request %s", msg[1])
                        request_id = msg[1]
                        trans_count = msg[2]
                        self.update_task(request_id, trans_count)
                        sock.send_multipart((identity, b"", b"ACK"))
                    else:
                        logger.error("Connection listener got unexpected message %s", msg)
                except Exception as e:
                    logger.error("Failed to decode message: %s", e)


class MooncakeLayerwiseConnectorMetadata(KVConnectorMetadata):
    def __init__(self):
        self.requests: dict[str, ReqMeta] = {}
        self.send_task: SendTask = SendTask()

    def add_new_req(
        self,
        request_id: str,
        local_block_ids: list[int],
        kv_transfer_params: dict[str, Any],
        token_ids: list[int] | None = None,
        chunk_finish: bool = False,
        prompt_len: int = 0,
        remote_cache_tokens: int = 0,
        local_computed_tokens: int = 0,
        local_transed_tokens: int = 0,
    ):
        self.requests[request_id] = ReqMeta(
            token_ids=token_ids or [],
            local_block_ids=local_block_ids,
            remote_block_ids=kv_transfer_params.get("remote_block_ids", []),
            remote_engine_id=kv_transfer_params.get("remote_engine_id"),
            remote_host=kv_transfer_params.get("remote_host"),
            remote_port=kv_transfer_params.get("remote_port"),
            remote_te_rpc_port=kv_transfer_params.get("remote_te_rpc_port"),
            remote_kv_caches_base_addr=kv_transfer_params.get("remote_kv_caches_base_addr"),
            metaserver=kv_transfer_params.get("metaserver"),
            remote_tp_size=kv_transfer_params.get("remote_tp_size"),
            remote_pcp_size=kv_transfer_params.get("remote_pcp_size"),
            remote_dcp_size=kv_transfer_params.get("remote_dcp_size"),
            chunk_finish=chunk_finish,
            remote_cache_tokens=remote_cache_tokens,
            local_computed_tokens=local_computed_tokens,
            prompt_len=prompt_len,
            local_transed_tokens=local_transed_tokens,
        )


class MooncakeLayerwiseConnector(KVConnectorBase_V1):
    def __init__(self, vllm_config: VllmConfig, role: KVConnectorRole, kv_cache_config: KVCacheConfig | None = None):
        super().__init__(vllm_config, role, kv_cache_config)
        assert vllm_config.kv_transfer_config is not None
        self.engine_id = vllm_config.kv_transfer_config.engine_id
        self._connector_metadata = MooncakeLayerwiseConnectorMetadata()

        if role == KVConnectorRole.SCHEDULER:
            self.connector_scheduler: MooncakeLayerwiseConnectorScheduler | None = MooncakeLayerwiseConnectorScheduler(
                vllm_config, str(self.engine_id)
            )
            self.connector_worker: MooncakeLayerwiseConnectorWorker | None = None
        elif role == KVConnectorRole.WORKER:
            self.connector_scheduler = None
            self.connector_worker = MooncakeLayerwiseConnectorWorker(vllm_config, str(self.engine_id))

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
        assert isinstance(self._connector_metadata, MooncakeLayerwiseConnectorMetadata)
        self.connector_worker.start_load_kv(self._connector_metadata)

    def wait_for_layer_load(self, layer_name: str) -> None:
        """MooncakeLayerwiseConnector does not do layerwise saving."""
        assert self.connector_worker is not None
        assert isinstance(self._connector_metadata, MooncakeLayerwiseConnectorMetadata)
        self.connector_worker.wait_for_layer_load(layer_name)

    def save_kv_layer(
        self, layer_name: str, kv_layer: torch.Tensor, attn_metadata: "AttentionMetadata", **kwargs
    ) -> None:
        """MooncakeLayerwiseConnector does not save explicitly."""
        assert self.connector_worker is not None
        assert isinstance(self._connector_metadata, MooncakeLayerwiseConnectorMetadata)
        self.connector_worker.save_kv_layer(layer_name, kv_layer, attn_metadata, self._connector_metadata)

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

        # Handshake base port
        self.side_channel_port = (
            vllm_config.kv_transfer_config.kv_port
            + vllm_config.parallel_config.data_parallel_rank * vllm_config.parallel_config.tensor_parallel_size
        )

        # Requests that need to start recv.
        # New requests are added by update_state_after_alloc in
        # the scheduler. Used to make metadata passed to Worker.
        self._reqs_need_recv: dict[str, tuple[Request, list[int], list[int]]] = {}
        self._reqs_need_send_layerwise: dict[str, SendReqInfo] = {}
        self.executor = ThreadPoolExecutor(32)
        tls_config: dict[str, Any] = vllm_config.kv_transfer_config.get_from_extra_config("tls_config", {})
        ssl_keyfile = tls_config.get("ssl_keyfile")
        ssl_certfile = tls_config.get("ssl_certfile")
        ssl_ca_certs = tls_config.get("ssl_ca_certs", False)
        ssl_keyfile_password = tls_config.get("ssl_keyfile_password")
        self.cert_path = (ssl_certfile, ssl_keyfile, ssl_keyfile_password)
        self.ssl_enable = tls_config.get("ssl_enable", False)
        self.ca_path = ssl_ca_certs
        if self.ssl_enable:
            self.metaserver_client = httpx.Client(
                limits=httpx.Limits(max_connections=100000), timeout=None, cert=self.cert_path, verify=self.ca_path
            )
        else:
            self.metaserver_client = httpx.Client(limits=httpx.Limits(max_connections=100000), timeout=None)

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
            "MooncakeLayerwiseConnector get_num_new_matched_tokens: num_computed_tokens=%s, kv_transfer_params=%s",
            num_computed_tokens,
            params,
        )

        if params is not None and params.get("do_remote_prefill"):
            # Remote prefill: get all prompt blocks from remote.
            assert num_computed_tokens % self.block_size == 0
            # Note: We use the full token count as transmit data here.
            count = max(len(request.prompt_token_ids) - num_computed_tokens, 0)
            return count, count > 0

        # No remote prefill for this request.
        return 0, False

    def update_state_after_alloc(self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int):
        params = request.kv_transfer_params
        logger.debug(
            "MooncakeLayerwiseConnector update_state_after_alloc: num_external_tokens=%s, kv_transfer_params=%s",
            num_external_tokens,
            params,
        )

        if params is not None and params.get("do_remote_prefill"):
            local_block_ids = (blocks.get_block_ids()[0]) if num_external_tokens > 0 else []
            remote_cached_tokens = request.num_computed_tokens
            # Get unhashed blocks to pull from remote.
            logger.debug(
                f"MooncakeLayerwiseConnector update_state_after_alloc: add {request.request_id} to need recv queue"
            )
            self._reqs_need_recv[request.request_id] = (
                request,
                [],  # request._all_token_ids,
                local_block_ids,
            )

            params["do_remote_prefill"] = False

            logger.info(f"Send request: {request.request_id} to proxy metaserver: {params.get('metaserver', None)}")
            # All parameters here should appear in the returned dict of
            # request_finished in the scheduler side except "request_id".
            # change the format of request_id if vllm-version >= 0.14.0
            external_req_id = get_external_request_id(request.request_id)
            kv_transfer_params = dict(
                token_ids=[],
                request_id=external_req_id,
                do_remote_prefill=False,
                do_remote_decode=True,
                remote_block_ids=local_block_ids,
                remote_engine_id=self.engine_id,
                remote_host=self.side_channel_host,
                remote_port=self.side_channel_port,
                remote_tp_size=self.vllm_config.parallel_config.tensor_parallel_size,
                remote_pcp_size=self.vllm_config.parallel_config.prefill_context_parallel_size,
                remote_dcp_size=self.vllm_config.parallel_config.decode_context_parallel_size,
                remote_cached_tokens=remote_cached_tokens,
            )

            future = self.executor.submit(
                self._access_metaserver, url=params.get("metaserver", None), message=kv_transfer_params
            )

            def handle_exception(future):
                if future.exception():
                    logger.error(f"Access metaserver fail: {future.exception()}")

            future.add_done_callback(handle_exception)

        # Layerwise prefiller add request need send
        if params is not None and params.get("do_remote_decode"):
            local_block_ids = blocks.get_block_ids()[0]
            logger.debug(
                f"MooncakeLayerwiseConnector update_state_after_alloc: add {request.request_id} to need send queue"
            )
            remote_block_ids = copy.deepcopy(params["remote_block_ids"])
            remote_cache_tokens = (
                (len(request.all_token_ids) + self.block_size - 1) // self.block_size - len(remote_block_ids)
            ) * self.block_size
            local_transferred_tokens = remote_cache_tokens
            local_computed_tokens = 0
            self._reqs_need_send_layerwise[request.request_id] = SendReqInfo(
                local_block_ids=local_block_ids,
                local_transferred_tokens=local_transferred_tokens,
                local_computed_tokens=local_computed_tokens,
                request=request,
            )

    def build_connector_meta(
        self,
        scheduler_output: SchedulerOutput,
    ) -> KVConnectorMetadata:
        meta = MooncakeLayerwiseConnectorMetadata()

        if self.vllm_config.kv_transfer_config.is_kv_consumer:
            # Loop through scheduled reqs and convert to ReqMeta.
            for req_id, (req, token_ids, block_ids) in self._reqs_need_recv.items():
                assert req.kv_transfer_params is not None
                # For the case where there are no remote blocks to pull
                # (block_ids is empty), we don't need to schedule
                # an async read on the worker side.
                meta.add_new_req(
                    request_id=req_id,
                    local_block_ids=block_ids,
                    kv_transfer_params=req.kv_transfer_params,
                    token_ids=token_ids,
                )

            # Clear the list once workers start the transfers
            self._reqs_need_recv.clear()
        else:
            cached_reqs = scheduler_output.scheduled_cached_reqs
            new_reqs = scheduler_output.scheduled_new_reqs
            scheduled_spec_decode_tokens = scheduler_output.scheduled_spec_decode_tokens
            for req_id, new_blocks in zip(cached_reqs.req_ids, cached_reqs.new_block_ids):
                if req_id in self._reqs_need_send_layerwise and new_blocks is not None:
                    self._reqs_need_send_layerwise[req_id].extend_local_block_ids(new_blocks[0])
            computed_tokens = dict(
                list(zip(cached_reqs.req_ids, cached_reqs.num_computed_tokens))
                + [(x.req_id, x.num_computed_tokens) for x in new_reqs]
            )
            for req_id, scheduled_tokens in scheduler_output.num_scheduled_tokens.items():
                if req_id in self._reqs_need_send_layerwise:
                    send_req_info = self._reqs_need_send_layerwise[req_id]
                    # update local transferred tokens
                    send_req_info.update_transferred_tokens(
                        round_down(send_req_info.local_computed_tokens, self.block_size)
                    )
                    # update local computed tokens, not transfer spec decode tokens
                    spec_decode_tokens = (
                        len(scheduled_spec_decode_tokens[req_id]) if (req_id in scheduled_spec_decode_tokens) else 0
                    )
                    send_req_info.update_computed_tokens(
                        computed_tokens.get(req_id, 0) + scheduled_tokens - spec_decode_tokens
                    )

                    def add_tranfer_task(req_id, send_req_info: SendReqInfo, chunk_finish=False):
                        (
                            local_block_ids,
                            local_transed_tokens,
                            local_computed_tokens,
                            request,
                        ) = send_req_info.unpack()
                        meta.add_new_req(
                            request_id=req_id,
                            local_block_ids=local_block_ids,
                            kv_transfer_params=request.kv_transfer_params,
                            token_ids=[],
                            chunk_finish=chunk_finish,
                            remote_cache_tokens=request.kv_transfer_params.get("remote_cached_tokens"),
                            prompt_len=len(request.all_token_ids),
                            local_computed_tokens=local_computed_tokens,
                            local_transed_tokens=local_transed_tokens,
                        )
                        logger.debug(
                            f"MooncakeLayerwiseConnector build_connector_meta: {req_id=}"
                            f"prompt_len={len(request.all_token_ids)} {local_computed_tokens=}"
                            f"{local_transed_tokens=}"
                            f"remote_cache_tokens={request.kv_transfer_params.get('remote_cached_tokens')}"
                            f"{chunk_finish=} {local_block_ids=}"
                            f"remote_block_ids={request.kv_transfer_params.get('remote_block_ids')}"
                        )

                    # whether chunk finish
                    chunk_finish = send_req_info.local_computed_tokens >= len(send_req_info.request.all_token_ids)

                    add_tranfer_task(req_id, send_req_info, chunk_finish=chunk_finish)
                    if chunk_finish:
                        self._reqs_need_send_layerwise.pop(req_id)
        return meta

    def _access_metaserver(self, url, message):
        success = False
        retry = 0
        while retry < 3 and success is False:
            retry += 1
            try:
                self.metaserver_client.post(url, json=message)
                success = True
            except Exception as e:
                logger.error(f"Failed to connect to metaserver: {url}, retry {retry} time.")
                if retry == 3:
                    raise e

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, dict[str, Any] | None]:
        """
        Once a request is finished, determine whether request blocks
        should be freed now or will be sent asynchronously and freed later.
        """
        # layer_wise push, not need delay_free_blocks
        return False, None


class MooncakeLayerwiseConnectorWorker:
    """Implementation of Worker side methods"""

    def __init__(self, vllm_config: VllmConfig, engine_id: str):
        os.environ["ASCEND_TRANSFER_TIMEOUT"] = str(get_transfer_timeout_value())

        if TransferEngine is None:
            raise RuntimeError("mooncake is not available")
        logger.info("Initializing Mooncake work %s", engine_id)

        # Metadata.
        self.vllm_config = vllm_config
        self.local_engine_id: str = " "
        self.engine_id = engine_id
        self.tp_rank = get_tensor_model_parallel_rank()
        self.tp_size = vllm_config.parallel_config.tensor_parallel_size
        self.pcp_size = vllm_config.parallel_config.prefill_context_parallel_size
        self.pcp_rank = get_pcp_group().rank_in_group if self.pcp_size > 1 else 0
        self.dcp_size = vllm_config.parallel_config.decode_context_parallel_size
        self.dcp_rank = get_decode_context_model_parallel_rank() if self.dcp_size > 1 else 0
        self.tp_group = get_tp_group()
        self._decode_tp_size: int | None = None
        self.kv_caches: dict[str, torch.Tensor] = {}
        self.side_channel_host = get_ip()
        self.total_layers = vllm_config.model_config.get_num_layers(vllm_config.parallel_config)
        self.use_mla = self.vllm_config.model_config.use_mla
        if self.use_mla:
            self.total_num_kv_heads = 1
        else:
            self.total_num_kv_heads = self.vllm_config.model_config.get_total_num_kv_heads()

        # Handshake base port
        self.side_channel_port = (
            vllm_config.kv_transfer_config.kv_port
            + vllm_config.parallel_config.data_parallel_rank * vllm_config.parallel_config.tensor_parallel_size
        )
        self.handshake_port = self.side_channel_port + self.tp_rank
        self.sockets: dict = {}
        logger.info("Initializing Mooncake work %s", engine_id)
        self.engine = global_te.get_transfer_engine(self.side_channel_host, device_name=None)
        self.te_rpc_port = self.engine.get_rpc_port()

        # Background thread for sending or receiving KV caches.
        self.kv_recv_layer_thread: KVCacheRecvingLayerThread | None = None
        self.kv_send_layer_thread: KVCacheSendingLayerThread | None = None

        self.vllm_config = vllm_config
        self.block_size = vllm_config.cache_config.block_size
        self.kv_caches_base_addr: list[int] = []

        self.pd_tp_ratio = get_ascend_config().pd_tp_ratio
        self.pd_head_ratio = get_ascend_config().pd_head_ratio
        self.num_head_replica = get_ascend_config().num_head_replica
        self.resharding_stream = None
        if self.pd_head_ratio > 1:
            self.resharding_stream = torch.npu.Stream()

        self.remote_poller = zmq.Poller()  # type: ignore
        self.decoder = msgspec.msgpack.Decoder(MooncakeAgentMetadata)
        self.encoder = msgspec.msgpack.Encoder()

        self.remote_kv_caches_base_addr: dict[str, dict[int, list[int]]] = SizedDict()
        self.remote_te_port: dict[str, dict[int, int]] = SizedDict()
        self.remote_sockets_lock = threading.Lock()
        self.remote_sockets: dict[  # type: ignore
            str, deque[zmq.Socket]
        ] = defaultdict(  # type: ignore
            deque
        )
        self.remote_poller = zmq.Poller()  # type: ignore
        self.timeout = 1.0  # seconds
        self.k_buffer: torch.Tensor | None = None
        self.v_buffer: torch.Tensor | None = None

    def create_kv_buffer(self, first_kv_cache):
        if self.pd_head_ratio > 1:
            # regesit kv buffer for tp inequal
            alignment = 2 * 1024 * 1024
            self.k_buffer = torch.zeros(
                first_kv_cache.numel() + alignment, dtype=first_kv_cache.dtype, device=first_kv_cache.device
            )
            self.k_buffer = align_memory(self.k_buffer, alignment)[: first_kv_cache.numel()].view(
                -1, first_kv_cache.shape[-1]
            )
            self.v_buffer = torch.zeros(
                first_kv_cache.numel() + alignment, dtype=first_kv_cache.dtype, device=first_kv_cache.device
            )
            self.v_buffer = align_memory(self.v_buffer, alignment)[: first_kv_cache.numel()].view(
                -1, first_kv_cache.shape[-1]
            )

            for tensor in (self.k_buffer, self.v_buffer):
                assert tensor.data_ptr() % alignment == 0, (
                    "The address of the registered kv cache should be aligned to 2M"
                )
                ret_value = self.engine.register_memory(tensor.data_ptr(), tensor.numel())
                logger.info(
                    f"Register memory for prefill when pd head ratio > 1 {tensor.data_ptr()} "
                    f"{tensor.numel()} {ret_value=}"
                )
                if ret_value != 0:
                    raise RuntimeError("Mooncake memory registration failed. ")

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        """Register the KV Cache data."""

        _, first_kv_cache_tuple = next(iter(kv_caches.items()))
        first_kv_cache = first_kv_cache_tuple[0]
        self.create_kv_buffer(first_kv_cache)

        # TODO(tms): Find a more robust way to detect and handle MLA
        self.use_mla = (
            first_kv_cache_tuple[0].size(-1) != first_kv_cache_tuple[1].size(-1) and len(first_kv_cache_tuple) == 2
        )
        self.use_sparse = len(first_kv_cache_tuple) == 3

        self.num_blocks = first_kv_cache.shape[0]
        logger.info("num_blocks: %s", self.num_blocks)
        block_rank = 3
        self.block_len = []
        if self.use_mla or self.use_sparse:
            for i in range(len(first_kv_cache_tuple)):
                block_shape = first_kv_cache_tuple[i].shape[-block_rank:]
                logger.info("block_shape: %s", block_shape)
                self.block_len.append(first_kv_cache[i].element_size() * math.prod(block_shape))
        else:
            # [num_block, block_size, num_head, hidden_dim]
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
        global_te.register_buffer(ptrs, lengths)
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
                k_buffer=self.k_buffer,
                v_buffer=self.v_buffer,
                resharding_stream=self.resharding_stream,
                callback_func=self.send_done_send_signal,
            )
            self.kv_send_layer_thread.start()
            ready_event.wait()

        if self.vllm_config.kv_transfer_config.is_kv_consumer:
            ready_event = threading.Event()
            self.kv_recv_layer_thread = KVCacheRecvingLayerThread(
                self.tp_rank,
                self.side_channel_port,
                self.tp_size,
                self.pd_head_ratio,
                self.engine_id,
                metadata,
                ready_event,
            )
            self.kv_recv_layer_thread.start()
            ready_event.wait()

    def get_finished(self) -> tuple[set[str], set[str]]:
        done_recving = (
            self.kv_recv_layer_thread.get_and_clear_finished_requests(  # type: ignore[union-attr]
            )
            if self.vllm_config.kv_transfer_config.is_kv_consumer
            else set()
        )
        if len(done_recving) > 0:
            logger.info(
                f"Number of completed KV cache recv requests: {len(done_recving)}, receive requests: {done_recving}"
            )
        return set(), done_recving

    # {(ip, port)]: {local_block_ids: [], remote_block_ids: {}}}
    def _get_kv_split_metadata(self, req_meta, req_idx, req_id):
        remote_pcp_size = req_meta.remote_pcp_size
        remote_dcp_size = req_meta.remote_dcp_size
        remote_tp_size = req_meta.remote_tp_size
        remote_hosts = [req_meta.remote_host]
        remote_port = req_meta.remote_port
        local_transed_tokens = max(req_meta.remote_cache_tokens, req_meta.local_transed_tokens)
        # local_transed_tokens tokens that have already been transmitted on the local side
        local_computed_tokens = req_meta.local_computed_tokens
        prompt_len = req_meta.prompt_len
        p_parallel_info = parallel_info(
            tp_size=self.tp_size,
            pcp_size=self.pcp_size,
            dcp_size=self.dcp_size,
            pd_head_ratio=self.pd_head_ratio,
            use_mla=self.use_mla,
        )
        d_parallel_info = parallel_info(
            tp_size=remote_tp_size,
            pcp_size=remote_pcp_size,
            dcp_size=remote_dcp_size,
            pd_head_ratio=self.pd_head_ratio,
            use_mla=self.use_mla,
        )
        cp_size = self.pcp_size * self.dcp_size
        # to_trans_idx all tokens that have been processed up to the current step
        if req_meta.chunk_finish:
            to_trans_idx = math.ceil(local_computed_tokens / self.block_size)
        else:
            to_trans_idx = math.floor(local_computed_tokens / self.block_size)
        prompt_block_size = math.ceil(prompt_len / self.block_size)
        #
        num_local_blocks = prompt_block_size // cp_size + int(
            (prompt_block_size % cp_size) > (self.pcp_rank * self.dcp_size + self.dcp_rank)
        )
        already_send_blocks = to_trans_idx // cp_size + int(
            (to_trans_idx % cp_size) > (self.pcp_rank * self.dcp_size + self.dcp_rank)
        )
        if num_local_blocks == already_send_blocks:
            req_meta.chunk_finish = True
        transed_idx = math.floor(local_transed_tokens / self.block_size)

        p_cp_group = get_cp_group(self.tp_size, self.total_num_kv_heads, self.dcp_size)
        d_cp_group = get_cp_group(remote_tp_size, self.total_num_kv_heads, remote_dcp_size)
        logger.debug(f"Compute cp group for P&D {req_id=} {p_cp_group=} {d_cp_group=}")

        cp_ratio = len(p_cp_group) // len(d_cp_group)
        if cp_ratio == 0:
            selected_p_cp_groups = p_cp_group
            selected_d_cp_groups = d_cp_group
        else:
            x = req_idx % cp_ratio
            start = x * len(d_cp_group)
            selected_p_cp_groups = p_cp_group[start : (start + len(d_cp_group))]
            selected_d_cp_groups = d_cp_group
        assert len(selected_p_cp_groups) == len(selected_d_cp_groups)

        p_head_group_rank = (self.tp_rank - self.dcp_rank) // self.dcp_size
        selected_p_cp_group = []
        selected_d_cp_group = []
        for idx, cp_group in enumerate(selected_p_cp_groups):
            if p_head_group_rank in cp_group:  # Check whether the rank is in selected_p_cp_groups
                selected_p_cp_group = cp_group
                selected_d_cp_group = selected_d_cp_groups[idx]
        if len(selected_p_cp_group) == 0:
            return {}

        logger.debug(
            f"MooncakeLayerwiseConnector _get_kv_split_metadata {req_id=} "
            f"P-side selected head_group cp group: {selected_p_cp_group}, "
            f"D-side selected head_group cp group: {selected_d_cp_group}"
        )

        context_parallel_parameters_check(
            remote_pcp_size, remote_dcp_size, p_parallel_info, d_parallel_info, self.total_num_kv_heads
        )
        p_rank_block_mapping, d_block_rank_mapping, pd_head_mapping, d_trans_count_mapping = (
            get_local_remote_block_port_mappings(
                to_trans_idx,
                p_parallel_info,
                d_parallel_info,
                remote_hosts,
                remote_port,
                selected_p_cp_group,
                selected_d_cp_group,
                prompt_len,
                self.block_size,
                req_meta,
                self.total_num_kv_heads,
                req_id,
            )
        )
        transfer_mappings = get_transfer_mappings(
            p_rank_block_mapping,
            d_block_rank_mapping,
            pd_head_mapping,
            d_trans_count_mapping,
            req_meta,
            p_parallel_info,
            req_id,
            transed_idx,
            to_trans_idx,
            self.tp_rank,
            self.pcp_rank,
            self.dcp_rank,
        )
        return transfer_mappings

    def start_load_kv(self, metadata: MooncakeLayerwiseConnectorMetadata):
        """Start loading KV blocks from remote engine."""
        self.current_layer = 0
        if self.vllm_config.kv_transfer_config.is_kv_consumer:
            for req_id, meta in metadata.requests.items():
                external_req_id = get_external_request_id(req_id)
                assert self.kv_recv_layer_thread is not None
                with self.kv_recv_layer_thread.lock:
                    self.kv_recv_layer_thread.task_tracker[external_req_id] = 0
                    self.kv_recv_layer_thread.request_map[external_req_id] = req_id
        elif self.vllm_config.kv_transfer_config.is_kv_producer:
            # update trans info
            update_metadata = {}
            for req_idx, (req_id, req_meta) in enumerate(metadata.requests.items()):
                self._decode_tp_size = req_meta.remote_tp_size
                transfer_mappings = self._get_kv_split_metadata(req_meta, req_idx, req_id)
                assert len(transfer_mappings) <= 1, f"Not support add mutil transfer task for req_id:{req_id}"
                update_req_meta = copy.deepcopy(req_meta)
                for (host, port), block_dict in transfer_mappings.items():
                    update_req_meta.remote_host = host
                    update_req_meta.remote_port = port
                    update_req_meta.local_block_ids = block_dict["local_block_ids"]
                    update_req_meta.remote_block_ids = block_dict["remote_block_ids"]
                    update_req_meta.trans_count = block_dict["trans_count"]
                    update_metadata[req_id] = update_req_meta
            metadata.requests = {}
            for req_id, req_meta in update_metadata.items():
                metadata.requests[req_id] = update_metadata[req_id]

            # update send task trans block info
            if self.pd_head_ratio != 1:
                send_task = metadata.send_task
                send_task.rearrange_block_ids = sorted(
                    {block_id for req_id in metadata.requests for block_id in metadata.requests[req_id].local_block_ids}
                )

                device = self.k_buffer.device  # type: ignore
                flat_block_ids = send_task.rearrange_block_ids
                block_ids_tensor = torch.tensor(flat_block_ids, dtype=torch.int32, device=device)
                send_task.num_blocks = len(flat_block_ids)
                send_task.num_tokens = send_task.num_blocks * self.block_size

                send_task.block_table = block_ids_tensor.view(1, -1)
                send_task.block_len_tensor = torch.tensor([send_task.num_tokens], dtype=torch.int32, device=device)
                send_task.seq_start_tensor = torch.tensor([0], dtype=torch.int32, device=device)

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: tuple[torch.Tensor, torch.Tensor],
        attn_metadata: "AttentionMetadata",
        connector_metadata: MooncakeLayerwiseConnectorMetadata,
        **kwargs,
    ) -> None:
        """MooncakeLayerwiseConnector does not save explicitly."""
        if self.vllm_config.kv_transfer_config.is_kv_producer and connector_metadata.requests.keys():
            # get reshape and cache event
            if self.use_mla or self.use_sparse:
                reshape_cache_event = attn_metadata[layer_name].reshape_cache_event
            else:
                reshape_cache_event = attn_metadata.reshape_cache_event

            send_task = connector_metadata.send_task
            if self.pd_head_ratio != 1:
                assert self.resharding_stream is not None
                with npu_stream_switch(self.resharding_stream):
                    reshape_cache_event.wait()
                    dtype = self.k_buffer.dtype  # type: ignore
                    device = self.k_buffer.device  # type: ignore
                    # Initialize buffers
                    keys = torch.empty((send_task.num_tokens, *kv_layer[0].size()[-2:]), dtype=dtype, device=device)
                    values = torch.empty((send_task.num_tokens, *kv_layer[1].size()[-2:]), dtype=dtype, device=device)

                    # Load cache data into buffers
                    torch_npu.atb.npu_paged_cache_load(
                        kv_layer[0],
                        kv_layer[1],
                        send_task.block_table,
                        send_task.block_len_tensor,
                        seq_starts=send_task.seq_start_tensor,
                        key=keys,
                        value=values,
                    )

                    # sort kv caches for each block
                    keys = (
                        keys.view(send_task.num_blocks, self.pd_head_ratio, -1, *keys.shape[1:])
                        .transpose(0, 1)
                        .reshape_as(keys)
                    )
                    values = (
                        values.view(send_task.num_blocks, self.pd_head_ratio, -1, *values.shape[1:])
                        .transpose(0, 1)
                        .reshape_as(values)
                    )
                    # reshard kv cache
                    keys = keys.reshape(-1, *kv_layer[0].shape[2:])
                    values = values.reshape(-1, *kv_layer[1].shape[2:])
                    (keys, values) = kv_alltoall_and_rearrange(self.pd_head_ratio, keys, values)
            else:
                keys = None
                values = None

            assert self.kv_send_layer_thread is not None
            assert reshape_cache_event is not None
            layer_send_task = SendTask(
                wait_event=reshape_cache_event,
                k_cache=keys,
                v_cache=values,
                layer_idx=self.current_layer,
                rearrange_block_ids=send_task.rearrange_block_ids,
            )
            for req_id, req_meta in connector_metadata.requests.items():
                req_meta_update = self.update_decoder_info(req_id, req_meta)
                logger.debug(f"Add request {req_id} to kv send layer thread. {req_meta_update=}")
                layer_send_task.send_request[req_id] = req_meta_update

            self.kv_send_layer_thread.send_queue.put(layer_send_task)
            self.current_layer += 1

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

    def update_decoder_info(self, req_id, req_meta):
        if (
            req_meta.remote_engine_id not in self.remote_kv_caches_base_addr
            or req_meta.remote_port not in self.remote_kv_caches_base_addr[req_meta.remote_engine_id]
        ):
            try:
                encoded_data = self.encoder.encode((GET_META_MSG, req_id))
                sock = self._get_remote_socket(req_meta.remote_host, req_meta.remote_port)
                path = f"{req_meta.remote_host}:{req_meta.remote_port}"
                ensure_zmq_send(sock, encoded_data, path)
                metadata_bytes = ensure_zmq_recv(sock, self.remote_poller, path)
                agent_meta = self.decoder.decode(metadata_bytes)
            except Exception as e:
                logger.error(
                    f"Query to port and kv base addr for request {req_id}"
                    f"from {req_meta.remote_host}:{req_meta.remote_port}"
                    f"fail with error: {e}"
                )
            assert req_meta.remote_engine_id != self.engine_id, (
                f"Conflict engine id {req_meta.remote_engine_id} with local engine id {self.local_engine_id}."
            )
            self.remote_kv_caches_base_addr[req_meta.remote_engine_id][req_meta.remote_port] = (
                agent_meta.kv_caches_base_addr
            )
            self.remote_te_port[req_meta.remote_engine_id][req_meta.remote_port] = agent_meta.te_rpc_port
            logger.info(
                f"Query to port and kv base addr for request {req_id}"
                f"from {req_meta.remote_host}:{req_meta.remote_port}"
                f"success {agent_meta.kv_caches_base_addr=} {agent_meta.te_rpc_port=}"
            )
            if self.pd_head_ratio > 1:
                # for tp inequal, pre-create link to prevent alltoall out of memory
                session_id = f"{req_meta.remote_host}:{agent_meta.te_rpc_port}"
                ret = self.engine.batch_transfer_sync_write(
                    session_id, [self.kv_caches_base_addr[0]], [agent_meta.kv_caches_base_addr[0]], [128]
                )
                if ret < 0:
                    logger.error(f"Mooncake transfer failed to create link to device {session_id}")
        req_meta.remote_te_rpc_port = self.remote_te_port[req_meta.remote_engine_id][req_meta.remote_port]
        req_meta.remote_kv_caches_base_addr = self.remote_kv_caches_base_addr[req_meta.remote_engine_id][
            req_meta.remote_port
        ]
        return req_meta

    def send_done_send_signal(self, req_id, req_meta):
        external_req_id = get_external_request_id(req_id)
        logger.info(
            "Sending done sending signal for request %s to %s:%d",
            external_req_id,
            req_meta.remote_host,
            req_meta.remote_port,
        )
        try:
            path = make_zmq_path("tcp", req_meta.remote_host, req_meta.remote_port)
            msg_encoder = msgspec.msgpack.Encoder()
            encoded_data = msg_encoder.encode((DONE_SENDING_MSG, external_req_id, req_meta.trans_count))
            with zmq_ctx(zmq.REQ, path) as sock:  # type: ignore
                ensure_zmq_send(sock, encoded_data, f"{req_meta.remote_host}:{req_meta.remote_port}")
                ack = sock.recv()
                if ack != b"ACK":
                    raise ValueError(f"Unexpected ACK response: {ack}")
        except Exception as e:
            logger.error(
                f"Sending done sending signal for request {external_req_id} to "
                f"{req_meta.remote_host}:{req_meta.remote_port} fail with error: {e}"
            )

    def wait_for_layer_load(self, layer_name: str) -> None:
        pass


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
    src: list[int], dst: list[int] | None = None
) -> tuple[list[npt.NDArray[np.int64]], list[npt.NDArray[np.int64]]]:
    """Vectorised NumPy implementation."""
    if dst is None:
        dst = []
    if not dst:
        src_only_indices: npt.NDArray[np.int64] = np.array(src, dtype=np.int64)

        if src_only_indices.size == 0:
            return [], []

        brk = np.where(np.diff(src_only_indices) != 1)[0] + 1
        src_groups = np.split(src_only_indices, brk)
        src_groups = [g.tolist() for g in src_groups]

        return src_groups, []

    else:
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
                logger.warning(f"Send failed: {e}, retrying... ({retries_left} attempts left)")
                time.sleep(0.1)
            else:
                logger.error(f"Send failed after all retries: {e}")
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
                logger.warning(f"Receive failed: {e}, retrying... ({retries_left} attempts left)")
                time.sleep(0.1)
            else:
                logger.error(f"Receive failed after all retries: {e}")
                raise RuntimeError(f"Failed to receive data from {path} after {max_retries} retries: {e}")


def get_external_request_id(request_id: str):
    # NOTE(zxr): vLLM PR #27987 add additional suffix
    # to EngineCore request_id with len(suffix) == 9
    return request_id[:-9]
