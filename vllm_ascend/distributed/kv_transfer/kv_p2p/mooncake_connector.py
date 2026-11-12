# SPDX-License-Identifier: Apache-2.0
import copy
import logging
import math
import os
import queue
import random
import threading
import time
from collections import OrderedDict, defaultdict, deque
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any

import msgspec
import numpy as np
import torch
import torch_npu  # noqa: F401  (entry patch surface used by existing tests)
import zmq
from mooncake.engine import TransferEngine  # type: ignore
from vllm.config import VllmConfig
from vllm.distributed import get_pcp_group
from vllm.distributed.kv_transfer.kv_connector.utils import BlockIds
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorHandshakeMetadata,
    KVConnectorMetadata,
    KVConnectorRole,
    KVConnectorTransferResults,
    SupportsHMA,
)
from vllm.distributed.parallel_state import (
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    get_tp_group,
)
from vllm.distributed.utils import get_pp_indices
from vllm.logger import logger

# Kept as the module attribute patched by the existing mamba-transfer tests; the
# reformat mixin resolves this helper through this module at call time.
from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first  # noqa: F401
from vllm.utils.network_utils import get_ip, make_zmq_path
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    MambaSpec,
    MLAAttentionSpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.request import RequestStatus

from vllm_ascend.ascend_config import get_ascend_config, init_ascend_config
from vllm_ascend.core.kv_cache_interface import AscendSFAIndexerCacheSpec, AscendSlidingWindowMLASpec
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_v1.common.block_selection import (
    GroupTransferInfo,
    build_group_transfer_info,
    group_unique_specs,
    state_prefill_token_count,
    swa_transfer_block_ids,
    transfer_block_ids,
    truncate_request_for_prefill,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_v1.common.endpoints import (
    pull_base_port,
    pull_handshake_port,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_v1.common.protocol import (
    DONE_RECVING_MSG,
    GET_META_MSG,
    PullAgentMetadata,
    RemotePortInfo,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_v1.common.tracker import KVCacheTaskTracker
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_v1.common.transport import (
    RemoteSocketPool,
    SizedDict,
    ensure_zmq_recv,
    ensure_zmq_send,
    group_concurrent_contiguous,
    split_if_not_byte_contiguous,
    string_to_int64_hash,
    zmq_ctx,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_v1.pull import reformat as _pull_reformat
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_v1.pull.metadata import (
    GroupPull,
    MooncakeConnectorMetadata,
    ReqMeta,
    TransferRequest,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_v1.pull.reformat import KvCacheReformatMixin
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_v1.pull.transfer_plan import (
    context_parallel_parameters_check as pull_context_parallel_parameters_check,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_v1.pull.transfer_plan import (
    cp_group_meta as pull_cp_group_meta,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_v1.pull.transfer_plan import (
    dcp_shard_pulls as pull_dcp_shard_pulls,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_v1.pull.transfer_plan import (
    group_kernel_params as pull_group_kernel_params,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_v1.pull.transfer_plan import (
    group_pulls_metadata as pull_group_pulls_metadata,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_v1.pull.transfer_plan import (
    hybrid_remote_rank_group_pulls as pull_hybrid_remote_rank_group_pulls,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_v1.pull.transfer_plan import (
    kernel_block_scale as pull_kernel_block_scale,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_v1.pull.transfer_plan import (
    kv_head_groups as pull_kv_head_groups,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_v1.pull.transfer_plan import (
    local_remote_block_port_mappings as pull_local_remote_block_port_mappings,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_v1.pull.transfer_plan import (
    local_remote_cp_params as pull_local_remote_cp_params,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_v1.pull.transfer_plan import (
    remote_host_info_by_port as pull_remote_host_info_by_port,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_v1.pull.transfer_plan import (
    remote_port_send_num as pull_remote_port_send_num,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_v1.pull.transfer_plan import (
    remote_ranks_for_req as pull_remote_ranks_for_req,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_v1.pull.transfer_plan import (
    set_hma_shared_port as pull_set_hma_shared_port,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_v1.pull.transfer_plan import (
    sfa_replicate_k_block_ids as pull_sfa_replicate_k_block_ids,
)
from vllm_ascend.distributed.kv_transfer.utils.mooncake_transfer_engine import global_te
from vllm_ascend.distributed.kv_transfer.utils.utils import (
    PD_QOS_DEFAULT,
    RegisterRegions,
    collect_storage_merged_register_regions,
    get_transfer_timeout_value,
    inject_qos,
    tensor_storage_key,
    validate_register_region_count,
)
from vllm_ascend.distributed.utils import (
    get_decode_context_model_parallel_rank,
    get_decode_context_model_parallel_world_size,
)
from vllm_ascend.utils import (
    enable_sfa_dcp_replicated_indexer,
    get_kv_cache_tensor_layers,
    model_uses_sfa_sparse,
)

# isort: off
if TYPE_CHECKING:
    from vllm.v1.attention.backend import AttentionMetadata  # type: ignore
    from vllm.forward_context import ForwardContext
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.request import Request
# isort: on

# Back-compat re-exports: the shared wire protocol and task tracker definitions
# moved to mooncake_v1/common. The imports above keep GET_META_MSG,
# DONE_RECVING_MSG, RemotePortInfo and KVCacheTaskTracker importable from this
# module; MooncakeAgentMetadata keeps its historical name and is bound here (not
# at the end of the module) because the annotations below reference it while the
# classes are being defined.
MooncakeAgentMetadata = PullAgentMetadata


# A busy peer can otherwise keep a global executor worker forever when the
# number of peers is larger than max_workers. Yield after a small FIFO batch so
# other peers already waiting in the global executor queue can make progress.
MAX_REQUESTS_PER_PEER_HANDLER = 5
KV_CACHE_BUFFER_ALIGNMENT = 2 * 1024 * 1024
# Poll/timeout used by the background threads so a stop request can interrupt a
# blocking wait. The threads observe the stop event and exit on their own; the
# ZMQ socket is never closed from another thread.
THREAD_STOP_POLL_SECONDS = 0.2
# Best-effort upper bound for joining a background thread during shutdown.
SHUTDOWN_JOIN_TIMEOUT_SECONDS = 5.0
# Startup wait bound for the transfer threads; on timeout the worker rolls back.
KV_CACHE_THREAD_STARTUP_TIMEOUT_SECONDS = 5 * 60


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
        self.pcp_size = get_pcp_group().world_size
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
        # Set by stop() to let the busy loop exit; see run_busy_loop().
        self._stop_event = threading.Event()
        # First fatal (unrecoverable) error raised by run(); read by the worker.
        self.fatal_error: BaseException | None = None

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

    def stop(self) -> None:
        """Signal the busy loop to exit; idempotent."""
        self._stop_event.set()

    def run(self):
        """Run the thread to handle KV cache transfer requests."""
        try:
            # Listen for new requests for metadata. NOTE(rob): we need each rank
            # to have a unique port. This hack to keeps us moving. We will
            # switch when moving to etcd or where we have a single ZMQ socket in
            # the scheduler.
            handshake_port = pull_handshake_port(
                base_port=self.side_channel_port,
                pp_rank=self.pp_rank,
                pcp_size=self.pcp_size,
                pcp_rank=self.pcp_rank,
                tp_size=self.tp_size,
                tp_rank=self.tp_rank,
            )
            path = make_zmq_path("tcp", self.side_channel_host, handshake_port)
            logger.info(
                "KVCacheSendingThread started listening on path: %s. Thread: tp_rank=%d, pp_rank=%d, pcp_rank=%d",
                path,
                self.tp_rank,
                self.pp_rank,
                self.pcp_rank,
            )
            with zmq_ctx(zmq.ROUTER, path) as sock:  # type: ignore
                self.ready_event.set()
                self.run_busy_loop(sock)
        except Exception as e:
            self.fatal_error = e
            logger.exception(
                "Mooncake KVCacheSendingThread encountered exception. "
                "Thread: tp_rank=%d, pp_rank=%d, listening_path=%s. "
                "Error: %s",
                self.tp_rank,
                self.pp_rank,
                path,
                e,
            )

    def run_busy_loop(self, sock: zmq.Socket):  # type: ignore
        encoder = msgspec.msgpack.Encoder()
        encoded_data = encoder.encode(self.metadata)
        size_in_bytes = len(encoded_data)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("Size of encoded MooncakeAgentMetadata: %s bytes", str(size_in_bytes))

        decoder = msgspec.msgpack.Decoder(type=tuple)
        while not self._stop_event.is_set():
            try:
                if sock.poll(int(THREAD_STOP_POLL_SECONDS * 1000)) == 0:
                    continue
                frames = sock.recv_multipart()
                if len(frames) < 2:
                    logger.error(
                        "Invalid message format in KVCacheSendingThread. "
                        "Expected: at least 2 frames (identity + payload). "
                        "Actual: %d frames. "
                        "Frames: %s. "
                        "Check: Verify message sender implementation.",
                        len(frames),
                        frames,
                    )
                    continue

                identity = frames[0]
                payload = [f for f in frames[1:] if f != b""]
                if len(payload) != 1:
                    logger.error(
                        "Invalid message format in KVCacheSendingThread. "
                        "Expected: exactly 1 payload frame. "
                        "Actual: %d payload frames. "
                        "Frames: %s. "
                        "Check: Verify message sender removes empty frames correctly.",
                        len(payload),
                        frames,
                    )
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
                        handshake_port = pull_handshake_port(
                            base_port=self.side_channel_port,
                            pp_rank=self.pp_rank,
                            pcp_size=self.pcp_size,
                            pcp_rank=self.pcp_rank,
                            tp_size=self.tp_size,
                            tp_rank=self.tp_rank,
                        )
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
                    logger.error(
                        "Connection listener received unexpected message type. "
                        "Expected: GET_META_MSG or DONE_RECVING_MSG. "
                        "Actual: %s. "
                        "Full message: %s. "
                        "Check: Verify message protocol implementation.",
                        msg[0] if msg else "empty",
                        msg,
                    )
            except Exception as e:
                logger.error(
                    "Connection listener encountered exception during message processing. "
                    "Exception type: %s. "
                    "Error: %s. "
                    "Context: Processing frames from socket. "
                    "Check: Review message handling logic and socket state.",
                    type(e).__name__,
                    e,
                )


class KVCacheRecvingThread(KvCacheReformatMixin, threading.Thread):
    def __init__(
        self,
        tp_rank: int,
        tp_size: int,
        _prefill_pp_size: int,
        engine: TransferEngine,
        local_engine_id: str,
        local_handshake_port: int,
        side_channel_port: int,
        local_kv_caches_base_addr: list[list[int]],
        block_len_per_addr: list[list[int]],
        block_stride_per_addr: list[list[int]],
        is_hma_required=False,
        ready_event: threading.Event | None = None,
        vllm_config: VllmConfig | None = None,
        kv_caches: dict[str, Any] | None = None,
        prefill_pp_layer_partition: str | None = None,
        kv_group2layeridx: dict[int, tuple[dict[str, Any], list[int]]] | None = None,
        block_size_scale: list[list[int]] | None = None,
        use_request_level_recv_errors: bool = False,
    ):
        super().__init__(daemon=True, name="KVCacheRecvingThread")
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self._prefill_pp_size = _prefill_pp_size
        self.local_engine_id = local_engine_id
        self.local_handshake_port = local_handshake_port
        self.side_channel_port = side_channel_port
        self.engine = engine
        if ready_event is None:
            ready_event = threading.Event()
        self.ready_event = ready_event

        if kv_caches is None:
            kv_caches = {}
        self.kv_caches = kv_caches
        self.kv_caches_base_addr: dict[str, dict[int, list[list[int]]]] = SizedDict()
        self.kv_caches_base_addr[local_engine_id][local_handshake_port] = local_kv_caches_base_addr
        self.block_len_per_addr = block_len_per_addr
        self.block_stride_per_addr = block_stride_per_addr
        if kv_group2layeridx is None:
            kv_group2layeridx = {}
        self.kv_group2layeridx = kv_group2layeridx
        self.group_compress_ratios: dict[int, int] = {}
        for group_id, (group_spec, _) in self.kv_group2layeridx.items():
            compress_ratio = 1
            kv_cache_spec = group_spec.get("kv_cache_spec")
            if isinstance(kv_cache_spec, dict):
                for spec in kv_cache_spec.values():
                    if isinstance(spec, dict) and isinstance(spec.get("compress_ratio"), int):
                        compress_ratio = max(1, spec["compress_ratio"])
                        break
            self.group_compress_ratios[group_id] = compress_ratio
        self.remote_te_port: dict[str, dict[int, int]] = SizedDict()
        self.remote_block_size_scale: dict[str, dict[int, list[list[int]]]] = SizedDict()
        self.remote_block_stride_per_addr: dict[str, dict[int, list[list[int]]]] = SizedDict()
        self.remote_kv_group2layeridx: dict[str, dict[int, dict[int, tuple[dict[str, Any], list[int]]]]] = SizedDict()
        self.remote_metadata_lock = threading.Lock()
        # Reformat metadata keyed by request_id then CP shard index. Populated by the
        # last TP-offset pull task for each shard; applied once all pull tasks finish.
        self.pending_reformat: defaultdict[str, dict[int, list[tuple[int, list[list[int]], int, list[int]]]]] = (
            defaultdict(dict)
        )
        self.pending_reformat_lock = threading.Lock()

        self.request_queue: queue.Queue[Any] = queue.Queue()
        first_kv_cache = next(iter(self.kv_caches.values()))
        # NPU device selection is thread-local. Executor workers do not inherit
        # the device selected by the model worker thread and would otherwise
        # use device 0 on their first NPU operation.
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
        self.decoder = msgspec.msgpack.Decoder(MooncakeAgentMetadata)
        self.remote_socket_pool = RemoteSocketPool(timeout=1.0)

        assert vllm_config is not None
        self.vllm_config: VllmConfig = vllm_config
        self.model_config = self.vllm_config.model_config
        self.mamba_cache_mode = getattr(self.vllm_config.cache_config, "mamba_cache_mode", None)
        self.num_speculative_tokens = (
            self.vllm_config.speculative_config.num_speculative_tokens
            if self.vllm_config.speculative_config is not None
            else 0
        )
        self.use_mla = self.model_config.is_deepseek_mla
        self.enable_sfa_dcp_replicated_indexer = enable_sfa_dcp_replicated_indexer(self.vllm_config)
        self.is_hma_required = is_hma_required
        self.use_request_level_recv_errors = use_request_level_recv_errors
        self.block_size = self.vllm_config.cache_config.block_size
        try:
            hf_text_config = self.model_config.hf_text_config
            if hf_text_config is None:
                raise AttributeError
        except AttributeError:
            hf_text_config = self.model_config.hf_config
        self.num_layers = hf_text_config.num_hidden_layers
        total_num_layers = self.vllm_config.model_config.get_total_num_hidden_layers()
        metadata_target_layers = total_num_layers if isinstance(total_num_layers, int) else self.num_layers
        # Reserve one full metadata plane for MTP/Eagle layers so that the SFA
        # plane is stable even when producer and consumer use different draft
        # configurations.
        self.index_cache_plane_base = metadata_target_layers * 2
        if block_size_scale is None:
            block_size_scale = []
        self.block_size_scale = block_size_scale
        self.pp_layer_indices = {
            rank: get_prefill_pp_indices(self.num_layers, rank, self._prefill_pp_size, prefill_pp_layer_partition)
            for rank in range(self._prefill_pp_size)
        }
        self.proc_not_transfer_request: dict[str, bool] = {}
        self.proc_not_transfer_request_lock = threading.Lock()
        self.failed_recv_requests: set[str] = set()
        self.invalid_block_ids: set[int] = set()
        self.failed_recv_requests_lock = threading.Lock()

        self.num_draft_layers = 0
        if self.vllm_config.speculative_config is not None:
            if self.vllm_config.speculative_config.method == "mtp":
                # all MTP layer use the same kv cache layer, so only need to transfer once
                self.num_draft_layers = 1
            elif (
                hasattr(self.vllm_config.speculative_config.draft_model_config, "hf_config")
                and getattr(self.vllm_config.speculative_config.draft_model_config.hf_config, "num_hidden_layers", None)
                is not None
            ):
                self.num_draft_layers = (
                    self.vllm_config.speculative_config.draft_model_config.hf_config.num_hidden_layers
                )
        # Set by stop() to let the request loop drain and exit; see run().
        self._stop_event = threading.Event()

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
        remote_block_size=None,
        remote_port_send_num: dict[int, RemotePortInfo] | None = None,
        num_computed_tokens: int = 0,
        all_task_done: bool = False,
        shard_idx: int = 0,
        local_block_ids_replicate_k: BlockIds | None = None,
        remote_block_ids_replicate_k: BlockIds | None = None,
        awaiting_kvs: bool = True,
    ):
        """Add a new request to the queue for processing."""
        transfer_request = TransferRequest.from_legacy_kwargs(
            request_id=request_id,
            remote_request_id=remote_request_id,
            local_block_ids=local_block_ids,
            remote_block_ids=remote_block_ids,
            group_pulls=group_pulls,
            remote_engine_id=remote_engine_id,
            remote_host=remote_host,
            remote_handshake_port=remote_handshake_port,
            remote_block_size=remote_block_size,
            remote_port_send_num=remote_port_send_num,
            num_computed_tokens=num_computed_tokens,
            all_task_done=all_task_done,
            shard_idx=shard_idx,
            local_block_ids_replicate_k=local_block_ids_replicate_k,
            remote_block_ids_replicate_k=remote_block_ids_replicate_k,
            awaiting_kvs=awaiting_kvs,
        )
        trans_info = transfer_request.as_legacy_dict()
        logger.debug("Adding request %s to the queue.Trans info:%s", request_id, trans_info)
        self.request_queue.put(trans_info)

    def get_and_clear_finished_requests(self) -> set[str]:
        """
        Get and clear the requests that have been completed.
        Returns:
            A set of request IDs that have been completed.
        """
        return self.task_tracker.get_and_clear_finished_requests()

    def get_and_clear_invalid_block_ids(self) -> set[int]:
        """Get and clear block ids that failed to load."""
        with self.failed_recv_requests_lock:
            invalid_block_ids = self.invalid_block_ids
            self.invalid_block_ids = set()
        return invalid_block_ids

    def _is_failed_recv_request(self, request_id: str) -> bool:
        with self.failed_recv_requests_lock:
            return request_id in self.failed_recv_requests

    def get_and_clear_failed_recv_requests(self, finished_req_ids: set[str]) -> set[str]:
        """Retain failures until every local pull and reformat has finished."""
        with self.failed_recv_requests_lock:
            failed_req_ids = self.failed_recv_requests & finished_req_ids
            self.failed_recv_requests.difference_update(failed_req_ids)
        return failed_req_ids if self.use_request_level_recv_errors else set()

    def _mark_failed_recv_request(self, request_id: str, local_block_ids: BlockIds) -> None:
        with self.failed_recv_requests_lock:
            self.failed_recv_requests.add(request_id)
            if not self.use_request_level_recv_errors:
                self.invalid_block_ids.update(local_block_ids[0])

    def stop(self) -> None:
        """Signal the request loop to exit; idempotent."""
        self._stop_event.set()

    def run(self):
        """Run the thread to handle KV cache transfer requests."""
        self.ready_event.set()
        while not self._stop_event.is_set():
            try:
                request_data = self.request_queue.get(timeout=THREAD_STOP_POLL_SECONDS)
            except queue.Empty:
                continue
            try:
                if request_data is None:
                    logger.warning("Received a None request. ")
                    self.request_queue.task_done()
                    continue
                self._submit_request(request_data)
            except Exception as e:
                logger.error("Error in KVCacheTransferThread. error=%s. ", e)

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
        requests_handled = 0
        while requests_handled < MAX_REQUESTS_PER_PEER_HANDLER:
            with self.peer_request_queues_lock:
                peer_queue = self.peer_request_queues.get(peer_key)
                if not peer_queue:
                    self.peer_request_queues.pop(peer_key, None)
                    self.active_peer_request_handlers.discard(peer_key)
                    return
                req_meta = peer_queue.popleft()

            requests_handled += 1
            try:
                self._handle_request(req_meta)
            except Exception:
                logger.exception(
                    "Error handling KV cache transfer request for peer %s:%d.",
                    peer_key[0],
                    peer_key[1],
                )

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
                logger.warning("Skipping KV cache transfer for request. remote_request_id=%s. ", remote_request_id)
            else:
                try:
                    logger.debug("Starting to transfer KV cache for request %s.", remote_request_id)
                    self._transfer_kv_cache_all_groups(req_meta)
                    logger.debug("Finished transferring KV cache for request %s.", remote_request_id)
                except Exception as e:
                    transfer_failed = True
                    self._mark_failed_recv_request(request_id, req_meta["local_block_ids"])
                    logger.exception("Failed to transfer KV cache for request %s: %s", remote_request_id, e)
        finally:
            all_tasks_done = self._mark_request_task_done(request_id, all_task_done)
            if all_tasks_done:
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
                            "Failed to reformat KV cache after all pulls for request %s: %s",
                            remote_request_id,
                            e,
                        )
                # Drain the failure with completion, after all local writes end.
                if req_meta.get("awaiting_kvs", True):
                    self.task_tracker.update_done_task_count(request_id)
                else:
                    self.task_tracker.update_done_task_count(request_id, report_completion=False)
                with self.proc_not_transfer_request_lock:
                    self.proc_not_transfer_request.pop(remote_request_id, None)
            self.request_queue.task_done()
            self._send_done_signal_to_free_remote_port(remote_request_id, remote_host, remote_port_send_num)
            # Always send the done signal to the remote host to ensure proper
            # resource cleanup. Failing to do so may cause a memory leak on the
            # remote host.
            self._send_done_recv_signal(remote_request_id, remote_host, remote_handshake_port, remote_port_send_num)

    def _send_done_signal_to_free_remote_port(
        self, request_id: str, remote_host: str, remote_port_send_num: dict[int, RemotePortInfo]
    ):
        if self.side_channel_port != self.local_handshake_port or not remote_port_send_num:
            return
        with self.proc_not_transfer_request_lock:
            if request_id not in self.proc_not_transfer_request:
                self.proc_not_transfer_request[request_id] = True
            should_send = self.proc_not_transfer_request[request_id]
            if should_send:
                self.proc_not_transfer_request[request_id] = False
        if should_send:
            for remote_port in remote_port_send_num:
                if remote_port_send_num[remote_port]["num"] == 0:
                    remote_host_ = remote_port_send_num[remote_port]["host"]
                    self._send_done_recv_signal(request_id, remote_host_, remote_port, remote_port_send_num)

    def _transfer_kv_cache_all_groups(self, req_meta: dict[str, Any]):
        """Handle a KV cache transfer request."""
        remote_request_id = req_meta["remote_request_id"]
        local_block_ids: BlockIds = req_meta["local_block_ids"]
        remote_block_ids: BlockIds = req_meta["remote_block_ids"]
        local_block_ids_replicate_k: BlockIds = req_meta.get("local_block_ids_replicate_k", tuple())
        remote_block_ids_replicate_k: BlockIds = req_meta.get("remote_block_ids_replicate_k", tuple())
        has_replicate_k_blocks = any(local_block_ids_replicate_k) and any(remote_block_ids_replicate_k)
        group_pulls: list[GroupPull] = req_meta["group_pulls"]
        remote_engine_id = req_meta["remote_engine_id"]
        remote_host = req_meta["remote_host"]
        remote_handshake_port = req_meta["remote_handshake_port"]
        # Full prefix cache hit: do not need to read remote blocks, just notify
        # P worker that we have the blocks we need.
        num_local_blocks = sum(len(group_block_ids) for group_block_ids in local_block_ids)
        if num_local_blocks == 0 and not has_replicate_k_blocks:
            return

        # Check if we have the remote metadata cached.
        with self.remote_metadata_lock:
            has_remote_metadata = (
                remote_engine_id in self.kv_caches_base_addr
                and remote_handshake_port in self.kv_caches_base_addr[remote_engine_id]
            )
        if not has_remote_metadata:
            self._get_remote_metadata(remote_host, remote_handshake_port)
        with self.remote_metadata_lock:
            remote_kv_caches_base_addrs = self.kv_caches_base_addr[remote_engine_id][remote_handshake_port]
            local_kv_caches_base_addrs = self.kv_caches_base_addr[self.local_engine_id][self.local_handshake_port]
            remote_transfer_port = self.remote_te_port[remote_engine_id][remote_handshake_port]
            remote_block_stride_per_addr = self.remote_block_stride_per_addr[remote_engine_id][remote_handshake_port]
            remote_kv_group2layeridx = self.remote_kv_group2layeridx.get(remote_engine_id, {}).get(
                remote_handshake_port,
                self.kv_group2layeridx,
            )
        remote_layer_name_to_idx = build_layer_name_to_metadata_idx(remote_kv_group2layeridx)
        session_id = f"{remote_host}:{remote_transfer_port}"

        req_start_time = time.perf_counter()
        src_list: list[int] = []
        dst_list: list[int] = []
        length_list: list[int] = []
        attention_group_reformat_block_ids: list[tuple[tuple[int, list[list[int]], int, list[int]], bool]] = []
        grouped_remote_k_block_ids: list[list[int]] = []
        grouped_local_k_block_ids: list[list[int]] = []
        if has_replicate_k_blocks:
            grouped_remote_k_block_ids, grouped_local_k_block_ids = group_concurrent_contiguous(
                remote_block_ids_replicate_k[0],
                local_block_ids_replicate_k[0],
            )

        def pp_layer_indices(layer_indices: list[int], prefill_pp_rank: int, group_spec: dict[str, Any]) -> list[int]:
            first_layer_index, end_layer_index = self.pp_layer_indices[prefill_pp_rank]
            if self.vllm_config.speculative_config is not None and prefill_pp_rank == self._prefill_pp_size - 1:
                end_layer_index += self.num_draft_layers
            is_index_cache_plane = group_spec.get("kv_cache_spec_type") == "AscendSFAIndexerCacheSpec"

            def in_partition(metadata_layer_idx: int) -> bool:
                transformer_layer = (
                    metadata_layer_idx - self.index_cache_plane_base
                    if is_index_cache_plane and metadata_layer_idx >= self.index_cache_plane_base
                    else metadata_layer_idx
                )
                return first_layer_index <= transformer_layer < end_layer_index

            return [layer_idx for layer_idx in layer_indices if in_partition(layer_idx)]

        use_transfer_group_block_ids = transfer_groups_need_independent_block_ids(
            self.kv_group2layeridx,
            self.block_size_scale,
        )

        def get_remote_layer_idx(
            local_layer_idx: int,
            group_spec: dict[str, Any],
            local_layer_indices: list[int],
        ) -> int:
            # Older peers and lightweight tests may not provide layer names in
            # their cache metadata. Identical layouts remain position-compatible.
            if not remote_layer_name_to_idx or not group_spec.get("layer_names"):
                return local_layer_idx
            return resolve_remote_layer_idx(
                local_layer_idx,
                group_spec,
                local_layer_indices,
                remote_layer_name_to_idx,
            )

        for group_pull in group_pulls:
            group_idx = group_pull.group_id
            group_spec, layer_indices = self.kv_group2layeridx[group_idx]
            kv_cache_group_id = group_spec.get("kv_cache_group_id", group_idx)
            raw_layer_indices = layer_indices
            layer_indices = pp_layer_indices(layer_indices, group_pull.prefill_pp_rank, group_spec)

            if not layer_indices:
                continue
            tp_num_need_pulls = group_pull.num_group_pulls
            inner_offset = group_pull.remote_tp_offset
            is_mamba_group = group_spec["kv_cache_spec_type"] == "MambaSpec"
            block_id_idx = group_idx if use_transfer_group_block_ids else kv_cache_group_id
            local_group_block_ids = local_block_ids[block_id_idx]
            remote_group_block_ids = remote_block_ids[block_id_idx]
            has_group_blocks = bool(local_group_block_ids)
            if not has_group_blocks and (is_mamba_group or not has_replicate_k_blocks):
                continue
            if not is_mamba_group:
                grouped_remote_block_ids: list[list[int]] = []
                grouped_local_block_ids: list[list[int]] = []
                if has_group_blocks:
                    is_group_transfer_end = group_pull.is_group_transfer_end
                    # Block ids are already expanded to kernel granularity and truncated in
                    # _get_kv_split_metadata, so consume them directly here.
                    kernel_remote_block_ids = remote_group_block_ids
                    kernel_local_block_ids = local_group_block_ids

                    if tp_num_need_pulls == 1:
                        grouped_remote_block_ids, grouped_local_block_ids = group_concurrent_contiguous(
                            kernel_remote_block_ids, kernel_local_block_ids
                        )
                    else:
                        grouped_remote_block_ids = [[block_id] for block_id in kernel_remote_block_ids]
                        grouped_local_block_ids = [[block_id] for block_id in kernel_local_block_ids]
                    attention_group_reformat_block_ids.append(
                        (
                            (group_idx, grouped_local_block_ids, tp_num_need_pulls, layer_indices),
                            is_group_transfer_end,
                        )
                    )
            else:
                # Ascend Hybrid Mamba supports "align" (prefix caching) and
                # "none" (no prefix caching), but not "all".
                if self.mamba_cache_mode == "align":
                    if len(remote_group_block_ids) != 1:
                        raise RuntimeError(
                            "Mooncake Mamba transfer requires exactly one normalized remote state block; "
                            f"request_id={remote_request_id}, group_idx={group_idx}, "
                            f"remote_block_count={len(remote_group_block_ids)}, "
                            f"local_block_count={len(local_group_block_ids)}."
                        )
                    remote_state_block_id = remote_group_block_ids[0]
                else:
                    transfer_block_idx = len(remote_group_block_ids) - self.num_speculative_tokens - 1
                    if transfer_block_idx < 0:
                        raise RuntimeError(
                            "Invalid non-aligned Mamba state block metadata: "
                            f"request_id={remote_request_id}, group_idx={group_idx}, "
                            f"remote_block_count={len(remote_group_block_ids)}, "
                            f"num_speculative_tokens={self.num_speculative_tokens}."
                        )
                    remote_state_block_id = remote_group_block_ids[transfer_block_idx]
                grouped_remote_block_ids = [[remote_state_block_id]]
                grouped_local_block_ids = [[local_group_block_ids[0]]]

            if is_mamba_group:
                for layer_idx in layer_indices:
                    remote_layer_idx = get_remote_layer_idx(
                        layer_idx,
                        group_spec,
                        raw_layer_indices,
                    )
                    start_meta_idx = len(src_list)
                    self._append_mamba_transfer_meta(
                        src_list,
                        dst_list,
                        length_list,
                        group_spec=group_spec,
                        src_layer_base_addr=local_kv_caches_base_addrs[layer_idx],
                        dst_layer_base_addr=remote_kv_caches_base_addrs[remote_layer_idx],
                        block_len=self.block_len_per_addr[layer_idx],
                        block_stride=self.block_stride_per_addr[layer_idx],
                        remote_block_stride=remote_block_stride_per_addr[remote_layer_idx],
                        remote_block_id=grouped_remote_block_ids[0][0],
                        local_block_id=grouped_local_block_ids[0][0],
                        tp_num_need_pulls=tp_num_need_pulls,
                        remote_tp_offset=inner_offset,
                    )
                    if logger.isEnabledFor(logging.DEBUG):
                        for src, dst, length in zip(
                            src_list[start_meta_idx:], dst_list[start_meta_idx:], length_list[start_meta_idx:]
                        ):
                            logger.debug(
                                "Mooncake mamba transfer meta: request_id=%s group_idx=%s layer_idx=%s "
                                "local_block_id=%s remote_block_id=%s tp_num_need_pulls=%s "
                                "remote_tp_offset=%s  session_id=%s",
                                remote_request_id,
                                group_idx,
                                layer_idx,
                                grouped_local_block_ids[0][0],
                                grouped_remote_block_ids[0][0],
                                tp_num_need_pulls,
                                inner_offset,
                                session_id,
                            )
                continue

            for layer_idx in layer_indices:
                remote_layer_idx = get_remote_layer_idx(
                    layer_idx,
                    group_spec,
                    raw_layer_indices,
                )
                for cache_idx in range(len(local_kv_caches_base_addrs[layer_idx])):
                    src_layer_base_addr = local_kv_caches_base_addrs[layer_idx][cache_idx]
                    dst_layer_base_addr = remote_kv_caches_base_addrs[remote_layer_idx][cache_idx]
                    block_len = self.block_len_per_addr[layer_idx][cache_idx]
                    block_stride = self.block_stride_per_addr[layer_idx][cache_idx]
                    remote_block_stride = remote_block_stride_per_addr[remote_layer_idx][cache_idx]
                    inner_block_len = block_len // tp_num_need_pulls
                    is_sfa_indexer_group = group_spec["kv_cache_spec_type"] == "AscendSFAIndexerCacheSpec"
                    if is_sfa_indexer_group and has_replicate_k_blocks:
                        transfer_remote_block_ids = grouped_remote_k_block_ids
                        transfer_local_block_ids = grouped_local_k_block_ids
                    else:
                        if not has_group_blocks:
                            continue
                        transfer_remote_block_ids, transfer_local_block_ids = split_if_not_byte_contiguous(
                            grouped_remote_block_ids,
                            grouped_local_block_ids,
                            src_block_stride=remote_block_stride,
                            dst_block_stride=block_stride,
                            block_len=inner_block_len,
                        )
                    for remote_block_id, local_block_id in zip(transfer_remote_block_ids, transfer_local_block_ids):
                        src = src_layer_base_addr + local_block_id[0] * block_stride + inner_offset * inner_block_len
                        dst = dst_layer_base_addr + remote_block_id[0] * remote_block_stride
                        length = inner_block_len * len(local_block_id)
                        src_list.append(src)
                        dst_list.append(dst)
                        length_list.append(length)
                    logger.debug(
                        "Mooncake kv transfer meta: request_id=%s group_idx=%s layer_idx=%s local_block_ids=%s "
                        "remote_block_ids=%s tp_num_need_pulls=%s remote_tp_offset=%s session_id=%s",
                        remote_request_id,
                        group_idx,
                        layer_idx,
                        grouped_local_block_ids,
                        grouped_remote_block_ids,
                        tp_num_need_pulls,
                        inner_offset,
                        session_id,
                    )
        if not src_list:
            return

        logger.debug(
            "Mooncake transfer request=%s session id=%s src=%s dst=%s length=%s",
            remote_request_id,
            session_id,
            src_list,
            dst_list,
            length_list,
        )
        ret = self.engine.batch_transfer_sync_read(session_id, src_list, dst_list, length_list)
        if ret < 0:
            logger.error(
                "Mooncake transfer failed for request. remote_request_id=%s, ret=%d. ",
                req_meta["remote_request_id"],
                ret,
            )
            raise RuntimeError(f"Mooncake transfer failed, ret: {ret}")

        req_end_time = time.perf_counter()
        req_transfer_elapsed = (req_end_time - req_start_time) * 1000
        logger.info(
            "KV cache transfer for request %s took %.2f ms. local_ip %s local_device_id %s remote_session_id %s",
            remote_request_id,
            req_transfer_elapsed,
            get_ip(),
            self.tp_rank,
            session_id,
        )

        ready_attention_group_reformat_block_ids = []
        for reformat_group, is_group_transfer_end in attention_group_reformat_block_ids:
            if is_group_transfer_end:
                ready_attention_group_reformat_block_ids.append(reformat_group)
        if ready_attention_group_reformat_block_ids:
            shard_idx = int(req_meta.get("shard_idx", 0))
            self._stash_pending_reformat(
                req_meta["request_id"],
                shard_idx,
                ready_attention_group_reformat_block_ids,
            )

    def _get_remote_metadata(self, remote_host: str, remote_handshake_port: int) -> None:
        """Get the metadata from the remote host."""
        sock: zmq.Socket | None = None  # type: ignore
        try:
            sock = self._get_remote_socket(remote_host, remote_handshake_port)
            ensure_zmq_send(sock, self.encoder.encode((GET_META_MSG, "")), f"{remote_host}:{remote_handshake_port}")
            metadata_bytes = ensure_zmq_recv(sock, f"{remote_host}:{remote_handshake_port}")
            agent_meta = self.decoder.decode(metadata_bytes)
            engine_id = agent_meta.engine_id
            assert engine_id != self.local_engine_id, (
                f"Conflict engine id {engine_id} with local engine id {self.local_engine_id}."
            )
            if agent_meta.kv_group2layeridx != self.kv_group2layeridx:
                logger.debug(
                    "Remote kv_group2layeridx differs from local. Inspect the remote and local metadata "
                    "to determine whether the difference is expected for the configured parallelism "
                    "and KV cache layouts. remote=%s, local=%s. ",
                    agent_meta.kv_group2layeridx,
                    self.kv_group2layeridx,
                )
            with self.remote_metadata_lock:
                self.remote_kv_group2layeridx[engine_id][remote_handshake_port] = agent_meta.kv_group2layeridx
                self.kv_caches_base_addr[engine_id][remote_handshake_port] = agent_meta.kv_caches_base_addr
                self.remote_te_port[engine_id][remote_handshake_port] = agent_meta.te_rpc_port
                self.remote_block_size_scale[engine_id][remote_handshake_port] = agent_meta.block_size_scale
                self.remote_block_stride_per_addr[engine_id][remote_handshake_port] = agent_meta.block_strides
        except Exception:
            if isinstance(sock, zmq.Socket):  # type: ignore
                sock.close()
                sock = None
            raise
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
            resp = ensure_zmq_recv(sock, f"{remote_host}:{remote_handshake_port}")
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("Received response for request %s: %s", request_id, resp.decode("utf-8"))
            if resp != b"ACK":
                logger.error(
                    "Failed to receive ACK for request. request_id=%s, source=%s:%d. ",
                    request_id,
                    remote_host,
                    remote_handshake_port,
                )
                raise RuntimeError(f"Failed to receive ACK, resp: {resp.decode('utf-8')}")
        except RuntimeError as e:
            if isinstance(sock, zmq.Socket):  # type: ignore
                sock.close()
                sock = None
                logger.warning("Unexpected error occurred in socket. error=%s. ", e)
        finally:
            if sock is not None:
                self._return_remote_socket(sock, remote_host, remote_handshake_port)
                logger.debug("Returned socket to pool for %s:%d", remote_host, remote_handshake_port)

    def _get_remote_socket(self, remote_host: str, remote_handshake_port: int) -> zmq.Socket:  # type: ignore
        """Get a socket to the remote host."""
        return self.remote_socket_pool.get(remote_host, remote_handshake_port)

    def _return_remote_socket(
        self,
        sock: zmq.Socket,  # type: ignore
        remote_host: str,
        remote_handshake_port: int,
    ) -> None:
        """Return the remote socket to the pool."""
        self.remote_socket_pool.return_socket(sock, remote_host, remote_handshake_port)


class MooncakeConnector(KVConnectorBase_V1, SupportsHMA):
    _shutdown_complete: bool = False

    @property
    def supports_divergent_local_hybrid_hits(self) -> bool:
        return True

    def __init__(self, vllm_config: VllmConfig, role: KVConnectorRole, kv_cache_config: KVCacheConfig | None = None):
        assert vllm_config.kv_transfer_config is not None
        # vLLM main's KVConnectorBase_V1.__init__ tracks `_kv_cache_config`
        # (and `_role`), which `get_loaded_kv_cache_group_ids` reads.
        super().__init__(vllm_config, role, kv_cache_config)
        self.engine_id = vllm_config.kv_transfer_config.engine_id
        self._connector_metadata = MooncakeConnectorMetadata()

        if role == KVConnectorRole.SCHEDULER:
            self.connector_scheduler: MooncakeConnectorScheduler | None = MooncakeConnectorScheduler(
                vllm_config, str(self.engine_id), kv_cache_config
            )
            self.connector_worker: MooncakeConnectorWorker | None = None
        elif role == KVConnectorRole.WORKER:
            self.connector_scheduler = None
            self.connector_worker = MooncakeConnectorWorker(vllm_config, str(self.engine_id), kv_cache_config)
        # Scheduler side holds no owned resources; the worker side owns threads,
        # the executor and the remote socket pool.
        self._shutdown_complete = False

    def shutdown(self) -> None:
        """Idempotent connector shutdown; delegates to the worker side."""
        if self._shutdown_complete:
            return
        self._shutdown_complete = True
        if self.connector_worker is not None:
            self.connector_worker.shutdown()

    def get_loaded_kv_cache_group_ids(self, request: "Request") -> tuple[int, ...]:
        # vLLM main's scheduler uses this for skip_zeroing_group_ids.
        # Mooncake restores the request's own transfer blocks.
        return self._kv_cache_config.transfer_group_ids

    def on_new_request(self, request: "Request") -> None:
        assert self.connector_scheduler is not None
        self.connector_scheduler.on_new_request(request)

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
        return self.connector_scheduler.request_finished(request, (block_ids,))

    def request_finished_all_groups(
        self,
        request: "Request",
        block_ids: tuple[list[int], ...],
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

    def get_block_ids_with_load_errors(self) -> set[int]:
        """Get the block ids whose KV load failed."""
        assert self.connector_worker is not None
        return self.connector_worker.get_block_ids_with_load_errors()

    def get_transfer_results(self, finished_req_ids: set[str]) -> KVConnectorTransferResults:
        assert self.connector_worker is not None
        return self.connector_worker.get_transfer_results()

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

    def set_xfer_handshake_metadata(
        self, metadata: Mapping[int | tuple[int, ...], KVConnectorHandshakeMetadata]
    ) -> None:
        """
        Set the KV connector handshake metadata for this connector.

        Args:
            metadata (dict): the handshake metadata to set.
        """
        assert self.connector_scheduler is not None
        self.connector_scheduler.set_xfer_handshake_metadata(metadata)

    def set_xfer_handshake_metadata_pp_aware(
        self, metadata: Mapping[int | tuple[int, ...], KVConnectorHandshakeMetadata]
    ) -> None:
        assert self.connector_scheduler is not None
        self.connector_scheduler.set_xfer_handshake_metadata_from_workers(metadata)


class MooncakeConnectorScheduler:
    """Implementation of Scheduler side methods"""

    def __init__(self, vllm_config: VllmConfig, engine_id: str, kv_cache_config: KVCacheConfig):
        self.vllm_config = vllm_config
        self.kv_cache_config = kv_cache_config
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

        # Use the serving index, which survives dense-engine DP rank reset.
        # Keep scheduler metadata and worker handshake ports consistent.
        self.side_channel_port = pull_base_port(
            kv_port=vllm_config.kv_transfer_config.kv_port,
            dp_index=vllm_config.parallel_config.data_parallel_index,
            tp_size=vllm_config.parallel_config.tensor_parallel_size,
            pp_size=vllm_config.parallel_config.pipeline_parallel_size,
            pcp_size=self.pcp_size,
        )
        # Requests that need to start recv.
        # New requests are added by update_state_after_alloc in
        # the scheduler. Used to make metadata passed to Worker.
        self._reqs_need_recv: dict[str, tuple[Request, BlockIds, BlockIds, int]] = {}
        self._reqs_need_send: dict[str, float] = {}
        self._reqs_in_batch: set[str] = set()

        # master-slave meta information for cross-nodes
        self.multi_nodes_meta_mapping: dict[str, dict[str, Any]] = {}
        self.kv_cache_groups = kv_cache_config.kv_cache_groups
        self.use_compress = self._model_uses_compress()
        self.group_transfer_info = [self._get_group_transfer_info(group) for group in kv_cache_config.kv_cache_groups]
        self.need_truncate = self.use_compress or any(info.is_state_group for info in self.group_transfer_info)

    def _model_uses_compress(self) -> bool:
        hf_config = getattr(self.vllm_config.model_config, "hf_config", None)
        compress_ratios = getattr(hf_config, "compress_ratios", None)
        return isinstance(compress_ratios, (list, tuple, dict))

    def _get_group_transfer_info(self, group: Any) -> GroupTransferInfo:
        return build_group_transfer_info(group, default_block_size=self.block_size)

    def _get_group_unique_specs(self, group: Any) -> list[Any]:
        return group_unique_specs(group)

    def _get_transfer_block_ids(self, block_ids: BlockIds, prompt_len: int) -> BlockIds:
        """Return blocks that contain prompt KV, dropping MTP extra blocks.

        In aligned Mamba mode, normalize each state group to the single block
        containing the final prompt state. This prevents the receiver from
        inferring a block index from a speculative *token* count. Non-aligned
        state groups keep their existing behavior. SWA tail clipping is handled
        as a separate step after this.
        """
        return transfer_block_ids(
            block_ids,
            prompt_len,
            group_transfer_info=self.group_transfer_info,
            mamba_cache_mode=getattr(self.vllm_config.cache_config, "mamba_cache_mode", None),
            dcp_size=self.dcp_size,
        )

    def _get_swa_transfer_block_ids(self, block_ids: BlockIds) -> BlockIds:
        """Clip SWA groups to their window tail and drop placeholder block 0."""
        return swa_transfer_block_ids(block_ids, group_transfer_info=self.group_transfer_info)

    def _state_prefill_token_count(self, num_prompt_tokens: int) -> int:
        """D-side only. Returns N-1 for Mamba models since the decoder
        always recomputes the last token and must start from h(N-1)."""
        return state_prefill_token_count(num_prompt_tokens, need_truncate=self.need_truncate)

    def _truncate_request_for_prefill(self, request: "Request") -> None:
        """P-side only: drop the last prompt token so the prefiller computes
        h(N-1) instead of h(N). The decoder recomputes the last token to
        derive h(N) correctly.

        Guarded by ``_p_side_truncated`` to avoid repeated truncation if the
        request is preempted and rescheduled."""
        truncate_request_for_prefill(request)

    def on_new_request(self, request: "Request") -> None:
        # Truncate before local prefix-cache lookup so its hit count stays
        # consistent with the prompt length used by the scheduler.
        params = request.kv_transfer_params
        if params is not None and params.get("do_remote_decode") and self.need_truncate:
            self._truncate_request_for_prefill(request)

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
            # Remote prefill: get all prompt blocks from remote.
            token_ids = request.prompt_token_ids or []
            actual = self._state_prefill_token_count(len(token_ids))
            params["num_computed_tokens"] = num_computed_tokens
            count = max(actual - num_computed_tokens, 0)
            if count > 0:
                return count, True

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
                    local_block_ids = blocks.get_unhashed_block_ids_all_groups() if num_external_tokens > 0 else []
                    local_full_block_ids = blocks.get_block_ids() if num_external_tokens > 0 else tuple()
                    # Get unhashed blocks to pull from remote.
                    self._reqs_need_recv[request.request_id] = (
                        request,
                        local_block_ids,
                        local_full_block_ids,
                        num_external_tokens,
                    )
                else:
                    logger.warning("Got invalid KVTransferParams. params=%s. ", params)
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
        for req_id, (req, block_ids, full_block_ids, num_external_tokens) in self._reqs_need_recv.items():
            assert req.kv_transfer_params is not None
            # For the case where there are no remote blocks to pull
            # (block_ids is empty), we don't need to schedule
            # an async read on the worker side.
            meta.add_new_req(
                request_id=req_id,
                local_block_ids=block_ids,
                local_full_block_ids=full_block_ids,
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
        block_ids: BlockIds,
    ) -> tuple[bool, dict[str, Any] | None]:
        """
        Once a request is finished, determine whether request blocks
        should be freed now or will be sent asynchronously and freed later.
        """

        params = request.kv_transfer_params
        logger.debug(
            "MooncakeConnector request_finished, request_status=%s, kv_transfer_params=%s", request.status, params
        )

        if params is None:
            return False, None

        # A remote-prefill request can be rejected before scheduler admission
        # (for example, when prompt + max_tokens exceeds max_model_len). In
        # that case update_state_after_alloc() never gets a chance to schedule
        # the receive, so explicitly enqueue an empty receive. The worker skips
        # the data transfer for empty block IDs but still sends the completion
        # signal to the P node, allowing it to release the stranded KV blocks.
        if params.get("do_remote_prefill") or request.request_id in self._reqs_need_recv:
            empty_block_ids: BlockIds = tuple([] for _ in self.kv_cache_groups)
            self._reqs_need_recv[request.request_id] = (
                request,
                empty_block_ids,
                empty_block_ids,
                0,
            )
            params["do_remote_prefill"] = False
            return False, None

        if not params.get("do_remote_decode") or request.status != RequestStatus.FINISHED_LENGTH_CAPPED:
            return False, None

        num_prompt_blocks = math.ceil(len(request.prompt_token_ids) / self.block_size)
        computed_block_ids = self._get_transfer_block_ids(block_ids, len(request.prompt_token_ids))
        computed_block_ids = self._get_swa_transfer_block_ids(computed_block_ids)
        computed_block_lens = [len(block_id_list) for block_id_list in computed_block_ids]
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
            remote_pcp_size=self.pcp_size,
            remote_dcp_size=self.dcp_size,
            remote_ptp_size=self.tp_size,
            last_token_id=request.output_token_ids[-1],
            remote_multi_nodes_meta_mapping=self.multi_nodes_meta_mapping,
            num_prompt_blocks=num_prompt_blocks,
            remote_block_size=self.block_size,
        )

    def _port_offset_from_handshake_metadata(
        self,
        rank_metadata: KVConnectorHandshakeMetadata,
        metadata_key: int | tuple[int, ...],
    ) -> int:
        kv_port = self.vllm_config.kv_transfer_config.kv_port
        handshake_port = getattr(rank_metadata, "handshake_port", 0)
        if handshake_port > 0:
            return handshake_port - kv_port
        if isinstance(metadata_key, int):
            return metadata_key
        raise ValueError(f"Mooncake handshake metadata missing handshake_port for worker key {metadata_key}")

    def set_xfer_handshake_metadata_from_workers(
        self,
        metadata: Mapping[int | tuple[int, ...], KVConnectorHandshakeMetadata],
    ) -> None:
        """Build host mapping for one DP group that may span multiple nodes."""
        if not metadata:
            return

        updated_mapping: dict[str, dict[str, Any]] = {}
        kv_port = self.vllm_config.kv_transfer_config.kv_port
        for metadata_key, rank_metadata in metadata.items():
            port_offset = self._port_offset_from_handshake_metadata(rank_metadata, metadata_key)
            updated_mapping[str(port_offset)] = {
                "host": rank_metadata.local_ip,
                "engine_id": rank_metadata.engine_id,
                "handshake_port": kv_port + port_offset,
            }

        self.multi_nodes_meta_mapping.update(updated_mapping)
        logger.info(
            "MooncakeConnector set_xfer_handshake_metadata: worker_count=%d, updated=%s, multi_nodes_meta_mapping=%s",
            len(metadata),
            updated_mapping,
            self.multi_nodes_meta_mapping,
        )

    def set_xfer_handshake_metadata(
        self, metadata: Mapping[int | tuple[int, ...], KVConnectorHandshakeMetadata]
    ) -> None:
        """Legacy int-keyed entry point (port offset keys)."""
        self.set_xfer_handshake_metadata_from_workers(metadata)


class MooncakeConnectorWorker:
    """Implementation of Worker side methods"""

    def __init__(self, vllm_config: VllmConfig, engine_id: str, kv_cache_config: KVCacheConfig):
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
        self.use_sfa_sparse = model_uses_sfa_sparse(vllm_config.model_config)
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
        self.total_layers = vllm_config.model_config.get_total_num_hidden_layers()
        # Assert that pp_size and pcp_size cannot both be greater than 1
        assert not (self.pp_size > 1 and self.pcp_size > 1), "pp and pcp cannot open in same time"
        self.pcp_rank = get_pcp_group().rank_in_group if self.pcp_size > 1 else 0
        self.dcp_size = get_decode_context_model_parallel_world_size()
        self.dcp_rank = get_decode_context_model_parallel_rank() if self.dcp_size > 1 else 0

        self.max_device_id = self.tp_size * self.dp_size * self.pcp_size * self.pp_size
        self.kv_role = vllm_config.kv_transfer_config.kv_role
        if self.kv_role == "kv_consumer" and self.pcp_size > 1:
            raise ValueError(
                "In P/D disaggregation, Mooncake supports PCP only on the prefill (kv_producer) engine. "
                "Set prefill_context_parallel_size=1 on the decode (kv_consumer) engine."
            )
        self.num_key_value_heads = self.vllm_config.model_config.hf_text_config.num_key_value_heads

        # kv cache config
        self.kv_cache_config = kv_cache_config
        self.num_blocks: int = kv_cache_config.num_blocks
        self.kv_group2layeridx: dict[int, tuple[dict[str, Any], list[int]]] = {}
        _is_fullattn_group = []
        for g in self.kv_cache_config.kv_cache_groups:
            if isinstance(g.kv_cache_spec, UniformTypeKVCacheSpecs):
                is_fullattn_spec = []
                for layer_name in g.layer_names:
                    layer_spec = g.kv_cache_spec.kv_cache_specs[layer_name]
                    is_fullattn_spec.append(isinstance(layer_spec, FullAttentionSpec))
                _is_fullattn_group.append(all(is_fullattn_spec))
            else:
                _is_fullattn_group.append(isinstance(g.kv_cache_spec, FullAttentionSpec))
        self.use_hybrid = (
            not self.vllm_config.scheduler_config.disable_hybrid_kv_cache_manager
            and not all(_is_fullattn_group)
            and len(self.kv_cache_config.kv_cache_groups) > 1
        )
        self._is_hma_required = not vllm_config.scheduler_config.disable_hybrid_kv_cache_manager and not all(
            _is_fullattn_group
        )
        # Use the logical cache layout and HMA state, before registration expands
        # _is_hma_required for transfer-only groups sharing one block table.
        self._use_request_level_recv_errors = self._is_hma_required or len(kv_cache_config.kv_cache_groups) > 1
        self._layer_specs = self._build_layer_specs_from_kv_cache_config(kv_cache_config)

        # Use the serving index, which survives dense-engine DP rank reset.
        # Keep scheduler metadata and worker handshake ports consistent.
        self.side_channel_port = pull_base_port(
            kv_port=vllm_config.kv_transfer_config.kv_port,
            dp_index=vllm_config.parallel_config.data_parallel_index,
            tp_size=vllm_config.parallel_config.tensor_parallel_size,
            pp_size=vllm_config.parallel_config.pipeline_parallel_size,
            pcp_size=self.pcp_size,
        )
        self.handshake_port = pull_handshake_port(
            base_port=self.side_channel_port,
            pp_rank=self.pp_rank,
            pcp_size=self.pcp_size,
            pcp_rank=self.pcp_rank,
            tp_size=self.tp_size,
            tp_rank=self.tp_rank,
        )
        self.sockets: dict = {}
        device_name = str(torch.npu.current_device()) if self.pp_size > 1 else None
        inject_qos(vllm_config.kv_transfer_config.get_from_extra_config("qos_priority", PD_QOS_DEFAULT))
        self.engine = global_te.get_transfer_engine(
            self.side_channel_host,
            device_name=device_name,
        )
        self.te_rpc_port = self.engine.get_rpc_port()

        # Background thread for sending or receiving KV caches.
        self.kv_send_thread: KVCacheSendingThread | None = None
        self.kv_recv_thread: KVCacheRecvingThread | None = None
        # Idempotent shutdown state; see shutdown().
        self._shutdown_complete = False
        self._shutdown_lock = threading.Lock()

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

    def shutdown(self) -> None:
        """Idempotent best-effort shutdown of worker-owned threads and resources."""
        with self._shutdown_lock:
            if self._shutdown_complete:
                return
            self._shutdown_complete = True
        threads = [t for t in (self.kv_recv_thread, self.kv_send_thread) if t is not None]
        for thread in threads:
            try:
                thread.stop()
            except Exception as error:
                logger.warning("Failed to signal mooncake thread stop. thread=%s. error=%s. ", thread.name, error)
        for thread in threads:
            try:
                thread.join(timeout=SHUTDOWN_JOIN_TIMEOUT_SECONDS)
                if thread.is_alive():
                    logger.warning(
                        "Mooncake thread did not stop within %.1f s. thread=%s. ",
                        SHUTDOWN_JOIN_TIMEOUT_SECONDS,
                        thread.name,
                    )
            except Exception as error:
                logger.warning("Failed to join mooncake thread. thread=%s. error=%s. ", thread.name, error)
        fatal_error = self.get_thread_fatal_error()
        if fatal_error is not None:
            logger.error(
                "Mooncake transfer thread reported a fatal error. engine_id=%s. error=%s. ",
                self.engine_id,
                fatal_error,
            )
        if self.kv_recv_thread is not None:
            try:
                self.kv_recv_thread.executor.shutdown(wait=True)
            except Exception as error:
                logger.warning("Failed to shutdown mooncake recv executor. error=%s. ", error)
            try:
                self.kv_recv_thread.remote_socket_pool.close()
            except Exception as error:
                logger.warning("Failed to close mooncake remote socket pool. error=%s. ", error)
        logger.info("MooncakeConnectorWorker shutdown complete. engine_id=%s", self.engine_id)

    def _wait_until_ready(self, thread, ready_event) -> None:
        """Wait for the transfer thread to become ready; roll back on failure."""
        start_wait_time = time.time()
        while not ready_event.is_set():
            if not thread.is_alive():
                self.shutdown()
                raise RuntimeError("KV Cache sending/receiving thread failed to start.")
            if time.time() - start_wait_time > KV_CACHE_THREAD_STARTUP_TIMEOUT_SECONDS:
                self.shutdown()
                raise RuntimeError("Timeout waiting for KV Cache thread to be ready.")
            time.sleep(3)

    def get_thread_fatal_error(self) -> BaseException | None:
        """Return the first fatal error reported by a transfer thread, if any."""
        for thread in (self.kv_recv_thread, self.kv_send_thread):
            error = getattr(thread, "fatal_error", None) if thread is not None else None
            if error is not None:
                return error
        return None

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

    @staticmethod
    def _serialize_kv_group_spec(
        group_spec: Any,
        layer_names: list[str] | None = None,
        kv_cache_spec: Any | None = None,
        kv_cache_group_id: int | None = None,
        total_num_kv_heads: int | None = None,
    ) -> dict[str, Any]:
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

        if layer_names is None:
            layer_names = list(group_spec.layer_names)
        if kv_cache_spec is None:
            kv_cache_spec = group_spec.kv_cache_spec
        spec = kv_cache_spec
        if isinstance(kv_cache_spec, UniformTypeKVCacheSpecs):
            spec = {layer_name: kv_cache_spec.kv_cache_specs[layer_name] for layer_name in layer_names}
        serialized_kv_cache_spec = to_msgpackable(spec)
        if not isinstance(serialized_kv_cache_spec, dict):
            serialized_kv_cache_spec = {"repr": serialized_kv_cache_spec}
        num_key_value_heads = MooncakeConnectorWorker._get_spec_num_key_value_heads(spec)
        if num_key_value_heads is not None:
            serialized_kv_cache_spec["num_kv_heads"] = num_key_value_heads
            serialized_kv_cache_spec["num_key_value_heads"] = num_key_value_heads
        if total_num_kv_heads is not None:
            serialized_kv_cache_spec["total_num_kv_heads"] = total_num_kv_heads

        serialized = {
            "layer_names": layer_names,
            "kv_cache_spec_type": type(kv_cache_spec).__name__,
            "kv_cache_spec": serialized_kv_cache_spec,
        }
        if kv_cache_group_id is not None:
            serialized["kv_cache_group_id"] = kv_cache_group_id
        if isinstance(kv_cache_spec, MambaSpec):
            serialized["shapes"] = [list(shape) for shape in kv_cache_spec.shapes]
            serialized["dtype_sizes"] = [
                torch.tensor([], dtype=dtype).element_size()
                for dtype in kv_cache_spec.dtypes  # type: ignore[misc]
            ]
        return serialized

    def _is_index_cache_layer(self, layer_name: str) -> bool:
        """Whether a layer needs the independent SFA metadata plane.

        Indexer cache names are model-specific (for example ``.index_cache``
        in MiniMax M3 and ``.indexer.k_cache`` in GLM-5.2), so identify the
        physical layout from its cache spec instead of its name.
        """
        return isinstance(self._get_layer_spec(layer_name), AscendSFAIndexerCacheSpec)

    @staticmethod
    def _build_layer_specs_from_kv_cache_config(
        kv_cache_config: KVCacheConfig,
    ) -> dict[str, Any]:
        """Return the real per-layer specs retained by the KV cache config."""
        layer_specs: dict[str, Any] = {}
        for group in kv_cache_config.kv_cache_groups:
            group_kv_spec = group.kv_cache_spec
            if isinstance(group_kv_spec, UniformTypeKVCacheSpecs):
                for layer_name in group.layer_names:
                    layer_specs[layer_name] = group_kv_spec.kv_cache_specs[layer_name]
                continue
            for layer_name in group.layer_names:
                layer_specs[layer_name] = group_kv_spec
        return layer_specs

    @staticmethod
    def _get_spec_num_key_value_heads(spec: Any) -> int | None:
        for key in ("num_kv_heads", "num_key_value_heads"):
            num_key_value_heads = getattr(spec, key, None)
            if isinstance(num_key_value_heads, int):
                return num_key_value_heads
        return None

    @classmethod
    def _get_kv_transfer_spec_key(
        cls,
        spec: Any,
        total_num_kv_heads: int | None,
    ) -> tuple[str, int | None, int | None]:
        # TODO: Extand this key with KV cache layout fields (for example num_dims)
        # if a future model has layers with the same number of kv heads but incompatiible
        # cache shapes.
        return (
            type(spec).__name__,
            cls._get_spec_num_key_value_heads(spec),
            total_num_kv_heads,
        )

    @staticmethod
    def _group_skip_kv_reformat(group_spec: dict[str, Any]) -> bool:
        # index_cache is key-only; never run K/V GQA transpose on it.
        return group_spec.get("kv_cache_spec_type") == "AscendSFAIndexerCacheSpec"

    def _get_spec_total_num_kv_heads(self, spec: Any, layer_idx: int) -> int | None:
        # AscendSFAIndexerCacheSpec inherits MLAAttentionSpec for scheduler
        # grouping, but remains a distinct key-only transfer layout.
        if isinstance(spec, AscendSFAIndexerCacheSpec):
            return 1
        if isinstance(spec, (MLAAttentionSpec, AscendSlidingWindowMLASpec)):
            return 1
        local_num_kv_heads = self._get_spec_num_key_value_heads(spec)
        if local_num_kv_heads is None:
            return local_num_kv_heads

        model_config = self.vllm_config.model_config
        speculative_config = self.vllm_config.speculative_config
        if (
            layer_idx >= self.total_layers
            and speculative_config is not None
            and speculative_config.draft_model_config is not None
        ):
            model_config = speculative_config.draft_model_config
        return model_config.get_total_num_kv_heads()

    def _build_kv_group2layeridx(self) -> dict[int, tuple[dict[str, Any], list[int]]]:
        from vllm.v1.worker.utils import extract_layer_index

        kv_group2layeridx: dict[int, tuple[dict[str, Any], list[int]]] = {}
        model_type = self.vllm_config.model_config.hf_text_config.model_type
        num_attn_module = 2 if model_type in ("longcat_flash", "longcat_flash_ngram") else 1
        index_cache_plane_base = self.total_layers * 2
        next_mtp_layer_idx = self.total_layers
        transfer_group_id = 0
        for kv_cache_group_id, group_spec in enumerate(self.kv_cache_config.kv_cache_groups):
            layer_entries: list[tuple[str, int]] = []
            # For eagle3 method there is no "mtp" in layer names, and upstream model initiation assigns the layer id
            # that is sliced by Pipeline Parallel. So the eagle layer id will confilt with target model layers.
            # Here we determine whether the current layer is an eagle layer based on whether the layer id has been
            # assigned to previous layers. If the layer id has been assigned, we treat the current layer as
            # an eagle layer and assign a new layer id starting from total_layers.
            for layer_name in group_spec.layer_names:
                if "mtp" in layer_name or "eagle" in layer_name:
                    layer_idx = next_mtp_layer_idx
                    next_mtp_layer_idx += 1
                elif self._is_index_cache_layer(layer_name):
                    layer_idx = index_cache_plane_base + extract_layer_index(layer_name, num_attn_module)
                else:
                    layer_idx = extract_layer_index(layer_name, num_attn_module)
                layer_entries.append((layer_name, layer_idx))

            spec_groups: OrderedDict[
                tuple[str, int | None, int | None],
                list[tuple[str, int, Any, int | None]],
            ] = OrderedDict()
            for layer_name, layer_idx in layer_entries:
                kv_cache_spec = self._get_layer_spec(layer_name)
                total_num_kv_heads = self._get_spec_total_num_kv_heads(kv_cache_spec, layer_idx)
                spec_key = self._get_kv_transfer_spec_key(kv_cache_spec, total_num_kv_heads)
                spec_groups.setdefault(spec_key, []).append((layer_name, layer_idx, kv_cache_spec, total_num_kv_heads))

            if len(spec_groups) > 1:
                logger.info(
                    "Split KV cache manager group %d into %d Mooncake transfer groups by KV spec: %s",
                    kv_cache_group_id,
                    len(spec_groups),
                    list(spec_groups),
                )

            for entries in spec_groups.values():
                layer_names = [layer_name for layer_name, _, _, _ in entries]
                layer_indices = [layer_idx for _, layer_idx, _, _ in entries]
                kv_cache_spec = entries[0][2]
                total_num_kv_heads = entries[0][3]
                kv_group2layeridx[transfer_group_id] = (
                    self._serialize_kv_group_spec(
                        group_spec,
                        layer_names=layer_names,
                        kv_cache_spec=kv_cache_spec,
                        kv_cache_group_id=kv_cache_group_id,
                        total_num_kv_heads=total_num_kv_heads,
                    ),
                    layer_indices,
                )
                transfer_group_id += 1
        return kv_group2layeridx

    def _has_mamba_group(self) -> bool:
        return any(group_spec["kv_cache_spec_type"] == "MambaSpec" for group_spec, _ in self.kv_group2layeridx.values())

    def _requires_group_aware_attention_transfer(self) -> bool:
        total_num_kv_heads = {
            self._get_attention_group_num_key_value_heads(group_spec)
            for group_spec, layer_indices in self.kv_group2layeridx.values()
            if layer_indices and group_spec["kv_cache_spec_type"] != "MambaSpec"
        }
        return len(total_num_kv_heads) > 1

    @staticmethod
    def _as_kv_cache_tuple(kv_cache_tuple: Any) -> list[torch.Tensor]:
        # MRv1 can expose Attention as [K/V, blocks, block_size, heads, dim].
        # Transfer metadata indexes blocks on dimension 0 of each K/V view.
        # Unbind preserves the physical strides of the non-contiguous cache.
        if isinstance(kv_cache_tuple, torch.Tensor) and kv_cache_tuple.ndim == 5 and kv_cache_tuple.shape[0] == 2:
            return list(kv_cache_tuple.unbind(0))
        if isinstance(kv_cache_tuple, (list, tuple)):
            return list(kv_cache_tuple)
        return [kv_cache_tuple]

    def _get_layer_spec(self, layer_name: str) -> Any:
        layer_spec = self._layer_specs[layer_name]
        if isinstance(layer_spec, UniformTypeKVCacheSpecs):
            layer_spec = layer_spec.kv_cache_specs[layer_name]
        return layer_spec

    @staticmethod
    def _recover_aligned_kv_tensor_base(
        shared_tensors: list[torch.Tensor],
        tensor_size: int,
    ) -> int | None:
        """Recover the aligned raw buffer base behind hybrid cache views."""
        candidates: set[int] = set()
        for tensor in shared_tensors:
            storage = tensor.untyped_storage()
            storage_base = tensor_storage_key(tensor)
            aligned_base = (
                (storage_base + KV_CACHE_BUFFER_ALIGNMENT - 1) // KV_CACHE_BUFFER_ALIGNMENT * KV_CACHE_BUFFER_ALIGNMENT
            )
            storage_end = storage_base + storage.nbytes()
            if aligned_base <= tensor.data_ptr() and aligned_base + tensor_size <= storage_end:
                candidates.add(aligned_base)

        if len(candidates) > 1:
            raise RuntimeError(
                "Unable to recover one aligned KV tensor base from hybrid cache views: "
                f"candidates={sorted(candidates)}, tensor_size={tensor_size}."
            )
        return candidates.pop() if candidates else None

    @staticmethod
    def _merge_overlapping_register_regions(ptrs: list[int], lengths: list[int]) -> tuple[list[int], list[int]]:
        """Coalesce ranges that Mooncake would reject as overlapping."""
        if len(ptrs) != len(lengths):
            raise ValueError("Mooncake register pointers and lengths must match.")

        ranges = sorted((ptr, ptr + length) for ptr, length in zip(ptrs, lengths) if length > 0)
        merged: list[tuple[int, int]] = []
        for start, end in ranges:
            if merged and start < merged[-1][1]:
                previous_start, previous_end = merged[-1]
                merged[-1] = (previous_start, max(previous_end, end))
            else:
                merged.append((start, end))

        if len(merged) != len(ranges):
            logger.warning(
                "Coalesced %d overlapping Mooncake KV registration ranges into %d ranges.",
                len(ranges),
                len(merged),
            )
        return (
            [start for start, _ in merged],
            [end - start for start, end in merged],
        )

    def _get_registered_kv_tensor_buffers(self, kv_caches: dict[str, torch.Tensor]) -> tuple[list[int], list[int]]:
        ptrs: list[int] = []
        lengths: list[int] = []
        private_layer_tensors: list[torch.Tensor] = []

        for kv_cache_tensor in self.kv_cache_config.kv_cache_tensors:
            shared_tensors: list[torch.Tensor] = []
            for layer_name in get_kv_cache_tensor_layers(kv_cache_tensor):
                for single_kv_cache in self._as_kv_cache_tuple(kv_caches[layer_name]):
                    shared_tensors.append(single_kv_cache)

            if not shared_tensors:
                continue
            # Hybrid cache views can begin after Mamba padding, and target and
            # draft MLA groups can use different padding.  Arithmetic based on
            # a previous group's padding can yield a wrong but still aligned
            # address, so always recover the allocation base from storage.
            base_addr = self._recover_aligned_kv_tensor_base(
                shared_tensors,
                kv_cache_tensor.size,
            )
            if base_addr is None:
                # vLLM #51718 describes all layers in one KVCacheTensor backed
                # by a shared allocation. Ascend's hybrid KV-transfer layout
                # can instead materialize one aligned allocation per layer,
                # including a single-layer descriptor with one private
                # storage. No individual storage then needs to span the
                # descriptor's total size, so register the real layer storage
                # ranges below.
                private_layer_tensors.extend(shared_tensors)
                continue
            if base_addr % KV_CACHE_BUFFER_ALIGNMENT != 0:
                raise RuntimeError(f"Tensor start addr {base_addr} is not aligned to 2 MiB.")
            ptrs.append(base_addr)
            lengths.append(kv_cache_tensor.size)

        if private_layer_tensors:
            regions_by_storage: OrderedDict[int, tuple[int, int]] = OrderedDict()
            for tensor in private_layer_tensors:
                if tensor.numel() == 0:
                    continue
                storage = tensor.untyped_storage()
                storage_base = tensor_storage_key(tensor)
                aligned_base = (
                    (storage_base + KV_CACHE_BUFFER_ALIGNMENT - 1)
                    // KV_CACHE_BUFFER_ALIGNMENT
                    * KV_CACHE_BUFFER_ALIGNMENT
                )
                tensor_span = tensor.element_size() + sum(
                    (size - 1) * stride * tensor.element_size() for size, stride in zip(tensor.shape, tensor.stride())
                )
                tensor_end = tensor.data_ptr() + tensor_span
                storage_end = storage_base + storage.nbytes()
                if not (aligned_base <= tensor.data_ptr() and tensor_end <= storage_end):
                    raise RuntimeError(
                        "Unable to recover an aligned private KV layer storage: "
                        f"data_ptr={tensor.data_ptr()}, tensor_end={tensor_end}, "
                        f"storage=[{storage_base}, {storage_end})."
                    )
                previous = regions_by_storage.get(storage_base)
                regions_by_storage[storage_base] = (
                    aligned_base,
                    max(previous[1] if previous is not None else aligned_base, tensor_end),
                )

            ptrs.extend(base for base, _ in regions_by_storage.values())
            lengths.extend(end - base for base, end in regions_by_storage.values())

        return self._merge_overlapping_register_regions(ptrs, lengths)

    def _get_registered_kv_tensor_buffers_hybrid(
        self, kv_caches: dict[str, torch.Tensor]
    ) -> tuple[list[int], list[int]]:
        ptrs: list[int] = []
        lengths: list[int] = []

        for kv_cache_tensor in self.kv_cache_config.kv_cache_tensors:
            shared_addrs: list[int] = []
            for layer_name in get_kv_cache_tensor_layers(kv_cache_tensor):
                for single_kv_cache in self._as_kv_cache_tuple(kv_caches[layer_name]):
                    shared_addrs.append(single_kv_cache.data_ptr())

            if not shared_addrs:
                continue
            base_addr = min(shared_addrs)
            if base_addr % KV_CACHE_BUFFER_ALIGNMENT != 0:
                raise RuntimeError(f"Tensor start addr {base_addr} is not aligned to 2 MiB.")
            ptrs.append(base_addr)
            lengths.append(kv_cache_tensor.size)

        return ptrs, lengths

    def _get_registered_layer_buffers(self, kv_caches: dict[str, torch.Tensor]) -> tuple[list[int], list[int]]:
        ptrs: list[int] = []
        lengths: list[int] = []

        for kv_cache_tuple in kv_caches.values():
            for single_kv_cache in self._as_kv_cache_tuple(kv_cache_tuple):
                ptrs.append(single_kv_cache.data_ptr())
                lengths.append(single_kv_cache.element_size() * math.prod(single_kv_cache.shape))

        return ptrs, lengths

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        """Register the KV Cache data."""
        self.use_mla = self.vllm_config.model_config.is_deepseek_mla
        self.use_sparse = hasattr(self.vllm_config.model_config.hf_text_config, "index_topk")
        self.enable_sfa_dcp_replicated_indexer = enable_sfa_dcp_replicated_indexer(self.vllm_config)

        self.num_blocks = self.kv_cache_config.num_blocks
        logger.info("num_blocks: %s", self.num_blocks)
        self.kv_caches = kv_caches
        # Maps each KV cache group to its serialized group spec and physical
        # layer indices: {group_id: (group_spec, [layer_idx0, layer_idx1, ...])}.
        self.kv_group2layeridx = self._build_kv_group2layeridx()
        self._is_hma_required = self._is_hma_required or self._requires_group_aware_attention_transfer()
        has_mamba_group = self._has_mamba_group()
        layer_name_to_idx = {
            layer_name: layer_idx
            for _, (group_spec, layer_indices) in self.kv_group2layeridx.items()
            for layer_name, layer_idx in zip(group_spec["layer_names"], layer_indices)
        }
        metadata_layers = max(layer_name_to_idx.values(), default=-1) + 1
        # Per-layer registered KV cache base addresses:
        # [layer_idx][cache_idx] -> data_ptr of one cache tensor, e.g. K/V.
        self.kv_caches_base_addr: list[list[int]] = [[] for _ in range(metadata_layers)]
        # Per-layer block scaling between logical KV blocks and tensor blocks:
        # [layer_idx][cache_idx] -> cache tensor num_blocks / logical num_blocks.
        self.block_size_scale: list[list[int]] = [[] for _ in range(metadata_layers)]
        # Per-layer byte length of one tensor block:
        # [layer_idx][cache_idx] -> element_size * prod(block_shape).
        self.block_len_per_addr: list[list[int]] = [[] for _ in range(metadata_layers)]
        # Per-layer full tensor shape for each registered KV cache address:
        # [layer_idx][cache_idx] -> cache tensor shape, including num_blocks.
        self.block_shape_per_addr: list[list[int]] = [[] for _ in range(metadata_layers)]
        # Per-layer byte stride between consecutive tensor blocks:
        # [layer_idx][cache_idx] -> stride(0) * element_size.
        self.block_stride_per_addr: list[list[int]] = [[] for _ in range(metadata_layers)]

        # TODO: For DSV4 use_compress, metadata/transfer can be optimized by
        # aggregating layer views that share the same raw KVCacheTensor.
        for layer_name, kv_cache_tuple in kv_caches.items():
            layer_idx = layer_name_to_idx[layer_name]
            for single_kv_cache in self._as_kv_cache_tuple(kv_cache_tuple):
                tensor_num_blocks = single_kv_cache.shape[0]
                block_size_scale = tensor_num_blocks // self.num_blocks
                block_shape = single_kv_cache.shape[1:]
                self.block_len_per_addr[layer_idx].append(single_kv_cache.element_size() * math.prod(block_shape))
                self.block_stride_per_addr[layer_idx].append(single_kv_cache.stride(0) * single_kv_cache.element_size())
                self.block_shape_per_addr[layer_idx].append(single_kv_cache.shape)
                self.block_size_scale[layer_idx].append(block_size_scale)
                self.kv_caches_base_addr[layer_idx].append(single_kv_cache.data_ptr())

        if has_mamba_group:
            ptrs, lengths = self._get_registered_kv_tensor_buffers(kv_caches)
            register_regions = RegisterRegions(ptrs=ptrs, lengths=lengths)
        elif self.use_hybrid:
            ptrs, lengths = self._get_registered_kv_tensor_buffers_hybrid(kv_caches)
            register_regions = RegisterRegions(ptrs=ptrs, lengths=lengths)
        else:
            # For normal attention / sparse-c8 KV cache, keep metadata at the
            # logical tensor level but merge registration ranges by underlying
            # storage to avoid exceeding the HCCL per-process region limit.
            register_regions = collect_storage_merged_register_regions(kv_caches)

        validate_register_region_count(register_regions)
        global_te.register_buffer(register_regions.ptrs, register_regions.lengths)

        logger.debug(
            "Mooncake register kv caches metadata: kv_group2layeridx=%s, kv_caches_base_addr=%s, "
            "block_len_per_addr=%s, block_stride_per_addr=%s, block_shape_per_addr=%s, "
            "block_size_scale=%s, ptrs=%s, lengths=%s",
            self.kv_group2layeridx,
            self.kv_caches_base_addr,
            self.block_len_per_addr,
            self.block_stride_per_addr,
            self.block_shape_per_addr,
            self.block_size_scale,
            register_regions.ptrs,
            register_regions.lengths,
        )
        # After KV Caches registered, start the sending or receiving thread.
        metadata = MooncakeAgentMetadata(
            engine_id=self.engine_id,
            te_rpc_port=self.te_rpc_port,
            kv_group2layeridx=self.kv_group2layeridx,
            block_size=self.block_size,
            kv_caches_base_addr=self.kv_caches_base_addr,
            block_size_scale=self.block_size_scale,
            num_blocks=self.num_blocks,
            block_lens=self.block_len_per_addr,
            block_strides=self.block_stride_per_addr,
            local_ip=get_ip(),
            handshake_port=self.handshake_port,
        )
        self.xfer_handshake_metadata = metadata

        try:
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
                    self.kv_caches_base_addr,
                    self.block_len_per_addr,
                    self.block_stride_per_addr,
                    self._is_hma_required,
                    ready_event,
                    self.vllm_config,
                    self.kv_caches,
                    self._prefill_pp_layer_partition,
                    self.kv_group2layeridx,
                    self.block_size_scale,
                    use_request_level_recv_errors=self._use_request_level_recv_errors,
                )
                self.kv_recv_thread.start()
            thread = self.kv_send_thread if self.kv_role == "kv_producer" else self.kv_recv_thread
            assert thread is not None
            self._wait_until_ready(thread, ready_event)
        except Exception:
            self.shutdown()
            raise

    def get_finished(self) -> tuple[set[str], set[str]]:
        results = self.get_transfer_results()
        return results.finished_sending, results.finished_recving

    def get_transfer_results(self) -> KVConnectorTransferResults:
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
        failed_recving = (
            self.kv_recv_thread.get_and_clear_failed_recv_requests(done_recving)  # type: ignore[union-attr]
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
        return KVConnectorTransferResults(
            finished_sending=done_sending,
            finished_recving=done_recving,
            failed_recving=failed_recving,
        )

    def get_block_ids_with_load_errors(self) -> set[int]:
        if self.kv_role == "kv_consumer" and self.kv_recv_thread is not None:
            return self.kv_recv_thread.get_and_clear_invalid_block_ids()
        return set()

    @staticmethod
    def _expand_block_ids(block_ids, scale):
        # Expand each logical block into its `scale` contiguous kernel blocks:
        # logical block b -> [b*scale, b*scale+1, ..., b*scale+scale-1].
        return [bid * scale + offset for bid in block_ids for offset in range(scale)]

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

        The shard (CP rank ``shard_cp_rank``) pulls ``num_blocks_to_pull`` P-blocks
        starting at this rank's local index ``shard_first_p_block``. The destination
        kernel position is derived directly from the CP rank and the block index,
        replacing the previous two-step pipeline (local_chunk_token_starts + per-token
        expansion). TP rank does not affect the block id (it only selects ports / the
        head-dim offset at the transfer stage).
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
            if (global_p_block // block_size_ratio) % local_cp_size != rank_first_d_block % local_cp_size:
                continue
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

    def _is_m3_index_cache_group(self, group_spec: dict[str, Any]) -> bool:
        if group_spec.get("kv_cache_spec_type") != "AscendSFAIndexerCacheSpec":
            return False
        if self.vllm_config.model_config.is_deepseek_mla:
            return False
        return any(".index_cache" in layer_name for layer_name in group_spec.get("layer_names", []))

    def _get_kernel_block_scale(self, layer_indices: list[int]) -> int:
        """Delegate kernel block scale computation to ``transfer_plan.kernel_block_scale``."""
        return pull_kernel_block_scale(layer_indices=layer_indices, block_size_scale=self.block_size_scale)

    def _get_kernel_block_ids(self, layer_indices, meta, group_idx, group_spec):
        """No-CP per-group block ids at kernel granularity: (local, remote).

        Mamba state is not block-sharded, so its logical ids pass through unchanged.
        Attention expands both sides to kernel blocks, skips the prefix-cached remote
        kernels (already on D, located via num_computed_tokens), and trims both lists
        to the shorter one so remote/local stay aligned.
        """
        kv_cache_group_id = self._get_kv_cache_group_id(group_idx, group_spec)
        if group_spec["kv_cache_spec_type"] == "MambaSpec":
            return list(meta.local_block_ids[kv_cache_group_id]), list(meta.remote_block_ids[kv_cache_group_id])

        remote_block_size = meta.remote_block_size or self.block_size

        # kernel_size is the shared (P==D) granularity; remote_scale is derived from it.
        local_scale = self._get_kernel_block_scale(layer_indices)
        kernel_size = self.block_size // local_scale
        assert remote_block_size % kernel_size == 0, (
            f"remote_block_size({remote_block_size}) not divisible by kernel_size({kernel_size})"
        )

        remote_scale = remote_block_size // kernel_size
        kernel_local = self._expand_block_ids(list(meta.local_block_ids[kv_cache_group_id]), local_scale)
        kernel_remote = self._expand_block_ids(list(meta.remote_block_ids[kv_cache_group_id]), remote_scale)
        # Skip prefix-cached remote kernels (D-side already holds them). The token size of one
        # remote kernel is kernel_size * compress_ratio, so the number to skip is
        # num_computed_tokens // (kernel_size * compress_ratio).
        remote_kernel_token_size = kernel_size * self._group_compress_ratio(group_spec)
        remote_start_idx = meta.num_computed_tokens // remote_kernel_token_size
        kernel_remote = kernel_remote[remote_start_idx:]
        num_kernel_blocks = min(len(kernel_remote), len(kernel_local))
        return kernel_local[:num_kernel_blocks], kernel_remote[:num_kernel_blocks]

    def _get_group_kernel_params(self, remote_block_size):
        """Delegate kernel-expansion params computation to ``transfer_plan.group_kernel_params``."""
        return pull_group_kernel_params(
            kv_group2layeridx=self.kv_group2layeridx,
            block_size=self.block_size,
            block_size_scale=self.block_size_scale,
            remote_block_size=remote_block_size,
        )

    def _get_local_remote_cp_params(self, meta: ReqMeta):
        """Delegate CP geometry resolution to ``transfer_plan.local_remote_cp_params``."""
        return pull_local_remote_cp_params(
            remote_block_size=meta.remote_block_size,
            block_size=self.block_size,
            dcp_rank=self.dcp_rank,
            dcp_size=self.dcp_size,
            remote_dcp_size=meta.remote_dcp_size,
        )

    def _get_decode_only_dcp_metadata(
        self,
        req_id: str,
        meta: ReqMeta,
        prefill_tp_size: int,
    ) -> tuple[list[list[int]], list[BlockIds], list[BlockIds]]:
        assert (meta.remote_block_size or self.block_size) == self.block_size, (
            "Decode-only DCP requires equal P/D block sizes."
        )
        if self._is_hma_required:
            chosen_rank_list, _ = self._get_hybrid_remote_rank_group_pulls(req_id, prefill_tp_size)
        else:
            chosen_rank_list = self._get_remote_rank(req_id, prefill_tp_size)
        pcp_offset = self._get_selected_pcp_rank(req_id, meta.remote_pcp_size) * prefill_tp_size
        remote_handshake_port_list = [[x + meta.remote_port + pcp_offset for x in chosen_rank_list]]
        use_transfer_group_block_ids = transfer_groups_need_independent_block_ids(
            self.kv_group2layeridx, self.block_size_scale
        )
        local_block_ids: list[list[int]] = [
            [] for _ in (self.kv_group2layeridx if use_transfer_group_block_ids else meta.local_block_ids)
        ]
        remote_block_ids: list[list[int]] = [
            [] for _ in (self.kv_group2layeridx if use_transfer_group_block_ids else meta.remote_block_ids)
        ]
        for group_idx, (group_spec, layer_indices) in self.kv_group2layeridx.items():
            spec_type = group_spec["kv_cache_spec_type"]
            group_id = self._get_kv_cache_group_id(group_idx, group_spec)
            block_id_idx = group_idx if use_transfer_group_block_ids else group_id
            if spec_type == "MambaSpec":
                # KDA keeps the full sequence state for this TP rank's heads.
                local_block_ids[block_id_idx], remote_block_ids[block_id_idx] = self._get_kernel_block_ids(
                    layer_indices, meta, group_idx, group_spec
                )
                continue
            if spec_type == "AscendSFAIndexerCacheSpec":
                # The full indexer cache is transferred separately.
                continue
            if spec_type not in ("MLAAttentionSpec", "AscendMLAAttentionSpec"):
                raise NotImplementedError(
                    f"Decode-only DCP does not support cache type {spec_type} "
                    f"in transfer group {group_idx} (layer indices {layer_indices})."
                )
            local_blocks = (meta.local_full_block_ids or meta.local_block_ids)[group_id]
            if not local_blocks:
                # Rejected requests enqueue an empty receive to release P-side KV.
                # Keep the selected ports and empty IDs for the completion signal.
                continue
            remote_blocks = meta.remote_block_ids[group_id]
            first_block = meta.num_computed_tokens // self.block_size
            first_block += (self.dcp_rank - first_block) % self.dcp_size
            # P owns the full sequence; D rank r owns r, r + DCP, ... .
            global_blocks = range(first_block, min(meta.num_prompt_blocks, len(remote_blocks)), self.dcp_size)
            scale = self._get_kernel_block_scale(layer_indices)
            local_block_ids[block_id_idx] = self._expand_block_ids(
                [local_blocks[block // self.dcp_size] for block in global_blocks], scale
            )
            remote_block_ids[block_id_idx] = self._expand_block_ids(
                [remote_blocks[block] for block in global_blocks], scale
            )
        return remote_handshake_port_list, [tuple(local_block_ids)], [tuple(remote_block_ids)]

    def _get_kv_split_metadata(
        self,
        req_id: str,
        meta: ReqMeta,
    ) -> tuple[list[list[int]], list[BlockIds], list[BlockIds]]:
        """Build per-transfer port and block-id metadata for remote KV reads.

        Args:
            req_id: Remote request id used as the stable hash key when choosing
                prefill TP ranks.
            meta: Request-level transfer metadata from the scheduler. It
                contains remote/local block ids, remote P-side port base,
                remote P-side PCP/DCP/PTP sizes, and prompt/prefix-cache
                token counts.

        Returns:
            A tuple of three aligned lists. Index ``i`` describes one transfer
            shard for this local D-side rank:
            * remote_handshake_port_list[i]: remote P worker handshake ports
              to pull from. The inner list length is the number of TP pulls
              needed for that shard.
            * local_block_ids_list[i]: local kernel block ids, grouped by KV cache
              group, where received blocks are written.
            * remote_block_ids_list[i]: remote kernel block ids, grouped by KV cache
              group, where blocks are read from.

        PCP selects one complete P-side KV replica; DCP splits prompt blocks
        across P workers. Unequal P/D prefix-cache hits reduce the number of
        remote blocks that still need to be pulled.
        """
        if meta.num_external_tokens == 0 and not any(meta.local_block_ids):
            # Prefix replay can discard a short remote hit entirely. Keep the
            # normal PCP/DCP/HMA notification routing, but provide empty groups
            # before any path indexes logical cache IDs. Clear remote IDs too:
            # Mamba's whole-state path must not request a copy into an empty group.
            meta = copy.copy(meta)
            num_groups = (
                max(
                    (self._get_kv_cache_group_id(idx, spec) for idx, (spec, _) in self.kv_group2layeridx.items()),
                    default=-1,
                )
                + 1
            )
            meta.local_block_ids = tuple([] for _ in range(num_groups))
            meta.remote_block_ids = tuple([] for _ in range(num_groups))

        prefill_tp_size: int = meta.remote_ptp_size if meta.remote_ptp_size is not None else self._prefill_tp_size

        is_decode_only_dcp = self.dcp_size > 1 and meta.remote_dcp_size == 1
        if is_decode_only_dcp:
            return self._get_decode_only_dcp_metadata(req_id, meta, prefill_tp_size)

        if self.dcp_size == meta.remote_dcp_size == 1:
            if self._is_hma_required:
                chosen_rank_list, _ = self._get_hybrid_remote_rank_group_pulls(req_id, prefill_tp_size)
            else:
                chosen_rank_list = self._get_remote_rank(req_id, prefill_tp_size)

            # Select the same TP rank in the chosen PCP replica.
            # E.g. TP2/PP1, PCP rank 1, TP rank 1: offset = 2, port = base + 3.
            pcp_offset = self._get_selected_pcp_rank(req_id, meta.remote_pcp_size) * prefill_tp_size
            remote_handshake_port_list = [[x + meta.remote_port + pcp_offset for x in chosen_rank_list]]
            # Complete KV replicas use the same logical-to-kernel block mapping
            # as the non-CP path.
            use_transfer_group_block_ids = transfer_groups_need_independent_block_ids(
                self.kv_group2layeridx,
                self.block_size_scale,
            )
            local_block_ids: list[list[int]]
            remote_block_ids: list[list[int]]
            if use_transfer_group_block_ids:
                local_block_ids = [[] for _ in self.kv_group2layeridx]
                remote_block_ids = [[] for _ in self.kv_group2layeridx]
            else:
                local_block_ids = [[] for _ in meta.local_block_ids]
                remote_block_ids = [[] for _ in meta.remote_block_ids]
            for group_idx, (group_spec, layer_indices) in self.kv_group2layeridx.items():
                local_kernel_block_ids, remote_kernel_block_ids = self._get_kernel_block_ids(
                    layer_indices, meta, group_idx, group_spec
                )
                block_id_idx = (
                    group_idx if use_transfer_group_block_ids else self._get_kv_cache_group_id(group_idx, group_spec)
                )
                local_block_ids[block_id_idx] = local_kernel_block_ids
                remote_block_ids[block_id_idx] = remote_kernel_block_ids
            local_block_ids_list = [tuple(local_block_ids) for _ in remote_handshake_port_list]
            remote_block_ids_list = [tuple(remote_block_ids) for _ in remote_handshake_port_list]
            return (
                remote_handshake_port_list,
                local_block_ids_list,
                remote_block_ids_list,
            )

        def context_parallel_parameters_check():
            pull_context_parallel_parameters_check(
                remote_cp_size=remote_cp_size,
                local_cp_size=local_cp_size,
                use_mla=self.use_mla,
                use_sparse=self.use_sparse,
                num_key_value_heads=self.num_key_value_heads,
                prefill_tp_size=prefill_tp_size,
                decode_tp_size=self.tp_size,
            )

        def get_kv_head_groups(tp_size):
            return pull_kv_head_groups(
                num_key_value_heads=self.num_key_value_heads,
                use_mla=self.use_mla,
                use_sparse=self.use_sparse,
                tp_size=tp_size,
            )

        def get_cp_group_meta(tp_size, pcp_size, dcp_size, port_base):
            return pull_cp_group_meta(
                num_key_value_heads=self.num_key_value_heads,
                use_mla=self.use_mla,
                use_sparse=self.use_sparse,
                tp_size=tp_size,
                pcp_size=pcp_size,
                dcp_size=dcp_size,
                port_base=port_base,
            )

        def get_local_remote_block_port_mappings():
            p_node_cp_group_meta, d_node_cp_group_meta, mappings = pull_local_remote_block_port_mappings(
                num_key_value_heads=self.num_key_value_heads,
                use_mla=self.use_mla,
                use_sparse=self.use_sparse,
                prefill_tp_size=prefill_tp_size,
                remote_cp_size=remote_cp_size,
                local_cp_size=local_cp_size,
                remote_pcp_size=meta.remote_pcp_size,
                remote_dcp_size=meta.remote_dcp_size,
                remote_port=meta.remote_port,
                decode_tp_size=self.tp_size,
                decode_pcp_size=self.pcp_size,
                decode_dcp_size=self.dcp_size,
                decode_side_channel_port=self.side_channel_port,
                r_blk=r_blk,
            )
            logger.info(
                "p_node_cp_group_meta is:: %s. d_node_cp_group_meta is:: %s. "
                "local_remote_block_port_mappings is:: %s. ",
                p_node_cp_group_meta,
                d_node_cp_group_meta,
                mappings,
            )

            return mappings

        def get_remote_port_send_num(
            local_remote_block_port_mappings: dict[int, list[list[int]]],
        ) -> dict[int, RemotePortInfo]:
            return pull_remote_port_send_num(
                remote_port=meta.remote_port,
                remote_pcp_size=meta.remote_pcp_size,
                prefill_tp_size=prefill_tp_size,
                kv_port=self.vllm_config.kv_transfer_config.kv_port,
                remote_multi_nodes_meta_mapping=meta.remote_multi_nodes_meta_mapping,
                remote_host=meta.remote_host,
                remote_engine_id=meta.remote_engine_id,
                local_remote_block_port_mappings=local_remote_block_port_mappings,
            )

        def _set_hma_shared_port(prefill_tp_size, meta, remote_handshake_port_list, req_id):
            """Delegate HMA attention replica balancing / Mamba port append to transfer_plan."""
            if self._is_hma_required and not (self.use_mla or self.use_sparse):
                chosen_tp_list = self._get_remote_rank(req_id, prefill_tp_size)
            else:
                chosen_tp_list = []
            return pull_set_hma_shared_port(
                is_hma_required=self._is_hma_required,
                use_mla=self.use_mla,
                use_sparse=self.use_sparse,
                num_key_value_heads=self.num_key_value_heads,
                prefill_tp_size=prefill_tp_size,
                remote_port=meta.remote_port,
                remote_dcp_size=meta.remote_dcp_size,
                decode_tp_size=self.tp_size,
                decode_tp_rank=self.tp_rank,
                chosen_tp_list=chosen_tp_list,
                remote_handshake_port_list=remote_handshake_port_list,
            )

        remote_block_size, local_cp_rank, local_cp_size, remote_cp_size, r_blk = self._get_local_remote_cp_params(meta)

        # Per attention group kernel-expansion params. remote_scale is derived locally from
        # the shared kernel size, so no remote handshake scale is needed here.
        group_kernel_params = self._get_group_kernel_params(remote_block_size)

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
        num_external_blocks_p = math.ceil(meta.num_external_tokens / remote_block_size)

        kv_group_items = list(self.kv_group2layeridx.items())
        use_transfer_group_block_ids = transfer_groups_need_independent_block_ids(
            self.kv_group2layeridx,
            self.block_size_scale,
        )
        sequence_group_idx = next(
            (
                group_spec.get("kv_cache_group_id", group_idx)
                for group_idx, (group_spec, _) in kv_group_items
                if group_spec["kv_cache_spec_type"] != "MambaSpec"
            ),
            0,
        )
        assert math.ceil(num_external_blocks / local_cp_size) == len(meta.local_block_ids[sequence_group_idx]), (
            f"num_external_blocks({num_external_blocks}), cp_size({local_cp_size}), "
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

        # make sure the last block (which may be unfull) of P nodes is put to the last block of D node
        remote_block_nums: list[int] = []
        shard_cp_ranks: list[int] = []
        final_block_idx: int | None = None

        for cp_rank, block_num in enumerate(remote_block_nums_all):
            # When r_blk = 1, it degrades to the original cp_rank % Lcp rule.
            if (cp_rank // r_blk - local_cp_rank) % min(local_cp_size, remote_cp_size) == 0:
                if last_block_location == cp_rank:
                    final_block_idx = len(remote_block_nums)
                remote_block_nums.append(block_num)
                shard_cp_ranks.append(cp_rank)

        assert local_remote_block_port_mapping is not None
        if final_block_idx is not None:
            final_block_num = remote_block_nums.pop(final_block_idx)
            shard_cp_ranks.append(shard_cp_ranks.pop(final_block_idx))
            remote_block_nums.append(final_block_num)
            for mapping in local_remote_block_port_mapping:
                final_block_port = mapping.pop(final_block_idx)
                mapping.append(final_block_port)

        # Number of matched P-blocks in the prefix (Note: use P-side unit)
        num_prefix_p_blocks = num_prefix_cached_blocks
        if r_blk > 1:
            # The prefix match granularity for D is Bd = r_blk * Bp, so P0 must be an integer multiple of r_blk.
            assert num_prefix_p_blocks % r_blk == 0, (
                f"P0({num_prefix_p_blocks}) should be  r_blk({r_blk}) integer multiple "
            )

        # The first D-block in the external zone (global block ID in D-units)
        # and the first external D-block owned by this rank.
        num_prefix_d_blocks = num_prefix_p_blocks // r_blk
        first_d = num_prefix_d_blocks + ((local_cp_rank - num_prefix_d_blocks) % local_cp_size)

        remote_handshake_port_list, local_block_ids_list, remote_block_ids_list = [], [], []
        for idx in range(len(local_remote_block_port_mapping[0])):
            mapping_list = []
            for mapping in local_remote_block_port_mapping:
                mapping_list.append(mapping[idx])
            remote_handshake_port_list.append(mapping_list)

        # Attention port TP offset = kv_head_group_offset + dcp_repeat_offset + dcp_rank
        # Within the same head group, only the TP replicas that share the same dcp_rank but have
        # different dcp_repeat hold the exact same attention KV shard. Therefore, substitution is
        # strictly limited to the dcp_repeat (replica) dimension. The head group and dcp_rank parts
        # must be preserved as-is; otherwise, different DCP shards will end up fetching duplicated KV caches.
        remote_handshake_port_list = _set_hma_shared_port(prefill_tp_size, meta, remote_handshake_port_list, req_id)

        # the local_block_ids_list and remote_block_ids_list are related with remote_handshake_port_list
        # such as: local_block_ids_list[[1],[2],[5],[6]], remote_block_ids_list[[1],[1],[1],[1]],
        # remote_handshake_port_list[[30000],[30001],[30004],[30005]]
        # D rank will get remote block 1 in port 30004 and save it in local block 5

        for remote_kv_id in range(len(remote_handshake_port_list)):
            num_blocks_to_pull = remote_block_nums[remote_kv_id]
            # rank-local index of this shard's first external block; used both to slice
            # the remote ids and to derive the matching local kernel positions.
            shard_cp_rank = shard_cp_ranks[remote_kv_id]
            remote_first = (num_prefix_p_blocks - shard_cp_rank + remote_cp_size - 1) // remote_cp_size

            group_remote_block_ids: list[list[int]]
            group_local_block_ids: list[list[int]]
            if use_transfer_group_block_ids:
                group_remote_block_ids = [[] for _ in self.kv_group2layeridx]
                group_local_block_ids = [[] for _ in self.kv_group2layeridx]
            else:
                group_remote_block_ids = [[] for _ in meta.remote_block_ids]
                group_local_block_ids = [[] for _ in meta.local_block_ids]
            is_final_shard = remote_kv_id == len(remote_handshake_port_list) - 1
            for group_idx, (group_spec, _) in kv_group_items:
                kv_cache_group_id = self._get_kv_cache_group_id(group_idx, group_spec)
                block_id_idx = group_idx if use_transfer_group_block_ids else kv_cache_group_id
                if group_spec["kv_cache_spec_type"] == "MambaSpec":
                    # Mamba state is not context-block sharded like attention
                    # KV. Transfer the final state from the final PCP/DCP shard.
                    group_remote_block_ids[block_id_idx] = (
                        list(meta.remote_block_ids[kv_cache_group_id]) if is_final_shard else []
                    )
                    group_local_block_ids[block_id_idx] = (
                        list(meta.local_block_ids[kv_cache_group_id]) if is_final_shard else []
                    )
                    continue
                # Attention: expand to kernel blocks here. Remote is sliced from remote_first
                # (skips this rank's prefix-cached blocks) then expanded; local kernels are
                # located directly from CP rank + block index. This removes the need to pass
                # chunk_starts down to the transfer stage. A shard that pulls nothing has
                # n == 0, so both kernel lists naturally come out empty.
                _, remote_scale, kernel_size = group_kernel_params[group_idx]
                remote_logical = list(
                    meta.remote_block_ids[kv_cache_group_id][remote_first : remote_first + num_blocks_to_pull]
                )
                if local_cp_size > remote_cp_size:
                    remote_logical = [
                        block_id
                        for offset, block_id in enumerate(remote_logical)
                        if ((remote_first + offset) * remote_cp_size + shard_cp_rank) // r_blk % local_cp_size
                        == local_cp_rank
                    ]
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
                    list(meta.local_block_ids[kv_cache_group_id]),
                )
                num_kernel_blocks = min(len(kernel_remote), len(kernel_local))
                group_remote_block_ids[block_id_idx] = kernel_remote[:num_kernel_blocks]

                group_local_block_ids[block_id_idx] = kernel_local[:num_kernel_blocks]
            remote_block_ids_list.append(tuple(group_remote_block_ids))
            local_block_ids_list.append(tuple(group_local_block_ids))

        tp_num_need_pulls = self._get_tp_num_need_pulls(prefill_tp_size)
        if self._is_hma_required:
            # HMA: The final shard might be padded with Mamba ports;
            # the total port count is permitted to exceed the number required by attention.
            assert len(remote_handshake_port_list[0]) >= tp_num_need_pulls, (
                f"tp_num_need_pulls: {tp_num_need_pulls}, remote_handshake_port_list: {remote_handshake_port_list}"
            )
        else:
            assert tp_num_need_pulls == len(remote_handshake_port_list[0]), (
                f"tp_num_need_pulls: {tp_num_need_pulls}, remote_handshake_port_list: {remote_handshake_port_list}"
            )

        return remote_handshake_port_list, local_block_ids_list, remote_block_ids_list

    def _get_dcp_shard_pulls(self, remote_handshake_port_list, prefill_tp_size, remote_base_port, remote_pcp_size):
        """Thin delegate for transfer_plan.dcp_shard_pulls."""
        return pull_dcp_shard_pulls(
            remote_handshake_port_list=remote_handshake_port_list,
            prefill_tp_size=prefill_tp_size,
            remote_base_port=remote_base_port,
            remote_pcp_size=remote_pcp_size,
            tp_size=self.tp_size,
            tp_rank=self.tp_rank,
            kv_group2layeridx=self.kv_group2layeridx,
            get_tp_num_need_pulls=self._get_tp_num_need_pulls,
        )

    def _get_group_pulls_metadata(
        self,
        req_id: str,
        remote_handshake_port_list: list[list[int]],
        prefill_tp_size: int,
        remote_base_port: int,
        remote_pcp_size: int = 1,
        remote_dcp_size: int = 1,
    ) -> list[list[list[GroupPull]]]:
        """Thin delegate for transfer_plan.group_pulls_metadata."""
        return pull_group_pulls_metadata(
            req_id=req_id,
            remote_handshake_port_list=remote_handshake_port_list,
            prefill_tp_size=prefill_tp_size,
            remote_base_port=remote_base_port,
            remote_pcp_size=remote_pcp_size,
            remote_dcp_size=remote_dcp_size,
            is_hma_required=self._is_hma_required,
            prefill_pp_size=self._prefill_pp_size,
            kv_group2layeridx=self.kv_group2layeridx,
            get_tp_num_need_pulls=self._get_tp_num_need_pulls,
            get_selected_pcp_rank=self._get_selected_pcp_rank,
            get_hybrid_remote_rank_group_pulls=self._get_hybrid_remote_rank_group_pulls,
            get_dcp_shard_pulls=self._get_dcp_shard_pulls,
        )

    def _get_hybrid_remote_rank_group_pulls(
        self,
        req_id: str,
        prefill_tp_size: int,
        tp_rank: int | None = None,
    ) -> tuple[list[int], dict[int, list[GroupPull]]]:
        """Thin delegate for transfer_plan.hybrid_remote_rank_group_pulls."""
        return pull_hybrid_remote_rank_group_pulls(
            req_id=req_id,
            prefill_tp_size=prefill_tp_size,
            tp_rank=tp_rank,
            default_tp_rank=self.tp_rank,
            tp_size=self.tp_size,
            prefill_pp_size=self._prefill_pp_size,
            kv_group2layeridx=self.kv_group2layeridx,
            get_attention_group_num_need_pulls=self._get_attention_group_num_need_pulls,
            get_attention_group_num_key_value_heads=self._get_attention_group_num_key_value_heads,
            get_attention_group_remote_rank=self._get_attention_group_remote_rank,
        )

    def _get_hybrid_remote_port_send_num(
        self,
        req_id: str,
        meta: ReqMeta,
        prefill_tp_size: int,
    ) -> dict[int, RemotePortInfo]:
        remote_port_send_num: dict[int, RemotePortInfo] = {}
        num_prefill_workers = prefill_tp_size * self._prefill_pp_size
        for remote_rank in range(num_prefill_workers):
            remote_port = meta.remote_port + remote_rank
            remote_host, _ = self._get_remote_host_info_by_port(
                meta.remote_port,
                remote_port,
                meta.remote_host,
                meta.remote_engine_id,
                meta.remote_multi_nodes_meta_mapping,
            )
            remote_port_send_num[remote_port] = {"num": 0, "host": remote_host}

        for decode_tp_rank in range(self.tp_size):
            remote_ranks, _ = self._get_hybrid_remote_rank_group_pulls(
                req_id,
                prefill_tp_size,
                decode_tp_rank,
            )
            for remote_rank in remote_ranks:
                port = meta.remote_port + remote_rank
                if port in remote_port_send_num:
                    remote_port_send_num[port]["num"] += 1
                else:
                    logger.warning("remote_rank %d is out of prefill workers range", remote_rank)

        return remote_port_send_num

    def _get_attention_group_num_need_pulls(self, group_spec: dict[str, Any], prefill_tp_size: int) -> int:
        return self._get_attention_group_num_need_pulls_for_decode_tp(group_spec, prefill_tp_size, self.tp_size)

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

    def _group_use_mla_rank_routing(self, group_spec: dict[str, Any]) -> bool:
        # MiniMax M3 index caches inherit MLAAttentionSpec only to preserve
        # their per-layer spec. Their TP layout is replicated, not MLA-sharded.
        if self._is_m3_index_cache_group(group_spec):
            return False
        spec_type = group_spec.get("kv_cache_spec_type", "")
        if spec_type not in ("MLAAttentionSpec", "AscendSFAIndexerCacheSpec"):
            return False
        return self.vllm_config.model_config.is_deepseek_mla

    def _get_attention_group_num_key_value_heads(self, group_spec: dict[str, Any]) -> int:
        def model_num_key_value_heads() -> int:
            num_key_value_heads = getattr(self, "num_key_value_heads", None)
            if isinstance(num_key_value_heads, int):
                return num_key_value_heads
            return self.vllm_config.model_config.get_total_num_kv_heads()

        spec_type = group_spec.get("kv_cache_spec_type", "")
        if spec_type == "AscendSFAIndexerCacheSpec" and not self.vllm_config.model_config.is_deepseek_mla:
            # MiniMax M3 index_cache: replicated per TP rank; use model KV head count for
            # routing table sizing (P8->D4 needs 4 decode slots), not spec num_kv_heads=1.
            return model_num_key_value_heads()
        kv_cache_spec = group_spec.get("kv_cache_spec", {})
        if isinstance(kv_cache_spec, dict):
            total_num_kv_heads = kv_cache_spec.get("total_num_kv_heads")
            if isinstance(total_num_kv_heads, int):
                return total_num_kv_heads
            for key in ("num_kv_heads", "num_key_value_heads"):
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
        return model_num_key_value_heads()

    def _get_attention_group_remote_rank(
        self,
        req_id: str,
        group_spec: dict[str, Any],
        prefill_tp_size: int,
        tp_rank: int | None = None,
    ) -> list[int]:
        if tp_rank is None:
            tp_rank = self.tp_rank
        num_key_value_heads = self._get_attention_group_num_key_value_heads(group_spec)
        num_group_pulls = self._get_attention_group_num_need_pulls(group_spec, prefill_tp_size)
        return self._get_remote_ranks_for_req(
            req_id,
            prefill_tp_size,
            num_key_value_heads=num_key_value_heads,
            tp_num_need_pulls=num_group_pulls,
            use_mla=self._group_use_mla_rank_routing(group_spec),
        )[tp_rank]

    def _get_sfa_replicate_k_block_ids(self, meta: ReqMeta) -> tuple[BlockIds, BlockIds]:
        """Thin delegate for transfer_plan.sfa_replicate_k_block_ids."""
        return pull_sfa_replicate_k_block_ids(
            meta=meta,
            use_sfa_sparse=self.use_sfa_sparse,
            enable_sfa_dcp_replicated_indexer=self.enable_sfa_dcp_replicated_indexer,
            dcp_size=self.dcp_size,
            block_size=self.block_size,
        )

    def start_load_kv(self, metadata: MooncakeConnectorMetadata):
        """Start loading KV blocks from remote engine."""
        for req_id in metadata.reqs_in_batch:
            if self.kv_send_thread is not None:
                self.kv_send_thread.task_tracker.add_req_to_process(req_id)
            if self.kv_recv_thread is not None:
                self.kv_recv_thread.task_tracker.add_req_to_process(req_id)

        for req_id, meta in metadata.requests.items():
            if meta.do_virtual:
                if self.kv_recv_thread is not None:
                    self.kv_recv_thread.task_tracker.add_not_transfer_request(req_id)
                continue
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "start_load_kv for request %s from remote engine %s. "
                    "Num local_block_ids: %s. Num remote_block_ids: %s. ",
                    req_id,
                    meta.remote_engine_id,
                    len(meta.local_block_ids),
                    len(meta.remote_block_ids),
                )

            remote_req_id = meta.remote_request_id
            prefill_tp_size: int = meta.remote_ptp_size if meta.remote_ptp_size is not None else self._prefill_tp_size
            (
                local_block_ids_replicate_k,
                remote_block_ids_replicate_k,
            ) = self._get_sfa_replicate_k_block_ids(meta)

            (
                remote_handshake_port_list,
                local_block_ids_list,
                remote_block_ids_list,
            ) = self._get_kv_split_metadata(remote_req_id, meta)
            has_replicate_k_blocks = any(local_block_ids_replicate_k) and any(remote_block_ids_replicate_k)
            remote_transfer_ports = [port for remote_ports in remote_handshake_port_list for port in remote_ports]
            replicate_k_transfer_port = (
                remote_transfer_ports[0] if has_replicate_k_blocks and remote_transfer_ports else None
            )
            group_pulls_list = self._get_group_pulls_metadata(
                remote_req_id,
                remote_handshake_port_list,
                prefill_tp_size,
                meta.remote_port,
                meta.remote_pcp_size,
                meta.remote_dcp_size,
            )
            if meta.remote_dcp_size > 1:
                remote_port_send_num = self.remote_port_send_num[meta.remote_engine_id]
            elif self._is_hma_required and meta.remote_pcp_size <= 1:
                remote_port_send_num = self._get_hybrid_remote_port_send_num(
                    remote_req_id,
                    meta,
                    prefill_tp_size,
                )
            else:
                remote_port_send_num = None

            for shard_idx, remote_ports in enumerate(remote_handshake_port_list):
                for remote_tp_offset, remote_handshake_port in enumerate(remote_ports):
                    assert self.kv_recv_thread is not None
                    remote_host, remote_engine_id = self._get_remote_host_info_by_port(
                        meta.remote_port,
                        remote_handshake_port,
                        meta.remote_host,
                        meta.remote_engine_id,
                        meta.remote_multi_nodes_meta_mapping,
                    )
                    local_block_ids_replicate_k_for_port = (
                        local_block_ids_replicate_k
                        if replicate_k_transfer_port is not None and remote_handshake_port == replicate_k_transfer_port
                        else None
                    )
                    remote_block_ids_replicate_k_for_port = (
                        remote_block_ids_replicate_k
                        if replicate_k_transfer_port is not None and remote_handshake_port == replicate_k_transfer_port
                        else None
                    )
                    group_pulls = group_pulls_list[shard_idx][remote_tp_offset]
                    if has_replicate_k_blocks and remote_handshake_port != replicate_k_transfer_port:
                        # The indexer is replicated, not an attention DCP shard.
                        # Other ports must not overwrite its full-cache transfer.
                        group_pulls = [
                            pull
                            for pull in group_pulls
                            if self.kv_group2layeridx[pull.group_id][0]["kv_cache_spec_type"]
                            != "AscendSFAIndexerCacheSpec"
                        ]
                    self.kv_recv_thread.add_request(
                        request_id=req_id,
                        remote_request_id=remote_req_id,
                        local_block_ids=local_block_ids_list[shard_idx],
                        remote_block_ids=remote_block_ids_list[shard_idx],
                        group_pulls=group_pulls,
                        remote_engine_id=remote_engine_id,
                        remote_host=remote_host,
                        remote_handshake_port=remote_handshake_port,
                        remote_port_send_num=remote_port_send_num,
                        num_computed_tokens=meta.num_computed_tokens,
                        all_task_done=(
                            shard_idx == len(remote_handshake_port_list) - 1
                            and remote_tp_offset == len(remote_ports) - 1
                        ),
                        shard_idx=shard_idx,
                        remote_block_size=meta.remote_block_size,
                        local_block_ids_replicate_k=local_block_ids_replicate_k_for_port,
                        remote_block_ids_replicate_k=remote_block_ids_replicate_k_for_port,
                        awaiting_kvs=meta.num_external_tokens > 0,
                    )

        if self.kv_send_thread is not None and self.dcp_size == 1:
            for req_id, delay_start_time in metadata.requests_to_send.items():
                # Unused PCP replicas report completion locally; only transfer
                # sources wait for the D-side completion signal.
                selected_pcp_rank = self._get_selected_pcp_rank(req_id, self.pcp_size)
                if self.pcp_rank == selected_pcp_rank and self.tp_rank in self._prefill_get_remote_rank(req_id):
                    self.kv_send_thread.add_delayed_request(req_id, delay_start_time)
                else:
                    self.kv_send_thread.add_not_transfer_request(req_id)

        if self.kv_send_thread is not None and self.dcp_size > 1:
            for req_id, delay_start_time in metadata.requests_to_send.items():
                self.kv_send_thread.add_delayed_request(req_id, delay_start_time)

    def _get_tp_num_need_pulls(self, prefill_tp_size: int | None) -> int:
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

    @staticmethod
    def _get_selected_pcp_rank(req_id: str, pcp_size: int) -> int:
        if pcp_size == 1:
            return 0
        # P and D use the P request ID to select the same replica, independently
        # of TP routing.
        return random.Random(string_to_int64_hash(f"pcp:{req_id}")).randrange(pcp_size)

    def _get_remote_host_info_by_port(
        self,
        base_port: int,
        remote_handshake_port: int,
        remote_host: str,
        remote_engine_id: str,
        remote_multi_nodes_meta_mapping: dict,
    ):
        return pull_remote_host_info_by_port(
            base_port=base_port,
            remote_handshake_port=remote_handshake_port,
            remote_host=remote_host,
            remote_engine_id=remote_engine_id,
            remote_multi_nodes_meta_mapping=remote_multi_nodes_meta_mapping,
            kv_port=self.vllm_config.kv_transfer_config.kv_port,
        )

    def _prefill_get_remote_rank(self, req_id: str) -> list[int]:
        if self._is_hma_required:
            prefill_ranks: set[int] = set()
            for group_spec, layer_indices in self.kv_group2layeridx.values():
                if layer_indices:
                    prefill_ranks.update(self._get_prefill_ranks_for_group(req_id, group_spec))
            return sorted(prefill_ranks)
        return sum(self._get_remote_ranks_for_req(req_id), [])

    def _get_prefill_ranks_for_group(self, req_id: str, group_spec: dict[str, Any]) -> set[int]:
        if group_spec["kv_cache_spec_type"] == "MambaSpec":
            assert self._prefill_tp_size % self._decode_tp_size == 0, (
                f"Hybrid Mamba prefill tp size({self._prefill_tp_size}) must be divisible by "
                f"decode tp size({self._decode_tp_size})."
            )
            return set(range(self._prefill_tp_size * self._prefill_pp_size))

        num_key_value_heads = self._get_attention_group_num_key_value_heads(group_spec)
        num_group_pulls = self._get_attention_group_num_need_pulls_for_decode_tp(
            group_spec,
            self._prefill_tp_size,
            self._decode_tp_size,
        )
        remote_ranks_by_decode_rank = self._get_remote_ranks_for_req(
            req_id,
            self._prefill_tp_size,
            num_key_value_heads=num_key_value_heads,
            tp_num_need_pulls=num_group_pulls,
            use_mla=self._group_use_mla_rank_routing(group_spec),
        )
        return {rank for remote_ranks in remote_ranks_by_decode_rank for rank in remote_ranks}

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
        # random split prefill tp list
        tp_sampled_nums = []
        if prefill_tp_size > num_key_value_heads or use_mla or self.use_sparse:
            tp_ori_data = tp_ori_data.reshape(-1, num_groups)
            chosen_group = tp_ori_data[:, [rand_group_index]]
            flattened = chosen_group.reshape(-1).tolist()
            tp_sampled_nums = [
                flattened[i : i + tp_num_need_pulls] for i in range(0, len(flattened), tp_num_need_pulls)
            ]
        # non-random split
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
        """Thin delegate for transfer_plan.remote_ranks_for_req."""
        return pull_remote_ranks_for_req(
            req_id=req_id,
            prefill_tp_size=prefill_tp_size,
            num_key_value_heads=num_key_value_heads,
            tp_num_need_pulls=tp_num_need_pulls,
            use_mla=use_mla,
            default_prefill_tp_size=self._prefill_tp_size,
            decode_tp_size=self._decode_tp_size,
            prefill_pp_size=self._prefill_pp_size,
            default_num_key_value_heads=self.num_key_value_heads,
            use_sparse=self.use_sparse,
            model_is_deepseek_mla=self.vllm_config.model_config.is_deepseek_mla,
            get_tp_num_need_pulls=self._get_tp_num_need_pulls,
            get_remote_tp_ranks=self._get_remote_tp_ranks,
        )


def transfer_groups_need_independent_block_ids(
    kv_group2layeridx: dict[int, tuple[dict[str, Any], list[int]]],
    block_size_scale: list[list[int]],
) -> bool:
    """Whether split transfer groups need separately expanded block IDs.

    Transfer groups that share one KV cache manager group normally share the
    same logical block table. They only need independent block-id lists when
    their tensor layouts use different logical-to-kernel block scales.
    """
    group_scales: dict[int, int] = {}
    for group_idx, (group_spec, layer_indices) in kv_group2layeridx.items():
        if group_spec.get("kv_cache_spec_type") == "MambaSpec":
            continue
        kv_cache_group_id = group_spec.get("kv_cache_group_id", group_idx)
        scale = 1
        if layer_indices and layer_indices[0] < len(block_size_scale) and block_size_scale[layer_indices[0]]:
            scale = block_size_scale[layer_indices[0]][0]
        previous_scale = group_scales.setdefault(kv_cache_group_id, scale)
        if previous_scale != scale:
            return True
    return False


# decode node should know pp_partition_layer in prefill node,
# it is configured in kv_transfer_config by partition_list_str,
# default using vllm layer split algorithm.
def build_layer_name_to_metadata_idx(
    kv_group2layeridx: dict[int, tuple[dict[str, Any], list[int]]],
) -> dict[str, int]:
    layer_name_to_idx: dict[str, int] = {}
    for group_spec, layer_indices in kv_group2layeridx.values():
        layer_names = group_spec.get("layer_names", [])
        for layer_name, layer_idx in zip(layer_names, layer_indices):
            layer_name_to_idx[layer_name] = layer_idx
    return layer_name_to_idx


def resolve_remote_layer_idx(
    local_layer_idx: int,
    group_spec: dict[str, Any],
    local_layer_indices: list[int],
    remote_layer_name_to_idx: dict[str, int],
) -> int:
    """Resolve a remote metadata index by the stable cache-layer name."""
    layer_names = group_spec.get("layer_names", [])
    try:
        position = local_layer_indices.index(local_layer_idx)
        layer_name = layer_names[position]
    except (ValueError, IndexError) as error:
        raise RuntimeError(f"Cannot resolve local KV metadata index {local_layer_idx} to a layer name.") from error
    try:
        return remote_layer_name_to_idx[layer_name]
    except KeyError as error:
        raise RuntimeError(f"Remote KV metadata does not contain layer {layer_name!r}.") from error


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


# Bind the class used verbatim by the extracted reformat mixin. This avoids a
# circular import between the entry module and mooncake_v1/pull/reformat.py.
_pull_reformat.MooncakeConnectorWorker = MooncakeConnectorWorker
