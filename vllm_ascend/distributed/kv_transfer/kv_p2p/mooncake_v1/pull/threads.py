# SPDX-License-Identifier: Apache-2.0
"""Data-plane threads of the pull (non-layerwise) mooncake V1 connector.

The classes used to live inside mooncake_connector. They are moved verbatim so
the entry module can stay a compatibility re-export layer. The entry module
keeps importing these names, so class identity and the existing test patches on
the entry module stay valid for the code that remains there.
"""

import logging
import queue
import threading
import time
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import msgspec
import torch
import zmq
from mooncake.engine import TransferEngine  # type: ignore
from vllm.config import VllmConfig
from vllm.distributed import get_pcp_group
from vllm.distributed.kv_transfer.kv_connector.utils import BlockIds
from vllm.distributed.parallel_state import (
    get_pp_group,
    get_tensor_model_parallel_world_size,
)
from vllm.logger import logger
from vllm.utils.network_utils import get_ip, make_zmq_path

from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_v1.common.block_selection import (
    build_layer_name_to_metadata_idx,
    get_prefill_pp_indices,
    resolve_remote_layer_idx,
    transfer_groups_need_independent_block_ids,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_v1.common.constants import (
    THREAD_STOP_POLL_SECONDS,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_v1.common.endpoints import (
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
    zmq_ctx,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_v1.pull.metadata import (
    GroupPull,
    TransferRequest,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_v1.pull.reformat import KvCacheReformatMixin
from vllm_ascend.utils import enable_sfa_dcp_replicated_indexer

MooncakeAgentMetadata = PullAgentMetadata


# A busy peer can otherwise keep a global executor worker forever when the
# number of peers is larger than max_workers. Yield after a small FIFO batch so
# other peers already waiting in the global executor queue can make progress.
MAX_REQUESTS_PER_PEER_HANDLER = 5


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
        # last TP-offset pull task for each PP segment within a shard; applied once all pull tasks finish.
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
                transfer_failed = transfer_failed or self._is_failed_recv_request(request_id)
                if transfer_failed:
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
                self.remote_socket_pool.discard(sock)
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
                self.remote_socket_pool.discard(sock)
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
