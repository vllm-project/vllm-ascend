# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Scheduler-side logic for Mooncake pull transfers."""

import queue
import threading
import time
from collections import OrderedDict
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import msgspec
import zmq
from vllm import envs
from vllm.distributed.kv_transfer.kv_connector.utils import BlockIds
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorHandshakeMetadata,
    KVConnectorMetadata,
)
from vllm.logger import logger
from vllm.utils.network_utils import make_zmq_path
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.outputs import KVConnectorOutput
from vllm.v1.request import RequestStatus

from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.base_scheduler import (
    MooncakeBaseConnectorScheduler,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.metadata import (
    MooncakeConnectorMetadata,
    MooncakePPTransferMetadata,
    MooncakeTPTransferMetadata,
    MooncakeTransferMetadata,
    MooncakeTransferMetadataGroups,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.utils import (
    ensure_zmq_recv,
    ensure_zmq_send,
    zmq_ctx,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request


GET_META_MSG = b"get_meta_msg"
DONE_RECVING_MSG = b"done_recving_msg"
ACK_MSG = b"ACK"


class MooncakeSchedulerSendingThread(threading.Thread):
    """Serve worker metadata and collect D-side completion messages."""

    def __init__(
        self,
        host: str,
        port: int,
        engine_id: str,
        metadata: Mapping[int | tuple[int, ...], KVConnectorHandshakeMetadata],
        tp_size: int,
        pp_size: int,
        pcp_size: int,
        dcp_size: int,
        ready_event: threading.Event,
    ) -> None:
        super().__init__(daemon=True, name="MooncakeSchedulerSendingThread")
        encoder = msgspec.msgpack.Encoder()
        self.host = host
        self.port = port
        self.tp_size = tp_size
        self.pp_size = pp_size
        self.pcp_size = pcp_size
        self.dcp_size = dcp_size
        self.engine_id = engine_id
        metadata_by_pp_rank = self._merge_metadata_by_pp_rank(metadata)
        self.encoded_metadata = encoder.encode(
            MooncakeTransferMetadataGroups(
                engine_id=engine_id,
                scheduler_host=host,
                scheduler_port=port,
                pp_size=pp_size,
                pcp_size=pcp_size,
                dcp_size=dcp_size,
                tp_size=tp_size,
                metadata_by_pp_rank=metadata_by_pp_rank,
            )
        )
        self.ready_event = ready_event
        self.delayed_free_requests: OrderedDict[str, float] = OrderedDict()
        self.finished_requests: queue.SimpleQueue[str] = queue.SimpleQueue()
        self.early_finished_requests: set[str] = set()
        self.finished_request_ids: set[str] = set()
        self.state_lock = threading.Lock()

    def _merge_metadata_by_pp_rank(
        self,
        metadata: Mapping[int | tuple[int, ...], KVConnectorHandshakeMetadata],
    ) -> dict[int, MooncakePPTransferMetadata]:
        """Validate worker metadata and merge TP-invariant fields by PP."""
        workers_by_pp_rank: dict[int, dict[int, MooncakeTransferMetadata]] = {}
        for metadata_key, rank_metadata in metadata.items():
            if isinstance(metadata_key, int):
                pp_rank, tp_rank = 0, metadata_key
            elif len(metadata_key) == 2:
                pp_rank, tp_rank = metadata_key
            else:
                raise ValueError(
                    f"Mooncake handshake metadata key must be tp_rank or (pp_rank, tp_rank), got {metadata_key!r}"
                )

            if not isinstance(rank_metadata, MooncakeTransferMetadata):
                raise ValueError(
                    "Mooncake scheduler expects MooncakeTransferMetadata, "
                    f"got {type(rank_metadata).__name__} for key {metadata_key!r}"
                )
            if rank_metadata.engine_id != self.engine_id:
                raise ValueError(
                    "Mooncake worker metadata engine ID mismatch: expected "
                    f"{self.engine_id!r}, got {rank_metadata.engine_id!r}"
                )

            workers_by_tp_rank = workers_by_pp_rank.setdefault(pp_rank, {})
            if tp_rank in workers_by_tp_rank:
                raise ValueError(f"Duplicate Mooncake metadata for PP rank {pp_rank}, TP rank {tp_rank}")
            workers_by_tp_rank[tp_rank] = rank_metadata

        expected_pp_ranks = set(range(self.pp_size))
        if set(workers_by_pp_rank) != expected_pp_ranks:
            raise ValueError(
                "Mooncake worker metadata has incomplete PP ranks: expected "
                f"{sorted(expected_pp_ranks)}, got "
                f"{sorted(workers_by_pp_rank)}"
            )

        common_fields = (
            "block_size",
            "num_blocks",
            "spec_block_sizes",
            "layer_names",
            "group_indices",
            "spec_indices",
            "block_strides",
            "block_lens",
            "block_shapes",
            "block_size_scales",
        )
        expected_tp_ranks = set(range(self.tp_size))
        merged: dict[int, MooncakePPTransferMetadata] = {}
        for pp_rank, workers_by_tp_rank in sorted(workers_by_pp_rank.items()):
            if set(workers_by_tp_rank) != expected_tp_ranks:
                raise ValueError(
                    "Mooncake worker metadata has incomplete TP ranks for "
                    f"PP rank {pp_rank}: expected {sorted(expected_tp_ranks)}, "
                    f"got {sorted(workers_by_tp_rank)}"
                )

            reference_tp_rank = min(workers_by_tp_rank)
            reference = workers_by_tp_rank[reference_tp_rank]

            for tp_rank, worker_metadata in workers_by_tp_rank.items():
                mismatched_fields = [
                    field_name
                    for field_name in common_fields
                    if getattr(worker_metadata, field_name) != getattr(reference, field_name)
                ]
                if mismatched_fields:
                    raise ValueError(
                        "Mooncake worker metadata differs across TP ranks for "
                        f"PP rank {pp_rank}: TP {reference_tp_rank} and "
                        f"TP {tp_rank} mismatch in {mismatched_fields}"
                    )

            merged[pp_rank] = MooncakePPTransferMetadata(
                block_size=reference.block_size,
                num_blocks=reference.num_blocks,
                spec_block_sizes=reference.spec_block_sizes,
                layer_names=reference.layer_names,
                group_indices=reference.group_indices,
                spec_indices=reference.spec_indices,
                block_strides=reference.block_strides,
                block_lens=reference.block_lens,
                block_shapes=reference.block_shapes,
                block_size_scales=reference.block_size_scales,
                metadata_by_tp_rank={
                    tp_rank: MooncakeTPTransferMetadata(
                        te_rpc_port=worker_metadata.te_rpc_port,
                        kv_caches_base_addr=(worker_metadata.kv_caches_base_addr),
                        local_ip=worker_metadata.local_ip,
                        handshake_port=worker_metadata.handshake_port,
                    )
                    for tp_rank, worker_metadata in sorted(workers_by_tp_rank.items())
                },
            )
        return merged

    def add_delayed_request(self, request_id: str, delay_start_time: float) -> None:
        with self.state_lock:
            self.delayed_free_requests[request_id] = delay_start_time
            if request_id in self.early_finished_requests:
                self.early_finished_requests.remove(request_id)
                self._mark_finished_locked(request_id)

    def get_and_clear_finished_requests(self) -> set[str]:
        finished: set[str] = set()
        with self.state_lock:
            self._retrieve_expired_requests_locked()
        while True:
            try:
                finished.add(self.finished_requests.get_nowait())
            except queue.Empty:
                return finished

    def run(self) -> None:
        path = make_zmq_path("tcp", self.host, self.port)
        try:
            logger.info("Mooncake scheduler sending thread listening on %s", path)
            with zmq_ctx(zmq.ROUTER, path) as sock:
                sock.setsockopt(zmq.RCVTIMEO, 1000)
                self.ready_event.set()
                self._run_busy_loop(sock)
        except Exception:
            self.ready_event.set()
            logger.exception("Mooncake scheduler sending thread failed on %s", path)

    def _run_busy_loop(self, sock: zmq.Socket) -> None:
        decoder = msgspec.msgpack.Decoder(type=tuple)
        while True:
            try:
                frames = sock.recv_multipart()
            except zmq.Again:
                continue

            identity = frames[0]
            payload = [frame for frame in frames[1:] if frame]
            if len(payload) != 1:
                logger.warning("Invalid Mooncake scheduler control frames: %s", frames)
                continue

            try:
                msg = decoder.decode(payload[0])
                if msg and msg[0] == GET_META_MSG:
                    sock.send_multipart((identity, b"", self.encoded_metadata))
                elif msg[0] == DONE_RECVING_MSG and len(msg) == 2:
                    self._handle_finished_request(str(msg[1]))
                    sock.send_multipart((identity, b"", ACK_MSG))
                else:
                    logger.warning("Unexpected Mooncake scheduler control message: %s", msg)
            except Exception:
                logger.exception("Failed to handle Mooncake scheduler control message")

    def _handle_finished_request(self, request_id: str) -> None:
        with self.state_lock:
            if request_id in self.finished_request_ids:
                return
            if request_id in self.delayed_free_requests:
                self._mark_finished_locked(request_id)
            else:
                self.early_finished_requests.add(request_id)

    def _mark_finished_locked(self, request_id: str) -> None:
        if request_id in self.finished_request_ids:
            return
        self.finished_request_ids.add(request_id)
        self.delayed_free_requests.pop(request_id, None)
        self.finished_requests.put(request_id)

    def _retrieve_expired_requests_locked(self) -> None:
        current_time = time.time()
        while self.delayed_free_requests:
            request_id = next(iter(self.delayed_free_requests))
            delay_start_time = self.delayed_free_requests[request_id]
            if current_time - delay_start_time <= envs.VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT:
                break

            self._mark_finished_locked(request_id)
            logger.error(
                "Force freed expired Mooncake request %s after %s seconds",
                request_id,
                envs.VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT,
            )


class MooncakeSchedulerRecvingThread(threading.Thread):
    """Send scheduler-level completion messages from D to P."""

    def __init__(self, ready_event: threading.Event) -> None:
        super().__init__(daemon=True, name="MooncakeSchedulerRecvingThread")
        self.ready_event = ready_event
        self.request_queue: queue.Queue[tuple[str, int, str]] = queue.Queue()
        self.encoder = msgspec.msgpack.Encoder()

    def add_request(self, remote_host: str, remote_port: int, request_id: str) -> None:
        self.request_queue.put((remote_host, remote_port, request_id))

    def run(self) -> None:
        self.ready_event.set()
        while True:
            request = self.request_queue.get()

            try:
                self._send_done_recving(*request)
            except Exception:
                logger.exception(
                    "Failed to send Mooncake scheduler completion for request %s",
                    request[2],
                )
                self.request_queue.put(request)
            finally:
                self.request_queue.task_done()

    def _send_done_recving(self, remote_host: str, remote_port: int, request_id: str) -> None:
        path = make_zmq_path("tcp", remote_host, remote_port)
        with zmq_ctx(zmq.REQ, path) as sock:
            sock.setsockopt(zmq.SNDTIMEO, 1000)
            sock.setsockopt(zmq.RCVTIMEO, 1000)
            ensure_zmq_send(
                sock,
                self.encoder.encode((DONE_RECVING_MSG, request_id)),
                path,
            )
            response = ensure_zmq_recv(sock, path)
            if response != ACK_MSG:
                raise RuntimeError(f"Unexpected Mooncake scheduler completion response: {response!r}")


class MooncakePullConnectorScheduler(MooncakeBaseConnectorScheduler):
    """Scheduler-side Mooncake pull connector implementation."""

    def __init__(
        self,
        vllm_config: "VllmConfig",
        engine_id: str,
        kv_cache_config: "KVCacheConfig",
    ) -> None:
        super().__init__(vllm_config, engine_id, kv_cache_config)

        # Requests waiting for the worker to start a READ transfer.
        self._reqs_need_recv: dict[str, tuple[Request, BlockIds, BlockIds, int]] = {}
        # Producer requests whose blocks must remain allocated until read.
        self._reqs_need_send: dict[str, float] = {}
        self._reqs_in_batch: set[str] = set()
        # D request -> (P scheduler host, port, P request id).
        self._reqs_recv_info: dict[str, tuple[str, int, str]] = {}
        self._sending_thread: MooncakeSchedulerSendingThread | None = None
        self._recving_thread: MooncakeSchedulerRecvingThread | None = None

        if self.kv_transfer_config.is_kv_consumer:
            recving_ready_event = threading.Event()
            self._recving_thread = MooncakeSchedulerRecvingThread(recving_ready_event)
            self._recving_thread.start()
            recving_ready_event.wait()

    def set_xfer_handshake_metadata_from_workers(
        self,
        metadata: Mapping[int | tuple[int, ...], KVConnectorHandshakeMetadata],
    ) -> None:
        if not self.kv_transfer_config.is_kv_producer or not metadata or self._sending_thread is not None:
            return

        ready_event = threading.Event()
        self._sending_thread = MooncakeSchedulerSendingThread(
            self.side_channel_host,
            self.side_channel_port,
            self.engine_id,
            metadata,
            self.tp_size,
            self.pp_size,
            self.pcp_size,
            self.dcp_size,
            ready_event,
        )
        self._sending_thread.start()
        if not ready_event.wait(timeout=10):
            raise RuntimeError("Timed out starting Mooncake scheduler sending thread")
        if not self._sending_thread.is_alive():
            raise RuntimeError("Mooncake scheduler sending thread failed to start")

    def get_num_new_matched_tokens(self, request: "Request", num_computed_tokens: int) -> tuple[int, bool]:
        """Return prompt tokens that will be loaded from a remote producer."""
        params = request.kv_transfer_params
        logger.debug(
            "MooncakeConnector get_num_new_matched_tokens: num_computed_tokens=%s, kv_transfer_params=%s",
            num_computed_tokens,
            params,
        )

        if params is not None and params.get("do_remote_prefill"):
            token_ids = request.prompt_token_ids or []
            actual = self._state_prefill_token_count(len(token_ids))
            params["num_computed_tokens"] = num_computed_tokens
            count = max(actual - num_computed_tokens, 0)
            if count > 0:
                return count, True

        if params is not None and params.get("do_remote_decode") and self.need_truncate:
            self._truncate_request_for_prefill(request)

        return 0, False

    def update_state_after_alloc(
        self,
        request: "Request",
        blocks: "KVCacheBlocks",
        num_external_tokens: int,
    ) -> None:
        params = request.kv_transfer_params
        logger.debug(
            "MooncakeConnector update_state_after_alloc: num_external_tokens=%s, kv_transfer_params=%s",
            num_external_tokens,
            params,
        )

        if params is not None and (params.get("do_remote_prefill", False) or params.get("do_remote_decode", False)):
            self._reqs_in_batch.add(request.request_id)

        if params is None or not params.get("do_remote_prefill"):
            return

        if params.get("remote_block_ids"):
            required_remote_fields = (
                "remote_engine_id",
                "remote_host",
                "remote_port",
                "remote_request_id",
            )
            if all(field in params for field in required_remote_fields):
                remote = (
                    params["remote_host"],
                    params["remote_port"],
                    params["remote_request_id"],
                )
                if num_external_tokens > 0:
                    self._reqs_need_recv[request.request_id] = (
                        request,
                        blocks.get_unhashed_block_ids_all_groups(),
                        blocks.get_block_ids(),
                        num_external_tokens,
                    )
                    self._reqs_recv_info[request.request_id] = remote
                else:
                    if self._recving_thread is None:
                        raise RuntimeError("Producer Mooncake scheduler cannot acknowledge a receive request")
                    self._recving_thread.add_request(*remote)
            else:
                logger.warning("Got invalid KVTransferParams. params=%s.", params)
        else:
            assert num_external_tokens == 0

        # Only trigger one transfer for a request.
        params["do_remote_prefill"] = False

    def build_connector_meta(self, scheduler_output: SchedulerOutput) -> KVConnectorMetadata:
        meta = MooncakeConnectorMetadata()

        for (
            req_id,
            (req, block_ids, full_block_ids, num_external_tokens),
        ) in self._reqs_need_recv.items():
            assert req.kv_transfer_params is not None
            meta.add_new_req(
                request_id=req_id,
                local_block_ids=block_ids,
                local_full_block_ids=full_block_ids,
                local_num_prompt_tokens=req.num_prompt_tokens,
                num_external_tokens=num_external_tokens,
                kv_transfer_params=req.kv_transfer_params,
            )

        self._reqs_need_recv.clear()
        meta.reqs_in_batch = self._reqs_in_batch
        self._reqs_in_batch = set()
        return meta

    def request_finished(
        self,
        request: "Request",
        block_ids: BlockIds,
    ) -> tuple[bool, dict[str, Any] | None]:
        """Expose completed producer blocks for a remote READ transfer."""
        params = request.kv_transfer_params
        logger.debug(
            "MooncakeConnector request_finished: request_status=%s, kv_transfer_params=%s",
            request.status,
            params,
        )

        if (
            params is None
            or not params.get("do_remote_decode")
            or request.status != RequestStatus.FINISHED_LENGTH_CAPPED
        ):
            return False, None

        prompt_token_ids = request.prompt_token_ids or []
        prompt_len = len(prompt_token_ids)
        computed_block_ids = self._get_transfer_block_ids(block_ids, prompt_len)
        num_computed_blocks = sum(len(group_block_ids) for group_block_ids in computed_block_ids)
        delay_free_blocks = num_computed_blocks > 0
        if delay_free_blocks:
            logger.info(
                "Delaying free of %d blocks for request %s",
                num_computed_blocks,
                request.request_id,
            )
            delay_start_time = time.time()
            self._reqs_need_send[request.request_id] = delay_start_time
            if self._sending_thread is None:
                raise RuntimeError("Mooncake scheduler metadata has not been initialized")
            self._sending_thread.add_delayed_request(
                request.request_id,
                delay_start_time,
            )

        return delay_free_blocks, {
            "do_remote_prefill": True,
            "do_remote_decode": False,
            "remote_block_ids": computed_block_ids,
            "remote_num_prompt_tokens": prompt_len,
            "remote_engine_id": self.engine_id,
            "remote_request_id": request.request_id,
            "remote_host": self.side_channel_host,
            "remote_port": self.side_channel_port,
            "last_token_id": request.output_token_ids[-1],
        }

    def on_new_request(self, request: "Request") -> None:
        pass

    def update_connector_output(
        self,
        connector_output: KVConnectorOutput,
    ) -> None:
        # D side: this output has already aggregated completion from all
        # workers. Send one scheduler-to-scheduler ACK for the request.
        for req_id in connector_output.finished_recving or ():
            remote = self._reqs_recv_info.pop(req_id, None)
            if remote is not None:
                if self._recving_thread is None:
                    raise RuntimeError("Producer Mooncake scheduler received a receive-completion event")
                self._recving_thread.add_request(*remote)

        # P side: feed scheduler-received ACKs into vLLM's standard delayed
        # free path. Scheduler._update_from_kv_xfer_finished reads this same
        # KVConnectorOutput immediately after this hook returns.
        finished_sending = (
            self._sending_thread.get_and_clear_finished_requests() if self._sending_thread is not None else set()
        )
        if finished_sending:
            for req_id in finished_sending:
                self._reqs_need_send.pop(req_id, None)
            if connector_output.finished_sending is None:
                connector_output.finished_sending = finished_sending
            else:
                connector_output.finished_sending.update(finished_sending)


__all__ = [
    "MooncakePullConnectorScheduler",
    "MooncakeSchedulerRecvingThread",
    "MooncakeSchedulerSendingThread",
]
