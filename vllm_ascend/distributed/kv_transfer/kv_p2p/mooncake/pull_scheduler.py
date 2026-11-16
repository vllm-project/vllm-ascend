# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Scheduler-side logic for Mooncake pull transfers."""

import queue
import threading
import time
from collections import OrderedDict, defaultdict, deque
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

import msgspec
import zmq
from cachetools import TTLCache
from vllm.distributed.kv_transfer.kv_connector.utils import BlockIds
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorHandshakeMetadata,
    KVConnectorMetadata,
)
from vllm.logger import logger
from vllm.utils.network_utils import make_zmq_path, make_zmq_socket
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.outputs import KVConnectorOutput
from vllm.v1.request import RequestStatus

from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.base_scheduler import (
    GET_META_MSG,
    MooncakeBaseConnectorScheduler,
    MooncakeMetadataThread,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.heartbeat import (
    DEFAULT_KV_LEASE_DURATION,
    HEARTBEAT_MSG,
    MooncakeHeartbeatThread,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.metadata import (
    MooncakeConnectorMetadata,
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


DONE_RECVING_MSG = b"done_recving_msg"
ACK_MSG = b"ACK"
PRODUCER_LEASE_CACHE_TTL_SECONDS = 600
PRODUCER_LEASE_CACHE_MAX_SIZE = 1024


class MooncakeSchedulerSendingThread(MooncakeMetadataThread):
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
        use_kv_pp: bool,
        ready_event: threading.Event,
        kv_lease_duration: int = DEFAULT_KV_LEASE_DURATION,
    ) -> None:
        super().__init__(host, port, engine_id, metadata, tp_size, pp_size, pcp_size, dcp_size, use_kv_pp, ready_event)
        self.delayed_free_requests: OrderedDict[str, float] = OrderedDict()
        self.renewed_requests: OrderedDict[str, float] = OrderedDict()
        self.kv_lease_duration = kv_lease_duration
        self.lease_extension = kv_lease_duration * 2 // 3
        self.finished_requests: queue.SimpleQueue[str] = queue.SimpleQueue()
        self.state_lock = threading.Lock()

    def add_delayed_request(self, request_id: str) -> None:
        with self.state_lock:
            if request_id not in self.delayed_free_requests and request_id not in self.renewed_requests:
                self.delayed_free_requests[request_id] = time.monotonic() + self.kv_lease_duration

    def _handle_heartbeat(self, request_ids: Sequence[str]) -> None:
        with self.state_lock:
            now = time.monotonic()
            for request_id in request_ids:
                expiry = self.delayed_free_requests.get(request_id)
                if expiry is None:
                    expiry = self.renewed_requests.get(request_id)
                if expiry is None:
                    continue
                if now >= expiry:
                    self._mark_finished_locked(request_id)
                    logger.error("Force freed expired Mooncake request %s after KV lease expired", request_id)
                elif now + self.lease_extension > expiry:
                    self.delayed_free_requests.pop(request_id, None)
                    self.renewed_requests[request_id] = now + self.lease_extension
                    self.renewed_requests.move_to_end(request_id)

    def get_and_clear_finished_requests(self) -> set[str]:
        finished: set[str] = set()
        with self.state_lock:
            self._retrieve_expired_requests_locked()
        while True:
            try:
                finished.add(self.finished_requests.get_nowait())
            except queue.Empty:
                return finished

    def shutdown(self) -> None:
        self.stop_event.set()
        if self.ident is not None:
            self.join()
        with self.state_lock:
            self.delayed_free_requests.clear()
            self.renewed_requests.clear()
        while not self.finished_requests.empty():
            self.finished_requests.get_nowait()

    def run(self) -> None:
        path = make_zmq_path("tcp", self.host, self.port)
        try:
            logger.info("Mooncake scheduler sending thread listening on %s", path)
            with zmq_ctx(zmq.ROUTER, path) as sock:  # type: ignore[attr-defined]
                sock.setsockopt(zmq.SNDTIMEO, 1000)  # type: ignore[attr-defined]
                sock.setsockopt(zmq.RCVTIMEO, 1000)  # type: ignore[attr-defined]
                self.ready_event.set()
                self._run_busy_loop(sock)
        except Exception:
            self.ready_event.set()
            logger.exception("Mooncake scheduler sending thread failed on %s", path)

    def _run_busy_loop(self, sock: Any) -> None:
        decoder = msgspec.msgpack.Decoder(type=tuple)
        while not self.stop_event.is_set():
            try:
                frames = sock.recv_multipart()
            except zmq.Again:  # type: ignore[attr-defined]
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
                elif msg[0] == HEARTBEAT_MSG and len(msg) == 2:
                    if not isinstance(msg[1], (list, tuple)) or not all(isinstance(req_id, str) for req_id in msg[1]):
                        raise ValueError("Invalid Mooncake heartbeat request IDs")
                    self._handle_heartbeat(msg[1])
                    sock.send_multipart((identity, b"", ACK_MSG))
                else:
                    logger.warning("Unexpected Mooncake scheduler control message: %s", msg)
            except Exception:
                logger.exception("Failed to handle Mooncake scheduler control message")

    def _handle_finished_request(self, request_id: str) -> None:
        with self.state_lock:
            self._mark_finished_locked(request_id)

    def _mark_finished_locked(self, request_id: str) -> None:
        # request_finished registers the request before exposing its transfer
        # parameters to D. Unknown IDs are therefore stale or duplicate DONEs;
        # the control loop still ACKs them so D can stop retrying.
        if request_id not in self.delayed_free_requests and request_id not in self.renewed_requests:
            logger.warning(
                "Received Mooncake DONE for request %s with no pending delayed free; ignoring completion",
                request_id,
            )
            return
        self.delayed_free_requests.pop(request_id, None)
        self.renewed_requests.pop(request_id, None)
        self.finished_requests.put(request_id)

    def _retrieve_expired_requests_locked(self) -> None:
        now = time.monotonic()
        # Each queue has one fixed duration. Renewals only move to the tail
        # when the deadline increases, keeping both queues expiry-ordered.
        for requests in (self.delayed_free_requests, self.renewed_requests):
            while requests:
                request_id, expiry = next(iter(requests.items()))
                if now < expiry:
                    break
                self._mark_finished_locked(request_id)
                logger.error("Force freed expired Mooncake request %s after KV lease expired", request_id)


class MooncakeSchedulerRecvingThread(threading.Thread):
    """Send scheduler-level completion messages from D to P."""

    def __init__(self, ready_event: threading.Event) -> None:
        super().__init__(daemon=True, name="MooncakeSchedulerRecvingThread")
        self.ready_event = ready_event
        self.stop_event = threading.Event()
        self.request_queue: queue.Queue[tuple[str, int, str] | None] = queue.Queue()
        self.encoder = msgspec.msgpack.Encoder()
        self.remote_sockets: dict[str, deque[Any]] = defaultdict(deque)
        self.remote_sockets_lock = threading.Lock()
        self.zmq_context: Any | None = None

    def add_request(self, remote_host: str, remote_port: int, request_id: str) -> None:
        self.request_queue.put((remote_host, remote_port, request_id))

    def shutdown(self) -> None:
        if not self.stop_event.is_set():
            self.stop_event.set()
            self.request_queue.put(None)
        if self.ident is not None:
            self.join()
        # Also handle shutdown before start and a sentinel enqueued while the
        # thread was already completing its final cleanup.
        self._clear_pending_requests()

    def _clear_pending_requests(self) -> None:
        while True:
            try:
                self.request_queue.get_nowait()
            except queue.Empty:
                break
            self.request_queue.task_done()

    def run(self) -> None:
        self.ready_event.set()
        try:
            while not self.stop_event.is_set():
                request = self.request_queue.get()
                try:
                    if request is None or self.stop_event.is_set():
                        break
                    self._send_done_recving(*request)
                except Exception:
                    # The socket helpers already retry. Let P's delayed-free
                    # timeout reclaim blocks if this notification cannot arrive.
                    logger.exception(
                        "Failed to send Mooncake scheduler completion for request %s; "
                        "producer cleanup will rely on its delayed-free timeout",
                        request[2] if request is not None else None,
                    )
                finally:
                    self.request_queue.task_done()
        finally:
            self.stop_event.set()
            # Close sockets in their owning thread, after the last send ends.
            with self.remote_sockets_lock:
                for sockets in self.remote_sockets.values():
                    for sock in sockets:
                        sock.close(linger=0)
                self.remote_sockets.clear()
                if self.zmq_context is not None:
                    self.zmq_context.destroy(linger=0)
                    self.zmq_context = None
            self._clear_pending_requests()

    def _send_done_recving(self, remote_host: str, remote_port: int, request_id: str) -> None:
        path = make_zmq_path("tcp", remote_host, remote_port)
        sock = self._get_remote_socket(path)
        try:
            ensure_zmq_send(
                sock,
                self.encoder.encode((DONE_RECVING_MSG, request_id)),
                path,
            )
            response = ensure_zmq_recv(sock, path)
            if response != ACK_MSG:
                raise RuntimeError(f"Unexpected Mooncake scheduler completion response: {response!r}")
        except Exception:
            sock.close(linger=0)
            raise
        self._return_remote_socket(path, sock)

    def _get_remote_socket(self, path: str) -> Any:
        """Borrow a persistent REQ socket for one producer scheduler."""
        with self.remote_sockets_lock:
            sockets = self.remote_sockets[path]
            if sockets:
                return sockets.popleft()

            if self.zmq_context is None:
                self.zmq_context = zmq.Context()  # type: ignore[attr-defined]
            sock = make_zmq_socket(
                ctx=self.zmq_context,
                path=path,
                socket_type=zmq.REQ,  # type: ignore[attr-defined]
                bind=False,
            )
            sock.setsockopt(zmq.SNDTIMEO, 1000)  # type: ignore[attr-defined]
            sock.setsockopt(zmq.RCVTIMEO, 1000)  # type: ignore[attr-defined]
            return sock

    def _return_remote_socket(self, path: str, sock: Any) -> None:
        """Return a healthy REQ socket to its endpoint pool."""
        with self.remote_sockets_lock:
            self.remote_sockets[path].append(sock)


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
        # Successful validations expire and have a bounded endpoint count.
        # Only the scheduler thread accesses this cache.
        self._validated_producer_leases: TTLCache[tuple[str, int], tuple[str, int]] = TTLCache(
            maxsize=PRODUCER_LEASE_CACHE_MAX_SIZE,
            ttl=PRODUCER_LEASE_CACHE_TTL_SECONDS,
            timer=time.monotonic,
        )
        self._sending_thread: MooncakeSchedulerSendingThread | None = None
        self._recving_thread: MooncakeSchedulerRecvingThread | None = None
        self._heartbeat_thread: MooncakeHeartbeatThread | None = None
        self._kv_lease_duration = self.kv_transfer_config.get_from_extra_config(
            "kv_lease_duration", DEFAULT_KV_LEASE_DURATION
        )
        if type(self._kv_lease_duration) is not int or self._kv_lease_duration < 6:
            raise ValueError("Mooncake kv_lease_duration must be an integer of at least 6 seconds")

        if self.kv_transfer_config.is_kv_consumer:
            recving_ready_event = threading.Event()
            self._recving_thread = MooncakeSchedulerRecvingThread(recving_ready_event)
            self._recving_thread.start()
            recving_ready_event.wait()
            self._heartbeat_thread = MooncakeHeartbeatThread(self._kv_lease_duration // 6)
            self._heartbeat_thread.start()

    def on_new_request(self, request: "Request") -> None:
        """Track remote KV leases from scheduler admission until receive completion."""
        super().on_new_request(request)
        params = request.kv_transfer_params
        if (
            self._heartbeat_thread is not None
            and params is not None
            and params.get("do_remote_prefill")
            and any(params.get("remote_block_ids", ()))
            and all(key in params for key in ("remote_host", "remote_port", "remote_request_id", "remote_engine_id"))
        ):
            endpoint = (params["remote_host"], params["remote_port"])
            remote_duration = params.get("remote_kv_lease_duration")
            producer_config = (params["remote_engine_id"], remote_duration)
            # Reuse successful validation while this P's instance and lease
            # configuration stay unchanged. Checking the advertised duration
            # also covers restarts that reuse an explicitly configured engine ID.
            if type(remote_duration) is not int or self._validated_producer_leases.get(endpoint) != producer_config:
                self._validate_remote_lease_duration(remote_duration, *endpoint)
                self._validated_producer_leases[endpoint] = producer_config
            self._heartbeat_thread.add_request(
                request.request_id, params["remote_host"], params["remote_port"], params["remote_request_id"]
            )

    def _validate_remote_lease_duration(self, remote_duration: Any, host: str, port: int) -> None:
        """Validate a new producer configuration before any heartbeat or READ."""
        if type(remote_duration) is not int or remote_duration < 6:
            raise ValueError(
                "Mooncake producer must provide remote_kv_lease_duration as an integer of at least 6 seconds; "
                f"got {remote_duration!r} from {host}:{port}. "
                "Upgrade P before D and preserve kv_transfer_params through the proxy."
            )
        if remote_duration != self._kv_lease_duration:
            raise ValueError(
                "Mooncake kv_lease_duration mismatch: "
                f"P ({host}:{port})={remote_duration}s, "
                f"D={self._kv_lease_duration}s. Configure the same kv_lease_duration on P and D."
            )

    def shutdown(self) -> None:
        if self._heartbeat_thread is not None:
            self._heartbeat_thread.shutdown()
        if self._sending_thread is not None:
            self._sending_thread.shutdown()
        if self._recving_thread is not None:
            self._recving_thread.shutdown()
        self._reqs_need_recv.clear()
        self._reqs_need_send.clear()
        self._reqs_in_batch.clear()
        self._reqs_recv_info.clear()
        self._validated_producer_leases.clear()

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
            self.ascend_config.kvpp_config.size > 1,
            ready_event,
            kv_lease_duration=self._kv_lease_duration,
        )
        self._sending_thread.start()
        if not ready_event.wait(timeout=10):
            raise RuntimeError("Timed out starting Mooncake scheduler sending thread")
        if not self._sending_thread.is_alive():
            raise RuntimeError("Mooncake scheduler sending thread failed to start")

    def update_state_after_alloc(
        self,
        request: "Request",
        blocks: "KVCacheBlocks",
        num_external_tokens: int,
    ) -> None:
        self._log_mamba_request_allocation(request.request_id, blocks)
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
                    if self._heartbeat_thread is not None:
                        self._heartbeat_thread.remove_request(request.request_id)
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
        """Release unstarted consumer transfers or expose completed producer blocks."""
        params = request.kv_transfer_params
        # Keep renewing in-flight READs even after client cancellation. Their
        # workers must finish before P can release the source blocks.
        if self._heartbeat_thread is not None and request.request_id not in self._reqs_recv_info:
            # The unallocated remote-prefill path below owns the DONE notification.
            self._heartbeat_thread.remove_request(request.request_id)
        logger.debug(
            "MooncakeConnector request_finished: request_status=%s, kv_transfer_params=%s",
            request.status,
            params,
        )

        if params is not None and params.get("do_remote_prefill"):
            # Allocation clears this flag before starting a READ. A rejected
            # or cancelled request that still has it never started receiving,
            # so notify P directly instead of leaving its blocks pinned.
            if params.get("remote_block_ids"):
                required_remote_fields = ("remote_host", "remote_port", "remote_request_id")
                if all(field in params for field in required_remote_fields):
                    if self._recving_thread is None:
                        raise RuntimeError("Producer Mooncake scheduler cannot acknowledge a receive request")
                    self._recving_thread.add_request(
                        params["remote_host"], params["remote_port"], params["remote_request_id"]
                    )
                else:
                    logger.warning("Got invalid KVTransferParams. params=%s.", params)
            # Make repeated finish callbacks harmless, including zero-block
            # requests. Transfers already handed to workers use their normal
            # aggregated receive-completion notification instead.
            params["do_remote_prefill"] = False
            return False, None

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
            delay_start_time = time.monotonic()
            self._reqs_need_send[request.request_id] = delay_start_time
            if self._sending_thread is None:
                raise RuntimeError("Mooncake scheduler metadata has not been initialized")
            self._sending_thread.add_delayed_request(request.request_id)

        return delay_free_blocks, {
            "do_remote_prefill": True,
            "do_remote_decode": False,
            "remote_block_ids": computed_block_ids,
            "remote_num_prompt_tokens": prompt_len,
            "remote_engine_id": self.engine_id,
            "remote_request_id": request.request_id,
            "remote_host": self.side_channel_host,
            "remote_port": self.side_channel_port,
            "remote_kv_lease_duration": self._kv_lease_duration,
            "last_token_id": request.output_token_ids[-1],
        }

    def update_connector_output(
        self,
        connector_output: KVConnectorOutput,
    ) -> None:
        # D side: this output has already aggregated completion from all
        # workers. Send one scheduler-to-scheduler ACK for the request.
        for req_id in connector_output.finished_recving or ():
            if self._heartbeat_thread is not None:
                self._heartbeat_thread.remove_request(req_id)
            remote = self._reqs_recv_info.pop(req_id, None)
            if remote is not None:
                if self._recving_thread is None:
                    raise RuntimeError("Producer Mooncake scheduler received a receive-completion event")
                self._recving_thread.add_request(*remote)
                if req_id in connector_output.failed_recving:
                    logger.error(
                        "Mooncake D receive failed after worker aggregation: request_id=%s, "
                        "remote_request_id=%s, producer=%s:%s, failure_policy=%s; producer release notification queued",
                        req_id,
                        remote[2],
                        remote[0],
                        remote[1],
                        self.kv_transfer_config.kv_load_failure_policy,
                    )

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
