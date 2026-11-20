# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Scheduler-side logic for Mooncake pull transfers."""

import hashlib
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
# Bidirectional KV liveness probe: P asks D's scheduler sending thread whether
# a session's blocks are still pinned before pulling. The proxy cannot sense
# early D-side eviction (LRU under budget pressure), and pulling freed blocks
# would fail or read reallocated memory.
CHECK_PENDING_MSG = b"check_pending_msg"
PENDING_YES_MSG = b"pending_yes"
PENDING_NO_MSG = b"pending_no"
PRODUCER_LEASE_CACHE_TTL_SECONDS = 600
PRODUCER_LEASE_CACHE_MAX_SIZE = 1024

# Prefix length (hex chars) of each content-hash in the bidirectional KV
# block-hash chain. 16 hex chars (64 bits) per block is collision-safe for
# alignment verification while keeping the per-request JSON payload compact.
BIDIR_BLOCK_HASH_HEX_LEN = 16

# Hash granularity for the bidirectional KV alignment chain, in tokens. This
# is deliberately independent of the scheduler-level block size (which for
# hybrid-attention models under HMA is the lcm of per-group block sizes, e.g.
# 896 for Qwen3.5 — far too coarse: a 4527-token prompt + 300-token decode
# output both span exactly 5 x 896-token blocks, so no decode-side KV would
# ever be recovered). 128 matches the physical attention-group block size
# (the unit of KV transfer) and keeps ~40 hashes per 5k-token request.
BIDIR_HASH_BLOCK_SIZE = 128


def _bidir_block_hash(parent: str, block_tokens: Sequence[int]) -> str:
    """Hash one chain step: ``sha256(parent + "," + tokens)`` truncated.

    Shared by the full-block chain and the sequence tail so both sides
    compute identical values for the same token span and parent.
    """
    digest = hashlib.sha256()
    digest.update(parent.encode("ascii"))
    digest.update(b",")
    digest.update(",".join(str(token) for token in block_tokens).encode("ascii"))
    return digest.hexdigest()[:BIDIR_BLOCK_HASH_HEX_LEN]


def _bidir_block_hash_chain(
    token_ids: Sequence[int],
    block_size: int,
    max_blocks: int | None = None,
) -> list[str]:
    """Content-addressed hash chain over consecutive blocks of ``token_ids``.

    Each block's hash incorporates its parent's hash (chained, like vLLM's
    prefix-cache block hashes), so a match at block ``i`` proves the whole
    prefix ``[0, (i + 1) * block_size)`` is token-identical. The chain is
    computed identically on the D side (over prompt + decode output) and on
    the P side (over the next turn's prompt), so the longest common chain
    prefix is the exact token-alignment boundary between D's cached KV and
    the new request.
    """
    num_blocks = len(token_ids) // block_size
    if max_blocks is not None:
        num_blocks = min(num_blocks, max_blocks)
    hashes: list[str] = []
    parent = ""
    for block_index in range(num_blocks):
        block_tokens = token_ids[block_index * block_size : (block_index + 1) * block_size]
        parent = _bidir_block_hash(parent, block_tokens)
        hashes.append(parent)
    return hashes


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
        # request_id -> block count; total feeds the D pinned-block budget.
        self.delayed_free_blocks: dict[str, int] = {}
        self.pinned_block_total = 0

    def add_delayed_request(self, request_id: str) -> None:
        with self.state_lock:
            if request_id not in self.delayed_free_requests and request_id not in self.renewed_requests:
                self.delayed_free_requests[request_id] = time.monotonic() + self.kv_lease_duration

    def add_delayed_request_with_blocks(
        self, request_id: str, num_blocks: int, ttl: int | None = None
    ) -> None:
        """Track a delayed-free request together with its block count.

        Same registration semantics as ``add_delayed_request`` (which uses
        the producer KV lease duration) but additionally accounts the
        request's block count against the D-node pinned-block budget;
        ``ttl`` lets the bidirectional decoder path expire blocks with
        its own TTL instead of the lease duration.
        """
        with self.state_lock:
            if request_id not in self.delayed_free_requests and request_id not in self.renewed_requests:
                self.delayed_free_requests[request_id] = time.monotonic() + (
                    ttl if ttl is not None else self.kv_lease_duration
                )
                self.delayed_free_blocks[request_id] = num_blocks
                self.pinned_block_total += num_blocks

    @property
    def total_pinned_blocks(self) -> int:
        with self.state_lock:
            return self.pinned_block_total

    def _handle_heartbeat(self, request_ids: Sequence[str]) -> None:
        extended = unchanged = unknown = 0
        expired: list[tuple[str, float]] = []
        with self.state_lock:
            now = time.monotonic()
            for request_id in request_ids:
                expiry = self.delayed_free_requests.get(request_id)
                if expiry is None:
                    expiry = self.renewed_requests.get(request_id)
                if expiry is None:
                    unknown += 1
                    continue
                if now >= expiry:
                    self._mark_finished_locked(request_id)
                    expired.append((request_id, now - expiry))
                elif now + self.lease_extension > expiry:
                    self.delayed_free_requests.pop(request_id, None)
                    self.renewed_requests[request_id] = now + self.lease_extension
                    self.renewed_requests.move_to_end(request_id)
                    extended += 1
                else:
                    unchanged += 1
        logger.debug(
            "MooncakeHeartbeat event=process engine_id=%s received=%d extended=%d unchanged=%d expired=%d unknown=%d",
            self.engine_id,
            len(request_ids),
            extended,
            unchanged,
            len(expired),
            unknown,
        )
        for request_id, expired_by in expired:
            self._log_lease_finish(request_id, expired_by)

    def get_and_clear_finished_requests(self) -> set[str]:
        finished: set[str] = set()
        with self.state_lock:
            expired = self._retrieve_expired_requests_locked()
        for request_id, expired_by in expired:
            self._log_lease_finish(request_id, expired_by)
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
        # Proactive expiry check interval: expired delayed-free blocks
        # are released periodically even when no new requests arrive, so
        # the pinned-block budget is replenished without depending on
        # traffic.
        last_expiry_sweep = time.monotonic()
        EXPIRY_SWEEP_INTERVAL_S = 30.0
        while not self.stop_event.is_set():
            try:
                frames = sock.recv_multipart()
            except zmq.Again:  # type: ignore[attr-defined]
                # ZMQ RCVTIMEO (1s) expired with no incoming message —
                # use this idle tick to proactively release expired
                # blocks.
                now = time.monotonic()
                if now - last_expiry_sweep >= EXPIRY_SWEEP_INTERVAL_S:
                    with self.state_lock:
                        self._retrieve_expired_requests_locked()
                    last_expiry_sweep = now
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
                elif msg[0] == CHECK_PENDING_MSG and len(msg) == 2:
                    # Bidirectional KV liveness probe: whether this session's
                    # blocks are still pinned (delay-free or heartbeat-renewed).
                    with self.state_lock:
                        pending = str(msg[1]) in self.delayed_free_requests or str(
                            msg[1]
                        ) in self.renewed_requests
                    sock.send_multipart((identity, b"", PENDING_YES_MSG if pending else PENDING_NO_MSG))
                else:
                    logger.warning("Unexpected Mooncake scheduler control message: %s", msg)
            except Exception:
                logger.exception("Failed to handle Mooncake scheduler control message")

    def _handle_finished_request(self, request_id: str) -> None:
        with self.state_lock:
            finished = self._mark_finished_locked(request_id)
        if finished:
            self._log_lease_finish(request_id)
        else:
            logger.warning(
                "Received Mooncake DONE for request %s with no pending delayed free; ignoring completion",
                request_id,
            )

    def _mark_finished_locked(self, request_id: str) -> bool:
        # request_finished registers the request before exposing its transfer
        # parameters to D. Unknown IDs are therefore stale or duplicate DONEs;
        # the control loop still ACKs them so D can stop retrying.
        if request_id not in self.delayed_free_requests and request_id not in self.renewed_requests:
            return False
        self.delayed_free_requests.pop(request_id, None)
        self.renewed_requests.pop(request_id, None)
        released = self.delayed_free_blocks.pop(request_id, None)
        if released is not None:
            self.pinned_block_total -= released
        self.finished_requests.put(request_id)
        return True

    def _log_lease_finish(self, request_id: str, expired_by: float | None = None) -> None:
        # Call only after releasing state_lock; logging must not hold up lease updates.
        if expired_by is None:
            logger.debug(
                "MooncakeHeartbeat event=lease_finish engine_id=%s request_id=%s reason=done action=queued_for_release",
                self.engine_id,
                request_id,
            )
        else:
            logger.error(
                "Force freed expired Mooncake request %s after KV lease expired; "
                "MooncakeHeartbeat event=lease_finish engine_id=%s request_id=%s "
                "reason=expired expired_by_s=%.3f action=queued_for_release",
                request_id,
                self.engine_id,
                request_id,
                expired_by,
            )

    def _retrieve_expired_requests_locked(self) -> list[tuple[str, float]]:
        now = time.monotonic()
        expired: list[tuple[str, float]] = []
        # Each queue has one fixed duration. Renewals only move to the tail
        # when the deadline increases, keeping both queues expiry-ordered.
        for requests in (self.delayed_free_requests, self.renewed_requests):
            while requests:
                request_id, expiry = next(iter(requests.items()))
                if now < expiry:
                    break
                self._mark_finished_locked(request_id)
                expired.append((request_id, now - expiry))
        return expired

    def sweep_expired_blocks(self) -> int:
        """Proactively release blocks whose TTL has expired.

        Called from the scheduler's budget check (before deciding whether to
        reject a new request) and from the busy-loop idle sweep. Returns the
        number of expired requests freed.
        """
        with self.state_lock:
            before = len(self.delayed_free_requests) + len(self.renewed_requests)
            self._retrieve_expired_requests_locked()
            freed = before - (len(self.delayed_free_requests) + len(self.renewed_requests))
        if freed > 0:
            logger.info(
                "Mooncake expiry sweep: freed %d expired delayed requests (%d blocks still pinned)",
                freed,
                self.pinned_block_total,
            )
        return freed

    def evict_oldest_for_budget(self, needed_blocks: int, budget: int) -> int:
        """Evict the oldest delayed blocks (LRU) until enough budget is free.

        Both queues are expiry-ordered (oldest first). Evicting from the
        front releases blocks closest to their TTL expiry — the least
        valuable for a next-turn pull. Returns the number of evicted
        requests.
        """
        evicted = 0
        with self.state_lock:
            while self.pinned_block_total + needed_blocks > budget and (
                self.delayed_free_requests or self.renewed_requests
            ):
                if self.delayed_free_requests:
                    request_id = next(iter(self.delayed_free_requests))
                else:
                    request_id = next(iter(self.renewed_requests))
                self._mark_finished_locked(request_id)
                evicted += 1
                logger.warning(
                    "Mooncake LRU eviction: force-freed request %s to reclaim budget "
                    "(pinned=%d, needed=%d, budget=%d)",
                    request_id,
                    self.pinned_block_total,
                    needed_blocks,
                    budget,
                )
        return evicted


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
        # Bidirectional pull-in-flight state on P: request -> (Request, external
        # token count). Used to undo the external-token accounting if the
        # pull fails after the pre-scheduling liveness check passed.
        self._bidir_pull_state: dict[str, tuple[Request, int]] = {}
        # Encoder for the bidirectional liveness probe (scheduler thread only).
        self._bidir_check_encoder = msgspec.msgpack.Encoder()
        self._bidir_check_ctx: Any | None = None
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

        # D-node bidirectional pinned-block budget: cap total blocks held
        # for D-to-P reuse so multi-turn bursts cannot exhaust the KV
        # pool. When the budget is exceeded, expired blocks are swept and
        # the oldest delayed blocks are evicted (LRU); only when the pool
        # itself is too small do we skip delay-free entirely.
        total_blocks = kv_cache_config.num_blocks
        self.decoder_kv_block_budget = int(
            vllm_config.kv_transfer_config.get_from_extra_config(
                "decoder_kv_block_budget", max(1, total_blocks // 5)
            )
        )
        logger.info(
            "Bidirectional KV: decoder pinned-block budget=%d (pool=%d blocks)",
            self.decoder_kv_block_budget,
            total_blocks,
        )

        # The recving thread handles scheduler-to-scheduler DONE
        # notifications after pull transfers complete. It is needed by
        # kv_consumer nodes (standard D pulls from P) and by kv_producer
        # nodes under bidirectional transfer (P pulls from D and must
        # ACK D on completion). The heartbeat thread stays D-only.
        needs_recving_thread = self.kv_transfer_config.is_kv_consumer or (
            self.is_bidirectional_kv_xfer_enabled and self.kv_transfer_config.is_kv_producer
        )
        if needs_recving_thread:
            recving_ready_event = threading.Event()
            self._recving_thread = MooncakeSchedulerRecvingThread(recving_ready_event)
            self._recving_thread.start()
            recving_ready_event.wait()
            if self.kv_transfer_config.is_kv_consumer:
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
        # The sending thread serves kv_producer nodes (standard P-to-D)
        # and also kv_consumer nodes when bidirectional KV transfer is
        # enabled (D must expose its blocks so P can pull them next turn).
        needs_sending_thread = self.kv_transfer_config.is_kv_producer or (
            self.is_bidirectional_kv_xfer_enabled and self.kv_transfer_config.is_kv_consumer
        )
        if not needs_sending_thread or not metadata or self._sending_thread is not None:
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

    def _check_remote_pending(self, params: dict[str, Any]) -> bool:
        """Ask D's scheduler whether this session's blocks are still pinned.

        The proxy cannot sense early D-side eviction (LRU under budget
        pressure, expiry sweep, engine restart); its session cache only
        knows the TTL-based ``remote_blocks_expiry_time``. A "no" (or any
        probe error, including timeout — correctness over performance)
        makes the caller fall back to a full local prefill.
        """
        path = make_zmq_path("tcp", str(params["remote_host"]), int(params["remote_port"]))
        sock = None
        try:
            if self._bidir_check_ctx is None:
                self._bidir_check_ctx = zmq.Context()  # type: ignore[attr-defined]
            sock = make_zmq_socket(
                ctx=self._bidir_check_ctx,
                path=path,
                socket_type=zmq.REQ,  # type: ignore[attr-defined]
                bind=False,
            )
            sock.setsockopt(zmq.SNDTIMEO, 2000)  # type: ignore[attr-defined]
            sock.setsockopt(zmq.RCVTIMEO, 2000)  # type: ignore[attr-defined]
            ensure_zmq_send(
                sock,
                self._bidir_check_encoder.encode(
                    (CHECK_PENDING_MSG, str(params["remote_request_id"]))
                ),
                path,
            )
            return ensure_zmq_recv(sock, path) == PENDING_YES_MSG
        except Exception:
            logger.warning(
                "Bidirectional KV liveness probe to %s failed for request %s; "
                "assuming remote blocks are gone (local prefill fallback)",
                path,
                params.get("remote_request_id"),
            )
            return False
        finally:
            if sock is not None:
                sock.close(linger=0)

    def get_num_new_matched_tokens(
        self, request: "Request", num_computed_tokens: int
    ) -> tuple[int, bool]:
        """Return prompt tokens that will be loaded from a remote producer."""
        standard_count, standard_load_async = super().get_num_new_matched_tokens(
            request, num_computed_tokens
        )
        if standard_count > 0:
            return standard_count, standard_load_async

        params = request.kv_transfer_params
        # Bidirectional KV transfer: P pulls blocks from D
        if (
            self.is_bidirectional_kv_xfer_enabled
            and params is not None
            and params.get("do_remote_decode")
            and params.get("remote_block_ids")
            and all(
                p in params
                for p in (
                    "remote_engine_id",
                    "remote_request_id",
                    "remote_host",
                    "remote_port",
                )
            )
        ):
            # Decode node has KV blocks for part of this prefill request, so
            # provide them as an external token count to the scheduler. Use
            # synchronous load (load_kv_async=False) with reserve-1 to avoid
            # the async-load deadlock (WAITING_FOR_REMOTE_KVS never resolves
            # because no step is produced to trigger start_load_kv).
            #
            # Liveness check: D may have evicted this session's blocks before
            # their TTL (LRU under budget pressure, expiry sweep, engine
            # restart). The proxy cannot sense early eviction, and pulling
            # freed blocks would fail or read reallocated memory. Ask D's
            # scheduler directly; "gone" (or any probe error) falls back to a
            # full local prefill, which is always correct.
            if not self._check_remote_pending(params):
                logger.info(
                    "Bidirectional KV: remote blocks for request %s are no longer "
                    "pinned on D; falling back to local prefill",
                    request.request_id,
                )
                return 0, False
            #
            # Alignment boundary (hash index preferred, conservative fallback):
            # 1. Hash index: if D reported a content-addressed block hash chain
            #    (remote_block_hashes), walk this request's prompt chain against
            #    it block by block. The longest matching chain prefix is the
            #    exact token-alignment boundary, which also recovers the
            #    assistant-text KV from the previous turn's decode output (the
            #    conservative bound deliberately excludes it).
            # 2. Conservative fallback: bound the pull by D's reported prompt
            #    length (remote_num_prompt_tokens). Only D's prompt-side KV is
            #    guaranteed token-aligned with this prompt; decode output
            #    (thinking tokens, template tails) does not reliably reappear.
            remote_block_hashes = params.get("remote_block_hashes") or None
            if remote_block_hashes:
                remote_block_size = params.get("remote_block_size") or self.block_size
                local_token_ids = list(request.prompt_token_ids or [])
                local_hashes = _bidir_block_hash_chain(
                    local_token_ids,
                    remote_block_size,
                    max_blocks=len(remote_block_hashes),
                )
                matched_blocks = 0
                for local_hash, remote_hash in zip(local_hashes, remote_block_hashes):
                    if local_hash != remote_hash:
                        break
                    matched_blocks += 1
                aligned_tokens = matched_blocks * remote_block_size
                # Full-sequence alignment check: all full blocks matched AND
                # the tail hash (tokens beyond the last full block) matches.
                # Only then does D's final state snapshot land exactly on the
                # resume boundary — required for stateful (hybrid) models.
                fully_aligned = False
                remote_full_len = params.get("remote_num_tokens") or 0
                remote_tail_hash = params.get("remote_tail_hash")
                if (
                    matched_blocks == len(remote_block_hashes)
                    and remote_tail_hash is not None
                    and remote_full_len > 0
                ):
                    tail_start = matched_blocks * remote_block_size
                    local_tail = local_token_ids[tail_start:remote_full_len]
                    if len(local_tail) == remote_full_len - tail_start:
                        tail_parent = remote_block_hashes[-1] if remote_block_hashes else ""
                        fully_aligned = (
                            _bidir_block_hash(tail_parent, local_tail) == remote_tail_hash
                        )
                        if fully_aligned:
                            aligned_tokens = remote_full_len
                if self.has_stateful_groups:
                    # Stateful models: D's mamba state is a single snapshot at
                    # the END of its sequence. Pulling a partial prefix would
                    # hand P a wrong initial state (fluent but incorrect
                    # output). Only a verified FULL alignment may be pulled,
                    # and only when the prompt continues past it (no
                    # reserve-1: the state is valid exactly at the boundary).
                    if not fully_aligned:
                        logger.info(
                            "Bidirectional KV: stateful model requires full-sequence "
                            "alignment, matched %d/%d blocks (tail=%s); skipping pull for %s",
                            matched_blocks,
                            len(remote_block_hashes),
                            "present" if remote_tail_hash is not None else "absent",
                            request.request_id,
                        )
                        return 0, False
                    count = aligned_tokens - num_computed_tokens
                    if count <= 0 or aligned_tokens >= request.num_prompt_tokens:
                        # Nothing new to compute: reserve-1 would shift the
                        # state boundary by one — unsafe on stateful models.
                        return 0, False
                    logger.info(
                        "Bidirectional KV: full-sequence alignment verified "
                        "(%d tokens), pulling for stateful request %s",
                        aligned_tokens,
                        request.request_id,
                    )
                    return count, False
                count = min(aligned_tokens, request.num_prompt_tokens) - num_computed_tokens
                logger.info(
                    "Bidirectional KV hash-align: matched %d blocks (%d tokens) of "
                    "%d remote hashes; prompt_tokens=%d, remote_prompt_tokens=%s, "
                    "num_computed_tokens=%d, external_count=%d",
                    matched_blocks,
                    aligned_tokens,
                    len(remote_block_hashes),
                    request.num_prompt_tokens,
                    params.get("remote_num_prompt_tokens"),
                    num_computed_tokens,
                    max(count, 0),
                )
            else:
                # Stateful models cannot use the conservative fallback: the
                # prompt-bound prefix alone never carries D's final state.
                if self.has_stateful_groups:
                    return 0, False
                # Conservative fallback: no hash chain reported (older D-side
                # metadata cached by the proxy, or D running a pre-hash build).
                remote_prompt_tokens = params.get("remote_num_prompt_tokens") or 0
                count = min(remote_prompt_tokens, request.num_prompt_tokens) - num_computed_tokens
            # Reserve the last token for local prefill to avoid num_new_tokens=0.
            if count >= request.num_prompt_tokens - num_computed_tokens:
                count = max(count - 1, 0)
            if count > 0:
                return count, False

        return 0, False

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

        if params is not None and params.get("do_remote_prefill"):
            # Standard P->D prefill path (fall through to the
            # bidirectional path below when the flag is absent).
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
            return

        # Bidirectional KV transfer: P pulls blocks from D
        if (
            self.is_bidirectional_kv_xfer_enabled
            and params is not None
            and params.get("do_remote_decode")
            and params.get("remote_block_ids")
            and not params.get("_remote_blocks_processed")
        ):
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
                    # Track the in-flight pull so update_connector_output can
                    # revert the external-token accounting if the pull fails
                    # after the pre-scheduling liveness check passed.
                    self._bidir_pull_state[request.request_id] = (
                        request,
                        num_external_tokens,
                    )
                else:
                    if self._recving_thread is None:
                        raise RuntimeError("Producer Mooncake scheduler cannot acknowledge a receive request")
                    self._recving_thread.add_request(*remote)
            else:
                logger.warning("Got invalid KVTransferParams for bidirectional transfer. params=%s.", params)

            # Mark as processed to avoid duplicate transfers
            params["_remote_blocks_processed"] = True

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
        # Drop any stale bidirectional pull-in-flight tracking for this request.
        self._bidir_pull_state.pop(request.request_id, None)
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

        # Bidirectional KV transfer: D advertises its full-sequence KV
        # (prompt + decode output) so the next turn's P can pull it.
        # On a bidir D node, input params have do_remote_prefill cleared
        # by update_state_after_alloc once the P-side pull started, so
        # the request falls through to here. On a non-bidir D node the
        # gate below returns None, matching upstream's behavior.
        is_d_node = self.kv_transfer_config.is_kv_consumer
        if not (self.is_bidirectional_kv_xfer_enabled and is_d_node):
            return False, None
        # Only delay free for successful completions.
        if request.status not in (
            RequestStatus.FINISHED_LENGTH_CAPPED,
            RequestStatus.FINISHED_STOPPED,
        ):
            return False, None

        # Extend the advertised block set to the FULL token sequence
        # (prompt + decode output) so the next turn's P node can also
        # pull the decode-generated KV (assistant text), not just the
        # prompt side.
        prompt_token_ids = request.prompt_token_ids or []
        prompt_len = len(prompt_token_ids)
        output_token_ids = list(request.output_token_ids or [])
        full_seq = list(prompt_token_ids) + output_token_ids
        advertised_len = len(full_seq)
        computed_block_ids = self._get_transfer_block_ids(block_ids, advertised_len)
        num_computed_blocks = sum(len(group_block_ids) for group_block_ids in computed_block_ids)
        if num_computed_blocks <= 0:
            return False, None

        # D-node memory budget: cap total pinned bidirectional blocks
        # so a burst of multi-turn sessions cannot exhaust the KV pool.
        # When the budget is exceeded, first sweep expired blocks; then
        # evict the OLDEST delayed blocks (LRU). Only when the pool
        # itself is too small do we skip delay-free entirely.
        current_pinned = (
            self._sending_thread.total_pinned_blocks if self._sending_thread is not None else 0
        )
        if current_pinned + num_computed_blocks > self.decoder_kv_block_budget:
            if self._sending_thread is not None:
                self._sending_thread.sweep_expired_blocks()
                evicted = self._sending_thread.evict_oldest_for_budget(
                    num_computed_blocks, self.decoder_kv_block_budget
                )
                if evicted > 0:
                    logger.info(
                        "Bidirectional KV: evicted %d oldest delayed blocks (LRU) "
                        "to make budget for request %s",
                        evicted,
                        request.request_id,
                    )
                current_pinned = self._sending_thread.total_pinned_blocks
            if current_pinned + num_computed_blocks > self.decoder_kv_block_budget:
                logger.warning(
                    "Bidirectional KV: block budget exhausted (%d+%d > %d after expiry "
                    "sweep + LRU eviction), skipping delay-free for request %s",
                    current_pinned,
                    num_computed_blocks,
                    self.decoder_kv_block_budget,
                    request.request_id,
                )
                return False, None

        # Content-addressed block hash chain over the full sequence.
        # Only the D node reports it; the next turn's P node walks its
        # own prompt chain against this chain to find the exact
        # token-alignment boundary.
        remote_block_hashes = _bidir_block_hash_chain(full_seq, BIDIR_HASH_BLOCK_SIZE)
        # Tail hash: covers the tokens beyond the last full hash block.
        # Stateful (hybrid) models may only pull when the ENTIRE D-side
        # sequence aligns with the next prompt prefix, because D's
        # mamba state is a single snapshot at the end of the sequence;
        # the tail hash makes the full-alignment boundary verifiable.
        remote_tail_hash: str | None = None
        tail_start = len(remote_block_hashes) * BIDIR_HASH_BLOCK_SIZE
        if advertised_len > tail_start:
            tail_parent = remote_block_hashes[-1] if remote_block_hashes else ""
            remote_tail_hash = _bidir_block_hash(tail_parent, full_seq[tail_start:advertised_len])

        delay_start_time = time.monotonic()
        self._reqs_need_send[request.request_id] = delay_start_time
        if self._sending_thread is None:
            raise RuntimeError("Mooncake scheduler metadata has not been initialized")
        self._sending_thread.add_delayed_request_with_blocks(
            request.request_id, num_computed_blocks, ttl=self.decoder_kv_blocks_ttl
        )
        logger.info(
            "Bidirectional KV: Delaying free of %d blocks for request %s with TTL %d seconds",
            num_computed_blocks,
            request.request_id,
            self.decoder_kv_blocks_ttl,
        )

        # remote_num_tokens is the EXACT advertised sequence length
        # (prompt + output) so the next-turn P node can verify
        # full-sequence alignment against the tail hash.
        # NOTE: remote_blocks_expiry_time uses the wall clock (not monotonic)
        # because it crosses process boundaries (D -> proxy -> P) and the
        # proxy compares it against time.time().
        return True, {
            "do_remote_prefill": False,
            "do_remote_decode": True,
            "remote_block_ids": computed_block_ids,
            "remote_num_tokens": advertised_len,
            "remote_num_prompt_tokens": prompt_len,
            "remote_block_hashes": remote_block_hashes,
            "remote_tail_hash": remote_tail_hash,
            "remote_block_size": BIDIR_HASH_BLOCK_SIZE,
            "remote_engine_id": self.engine_id,
            "remote_request_id": request.request_id,
            "remote_host": self.side_channel_host,
            "remote_port": self.side_channel_port,
            "remote_blocks_expiry_time": time.time() + self.decoder_kv_blocks_ttl,
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

        # Bidirectional KV transfer on P: the synchronous bidir load already
        # completed within the scheduling step, and the request has since been
        # freed. Scheduler._update_from_kv_xfer_finished would assert
        # (req_id not in self.requests) if it saw these sets again —
        # consume and drop them after the ACK to D was queued above.
        if self.is_bidirectional_kv_xfer_enabled and self.kv_transfer_config.is_kv_producer:
            for req_id in connector_output.finished_recving or ():
                pull_state = self._bidir_pull_state.pop(req_id, None)
                if (
                    pull_state is not None
                    and req_id in (connector_output.failed_recving or ())
                ):
                    # The pull failed AFTER the pre-scheduling liveness check
                    # passed (D evicted this session's blocks mid-flight). The
                    # request is still mid-scheduling and its allocated blocks
                    # were never loaded: undo the external-token accounting so
                    # the next scheduling round re-prefills those tokens
                    # locally. (Mirrors upstream _handle_failed_recving's
                    # recompute semantics; the upstream WAITING_FOR_REMOTE_KVS
                    # gate cannot reach the synchronous bidir load path.)
                    failed_request, external_count = pull_state
                    failed_request.num_computed_tokens = max(
                        0, failed_request.num_computed_tokens - external_count
                    )
                    logger.error(
                        "Bidirectional KV pull failed for request %s; reverting "
                        "%d external tokens to local prefill",
                        req_id,
                        external_count,
                    )
            connector_output.finished_recving = None
            connector_output.failed_recving = None

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
