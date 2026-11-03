# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Whole-request Mooncake WRITE and its worker-owned control protocol.

One background thread owns all ZMQ sockets, rendezvous tables and synchronous
WRITE calls on a rank. Main-thread methods only record readiness and exchange
queue messages. A failed WRITE has no assumed drain guarantee: both endpoints
retain the affected pages until the transport can be proven quiescent.
"""

import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

import msgspec
import torch
import zmq
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorTransferResults
from vllm.logger import logger
from vllm.utils.network_utils import make_zmq_path

from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.base_worker import MooncakeBaseConnectorWorker
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.metadata import MooncakeTransferMetadata
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.push_config import PushConfig
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.push_metadata import (
    MooncakePushMetadata,
    PushReceive,
    PushSource,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.push_planner import plan_push

PROTOCOL_VERSION = 1
POLL_INTERVAL_MS = 10
MAX_INBOX_BATCH = 64
MAX_CONTROL_MESSAGE_BYTES = 64 * 1024 * 1024
CANCEL_RETRY_SECONDS = 1.0
SHUTDOWN_TIMEOUT_SECONDS = 10.0


@dataclass
class ProducerPeer:
    metadata: MooncakeTransferMetadata
    last_sequence: int = 0


@dataclass
class Registration:
    identity: bytes
    sequence: int
    transfer_id: str
    block_ids: tuple[list[int], ...]
    num_tokens: int
    deadline: float
    quarantined: bool = False


@dataclass
class SourceState:
    source: PushSource
    event: Any
    deadline: float
    quarantined: bool = False


@dataclass
class ConsumerPeer:
    socket: Any
    producer_epoch: str
    next_sequence: int = 0


@dataclass
class ReceiveState:
    receive: PushReceive
    peer: ConsumerPeer
    sequence: int
    deadline: float
    cancelling: bool = False
    error: str | None = None
    last_cancel: float = 0.0


@dataclass
class PushCompletion:
    sent: set[str] = field(default_factory=set)
    received: set[str] = field(default_factory=set)
    failed: set[str] = field(default_factory=set)
    invalid_blocks: set[int] = field(default_factory=set)


class MooncakePushConnectorWorker(MooncakeBaseConnectorWorker):
    def __init__(self, config, engine_id: str, cache) -> None:
        super().__init__(config, engine_id, cache)
        self.push_config = PushConfig.from_config(config, cache)
        self._inbox: queue.SimpleQueue = queue.SimpleQueue()
        self._completions: queue.SimpleQueue[PushCompletion] = queue.SimpleQueue()
        self._invalid_blocks: set[int] = set()
        self._pending_results = KVConnectorTransferResults()
        self._ready = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._startup_error: Exception | None = None
        self._epoch = uuid4().hex
        self._device_index = torch.npu.current_device()
        self._producer_peers: dict[bytes, ProducerPeer] = {}
        self._consumer_peers: dict[str, ConsumerPeer] = {}
        self._sources: dict[str, SourceState] = {}
        self._registrations: dict[str, Registration] = {}
        self._receives: dict[str, ReceiveState] = {}
        self._peer_receives: dict[tuple[int, int], str] = {}
        self._router: Any = None
        self._context: Any = None
        self._poller: Any = None
        self._dtype = str(cache.kv_cache_groups[0].kv_cache_spec.dtype)

    def register_kv_caches(self, kv_caches) -> None:
        super().register_kv_caches(kv_caches)
        assert isinstance(self.xfer_handshake_metadata, MooncakeTransferMetadata)
        self._metadata = self.xfer_handshake_metadata
        self._thread = threading.Thread(target=self._run, name="mooncake-push-writer", daemon=True)
        self._thread.start()
        if not self._ready.wait(self.push_config.handshake_timeout):
            raise RuntimeError("Timed out starting Mooncake push writer")
        if self._startup_error is not None:
            raise RuntimeError("Failed starting Mooncake push writer") from self._startup_error

    def start_load_kv(self, metadata: MooncakePushMetadata) -> None:
        for receive in metadata.push_receives:
            self._inbox.put(("receive", receive))
        for source in metadata.push_sources:
            # Eager, synchronous scheduling is required by the initial version.
            # The event follows the completed prefill on this worker's stream.
            event = torch.npu.Event()
            event.record()
            self._inbox.put(("source", (source, event)))
        for request_id in metadata.push_cancellations:
            self._inbox.put(("cancel", request_id))

    def _drain_completions(self) -> None:
        while True:
            try:
                completion = self._completions.get_nowait()
            except queue.Empty:
                break
            self._pending_results.finished_sending.update(completion.sent)
            self._pending_results.finished_recving.update(completion.received)
            self._pending_results.failed_recving.update(completion.failed)
            self._invalid_blocks.update(completion.invalid_blocks)

    def get_transfer_results(self) -> KVConnectorTransferResults:
        self._drain_completions()
        results, self._pending_results = self._pending_results, KVConnectorTransferResults()
        return results

    def get_block_ids_with_load_errors(self) -> set[int]:
        # Runners can ask for invalid pages before asking for completion results.
        self._drain_completions()
        invalid, self._invalid_blocks = self._invalid_blocks, set()
        return invalid

    def shutdown(self) -> None:
        if self._thread is None:
            return
        self._inbox.put(("shutdown", None))
        self._thread.join(SHUTDOWN_TIMEOUT_SECONDS)
        if self._thread.is_alive():
            raise RuntimeError("Mooncake push has undrained transfers; retain KV allocations until process shutdown")

    def _send(self, socket, message: dict, identity: bytes | None = None) -> None:
        data = msgspec.msgpack.encode(message)
        if identity is None:
            socket.send(data)
        else:
            socket.send_multipart([identity, data])

    def _reply(self, registration: Registration, status: str, error: str = "") -> None:
        self._send(
            self._router,
            {
                "kind": status,
                "producer_epoch": self._epoch,
                "sequence": registration.sequence,
                "transfer_id": registration.transfer_id,
                "error": error,
            },
            registration.identity,
        )

    def _complete_source(self, transfer_id: str) -> None:
        state = self._sources.pop(transfer_id, None)
        if state is not None and state.source.delayed_free:
            self._completions.put(PushCompletion(sent={state.source.request_id}))

    def _complete_receive(self, request_id: str, error: str | None = None) -> None:
        state = self._receives.pop(request_id)
        self._peer_receives.pop((id(state.peer.socket), state.sequence), None)
        if state.receive.num_tokens == 0:
            return
        invalid = {block for group in state.receive.block_ids for block in group} if error else set()
        self._completions.put(
            PushCompletion(
                received={request_id},
                failed={request_id} if error else set(),
                invalid_blocks=invalid,
            )
        )

    def _try_push(self, transfer_id: str) -> None:
        source = self._sources.get(transfer_id)
        registration = self._registrations.get(transfer_id)
        if source is None or registration is None or source.quarantined:
            return
        if source.source.cancelled:
            self._reply(registration, "ERROR", "Producer request was cancelled")
            self._registrations.pop(transfer_id)
            self._complete_source(transfer_id)
            return
        try:
            if source.source.num_tokens != registration.num_tokens:
                raise ValueError("Producer and consumer token coverage differ")
            destination = self._producer_peers[registration.identity].metadata
            src, dst, lengths = plan_push(
                self._metadata, destination, source.source.block_ids, registration.block_ids, registration.num_tokens
            )
        except (ValueError, KeyError) as error:
            self._reply(registration, "ERROR", str(error))
            self._registrations.pop(transfer_id)
            self._complete_source(transfer_id)
            return
        try:
            source.event.synchronize()
            if lengths:
                session = f"{destination.local_ip}:{destination.te_rpc_port}"
                result = self.engine.batch_transfer_sync_write(session, src, dst, lengths)
                if result < 0:
                    raise RuntimeError(f"Mooncake WRITE returned {result}")
        except Exception:
            # A negative return/exception is not proof that device DMA has drained.
            # Do not acknowledge cancellation or return either endpoint's pages.
            source.quarantined = registration.quarantined = True
            logger.exception("Mooncake push transfer %s is quarantined; transport drain is unconfirmed", transfer_id)
            self._reply(registration, "QUARANTINED", "Transport drain is unconfirmed")
            return
        self._reply(registration, "COMPLETE")
        self._registrations.pop(transfer_id)
        self._complete_source(transfer_id)

    def _handle_producer_message(self, identity: bytes, message: dict) -> None:
        kind = message.get("kind")
        if kind == "HELLO":
            if (
                message.get("version") != PROTOCOL_VERSION
                or message.get("producer_engine_id") != self.engine_id
                or message.get("tp_size") != self.tp_size
                or message.get("tp_rank") != self.tp_rank
                or message.get("dtype") != self._dtype
            ):
                self._send(self._router, {"kind": "HELLO_ERROR", "error": "Incompatible push peer"}, identity)
                return
            metadata = msgspec.convert(message["metadata"], type=MooncakeTransferMetadata)
            existing = self._producer_peers.get(identity)
            if existing is not None and existing.metadata != metadata:
                self._send(self._router, {"kind": "HELLO_ERROR", "error": "Peer metadata changed"}, identity)
                return
            if existing is None:
                if len(self._producer_peers) >= self.push_config.max_pending:
                    self._send(self._router, {"kind": "HELLO_ERROR", "error": "Push peer limit reached"}, identity)
                    return
                self._producer_peers[identity] = ProducerPeer(metadata)
            self._send(
                self._router,
                {"kind": "HELLO_OK", "version": PROTOCOL_VERSION, "producer_epoch": self._epoch},
                identity,
            )
            return
        peer = self._producer_peers.get(identity)
        if peer is None or message.get("producer_epoch") != self._epoch:
            self._send(self._router, {"kind": "RESTARTED"}, identity)
            return
        sequence = int(message["sequence"])
        transfer_id = message["transfer_id"]
        registration = self._registrations.get(transfer_id)
        same_registration = (
            registration is not None and registration.identity == identity and registration.sequence == sequence
        )
        if kind == "CANCEL":
            peer.last_sequence = max(peer.last_sequence, sequence)
            if same_registration:
                assert registration is not None
                if registration.quarantined:
                    self._reply(registration, "QUARANTINED", "Transport drain is unconfirmed")
                    return
                self._registrations.pop(transfer_id)
                self._complete_source(transfer_id)
            # This thread cannot process CANCEL while its synchronous WRITE is
            # running. Reaching here therefore fences every successful WRITE.
            response = Registration(identity, sequence, transfer_id, tuple(), 0, 0)
            self._reply(response, "CANCELLED")
            return
        if kind != "REGISTER":
            return
        if same_registration:
            return
        response = Registration(identity, sequence, transfer_id, tuple(), 0, 0)
        if sequence <= peer.last_sequence:
            # A single monotonically ordered stream prevents a replay from
            # re-authorizing writes after COMPLETE or CANCELLED, without an
            # unbounded set of per-request tombstones.
            self._reply(response, "RETIRED")
            return
        peer.last_sequence = sequence
        if registration is not None or len(self._registrations) >= self.push_config.max_pending:
            self._reply(response, "ERROR", "Push transfer is duplicated or the pending limit is reached")
            return
        registration = Registration(
            identity=identity,
            sequence=sequence,
            transfer_id=transfer_id,
            block_ids=tuple(message["block_ids"]),
            num_tokens=int(message["num_tokens"]),
            deadline=time.monotonic() + self.push_config.timeout,
        )
        self._registrations[transfer_id] = registration
        self._try_push(transfer_id)

    def _consumer_peer(self, receive: PushReceive) -> ConsumerPeer:
        path = make_zmq_path("tcp", receive.producer_host, receive.producer_port + self.tp_rank)
        if path in self._consumer_peers:
            return self._consumer_peers[path]
        if len(self._consumer_peers) >= self.push_config.max_pending:
            raise ValueError("Push consumer peer limit reached")
        socket = self._context.socket(zmq.DEALER)
        socket.setsockopt(zmq.IDENTITY, f"{self.engine_id}:{self._epoch}".encode())
        socket.setsockopt(zmq.LINGER, 0)
        socket.setsockopt(zmq.SNDTIMEO, int(self.push_config.handshake_timeout * 1000))
        socket.setsockopt(zmq.MAXMSGSIZE, MAX_CONTROL_MESSAGE_BYTES)
        socket.connect(path)
        try:
            self._send(
                socket,
                {
                    "kind": "HELLO",
                    "version": PROTOCOL_VERSION,
                    "producer_engine_id": receive.producer_engine_id,
                    "tp_size": self.tp_size,
                    "tp_rank": self.tp_rank,
                    "dtype": self._dtype,
                    "metadata": msgspec.to_builtins(self._metadata),
                },
            )
            if not socket.poll(int(self.push_config.handshake_timeout * 1000), zmq.POLLIN):
                raise TimeoutError("Push peer handshake timed out before target blocks were published")
            reply = msgspec.msgpack.decode(socket.recv())
            if reply.get("kind") != "HELLO_OK" or reply.get("version") != PROTOCOL_VERSION:
                raise ValueError(reply.get("error", "Invalid push handshake"))
            peer = ConsumerPeer(socket, reply["producer_epoch"])
            self._consumer_peers[path] = peer
            self._poller.register(socket, zmq.POLLIN)
            return peer
        except Exception:
            socket.close()
            raise

    def _register_receive(self, receive: PushReceive) -> None:
        if len(self._receives) >= self.push_config.max_pending:
            if receive.num_tokens:
                self._completions.put(
                    PushCompletion(
                        received={receive.request_id},
                        failed={receive.request_id},
                        invalid_blocks={block for group in receive.block_ids for block in group},
                    )
                )
            return
        try:
            peer = self._consumer_peer(receive)
        except Exception:
            # HELLO contains static pool metadata but no block authorization.
            # No REGISTER was sent, so these target blocks are safe to release.
            logger.exception("Mooncake push handshake failed for %s", receive.request_id)
            if receive.num_tokens:
                self._completions.put(
                    PushCompletion(
                        received={receive.request_id},
                        failed={receive.request_id},
                        invalid_blocks={block for group in receive.block_ids for block in group},
                    )
                )
            return
        peer.next_sequence += 1
        state = ReceiveState(receive, peer, peer.next_sequence, time.monotonic() + self.push_config.timeout)
        self._receives[receive.request_id] = state
        self._peer_receives[(id(peer.socket), state.sequence)] = receive.request_id
        self._send(
            peer.socket,
            {
                "kind": "REGISTER",
                "producer_epoch": peer.producer_epoch,
                "sequence": state.sequence,
                "transfer_id": receive.transfer_id,
                "block_ids": receive.block_ids,
                "num_tokens": receive.num_tokens,
            },
        )

    def _cancel_receive(self, state: ReceiveState) -> None:
        state.cancelling = True
        state.last_cancel = time.monotonic()
        self._send(
            state.peer.socket,
            {
                "kind": "CANCEL",
                "producer_epoch": state.peer.producer_epoch,
                "sequence": state.sequence,
                "transfer_id": state.receive.transfer_id,
            },
        )

    def _handle_consumer_message(self, socket, message: dict) -> None:
        request_id = self._peer_receives.get((id(socket), message.get("sequence")))
        if request_id is None:
            return
        state = self._receives[request_id]
        if (
            message.get("producer_epoch") != state.peer.producer_epoch
            or message.get("transfer_id") != state.receive.transfer_id
        ):
            return
        kind = message.get("kind")
        if kind == "COMPLETE":
            self._complete_receive(request_id, state.error)
        elif kind in ("CANCELLED", "ERROR", "RETIRED"):
            error = state.error
            if kind != "CANCELLED" or not state.cancelling:
                error = error or message.get("error") or "Push transfer was retired"
            self._complete_receive(request_id, error)
        elif kind == "QUARANTINED":
            state.error = "Remote WRITE may still be in flight; KV pages remain quarantined"

    def _expire(self) -> None:
        now = time.monotonic()
        for transfer_id, registration in list(self._registrations.items()):
            if not registration.quarantined and registration.deadline <= now:
                self._reply(registration, "ERROR", "Push registration expired before KV became ready")
                self._registrations.pop(transfer_id)
        for transfer_id, source in list(self._sources.items()):
            if not source.quarantined and source.deadline <= now:
                self._complete_source(transfer_id)
        for state in list(self._receives.values()):
            if state.deadline <= now and not state.cancelling:
                state.error = state.error or "Push receive timed out"
                logger.error(
                    "Mooncake push %s timed out; retaining targets until cancellation is fenced",
                    state.receive.transfer_id,
                )
                self._cancel_receive(state)
            elif state.cancelling and now - state.last_cancel >= CANCEL_RETRY_SECONDS:
                self._cancel_receive(state)

    def _drain_inbox(self) -> None:
        # Keep inbound control/cancellation responsive under sustained arrivals.
        for _ in range(MAX_INBOX_BATCH):
            try:
                kind, value = self._inbox.get_nowait()
            except queue.Empty:
                return
            if kind == "receive":
                self._register_receive(value)
            elif kind == "source":
                source, event = value
                if source.transfer_id in self._sources:
                    raise RuntimeError("Duplicate producer transfer ID")
                if len(self._sources) >= self.push_config.max_pending and source.transfer_id not in self._registrations:
                    # No WRITE has been authorized or submitted for this source.
                    if source.delayed_free:
                        self._completions.put(PushCompletion(sent={source.request_id}))
                    continue
                self._sources[source.transfer_id] = SourceState(
                    source, event, time.monotonic() + self.push_config.timeout
                )
                self._try_push(source.transfer_id)
            elif kind == "cancel":
                state = self._receives.get(value)
                if state is not None:
                    self._cancel_receive(state)
            elif kind == "shutdown":
                self._stop.set()
                for state in list(self._receives.values()):
                    self._cancel_receive(state)

    def _run(self) -> None:
        try:
            torch.npu.set_device(self._device_index)
            self._context = zmq.Context()
            self._poller = zmq.Poller()
            if self.kv_transfer_config.is_kv_producer:
                self._router = self._context.socket(zmq.ROUTER)
                self._router.setsockopt(zmq.LINGER, 0)
                self._router.setsockopt(zmq.SNDTIMEO, int(self.push_config.handshake_timeout * 1000))
                self._router.setsockopt(zmq.MAXMSGSIZE, MAX_CONTROL_MESSAGE_BYTES)
                self._router.bind(make_zmq_path("tcp", self.side_channel_host, self.handshake_port))
                self._poller.register(self._router, zmq.POLLIN)
            self._ready.set()
            while (
                not self._stop.is_set()
                or self._receives
                or any(source.quarantined for source in self._sources.values())
            ):
                try:
                    self._drain_inbox()
                    for socket in dict(self._poller.poll(POLL_INTERVAL_MS)):
                        if socket is self._router:
                            identity, payload = socket.recv_multipart()
                            self._handle_producer_message(identity, msgspec.msgpack.decode(payload))
                        else:
                            self._handle_consumer_message(socket, msgspec.msgpack.decode(socket.recv()))
                    self._expire()
                except Exception:
                    # Never release pages merely because control handling failed.
                    logger.exception("Mooncake push control operation failed; retaining pending transfers")
        except Exception as error:
            self._startup_error = error
            self._ready.set()
            logger.exception("Mooncake push writer terminated")
        finally:
            if self._router is not None:
                self._router.close()
            for peer in self._consumer_peers.values():
                peer.socket.close()
            if self._context is not None:
                self._context.term()
