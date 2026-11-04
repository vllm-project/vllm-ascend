# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""D-side Push control: one Ready sender and one completion listener."""

import queue
import threading
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import httpx
import msgspec
import zmq
from vllm import envs
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorTransferResults
from vllm.logger import logger
from vllm.utils.network_utils import make_zmq_path, make_zmq_socket

from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.base_worker import MooncakeBaseConnectorWorker
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.metadata import MooncakeConnectorMetadata, PushReqMeta
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.utils import (
    SizedDict,
    ensure_zmq_recv,
    ensure_zmq_send,
    zmq_ctx,
)

if TYPE_CHECKING:
    import torch


# Rank tuples are (PP, PCP, TP), matching scheduler handshake keys.
WorkerRank = tuple[int, int, int]


@dataclass
class _ReceiveState:
    request_id: str
    metadata: PushReqMeta
    start_time: float
    expected_ranks: set[WorkerRank] = field(default_factory=set)
    finished_ranks: set[WorkerRank] = field(default_factory=set)
    failed: bool = False


class MooncakeReadySendingThread(threading.Thread):
    """Own bootstrap HTTP calls and Ready sockets; never touch receive state."""

    def __init__(self, worker: "MooncakePushConnectorWorker") -> None:
        super().__init__(daemon=True, name="MooncakeReadySendingThread")
        self.stop_event = worker._stop_event
        self.ready_event = threading.Event()
        self.error: Exception | None = None
        self.engine_id = worker.engine_id
        self.rank = (worker.pp_rank, worker.pcp_rank, worker.tp_rank)
        self.completion_address = worker._recving_thread.address
        self.request_queue = worker._ready_queue
        self.result_queue = worker._result_queue
        self.remote_worker_addresses: SizedDict = SizedDict(max_size=128)
        self.timeout = float(envs.VLLM_MOONCAKE_CONNECTOR_TIMEOUT)
        self.zmq_context: zmq.Context | None = None

    def _get_remote_worker_addresses(self, client: httpx.Client, metadata: PushReqMeta) -> dict[WorkerRank, str]:
        key = (metadata.remote_bootstrap_addr, metadata.remote_engine_id)
        cached = self.remote_worker_addresses.get(key)
        if cached is not None:
            return cached
        response = client.get(metadata.remote_bootstrap_addr.rstrip("/") + "/query")
        response.raise_for_status()
        entries = [entry for entry in response.json().values() if entry["engine_id"] == metadata.remote_engine_id]
        if len(entries) != 1:
            raise ValueError("P engine is missing or ambiguous in bootstrap")
        entry = entries[0]
        sizes = (entry["pp_size"], entry["pcp_size"], entry["tp_size"])
        if any(type(size) is not int or size <= 0 for size in sizes):
            raise ValueError("Invalid P topology in bootstrap")
        addresses = {
            (int(pp), int(pcp), int(tp)): addr
            for pcp, tp_map in entry["worker_addr"].items()
            for tp, pp_map in tp_map.items()
            for pp, addr in pp_map.items()
        }
        expected_ranks = {(pp, pcp, tp) for pp in range(sizes[0]) for pcp in range(sizes[1]) for tp in range(sizes[2])}
        if set(addresses) != expected_ranks or any(
            not isinstance(addr, str) or not addr.startswith("tcp://") for addr in addresses.values()
        ):
            raise ValueError("P bootstrap registration is incomplete or invalid")
        self.remote_worker_addresses[key] = addresses
        return addresses

    def _send_ready(self, address: str, message: dict[str, Any]) -> None:
        # REQ sockets belong exclusively to this sender thread. DONE uses a
        # separate callback connection, not this Ready/ACK exchange.
        assert self.zmq_context is not None
        sock = make_zmq_socket(ctx=self.zmq_context, path=address, socket_type=zmq.REQ, bind=False)
        try:
            sock.setsockopt(zmq.SNDTIMEO, max(1, int(self.timeout * 1000)))
            sock.setsockopt(zmq.RCVTIMEO, 100)
            # Keep this exchange's existing timeout budget. The helpers must
            # not add retry windows around the interruptible ACK polling loop.
            ensure_zmq_send(sock, msgspec.msgpack.encode(message), address, max_retries=1)
            deadline = time.monotonic() + self.timeout
            while not self.stop_event.is_set():
                if sock.poll(100):
                    reply = msgspec.msgpack.decode(ensure_zmq_recv(sock, address, max_retries=1))
                    if not isinstance(reply, dict) or reply.get("kind") != "ready_ack":
                        raise ValueError("Invalid Ready ACK")
                    for key in (
                        "transfer_id",
                        "producer_engine_id",
                        "consumer_engine_id",
                        "producer_rank",
                        "consumer_rank",
                    ):
                        if reply.get(key) != message[key]:
                            raise ValueError(f"Ready ACK mismatch: {key}")
                    return
                if time.monotonic() >= deadline:
                    raise TimeoutError("Timed out waiting for Ready ACK")
            raise RuntimeError("Push sender stopped")
        finally:
            # Discard a timed-out REQ socket without terminating the context.
            sock.close(linger=0)

    def start(self) -> None:
        super().start()
        if not self.ready_event.wait(timeout=10):
            self.shutdown()
            raise TimeoutError(f"Timed out starting {self.name}")
        if self.error is not None:
            self.shutdown()
            raise RuntimeError(f"{self.name} failed to start") from self.error

    def shutdown(self) -> None:
        self.stop_event.set()
        self.request_queue.put(None)
        if self.ident is not None:
            self.join()

    def run(self) -> None:
        try:
            # Created, used and destroyed exclusively by this sender thread.
            self.zmq_context = zmq.Context()
            with httpx.Client(timeout=self.timeout) as client:
                self.ready_event.set()
                while not self.stop_event.is_set():
                    try:
                        item = self.request_queue.get(timeout=0.1)
                    except queue.Empty:
                        continue
                    if item is None or self.stop_event.is_set():
                        return
                    self._handle_request(client, item)
        except Exception as exc:
            self.error = exc
            logger.exception("%s failed", self.name)
        finally:
            if self.zmq_context is not None:
                self.zmq_context.destroy(linger=0)
                self.zmq_context = None
            self.ready_event.set()

    def _handle_request(self, client: httpx.Client, item: tuple[str, PushReqMeta]) -> None:
        request_id, metadata = item
        issued = False
        try:
            remote_worker_addresses = self._get_remote_worker_addresses(client, metadata)
            # Publish expected sources before any P can send DONE.
            self.result_queue.put(("peers", metadata.transfer_id, set(remote_worker_addresses)))
            # Every P worker receives control only. Each must send
            # terminal DONE, including workers with no owned slices.
            # Actual KV ownership/routing belongs to the P planner.
            for rank, address in remote_worker_addresses.items():
                if self.stop_event.is_set():
                    return
                message = {
                    "kind": "ready",
                    "transfer_id": metadata.transfer_id,
                    "request_id": request_id,
                    "producer_engine_id": metadata.remote_engine_id,
                    "producer_rank": list(rank),
                    "consumer_engine_id": self.engine_id,
                    "consumer_rank": list(self.rank),
                    "completion_address": self.completion_address,
                    "scheduler_address": make_zmq_path("tcp", metadata.scheduler_host, metadata.scheduler_port),
                    "block_ids": metadata.local_block_ids,
                    "num_computed_tokens": metadata.num_computed_tokens,
                    "num_external_tokens": metadata.num_external_tokens,
                }
                # From this point a lost ACK cannot prove P won't WRITE.
                issued = True
                self._send_ready(address, message)
        except Exception as exc:
            if not self.stop_event.is_set():
                self.result_queue.put(("uncertain" if issued else "failed", metadata.transfer_id, exc))


class MooncakeCompletionThread(threading.Thread):
    """Receive terminal per-source DONE; main thread owns request accounting."""

    def __init__(self, worker: "MooncakePushConnectorWorker") -> None:
        super().__init__(daemon=True, name="MooncakeCompletionThread")
        self.stop_event = worker._stop_event
        self.ready_event = threading.Event()
        self.error: Exception | None = None
        self.engine_id = worker.engine_id
        self.rank = [worker.pp_rank, worker.pcp_rank, worker.tp_rank]
        self.address = make_zmq_path("tcp", worker.side_channel_host, worker.handshake_port)
        self.result_queue = worker._result_queue

    def start(self) -> None:
        super().start()
        if not self.ready_event.wait(timeout=10):
            self.shutdown()
            raise TimeoutError(f"Timed out starting {self.name}")
        if self.error is not None:
            self.shutdown()
            raise RuntimeError(f"{self.name} failed to start") from self.error

    def shutdown(self) -> None:
        self.stop_event.set()
        if self.ident is not None:
            self.join()

    def run(self) -> None:
        try:
            with zmq_ctx(zmq.ROUTER, self.address) as sock:
                sock.setsockopt(zmq.RCVTIMEO, 100)
                sock.setsockopt(zmq.SNDTIMEO, 1000)
                self.address = sock.getsockopt(zmq.LAST_ENDPOINT).decode()
                self.ready_event.set()
                while not self.stop_event.is_set():
                    try:
                        frames = sock.recv_multipart()
                    except zmq.Again:
                        continue
                    if self.stop_event.is_set():
                        return
                    self._handle_frames(sock, frames)
        except Exception as exc:
            self.error = exc
            logger.exception("%s failed", self.name)
        finally:
            self.ready_event.set()

    def _handle_frames(self, sock: Any, frames: list[bytes]) -> None:
        identity, *payload = frames
        # Support REQ (empty delimiter) and DEALER callbacks.
        delimiter = [b""] if payload and payload[0] == b"" else []
        if delimiter:
            payload = payload[1:]
        try:
            if len(payload) != 1:
                raise ValueError("Expected one completion payload")
            message = msgspec.msgpack.decode(payload[0])
            if not isinstance(message, dict) or message.get("kind") != "done":
                raise ValueError("Expected DONE")
            if message.get("consumer_engine_id") != self.engine_id or message.get("consumer_rank") != self.rank:
                raise ValueError("DONE belongs to another consumer")
            rank = message.get("producer_rank")
            if not isinstance(rank, list) or len(rank) != 3 or any(type(v) is not int or v < 0 for v in rank):
                raise ValueError("Invalid producer rank")
            for key in ("transfer_id", "request_id", "producer_engine_id"):
                if not isinstance(message.get(key), str) or not message[key]:
                    raise ValueError(f"Missing completion identity: {key}")
            if message.get("status") not in ("ok", "error") or message.get("writes_drained") is not True:
                raise ValueError("DONE must certify all source WRITEs are drained")
            self.result_queue.put(("done", message["transfer_id"], message))
            reply = {"kind": "done_ack", "transfer_id": message["transfer_id"]}
        except (ValueError, msgspec.DecodeError) as exc:
            reply = {"kind": "error", "error": str(exc)}
        try:
            sock.send_multipart([identity, *delimiter, msgspec.msgpack.encode(reply)])
        except zmq.Again:
            # P may retry DONE; main-thread source accounting is idempotent.
            logger.warning("Timed out acknowledging Push completion")


class MooncakePushConnectorWorker(MooncakeBaseConnectorWorker):
    """Bridge the model thread to independent D Ready and DONE threads."""

    def __init__(self, vllm_config, engine_id, kv_cache_config) -> None:
        super().__init__(vllm_config, engine_id, kv_cache_config)
        self._stop_event = threading.Event()
        self._ready_queue: queue.Queue[tuple[str, PushReqMeta] | None] = queue.Queue()
        self._result_queue: queue.SimpleQueue = queue.SimpleQueue()
        self._reqs_recv_state: dict[str, _ReceiveState] = {}
        self._finished_transfers: SizedDict = SizedDict(max_size=16000)
        self._recving_thread: MooncakeCompletionThread | None = None
        self._sending_thread: MooncakeReadySendingThread | None = None
        self._control_error: Exception | None = None

    def register_kv_caches(self, kv_caches: dict[str, "torch.Tensor | list[torch.Tensor]"]) -> None:
        if self._recving_thread is not None:
            raise RuntimeError("Replacing active D KV storage is not supported")
        super().register_kv_caches(kv_caches)
        if self.kv_transfer_config.is_kv_producer:
            return
        self._recving_thread = MooncakeCompletionThread(self)
        try:
            self._recving_thread.start()
            self._sending_thread = MooncakeReadySendingThread(self)
            self._sending_thread.start()
        except Exception:
            self.shutdown()
            raise

    def start_load_kv(self, metadata: MooncakeConnectorMetadata) -> None:
        self.maybe_raise_thread_error()
        if self.kv_transfer_config.is_kv_producer:
            pass
        else:
            if self._sending_thread is None:
                raise RuntimeError("Register D KV caches before submitting Ready")
            for request_id, request in metadata.push_requests.items():
                if request.transfer_id in self._finished_transfers:
                    if self._finished_transfers[request.transfer_id] != (request_id, request):
                        raise ValueError("Reused retired Push transfer ID")
                    continue
                existing = self._reqs_recv_state.get(request.transfer_id)
                if existing is not None:
                    if (existing.request_id, existing.metadata) != (request_id, request):
                        raise ValueError("Conflicting Ready for an active transfer")
                    continue
                if (
                    not request.transfer_id
                    or not request.remote_engine_id
                    or request.num_computed_tokens < 0
                    or request.num_external_tokens <= 0
                ):
                    raise ValueError("Invalid Push request identity or token range")
                if any(block < 0 or block >= self.num_blocks for group in request.local_block_ids for block in group):
                    raise ValueError("Invalid D destination block ID")
                self._reqs_recv_state[request.transfer_id] = _ReceiveState(request_id, request, time.monotonic())
                self._ready_queue.put((request_id, request))

    def maybe_raise_thread_error(self) -> None:
        for thread in (self._sending_thread, self._recving_thread):
            if thread is not None and thread.error is not None:
                self._control_error = thread.error
        if self._control_error is not None:
            raise RuntimeError(
                "Push control failed with possible outstanding WRITEs; do not recycle destination blocks"
            ) from self._control_error

    def get_transfer_results(self) -> KVConnectorTransferResults:
        self.maybe_raise_thread_error()
        results = KVConnectorTransferResults()
        if self.kv_transfer_config.is_kv_producer:
            pass
        else:
            while True:
                try:
                    kind, transfer_id, payload = self._result_queue.get_nowait()
                except queue.Empty:
                    break
                state = self._reqs_recv_state.get(transfer_id)
                if state is None:
                    continue  # Duplicate completion for an already retired transfer.
                if kind == "peers":
                    state.expected_ranks = payload
                elif kind == "done":
                    if (
                        payload["producer_engine_id"] != state.metadata.remote_engine_id
                        or payload["request_id"] != state.request_id
                    ):
                        continue
                    source = tuple(payload["producer_rank"])
                    if source not in state.expected_ranks or source in state.finished_ranks:
                        continue
                    state.finished_ranks.add(source)
                    state.failed |= payload["status"] == "error"
                elif kind == "failed":
                    # Discovery failed before publishing any destination to P.
                    state.failed = True
                    logger.error("Push discovery failed for %s: %s", state.request_id, payload)
                elif kind == "uncertain":
                    self._control_error = payload
                    self.maybe_raise_thread_error()
                if kind == "failed" or (state.expected_ranks and state.finished_ranks == state.expected_ranks):
                    results.finished_recving.add(state.request_id)
                    if state.failed:
                        results.failed_recving.add(state.request_id)
                    self._finished_transfers[transfer_id] = (state.request_id, state.metadata)
                    del self._reqs_recv_state[transfer_id]
            for state in self._reqs_recv_state.values():
                if time.monotonic() - state.start_time > envs.VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT:
                    self._control_error = TimeoutError(f"Missing terminal Push completion for {state.request_id}")
                    self.maybe_raise_thread_error()
        return results

    def get_block_ids_with_load_errors(self) -> set[int]:
        # Failed requests are reported through failed_recving, after writes drain.
        self.maybe_raise_thread_error()
        return set()

    def shutdown(self) -> None:
        self._stop_event.set()
        for thread in (self._sending_thread, self._recving_thread):
            if thread is not None:
                thread.shutdown()
