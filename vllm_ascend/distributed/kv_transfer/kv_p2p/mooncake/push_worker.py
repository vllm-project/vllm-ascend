# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Layerwise Push: P Ready listener and WRITE pool, D Ready/DONE threads."""

import queue
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

import httpx
import msgspec
import torch
import zmq
from vllm import envs
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorTransferResults
from vllm.logger import logger
from vllm.utils.network_utils import make_zmq_path, make_zmq_socket
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    MambaSpec,
    MLAAttentionSpec,
    SlidingWindowMLASpec,
    SlidingWindowSpec,
)

from vllm_ascend.core.kv_cache_interface import is_circular_kv_cache_spec
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.base_worker import MooncakeBaseConnectorWorker
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.metadata import (
    MooncakeConnectorMetadata,
    MooncakeTransferMetadataGroups,
    PushReqMeta,
    PushSendReqMeta,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.utils import (
    SizedDict,
    ensure_zmq_recv,
    ensure_zmq_send,
    zmq_ctx,
)

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
    ready_sent: bool = False


class MooncakeReadySendingThread(threading.Thread):
    """Own bootstrap HTTP calls and Ready sockets; never touch receive state."""

    def __init__(self, worker: "MooncakePushConnectorWorker") -> None:
        super().__init__(daemon=True, name="MooncakeReadySendingThread")
        self.stop_event = worker._stop_event
        self.ready_event = threading.Event()
        self.error: Exception | None = None
        self.engine_id = worker.engine_id
        self.rank = (worker.pp_rank, worker.pcp_rank, worker.tp_rank)
        speculative_config = worker.vllm_config.speculative_config
        self.num_speculative_tokens = speculative_config.num_speculative_tokens if speculative_config else 0
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
                    "num_speculative_tokens": self.num_speculative_tokens,
                }
                # From this point a lost ACK cannot prove P won't WRITE.
                issued = True
                self._send_ready(address, message)
            # DONE may arrive before the last Ready ACK. Do not report success
            # until both the sender outcome and all source outcomes are known.
            self.result_queue.put(("ready_sent", metadata.transfer_id, None))
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
        completion = None
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
            if message.get("status") not in ("ok", "error") or type(message.get("writes_drained")) is not bool:
                raise ValueError("DONE requires a status and WRITE drain result")
            if message["status"] == "ok" and not message["writes_drained"]:
                raise ValueError("Successful DONE must certify drained WRITEs")
            completion = message
            reply = {"kind": "done_ack", "transfer_id": message["transfer_id"]}
        except (ValueError, msgspec.DecodeError) as exc:
            reply = {"kind": "error", "error": str(exc)}
        try:
            sock.send_multipart([identity, *delimiter, msgspec.msgpack.encode(reply)])
        except zmq.ZMQError:
            logger.exception("Failed to acknowledge Push completion")
            if completion is not None:
                # Source WRITEs are already drained, so this request can fail
                # safely after all other sources have also finished.
                completion["status"] = "error"
        if completion is not None:
            self.result_queue.put(("done", completion["transfer_id"], completion))


@dataclass(frozen=True)
class LayerTransferRequest:
    transfer_id: str
    local_block_ids: tuple[int, ...]
    remote_block_ids: tuple[int, ...]
    start_token: int
    end_token: int


@dataclass(frozen=True)
class LayerTransferBatch:
    """One layer/destination WRITE batch; snapshots outlive the model step."""

    layer_name: str
    remote_rank: WorkerRank
    remote_metadata: MooncakeTransferMetadataGroups
    requests: tuple[LayerTransferRequest, ...]
    event: Any


@dataclass
class RemoteReqInfo:
    ready: dict[str, Any]
    metadata_future: Future
    required_layers: set[str] | None = None
    sent_tokens: dict[str, int] = field(default_factory=dict)
    pending_layers: set[str] = field(default_factory=set)
    completion: Future | None = None


@dataclass
class SendReqInfo:
    request_id: str | None = None
    metadata: PushSendReqMeta | None = None
    start_time: float = field(default_factory=time.monotonic)
    # Available end token and device fence for each persistent cache layer.
    available_layers: dict[str, tuple[int, Any]] = field(default_factory=dict)
    destinations: dict[WorkerRank, RemoteReqInfo] = field(default_factory=dict)
    expected_ranks: set[WorkerRank] | None = None
    consumer_engine_id: str | None = None
    inference_finished: bool = False
    error: str | None = None


class MooncakeReadyRecvingThread(threading.Thread):
    """P listener only acknowledges/enqueues Ready; the model thread owns state."""

    def __init__(self, worker: "MooncakePushConnectorWorker") -> None:
        super().__init__(daemon=True, name="MooncakeReadyRecvingThread")
        self.stop_event = worker._stop_event
        self.ready_event = threading.Event()
        self.error: Exception | None = None
        self.engine_id = worker.engine_id
        self.rank = [worker.pp_rank, worker.pcp_rank, worker.tp_rank]
        self.address = make_zmq_path("tcp", worker.side_channel_host, worker.handshake_port)
        self.request_queue = worker._ready_queue

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
                    identity, *payload = frames
                    delimiter = [b""] if payload and payload[0] == b"" else []
                    if delimiter:
                        payload = payload[1:]
                    try:
                        if len(payload) != 1:
                            raise ValueError("Expected one Ready payload")
                        ready = msgspec.msgpack.decode(payload[0])
                        if ready["kind"] != "ready":
                            raise ValueError("Expected Ready")
                        if ready["producer_engine_id"] != self.engine_id or ready["producer_rank"] != self.rank:
                            raise ValueError("Ready belongs to another producer")
                        for key in (
                            "transfer_id",
                            "request_id",
                            "consumer_engine_id",
                            "scheduler_address",
                            "completion_address",
                        ):
                            if not ready[key]:
                                raise ValueError(f"Missing Ready field: {key}")
                        if (
                            len(ready["consumer_rank"]) != 3
                            or ready["num_computed_tokens"] < 0
                            or ready["num_external_tokens"] <= 0
                        ):
                            raise ValueError("Invalid Ready rank or token range")
                        reply = {
                            key: ready[key]
                            for key in (
                                "transfer_id",
                                "producer_engine_id",
                                "consumer_engine_id",
                                "producer_rank",
                                "consumer_rank",
                            )
                        }
                        reply["kind"] = "ready_ack"
                    except (KeyError, TypeError, ValueError, msgspec.DecodeError) as exc:
                        sock.send_multipart(
                            [identity, *delimiter, msgspec.msgpack.encode({"kind": "error", "error": str(exc)})]
                        )
                        continue
                    # Publish only after ACK was queued successfully. A send
                    # failure makes the listener fatal; it must not start WRITE.
                    sock.send_multipart([identity, *delimiter, msgspec.msgpack.encode(reply)])
                    self.request_queue.put(ready)
        except Exception as exc:
            self.error = exc
            logger.exception("%s failed", self.name)
        finally:
            self.ready_event.set()


class MooncakePushConnectorWorker(MooncakeBaseConnectorWorker):
    """Coordinate P layer batches and D Ready/DONE control threads."""

    def __init__(self, vllm_config, engine_id, kv_cache_config) -> None:
        if vllm_config.parallel_config.decode_context_parallel_size != 1:
            raise ValueError("Mooncake Push worker currently requires DCP=1")
        super().__init__(vllm_config, engine_id, kv_cache_config)
        self._stop_event = threading.Event()
        self._ready_queue: queue.Queue = queue.Queue()
        self._result_queue: queue.SimpleQueue = queue.SimpleQueue()
        self._reqs_recv_state: dict[str, _ReceiveState] = {}
        self._finished_transfers: SizedDict = SizedDict(max_size=16000)
        self._recving_thread: MooncakeCompletionThread | MooncakeReadyRecvingThread | None = None
        self._sending_thread: MooncakeReadySendingThread | None = None
        self._control_error: Exception | None = None
        self._reqs_send_state: dict[str, SendReqInfo] = {}
        self._prefilling_transfer_ids: tuple[str, ...] = ()
        self._finished_sending: set[str] = set()
        self._pending_sends: list[tuple[LayerTransferBatch, Future]] = []
        self._callback_futures: list[Future] = []
        self._remote_metadata: SizedDict = SizedDict(max_size=128)
        self._sender_executor: ThreadPoolExecutor | None = None
        self._zmq_context: zmq.Context | None = None

    def register_kv_caches(self, kv_caches: dict[str, "torch.Tensor | list[torch.Tensor]"]) -> None:
        if self._recving_thread is not None:
            raise RuntimeError("Replacing active Push KV storage is not supported")
        super().register_kv_caches(kv_caches)
        if self.kv_transfer_config.is_kv_producer:
            if self.ascend_config.kvpp_config.size > 1:
                raise ValueError("Mooncake Push does not support KVPP yet")
            self._device = torch.npu.current_device()
            self._layer_indices = {name: i for i, name in enumerate(self.transfer_metadata.layer_names)}
            # Shared Mamba address calculation uses the same registered descriptors as Pull.
            for name in ("layer_names", "kv_caches_base_addr", "block_shapes", "block_strides", "block_lens"):
                setattr(self, name, getattr(self.transfer_metadata, name))
            self._zmq_context = zmq.Context()
            self._sender_executor = ThreadPoolExecutor(
                max_workers=32,
                initializer=torch.npu.set_device,
                initargs=(self._device,),
                thread_name_prefix="MooncakePushSending",
            )
            self._recving_thread = MooncakeReadyRecvingThread(self)
            try:
                self._recving_thread.start()
            except Exception:
                self.shutdown()
                raise
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
            # Also drain late-Ready sends submitted by get_transfer_results.
            while self._pending_sends:
                self._process_sending_requests(wait=True)
            self._process_ready_requests()
            for request_id, request in metadata.push_send_requests.items():
                if request.transfer_id in self._finished_transfers:
                    raise ValueError("Reused retired Push transfer ID")
                state = self._reqs_send_state.setdefault(request.transfer_id, SendReqInfo())
                state.request_id = request_id
                state.metadata = request
                # A prefix hit is already resident before this forward.
                if not state.available_layers and request.num_computed_tokens:
                    event = torch.npu.Event()
                    event.record()
                    state.available_layers = {
                        name: (request.num_computed_tokens, event) for name in self._layer_indices
                    }
            for state in self._reqs_send_state.values():
                if state.request_id in metadata.requests_to_send:
                    state.inference_finished = True
            for request_id, transfer_id in metadata.push_aborted_requests.items():
                state = self._reqs_send_state.setdefault(transfer_id, SendReqInfo())
                state.request_id = request_id
                state.inference_finished = True
                state.error = "P request aborted"
            self._prefilling_transfer_ids = tuple(req.transfer_id for req in metadata.push_send_requests.values())
            self._process_sending_requests()
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

    def _get_remote_metadata(self, ready: dict[str, Any]) -> MooncakeTransferMetadataGroups:
        path = ready["scheduler_address"]
        sock = make_zmq_socket(self._zmq_context, path, zmq.REQ, bind=False)
        try:
            timeout = max(1, int(envs.VLLM_MOONCAKE_CONNECTOR_TIMEOUT * 1000))
            sock.setsockopt(zmq.SNDTIMEO, timeout)
            sock.setsockopt(zmq.RCVTIMEO, timeout)
            ensure_zmq_send(sock, msgspec.msgpack.encode((b"get_meta_msg", ready["consumer_engine_id"])), path)
            metadata = msgspec.msgpack.decode(ensure_zmq_recv(sock, path), type=MooncakeTransferMetadataGroups)
        finally:
            sock.close(linger=0)
        if metadata.engine_id != ready["consumer_engine_id"] or metadata.dcp_size != 1:
            raise ValueError("Push requires matching D engine metadata with DCP=1")
        if metadata.use_kv_pp:
            raise ValueError("Push does not support KVPP yet")
        return metadata

    def _send_done(self, ready: dict[str, Any], error: str | None, writes_drained: bool = True) -> bool:
        """A failed callback never retries the request or retains P source blocks."""
        path = ready["completion_address"]
        sock = None
        try:
            sock = make_zmq_socket(self._zmq_context, path, zmq.REQ, bind=False)
            timeout = max(1, int(envs.VLLM_MOONCAKE_CONNECTOR_TIMEOUT * 1000))
            sock.setsockopt(zmq.SNDTIMEO, timeout)
            sock.setsockopt(zmq.RCVTIMEO, timeout)
            message = {
                key: ready[key]
                for key in (
                    "transfer_id",
                    "request_id",
                    "producer_engine_id",
                    "producer_rank",
                    "consumer_engine_id",
                    "consumer_rank",
                )
            }
            message.update(kind="done", status="error" if error else "ok", writes_drained=writes_drained)
            if error:
                message["error"] = error
            ensure_zmq_send(sock, msgspec.msgpack.encode(message), path)
            reply = msgspec.msgpack.decode(ensure_zmq_recv(sock, path))
            if reply != {"kind": "done_ack", "transfer_id": ready["transfer_id"]}:
                raise ValueError("Invalid DONE ACK")
            return True
        except Exception:
            # D may have died. The caller may release blocks only when WRITE
            # drain is known; an undrained failure stops P independently.
            logger.exception("Failed to notify Push completion for %s at %s", ready["transfer_id"], path)
            return False
        finally:
            if sock is not None:
                sock.close(linger=0)

    def _process_ready_requests(self) -> None:
        while True:
            try:
                ready = self._ready_queue.get_nowait()
            except queue.Empty:
                break
            transfer_id = ready["transfer_id"]
            if transfer_id in self._finished_transfers:
                self._callback_futures.append(
                    self._sender_executor.submit(self._send_done, ready, "Push transfer has already retired")
                )
                continue
            state = self._reqs_send_state.setdefault(transfer_id, SendReqInfo())
            if state.consumer_engine_id is not None and state.consumer_engine_id != ready["consumer_engine_id"]:
                self._callback_futures.append(
                    self._sender_executor.submit(self._send_done, ready, "Transfer belongs to another D engine")
                )
                continue
            state.consumer_engine_id = ready["consumer_engine_id"]
            rank = tuple(ready["consumer_rank"])
            if rank in state.destinations:
                if state.destinations[rank].ready != ready:
                    state.error = "Conflicting Ready for the same destination"
                continue
            key = (ready["consumer_engine_id"], ready["scheduler_address"])
            future = self._remote_metadata.get(key)
            if future is None:
                future = self._sender_executor.submit(self._get_remote_metadata, ready)
                self._remote_metadata[key] = future
            state.destinations[rank] = RemoteReqInfo(ready, future)

    def _get_transfer_partition(
        self, layer_name: str, remote: MooncakeTransferMetadataGroups, rank: WorkerRank
    ) -> tuple[int, int, int] | None:
        """Return source/destination head offsets and count, or None for a non-owner."""
        local_index = self._layer_indices[layer_name]
        pp = remote.metadata_by_pp_rank[rank[0]]
        if layer_name not in pp.layer_names:
            return None
        remote_index = pp.layer_names.index(layer_name)
        worker = pp.metadata_by_pcp_rank[rank[1]].metadata_by_tp_rank[rank[2]]
        if remote_index not in worker.layer_indices:
            return None
        spec = self.kv_cache_specs[self.layer_name_to_spec_index[layer_name]]
        local_rank = self.tp_rank
        remote_rank = rank[2]
        local_size, remote_size = self.tp_size, remote.tp_size
        if isinstance(spec, MambaSpec):
            if getattr(spec, "tp_replicated", False):
                owner = (rank[2] * remote.pcp_size + rank[1]) % (self.tp_size * self.pcp_size)
                return (0, 0, 1) if self.tp_rank * self.pcp_size + self.pcp_rank == owner else None
            local_rank = self.tp_rank * self.pcp_size + self.pcp_rank
            remote_rank = rank[2] * remote.pcp_size + rank[1]
            local_size *= self.pcp_size
            remote_size *= remote.pcp_size
            larger, smaller = max(local_size, remote_size), min(local_size, remote_size)
            if larger % smaller:
                raise ValueError("Mamba P/D PCP*TP sizes must have an integer ratio")
            ratio = larger // smaller
            matched = (
                local_rank // ratio == remote_rank if local_size >= remote_size else remote_rank // ratio == local_rank
            )
            return (0, 0, 1) if matched else None
        whole = isinstance(spec, (MLAAttentionSpec, SlidingWindowMLASpec)) or is_circular_kv_cache_spec(spec)
        if whole:
            # Replicated pages: exactly one P worker writes each D replica.
            owner = (rank[2] * remote.pcp_size + rank[1]) % (self.tp_size * self.pcp_size)
            return (0, 0, 1) if self.tp_rank * self.pcp_size + self.pcp_rank == owner else None
        if not isinstance(spec, (FullAttentionSpec, SlidingWindowSpec)):
            raise ValueError(f"Unsupported Push cache spec: {type(spec).__name__}")
        if self.pcp_rank != rank[1] % self.pcp_size:
            return None
        local_heads = self.transfer_metadata.block_shapes[local_index][0][0]
        remote_heads = pp.block_shapes[remote_index][0][0]
        totals = {heads * size for heads, size in ((local_heads, local_size), (remote_heads, remote_size)) if heads > 1}
        if len(totals) > 1:
            raise ValueError("Inconsistent P/D attention head counts")
        total_heads = next(iter(totals), min(local_size, remote_size))
        if max(total_heads, local_size) % min(total_heads, local_size) or max(total_heads, remote_size) % min(
            total_heads, remote_size
        ):
            raise ValueError("Unsupported attention head partition")
        # Where TP exceeds KV heads, choose only one P replica for each head.
        replicas = max(1, local_size // total_heads)
        if local_rank % replicas != remote_rank % replicas:
            return None
        local_start = local_rank // replicas * local_heads
        remote_start = remote_rank // max(1, remote_size // total_heads) * remote_heads
        start, end = max(local_start, remote_start), min(local_start + local_heads, remote_start + remote_heads)
        return (start - local_start, start - remote_start, end - start) if start < end else None

    def _build_layer_transfer_request(
        self, transfer_id: str, state: SendReqInfo, destination: RemoteReqInfo, layer_name: str
    ) -> tuple[LayerTransferRequest, Any] | None:
        meta = state.metadata
        ready = destination.ready
        remote = destination.metadata_future.result()
        pp = remote.metadata_by_pp_rank[ready["consumer_rank"][0]]
        local_index = self._layer_indices[layer_name]
        remote_index = pp.layer_names.index(layer_name)
        spec = self.kv_cache_specs[self.layer_name_to_spec_index[layer_name]]
        available, event = state.available_layers[layer_name]
        final = available >= meta.num_prompt_tokens
        final_only = (
            isinstance(spec, MambaSpec)
            or is_circular_kv_cache_spec(spec)
            or getattr(spec, "sliding_window", None) is not None
        )
        if final_only and not final:
            return None
        end = min(available, ready["num_computed_tokens"] + ready["num_external_tokens"])
        start = destination.sent_tokens.get(layer_name, ready["num_computed_tokens"])
        if start >= end:
            return None
        local_blocks = meta.local_block_ids[self.transfer_metadata.group_indices[local_index]]
        remote_blocks = ready["block_ids"][pp.group_indices[remote_index]]
        if isinstance(spec, MambaSpec):
            num_speculative_tokens = ready.get("num_speculative_tokens", 0)
            local_ids = (local_blocks[-1],)
            remote_ids = (remote_blocks[-num_speculative_tokens - 1],)
            if not local_ids[0] or not remote_ids[0]:
                raise ValueError("Required Mamba state block is not resident")
        elif is_circular_kv_cache_spec(spec):
            if len(local_blocks) != 1 or len(remote_blocks) != 1:
                raise ValueError("Circular Push requires one state block per request")
            local_scale = self.transfer_metadata.block_size_scales[local_index][0]
            remote_scale = pp.block_size_scales[remote_index][0]
            if local_scale != remote_scale:
                raise ValueError("Circular Push requires equal P/D physical block counts")
            local_ids = tuple(local_blocks[0] * local_scale + i for i in range(local_scale))
            remote_ids = tuple(remote_blocks[0] * remote_scale + i for i in range(remote_scale))
        else:
            local_scale = self.transfer_metadata.block_size_scales[local_index][0]
            remote_scale = pp.block_size_scales[remote_index][0]
            block_size = self.transfer_metadata.layer_block_sizes[local_index] // local_scale
            if block_size != pp.layer_block_sizes[remote_index] // remote_scale:
                raise ValueError("Push requires equal P/D kernel block sizes")
            if not final:
                end = end // block_size * block_size
            if end <= start:
                return None
            if final_only:
                # Window tables retain logical positions, including null slots.
                start = max(start, end - getattr(spec, "num_retained_tokens", spec.sliding_window - 1) - 1)
            first, last = start // block_size, (end + block_size - 1) // block_size
            local_ids, remote_ids = [], []
            for index in range(first, last):
                local_block = local_blocks[index // local_scale]
                remote_block = remote_blocks[index // remote_scale]
                if final_only and remote_block == 0:
                    continue
                if local_block == 0 or remote_block == 0:
                    raise ValueError("Required Push block is not resident")
                local_ids.append(local_block * local_scale + index % local_scale)
                remote_ids.append(remote_block * remote_scale + index % remote_scale)
            local_ids, remote_ids = tuple(local_ids), tuple(remote_ids)
        return LayerTransferRequest(transfer_id, tuple(local_ids), tuple(remote_ids), start, end), event

    def _execute_layer_transfer(self, batch: LayerTransferBatch) -> tuple[str | None, bool]:
        src_list, dst_list, length_list = [], [], []
        try:
            local_index = self._layer_indices[batch.layer_name]
            remote = batch.remote_metadata
            pp = remote.metadata_by_pp_rank[batch.remote_rank[0]]
            remote_index = pp.layer_names.index(batch.layer_name)
            worker = pp.metadata_by_pcp_rank[batch.remote_rank[1]].metadata_by_tp_rank[batch.remote_rank[2]]
            spec = self.kv_cache_specs[self.layer_name_to_spec_index[batch.layer_name]]
            if isinstance(spec, MambaSpec):
                self._append_mamba_transfer_addresses(
                    spec,
                    batch.remote_rank[1],
                    batch.remote_rank[2],
                    remote.tp_size,
                    {
                        (local_index, remote_index): [
                            (req.transfer_id, list(req.local_block_ids), list(req.remote_block_ids))
                            for req in batch.requests
                        ]
                    },
                    pp,
                    src_list,
                    dst_list,
                    length_list,
                )
            else:
                local_offset, remote_offset, heads = self._get_transfer_partition(
                    batch.layer_name, remote, batch.remote_rank
                )
                whole = isinstance(spec, (MLAAttentionSpec, SlidingWindowMLASpec)) or is_circular_kv_cache_spec(spec)
                local = self.transfer_metadata
                for cache_index, (local_addr, remote_addr) in enumerate(
                    zip(local.kv_caches_base_addr[local_index], worker.kv_caches_base_addr[remote_index], strict=True)
                ):
                    local_len = local.block_lens[local_index][cache_index]
                    remote_len = pp.block_lens[remote_index][cache_index]
                    if not whole:
                        local_len //= local.block_shapes[local_index][cache_index][0]
                        remote_len //= pp.block_shapes[remote_index][cache_index][0]
                    if local_len != remote_len:
                        raise ValueError("P/D cache slice lengths differ")
                    for req in batch.requests:
                        for local_block, remote_block in zip(req.local_block_ids, req.remote_block_ids, strict=True):
                            src_list.append(
                                local_addr
                                + local_block * local.block_strides[local_index][cache_index]
                                + local_offset * local_len
                            )
                            dst_list.append(
                                remote_addr
                                + remote_block * pp.block_strides[remote_index][cache_index]
                                + remote_offset * remote_len
                            )
                            length_list.append(local_len * heads)
            batch.event.synchronize()
        except Exception as exc:
            # No WRITE was submitted; this failure can be reported safely.
            return str(exc), True
        if src_list:
            session = f"{worker.local_ip}:{worker.te_rpc_port}"
            try:
                ret = self.engine.batch_transfer_sync_write(session, src_list, dst_list, length_list)
            except Exception as exc:
                return str(exc), False
            if ret < 0:
                # The API also returns -1 on timeout, without a drain ticket.
                return f"Mooncake WRITE failed: session={session}, ret={ret}", False
        return None, True

    def _process_sending_requests(self, wait: bool = False) -> None:
        self._process_ready_requests()
        pending = []
        for batch, future in self._pending_sends:
            if not wait and not future.done():
                pending.append((batch, future))
                continue
            try:
                error, writes_drained = future.result()
            except Exception as exc:
                # An exception crossing the native WRITE boundary does not
                # certify drain. Do not send DONE or release these source pages.
                error, writes_drained = str(exc), False
            if not writes_drained:
                notifications = []
                for req in batch.requests:
                    state = self._reqs_send_state[req.transfer_id]
                    for destination in state.destinations.values():
                        notifications.append(
                            self._sender_executor.submit(self._send_done, destination.ready, error, False)
                        )
                # Best effort failure notification, including a dead D. Never
                # let the next model step recycle source pages after an unknown
                # native WRITE outcome, even if the callback itself failed.
                for notification in notifications:
                    notification.result()
                self._control_error = RuntimeError(error)
                self.maybe_raise_thread_error()
            for req in batch.requests:
                state = self._reqs_send_state[req.transfer_id]
                destination = state.destinations[batch.remote_rank]
                destination.pending_layers.remove(batch.layer_name)
                if error:
                    state.error = error
                else:
                    destination.sent_tokens[batch.layer_name] = req.end_token
        self._pending_sends = pending
        self._callback_futures = [future for future in self._callback_futures if not future.done()]
        batches = {}
        now = time.monotonic()
        for transfer_id, state in list(self._reqs_send_state.items()):
            expired = now - state.start_time > envs.VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT
            if expired:
                state.error = "Push transfer timed out"
            for rank, destination in state.destinations.items():
                if destination.completion is not None or not destination.metadata_future.done():
                    continue
                try:
                    remote = destination.metadata_future.result()
                    expected = {
                        (pp, pcp, tp)
                        for pp in range(remote.pp_size)
                        for pcp in range(remote.pcp_size)
                        for tp in range(remote.tp_size)
                    }
                    if rank not in expected:
                        raise ValueError("Ready rank is absent from D topology")
                    state.expected_ranks = expected
                    if destination.required_layers is None:
                        destination.required_layers = {
                            name
                            for name in self._layer_indices
                            if self._get_transfer_partition(name, remote, rank) is not None
                        }
                    if state.error is None and state.metadata is not None:
                        for layer_name in destination.required_layers:
                            if layer_name not in state.available_layers or layer_name in destination.pending_layers:
                                continue
                            task = self._build_layer_transfer_request(transfer_id, state, destination, layer_name)
                            if task is not None:
                                request, event = task
                                key = (state.consumer_engine_id, rank, layer_name, id(event))
                                entry = batches.setdefault(key, (remote, event, []))
                                entry[2].append(request)
                                destination.pending_layers.add(layer_name)
                except Exception as exc:
                    state.error = str(exc)
                    logger.exception("Failed to prepare Push transfer %s", transfer_id)
            # A request-wide failure waits for ALL destination tasks to drain.
            in_flight = any(dest.pending_layers for dest in state.destinations.values())
            for destination in state.destinations.values():
                if destination.completion is not None or in_flight:
                    continue
                end = destination.ready["num_computed_tokens"] + destination.ready["num_external_tokens"]
                complete = destination.required_layers is not None and all(
                    destination.sent_tokens.get(layer, -1) >= end for layer in destination.required_layers
                )
                if state.error or complete:
                    destination.completion = self._sender_executor.submit(
                        self._send_done, destination.ready, state.error
                    )
            all_done = all(
                dest.completion is not None and dest.completion.done() for dest in state.destinations.values()
            )
            all_ready = state.expected_ranks is not None and set(state.destinations) == state.expected_ranks
            if not in_flight and all_done and (all_ready or expired):
                if state.inference_finished or (state.request_id is None and expired):
                    if state.request_id is not None:
                        self._finished_sending.add(state.request_id)
                    self._finished_transfers[transfer_id] = True
                    del self._reqs_send_state[transfer_id]
        for (_, rank, layer_name, _), (remote, event, requests) in batches.items():
            # A later layer/destination may have failed while building this pass.
            active = tuple(req for req in requests if self._reqs_send_state[req.transfer_id].error is None)
            for req in requests:
                if req not in active:
                    self._reqs_send_state[req.transfer_id].destinations[rank].pending_layers.remove(layer_name)
            if active:
                batch = LayerTransferBatch(layer_name, rank, remote, active, event)
                self._pending_sends.append((batch, self._sender_executor.submit(self._execute_layer_transfer, batch)))

    def save_kv_layer(self, layer_name: str) -> None:
        if not self.kv_transfer_config.is_kv_producer:
            return
        self.maybe_raise_thread_error()
        if layer_name in self._layer_indices:
            event = torch.npu.Event()
            event.record()
            for transfer_id in self._prefilling_transfer_ids:
                state = self._reqs_send_state[transfer_id]
                end = state.metadata.num_computed_tokens + state.metadata.num_scheduled_tokens
                state.available_layers[layer_name] = end, event
        self._process_sending_requests()

    def wait_for_save(self) -> None:
        if not self.kv_transfer_config.is_kv_producer:
            return
        # End-of-forward fallback for Mamba/auxiliary caches without named hooks.
        event = torch.npu.Event()
        event.record()
        for transfer_id in self._prefilling_transfer_ids:
            state = self._reqs_send_state[transfer_id]
            end = state.metadata.num_computed_tokens + state.metadata.num_scheduled_tokens
            for layer_name in self._layer_indices:
                state.available_layers[layer_name] = end, event
        self._process_sending_requests(wait=True)
        # Drain tasks just submitted above before the next step can overwrite
        # window/state pages or the scheduler can recycle any source blocks.
        while self._pending_sends:
            self._process_sending_requests(wait=True)
        self._prefilling_transfer_ids = ()

    def maybe_raise_thread_error(self) -> None:
        for thread in (self._sending_thread, self._recving_thread):
            if thread is not None and thread.error is not None:
                self._control_error = thread.error
        if self._control_error is not None:
            # Stop publishing destinations once the control path is fatal.
            self._stop_event.set()
            raise RuntimeError(
                "Push control failed with possible outstanding WRITEs; do not recycle destination blocks"
            ) from self._control_error

    def get_transfer_results(self) -> KVConnectorTransferResults:
        self.maybe_raise_thread_error()
        results = KVConnectorTransferResults()
        if self.kv_transfer_config.is_kv_producer:
            self._process_sending_requests()
            results.finished_sending = self._finished_sending
            self._finished_sending = set()
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
                elif kind == "ready_sent":
                    state.ready_sent = True
                elif kind == "done":
                    if (
                        payload["producer_engine_id"] != state.metadata.remote_engine_id
                        or payload["request_id"] != state.request_id
                    ):
                        continue
                    source = tuple(payload["producer_rank"])
                    if source not in state.expected_ranks or source in state.finished_ranks:
                        continue
                    if not payload["writes_drained"]:
                        self._control_error = RuntimeError(payload.get("error", "P WRITE drain is unknown"))
                        self.maybe_raise_thread_error()
                    state.finished_ranks.add(source)
                    state.failed |= payload["status"] == "error"
                elif kind == "failed":
                    # Discovery failed before publishing any destination to P.
                    state.failed = True
                    logger.error("Push discovery failed for %s: %s", state.request_id, payload)
                elif kind == "uncertain":
                    state.failed = True
                    if state.expected_ranks and state.finished_ranks == state.expected_ranks:
                        # Ready failed, but every P has already certified that
                        # its WRITEs ended. Report failure rather than success.
                        state.ready_sent = True
                        logger.error("Push Ready failed for %s: %s", state.request_id, payload)
                    else:
                        self._control_error = payload
                        self.maybe_raise_thread_error()
                if kind == "failed" or (
                    state.ready_sent and state.expected_ranks and state.finished_ranks == state.expected_ranks
                ):
                    results.finished_recving.add(state.request_id)
                    if state.failed:
                        results.failed_recving.add(state.request_id)
                    self._finished_transfers[transfer_id] = (state.request_id, state.metadata)
                    del self._reqs_recv_state[transfer_id]
            for state in self._reqs_recv_state.values():
                if time.monotonic() - state.start_time > envs.VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT:
                    state.failed = True
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
        if self._sender_executor is not None:
            self._sender_executor.shutdown(wait=True)
            self._sender_executor = None
        if self._zmq_context is not None:
            self._zmq_context.destroy(linger=0)
            self._zmq_context = None
