# SPDX-License-Identifier: Apache-2.0
"""Existing MemFabric data path with a small JSON/ZMQ lease control plane."""

from __future__ import annotations

import base64
import hashlib
import json
import threading
import time

import zmq
from zmq.utils import jsonapi

from .configuration import DEFAULT_URMA_SWAP_MIB, configure_urma_staging, parse_urma_swap_mib
from .protocol import STAGE_BYTES, LeaseRegistry, validate_wire

# Failure-only process owners. Each reader serializes mutation with its lock;
# entries disappear only after both transport teardown stages have succeeded.
# Keeping the actual owners here is necessary if an engine drops its connector.
_retained_readers: dict[int, MemfabricReader] = {}


class TransferCancelled(Exception):
    """Cancellation before any transport read; safe to fail the async load."""


class LeaseServer:
    def __init__(self, endpoint: str, registry: LeaseRegistry, stage=None):
        self.registry = registry
        self.stage = stage
        self.endpoint = endpoint
        self.stop = threading.Event()
        self.ready = threading.Event()
        self.error: BaseException | None = None
        self.thread = threading.Thread(target=self._run, name="hisparse-pd-control", daemon=True)
        self.thread.start()
        if not self.ready.wait(10) or self.error:
            raise RuntimeError("HiSparse P/D control server failed to bind") from self.error

    def _run(self):
        try:
            with zmq.Context() as context, context.socket(zmq.REP) as socket:
                socket.setsockopt(zmq.LINGER, 0)
                socket.setsockopt(zmq.MAXMSGSIZE, 8 * 1024 * 1024)
                socket.bind(self.endpoint)
                self.ready.set()
                while not self.stop.is_set():
                    if not socket.poll(100):
                        continue
                    try:
                        raw_message = socket.recv()
                        message = jsonapi.loads(raw_message)
                        validate_wire(message)
                        if message["op"] == "lookup":
                            lookup = getattr(self.registry, "lookup_wire", None)
                            if lookup is None:
                                raise ValueError("This producer does not publish shared snapshots")
                            response = lookup(message)
                        elif message["op"] == "acquire":
                            response = self.registry.acquire(
                                message["lease_id"],
                                message["consumer"],
                                message["identity"],
                                message["rank"],
                                message["world_size"],
                            )
                        elif message["op"] == "release":
                            self.registry.release(
                                message["lease_id"], message["consumer"], drained=message["drained"] is True
                            )
                            response = {"status": "released"}
                        elif message["op"] in ("stage_prepare", "stage_release", "stage_advance"):
                            if self.stage is None:
                                raise ValueError("Producer HBM staging is not available")
                            method = {
                                "stage_prepare": self.stage.prepare,
                                "stage_release": self.stage.release,
                                "stage_advance": self.stage.advance,
                            }[message["op"]]
                            response = method(message)
                        else:
                            raise ValueError("Unknown HiSparse P/D control operation")
                    except (KeyError, TypeError, ValueError, RuntimeError) as exc:
                        response = {"status": "error", "error": str(exc)}
                        if isinstance(exc, json.JSONDecodeError):
                            # Preserve the failing control bytes before any
                            # subsequent receive can obscure their provenance.
                            start = max(0, exc.pos - 64)
                            response["wire_failure"] = {
                                "bytes": len(raw_message),
                                "sha256": hashlib.sha256(raw_message).hexdigest(),
                                "position": exc.pos,
                                "context_start": start,
                                "context_base64": base64.b64encode(raw_message[start : start + 192]).decode(),
                                "raw_base64": base64.b64encode(raw_message).decode()
                                if len(raw_message) <= 4096
                                else None,
                            }
                    socket.send_json(response)
        except BaseException as exc:
            self.error = exc
            self.ready.set()

    def close(self):
        self.stop.set()
        self.thread.join(timeout=10)
        if self.thread.is_alive():
            raise RuntimeError("HiSparse P/D control server did not stop")


class RpcClient:
    """One request's control connection, exclusively owned by its receive thread.

    Only a matching reply permits reuse. Any failure discards that REQ socket;
    a later explicit operation can open a fresh socket, never retry this one.
    The request's context manager closes everything on the same thread.
    """

    def __init__(self, endpoint: str, *, on_stall=None):
        self.endpoint = endpoint
        self.on_stall = on_stall
        self._owner = threading.get_ident()
        self._context = None
        self._socket = None
        self._closed = False

    def _check_owner(self):
        if threading.get_ident() != self._owner:
            raise RuntimeError("HiSparse control connection belongs to a different thread")
        if self._closed:
            raise RuntimeError("HiSparse control connection is closed")

    def __enter__(self):
        self._check_owner()
        return self

    def __exit__(self, *_):
        self.close()

    def _discard_socket(self):
        if self._socket is not None:
            self._socket.close(linger=0)
            self._socket = None

    def call(self, message: dict, timeout_seconds: float) -> dict:
        self._check_owner()
        try:
            if self._context is None:
                self._context = zmq.Context()
            if self._socket is None:
                self._socket = self._context.socket(zmq.REQ)
                self._socket.setsockopt(zmq.LINGER, 0)
                self._socket.setsockopt(zmq.MAXMSGSIZE, 64 * 1024 * 1024)
                self._socket.connect(self.endpoint)
            return self._exchange(self._socket, message, timeout_seconds)
        except BaseException:
            self._discard_socket()
            raise

    def close(self):
        if self._closed:
            return
        self._check_owner()
        self._discard_socket()
        if self._context is not None:
            self._context.term()
            self._context = None
        self._closed = True

    def _exchange(self, socket, message, timeout_seconds):
        # Preserve the actual encoded bytes before sending.
        # The old failure path re-encoded the dict after the error arrived.
        wire = jsonapi.dumps(message)
        sent_sha256_before = hashlib.sha256(wire).hexdigest()
        sent_base64_before = base64.b64encode(wire).decode() if len(wire) <= 4096 else None
        socket.send(wire, copy=True)
        deadline = time.monotonic() + timeout_seconds
        captured = False
        while True:
            remaining_ms = max(1, int((deadline - time.monotonic()) * 1000))
            interval = remaining_ms if self.on_stall is None else min(1000, remaining_ms)
            if socket.poll(interval):
                break
            if self.on_stall is not None and not captured and time.monotonic() >= deadline - timeout_seconds + 5:
                self.on_stall(self.endpoint, message, wire, sent_sha256_before, sent_base64_before)
                captured = True
            if time.monotonic() >= deadline:
                raise TimeoutError("HiSparse P/D control request timed out; source lease remains retained")
        result = socket.recv_json()
        if result.get("status") == "error":
            evidence = result.get("wire_failure")
            if evidence is not None:
                expected = wire
                details = {
                    "received": evidence,
                    "sent_bytes": len(expected),
                    "sent_sha256": sent_sha256_before,
                    "send_buffer_sha256_after": hashlib.sha256(expected).hexdigest(),
                    "send_buffer_unchanged": hashlib.sha256(expected).hexdigest() == sent_sha256_before,
                    "sent_base64": sent_base64_before,
                }
                raise ValueError(result["error"] + "; control_wire=" + json.dumps(details, sort_keys=True))
            raise ValueError(result["error"])
        return result


def rpc(endpoint: str, message: dict, timeout_seconds: float, *, client=None) -> dict:
    if client is not None:
        if endpoint != client.endpoint:
            raise ValueError("Control connection endpoint differs from this request")
        return client.call(message, timeout_seconds)
    with RpcClient(endpoint) as temporary:
        return temporary.call(message, timeout_seconds)


def acquire(endpoint: str, message: dict, timeout_seconds: float, cancelled: threading.Event, *, client=None) -> dict:
    deadline = time.monotonic() + timeout_seconds
    while not cancelled.is_set():
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("Producer did not publish its completed KV before the deadline")
        result = rpc(endpoint, {"op": "acquire", **message}, remaining, client=client)
        if result["status"] == "ready":
            return result["manifest"]
        if result["status"] != "pending":
            raise ValueError("Unexpected producer lease state")
        cancelled.wait(min(remaining, 0.1))
    raise TransferCancelled("HiSparse P/D request was cancelled before acquisition")


def validate_prepared_stage(response, message):
    validate_wire(response)
    if (
        response.get("status") != "ready"
        or response.get("chunk_id") != message["chunk_id"]
        or response.get("length") != message["length"]
        or type(response.get("generation")) is not int
        or response["generation"] <= 0
        or type(response.get("stage_address")) is not int
        or response["stage_address"] <= 0
        or response.get("capacity") != STAGE_BYTES
    ):
        raise ValueError("Producer stage response differs from the bounded request")
    return response


def advance_stage(endpoint, released, following, timeout_seconds, *, client=None):
    response = rpc(
        endpoint,
        {**released, "op": "stage_advance", "released": released, "next": following},
        timeout_seconds,
        client=client,
    )
    ack = response.get("released", {})
    if (
        response.get("status") != "stage_advanced"
        or ack.get("status") != "stage_released"
        or ack.get("generation") != released["generation"]
        or ack.get("chunk_id") != released["chunk_id"]
    ):
        raise ValueError("Stage advance has no matching drained release acknowledgement")
    return validate_prepared_stage(response.get("prepared", {}), following)


def prepare_stage(endpoint, message, timeout_seconds, *, client=None):
    """Poll a busy slot within the caller's transfer deadline.

    The control server also completes synchronous H2D copies. A five-second
    polling interval must not shorten its response budget during another
    request's prefill. Uncertain replies still retain the source lease.
    """
    deadline = time.monotonic() + timeout_seconds
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("Producer stage remains owned; no timeout reuse is permitted")
        response = rpc(endpoint, {"op": "stage_prepare", **message}, remaining, client=client)
        if response.get("status") == "ready":
            return validate_prepared_stage(response, message)
        if response.get("status") != "pending":
            raise ValueError("Unexpected producer stage state")
        time.sleep(min(remaining, 0.01))


class MemfabricReader:
    """Register only real HBM; Host SOURCE ranges never enter MemFabric.

    Producer HBM staging is separate from MemFabric's own 16 MiB URMA scratch.
    Consumer SOURCE destinations remain raw Host addresses. Whole real owners
    remain retained until device drain plus both engine teardown stages pass.
    """

    def __init__(
        self,
        *,
        producer: bool,
        device: int,
        host: str,
        protocol: str,
        regions: dict,
        owners: list,
        urma_swap_space_mib=DEFAULT_URMA_SWAP_MIB,
    ):
        # Construction only captures owners. The worker must store this reader
        # before initialize() can register pointers or raise during rollback.
        self.producer = producer
        self.device = device
        self.host = host
        self.protocol = protocol
        if protocol != "device_urma":
            raise ValueError("HiSparse P/D mapped-to-host transport requires device_urma")
        self.urma_swap_space_mib = parse_urma_swap_mib(urma_swap_space_mib)
        self.staging_config = None
        self.regions = dict(regions)
        self.owners = list(owners)
        self.engine = None
        self.session = None
        self.initialized = False
        self.closed = False
        self._started = False
        self._closing = False
        self._destroyed = False
        self._uninitialized = False
        self._reads_started = False
        self._failed_reads = []
        self._lock = threading.RLock()

    def _teardown_methods(self):
        raw_engine = self.engine._engine
        methods = tuple(getattr(raw_engine, name, None) for name in ("destroy", "unInitialize"))
        if not all(callable(method) for method in methods):
            raise RuntimeError("MemFabric requires both destroy and unInitialize before borrowing HiSparse mappings")
        return methods

    def initialize(self):
        with self._lock:
            if self._started or self.closed:
                raise RuntimeError("MemFabric reader cannot be initialized twice")
            self._started = True
            try:
                self.staging_config = configure_urma_staging(self.urma_swap_space_mib)
                from vllm_ascend.distributed.kv_transfer.utils.memfabric_transfer_engine import global_memfabric_te

                global_memfabric_te.configure(
                    role="Prefill" if self.producer else "Decode",
                    device_id=self.device,
                    transfer_protocol=self.protocol,
                )
                self.engine = global_memfabric_te.get_transfer_engine(self.host)
                self.session = global_memfabric_te.unique_id
                self._teardown_methods()  # Reject incomplete APIs before registration.
                for (address, size), role in self.regions.items():
                    if role != "hisparse_source":
                        if self.engine.register_memory(address, size) != 0:
                            raise RuntimeError(
                                f"MemFabric cannot register {role} allocation; no alternate source address is attempted"
                            )
                self.initialized = True
            except BaseException as initialize_error:
                try:
                    self.close()
                except BaseException as cleanup_error:
                    raise initialize_error from cleanup_error
                raise

    def read(self, session: str, descriptors: list[tuple[int, int, int]]) -> None:
        with self._lock:
            if not self.initialized or self.closed or self._closing:
                raise RuntimeError("MemFabric reader is not available for reads")
            if not descriptors:
                raise ValueError("Empty transfer")
            destinations, sources, lengths = map(list, zip(*descriptors))
            self._reads_started = True
            try:
                if self.engine.batch_transfer_sync_read(session, destinations, sources, lengths) != 0:
                    raise RuntimeError("MemFabric read failed; source and destination ownership must be retained")
            except BaseException:
                # A failed synchronous API is not proof that partially queued
                # work has drained. Do not reuse this reader or its descriptors.
                self._failed_reads.append((session, destinations, sources, lengths))
                self._closing = True
                _retained_readers[id(self)] = self
                raise

    def close(self, *, device_drained: bool = False):
        with self._lock:
            if self.closed:
                return
            self._closing = True
            try:
                if self._reads_started and not device_drained:
                    raise RuntimeError("Device drain must succeed before MemFabric read owners can be released")
                if self.engine is not None:
                    destroy, uninitialize = self._teardown_methods()
                    if not self._destroyed:
                        result = destroy()
                        if result not in (None, 0):
                            raise RuntimeError(f"MemFabric destroy returned {result}")
                        self._destroyed = True
                    if not self._uninitialized:
                        result = uninitialize()
                        if result not in (None, 0):
                            raise RuntimeError(f"MemFabric unInitialize returned {result}")
                        self._uninitialized = True
            except BaseException:
                _retained_readers[id(self)] = self
                raise
            self.closed = True
            self.initialized = False
            self._failed_reads.clear()
            self.regions.clear()
            self.owners.clear()
            _retained_readers.pop(id(self), None)
