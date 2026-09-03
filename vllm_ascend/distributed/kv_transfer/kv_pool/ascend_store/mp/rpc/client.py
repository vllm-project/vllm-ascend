"""Multiprocess RPC client.

Thread ownership rules:

- Public request methods may be called from application threads.
- The outbound queue is a multi-producer, single-consumer queue.
- The DEALER socket and pending request map are owned by the I/O thread.
- Future callbacks run synchronously in the I/O thread and must not call
  blocking methods on this client.
"""

import contextlib
import itertools
import logging
import math
import queue
import socket
import threading
import time
from collections.abc import Sequence
from concurrent.futures import Future
from dataclasses import dataclass
from enum import Enum, auto

import zmq
from zmq.utils.monitor import recv_monitor_message

from .error import (
    MPClientClosedError,
    MPProtocolError,
    MPRemoteError,
    MPRequestTimeoutError,
    MPServerAbortedError,
    MPServerBusyError,
    MPServerUnavailableError,
)
from .protocol import MultipartMessage, ResponseStatus, SystemMethod, decode_response, encode_request, normalize_method

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _OutboundRequest:
    request_id: bytes
    method: str
    frames: MultipartMessage
    future: Future[list[bytes]]
    deadline_ns: int | None


@dataclass(frozen=True)
class _PendingRequest:
    method: str
    future: Future[list[bytes]]
    deadline_ns: int | None


class _ClientState(Enum):
    """State of the client I/O thread and its transport resources.

    Only the transitions shown below are supported. Calls that leave the state
    unchanged are omitted.

        +----------+                +--------------+                    +-----------+
        | STARTING |---- ready ---->| DISCONNECTED |---- connected ---->| CONNECTED |
        +----------+                +--------------+<-- disconnected ---+-----------+
             |                             |                                  |
             +-----------------------------+----------------------------------+
                                           |
                            +--------------+--------------+
                            |                             |
                       I/O failure                     close()
                            |                             |
                            v                             v
                        +--------+                   +---------+
                        | FAILED |----- close() ---->| CLOSING |
                        +--------+                   +---------+
                                                          | I/O thread exits
                                                          v
                                                      +--------+
                                                      | CLOSED |
                                                      +--------+
    """

    STARTING = auto()
    DISCONNECTED = auto()
    CONNECTED = auto()
    CLOSING = auto()
    FAILED = auto()
    CLOSED = auto()


class MPClient:
    """Thread-safe RPC facade backed by one DEALER-owning I/O thread."""

    def __init__(self, server_url: str):
        self._context = zmq.Context()
        self._server_url = server_url
        self._request_ids = itertools.count()

        self._outbound_queue: queue.Queue[_OutboundRequest] = queue.Queue()
        self._pending_requests: dict[bytes, _PendingRequest] = {}

        self._client_lifecycle_condition = threading.Condition()
        self._lifecycle_state = _ClientState.STARTING
        self._io_error: Exception | None = None

        self._notify_reader, self._notify_writer = socket.socketpair()
        self._notify_writer.setblocking(False)
        self._io_thread = threading.Thread(target=self._io_loop, daemon=True, name="ascend-store-mp-client")
        self._io_thread.start()

        with self._client_lifecycle_condition:
            self._client_lifecycle_condition.wait_for(lambda: self._lifecycle_state is not _ClientState.STARTING)
            io_error = self._io_error if self._lifecycle_state is _ClientState.FAILED else None

        if io_error is not None:
            self.close()
            raise RuntimeError("Failed to start MP client I/O thread") from io_error

    # ==============================
    # Public API
    # ==============================

    # Application threads use these methods without touching transport-owned
    # state. Requests cross to the I/O thread through the outbound queue.

    def __enter__(self) -> "MPClient":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    @property
    def is_transport_connected(self) -> bool:
        """Whether ZMQ reports an active transport connection."""
        with self._client_lifecycle_condition:
            return self._lifecycle_state is _ClientState.CONNECTED

    def wait_until_connected(self, timeout_ms: int = 5000) -> None:
        if timeout_ms <= 0:
            raise ValueError(f"timeout_ms must be greater than 0, got {timeout_ms}")

        terminal_states = {_ClientState.CONNECTED, _ClientState.CLOSING, _ClientState.FAILED, _ClientState.CLOSED}
        with self._client_lifecycle_condition:
            reached_terminal_state = self._client_lifecycle_condition.wait_for(
                lambda: self._lifecycle_state in terminal_states,
                timeout_ms / 1000,
            )
            if not reached_terminal_state:
                raise MPServerUnavailableError(f"Timed out connecting to MP server: {self._server_url}")
            if self._lifecycle_state is _ClientState.CONNECTED:
                return
            if self._lifecycle_state in {_ClientState.CLOSING, _ClientState.CLOSED}:
                raise MPClientClosedError("MP client is closed")
            if self._lifecycle_state is _ClientState.FAILED:
                raise MPServerUnavailableError("MP client I/O thread is unavailable") from self._io_error
            raise RuntimeError(f"Unexpected MP client lifecycle state: {self._lifecycle_state.name}")

    def submit_request(
        self,
        method: str,
        payloads: Sequence[bytes] | None = None,
        timeout_ms: int | None = None,
    ) -> Future[list[bytes]]:
        method_name = normalize_method(method)

        with self._client_lifecycle_condition:
            if self._lifecycle_state in {_ClientState.CLOSING, _ClientState.CLOSED}:
                raise MPClientClosedError("MP client is closed")
            if self._lifecycle_state is not _ClientState.CONNECTED:
                if self._lifecycle_state is _ClientState.FAILED:
                    raise MPServerUnavailableError("MP client I/O thread is unavailable") from self._io_error
                raise MPServerUnavailableError("MP server is unavailable")

            request_id = str(next(self._request_ids)).encode()
            future: Future[list[bytes]] = Future()
            deadline_ns = self._deadline_from_timeout(timeout_ms)
            frames = encode_request(request_id, method_name, payloads or (), deadline_ns)

            self._outbound_queue.put(_OutboundRequest(request_id, method_name, frames, future, deadline_ns))
            self._notify_io_thread()

        return future

    def request(
        self, method: str, payloads: Sequence[bytes] | None = None, timeout_ms: int | None = 5000
    ) -> list[bytes]:
        return self.submit_request(method, payloads, timeout_ms=timeout_ms).result()

    def ping(self, timeout_ms: int = 5000) -> str:
        return self.request(SystemMethod.PING, timeout_ms=timeout_ms)[0].decode()

    def echo(self, payload: bytes, timeout_ms: int = 5000) -> bytes:
        return self.request(SystemMethod.ECHO, [payload], timeout_ms=timeout_ms)[0]

    # ==============================
    # I/O thread
    # ==============================

    # This thread exclusively owns the DEALER socket and pending-request map. It
    # sends queued frames, completes responses, observes connection changes, and
    # expires deadlines without exposing transport state to application threads.

    def _io_loop(self) -> None:
        zmq_socket = None
        monitor_socket = None
        try:
            zmq_socket = self._context.socket(zmq.DEALER)
            monitor_socket = zmq_socket.get_monitor_socket(events=zmq.EVENT_CONNECTED | zmq.EVENT_DISCONNECTED)
            zmq_socket.connect(self._server_url)

            poller = zmq.Poller()
            poller.register(zmq_socket, zmq.POLLIN)
            poller.register(monitor_socket, zmq.POLLIN)
            poller.register(self._notify_reader.fileno(), zmq.POLLIN)
            with self._client_lifecycle_condition:
                if self._lifecycle_state is _ClientState.STARTING:
                    self._lifecycle_state = _ClientState.DISCONNECTED
                    self._client_lifecycle_condition.notify()

            while True:
                timeout_ms = self._next_poll_timeout_ms()
                events = dict(poller.poll() if timeout_ms is None else poller.poll(timeout_ms))

                if self._notify_reader.fileno() in events:
                    self._notify_reader.recv(4096)
                    with self._client_lifecycle_condition:
                        closing = self._lifecycle_state is _ClientState.CLOSING
                    if closing:
                        self._fail_pending(MPClientClosedError("MP client was closed"))
                        break
                    self._process_outbound(zmq_socket)

                if zmq_socket in events:
                    self._process_inbound(zmq_socket)
                if monitor_socket in events:
                    self._process_monitor_event(zmq_socket, monitor_socket)
                self._expire_pending_requests()
        except Exception as exc:
            failure = self._as_server_unavailable(exc)
            with self._client_lifecycle_condition:
                self._io_error = exc
                if self._lifecycle_state is not _ClientState.CLOSING:
                    self._lifecycle_state = _ClientState.FAILED
                self._client_lifecycle_condition.notify_all()
            self._fail_pending(failure)
        finally:
            with self._client_lifecycle_condition:
                if self._lifecycle_state is _ClientState.CONNECTED:
                    self._lifecycle_state = _ClientState.DISCONNECTED
            if monitor_socket is not None:
                monitor_socket.close(linger=0)
            if zmq_socket is not None:
                zmq_socket.close(linger=0)
            self._notify_reader.close()

    @staticmethod
    def _as_server_unavailable(exc: Exception) -> MPServerUnavailableError:
        """Expose a failed client I/O loop through the degradable RPC contract."""
        if isinstance(exc, MPServerUnavailableError):
            return exc
        failure = MPServerUnavailableError("MP client I/O thread is unavailable")
        failure.__cause__ = exc
        return failure

    @staticmethod
    def _deadline_from_timeout(timeout_ms: int | None) -> int | None:
        if timeout_ms is None:
            return None
        if timeout_ms <= 0:
            raise ValueError(f"timeout_ms must be greater than 0, got {timeout_ms}")
        return time.monotonic_ns() + timeout_ms * 1_000_000

    def _notify_io_thread(self) -> None:
        with contextlib.suppress(BlockingIOError):
            self._notify_writer.send(b"\x01")

    def _process_outbound(self, zmq_socket: zmq.Socket) -> None:
        while True:
            try:
                request = self._outbound_queue.get_nowait()
            except queue.Empty:
                return

            if not request.future.set_running_or_notify_cancel():
                continue
            if request.deadline_ns is not None and request.deadline_ns <= time.monotonic_ns():
                self._set_request_timeout(request.method, request.future)
                continue

            try:
                zmq_socket.send_multipart(request.frames, flags=zmq.NOBLOCK)
            except zmq.Again:
                # Non-blocking send reports transport backpressure through
                # Again. The server has not accepted this request, so fail
                # it as busy instead of blocking the client I/O loop.
                request.future.set_exception(MPServerBusyError("MP client outbound transport is busy"))
                continue
            except Exception as exc:
                # Removing a request from the outbound queue and adding it to
                # the pending map is one ownership handoff. If transport send
                # fails between those steps, this thread still owns and
                # must complete the request before the I/O loop terminates.
                failure = self._as_server_unavailable(exc)
                request.future.set_exception(failure)
                raise failure from exc

            self._pending_requests[request.request_id] = _PendingRequest(
                request.method, request.future, request.deadline_ns
            )

    def _process_inbound(self, zmq_socket: zmq.Socket) -> None:
        frames = zmq_socket.recv_multipart()
        if not frames:
            logger.error("Discarding malformed response without a request ID")
            return

        request_id = frames[0]
        pending = self._pending_requests.pop(request_id, None)
        if pending is None:
            logger.debug("Discarding response for inactive request ID %r", request_id)
            return
        if pending.future.done():
            return

        try:
            _, response_method, status, responses = decode_response(frames)
        except MPProtocolError as exc:
            pending.future.set_exception(exc)
            return

        if response_method != pending.method:
            pending.future.set_exception(
                MPProtocolError(f"Response method mismatch: expected {pending.method!r}, got {response_method!r}")
            )
            return

        if status is not ResponseStatus.OK:
            message = responses[0].decode(errors="replace") if responses else "Unknown server error"
            remote_traceback = responses[1].decode(errors="replace") if len(responses) > 1 else None
            if status is ResponseStatus.BUSY:
                error = MPServerBusyError(message)
            elif status is ResponseStatus.ABORTED:
                error = MPServerAbortedError(message)
            else:
                error = MPRemoteError(
                    message,
                    method=response_method,
                    request_id=request_id,
                    remote_traceback=remote_traceback,
                )
            pending.future.set_exception(error)
            return

        pending.future.set_result(list(responses))

    def _drain_inbound(self, zmq_socket: zmq.Socket) -> None:
        while zmq_socket.poll(timeout=0, flags=zmq.POLLIN):
            self._process_inbound(zmq_socket)

    def _handle_transport_disconnected(self, zmq_socket: zmq.Socket) -> None:
        with self._client_lifecycle_condition:
            if self._lifecycle_state is not _ClientState.CONNECTED:
                return
            self._lifecycle_state = _ClientState.DISCONNECTED

        self._drain_inbound(zmq_socket)
        self._fail_pending(MPServerUnavailableError(f"MP server disconnected: {self._server_url}"))

    def _handle_transport_connected(self) -> None:
        with self._client_lifecycle_condition:
            if self._lifecycle_state not in {_ClientState.STARTING, _ClientState.DISCONNECTED}:
                return
            self._lifecycle_state = _ClientState.CONNECTED
            self._client_lifecycle_condition.notify_all()

    def _process_monitor_event(self, zmq_socket: zmq.Socket, monitor_socket: zmq.Socket) -> None:
        monitor_event = recv_monitor_message(monitor_socket)
        event = monitor_event["event"]
        if event == zmq.EVENT_DISCONNECTED:
            self._handle_transport_disconnected(zmq_socket)
        elif event == zmq.EVENT_CONNECTED:
            self._handle_transport_connected()

    @staticmethod
    def _set_request_timeout(method: str, future: Future[list[bytes]]) -> None:
        if not future.done():
            future.set_exception(MPRequestTimeoutError(f"Timed out waiting for response to {method}"))

    def _next_poll_timeout_ms(self) -> int | None:
        deadlines_ns = [
            request.deadline_ns for request in self._pending_requests.values() if request.deadline_ns is not None
        ]
        if not deadlines_ns:
            return None

        remaining_ns = min(deadlines_ns) - time.monotonic_ns()
        return max(0, math.ceil(remaining_ns / 1_000_000))

    def _expire_pending_requests(self) -> None:
        now_ns = time.monotonic_ns()
        expired_request_ids = [
            request_id
            for request_id, request in self._pending_requests.items()
            if request.deadline_ns is not None and request.deadline_ns <= now_ns
        ]
        for request_id in expired_request_ids:
            request = self._pending_requests.pop(request_id)
            self._set_request_timeout(request.method, request.future)

    def _fail_pending(self, exc: Exception) -> None:
        while True:
            try:
                request = self._outbound_queue.get_nowait()
            except queue.Empty:
                break
            if request.future.set_running_or_notify_cancel():
                request.future.set_exception(exc)

        for request in self._pending_requests.values():
            if not request.future.done():
                request.future.set_exception(exc)
        self._pending_requests.clear()

    # ==============================
    # Shutdown
    # ==============================

    # Closing is a cross-thread handoff. New calls are rejected first; the I/O
    # thread then completes all queued and pending requests and closes its sockets.
    # The caller joins that thread before releasing the remaining shared resources.

    def close(self) -> None:
        with self._client_lifecycle_condition:
            if self._lifecycle_state is _ClientState.CLOSED:
                return

            if self._lifecycle_state is not _ClientState.CLOSING:
                self._lifecycle_state = _ClientState.CLOSING
                self._client_lifecycle_condition.notify_all()
                if self._io_thread.is_alive():
                    with contextlib.suppress(OSError):
                        self._notify_io_thread()

        self._io_thread.join()

        with self._client_lifecycle_condition:
            if self._lifecycle_state is _ClientState.CLOSED:
                return
            self._notify_writer.close()
            self._context.term()
            self._lifecycle_state = _ClientState.CLOSED
