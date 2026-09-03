"""Multiprocess RPC server.

The ROUTER socket is owned exclusively by the thread running ``run``.
Completed handler responses return through a queue, and a notification socket
wakes that thread.
"""

import logging
import queue
import socket
import threading
import time
import traceback
from collections import deque
from collections.abc import Callable, Hashable, Iterable
from concurrent.futures import Future
from dataclasses import dataclass
from enum import Enum, auto
from functools import partial

import zmq

from .error import MPServerBusyError
from .executor import InlineExecutor, TaskExecutor
from .protocol import MultipartMessage, ResponseStatus, SystemMethod, decode_request, encode_response, normalize_method

logger = logging.getLogger(__name__)

RequestHandler = Callable[[tuple[bytes, ...]], Iterable[bytes]]
ExecutionKeyFactory = Callable[[bytes, tuple[bytes, ...]], Hashable]
ServerResponse = MultipartMessage
_RequestKey = tuple[bytes, bytes]

_RESPONSE_LINGER_MS = 1000


@dataclass(frozen=True)
class Route:
    """Bind one RPC method to its handler and executor.

    The optional key factory runs in the server I/O thread and must remain
    lightweight. It is intended for executors that require keyed execution.
    """

    method: str
    handler: RequestHandler
    executor: TaskExecutor
    key_factory: ExecutionKeyFactory | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "method", normalize_method(self.method))
        if not callable(self.handler):
            raise TypeError("handler must be callable")
        if not callable(getattr(self.executor, "submit", None)):
            raise TypeError("executor must define submit")
        if not callable(getattr(self.executor, "shutdown", None)):
            raise TypeError("executor must define shutdown")
        if self.key_factory is not None and not callable(self.key_factory):
            raise TypeError("key_factory must be callable")


@dataclass(frozen=True)
class _AcceptedRequest:
    identity: bytes
    request_id: bytes
    method: str
    deadline_ns: int | None

    @property
    def key(self) -> _RequestKey:
        return self.identity, self.request_id


@dataclass(frozen=True)
class _ResponseEnvelope:
    frames: ServerResponse
    request_key: _RequestKey | None = None
    is_abort_response: bool = False


class _ServerState(Enum):
    """State of server admission, request draining, and resource ownership.

    Only the transitions shown below are supported. Calls that leave the state
    unchanged are omitted.

        +-------+                  +---------+
        | READY |----- run() ----->| RUNNING |---------------------- run failure -----------------------+
        +-------+                  +---------+                                                          |
            |                           |                                                               |
            +---------------------------+                                                               |
                                        |                                                               |
                      +-----------------+---------------------------------------------------------------+
                      |                                                                                 |
               request_stop()                                                                        abort()
                      |                                                                                 |
                      v                                                                                 v
                +----------+                                +---------+                  +--------+     |
                | DRAINING |--- accepted responses sent --->| DRAINED |---- close() ---->| CLOSED |     |
                +----------+                                +---------+                  +--------+     |
                      |                                          |                                      |
            abort() / run failure                             abort()                                   |
                      |                                          |                                      |
                      +------------------------------------------+--------------------------------------+
                                                                 |
                                                                 v
                                                            +----------+
                                                            | ABORTING |
                                                            +----------+
                                                                 | forced cleanup
                                                                 v
                                                            +---------+
                                                            | ABORTED |
                                                            +---------+

    Construction failure closes resources before an MPServer is returned and
    therefore sits outside this state machine.
    """

    READY = auto()
    RUNNING = auto()
    DRAINING = auto()
    DRAINED = auto()
    ABORTING = auto()
    ABORTED = auto()
    CLOSED = auto()


class _AdmissionResult(Enum):
    """Outcome of claiming one request in the server lifecycle."""

    ACCEPTED = auto()
    STOPPING = auto()
    DUPLICATE = auto()


class MPServer:
    """Serve RPC routes and own their executors until the server is closed.

    Graceful shutdown moves from RUNNING through DRAINING and DRAINED to
    CLOSED. Forced shutdown moves from any live state through ABORTING to
    ABORTED.
    """

    def __init__(self, bind_url: str, routes: Iterable[Route] = ()):
        system_executor = InlineExecutor()
        all_routes = (
            Route(SystemMethod.PING, self._handle_ping, system_executor),
            Route(SystemMethod.ECHO, self._handle_echo, system_executor),
            *routes,
        )
        # The route table dispatches requests, while a separate executor list owns
        # shutdown because multiple routes may share one executor. Collect that
        # list first so route validation failures can still shut down every
        # referenced executor exactly once.
        self._executors = self._collect_owned_executors(all_routes)
        try:
            self._routes = self._index_routes(all_routes)
        except BaseException:
            self._shutdown_executors()
            raise
        try:
            self._notify_reader, self._notify_writer = socket.socketpair()
        except BaseException:
            self._shutdown_executors()
            raise

        self._completed_response_queue: queue.Queue[_ResponseEnvelope] = queue.Queue()
        self._send_backlog: deque[_ResponseEnvelope] = deque()
        self._notify_writer.setblocking(False)
        self._notify_lock = threading.Lock()
        self._close_lock = threading.Lock()
        self._state_condition = threading.Condition()
        self._state = _ServerState.READY
        self._run_thread: threading.Thread | None = None
        self._accepted_requests: dict[_RequestKey, _AcceptedRequest] = {}
        self._graceful_deadline_ns: int | None = None
        self._transport_closed = False

        context = None
        zmq_socket = None
        try:
            context = zmq.Context()
            zmq_socket = context.socket(zmq.ROUTER)
            self.endpoint = self._bind(zmq_socket, bind_url)
        except BaseException:
            if zmq_socket is not None:
                zmq_socket.close(linger=0)
            if context is not None:
                context.term()
            self._notify_reader.close()
            self._notify_writer.close()
            self._shutdown_executors()
            self._state = _ServerState.CLOSED
            raise

        self._context = context
        self._socket = zmq_socket

    def __enter__(self) -> "MPServer":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    @staticmethod
    def _index_routes(route_definitions: Iterable[object]) -> dict[str, Route]:
        indexed_routes = {}
        for route in route_definitions:
            if not isinstance(route, Route):
                raise TypeError(f"routes must contain Route instances, got {type(route).__name__}")
            if route.method in indexed_routes:
                raise ValueError(f"Duplicate RPC method: {route.method}")
            indexed_routes[route.method] = route
        return indexed_routes

    @staticmethod
    def _collect_owned_executors(routes: Iterable[object]) -> tuple[TaskExecutor, ...]:
        executors = {}
        for route in routes:
            # Route validation runs after ownership is claimed so every valid
            # executor can still be shut down when another entry is malformed.
            if not isinstance(route, Route):
                continue
            executors.setdefault(id(route.executor), route.executor)
        return tuple(executors.values())

    @staticmethod
    def _bind(zmq_socket: zmq.Socket, bind_url: str) -> str:
        if bind_url.endswith(":*"):
            base_url = bind_url[:-2]
            port = zmq_socket.bind_to_random_port(base_url)
            return f"{base_url}:{port}"

        zmq_socket.bind(bind_url)
        return bind_url

    # ==============================
    # Server lifecycle
    # ==============================

    # Public lifecycle methods control admission but leave ROUTER I/O to the run
    # thread. Graceful stop keeps accepted requests until their responses are sent
    # and uses their client deadlines to bound the drain; abort cancels queued work
    # and stops without waiting for running handlers.

    def _should_stop_run(self) -> bool:
        with self._state_condition:
            stopping = self._state is _ServerState.DRAINING or self._state is _ServerState.ABORTING
            return stopping and not self._accepted_requests

    def run(self) -> None:
        with self._state_condition:
            if self._state in {_ServerState.ABORTED, _ServerState.CLOSED}:
                raise RuntimeError("MPServer is closed")
            # request_stop() may win the race with a newly started run thread.
            if self._run_thread is not None or self._state not in {_ServerState.READY, _ServerState.DRAINING}:
                raise RuntimeError("MPServer.run() can only be called once")
            if self._state is _ServerState.READY:
                self._state = _ServerState.RUNNING
            self._run_thread = threading.current_thread()

        try:
            poller = zmq.Poller()
            poller.register(self._socket, zmq.POLLIN)
            poller.register(self._notify_reader.fileno(), zmq.POLLIN)
            socket_events = zmq.POLLIN

            while not self._should_stop_run():
                # Backpressure must drain queued responses before more requests
                # are accepted, otherwise memory use can grow without bound.
                expected_socket_events = zmq.POLLOUT if self._send_backlog else zmq.POLLIN
                if socket_events != expected_socket_events:
                    socket_events = expected_socket_events
                    poller.modify(self._socket, socket_events)

                events = dict(poller.poll())

                if self._notify_reader.fileno() in events:
                    self._process_run_notification()
                    self._send_responses()

                socket_event = events.get(self._socket, 0)
                if socket_event & zmq.POLLOUT:
                    self._send_responses()
                if socket_event & zmq.POLLIN and not self._send_backlog:
                    self._receive_and_dispatch_request()
        except BaseException:
            self.abort()
            raise
        finally:
            with self._state_condition:
                controlled_stop = self._state in {_ServerState.DRAINING, _ServerState.ABORTING}
            # A stopped run loop must never leave its ROUTER socket doing I/O.
            with self._close_lock:
                self._close_transport(_RESPONSE_LINGER_MS if controlled_stop else 0)

            with self._state_condition:
                self._run_thread = None
                if self._state is _ServerState.DRAINING:
                    self._state = _ServerState.DRAINED
                aborting = self._state is _ServerState.ABORTING
                # wait_until_stopped() waits for this exact predicate.
                self._state_condition.notify_all()
            if aborting:
                self._release_resources(_ServerState.ABORTED, shutdown_executors=False)

    def request_stop(self) -> bool:
        """Stop accepting work if every accepted request has a finite deadline."""
        with self._state_condition:
            if self._state in {_ServerState.DRAINED, _ServerState.CLOSED}:
                return True
            if self._state in {_ServerState.ABORTING, _ServerState.ABORTED}:
                return False
            if self._state is _ServerState.DRAINING:
                return True

            requests = tuple(self._accepted_requests.values())
            indefinite_count = sum(request.deadline_ns is None for request in requests)
            if indefinite_count:
                logger.warning(
                    "Graceful MP server shutdown rejected: %d accepted request(s) have no deadline; call abort()",
                    indefinite_count,
                )
                return False

            deadlines_ns = [request.deadline_ns for request in requests if request.deadline_ns is not None]
            # Reuse client deadlines instead of inventing a shorter server-side
            # shutdown timeout that could terminate otherwise valid requests.
            self._graceful_deadline_ns = max(deadlines_ns, default=None)
            self._state = _ServerState.DRAINING
            run_active = self._run_thread is not None
        if run_active:
            self._wake_run_loop()
        return True

    def wait_until_stopped(self, timeout: float | None = None) -> bool:
        """Wait until the run loop stops, whether by draining, abort, or failure."""
        with self._state_condition:
            if self._run_thread is None:
                return True
            if self._run_thread is threading.current_thread():
                # The run thread cannot wait for its own finally block.
                return False
            return self._state_condition.wait_for(lambda: self._run_thread is None, timeout)

    def wait_for_drain(self) -> bool:
        """Wait no longer than the latest accepted request deadline."""
        with self._state_condition:
            if self._state in {_ServerState.DRAINED, _ServerState.CLOSED}:
                return not self._accepted_requests
            if self._state is not _ServerState.DRAINING:
                return False
            if self._run_thread is None:
                # Without an I/O thread, accepted responses can never be sent.
                if self._accepted_requests:
                    return False
                self._state = _ServerState.DRAINED
                return True
            deadline_ns = self._graceful_deadline_ns

        # request_stop() guarantees that every accepted request contributing to
        # this upper bound has a finite deadline.
        timeout = None if deadline_ns is None else max(0.0, (deadline_ns - time.monotonic_ns()) / 1_000_000_000)
        if not self.wait_until_stopped(timeout):
            return False
        with self._state_condition:
            # A run loop stopped by abort is not a successful drain.
            return self._state in {_ServerState.DRAINED, _ServerState.CLOSED} and not self._accepted_requests

    def abort(self) -> None:
        """Fail accepted requests and stop without waiting for running handlers."""
        with self._state_condition:
            if self._state in {_ServerState.ABORTING, _ServerState.ABORTED, _ServerState.CLOSED}:
                return
            self._state = _ServerState.ABORTING
            requests = tuple(self._accepted_requests.values())
            for request in requests:
                self._completed_response_queue.put(
                    _ResponseEnvelope(self._encode_abort_response(request), request.key, is_abort_response=True)
                )
            run_active = self._run_thread is not None

        self._shutdown_executors(wait=False, cancel_futures=True)
        self._wake_run_loop()
        if not run_active:
            self._release_resources(_ServerState.ABORTED, linger_ms=0, shutdown_executors=False)

    def close(self) -> bool:
        """Release resources after a bounded drain, or return ``False``."""
        with self._state_condition:
            state = self._state
        if state is _ServerState.CLOSED:
            # The state becomes CLOSED before resources are released. Wait for
            # the closing thread so every successful close observes completion.
            with self._close_lock:
                pass
            return True
        if state in {_ServerState.ABORTING, _ServerState.ABORTED}:
            return False
        if not self.request_stop():
            return False
        if not self.wait_for_drain():
            logger.warning("Graceful MP server shutdown timed out; call abort()")
            return False
        self._release_resources(_ServerState.CLOSED)
        with self._state_condition:
            return self._state is _ServerState.CLOSED

    # ==============================
    # System routes and response encoding
    # ==============================

    # Built-in health routes and response builders define the behavior owned by
    # the transport, including status and remote error details. They produce
    # complete response frames but never send them; the run thread remains the
    # sole ROUTER owner.

    @staticmethod
    def _handle_ping(payloads: tuple[bytes, ...]) -> tuple[bytes, ...]:
        if payloads:
            raise ValueError("PING does not accept payloads")
        return (b"OK",)

    @staticmethod
    def _handle_echo(payloads: tuple[bytes, ...]) -> tuple[bytes, ...]:
        if len(payloads) != 1:
            raise ValueError(f"ECHO expects 1 payload, got {len(payloads)}")
        return payloads

    @staticmethod
    def _encode_error_response(identity: bytes, request_id: bytes, method: str, exc: BaseException) -> ServerResponse:
        status = ResponseStatus.BUSY if isinstance(exc, MPServerBusyError) else ResponseStatus.ERROR
        error_payloads = [f"{type(exc).__name__}: {exc}".encode()]
        if exc.__traceback__ is not None:
            remote_traceback = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
            error_payloads.append(remote_traceback.encode(errors="replace"))
        response_frames = encode_response(request_id, method, status, error_payloads)
        return identity, *response_frames

    @staticmethod
    def _encode_abort_response(request: _AcceptedRequest) -> ServerResponse:
        response_frames = encode_response(
            request.request_id,
            request.method,
            ResponseStatus.ABORTED,
            (b"MP server was force-aborted",),
        )
        return request.identity, *response_frames

    # ==============================
    # Request admission and execution
    # ==============================

    # Admission claims each (client identity, request ID) before route dispatch so
    # accepted work has one tracked completion. Executors own handler ordering, but
    # completion only queues a response; the request remains accepted until the run
    # thread sends or aborts it.

    def _receive_and_dispatch_request(self) -> None:
        frames = self._socket.recv_multipart()
        if len(frames) < 4:
            logger.warning(
                "Discarding malformed request: expected "
                "[identity, request_id, method, deadline, *payloads], got %d frames",
                len(frames),
            )
            return

        identity, *request_frames = frames
        try:
            request_id, method, deadline_ns, payloads = decode_request(request_frames)
        except Exception:
            logger.exception("Discarding malformed MP server request")
            return

        self._dispatch_request(identity, request_id, method, payloads, deadline_ns)

    def _dispatch_request(
        self,
        identity: bytes,
        request_id: bytes,
        method: str,
        payloads: tuple[bytes, ...],
        deadline_ns: int | None = None,
    ) -> None:
        request = _AcceptedRequest(identity, request_id, method, deadline_ns)
        admission = self._try_accept_request(request)
        if admission is _AdmissionResult.STOPPING:
            self._reject_stopping_request(identity, request_id, method)
            return
        if admission is _AdmissionResult.DUPLICATE:
            self._publish_response(
                self._encode_error_response(
                    identity,
                    request_id,
                    method,
                    MPServerBusyError("Duplicate in-flight MP request"),
                )
            )
            return

        route = self._routes.get(method)
        if route is None:
            self._complete_execution(
                request.key,
                self._encode_error_response(identity, request_id, method, ValueError(f"Unsupported method: {method}")),
            )
            return

        try:
            callback = partial(self._execute_handler, identity, request_id, method, payloads, route.handler)
            key = None if route.key_factory is None else route.key_factory(identity, payloads)
            future = route.executor.submit(callback, key)
        except BaseException as exc:
            self._complete_execution(request.key, self._encode_error_response(identity, request_id, method, exc))
            return

        future.add_done_callback(partial(self._on_execution_done, request))

    def _try_accept_request(self, request: _AcceptedRequest) -> _AdmissionResult:
        with self._state_condition:
            if self._state is not _ServerState.READY and self._state is not _ServerState.RUNNING:
                return _AdmissionResult.STOPPING
            if request.key in self._accepted_requests:
                return _AdmissionResult.DUPLICATE
            self._accepted_requests[request.key] = request
            return _AdmissionResult.ACCEPTED

    def _reject_stopping_request(self, identity: bytes, request_id: bytes, method: str) -> None:
        with self._state_condition:
            if self._state is _ServerState.ABORTING or self._state is _ServerState.ABORTED:
                return
        self._publish_response(
            self._encode_error_response(identity, request_id, method, MPServerBusyError("MP server is stopping"))
        )

    def _execute_handler(
        self,
        identity: bytes,
        request_id: bytes,
        method: str,
        payloads: tuple[bytes, ...],
        handler: RequestHandler,
    ) -> ServerResponse:
        try:
            response_frames = encode_response(request_id, method, ResponseStatus.OK, handler(payloads))
            return identity, *response_frames
        except BaseException as exc:
            logger.error(
                "MP RPC handler failed. method=%s identity=%r request_id=%r",
                method,
                identity,
                request_id,
                exc_info=(type(exc), exc, exc.__traceback__),
            )
            return self._encode_error_response(identity, request_id, method, exc)

    def _on_execution_done(self, request: _AcceptedRequest, future: Future[ServerResponse]) -> None:
        if future.cancelled():
            response = self._encode_error_response(
                request.identity,
                request.request_id,
                request.method,
                RuntimeError("Request execution was cancelled"),
            )
            self._complete_execution(request.key, response)
            return

        try:
            response = future.result()
        except BaseException as exc:
            logger.exception("Failed to process MP server request")
            response = self._encode_error_response(request.identity, request.request_id, request.method, exc)
        self._complete_execution(request.key, response)

    def _complete_execution(self, request_key: _RequestKey, response: ServerResponse) -> None:
        with self._state_condition:
            if request_key not in self._accepted_requests:
                return
            if self._state is _ServerState.ABORTING or self._state is _ServerState.ABORTED:
                return
            self._completed_response_queue.put(_ResponseEnvelope(response, request_key))
        self._wake_run_loop()

    # ==============================
    # Response delivery and backpressure
    # ==============================

    # Executor threads put completed responses on the queue. The run thread keeps
    # any response rejected by non-blocking send and stops accepting new requests
    # until that response is sent.

    def _wake_run_loop(self) -> None:
        with self._notify_lock:
            try:
                self._notify_writer.send(b"\x01")
            except BlockingIOError:
                # A full notification socket already guarantees a poll wakeup.
                pass
            except OSError:
                with self._state_condition:
                    stopped = self._state in {_ServerState.ABORTING, _ServerState.ABORTED, _ServerState.CLOSED}
                if not stopped:
                    logger.exception("Failed to notify MP server response loop")

    def _publish_response(self, response: ServerResponse) -> None:
        self._completed_response_queue.put(_ResponseEnvelope(response))
        self._wake_run_loop()

    def _process_run_notification(self) -> None:
        self._notify_reader.recv(4096)

        while True:
            try:
                response = self._completed_response_queue.get_nowait()
            except queue.Empty:
                return
            self._send_backlog.append(response)

    def _send_responses(self) -> None:
        # Responses belong to work the server has already accepted. NOBLOCK
        # keeps the run loop responsive; Again leaves the head queued for the
        # next POLLOUT notification instead of dropping an owed response.
        while self._send_backlog:
            response = self._send_backlog[0]
            if response.request_key is None:
                try:
                    self._socket.send_multipart(response.frames, flags=zmq.NOBLOCK)
                except zmq.Again:
                    return
                self._send_backlog.popleft()
                continue

            with self._state_condition:
                if response.request_key not in self._accepted_requests:
                    self._send_backlog.popleft()
                    continue
                # Once abort owns a request, any normal response queued earlier
                # must yield to the abort response without completing the request.
                if self._state is _ServerState.ABORTING and not response.is_abort_response:
                    self._send_backlog.popleft()
                    continue

                try:
                    self._socket.send_multipart(response.frames, flags=zmq.NOBLOCK)
                except zmq.Again:
                    return
                del self._accepted_requests[response.request_key]
                self._send_backlog.popleft()

    # ==============================
    # Final resource release
    # ==============================

    # All final paths use one shared cleanup sequence. Executor callbacks stop
    # using notification sockets before those sockets close, and the ZMQ context
    # closes last.

    def _shutdown_executors(self, wait: bool = True, cancel_futures: bool = True) -> None:
        for executor in self._executors:
            try:
                executor.shutdown(wait=wait, cancel_futures=cancel_futures)
            except Exception:
                logger.exception("Failed to shut down MP server executor")

    def _close_transport(self, linger_ms: int) -> None:
        if self._transport_closed:
            return
        self._transport_closed = True
        self._socket.close(linger=linger_ms)

    def _release_resources(
        self,
        final_state: _ServerState,
        linger_ms: int = _RESPONSE_LINGER_MS,
        shutdown_executors: bool = True,
    ) -> None:
        if final_state not in {_ServerState.ABORTED, _ServerState.CLOSED}:
            raise ValueError(f"Invalid terminal MP server state: {final_state}")
        # Only a valid DRAINED -> CLOSED or ABORTING -> ABORTED transition may
        # start the shared cleanup sequence.
        expected_state = _ServerState.ABORTING if final_state is _ServerState.ABORTED else _ServerState.DRAINED

        with self._close_lock:
            with self._state_condition:
                if self._state in {_ServerState.ABORTED, _ServerState.CLOSED}:
                    return
                if self._state is not expected_state:
                    return
                # Set the final state before slow cleanup. Concurrent close()
                # calls wait on _close_lock before reporting success.
                self._state = final_state

            self._close_transport(linger_ms)
            if shutdown_executors:
                aborted = final_state is _ServerState.ABORTED
                self._shutdown_executors(wait=not aborted, cancel_futures=aborted)
            # Notification sockets outlive executor shutdown, and the ZMQ
            # context outlives its ROUTER socket.
            self._notify_reader.close()
            self._notify_writer.close()
            self._context.term()
