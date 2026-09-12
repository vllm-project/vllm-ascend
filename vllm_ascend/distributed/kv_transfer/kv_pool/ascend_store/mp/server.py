"""Child-side transport server and shutdown state for KV transfers."""

from __future__ import annotations

import os
import queue
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from enum import Enum, auto
from functools import partial
from typing import Any, Protocol

import msgspec
from zmq import DONTWAIT, LINGER, POLLIN, ROUTER, ROUTER_MANDATORY, Context, Poller  # type: ignore[attr-defined]


class TransferServiceProtocol(Protocol):
    """Business operations consumed by the child transport server."""

    def execute(self, operation: str, payload: Any) -> Any: ...

    def submit(self, operation: str, payload: Any) -> Future: ...

    def close(self) -> None: ...


class _ServerState(Enum):
    RUNNING = auto()
    DRAINING = auto()
    DRAINED = auto()


class TransferServer:
    """Serve one parent until graceful close or parent-process death.

    The server owns the ROUTER socket, request admission and the control
    executor. The injected service owns backend work and device resources.
    """

    def __init__(
        self,
        endpoint: str,
        parent_fd: int,
        service_factory: Callable[[dict[str, Any]], TransferServiceProtocol],
    ):
        self._endpoint = endpoint
        self._parent_fd = parent_fd
        self._service_factory = service_factory
        self._context = Context()
        self._socket = self._context.socket(ROUTER)
        self._socket.setsockopt(LINGER, 0)
        self._socket.setsockopt(ROUTER_MANDATORY, 1)
        self._completions: queue.Queue = queue.Queue()
        self._control = ThreadPoolExecutor(max_workers=1, thread_name_prefix="KVTransferBackend")
        self._service: TransferServiceProtocol | None = None
        self._state = _ServerState.RUNNING
        self._closed_exit_code: int | None = None

    def run(self) -> int:
        """Run until the parent closes its lifetime pipe."""
        try:
            self._socket.bind(self._endpoint)
            poller = Poller()
            poller.register(self._socket, POLLIN)
            poller.register(self._parent_fd, POLLIN)
            while True:
                events = dict(poller.poll(10))
                if self._parent_fd in events and not os.read(self._parent_fd, 1):
                    return 1 if self._closed_exit_code is None else self._closed_exit_code
                if self._socket in events:
                    self._receive_request()
                self._send_completions()
        finally:
            # Parent death may leave device work blocked indefinitely. The
            # private entry point exits its interpreter after run() returns.
            self._control.shutdown(wait=False, cancel_futures=True)
            self._socket.close()
            self._context.term()

    def _receive_request(self) -> None:
        identity, encoded = self._socket.recv_multipart()
        try:
            operation_id, operation, payload = msgspec.msgpack.decode(encoded)
        except Exception as exc:
            raise RuntimeError("Failed to decode KV transfer command") from exc

        try:
            future = self._dispatch(operation, payload)
        except Exception as exc:
            future = Future()
            future.set_exception(exc)
        future.add_done_callback(partial(self._completed, identity, operation_id, operation))

    def _dispatch(self, operation: str, payload: Any) -> Future:
        if self._state is not _ServerState.RUNNING:
            raise RuntimeError("KV transfer process is closing")
        if operation == "init":
            if self._service is not None:
                raise RuntimeError("KV transfer process is already initialized")
            return self._control.submit(self._initialize_service, payload)
        if operation == "close":
            self._state = _ServerState.DRAINING
            return self._control.submit(self._close_service)
        if self._service is None:
            raise RuntimeError("KV transfer process is not initialized")
        if operation in ("store", "load"):
            return self._service.submit(operation, payload)
        return self._control.submit(self._service.execute, operation, payload)

    def _initialize_service(self, config: dict[str, Any]) -> None:
        self._service = self._service_factory(config)

    def _close_service(self) -> None:
        if self._service is not None:
            self._service.close()

    def _completed(self, identity: bytes, operation_id: int, operation: str, future: Future) -> None:
        exit_code = 0
        try:
            encoded = msgspec.msgpack.encode((operation_id, future.result(), None))
        except BaseException as exc:
            exit_code = 1
            encoded = msgspec.msgpack.encode((operation_id, None, f"{type(exc).__name__}: {exc}"))
        self._completions.put((identity, encoded, operation == "close", exit_code))

    def _send_completions(self) -> None:
        while not self._completions.empty():
            identity, encoded, is_close, exit_code = self._completions.get_nowait()
            self._socket.send_multipart((identity, encoded), DONTWAIT)
            if is_close:
                # send() only queues a reply. Keep the socket alive until the
                # parent consumes it and closes the lifetime pipe.
                self._state = _ServerState.DRAINED
                self._closed_exit_code = exit_code


__all__ = ["TransferServer"]
