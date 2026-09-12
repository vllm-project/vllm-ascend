"""Parent-side request channel for one local KV transfer process."""

from __future__ import annotations

import queue
import threading
import time
from collections.abc import Callable
from concurrent.futures import Future
from dataclasses import dataclass
from typing import Any

import msgspec
from zmq import DEALER, DONTWAIT, IMMEDIATE, LINGER, SNDHWM, Again, Context  # type: ignore[attr-defined]

POLL_INTERVAL_MS = 10
MAX_PENDING_TRANSFERS = 256
TRANSFER_TIMEOUT_SECONDS = 120.0


@dataclass
class _Command:
    operation_id: int
    operation: str
    payload: bytes
    future: Future
    deadline: float


class TransferClient:
    """Map concurrent calls onto a socket owned by one I/O thread.

    The client owns request IDs, deadlines, pending futures and its DEALER
    socket. Process creation, parent-death signalling and child reaping belong
    to ``TransferProcess``.
    """

    def __init__(
        self,
        endpoint: str,
        child_status: Callable[[], int | None],
        timeout: float = TRANSFER_TIMEOUT_SECONDS,
    ):
        if timeout <= 0:
            raise ValueError("Transfer timeout must be positive")
        self.timeout = timeout
        self._child_status = child_status
        self._commands: queue.Queue[_Command] = queue.Queue(MAX_PENDING_TRANSFERS)
        self._lock = threading.Lock()
        self._close_lock = threading.Lock()
        self._closed = False
        self._error: BaseException | None = None
        self._next_id = 0
        self._stop = threading.Event()
        self._ready: Future = Future()
        self._io = threading.Thread(target=self._run, args=(endpoint,), name="KVTransferControl", daemon=True)
        self._io.start()
        try:
            self._ready.result(timeout)
        except BaseException as exc:
            self._stop.set()
            self._io.join()
            raise RuntimeError("Failed to start KV transfer client") from exc

    def submit(self, operation: str, payload: Any = None) -> Future:
        with self._lock:
            self.raise_if_failed()
            if self._closed:
                raise RuntimeError("KV transfer client is closed")
            return self._enqueue(operation, payload)

    def call(self, operation: str, payload: Any = None) -> Any:
        return self.wait(self.submit(operation, payload))

    def wait(self, future: Future) -> Any:
        try:
            return future.result(self.timeout)
        except TimeoutError as exc:
            self._fail(exc)
            raise RuntimeError("KV transfer operation timed out; the process must be closed") from exc

    def raise_if_failed(self) -> None:
        if self._error is not None:
            raise RuntimeError("KV transfer process failed") from self._error

    def close(self) -> None:
        """Drain accepted calls through the child close command, then stop I/O."""
        with self._close_lock:
            with self._lock:
                closed = None
                if not self._closed and self._error is None:
                    try:
                        closed = self._enqueue("close")
                    except RuntimeError as exc:
                        self._fail(exc)
                self._closed = True
            try:
                if closed is not None:
                    self.wait(closed)
                elif self._error is not None:
                    self.raise_if_failed()
            finally:
                self._stop.set()
                self._io.join()

    def _enqueue(self, operation: str, payload: Any = None) -> Future:
        self._next_id += 1
        future: Future = Future()
        command = _Command(
            self._next_id,
            operation,
            msgspec.msgpack.encode((self._next_id, operation, payload)),
            future,
            time.monotonic() + self.timeout,
        )
        try:
            self._commands.put_nowait(command)
        except queue.Full as exc:
            raise RuntimeError("KV transfer command queue is full") from exc
        return future

    def _fail(self, error: BaseException) -> None:
        if self._error is None:
            self._error = error
        self._stop.set()

    def _run(self, endpoint: str) -> None:
        context = None
        socket = None
        # Commands move queue -> outgoing -> pending. Keeping one outgoing
        # command preserves FIFO when a non-blocking send must be retried.
        pending: dict[int, _Command] = {}
        outgoing: _Command | None = None
        try:
            context = Context()
            socket = context.socket(DEALER)
            socket.setsockopt(LINGER, 0)
            socket.setsockopt(SNDHWM, MAX_PENDING_TRANSFERS)
            socket.setsockopt(IMMEDIATE, 1)
            socket.connect(endpoint)
            self._ready.set_result(None)
            while not self._stop.is_set():
                while len(pending) < MAX_PENDING_TRANSFERS:
                    if outgoing is None:
                        try:
                            outgoing = self._commands.get_nowait()
                        except queue.Empty:
                            break
                    if time.monotonic() >= outgoing.deadline:
                        raise TimeoutError("Timed out submitting a KV transfer command")
                    try:
                        socket.send(outgoing.payload, DONTWAIT)
                    except Again:
                        break
                    else:
                        pending[outgoing.operation_id] = outgoing
                        outgoing = None
                readable = socket.poll(POLL_INTERVAL_MS)
                while readable:
                    operation_id, result, error = msgspec.msgpack.decode(socket.recv())
                    completed = pending.pop(operation_id, None)
                    if completed is None:
                        raise RuntimeError(f"Received completion for unknown KV transfer operation {operation_id}")
                    if completed.operation == "close":
                        self._stop.set()
                    if error is not None:
                        completed.future.set_exception(RuntimeError(error))
                    else:
                        completed.future.set_result(result)
                    readable = socket.poll(0)
                child_status = self._child_status()
                if not self._stop.is_set() and child_status is not None:
                    # Read queued replies before interpreting process exit; the
                    # final close reply may already be in the socket.
                    if not socket.poll(0):
                        raise RuntimeError(f"KV transfer subprocess exited with code {child_status}")
                if any(time.monotonic() >= item.deadline for item in pending.values()):
                    raise TimeoutError("Timed out waiting for KV transfer completion")
        except BaseException as exc:
            self._fail(exc)
            if not self._ready.done():
                self._ready.set_exception(exc)
        finally:
            # Freeze admission before failing commands from every accepted state.
            with self._lock:
                self._closed = True
                if outgoing is not None:
                    pending[outgoing.operation_id] = outgoing
                while not self._commands.empty():
                    item = self._commands.get_nowait()
                    pending[item.operation_id] = item
            for item in pending.values():
                if not item.future.done():
                    item.future.set_exception(RuntimeError(f"KV transfer channel stopped: {self._error}"))
            if socket is not None:
                socket.close()
            if context is not None:
                context.term()


__all__ = ["TRANSFER_TIMEOUT_SECONDS", "TransferClient"]
