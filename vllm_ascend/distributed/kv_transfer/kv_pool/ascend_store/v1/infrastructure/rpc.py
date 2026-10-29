"""Synchronous byte-frame RPC and its connection/thread lifecycle."""

from __future__ import annotations

import threading
from collections.abc import Callable, Sequence
from math import ceil
from time import monotonic
from typing import TypeAlias

from vllm.logger import logger
from vllm.utils.network_utils import make_zmq_socket
from zmq.constants import DONTWAIT, POLLIN, POLLOUT, RCVTIMEO, SNDTIMEO, SocketType
from zmq.error import Again
from zmq.sugar.context import Context
from zmq.sugar.socket import Socket

WireFrame: TypeAlias = bytes | bytearray | memoryview
MAX_TIMEOUT_MS = (1 << 31) - 1
RPC_CLOSE_TIMEOUT_S = 5.0
_MILLISECONDS_PER_SECOND = 1000
_SERVER_STOP_POLL_INTERVAL_MS = 100
_SERVER_REPLY_LINGER_MS = 100


class RpcClient:
    """Own a lazy REQ connection on one caller thread, with one exchange at a time."""

    def __init__(self, address: str) -> None:
        self._address = address
        self._context: Context | None = None
        self._socket: Socket | None = None
        self._closed = False

    def request(self, frames: Sequence[WireFrame], *, deadline: float | None) -> bytes:
        """Share one monotonic deadline across send/receive; failures do not retry the request."""
        if self._closed:
            raise RuntimeError("RPC client is closed")
        try:
            if self._context is None:
                self._context = Context()
            if self._socket is None:
                self._socket = make_zmq_socket(self._context, self._address, SocketType.REQ, bind=False)
            # Sending and receiving share one budget; receiving must not restart it.
            self._socket.setsockopt(SNDTIMEO, _remaining_timeout_ms(deadline))
            self._socket.send_multipart(frames, copy=False)
            self._socket.setsockopt(RCVTIMEO, _remaining_timeout_ms(deadline))
            return self._socket.recv()
        except Exception:
            # A timed-out REQ is still waiting for its reply. A new socket also
            # prevents that reply from being mistaken for a later request's result.
            self.reset_connection()
            raise

    def close(self) -> None:
        self._closed = True
        self.reset_connection()

    def reset_connection(self) -> None:
        """Discard an unusable connection without closing the client permanently."""
        if self._socket is not None:
            self._socket.close(linger=0)
            self._socket = None
        if self._context is not None:
            # make_zmq_socket may fail before returning its created socket. All
            # sockets in this Context belong to the client's calling thread.
            self._context.destroy(linger=0)
            self._context = None


class RpcServer:
    """Run a byte-frame handler on the socket-owning thread and join before resource release."""

    def __init__(self, handler: Callable[[Sequence[WireFrame]], bytes], address: str) -> None:
        self._address = address
        self._handler = handler
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._failure: BaseException | None = None
        self._thread = threading.Thread(target=self._serve, name="RpcServer", daemon=True)
        self._thread.start()
        self._ready.wait()
        if self._failure is not None:
            self._thread.join()
            raise RuntimeError(f"Failed to start RPC server at {address}") from self._failure

    def close(self, *, timeout: float = RPC_CLOSE_TIMEOUT_S) -> None:
        """Confirm handler exit, not reply delivery; a timeout leaves its resources owned."""

        self._stop.set()
        self._thread.join(timeout=timeout)
        if self._thread.is_alive():
            raise RuntimeError("RPC server is still running; handler resources must remain open")

    def _serve(self) -> None:
        try:
            with Context() as context:
                socket = make_zmq_socket(context, self._address, SocketType.REP, bind=True)
                try:
                    self._ready.set()
                    while not self._stop.is_set():
                        if not socket.poll(_SERVER_STOP_POLL_INTERVAL_MS, POLLIN):
                            continue
                        frames = socket.recv_multipart(copy=False, flags=DONTWAIT)
                        # Expose buffers, not ZeroMQ Frame objects, to the application.
                        response = self._handler(tuple(frame.buffer for frame in frames))
                        # Complete an accepted query before stop; never close the
                        # socket from another thread to interrupt its handler.
                        while True:
                            try:
                                socket.send(response, flags=DONTWAIT)
                            except Again:
                                if self._stop.is_set():
                                    logger.warning("RPC reply could not drain during shutdown at %s", self._address)
                                    return
                                socket.poll(_SERVER_STOP_POLL_INTERVAL_MS, POLLOUT)
                                continue
                            break
                finally:
                    socket.close(linger=_SERVER_REPLY_LINGER_MS)
        except BaseException as error:
            self._failure = error
            logger.exception("RPC server stopped at %s", self._address)
        finally:
            self._ready.set()


def deadline_after(timeout_ms: int) -> float | None:
    """Start the exchange budget before application encoding; -1 means no deadline."""
    return None if timeout_ms == -1 else monotonic() + timeout_ms / _MILLISECONDS_PER_SECOND


def _remaining_timeout_ms(deadline: float | None) -> int:
    if deadline is None:
        return -1
    return min(MAX_TIMEOUT_MS, max(0, ceil((deadline - monotonic()) * _MILLISECONDS_PER_SECOND)))
