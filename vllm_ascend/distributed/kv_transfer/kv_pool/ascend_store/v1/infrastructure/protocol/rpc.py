"""Lookup RPC transport between Scheduler and Worker processes."""

from __future__ import annotations

import threading
from collections.abc import Callable
from math import ceil
from time import monotonic

from vllm.logger import logger
from vllm.utils.network_utils import make_zmq_socket
from zmq.constants import DONTWAIT, POLLIN, POLLOUT, RCVTIMEO, SNDTIMEO, SocketType
from zmq.error import Again
from zmq.sugar.context import Context
from zmq.sugar.socket import Socket

from .lookup import LookupCodec
from .messages import LookupRequest, LookupResult

_MILLISECONDS_PER_SECOND = 1000
_MAX_TIMEOUT_MS = (1 << 31) - 1
_SERVER_STOP_POLL_INTERVAL_MS = 100
_SERVER_REPLY_LINGER_MS = 100
_SERVER_CLOSE_TIMEOUT_S = 5.0


class LookupClient:
    """Own a lazy REQ connection and discard it after an unsuccessful exchange."""

    def __init__(self, address: str, *, timeout_ms: int = -1) -> None:
        if type(timeout_ms) is not int or not -1 <= timeout_ms <= _MAX_TIMEOUT_MS:
            raise ValueError(f"lookup_rpc_timeout_ms must be an integer between -1 and {_MAX_TIMEOUT_MS}")
        self._address = address
        self._timeout_ms = timeout_ms
        self._codec = LookupCodec()
        self._context: Context | None = None
        self._socket: Socket | None = None
        self._closed = False

    def lookup(self, request: LookupRequest) -> LookupResult:
        if self._closed:
            raise RuntimeError("Lookup client is closed")
        deadline = None if self._timeout_ms == -1 else monotonic() + self._timeout_ms / _MILLISECONDS_PER_SECOND
        frames = self._codec.encode_request(request)
        try:
            if self._context is None:
                self._context = Context()
            if self._socket is None:
                self._socket = make_zmq_socket(self._context, self._address, SocketType.REQ, bind=False)
            # Sending and receiving share one budget; receiving must not restart it.
            self._socket.setsockopt(SNDTIMEO, _remaining_timeout_ms(deadline))
            self._socket.send_multipart(frames, copy=False)
            self._socket.setsockopt(RCVTIMEO, _remaining_timeout_ms(deadline))
            result = self._codec.decode_result(self._socket.recv())
            if result.available_end_token > request.end_token:
                raise ValueError("Lookup response exceeds the requested token range")
            return result
        except Exception:
            logger.exception("Lookup RPC failed at %s; treating this query as an external cache miss", self._address)
            # A timed-out REQ is still waiting for its reply. A new socket also
            # prevents that reply from being mistaken for a later request's result.
            self._close_connection()
            return LookupResult(0)

    def close(self) -> None:
        self._closed = True
        self._close_connection()

    def _close_connection(self) -> None:
        if self._socket is not None:
            self._socket.close(linger=0)
            self._socket = None
        if self._context is not None:
            # make_zmq_socket may fail before returning its created socket. All
            # sockets in this Context belong to the Scheduler's calling thread.
            self._context.destroy(linger=0)
            self._context = None


class LookupServer:
    """Serve Lookup on its socket-owning thread and join before Worker release."""

    def __init__(self, lookup: Callable[[LookupRequest], LookupResult], address: str) -> None:
        self._codec = LookupCodec()
        self._address = address
        self._lookup = lookup
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._failure: BaseException | None = None
        self._thread = threading.Thread(target=self._serve, name="KVPoolLookupServer", daemon=True)
        self._thread.start()
        self._ready.wait()
        if self._failure is not None:
            self._thread.join()
            raise RuntimeError(f"Failed to start Lookup server at {address}") from self._failure

    def close(self, *, timeout: float = _SERVER_CLOSE_TIMEOUT_S) -> None:
        """Request stop, retaining Worker ownership if an active query cannot exit."""

        self._stop.set()
        self._thread.join(timeout=timeout)
        if self._thread.is_alive():
            raise RuntimeError("Lookup server is still running; Worker resources must remain open")

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
                        try:
                            request = self._codec.decode_request(frames)
                            response = self._codec.encode_result(self._lookup(request))
                        except Exception:
                            logger.exception(
                                "Lookup request failed at %s; returning an external cache miss", self._address
                            )
                            response = self._codec.encode_result(LookupResult(0))
                        # Complete an accepted query before stop; never close the
                        # socket from another thread to interrupt its Backend call.
                        while True:
                            try:
                                socket.send(response, flags=DONTWAIT)
                            except Again:
                                if self._stop.is_set():
                                    logger.warning("Lookup reply could not drain during shutdown at %s", self._address)
                                    return
                                socket.poll(_SERVER_STOP_POLL_INTERVAL_MS, POLLOUT)
                                continue
                            break
                finally:
                    socket.close(linger=_SERVER_REPLY_LINGER_MS)
        except BaseException as error:
            self._failure = error
            logger.exception("Lookup RPC server stopped at %s", self._address)
        finally:
            self._ready.set()


def _remaining_timeout_ms(deadline: float | None) -> int:
    if deadline is None:
        return -1
    return min(_MAX_TIMEOUT_MS, max(0, ceil((deadline - monotonic()) * _MILLISECONDS_PER_SECOND)))
