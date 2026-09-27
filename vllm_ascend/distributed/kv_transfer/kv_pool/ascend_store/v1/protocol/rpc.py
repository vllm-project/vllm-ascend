"""Lookup RPC transport between Scheduler and Worker processes."""

from __future__ import annotations

import threading
from collections.abc import Callable

from vllm.utils.network_utils import make_zmq_socket
from zmq.constants import SocketType
from zmq.sugar.context import Context

from .lookup import LookupCodec, LookupRequest, LookupResult


class LookupClient:
    def __init__(self, address: str) -> None:
        self._codec = LookupCodec()
        self._context = Context()
        self._socket = make_zmq_socket(self._context, address, SocketType.REQ, bind=False)

    def lookup(self, request: LookupRequest) -> LookupResult:
        self._socket.send_multipart(self._codec.encode_request(request), copy=False)
        return self._codec.decode_result(self._socket.recv())

    def close(self) -> None:
        self._socket.close(linger=0)
        self._context.term()


class LookupServer:
    """Forward decoded Lookup requests to the KV Pool graph that owns the Backend."""

    def __init__(self, lookup: Callable[[LookupRequest], LookupResult], address: str) -> None:
        self._codec = LookupCodec()
        self._context = Context()
        self._socket = make_zmq_socket(self._context, address, SocketType.REP, bind=True)
        self._lookup = lookup
        self._running = True
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    def _serve(self) -> None:
        while self._running:
            request = self._codec.decode_request(self._socket.recv_multipart(copy=False))
            self._socket.send(self._codec.encode_result(self._lookup(request)))

    def close(self) -> None:
        self._running = False
        self._socket.close(linger=0)
