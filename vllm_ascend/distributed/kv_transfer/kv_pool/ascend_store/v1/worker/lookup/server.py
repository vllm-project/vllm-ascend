"""Worker-side Lookup RPC server."""

from __future__ import annotations

import threading
from collections.abc import Callable

import zmq
from vllm.utils.network_utils import make_zmq_socket

from ...protocol.lookup import LookupCodec, LookupRequest, LookupResult


class LookupServer:
    """Forward decoded Lookup requests to the Worker that owns the Backend."""

    def __init__(self, lookup: Callable[[LookupRequest], LookupResult], address: str) -> None:
        self._codec = LookupCodec()
        self._context = zmq.Context()
        self._socket = make_zmq_socket(self._context, address, zmq.REP, bind=True)
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
