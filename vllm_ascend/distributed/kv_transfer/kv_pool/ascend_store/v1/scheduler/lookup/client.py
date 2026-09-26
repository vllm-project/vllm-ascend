"""Scheduler-owned Lookup RPC client."""

from __future__ import annotations

import zmq
from vllm.utils.network_utils import make_zmq_socket

from ...protocol.lookup import LookupCodec, LookupRequest, LookupResult


class LookupClient:
    def __init__(self, address: str) -> None:
        self._codec = LookupCodec()
        self._context = zmq.Context()
        self._socket = make_zmq_socket(self._context, address, zmq.REQ, bind=False)

    def lookup(self, request: LookupRequest) -> LookupResult:
        self._socket.send_multipart(self._codec.encode_request(request), copy=False)
        return self._codec.decode_result(self._socket.recv())

    def close(self) -> None:
        self._socket.close(linger=0)
        self._context.term()
