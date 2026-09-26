"""Worker-side Lookup RPC server."""

from __future__ import annotations

import threading
from collections.abc import Callable

import zmq
from vllm.utils.network_utils import make_zmq_socket
from vllm.v1.serial_utils import MsgpackDecoder

from .request import WorkerLookupRequest


class LookupServer:
    """Forward decoded Lookup requests to the Worker that owns the Backend."""

    def __init__(self, lookup: Callable[[WorkerLookupRequest], int], address: str) -> None:
        self.decoder = MsgpackDecoder()
        self.ctx = zmq.Context()
        self.socket = make_zmq_socket(self.ctx, address, zmq.REP, bind=True)
        self.lookup = lookup
        self.running = True
        self.thread = threading.Thread(target=self._serve, daemon=True)
        self.thread.start()

    def _serve(self) -> None:
        while self.running:
            frames = self.socket.recv_multipart(copy=False)
            lookup_end_token = int.from_bytes(frames[0], byteorder="big")
            transfer_group_ids = tuple(self.decoder.decode([frames[1]]))
            local_cached_tokens = int.from_bytes(frames[2], byteorder="big")
            hash_strings = self.decoder.decode(frames[3:])
            kv_pool_cached_tokens = self.lookup(
                WorkerLookupRequest(lookup_end_token, transfer_group_ids, local_cached_tokens, tuple(hash_strings))
            )
            self.socket.send(kv_pool_cached_tokens.to_bytes(4, "big"))

    def close(self) -> None:
        self.running = False
        self.socket.close(linger=0)
