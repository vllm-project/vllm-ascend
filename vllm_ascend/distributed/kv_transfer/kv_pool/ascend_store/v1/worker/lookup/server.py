"""Worker-side Lookup RPC server for the classic path."""

from __future__ import annotations

import threading
from collections.abc import Callable

import zmq
from vllm.utils.network_utils import make_zmq_socket
from vllm.v1.serial_utils import MsgpackDecoder


class LookupKeyServer:
    """Forward decoded Lookup requests to the Worker that owns the Backend."""

    def __init__(self, lookup: Callable[[int, list[str]], int], address: str) -> None:
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
            token_len = int.from_bytes(frames[0], byteorder="big")
            group_ids = self.decoder.decode([frames[1]])
            # Retain the classic wire frame, but HBM hits are not used without a coordinator.
            _hbm_hit_tokens = int.from_bytes(frames[2], byteorder="big")
            hash_strings = self.decoder.decode(frames[3:])
            if group_ids != [0]:
                raise ValueError("AscendStore v1 classic Lookup requires one KV group")
            result = self.lookup(token_len, hash_strings)
            self.socket.send(result.to_bytes(4, "big"))

    def close(self) -> None:
        self.running = False
        self.socket.close(linger=0)
