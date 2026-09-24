"""Scheduler-owned Lookup RPC client for the classic path."""

from __future__ import annotations

import zmq
from vllm.utils.network_utils import make_zmq_socket
from vllm.v1.serial_utils import MsgpackEncoder


class LookupKeyClient:
    def __init__(self, address: str) -> None:
        self.encoder = MsgpackEncoder()
        self.ctx = zmq.Context()
        self.socket = make_zmq_socket(self.ctx, address, zmq.REQ, bind=False)

    def lookup(self, token_len: int, block_hashes: list[bytes], hbm_hit_tokens: int) -> int:
        hash_frames = self.encoder.encode([block_hash.hex() for block_hash in block_hashes])
        group_frames = self.encoder.encode([0])
        self.socket.send_multipart(
            [
                token_len.to_bytes(4, byteorder="big"),
                *group_frames,
                hbm_hit_tokens.to_bytes(4, byteorder="big"),
                *hash_frames,
            ],
            copy=False,
        )
        return int.from_bytes(self.socket.recv(), "big")

    def close(self) -> None:
        self.socket.close(linger=0)
        self.ctx.term()
