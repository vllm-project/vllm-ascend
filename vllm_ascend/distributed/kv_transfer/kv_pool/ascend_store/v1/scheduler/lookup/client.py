"""Scheduler-owned Lookup RPC client."""

from __future__ import annotations

import zmq
from vllm.utils.network_utils import make_zmq_socket
from vllm.v1.serial_utils import MsgpackEncoder


class LookupClient:
    def __init__(self, address: str) -> None:
        self.encoder = MsgpackEncoder()
        self.ctx = zmq.Context()
        self.socket = make_zmq_socket(self.ctx, address, zmq.REQ, bind=False)

    def lookup(
        self,
        lookup_end_token: int,
        transfer_group_ids: tuple[int, ...],
        block_hashes: list[bytes],
        local_cached_tokens: int,
    ) -> int:
        hash_frames = self.encoder.encode([block_hash.hex() for block_hash in block_hashes])
        group_frames = self.encoder.encode(list(transfer_group_ids))
        self.socket.send_multipart(
            [
                lookup_end_token.to_bytes(4, byteorder="big"),
                *group_frames,
                local_cached_tokens.to_bytes(4, byteorder="big"),
                *hash_frames,
            ],
            copy=False,
        )
        return int.from_bytes(self.socket.recv(), "big")

    def close(self) -> None:
        self.socket.close(linger=0)
        self.ctx.term()
