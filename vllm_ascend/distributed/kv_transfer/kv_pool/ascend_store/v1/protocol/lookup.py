"""Messages and wire encoding for Scheduler-to-Worker Lookup."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import zmq
from vllm.v1.core.kv_cache_utils import BlockHash
from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder

WireFrame = bytes | bytearray | memoryview | zmq.Frame
_TOKEN_COUNT_BYTES = 4
_REQUEST_FRAME_COUNT = 4


@dataclass(frozen=True, slots=True)
class LookupRequest:
    """Content hashes and cache groups participating in one remote Lookup."""

    lookup_end_token: int
    transfer_group_ids: tuple[int, ...]
    local_cached_tokens: int
    block_hashes: tuple[BlockHash, ...]


@dataclass(frozen=True, slots=True)
class LookupResult:
    """Contiguous token prefix available from the KV pool."""

    kv_pool_cached_tokens: int


class LookupCodec:
    """Keep Lookup wire frames out of Scheduler and Worker business services."""

    def __init__(self) -> None:
        self._encoder = MsgpackEncoder()
        self._decoder = MsgpackDecoder()

    def encode_request(self, request: LookupRequest) -> list[WireFrame]:
        group_frames = self._encoder.encode(list(request.transfer_group_ids))
        hash_frames = self._encoder.encode([block_hash.hex() for block_hash in request.block_hashes])
        return [
            request.lookup_end_token.to_bytes(_TOKEN_COUNT_BYTES, byteorder="big"),
            *group_frames,
            request.local_cached_tokens.to_bytes(_TOKEN_COUNT_BYTES, byteorder="big"),
            *hash_frames,
        ]

    def decode_request(self, frames: Sequence[WireFrame]) -> LookupRequest:
        if len(frames) != _REQUEST_FRAME_COUNT:
            raise ValueError(f"Lookup request requires {_REQUEST_FRAME_COUNT} frames, received {len(frames)}")
        hash_strings = self._decoder.decode(frames[3:])
        return LookupRequest(
            lookup_end_token=int.from_bytes(frames[0], byteorder="big"),
            transfer_group_ids=tuple(self._decoder.decode([frames[1]])),
            local_cached_tokens=int.from_bytes(frames[2], byteorder="big"),
            block_hashes=tuple(BlockHash(bytes.fromhex(block_hash)) for block_hash in hash_strings),
        )

    @staticmethod
    def encode_result(result: LookupResult) -> bytes:
        return result.kv_pool_cached_tokens.to_bytes(_TOKEN_COUNT_BYTES, byteorder="big")

    @staticmethod
    def decode_result(frame: WireFrame) -> LookupResult:
        return LookupResult(int.from_bytes(frame, byteorder="big"))
