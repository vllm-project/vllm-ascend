"""Messages and wire encoding for Scheduler-to-Worker Lookup."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TypeAlias

from vllm.v1.core.kv_cache_utils import BlockHash
from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder

from ..coordinates import TokenRange

WireFrame: TypeAlias = bytes | bytearray | memoryview
_TOKEN_COUNT_BYTES = 4
_REQUEST_FRAME_COUNT = 4
_TAIL_KEY_BOUNDARY_BYTES = 8


@dataclass(frozen=True, slots=True)
class LookupRequest:
    """Content hashes and cache groups participating in one remote Lookup."""

    query_range: TokenRange
    transfer_group_ids: tuple[int, ...]
    block_hashes: tuple[BlockHash, ...]


@dataclass(frozen=True, slots=True)
class TailKeyBoundary:
    """Hash boundary that identifies one cache group's remote tail object."""

    group_id: int
    boundary_token: int


@dataclass(frozen=True, slots=True)
class LookupResult:
    """Contiguous token prefix available from the KV pool."""

    available_end_token: int
    tail_key_boundaries: tuple[TailKeyBoundary, ...] = ()


class LookupCodec:
    """Keep Lookup wire frames out of the Scheduler planner and Worker internals."""

    def __init__(self) -> None:
        self._encoder = MsgpackEncoder()
        self._decoder = MsgpackDecoder()

    def encode_request(self, request: LookupRequest) -> list[WireFrame]:
        group_frames = self._encoder.encode(list(request.transfer_group_ids))
        hash_frames = self._encoder.encode([block_hash.hex() for block_hash in request.block_hashes])
        return [
            request.query_range.end_token.to_bytes(_TOKEN_COUNT_BYTES, byteorder="big"),
            *group_frames,
            request.query_range.start_token.to_bytes(_TOKEN_COUNT_BYTES, byteorder="big"),
            *hash_frames,
        ]

    def decode_request(self, frames: Sequence[WireFrame]) -> LookupRequest:
        if len(frames) != _REQUEST_FRAME_COUNT:
            raise ValueError(f"Lookup request requires {_REQUEST_FRAME_COUNT} frames, received {len(frames)}")
        hash_strings = self._decoder.decode(frames[3:])
        return LookupRequest(
            query_range=TokenRange(
                int.from_bytes(frames[2], byteorder="big"),
                int.from_bytes(frames[0], byteorder="big"),
            ),
            transfer_group_ids=tuple(self._decoder.decode([frames[1]])),
            block_hashes=tuple(BlockHash(bytes.fromhex(block_hash)) for block_hash in hash_strings),
        )

    @staticmethod
    def encode_result(result: LookupResult) -> bytes:
        payload = bytearray(result.available_end_token.to_bytes(_TOKEN_COUNT_BYTES, byteorder="big"))
        for boundary in result.tail_key_boundaries:
            payload.extend(boundary.group_id.to_bytes(_TOKEN_COUNT_BYTES, byteorder="big"))
            payload.extend(boundary.boundary_token.to_bytes(_TOKEN_COUNT_BYTES, byteorder="big"))
        return bytes(payload)

    @staticmethod
    def decode_result(frame: WireFrame) -> LookupResult:
        payload = bytes(frame)
        if len(payload) < _TOKEN_COUNT_BYTES or (len(payload) - _TOKEN_COUNT_BYTES) % _TAIL_KEY_BOUNDARY_BYTES:
            raise ValueError("Invalid Lookup result payload")
        boundaries = tuple(
            TailKeyBoundary(
                int.from_bytes(payload[offset : offset + _TOKEN_COUNT_BYTES], byteorder="big"),
                int.from_bytes(
                    payload[offset + _TOKEN_COUNT_BYTES : offset + _TAIL_KEY_BOUNDARY_BYTES], byteorder="big"
                ),
            )
            for offset in range(_TOKEN_COUNT_BYTES, len(payload), _TAIL_KEY_BOUNDARY_BYTES)
        )
        return LookupResult(int.from_bytes(payload[:_TOKEN_COUNT_BYTES], byteorder="big"), boundaries)
