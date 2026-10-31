"""Adapt Lookup messages, validation, and cache-miss fallback to byte-frame RPC."""

from __future__ import annotations

from collections.abc import Callable, Sequence

from vllm.logger import logger
from vllm.v1.core.kv_cache_utils import BlockHash
from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder

from ..infrastructure.rpc import MAX_TIMEOUT_MS, RPC_CLOSE_TIMEOUT_S, RpcClient, RpcServer, WireFrame, deadline_after
from .messages import LookupRequest, LookupResult, TailKeyBoundary

_TOKEN_COUNT_BYTES = 4
_REQUEST_FRAME_COUNT = 4
_TAIL_KEY_BOUNDARY_BYTES = 8


class LookupClient:
    """Interpret unsuccessful remote queries as misses, reconnecting on the next Lookup."""

    def __init__(self, address: str, *, timeout_ms: int = -1) -> None:
        if type(timeout_ms) is not int or not -1 <= timeout_ms <= MAX_TIMEOUT_MS:
            raise ValueError(f"lookup_rpc_timeout_ms must be an integer between -1 and {MAX_TIMEOUT_MS}")
        self._address = address
        self._timeout_ms = timeout_ms
        self._codec = LookupCodec()
        self._rpc = RpcClient(address)
        self._closed = False

    def lookup(self, request: LookupRequest) -> LookupResult:
        if self._closed:
            raise RuntimeError("Lookup client is closed")
        deadline = deadline_after(self._timeout_ms)
        frames = self._codec.encode_request(request)
        try:
            result = self._codec.decode_result(self._rpc.request(frames, deadline=deadline))
            if result.available_end_token > request.end_token:
                raise ValueError("Lookup response exceeds the requested token range")
            return result
        except Exception:
            logger.exception("Lookup RPC failed at %s; treating this query as an external cache miss", self._address)
            # Invalid replies also invalidate the connection, even after a completed exchange.
            self._rpc.reset_connection()
            return LookupResult(0)

    def close(self) -> None:
        self._closed = True
        self._rpc.close()

    def reset(self) -> bool:
        if self._closed:
            raise RuntimeError("Lookup client is closed")
        try:
            return self._rpc.request([b"reset"], deadline=deadline_after(self._timeout_ms)) == b"1"
        except Exception:
            self._rpc.reset_connection()
            logger.exception("External cache reset failed at %s", self._address)
            return False


class LookupServer:
    """Translate Lookup failures into miss replies while retaining the Worker callback until close."""

    def __init__(
        self, lookup: Callable[[LookupRequest], LookupResult], address: str, *, reset: Callable[[], bool] | None = None
    ) -> None:
        self._codec = LookupCodec()
        self._address = address
        self._lookup = lookup
        self._reset = reset
        self._rpc = RpcServer(self._handle_request, address)

    def close(self, *, timeout: float = RPC_CLOSE_TIMEOUT_S) -> None:
        """The owning Connector may release Worker resources only after this succeeds."""
        self._rpc.close(timeout=timeout)

    def _handle_request(self, frames: Sequence[WireFrame]) -> bytes:
        if len(frames) == 1 and bytes(frames[0]) == b"reset":
            try:
                return b"1" if self._reset is not None and self._reset() else b"0"
            except Exception:
                logger.exception("External cache reset failed at %s", self._address)
                return b"0"
        try:
            request = self._codec.decode_request(frames)
            return self._codec.encode_result(self._lookup(request))
        except Exception:
            logger.exception("Lookup request failed at %s; returning an external cache miss", self._address)
            return self._codec.encode_result(LookupResult(0))


class LookupCodec:
    """Keep Lookup wire frames out of the Scheduler planner and Worker internals."""

    def __init__(self) -> None:
        self._encoder = MsgpackEncoder()
        self._decoder = MsgpackDecoder()

    def encode_request(self, request: LookupRequest) -> list[WireFrame]:
        group_frames = self._encoder.encode(list(request.transfer_group_ids))
        hash_frames = self._encoder.encode([block_hash.hex() for block_hash in request.block_hashes])
        return [
            request.end_token.to_bytes(_TOKEN_COUNT_BYTES, byteorder="big"),
            *group_frames,
            request.local_available_tokens.to_bytes(_TOKEN_COUNT_BYTES, byteorder="big"),
            *hash_frames,
        ]

    def decode_request(self, frames: Sequence[WireFrame]) -> LookupRequest:
        if len(frames) != _REQUEST_FRAME_COUNT:
            raise ValueError(f"Lookup request requires {_REQUEST_FRAME_COUNT} frames, received {len(frames)}")
        local_available_tokens = int.from_bytes(frames[2], byteorder="big")
        end_token = int.from_bytes(frames[0], byteorder="big")
        if end_token < local_available_tokens:
            raise ValueError(f"Lookup range end {end_token} precedes local prefix {local_available_tokens}")
        hash_strings = self._decoder.decode(frames[3:])
        return LookupRequest(
            local_available_tokens=local_available_tokens,
            end_token=end_token,
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
