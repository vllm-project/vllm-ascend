"""Tests for infrastructure.protocol.lookup."""

from __future__ import annotations

import pytest

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.infrastructure.protocol.lookup import LookupCodec
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.infrastructure.protocol.messages import (
    LookupRequest,
    LookupResult,
    TailKeyBoundary,
)


def test_lookup_codec_roundtrip_preserves_groups_and_partial_tail_boundaries() -> None:
    codec = LookupCodec()
    request = LookupRequest(4, 12, (1, 3), (b"a", b"b"))
    result = LookupResult(8, (TailKeyBoundary(1, 12), TailKeyBoundary(3, 8)))
    assert codec.decode_request(codec.encode_request(request)) == request
    assert codec.decode_result(codec.encode_result(result)) == result


def test_lookup_codec_rejects_reversed_wire_interval() -> None:
    codec = LookupCodec()
    frames = codec.encode_request(LookupRequest(8, 4, (0,), (b"a",)))
    with pytest.raises(ValueError):
        codec.decode_request(frames)
