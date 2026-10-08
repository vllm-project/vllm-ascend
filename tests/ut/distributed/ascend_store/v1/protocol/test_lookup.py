"""Lookup wire encoding preserves cache-group identities and tail boundaries."""

from __future__ import annotations

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.coordinates import TokenRange
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.protocol.lookup import (
    LookupCodec,
    LookupRequest,
    LookupResult,
    TailKeyBoundary,
)


def test_lookup_codec_roundtrip_preserves_groups_and_partial_tail_boundaries() -> None:
    codec = LookupCodec()
    request = LookupRequest(TokenRange(4, 12), (1, 3), (b"a", b"b"))
    result = LookupResult(8, (TailKeyBoundary(1, 12), TailKeyBoundary(3, 8)))
    assert codec.decode_request(codec.encode_request(request)) == request
    assert codec.decode_result(codec.encode_result(result)) == result
