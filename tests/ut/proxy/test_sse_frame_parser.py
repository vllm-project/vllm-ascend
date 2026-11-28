"""Unit tests for the incremental SSE frame parser used by the PD proxies.

The parser's job is to reassemble the decode backend's SSE stream into frames
when the network does not respect frame boundaries.
"""

import json

import pytest

from examples.disaggregated_prefill_v1 import (
    load_balance_proxy_layerwise_server_example as layerwise_proxy,
)
from examples.disaggregated_prefill_v1 import (
    load_balance_proxy_server_example as proxy,
)


@pytest.fixture(params=[proxy.SSEFrameParser, layerwise_proxy.SSEFrameParser], ids=["standard", "layerwise"])
def parser(request):
    return request.param()


def _frame(content: str, **extra) -> bytes:
    payload = {
        "id": "x",
        "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": None}],
    }
    payload.update(extra)
    return b"data: " + json.dumps(payload).encode("utf-8") + b"\n\n"


def _payload(content: str, **extra) -> dict:
    payload = {
        "id": "x",
        "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": None}],
    }
    payload.update(extra)
    return payload


def test_complete_frame_is_parsed(parser):
    """A frame that arrives in one piece is returned as-is."""
    frames = parser.feed(_frame("hello"))
    assert len(frames) == 1
    assert frames[0].payload == _payload("hello")
    assert frames[0].is_sse is True


def test_frame_split_across_chunks_is_reassembled(parser):
    """A frame whose JSON is split across two chunks is parsed once complete."""
    frame = _frame("split")
    cut = len(frame) // 2

    assert parser.feed(frame[:cut]) == []
    frames = parser.feed(frame[cut:])

    assert len(frames) == 1
    assert frames[0].payload == _payload("split")


def test_two_frames_coalesced_in_one_chunk_are_both_parsed(parser):
    """Two frames delivered in one chunk are both returned."""
    frames = parser.feed(_frame("first") + _frame("second"))

    assert [f.payload for f in frames] == [_payload("first"), _payload("second")]


def test_byte_at_a_time_delivery_is_reassembled(parser):
    """A frame delivered one byte per chunk is parsed once complete."""
    frame = _frame("dripped")

    got = []
    for i in range(len(frame)):
        got.extend(parser.feed(frame[i : i + 1]))

    assert [f.payload for f in got] == [_payload("dripped")]


def test_done_sentinel_terminates_the_stream(parser):
    """The [DONE] sentinel yields no frame and drops trailing bytes."""
    assert parser.feed(b"data: [DONE]\n\n") == []
    assert parser.pending == b""


def test_done_sentinel_after_a_frame_keeps_the_frame(parser):
    """A frame followed by [DONE] in one chunk returns only the frame."""
    frames = parser.feed(_frame("before-done") + b"data: [DONE]\n\n")

    assert [f.payload for f in frames] == [_payload("before-done")]


def test_non_sse_frame_is_reported_as_not_sse(parser):
    """A frame without the data: prefix is parsed and flagged."""
    frames = parser.feed(json.dumps(_payload("plain")).encode("utf-8") + b"\n\n")

    assert len(frames) == 1
    assert frames[0].is_sse is False
    assert frames[0].payload == _payload("plain")


def test_flush_parses_a_frame_without_a_terminator(parser):
    """A trailing frame that was never terminated by a blank line is parsed."""
    parser.feed(b"data: " + json.dumps(_payload("tail")).encode("utf-8"))

    frame = parser.flush()
    assert frame is not None
    assert frame.payload == _payload("tail")


def test_flush_returns_none_on_an_empty_buffer(parser):
    """Flushing an empty buffer yields nothing."""
    assert parser.flush() is None


def test_malformed_frame_does_not_stall_the_stream(parser):
    """A frame that is not valid JSON is dropped, later frames still parse."""
    frames = parser.feed(b"data: {not json}\n\n" + _frame("after"))

    assert [f.payload for f in frames] == [_payload("after")]


@pytest.mark.parametrize("split_at", [1, 5, 17, 40])
def test_every_split_point_yields_the_same_frame(parser, split_at):
    """Splitting a frame anywhere yields the same parsed result."""
    frame = _frame("split-anywhere")

    assert parser.feed(frame[:split_at]) == []
    frames = parser.feed(frame[split_at:])

    assert [f.payload for f in frames] == [_payload("split-anywhere")]
