"""Unit tests for the incremental SSE frame parser used by the PD proxies.

The parser's job is to reassemble the decode backend's SSE stream into frames
when the network does not respect frame boundaries.
"""

import json
import sys
from pathlib import Path

import pytest

# The parser lives next to the proxy example scripts, which are run as
# standalone files. It is imported through the repository root so that a
# static checker (mypy) resolves it as well: the bare module name is only
# importable at runtime, because the examples directory is not on the
# checker's search path.
EXAMPLES_DIR = Path(__file__).resolve().parents[3] / "examples" / "disaggregated_prefill_v1"
sys.path.insert(0, str(EXAMPLES_DIR))

from examples.disaggregated_prefill_v1.sse_frame_parser import SSEFrameParser  # noqa: E402


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


def test_complete_frame_is_parsed():
    """A frame that arrives in one piece is returned as-is."""
    parser = SSEFrameParser()
    frames = parser.feed(_frame("hello"))
    assert len(frames) == 1
    assert frames[0].payload == _payload("hello")
    assert frames[0].is_sse is True


def test_frame_split_across_chunks_is_reassembled():
    """A frame whose JSON is split across two chunks is parsed once complete."""
    parser = SSEFrameParser()
    frame = _frame("split")
    cut = len(frame) // 2

    assert parser.feed(frame[:cut]) == []
    frames = parser.feed(frame[cut:])

    assert len(frames) == 1
    assert frames[0].payload == _payload("split")


def test_two_frames_coalesced_in_one_chunk_are_both_parsed():
    """Two frames delivered in one chunk are both returned."""
    parser = SSEFrameParser()
    frames = parser.feed(_frame("first") + _frame("second"))

    assert [f.payload for f in frames] == [_payload("first"), _payload("second")]


def test_byte_at_a_time_delivery_is_reassembled():
    """A frame delivered one byte per chunk is parsed once complete."""
    parser = SSEFrameParser()
    frame = _frame("dripped")

    got = []
    for i in range(len(frame)):
        got.extend(parser.feed(frame[i : i + 1]))

    assert [f.payload for f in got] == [_payload("dripped")]


def test_done_sentinel_terminates_the_stream():
    """The [DONE] sentinel yields no frame and drops trailing bytes."""
    parser = SSEFrameParser()
    assert parser.feed(b"data: [DONE]\n\n") == []
    assert parser.pending == b""


def test_done_sentinel_after_a_frame_keeps_the_frame():
    """A frame followed by [DONE] in one chunk returns only the frame."""
    parser = SSEFrameParser()
    frames = parser.feed(_frame("before-done") + b"data: [DONE]\n\n")

    assert [f.payload for f in frames] == [_payload("before-done")]


def test_non_sse_frame_is_reported_as_not_sse():
    """A frame without the data: prefix is parsed and flagged."""
    parser = SSEFrameParser()
    frames = parser.feed(json.dumps(_payload("plain")).encode("utf-8") + b"\n\n")

    assert len(frames) == 1
    assert frames[0].is_sse is False
    assert frames[0].payload == _payload("plain")


def test_flush_parses_a_frame_without_a_terminator():
    """A trailing frame that was never terminated by a blank line is parsed."""
    parser = SSEFrameParser()
    parser.feed(b"data: " + json.dumps(_payload("tail")).encode("utf-8"))

    frame = parser.flush()
    assert frame is not None
    assert frame.payload == _payload("tail")


def test_flush_returns_none_on_an_empty_buffer():
    """Flushing an empty buffer yields nothing."""
    parser = SSEFrameParser()
    assert parser.flush() is None


def test_malformed_frame_does_not_stall_the_stream():
    """A frame that is not valid JSON is dropped, later frames still parse."""
    parser = SSEFrameParser()
    frames = parser.feed(b"data: {not json}\n\n" + _frame("after"))

    assert [f.payload for f in frames] == [_payload("after")]


@pytest.mark.parametrize("split_at", [1, 5, 17, 40])
def test_every_split_point_yields_the_same_frame(split_at):
    """Splitting a frame anywhere yields the same parsed result."""
    frame = _frame("split-anywhere")
    parser = SSEFrameParser()

    assert parser.feed(frame[:split_at]) == []
    frames = parser.feed(frame[split_at:])

    assert [f.payload for f in frames] == [_payload("split-anywhere")]
