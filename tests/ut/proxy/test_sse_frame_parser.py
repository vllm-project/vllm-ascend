"""Unit tests for the incremental SSE frame parser used by the PD proxies.

The parser's job is to reassemble the decode backend's SSE stream into frames
when the network does not respect frame boundaries.
"""

import asyncio
import json
from contextlib import aclosing

import pytest

from examples.disaggregated_prefill_v1 import (
    load_balance_proxy_layerwise_server_example as layerwise_proxy,
)
from examples.disaggregated_prefill_v1 import (
    load_balance_proxy_server_example as standard_proxy,
)


@pytest.fixture(params=[standard_proxy, layerwise_proxy], ids=["standard", "layerwise"])
def proxy_module(request):
    return request.param


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


def test_complete_frame_is_parsed(proxy_module):
    """A frame that arrives in one piece is returned as-is."""
    parser = proxy_module.SSEFrameParser()
    frames = parser.feed(_frame("hello"))
    assert len(frames) == 1
    assert frames[0].payload == _payload("hello")
    assert frames[0].is_sse is True


def test_frame_split_across_chunks_is_reassembled(proxy_module):
    """A frame whose JSON is split across two chunks is parsed once complete."""
    parser = proxy_module.SSEFrameParser()
    frame = _frame("split")
    cut = len(frame) // 2

    assert parser.feed(frame[:cut]) == []
    frames = parser.feed(frame[cut:])

    assert len(frames) == 1
    assert frames[0].payload == _payload("split")


def test_two_frames_coalesced_in_one_chunk_are_both_parsed(proxy_module):
    """Two frames delivered in one chunk are both returned."""
    parser = proxy_module.SSEFrameParser()
    frames = parser.feed(_frame("first") + _frame("second"))

    assert [f.payload for f in frames] == [_payload("first"), _payload("second")]


def test_byte_at_a_time_delivery_is_reassembled(proxy_module):
    """A frame delivered one byte per chunk is parsed once complete."""
    parser = proxy_module.SSEFrameParser()
    frame = _frame("dripped")

    got = []
    for i in range(len(frame)):
        got.extend(parser.feed(frame[i : i + 1]))

    assert [f.payload for f in got] == [_payload("dripped")]


def test_done_sentinel_terminates_the_stream(proxy_module):
    """The [DONE] sentinel yields no frame and drops trailing bytes."""
    parser = proxy_module.SSEFrameParser()
    assert parser.feed(b"data: [DONE]\n\n")[0].raw == b"data: [DONE]\n\n"
    assert parser.pending == b""


def test_done_sentinel_after_a_frame_keeps_the_frame(proxy_module):
    """A frame followed by [DONE] in one chunk returns only the frame."""
    parser = proxy_module.SSEFrameParser()
    frames = parser.feed(_frame("before-done") + b"data: [DONE]\n\n")

    assert [f.payload for f in frames] == [_payload("before-done"), None]
    assert frames[-1].raw == b"data: [DONE]\n\n"


def test_non_sse_frame_is_reported_as_not_sse(proxy_module):
    """A frame without the data: prefix is parsed and flagged."""
    parser = proxy_module.SSEFrameParser()
    assert parser.feed(json.dumps(_payload("plain")).encode("utf-8") + b"\n\n") == []
    frames = [parser.flush()]

    assert len(frames) == 1
    assert frames[0].is_sse is False
    assert frames[0].payload == _payload("plain")


def test_flush_parses_a_frame_without_a_terminator(proxy_module):
    """A trailing frame that was never terminated by a blank line is parsed."""
    parser = proxy_module.SSEFrameParser()
    parser.feed(b"data: " + json.dumps(_payload("tail")).encode("utf-8"))

    frame = parser.flush()
    assert frame is not None
    assert frame.payload == _payload("tail")


def test_flush_returns_none_on_an_empty_buffer(proxy_module):
    """Flushing an empty buffer yields nothing."""
    parser = proxy_module.SSEFrameParser()
    assert parser.flush() is None


def test_malformed_frame_does_not_stall_the_stream(proxy_module):
    """A frame that is not valid JSON is dropped, later frames still parse."""
    parser = proxy_module.SSEFrameParser()
    frames = parser.feed(b"data: {not json}\n\n" + _frame("after"))

    assert [f.payload for f in frames] == [_payload("after")]


@pytest.mark.parametrize("split_at", [1, 5, 17, 40])
def test_every_split_point_yields_the_same_frame(proxy_module, split_at):
    """Splitting a frame anywhere yields the same parsed result."""
    frame = _frame("split-anywhere")
    parser = proxy_module.SSEFrameParser()

    assert parser.feed(frame[:split_at]) == []
    frames = parser.feed(frame[split_at:])

    assert [f.payload for f in frames] == [_payload("split-anywhere")]


@pytest.mark.parametrize("separator", [b"\n\n", b"\r\n\r\n"])
def test_raw_bytes_and_done_survive_every_split(proxy_module, separator):
    first = _frame("one").replace(b"\n\n", separator)
    second = _frame("two").replace(b"\n\n", separator)
    done = b"data: [DONE]" + separator
    wire = first + second + done
    for cut in range(len(wire) + 1):
        parser = proxy_module.SSEFrameParser()
        frames = parser.feed(wire[:cut]) + parser.feed(wire[cut:])
        assert [f.raw for f in frames] == [first, second, done]
        assert parser.feed(_frame("after-done")) == []
        assert parser.flush() is None


def test_pretty_json_waits_for_eof_even_with_blank_lines(proxy_module):
    raw = b'{\n\n"choices": [], "text": "hi"\n}'
    parser = proxy_module.SSEFrameParser()
    for value in raw:
        assert parser.feed(bytes([value])) == []
    frame = parser.flush()
    assert frame.raw == raw
    assert frame.payload == json.loads(raw)
    assert not frame.is_sse


def test_multiline_data_and_comments_preserve_wire_bytes(proxy_module):
    raw = b'event: message\r\ndata: {"choices": [],\r\ndata: "text": "hi"}\r\n\r\n'
    frames = proxy_module.SSEFrameParser().feed(b": ping\n\n" + raw)
    assert frames[0].raw == b": ping\n\n"
    assert frames[0].payload is None
    assert frames[1].raw == raw
    assert frames[1].payload == {"choices": [], "text": "hi"}


def test_incomplete_json_is_not_forwarded_at_eof(proxy_module):
    parser = proxy_module.SSEFrameParser()
    assert parser.feed(b'data: {"choices":') == []
    assert parser.flush() is None


def test_early_exit_closes_decoder_stream(proxy_module):
    async def run():
        closed = False

        async def source():
            nonlocal closed
            try:
                yield _frame("first") + _frame("unused")
            finally:
                closed = True

        async with aclosing(proxy_module.iter_response_frames(source())) as frames:
            async for frame in frames:
                assert frame.payload == _payload("first")
                break
        assert closed

    asyncio.run(run())


def test_transport_error_does_not_flush_partial_json(proxy_module):
    async def run():
        async def source():
            yield b'{"choices": []'
            raise ConnectionError("decoder disconnected")

        with pytest.raises(ConnectionError, match="decoder disconnected"):
            async for _ in proxy_module.iter_response_frames(source()):
                pytest.fail("Partial JSON must not reach the client")

    asyncio.run(run())
