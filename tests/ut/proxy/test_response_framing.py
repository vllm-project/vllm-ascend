# SPDX-License-Identifier: Apache-2.0
"""Deterministic framing boundaries that TCP may otherwise coalesce."""

import asyncio
import json

import pytest

from examples.disaggregated_prefill_v1 import load_balance_proxy_server_example as proxy


async def collect(chunks, stream=True):
    async def source():
        for chunk in chunks:
            yield chunk

    return [frame async for frame in proxy.iter_response_frames(source(), stream)]


@pytest.mark.parametrize("newline", [b"\n", b"\r\n", b"\r"])
def test_every_split_and_bytewise_utf8(newline):
    event = b"data: " + json.dumps({"choices": [], "text": "中文🙂"}, ensure_ascii=False).encode() + newline * 2
    frames = [event, b": heartbeat" + newline * 2, b"data: [DONE]" + newline * 2]
    raw = b"".join(frames)

    async def check():
        for split in range(len(raw) + 1):
            assert await collect([raw[:split], b"", raw[split:]]) == frames
        assert await collect([bytes([byte]) for byte in raw]) == frames

    asyncio.run(check())


def test_event_is_yielded_before_upstream_finishes():
    async def check():
        event = b'data: {"choices": []}\n\n'

        async def source():
            yield event
            raise AssertionError("Must not consume the next event before yielding this one")

        frames = proxy.iter_response_frames(source(), True)
        assert await anext(frames) == event
        await frames.aclose()

    asyncio.run(check())


def test_json_reassembly_and_incomplete_tail_passthrough():
    async def check():
        raw = '{"text":"中文🙂"}'.encode()
        assert await collect([bytes([byte]) for byte in raw], False) == [raw]
        assert await collect([], False) == []
        assert await collect([], True) == []
        assert await collect([b'data: {"unfinished":']) == [b'data: {"unfinished":']

    asyncio.run(check())


def test_multiline_data_and_headers_preserved():
    frame = b': comment\r\nid: 7\r\nevent: message\r\ndata: {"choices": [],\r\ndata: "usage": {}}\r\n\r\n'
    obj = json.loads(proxy.response_chunk_payload(frame, True))
    proxy.update_cached_tokens_in_chunk(obj, 7)
    rewritten = proxy.encode_response_chunk(obj, True, frame)
    assert rewritten.startswith(b": comment\r\nid: 7\r\nevent: message\r\n")
    assert rewritten.endswith(b"\r\n\r\n")
    assert rewritten.count(b"data:") == 1
    assert json.loads(proxy.response_chunk_payload(rewritten, True)) == obj


def test_upstream_error_and_cancellation_propagate():
    async def check(failure):
        async def source():
            yield b"data: {}\n\n"
            raise failure

        frames = proxy.iter_response_frames(source(), True)
        assert await anext(frames) == b"data: {}\n\n"
        with pytest.raises(type(failure)):
            await anext(frames)

    for failure in (RuntimeError("upstream disconnected"), asyncio.CancelledError()):
        asyncio.run(check(failure))
