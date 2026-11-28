"""Tests for the layerwise PD load balance proxy's response reassembly.

The decode backend answers with SSE, but the network does not respect frame
boundaries: a frame can arrive split across two reads, and two frames can be
coalesced into a single read. The proxy must reassemble frames before parsing
them, otherwise it misses the frame's ``stop_reason`` and never runs its
recompute retry.
"""

import asyncio
import json
import sys
from pathlib import Path

import pytest

EXAMPLES_DIR = Path(__file__).resolve().parents[3] / "examples" / "disaggregated_prefill_v1"
sys.path.insert(0, str(EXAMPLES_DIR))

from examples.disaggregated_prefill_v1 import (  # noqa: E402
    load_balance_proxy_layerwise_server_example as proxy,
)


def _frame(content: str, stop_reason: str | None = None) -> bytes:
    payload = {
        "id": "x",
        "choices": [
            {
                "index": 0,
                "delta": {"content": content},
                "finish_reason": None,
                "stop_reason": stop_reason,
            },
        ],
    }
    return b"data: " + json.dumps(payload).encode("utf-8") + b"\n\n"


DONE = b"data: [DONE]\n\n"


class _FakeResponse:
    """Minimal stand-in for the Starlette request object."""

    def __init__(self, body: dict) -> None:
        self._body = body

    async def json(self) -> dict:
        return self._body

    async def body(self) -> bytes:
        return json.dumps(self._body).encode("utf-8")


class _FakeProxyState:
    """Minimal stand-in for the proxy's scheduling state."""

    class _FakeDecoder:
        class client:  # noqa: N801 - mimics the httpx client attribute
            pass

        url = "http://decode"

    def __init__(self) -> None:
        self.decoders = [self._FakeDecoder()]
        self.req_to_prefiller: dict[str, str] = {}

    async def next_req_id(self) -> str:
        return "req-0"

    def register_request_batch(self, *args, **kwargs) -> None:
        pass

    def calculate_decode_scores(self, *args, **kwargs) -> float:
        return 0.0

    def select_decoder(self, *args, **kwargs) -> int:
        return 0

    def release_decoder(self, *args, **kwargs) -> None:
        pass

    async def cleanup_request_batch(self, *args, **kwargs) -> None:
        pass


async def _run_proxy(chunks_per_attempt: list[list[bytes]]) -> list[bytes]:
    """Drive the layerwise proxy handler and return the forwarded chunks.

    ``chunks_per_attempt`` holds the chunks each decoder attempt emits; the
    second attempt is only reached when the proxy detects a recompute.
    """
    attempts = iter(chunks_per_attempt)

    async def fake_stream(*args, **kwargs):
        for chunk in next(attempts):
            yield chunk

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(proxy, "proxy_state", _FakeProxyState())
    monkeypatch.setattr(proxy, "stream_service_response_with_retry", fake_stream)
    monkeypatch.setattr(
        proxy,
        "global_args",
        proxy.argparse.Namespace(max_retries=0, retry_delay=0.0, host="127.0.0.1", port=8000),
        raising=False,
    )

    request = _FakeResponse(
        {
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 20,
            "stream": True,
        }
    )

    try:
        response = await proxy._handle_completions("/v1/chat/completions", request)
        return [chunk async for chunk in response.body_iterator]
    finally:
        monkeypatch.undo()


def test_frame_split_across_chunks_is_reassembled():
    """A frame split across two reads still drives the recompute retry.

    The recompute signal travels in the frame's stop_reason. When the frame is
    split across reads, a per-chunk parse misses it and the proxy never
    retries, so the aborted generation reaches the client.
    """
    frame = _frame("gen", stop_reason="recomputed")
    cut = len(frame) // 2

    chunks = asyncio.run(
        _run_proxy(
            [
                [frame[:cut], frame[cut:]],
                [_frame("retried"), DONE],
            ]
        )
    )

    body = b"".join(chunks)
    assert b"retried" in body
    assert b"recomputed" not in body


def test_coalesced_frames_are_reassembled():
    """A frame coalesced with [DONE] into one read still drives the retry."""
    chunks = asyncio.run(
        _run_proxy(
            [
                [_frame("gen", stop_reason="recomputed") + DONE],
                [_frame("retried"), DONE],
            ]
        )
    )

    body = b"".join(chunks)
    assert b"retried" in body
    assert b"recomputed" not in body


def test_plain_stream_is_forwarded_unchanged():
    """A frame that arrives in one read is forwarded as it is."""
    chunks = asyncio.run(_run_proxy([[_frame("plain-token"), DONE]]))

    body = b"".join(chunks)
    assert b"plain-token" in body
