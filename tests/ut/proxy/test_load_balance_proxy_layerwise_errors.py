"""Error-path tests for load_balance_proxy_layerwise_server_example.

The layerwise proxy books a decoder's load and registers the request batch
before it builds the streaming response, but releases both only from
``generate_stream``'s ``finally`` - which runs when the response body is
iterated, not when the handler returns. Any failure in between used to leak
permanently.
"""

import argparse
import asyncio
import json

import pytest

pytest.importorskip("vllm")

from examples.disaggregated_prefill_v1 import (  # noqa: E402
    load_balance_proxy_layerwise_server_example as proxy,
)


class _FakeRequest:
    """Minimal stand-in for a starlette Request."""

    def __init__(self, payload: dict):
        self._payload = payload

    async def json(self):
        return self._payload

    async def body(self):
        return json.dumps(self._payload).encode()


@pytest.fixture
def state():
    proxy.global_args = argparse.Namespace(host="127.0.0.1", port=9000, max_retries=1, retry_delay=0.0)
    proxy.proxy_state = proxy.ProxyState([("127.0.0.1", 8100)], [("127.0.0.1", 8200)])
    return proxy.proxy_state


def _assert_nothing_booked(state) -> None:
    assert state.decoders[0].active_tokens == 0
    assert state.req_data_dict == {}
    assert state.metaserver_expected_ids == {}
    assert state.metaserver_params == {}
    assert state.metaserver_ready_events == {}


def test_a_malformed_chat_body_books_nothing(state):
    """``messages[0]`` runs after the decoder is booked and the batch
    registered, so its IndexError used to strand both forever."""
    request = _FakeRequest({"messages": [], "max_tokens": 4})

    with pytest.raises(IndexError):
        asyncio.run(proxy._handle_completions("/chat/completions", request))

    _assert_nothing_booked(state)


def test_a_completed_stream_releases_its_booking(state, monkeypatch):
    """The happy path must still release exactly once - a double release would
    drive the decoder's load below zero."""

    async def _no_chunks(*args, **kwargs):
        if False:  # pragma: no cover - makes this an async generator
            yield b""

    monkeypatch.setattr(proxy, "stream_service_response_with_retry", _no_chunks)

    response = asyncio.run(
        proxy._handle_completions(
            "/chat/completions",
            _FakeRequest({"messages": [{"role": "user", "content": "hi"}]}),
        )
    )
    assert state.decoders[0].active_tokens > 0, "the decode leg must book while it streams"

    async def _drain():
        async for _ in response.body_iterator:
            pass

    asyncio.run(_drain())

    _assert_nothing_booked(state)
