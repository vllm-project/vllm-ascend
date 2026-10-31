"""Regression tests for layerwise proxy setup failures and cancellation cleanup."""

import argparse
import asyncio
import json

import pytest

pytest.importorskip("vllm")

from examples.disaggregated_prefill_v1 import (  # noqa: E402
    load_balance_proxy_layerwise_server_example as proxy,
)


class _FakeRequest:
    def __init__(self, payload: dict):
        self._payload = payload

    async def json(self):
        return self._payload

    async def body(self):
        return json.dumps(self._payload).encode()


@pytest.fixture
def state(monkeypatch):
    monkeypatch.setattr(
        proxy,
        "global_args",
        argparse.Namespace(host="127.0.0.1", port=9000, max_retries=1, retry_delay=0.0),
        raising=False,
    )
    monkeypatch.setattr(
        proxy, "proxy_state", proxy.ProxyState([("127.0.0.1", 8100)], [("127.0.0.1", 8200)]), raising=False
    )
    return proxy.proxy_state


def _assert_nothing_booked(state) -> None:
    assert state.decoders[0].active_tokens == 0
    assert state.req_data_dict == {}
    assert state.metaserver_expected_ids == {}
    assert state.metaserver_params == {}
    assert state.metaserver_ready_events == {}


def test_a_malformed_chat_body_books_nothing(state):
    """A setup failure must release decoder load and request bookkeeping."""
    request = _FakeRequest({"messages": [], "max_tokens": 4})

    with pytest.raises(IndexError):
        asyncio.run(proxy._handle_completions("/chat/completions", request))

    _assert_nothing_booked(state)


def test_cancelled_response_setup_releases_booking_without_error_output(state, monkeypatch, capsys):
    def cancel_response(*args, **kwargs):
        raise asyncio.CancelledError()

    monkeypatch.setattr(proxy, "StreamingResponse", cancel_response)
    request = _FakeRequest({"messages": [{"role": "user", "content": "hi"}]})

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(proxy._handle_completions("/chat/completions", request))

    _assert_nothing_booked(state)
    output = capsys.readouterr()
    assert output.out == ""
    assert output.err == ""


def test_cancelled_stream_releases_its_booking(state, monkeypatch):
    async def cancel_stream(*args, **kwargs):
        yield b"data: [DONE]\n\n"
        raise asyncio.CancelledError()

    monkeypatch.setattr(proxy, "stream_service_response_with_retry", cancel_stream)

    async def run():
        response = await proxy._handle_completions(
            "/chat/completions", _FakeRequest({"messages": [{"role": "user", "content": "hi"}]})
        )
        with pytest.raises(asyncio.CancelledError):
            async for _ in response.body_iterator:
                pass

    asyncio.run(run())
    _assert_nothing_booked(state)


def test_a_completed_stream_releases_its_booking(state, monkeypatch):
    """A completed stream must release decoder load exactly once."""

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
