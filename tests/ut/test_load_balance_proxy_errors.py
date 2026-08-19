import argparse
import asyncio
import json
from typing import Any

import httpx
import pytest
from fastapi.responses import Response, StreamingResponse
from starlette.requests import Request

from examples.disaggregated_prefill_v1 import load_balance_proxy_server_example as proxy


def make_request(payload: dict) -> Request:
    body = json.dumps(payload).encode("utf-8")
    sent = False

    async def receive():
        nonlocal sent
        if sent:
            return {"type": "http.disconnect"}
        sent = True
        return {"type": "http.request", "body": body, "more_body": False}

    return Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/v1/chat/completions",
            "headers": [(b"content-type", b"application/json")],
        },
        receive,
    )


class FakeRuntime:
    def __init__(self, decoder_client: httpx.AsyncClient):
        self.decoder_client = decoder_client
        self.schedule_calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []

    async def get_client(self, role: proxy.ServerRole, key: str) -> httpx.AsyncClient:
        assert role is proxy.ServerRole.DECODE
        assert key == "decode"
        return self.decoder_client

    async def schedule(self, method: str, /, *args: Any, **kwargs: Any) -> None:
        self.schedule_calls.append((method, args, kwargs))


class BlockingStream(httpx.AsyncByteStream):
    def __init__(self):
        self.started = asyncio.Event()
        self.closed = False

    async def __aiter__(self):
        self.started.set()
        await asyncio.Event().wait()
        yield b""

    async def aclose(self) -> None:
        self.closed = True


class ListStream(httpx.AsyncByteStream):
    def __init__(self, chunks: list[bytes]):
        self.chunks = chunks
        self.closed = False

    async def __aiter__(self):
        for chunk in self.chunks:
            yield chunk

    async def aclose(self) -> None:
        self.closed = True


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
async def test_decode_4xx_response_is_returned_before_streaming(monkeypatch, stream: bool):
    requests_seen = 0
    error_body = {
        "error": {
            "message": "This model's maximum context length is 8192 tokens.",
            "type": "BadRequestError",
            "code": 400,
        }
    }

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal requests_seen
        requests_seen += 1
        assert request.url.path == "/v1/chat/completions"
        return httpx.Response(400, json=error_body)

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        base_url="http://decoder/v1",
    ) as decoder_client:
        fake_runtime = FakeRuntime(decoder_client)

        async def fake_assign_instances(*args, **kwargs):
            return proxy.InstanceInfo(
                request_id="request-id",
                prefiller_key="prefill",
                prefiller_score=1.0,
                decoder_key="decode",
                decoder_score=1.0,
                decoder_host="decoder",
                decoder_port=8000,
                prefiller_cached_tokens=0,
            )

        monkeypatch.setattr(proxy, "get_runtime", lambda: fake_runtime)
        monkeypatch.setattr(proxy, "get_global_args", lambda: argparse.Namespace(max_retries=3, retry_delay=0))
        monkeypatch.setattr(proxy, "assign_instances", fake_assign_instances)

        response = await proxy.handle_completions_impl(
            "/chat/completions",
            make_request(
                {
                    "model": "m",
                    "messages": [{"role": "user", "content": "Hello"}],
                    "max_tokens": 9000,
                    "stream": stream,
                }
            ),
        )

    assert isinstance(response, Response)
    assert not isinstance(response, StreamingResponse)
    assert response.status_code == 400
    assert json.loads(response.body) == error_body
    assert requests_seen == 1
    assert fake_runtime.schedule_calls == [("finish_request", ("prefill", 1.0, "decode", 1.0, True), {})]


@pytest.mark.asyncio
async def test_decode_5xx_response_is_retried_then_returned():
    requests_seen = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal requests_seen
        requests_seen += 1
        return httpx.Response(503, json={"error": {"message": "busy"}})

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        base_url="http://decoder/v1",
    ) as decoder_client:
        response, _, _ = await proxy.open_stream_service_response_with_retry(
            decoder_client,
            "/chat/completions",
            {},
            "request-id",
            max_retries=2,
            base_delay=0,
        )
        try:
            assert response.status_code == 503
            assert response.json() == {"error": {"message": "busy"}}
        finally:
            await response.aclose()

    assert requests_seen == 2


@pytest.mark.asyncio
async def test_success_response_replays_first_chunk_once():
    chunks = [b"first", b"second"]
    stream = ListStream(chunks)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, stream=stream)

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        base_url="http://decoder/v1",
    ) as decoder_client:
        response, chunk_iterator, first_chunk = await proxy.open_stream_service_response_with_retry(
            decoder_client,
            "/chat/completions",
            {},
            "request-id",
            max_retries=1,
            base_delay=0,
        )
        try:
            observed = [first_chunk]
            async for chunk in chunk_iterator:
                observed.append(chunk)
        finally:
            await response.aclose()

    assert observed == chunks
    assert stream.closed


@pytest.mark.asyncio
async def test_decode_connection_error_returns_502(monkeypatch):
    async def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        base_url="http://decoder/v1",
    ) as decoder_client:
        fake_runtime = FakeRuntime(decoder_client)

        async def fake_assign_instances(*args, **kwargs):
            return proxy.InstanceInfo(
                request_id="request-id",
                prefiller_key="prefill",
                prefiller_score=1.0,
                decoder_key="decode",
                decoder_score=1.0,
                decoder_host="decoder",
                decoder_port=8000,
                prefiller_cached_tokens=0,
            )

        monkeypatch.setattr(proxy, "get_runtime", lambda: fake_runtime)
        monkeypatch.setattr(proxy, "get_global_args", lambda: argparse.Namespace(max_retries=1, retry_delay=0))
        monkeypatch.setattr(proxy, "assign_instances", fake_assign_instances)

        response = await proxy.handle_completions_impl(
            "/chat/completions",
            make_request(
                {
                    "model": "m",
                    "messages": [{"role": "user", "content": "Hello"}],
                    "max_tokens": 16,
                    "stream": False,
                }
            ),
        )

    assert isinstance(response, Response)
    assert response.status_code == 502
    assert json.loads(response.body)["error"]["code"] == "decode_backend_unavailable"
    assert fake_runtime.schedule_calls == [("finish_request", ("prefill", 1.0, "decode", 1.0, True), {})]


@pytest.mark.asyncio
async def test_cancelling_during_first_decode_chunk_releases_resources(monkeypatch):
    stream = BlockingStream()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, stream=stream)

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        base_url="http://decoder/v1",
    ) as decoder_client:
        fake_runtime = FakeRuntime(decoder_client)

        async def fake_assign_instances(*args, **kwargs):
            return proxy.InstanceInfo(
                request_id="request-id",
                prefiller_key="prefill",
                prefiller_score=1.0,
                decoder_key="decode",
                decoder_score=1.0,
                decoder_host="decoder",
                decoder_port=8000,
                prefiller_cached_tokens=0,
            )

        monkeypatch.setattr(proxy, "get_runtime", lambda: fake_runtime)
        monkeypatch.setattr(proxy, "get_global_args", lambda: argparse.Namespace(max_retries=1, retry_delay=0))
        monkeypatch.setattr(proxy, "assign_instances", fake_assign_instances)

        task = asyncio.create_task(
            proxy.handle_completions_impl(
                "/chat/completions",
                make_request(
                    {
                        "model": "m",
                        "messages": [{"role": "user", "content": "Hello"}],
                        "max_tokens": 16,
                        "stream": False,
                    }
                ),
            )
        )
        await asyncio.wait_for(stream.started.wait(), timeout=1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    assert stream.closed
    assert fake_runtime.schedule_calls == [("finish_request", ("prefill", 1.0, "decode", 1.0, True), {})]
