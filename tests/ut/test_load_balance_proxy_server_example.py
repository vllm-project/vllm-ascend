# SPDX-License-Identifier: Apache-2.0

import argparse
import asyncio
import importlib.util
import json
import sys
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from typing import Any

import httpx
import pytest
from starlette.requests import Request
from starlette.responses import StreamingResponse

EXAMPLE_PATH = (
    Path(__file__).parents[2] / "examples" / "disaggregated_prefill_v1" / "load_balance_proxy_server_example.py"
)
SPEC = importlib.util.spec_from_file_location("load_balance_proxy_server_example_under_test", EXAMPLE_PATH)
assert SPEC is not None and SPEC.loader is not None
proxy: Any = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = proxy
SPEC.loader.exec_module(proxy)

Responder = Callable[[asyncio.StreamReader, asyncio.StreamWriter], Awaitable[None]]


class FakeRequest:
    def __init__(self, payload: dict):
        self._payload = payload
        self._body = json.dumps(payload).encode()
        self.disconnect = asyncio.Event()

    async def json(self) -> dict:
        return self._payload.copy()

    async def body(self) -> bytes:
        return self._body

    async def receive(self) -> dict:
        await self.disconnect.wait()
        return {"type": "http.disconnect"}


class TcpBackend:
    def __init__(self, responders: Responder | list[Responder]):
        self.responders = responders if isinstance(responders, list) else [responders]
        self.connection_count = 0
        self._tasks: set[asyncio.Task] = set()
        self.server: asyncio.Server | None = None

    async def start(self) -> None:
        self.server = await asyncio.start_server(self._handle, "127.0.0.1", 0)

    @property
    def port(self) -> int:
        assert self.server is not None
        return self.server.sockets[0].getsockname()[1]

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        task = asyncio.current_task()
        assert task is not None
        self._tasks.add(task)
        index = min(self.connection_count, len(self.responders) - 1)
        self.connection_count += 1
        try:
            await read_http_request(reader)
            await self.responders[index](reader, writer)
        except (asyncio.IncompleteReadError, ConnectionError):
            pass
        finally:
            writer.close()
            with suppress(ConnectionError):
                await writer.wait_closed()
            self._tasks.discard(task)

    async def close(self) -> None:
        if self.server is not None:
            self.server.close()
            await self.server.wait_closed()
        for task in list(self._tasks):
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)


async def read_http_request(reader: asyncio.StreamReader) -> None:
    headers = await reader.readuntil(b"\r\n\r\n")
    content_length = 0
    for line in headers.split(b"\r\n"):
        if line.lower().startswith(b"content-length:"):
            content_length = int(line.split(b":", 1)[1])
    if content_length:
        await reader.readexactly(content_length)


def json_responder(payload: dict) -> Responder:
    async def respond(_reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        body = json.dumps(payload).encode()
        writer.write(
            b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: "
            + str(len(body)).encode()
            + b"\r\nConnection: close\r\n\r\n"
            + body
        )
        await writer.drain()

    return respond


def chunked_responder(chunks: list[bytes], *, stall: bool = False, disconnected=None) -> Responder:
    async def respond(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        writer.write(
            b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n"
            b"Transfer-Encoding: chunked\r\nConnection: keep-alive\r\n\r\n"
        )
        for chunk in chunks:
            writer.write(f"{len(chunk):x}\r\n".encode() + chunk + b"\r\n")
            await writer.drain()
        if stall:
            await reader.read()
            if disconnected is not None:
                disconnected.set()
        else:
            writer.write(b"0\r\n\r\n")
            await writer.drain()

    return respond


async def hung_responder(_reader: asyncio.StreamReader, _writer: asyncio.StreamWriter) -> None:
    await asyncio.Event().wait()


async def close_without_response(_reader: asyncio.StreamReader, _writer: asyncio.StreamWriter) -> None:
    return


@asynccontextmanager
async def proxy_harness(
    decoder_responders: Responder | list[Responder],
    *,
    read_timeout: float = 0.1,
    max_retries: int = 1,
):
    prefill = TcpBackend(json_responder({"kv_transfer_params": {}}))
    decoder = TcpBackend(decoder_responders)
    await prefill.start()
    await decoder.start()
    scheduler = proxy.SharedProxyScheduler([("127.0.0.1", prefill.port)], [("127.0.0.1", decoder.port)])
    args = argparse.Namespace(
        max_retries=max_retries,
        retry_delay=0.001,
        connect_timeout=0.1,
        write_timeout=0.1,
        pool_timeout=0.1,
        read_timeout=read_timeout,
    )
    test_runtime = proxy.WorkerRuntime(scheduler, proxy.build_http_timeout(args))
    proxy.runtime = test_runtime
    proxy.global_args = args
    try:
        await test_runtime.sync_clients()
        yield scheduler, decoder
    finally:
        await test_runtime.close()
        proxy.runtime = None
        proxy.global_args = None
        await prefill.close()
        await decoder.close()


async def make_request(payload: dict | None = None):
    return await proxy.handle_completions_impl(
        "/completions",
        FakeRequest(payload or {"model": "test", "prompt": "hi", "max_tokens": 1}),
    )


async def response_body(response) -> bytes:
    if not isinstance(response, StreamingResponse):
        return response.body

    async def collect() -> bytes:
        return b"".join([chunk async for chunk in response.body_iterator])

    return await asyncio.wait_for(collect(), timeout=1.0)


def assert_loads_released(scheduler: Any) -> None:
    health = scheduler.healthcheck()
    assert health["request_num"] == 0
    assert health["oldest_request_age_seconds"] == 0.0
    assert health["max_request_idle_seconds"] == 0.0
    assert all(server.active_kv_cache == 0 for server in scheduler.prefillers.values())
    assert all(server.active_tokens == 0 for server in scheduler.decoders.values())


@pytest.mark.asyncio
async def test_hung_response_headers_return_504_and_finalize_tainted_instance():
    async with proxy_harness(hung_responder) as (scheduler, decoder):
        request_task = asyncio.create_task(make_request())
        await asyncio.wait_for(wait_for_request_count(scheduler, 1), timeout=1.0)
        scheduler.remove_instances(proxy.ServerRole.DECODE, [("127.0.0.1", decoder.port)])

        response = await asyncio.wait_for(request_task, timeout=1.0)

        assert response.status_code == 504
        assert json.loads(response.body) == {
            "error": {
                "message": "Upstream request timed out",
                "type": "upstream_timeout",
            }
        }
        assert_loads_released(scheduler)
        scheduler.finalize_tainted_instances()
        assert scheduler.decoders == {}


async def wait_for_request_count(scheduler: Any, expected: int) -> None:
    while scheduler.healthcheck()["request_num"] != expected:
        await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_timeout_after_first_sse_chunk_does_not_retry_or_duplicate():
    chunk = b'data: {"choices":[{"text":"one"}]}\n\n'
    async with proxy_harness(chunked_responder([chunk], stall=True)) as (scheduler, decoder):
        response = await asyncio.wait_for(make_request({"prompt": "hi", "stream": True}), timeout=1.0)
        body = await response_body(response)

        assert body == chunk
        assert decoder.connection_count == 1
        assert_loads_released(scheduler)


@pytest.mark.asyncio
async def test_client_disconnect_cancels_upstream_and_releases_load():
    disconnected = asyncio.Event()
    chunk = b'data: {"choices":[{"text":"one"}]}\n\n'
    async with proxy_harness(
        chunked_responder([chunk], stall=True, disconnected=disconnected),
        read_timeout=5.0,
    ) as (scheduler, _decoder):
        response = await asyncio.wait_for(make_request({"prompt": "hi", "stream": True}), timeout=1.0)
        disconnect = asyncio.Event()
        sent_bodies: list[bytes] = []

        async def receive():
            await disconnect.wait()
            return {"type": "http.disconnect"}

        async def send(message):
            if message["type"] == "http.response.body":
                sent_bodies.append(message.get("body", b""))
                disconnect.set()

        scope = {
            "type": "http",
            "asgi": {"version": "3.0", "spec_version": "2.3"},
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": "/v1/completions",
            "raw_path": b"/v1/completions",
            "query_string": b"",
            "headers": [],
            "client": ("127.0.0.1", 1),
            "server": ("127.0.0.1", 2),
        }
        await asyncio.wait_for(response(scope, receive, send), timeout=1.0)
        await asyncio.wait_for(disconnected.wait(), timeout=1.0)

        assert b"".join(sent_bodies) == chunk
        assert_loads_released(scheduler)


@pytest.mark.asyncio
async def test_disconnect_before_response_headers_cancels_handler_and_upstream(monkeypatch):
    upstream_cancelled = asyncio.Event()
    original_stream = proxy.stream_service_response_with_retry

    async def observed_stream(*args, **kwargs):
        try:
            async for chunk in original_stream(*args, **kwargs):
                yield chunk
        except asyncio.CancelledError:
            upstream_cancelled.set()
            raise

    monkeypatch.setattr(proxy, "stream_service_response_with_retry", observed_stream)
    async with proxy_harness(hung_responder, read_timeout=5.0) as (scheduler, decoder):
        request = FakeRequest({"prompt": "hi", "stream": True})
        handler = asyncio.create_task(proxy.handle_completions(request=request))
        await asyncio.wait_for(wait_for_request_count(scheduler, 1), timeout=1.0)
        await asyncio.wait_for(wait_for_connection_count(decoder, 1), timeout=1.0)

        request.disconnect.set()

        assert await asyncio.wait_for(handler, timeout=1.0) is None
        await asyncio.wait_for(upstream_cancelled.wait(), timeout=1.0)
        assert_loads_released(scheduler)


@pytest.mark.asyncio
async def test_disconnect_listener_does_not_compete_for_chunked_request_body():
    messages = [
        {"type": "http.request", "body": b'{"prompt":', "more_body": True},
        {"type": "http.request", "body": b'"hi"}', "more_body": False},
    ]
    receive_calls = 0
    body_messages_received = 0
    active_receivers = 0
    max_active_receivers = 0

    async def receive():
        nonlocal active_receivers, body_messages_received
        nonlocal max_active_receivers, receive_calls
        active_receivers += 1
        max_active_receivers = max(max_active_receivers, active_receivers)
        try:
            await asyncio.sleep(0)
            receive_calls += 1
            if messages:
                message = messages.pop(0)
                body_messages_received += 1
                return message
            await asyncio.Event().wait()
        finally:
            active_receivers -= 1

    request = Request(
        {"type": "http", "method": "POST", "path": "/v1/completions", "headers": []},
        receive,
    )

    @proxy.with_cancellation
    async def read_json(*, request):
        return await request.json()

    result = await asyncio.wait_for(read_json(request=request), timeout=1.0)

    assert result == {"prompt": "hi"}
    assert receive_calls == 3
    assert body_messages_received == 2
    assert max_active_receivers == 1


async def wait_for_connection_count(backend: TcpBackend, expected: int) -> None:
    while backend.connection_count != expected:
        await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_normal_non_streaming_and_sse_responses_succeed():
    non_stream = {"choices": [{"text": "ok"}], "usage": {"completion_tokens": 1}}
    async with proxy_harness(json_responder(non_stream)) as (scheduler, _decoder):
        response = await asyncio.wait_for(make_request(), timeout=1.0)
        response_json = json.loads(await response_body(response))
        assert response_json["choices"] == non_stream["choices"]
        assert response_json["usage"]["completion_tokens"] == 1
        assert response_json["usage"]["prompt_tokens_details"]["cached_tokens"] == 0
        assert_loads_released(scheduler)

    sse = b'data: {"choices":[{"text":"ok"}]}\n\n'
    async with proxy_harness(chunked_responder([sse])) as (scheduler, _decoder):
        response = await asyncio.wait_for(make_request({"prompt": "hi", "stream": True}), timeout=1.0)
        assert await response_body(response) == sse
        assert_loads_released(scheduler)


@pytest.mark.asyncio
async def test_retry_before_first_chunk_recovers():
    body = {"choices": [{"text": "recovered"}]}
    async with proxy_harness([close_without_response, json_responder(body)], max_retries=2) as (scheduler, decoder):
        response = await asyncio.wait_for(make_request(), timeout=1.0)

        assert json.loads(await response_body(response)) == body
        assert decoder.connection_count == 2
        assert_loads_released(scheduler)


def test_finish_request_is_idempotent_and_health_reports_lease_age():
    scheduler = proxy.SharedProxyScheduler([("127.0.0.1", 1)], [("127.0.0.1", 2)])
    scheduler.begin_request("request-id", 12.0)
    scheduler.pick_decoder("request-id", 8.0)
    scheduler.release_prefill_kv("request-id")
    scheduler.release_prefill_kv("request-id")

    health = scheduler.healthcheck()
    assert health["request_num"] == 1
    assert health["oldest_request_age_seconds"] >= 0.0
    assert health["max_request_idle_seconds"] >= 0.0
    assert scheduler.finish_request("request-id") is True
    assert scheduler.finish_request("request-id") is False
    assert_loads_released(scheduler)


@pytest.mark.asyncio
async def test_cleanup_finishes_before_repeated_cancellation_is_propagated():
    class BlockingRuntime:
        def __init__(self):
            self.started = asyncio.Event()
            self.release = asyncio.Event()
            self.finished = False

        async def schedule(self, _method, _request_id):
            self.started.set()
            await self.release.wait()
            self.finished = True

    test_runtime = BlockingRuntime()
    cleanup = asyncio.create_task(proxy._run_cleanup(test_runtime, "finish_request", "request-id"))
    await asyncio.wait_for(test_runtime.started.wait(), timeout=1.0)

    cleanup.cancel()
    await asyncio.sleep(0)
    cleanup.cancel()
    await asyncio.sleep(0)
    test_runtime.release.set()

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(cleanup, timeout=1.0)
    assert test_runtime.finished is True


def test_timeout_cli_defaults_and_validation(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["proxy"])
    args = proxy.parse_args()
    assert (args.connect_timeout, args.write_timeout, args.pool_timeout, args.read_timeout) == (
        10.0,
        30.0,
        10.0,
        300.0,
    )
    for invalid in ("0", "-1", "nan", "inf", "-inf"):
        with pytest.raises(argparse.ArgumentTypeError):
            proxy.positive_finite_float(invalid)

    timeout = proxy.build_http_timeout(args)
    assert timeout == httpx.Timeout(connect=10, write=30, pool=10, read=300)
