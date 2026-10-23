"""Regression tests for Messages/Responses forwarding."""

import argparse
import asyncio
import json
from typing import Any

import httpx
import pytest
from starlette.requests import Request

from examples.disaggregated_prefill_v1 import load_balance_proxy_server_example as proxy


def _request(body: dict) -> Request:
    raw = json.dumps(body).encode("utf-8")

    async def receive():
        return {"type": "http.request", "body": raw, "more_body": False}

    return Request(
        {
            "type": "http",
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": "/v1/messages",
            "raw_path": b"/v1/messages",
            "query_string": b"",
            "headers": [(b"content-type", b"application/json")],
            "client": ("127.0.0.1", 12345),
            "server": ("127.0.0.1", 8000),
        },
        receive,
    )


class _Response:
    def __init__(self, payload: dict, status: int = 200):
        self.status_code = status
        self._payload = payload

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise AssertionError(f"unexpected prefill status {self.status_code}")


class _Stream:
    def __init__(self, chunks: list[bytes], status: int = 200):
        self.status_code = status
        self._chunks = list(chunks)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    def raise_for_status(self):
        if self.status_code >= 400:
            raise AssertionError(f"unexpected decode status {self.status_code}")

    async def aread(self):
        return b"".join(self._chunks)

    async def aiter_bytes(self):
        for chunk in self._chunks:
            yield chunk


class _Client:
    def __init__(self, *, payload: dict | None = None, chunks: list[bytes] | None = None):
        self.base_url = "http://127.0.0.1/v1"
        self.posts: list[dict] = []
        self.payload = payload or {}
        self.chunks = chunks or []

    async def post(self, endpoint, json=None, headers=None):
        self.posts.append({"endpoint": endpoint, "json": json, "headers": headers})
        return _Response(self.payload)

    def stream(self, method, endpoint, json=None, headers=None):
        self.posts.append({"endpoint": endpoint, "json": json, "headers": headers, "stream": True})
        return _Stream(self.chunks)

    async def aclose(self):
        return None


def _scheduler(prefill_ports=(8100, 8101), decode_ports=(8200, 8201)):
    return proxy.SharedProxyScheduler(
        [("127.0.0.1", port) for port in prefill_ports],
        [("127.0.0.1", port) for port in decode_ports],
    )


@pytest.fixture
def installed_runtime():
    """One prefiller and one decoder, with fake HTTP clients already registered."""
    prefill = _Client(
        payload={
            "kv_transfer_params": {"do_remote_decode": True, "remote_block_ids": [1]},
            "usage": {
                "cache_read_input_tokens": 32,
                "input_tokens_details": {"cached_tokens": 7},
                "prompt_tokens_details": {"cached_tokens": 4},
            },
        }
    )
    decode = _Client()
    scheduler = _scheduler(prefill_ports=(8100,), decode_ports=(8200,))
    runtime = proxy.WorkerRuntime(scheduler)
    runtime._clients[proxy.ServerRole.PREFILL][proxy.server_key("127.0.0.1", 8100)] = prefill
    runtime._clients[proxy.ServerRole.DECODE][proxy.server_key("127.0.0.1", 8200)] = decode
    old_runtime, old_args = proxy.runtime, proxy.global_args
    proxy.runtime = runtime
    proxy.global_args = argparse.Namespace(
        max_retries=1, retry_delay=0.0, workers=1, log_level="ERROR", enable_prefill_affinity=False
    )
    try:
        yield scheduler, prefill, decode
    finally:
        proxy.runtime = old_runtime
        proxy.global_args = old_args


async def _body(response) -> bytes:
    chunks = []
    async for chunk in response.body_iterator:
        chunks.append(chunk if isinstance(chunk, bytes) else bytes(chunk))
    return b"".join(chunks)


def test_prefill_payload_depends_on_api():
    original = {
        "model": "m",
        "input": "hello",
        "max_output_tokens": 500,
        "max_tokens": 80,
        "min_tokens": 2,
        "max_completion_tokens": 80,
        "stream": True,
        "stream_options": {"include_usage": True},
    }

    responses = proxy.build_prefill_request_for_api("/responses", original)
    assert responses["max_output_tokens"] == 1
    assert responses["stream"] is False
    assert responses["kv_transfer_params"]["do_remote_decode"] is True
    for key in ("max_tokens", "min_tokens", "max_completion_tokens", "stream_options"):
        assert key not in responses
    assert original["max_output_tokens"] == 500

    messages = proxy.build_prefill_request_for_api("/messages", dict(original))
    assert messages["max_tokens"] == 1
    assert messages["min_tokens"] == 1
    assert messages["max_completion_tokens"] == 1
    assert "max_output_tokens" in messages

    chat = proxy.build_prefill_request_for_api("/chat/completions", {"prompt": "hi", "max_tokens": 16})
    assert chat["max_tokens"] == 1
    assert chat["min_tokens"] == 1
    assert "max_output_tokens" not in chat


def test_cached_token_fields_follow_the_protocol():
    messages = {"usage": {"cache_read_input_tokens": 11}}
    responses = {"usage": {"input_tokens_details": {"cached_tokens": 22}}}
    openai = {"usage": {"prompt_tokens_details": {"cached_tokens": 33}}}
    assert proxy.extract_cached_tokens_for_api("/messages", messages) == 11
    assert proxy.extract_cached_tokens_for_api("/responses", responses) == 22
    assert proxy.extract_cached_tokens_for_api("/completions", openai) == 33
    assert proxy.extract_cached_tokens_for_api("/messages", {"usage": {}}) == 0

    message_out = {"usage": {"cache_read_input_tokens": 0}}
    assert proxy.write_cached_tokens_for_api("/messages", message_out, 32) is True
    assert message_out["usage"]["cache_read_input_tokens"] == 32
    assert "input_tokens" not in message_out["usage"]

    completed: dict[str, Any] = {"type": "response.completed", "response": {"usage": {"input_tokens": 10}}}
    assert proxy.write_cached_tokens_for_api("/responses", completed, 7) is True
    assert completed["response"]["usage"]["input_tokens_details"]["cached_tokens"] == 7

    untouched = {"type": "message"}
    assert proxy.write_cached_tokens_for_api("/messages", untouched, 1) is False
    assert "usage" not in untouched
    assert proxy.write_cached_tokens_for_api("/messages", {"usage": {}}, None) is False


def test_messages_passthrough_rejoins_split_json_and_releases_load(installed_runtime):
    scheduler, prefill, decode = installed_runtime
    raw = json.dumps(
        {
            "type": "message",
            "content": [{"type": "text", "text": "hello"}],
            "usage": {"cache_read_input_tokens": 0},
        }
    ).encode("utf-8")
    decode.chunks = [raw[:20], raw[20:]]

    async def run():
        response = await proxy.handle_completions_impl(
            "/messages",
            _request({"model": "m", "max_tokens": 80, "messages": [{"role": "user", "content": "hi"}]}),
        )
        assert response.media_type == "application/json"
        return json.loads(await _body(response))

    body = asyncio.run(run())

    assert body["content"][0]["text"] == "hello"
    assert body["usage"]["cache_read_input_tokens"] == 32
    assert "choices" not in body
    posted = prefill.posts[0]["json"]
    assert posted["max_tokens"] == 1
    assert posted["min_tokens"] == 1
    assert posted["kv_transfer_params"]["do_remote_decode"] is True
    assert len(prefill.posts) == 1
    assert scheduler.request_num == 0
    assert all(entry.active_kv_cache == 0 for entry in scheduler.prefillers.values())
    assert all(entry.active_tokens == 0 for entry in scheduler.decoders.values())


@pytest.mark.parametrize("event_type", ["message", "message_start", "message_delta"])
@pytest.mark.parametrize("old_cached, new_cached", [(100, 20), (100, 0), (0, 100), (None, 20)])
def test_messages_cached_usage_preserves_total_input_tokens(installed_runtime, event_type, old_cached, new_cached):
    scheduler, prefill, decode = installed_runtime
    prefill.payload["usage"]["cache_read_input_tokens"] = new_cached
    usage = {"input_tokens": 100 - (old_cached or 0), "cache_creation_input_tokens": 7, "output_tokens": 17}
    if old_cached is not None:
        usage["cache_read_input_tokens"] = old_cached
    event: dict[str, Any] = {"type": event_type}
    if event_type == "message_start":
        event["message"] = {"usage": usage}
    else:
        event["usage"] = usage
    streaming = event_type != "message"
    payload = json.dumps(event).encode()
    if streaming:
        payload = b"event: " + event_type.encode() + b"\ndata: " + payload + b"\n\n"
    decode.chunks = [payload[:20], payload[20:]]

    async def run():
        response = await proxy.handle_completions_impl("/messages", _request({"model": "m", "stream": streaming}))
        return await _body(response)

    body = asyncio.run(run())
    if streaming:
        assert body.startswith(b"event: " + event_type.encode() + b"\n")
        body = body.split(b"data: ", 1)[1].strip()
    patched = json.loads(body)
    patched_usage = patched["message"]["usage"] if event_type == "message_start" else patched["usage"]
    assert patched_usage["input_tokens"] == 100 - new_cached
    assert patched_usage["cache_read_input_tokens"] == new_cached
    assert patched_usage["cache_creation_input_tokens"] == 7
    assert patched_usage["output_tokens"] == 17
    assert (
        sum(patched_usage[key] for key in ("input_tokens", "cache_read_input_tokens", "cache_creation_input_tokens"))
        == 107
    )
    assert scheduler.request_num == 0


def test_responses_prefill_is_one_token_and_streaming_usage_is_patched(installed_runtime):
    scheduler, prefill, decode = installed_runtime
    delta = b'data: {"type":"response.output_text.delta","delta":"hi"}\n\n'
    completed = b'data: {"type":"response.completed","response":{"usage":{"input_tokens":10,"output_tokens":1}}}\n\n'
    decode.chunks = [delta, completed]
    request_body = {
        "model": "m",
        "input": "hello",
        "stream": True,
        "max_output_tokens": 500,
        "max_tokens": 500,
        "min_tokens": 2,
        "max_completion_tokens": 500,
        "stream_options": {"include_usage": True},
    }

    async def run():
        response = await proxy.handle_completions_impl("/responses", _request(request_body))
        assert response.media_type == "text/event-stream; charset=utf-8"
        return (await _body(response)).decode("utf-8")

    text = asyncio.run(run())

    posted = prefill.posts[0]["json"]
    assert posted["max_output_tokens"] == 1
    assert posted["stream"] is False
    for key in ("max_tokens", "min_tokens", "max_completion_tokens", "stream_options"):
        assert key not in posted
    assert request_body["max_output_tokens"] == 500
    assert '"delta": "hi"' in text or '"delta":"hi"' in text
    usage_line = text.strip().split("\n\n")[1]
    event = json.loads(usage_line.removeprefix("data: "))
    assert event["response"]["usage"]["input_tokens_details"]["cached_tokens"] == 7
    assert len(prefill.posts) == 1
    assert scheduler.request_num == 0


@pytest.mark.parametrize("api", ["/messages", "/responses"])
@pytest.mark.parametrize("chunking", ["split-utf8", "coalesced"])
def test_stream_usage_is_patched_across_http_chunk_boundaries(installed_runtime, api, chunking):
    scheduler, _prefill, decode = installed_runtime
    if api == "/messages":
        event = {"type": "message_start", "message": {"usage": {"cache_read_input_tokens": 0}}}
        expected_cached = 32
    else:
        event = {"type": "response.completed", "response": {"usage": {"input_tokens_details": {"cached_tokens": 0}}}}
        expected_cached = 7
    event["text"] = "你好"
    usage_frame = b"event: usage\r\ndata: " + json.dumps(event, ensure_ascii=False).encode() + b"\r\n\r\n"
    tail = b"event: ping\r\ndata: [DONE]\r\n\r\n"
    if chunking == "split-utf8":
        split = usage_frame.index("你".encode()) + 1
        decode.chunks = [usage_frame[:split], usage_frame[split:] + tail]
    else:
        decode.chunks = [usage_frame + tail]

    async def run():
        response = await proxy.handle_completions_impl(api, _request({"model": "m", "stream": True}))
        return await _body(response)

    body = asyncio.run(run())
    frame, done, _ = body.split(b"\r\n\r\n")
    assert frame.startswith(b"event: usage\r\n")
    patched = json.loads(frame.split(b"data: ", 1)[1])
    assert patched["text"] == "你好"
    if api == "/messages":
        assert patched["message"]["usage"]["cache_read_input_tokens"] == expected_cached
    else:
        assert patched["response"]["usage"]["input_tokens_details"]["cached_tokens"] == expected_cached
    assert done + b"\r\n\r\n" == tail
    assert scheduler.request_num == 0
    assert all(entry.active_tokens == 0 for entry in scheduler.decoders.values())


def test_messages_does_not_retry_when_body_has_no_choices(installed_runtime):
    scheduler, prefill, decode = installed_runtime
    decode.chunks = [json.dumps({"type": "message", "stop_reason": "recomputed", "content": "done"}).encode()]

    async def run():
        response = await proxy.handle_completions_impl(
            "/messages",
            _request({"model": "m", "messages": [{"role": "user", "content": "hi"}]}),
        )
        return json.loads(await _body(response))

    body = asyncio.run(run())

    assert body["stop_reason"] == "recomputed"
    assert body["content"] == "done"
    assert len(prefill.posts) == 1
    assert scheduler.request_num == 0


def test_incomplete_nonstream_tail_is_forwarded(installed_runtime):
    _scheduler_obj, _prefill, decode = installed_runtime
    decode.chunks = [b'{"type":"message"', b', "cut":']

    async def run():
        response = await proxy.handle_completions_impl("/messages", _request({"model": "m"}))
        return await _body(response)

    assert asyncio.run(run()) == b'{"type":"message", "cut":'


def test_chat_completions_still_uses_choices_and_openai_cached_tokens(installed_runtime):
    scheduler, prefill, decode = installed_runtime
    decode.chunks = [
        json.dumps(
            {
                "choices": [{"message": {"role": "assistant", "content": "hello"}, "stop_reason": "stop"}],
                "usage": {"completion_tokens": 1, "prompt_tokens_details": {"cached_tokens": 99}},
            }
        ).encode()
    ]

    async def run():
        response = await proxy.handle_completions_impl(
            "/chat/completions",
            _request({"model": "m", "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]}),
        )
        return json.loads(await _body(response))

    body = asyncio.run(run())

    assert body["choices"][0]["message"]["content"] == "hello"
    assert body["usage"]["prompt_tokens_details"]["cached_tokens"] == 4
    posted = prefill.posts[0]["json"]
    assert posted["max_tokens"] == 1
    assert "max_output_tokens" not in posted
    assert scheduler.request_num == 0


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize(
    "options, parameter",
    [
        ({"background": True}, "background"),
        ({"previous_response_id": "resp_from_decoder"}, "previous_response_id"),
        ({"previous_response_id": ""}, "previous_response_id"),
        ({"background": True, "previous_response_id": "resp_1"}, "background"),
    ],
)
def test_responses_rejects_unsupported_state_before_scheduling(installed_runtime, streaming, options, parameter):
    scheduler, prefill, decode = installed_runtime

    async def run():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=proxy.app), base_url="http://proxy") as client:
            return await client.post(
                "/v1/responses",
                headers={"X-Request-Id": "rejected-request"},
                json={"input": "hi", "stream": streaming, **options},
            )

    response = asyncio.run(run())
    assert response.status_code == 400
    assert response.headers["content-type"] == "application/json"
    error = response.json()["error"]
    assert error["type"] == "invalid_request_error"
    assert error["code"] == "unsupported_parameter"
    assert error["param"] == parameter
    assert parameter in error["message"]
    assert prefill.posts == decode.posts == []
    assert scheduler.request_num == 0
    assert all(entry.active_kv_cache == 0 for entry in scheduler.prefillers.values())
    assert all(entry.active_tokens == 0 for entry in scheduler.decoders.values())


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("affinity_enabled", [False, True])
@pytest.mark.parametrize(
    "headers, body_id, expected_id",
    [
        ({"X-Request-Id": "header-id"}, None, "header-id"),
        ({"X-Request-Id": "header-id"}, "body-id", "header-id"),
        ({}, "body-id", "body-id"),
    ],
)
def test_responses_preserves_request_id(installed_runtime, streaming, affinity_enabled, headers, body_id, expected_id):
    scheduler, prefill, decode = installed_runtime
    proxy.global_args.enable_prefill_affinity = affinity_enabled
    result = {"id": "resp_id", "output": []}
    payload = json.dumps({"type": "response.completed", "response": result} if streaming else result).encode()
    decode.chunks = [b"event: response.completed\ndata: " + payload + b"\n\n" if streaming else payload]

    async def run():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=proxy.app), base_url="http://proxy") as client:
            return await client.post(
                "/v1/responses",
                headers={**headers, "X-Session-Id": "header-session"},
                json={"input": "hi", "stream": streaming, "request_id": body_id, "user": "body-session"},
            )

    response = asyncio.run(run())
    assert response.status_code == 200
    assert len(prefill.posts) == len(decode.posts) == 1
    for posted in (prefill.posts[0], decode.posts[0]):
        assert posted["json"]["request_id"] == expected_id
        assert posted["headers"]["X-Request-Id"] == expected_id
        assert posted["json"]["user"] == ("header-session" if affinity_enabled else "body-session")
    assert scheduler.request_num == 0


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("options", [{}, {"background": False}, {"previous_response_id": None}])
@pytest.mark.parametrize("old_per_turn", [None, [], [100], [100, 11, 3]])
@pytest.mark.parametrize("prefill_cached", [0, 20])
def test_responses_reconciles_per_turn_usage(installed_runtime, streaming, options, old_per_turn, prefill_cached):
    scheduler, prefill, decode = installed_runtime
    prefill.payload["usage"]["input_tokens_details"]["cached_tokens"] = prefill_cached
    input_per_turn = ([160] if len(old_per_turn) == 1 else [120, 30, 10]) if old_per_turn else []
    output_per_turn = ([17] if len(old_per_turn) == 1 else [10, 4, 3]) if old_per_turn else []
    output_details = {"output_tokens_per_turn": output_per_turn}
    details = {
        "cached_tokens": sum(old_per_turn) if old_per_turn else 114,
        "input_tokens_per_turn": input_per_turn,
        "cache_write_tokens": 2,
    }
    if old_per_turn is not None:
        details["cached_tokens_per_turn"] = old_per_turn
    usage = {
        "input_tokens": 160,
        "output_tokens": 17,
        "total_tokens": 177,
        "input_tokens_details": details,
        "output_tokens_details": output_details,
    }
    result = {"id": "resp_new", "usage": usage, "output": []}
    payload = json.dumps({"type": "response.completed", "response": result} if streaming else result).encode()
    if streaming:
        payload = b"event: response.completed\ndata: " + payload + b"\n\n"
    decode.chunks = [payload[:23], payload[23:]]

    async def run():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=proxy.app), base_url="http://proxy") as client:
            return await client.post(
                "/v1/responses",
                json={"model": "m", "input": "hi", "stream": streaming, "max_output_tokens": 80, **options},
            )

    response = asyncio.run(run())
    assert response.status_code == 200
    if streaming:
        assert response.content.startswith(b"event: response.completed\n")
        patched = json.loads(response.content.split(b"data: ", 1)[1])["response"]
    else:
        patched = response.json()
    actual = patched["usage"]["input_tokens_details"]
    if old_per_turn is None:
        assert "cached_tokens_per_turn" not in actual
        assert actual["cached_tokens"] == prefill_cached
    elif not old_per_turn:
        assert actual["cached_tokens_per_turn"] == []
        assert actual["cached_tokens"] == prefill_cached
    else:
        expected = [prefill_cached, *old_per_turn[1:]]
        assert actual["cached_tokens_per_turn"] == expected
        assert actual["cached_tokens"] == sum(expected)
    assert actual["input_tokens_per_turn"] == input_per_turn
    assert patched["usage"]["output_tokens_details"] == output_details
    assert actual["cache_write_tokens"] == 2
    assert patched["usage"]["input_tokens"] == 160
    assert patched["usage"]["output_tokens"] == 17
    assert patched["usage"]["total_tokens"] == 177
    assert prefill.posts[0]["json"]["max_output_tokens"] == 1
    assert decode.posts[0]["json"]["max_output_tokens"] == 80
    assert decode.posts[0]["json"]["kv_transfer_params"]["remote_block_ids"] == [1]
    assert len(prefill.posts) == len(decode.posts) == 1
    assert scheduler.request_num == 0
    assert all(entry.active_kv_cache == 0 for entry in scheduler.prefillers.values())
    assert all(entry.active_tokens == 0 for entry in scheduler.decoders.values())
