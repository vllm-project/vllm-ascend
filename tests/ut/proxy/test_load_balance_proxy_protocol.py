"""Regression tests for PD proxy fairness and Messages/Responses forwarding."""

import argparse
import asyncio
import json
from typing import Any

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
    proxy.global_args = argparse.Namespace(max_retries=1, retry_delay=0.0, workers=1, log_level="ERROR")
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


def test_heap_tie_break_uses_push_counter_not_ordinal():
    scheduler = _scheduler()
    heap = scheduler._pool(proxy.ServerRole.PREFILL).heap
    counters = [item[1] for item in heap]
    ordinals = [scheduler.prefillers[item[3]].ordinal for item in heap]
    assert counters == [1, 2]
    assert ordinals == [0, 1]
    assert counters != ordinals


def test_equal_priority_rotates_across_prefillers_and_decoders():
    scheduler = _scheduler()

    def cycle(pick, release):
        first = pick(50)
        release(first["key"], 50)
        second = pick(50)
        release(second["key"], 50)
        third = pick(50)
        release(third["key"], 50)
        assert first["port"] != second["port"]
        assert third["port"] == first["port"]
        return first, second

    cycle(scheduler.begin_request, scheduler.release_prefill_kv)
    cycle(scheduler.pick_decoder, scheduler.release_decoder)
    assert scheduler.request_num == 3
    for entry in scheduler.prefillers.values():
        assert entry.active_kv_cache == 0
    for entry in scheduler.decoders.values():
        assert entry.active_tokens == 0


def test_lower_load_wins_when_priorities_differ():
    scheduler = _scheduler()
    busy = scheduler.begin_request(1000)
    idle = scheduler.begin_request(10)
    assert busy["port"] == 8100
    assert idle["port"] == 8101
    assert scheduler.prefillers[busy["key"]].active_kv_cache == 1000
    assert scheduler.prefillers[idle["key"]].active_kv_cache == 10

    busy_decoder = scheduler.pick_decoder(1000)
    idle_decoder = scheduler.pick_decoder(10)
    assert busy_decoder["port"] == 8200
    assert idle_decoder["port"] == 8201


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
