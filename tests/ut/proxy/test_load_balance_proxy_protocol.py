"""Proxy protocol contracts, exercised through HTTP with mock P/D backends."""

import argparse
import asyncio
import json
from contextlib import AsyncExitStack
from typing import Any
from unittest.mock import Mock

import httpx
import pytest
from starlette.requests import ClientDisconnect

from examples.disaggregated_prefill_v1 import load_balance_proxy_server_example as proxy


class Body(httpx.AsyncByteStream):
    def __init__(self, data, error=None):
        self.data, self.error, self.closed = data, error, False

    async def __aiter__(self):
        # Split every byte, including UTF-8 and SSE delimiters.
        for byte in self.data:
            yield bytes([byte])
        if self.error:
            raise self.error

    async def aclose(self):
        self.closed = True


def wire(api, obj, streaming, *, omit_final_output=False, sequence_start=0):
    obj = dict(obj)
    text = obj.pop("text", None)
    if text is not None:
        obj.setdefault("usage", {"output_tokens": 1})
        if api == "/messages":
            obj.setdefault("content", [{"type": "text", "text": text}])
        else:
            obj.setdefault(
                "output",
                [{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": text}]}],
            )
    if not streaming:
        return json.dumps(obj, ensure_ascii=False, indent=2).encode()
    events: list[dict[str, Any]] = []
    if api == "/messages":
        events.append({"type": "message_start", "message": {"id": "msg", "content": []}})
        for index, block in enumerate(obj.get("content", [])):
            events.append({"type": "content_block_start", "index": index, "content_block": {**block, "text": ""}})
            if block.get("type") == "text":
                events.append(
                    {
                        "type": "content_block_delta",
                        "index": index,
                        "delta": {"type": "text_delta", "text": block["text"]},
                    }
                )
            events.append({"type": "content_block_stop", "index": index})
        events.append(
            {
                "type": "message_delta",
                "delta": {k: v for k, v in obj.items() if k not in ("content", "usage")},
                "usage": obj.get("usage"),
            }
        )
        events.append({"type": "message_stop"})
    else:
        response_id = obj.get("id", "resp")
        events.append({"type": "response.created", "response": {"id": response_id, "output": []}})
        for index, item in enumerate(obj.get("output", [])):
            events.append(
                {"type": "response.output_item.added", "output_index": index, "item": {**item, "content": []}}
            )
            for content_index, part in enumerate(item.get("content", [])):
                if part.get("type") != "output_text":
                    continue
                ids = {
                    "item_id": item.get("id", "item"),
                    "output_index": index,
                    "content_index": content_index,
                }
                events.extend(
                    [
                        {"type": "response.content_part.added", **ids, "part": {**part, "text": ""}},
                        {"type": "response.output_text.delta", **ids, "delta": part["text"]},
                        {"type": "response.output_text.done", **ids, "text": part["text"]},
                        {"type": "response.content_part.done", **ids, "part": part},
                    ]
                )
            events.append({"type": "response.output_item.done", "output_index": index, "item": item})
        terminal = "response.incomplete" if obj.get("status") == "incomplete" else "response.completed"
        final = {**obj, "output": []} if omit_final_output else obj
        events.append({"type": terminal, "response": final})
        for sequence_number, event in enumerate(events, sequence_start):
            event["sequence_number"] = sequence_number
    # Preserve comments, IDs, event names, multiline data, and CRLF framing.
    frames = []
    for event in events:
        data = b"\r\n".join(
            b"data: " + line for line in json.dumps(event, ensure_ascii=False, indent=2).encode().splitlines()
        )
        frames.append(
            b": keepalive\r\nid: event-1\r\nretry: 2000\r\nevent: "
            + event["type"].encode()
            + b"\r\n"
            + data
            + b"\r\n\r\n"
        )
    return b"".join(frames)


def sse_events(body: bytes) -> list[dict]:
    return [
        json.loads(b"\n".join(line[5:].lstrip() for line in frame.splitlines() if line.startswith(b"data:")))
        for frame in body.replace(b"\r\n", b"\n").split(b"\n\n")
        if b"data:" in frame
    ]


def response_text(api, response, streaming):
    if not streaming:
        obj = response.json()
        parts = (
            obj.get("content", [])
            if api == "/messages"
            else [part for item in obj.get("output", []) for part in item.get("content", [])]
        )
        return "".join(part.get("text", "") for part in parts)
    events = sse_events(response.content)
    if api == "/messages":
        assert sum(event["type"] == "message_start" for event in events) == 1
        assert sum(event["type"] == "message_stop" for event in events) == 1
        indexes = [event["index"] for event in events if event["type"] == "content_block_start"]
        assert indexes == list(range(len(indexes)))
        return "".join(
            event.get("content_block", {}).get("text", "") + event.get("delta", {}).get("text", "") for event in events
        )
    assert sum(event["type"] == "response.created" for event in events) == 1
    first_sequence = events[0]["sequence_number"]
    assert [event["sequence_number"] for event in events] == list(range(first_sequence, first_sequence + len(events)))
    indexes = [event["output_index"] for event in events if event["type"] == "response.output_item.added"]
    assert indexes == list(range(len(indexes)))
    text = "".join(event["delta"] for event in events if event["type"] == "response.output_text.delta")
    final = events[-1]["response"]
    assert text == "".join(part["text"] for item in final.get("output", []) for part in item.get("content", []))
    return text


@pytest.mark.parametrize("api", ["/chat/completions", "/completions", "/messages", "/responses"])
@pytest.mark.parametrize("disconnect", [False, True])
def test_client_receives_text_before_backend_finishes(monkeypatch, api, disconnect):
    """The backend cannot finish until ASGI sends text; also exercise a real disconnect."""

    async def run():
        gate, disconnected = asyncio.Event(), asyncio.Event()
        if api in ("/messages", "/responses"):
            raw = wire(api, {"text": "prefix", "stop_reason": "end_turn"}, True)
            cut = raw.index(b"\r\n\r\n", raw.index(b"prefix")) + 4
        else:
            choice: dict[str, Any] = (
                {"delta": {"content": "prefix"}} if api == "/chat/completions" else {"text": "prefix"}
            )
            raw = b"data: " + json.dumps({"choices": [choice]}).encode() + b"\n\ndata: [DONE]\n\n"
            cut = raw.index(b"\n\n") + 2

        class GatedBody(httpx.AsyncByteStream):
            closed = False

            async def __aiter__(self):
                yield raw[:cut]
                await gate.wait()
                yield raw[cut:]

            async def aclose(self):
                self.closed = True

        body = GatedBody()
        scheduler = proxy.SharedProxyScheduler([("127.0.0.1", 8100)], [("127.0.0.1", 8200)])
        runtime = proxy.WorkerRuntime(scheduler)
        monkeypatch.setattr(proxy, "runtime", runtime)
        monkeypatch.setattr(
            proxy, "global_args", argparse.Namespace(max_retries=2, retry_delay=0, enable_prefill_affinity=False)
        )

        async def backend(request):
            if request.url.host == "prefill":
                return httpx.Response(200, json={"kv_transfer_params": {"remote_block_ids": [1]}})
            return httpx.Response(200, stream=body)

        payload = {
            "model": "m",
            "stream": True,
            "max_tokens": 20,
            "max_output_tokens": 20,
            "messages": [{"role": "user", "content": "hi"}],
            "prompt": "hi",
            "input": "hi",
        }
        # Use only the input field belonging to the endpoint.
        field = {
            "/chat/completions": "messages",
            "/completions": "prompt",
            "/messages": "messages",
            "/responses": "input",
        }[api]
        for key in ("messages", "prompt", "input"):
            if key != field:
                payload.pop(key)
        requested = False
        sent = []

        async def receive():
            nonlocal requested
            if not requested:
                requested = True
                return {"type": "http.request", "body": json.dumps(payload).encode(), "more_body": False}
            await disconnected.wait()
            return {"type": "http.disconnect"}

        async def send(message):
            sent.append(message)
            if message["type"] == "http.response.body" and b"prefix" in message.get("body", b""):
                if disconnect:
                    disconnected.set()
                else:
                    gate.set()

        transport = httpx.MockTransport(backend)
        async with (
            httpx.AsyncClient(base_url="http://prefill/v1", transport=transport) as prefill,
            httpx.AsyncClient(base_url="http://decode/v1", transport=transport) as decode,
        ):
            runtime._clients[proxy.ServerRole.PREFILL][proxy.server_key("127.0.0.1", 8100)] = prefill
            runtime._clients[proxy.ServerRole.DECODE][proxy.server_key("127.0.0.1", 8200)] = decode
            scope = {
                "type": "http",
                "asgi": {"version": "3.0", "spec_version": "2.0"},
                "http_version": "1.1",
                "method": "POST",
                "scheme": "http",
                "path": "/v1" + api,
                "query_string": b"",
                "headers": [(b"content-type", b"application/json")],
                "client": ("127.0.0.1", 1),
                "server": ("127.0.0.1", 2),
            }
            await asyncio.wait_for(proxy.app(scope, receive, send), timeout=3)
        assert gate.is_set() != disconnect
        assert any(b"prefix" in message.get("body", b"") for message in sent)
        assert body.closed
        assert scheduler.request_num == 0
        assert all(entry.active_kv_cache == 0 for entry in scheduler.prefillers.values())
        assert all(entry.active_tokens == 0 for entry in scheduler.decoders.values())

    asyncio.run(run())


@pytest.fixture
def exchange(monkeypatch):
    """One real httpx client per backend, with load cleanup checked on every path."""

    async def run(
        api,
        payload,
        scripts,
        *,
        headers=None,
        prefill_error=None,
        affinity=False,
        max_retries=2,
        reselect=False,
        disconnect_at_headers=False,
    ):
        payload = dict(payload)
        if api == "/messages":
            payload.setdefault("messages", [{"role": "user", "content": "hi"}])
        elif api == "/responses":
            payload.setdefault("input", "hi")
        calls: dict[str, list[dict]] = {"prefill": [], "decode": []}
        host = "127.0.0.1"
        scheduler = proxy.SharedProxyScheduler([(host, 8100)], [(host, 8200)])
        runtime = proxy.WorkerRuntime(scheduler)
        monkeypatch.setattr(proxy, "runtime", runtime)
        monkeypatch.setattr(
            proxy,
            "global_args",
            argparse.Namespace(max_retries=max_retries, retry_delay=0, enable_prefill_affinity=affinity),
        )
        release = Mock(wraps=scheduler._release_load)
        if reselect:
            monkeypatch.setattr(scheduler, "_release_load", release)

        async def backend(request):
            role, port = request.url.host, request.url.port
            if reselect:
                assert scheduler.request_num == 1 + bool(calls["decode"])
            calls[role].append(
                {
                    "body": json.loads(request.content),
                    "headers": request.headers,
                    "path": request.url.path,
                    "port": port,
                }
            )
            if role == "prefill":
                if calls["decode"]:
                    # Close the abandoned decode before starting another P/D attempt.
                    assert scripts[len(calls["decode"]) - 1].is_closed
                if prefill_error and len(calls[role]) >= 2:
                    raise prefill_error
                cached = 2 if port == 8100 else 3
                return httpx.Response(
                    200,
                    json={
                        "kv_transfer_params": {"remote_block_ids": [len(calls[role])]},
                        "usage": {"cache_read_input_tokens": cached, "input_tokens_details": {"cached_tokens": cached}},
                    },
                )
            if reselect and len(calls["decode"]) == 1:
                # Leave the original P/D busy with an unrelated request.
                scheduler.begin_request(1000)
                scheduler.pick_decoder(1000)
                scheduler.activate_waiting_instance(proxy.ServerRole.PREFILL, host, 8101)
                scheduler.activate_waiting_instance(proxy.ServerRole.DECODE, host, 8201)
            script = scripts[len(calls[role]) - 1]
            if isinstance(script, BaseException):
                raise script
            return script

        async def app(scope, receive, send):
            # ASGI 2.4 propagates backend cancellation without the older disconnect task.
            scope["asgi"]["spec_version"] = "2.4"
            if disconnect_at_headers:

                async def disconnect(message):
                    assert message["type"] == "http.response.start"
                    raise OSError("client disconnected before the response body")

                await proxy.app(scope, receive, disconnect)
            else:
                await proxy.app(scope, receive, send)

        transport = httpx.MockTransport(backend)
        async with AsyncExitStack() as stack:
            for role, port in ((proxy.ServerRole.PREFILL, 8100), (proxy.ServerRole.DECODE, 8200)):
                for selected_port in (port, port + 1) if reselect else (port,):
                    backend_client = await stack.enter_async_context(
                        httpx.AsyncClient(base_url=f"http://{role.value}:{selected_port}/v1", transport=transport)
                    )
                    runtime._clients[role][proxy.server_key(host, selected_port)] = backend_client
            client = await stack.enter_async_context(
                httpx.AsyncClient(base_url="http://proxy", transport=httpx.ASGITransport(app=app))
            )
            try:
                response = await client.post("/v1" + api, json=payload, headers=headers)
            finally:
                if reselect:
                    assert scheduler.request_num == 1  # Only the unrelated request remains.
                    expected = [
                        proxy.server_key(host, call["port"])
                        for attempt in zip(calls["prefill"], calls["decode"], strict=True)
                        for call in attempt
                    ]
                    assert [call.args[1] for call in release.call_args_list] == expected
                    scheduler.finish_request(
                        proxy.server_key(host, 8100), 1000, proxy.server_key(host, 8200), 1000, release_prefill_kv=True
                    )
                assert scheduler.request_num == 0
                assert all(entry.active_kv_cache == pytest.approx(0) for entry in scheduler.prefillers.values())
                assert all(entry.active_tokens == pytest.approx(0) for entry in scheduler.decoders.values())
        return response, calls

    return lambda *args, **kwargs: asyncio.run(run(*args, **kwargs))


@pytest.mark.parametrize("api", ["/chat/completions", "/completions", "/messages", "/responses"])
@pytest.mark.parametrize("fragmented", [False, True])
def test_shared_frame_path_never_duplicates_or_drops_events(exchange, api, fragmented):
    payload: dict[str, Any] = {"stream": True, "messages": [{"role": "user", "content": "hi"}]}
    if api == "/completions":
        payload = {"stream": True, "prompt": "hi"}
    if api in ("/messages", "/responses"):
        raw = wire(api, {"text": "你好", "stop_reason": "end_turn", "usage": None}, True)
    else:
        frames = []
        for text in ("你", "好"):
            choice: dict[str, Any] = {"delta": {"content": text}} if api == "/chat/completions" else {"text": text}
            frames.append(b"data: " + json.dumps({"choices": [choice]}, ensure_ascii=False).encode() + b"\n\n")
        raw = b"".join(frames) + b"data: [DONE]\n\n"
    script = httpx.Response(200, stream=Body(raw)) if fragmented else httpx.Response(200, content=raw)
    response, _ = exchange(api, payload, [script])
    assert response.status_code == 200
    assert response.content == raw


@pytest.mark.parametrize("api", ["/chat/completions", "/completions", "/messages", "/responses"])
@pytest.mark.parametrize("stage", ["prefill", "decode"])
@pytest.mark.parametrize("failure", ["connect", "read", "http"])
def test_transport_retries_are_shared_across_apis(api, stage, failure):
    async def run():
        calls = []
        final = b'{"text":"done"}'

        async def backend(request):
            calls.append(json.loads(request.content))
            if len(calls) == 1:
                if failure == "connect":
                    raise httpx.ConnectError("connection failed")
                if failure == "read":
                    return httpx.Response(200, stream=Body(b"", httpx.ReadError("failed before first byte")))
                return httpx.Response(503, json={"error": "unavailable"})
            return httpx.Response(200, content=final)

        async with httpx.AsyncClient(base_url="http://backend/v1", transport=httpx.MockTransport(backend)) as client:
            args = (client, api, {"model": "m", "stream": False}, "request-id")
            if stage == "prefill":
                response = await proxy.send_request_to_service(*args, max_retries=2, base_delay=0)
                body = response.content
            else:
                body = b"".join(
                    [chunk async for chunk in proxy.stream_service_response(*args, max_retries=2, base_delay=0)]
                )
        assert body == final
        assert len(calls) == 2
        assert calls[0] == calls[1]

    asyncio.run(run())


@pytest.mark.parametrize("api", ["/chat/completions", "/completions", "/messages", "/responses"])
def test_all_protocols_reassign_after_recompute(exchange, api):
    payload = {"model": "m", "stream": True}
    if api in ("/chat/completions", "/messages"):
        payload.update(messages=[{"role": "user", "content": "hi"}], max_tokens=80)
    elif api == "/completions":
        payload.update(prompt="hi", max_tokens=80)
    else:
        payload.update(input="hi", max_output_tokens=80)

    bodies = []
    for text, stop_reason in [("part", "recomputed"), ("done", None)]:
        result: dict[str, Any] = {"stop_reason": stop_reason}
        if api == "/chat/completions":
            result["delta"] = {"content": text}
        else:
            result["text"] = text
        if api in ("/chat/completions", "/completions"):
            raw = json.dumps({"choices": [result], "usage": {"completion_tokens": 1}}).encode()
            bodies.append(b"data: " + raw + b"\n\n")
        else:
            bodies.append(wire(api, result, True))

    response, calls = exchange(api, payload, [httpx.Response(200, content=body) for body in bodies])
    assert response.status_code == 200
    assert b"recomputed" not in response.content
    assert b"done" in response.content
    assert len(calls["prefill"]) == len(calls["decode"]) == 2
    first, second = [call["body"] for call in calls["decode"]]
    assert first.pop("kv_transfer_params") == {"remote_block_ids": [1]}
    assert second.pop("kv_transfer_params") == {"remote_block_ids": [2]}
    assert first == payload
    if api == "/chat/completions":
        assert second == {**payload, "messages": [{"role": "user", "content": "hipart"}]}
    elif api == "/completions":
        assert second == {**payload, "prompt": "hipart"}
    elif api == "/messages":
        assert second == {**payload, "messages": [{"role": "user", "content": "hipart"}], "max_tokens": 79}
        assert response_text(api, response, True) == "partdone"
    else:
        assert second == {**payload, "input": "hipart", "max_output_tokens": 79}
        assert response_text(api, response, True) == "partdone"
    for role in calls.values():
        assert all(call["path"] == "/v1" + api for call in role)


@pytest.mark.parametrize("api", ["/messages", "/responses"])
@pytest.mark.parametrize("streaming", [False, True])
def test_native_passthrough_and_prefill_payload(exchange, api, streaming):
    result: dict[str, Any] = {
        "id": "native-id",
        "stop_reason": "end_turn",
        "text": "你好",
        "content": [{"type": "thinking", "thinking": "reason"}, {"type": "tool_use", "id": "tool-1", "input": {}}],
        "usage": {
            "input_tokens": 10,
            "cache_read_input_tokens": 99,
            "input_tokens_details": {"cached_tokens": 114, "cached_tokens_per_turn": [100, 11, 3]},
        },
    }
    raw = wire(api, result, streaming)
    stream = Body(raw)
    payload = {
        "model": "m",
        "messages": [{"role": "user", "content": "hi"}],
        "input": "hi",
        "stream": streaming,
        "max_tokens": 80,
        "max_completion_tokens": 80,
        "max_output_tokens": 80,
        "stream_options": {"include_usage": True},
        "vllm_xargs": {"existing_option": 1},
    }
    response, calls = exchange(api, payload, [httpx.Response(200, stream=stream)])
    assert response.status_code == 200
    expected_usage = dict(result["usage"])
    if api == "/messages":
        expected_usage.update(input_tokens=107, cache_read_input_tokens=2)
    else:
        expected_usage["input_tokens_details"] = {"cached_tokens": 16, "cached_tokens_per_turn": [2, 11, 3]}
    if streaming:
        expected_events = sse_events(raw)
        terminal = expected_events[-2] if api == "/messages" else expected_events[-1]["response"]
        terminal["usage"] = expected_usage
        assert sse_events(response.content) == expected_events
        for field in (b": keepalive\r\n", b"id: event-1\r\n", b"retry: 2000\r\n"):
            assert response.content.count(field) == raw.count(field)
    else:
        expected_result = json.loads(raw)
        expected_result["usage"] = expected_usage
        assert response.json() == expected_result
    assert response.headers["content-type"].startswith("text/event-stream" if streaming else "application/json")
    assert stream.closed
    assert len(calls["prefill"]) == len(calls["decode"]) == 1
    prefill, decode = calls["prefill"][0]["body"], calls["decode"][0]["body"]
    assert prefill["stream"] is False
    assert "stream_options" not in prefill
    if api == "/responses":
        assert prefill["max_output_tokens"] == 1
        assert not {"max_tokens", "min_tokens", "max_completion_tokens"} & prefill.keys()
    else:
        assert prefill["max_tokens"] == prefill["min_tokens"] == prefill["max_completion_tokens"] == 1
    expected = {**payload, "kv_transfer_params": {"remote_block_ids": [1]}}
    if not streaming:
        expected.pop("stream_options")
    assert decode == expected
    assert prefill["kv_transfer_params"]["do_remote_decode"] is True
    assert calls["prefill"][0]["path"] == calls["decode"][0]["path"] == "/v1" + api


@pytest.mark.parametrize("api", ["/messages", "/responses"])
@pytest.mark.parametrize("streaming", [False, True])
def test_recompute_appends_text_and_reduces_budget(exchange, api, streaming):
    # Responses needs a backend extension to expose this marker; upstream omits it.
    stopped = {"stop_reason": "recomputed", "text": "prefix", "usage": {"output_tokens": 7}}
    if api == "/messages":
        stopped.update(stop_reason="stop_sequence", stop_sequence="recomputed")
    final = {"stop_reason": "end_turn", "text": "done"}
    first = Body(wire(api, stopped, streaming))
    last = Body(wire(api, final, streaming))
    payload = {
        "model": "m",
        "stream": streaming,
        "input": "hi",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 80,
        "max_output_tokens": 80,
    }
    response, calls = exchange(api, payload, [httpx.Response(200, stream=first), httpx.Response(200, stream=last)])
    assert response.status_code == 200
    assert response_text(api, response, streaming) == "prefixdone"
    assert first.closed and last.closed
    assert len(calls["prefill"]) == len(calls["decode"]) == 2
    for index, attempt in enumerate(calls["decode"], 1):
        expected = {**payload, "kv_transfer_params": {"remote_block_ids": [index]}}
        if index == 2:
            if api == "/messages":
                expected.update(messages=[{"role": "user", "content": "hiprefix"}], max_tokens=73)
            else:
                expected.update(input="hiprefix", max_output_tokens=73)
        assert attempt["body"] == expected


@pytest.mark.parametrize("api", ["/messages", "/responses"])
def test_recompute_in_unterminated_final_event(exchange, api):
    stopped = wire(api, {"stop_reason": "recomputed"}, True).rstrip(b"\r\n")
    final = wire(api, {"text": "done"}, True)
    response, calls = exchange(
        api, {"stream": True}, [httpx.Response(200, content=stopped), httpx.Response(200, content=final)]
    )
    assert response.status_code == 200
    assert response_text(api, response, True) == "done"
    assert len(calls["prefill"]) == len(calls["decode"]) == 2


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize(
    "api, input_form",
    [
        ("/messages", "string-content"),
        ("/messages", "list-content"),
        ("/responses", "string-content"),
        ("/responses", "list-content"),
        ("/responses", "string-input"),
    ],
)
def test_multiple_recomputes_preserve_history_and_accumulate_text(exchange, api, streaming, input_form):
    array_content = input_form == "list-content"
    part_type = "text" if api == "/messages" else "input_text"
    content = [{"type": part_type, "text": "hi"}] if array_content else "hi"
    history: list[dict[str, Any]] = [{"role": "user", "content": content}, {"role": "assistant", "content": "history"}]
    field = "messages" if api == "/messages" else "input"
    limit = "max_tokens" if api == "/messages" else "max_output_tokens"
    initial_input = content if input_form == "string-input" else history
    payload = {"stream": streaming, field: initial_input, limit: 20}
    results: list[dict[str, Any]] = [
        {"id": "reply-0", "text": "你", "stop_reason": "recomputed", "usage": {"output_tokens": 2}},
        {"id": "reply-1", "text": "好", "stop_reason": "recomputed", "usage": {"output_tokens": 3}},
        {
            "id": "reply-2",
            "text": "！",
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 1, "total_tokens": 11},
        },
    ]
    if api == "/messages":
        results[0].update(stop_reason="stop_sequence", stop_sequence="recomputed")
    else:
        results[0]["status"] = "incomplete"
        for i, result in enumerate(results):
            result["output"] = [
                {
                    "id": f"item-{i}",
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": result["text"]}],
                }
            ]
    first_sequence = 10 if api == "/responses" and not array_content else 0
    replies = [
        httpx.Response(
            200,
            stream=Body(
                wire(
                    api,
                    obj,
                    streaming,
                    omit_final_output=api == "/responses" and i == 0 and array_content,
                    sequence_start=first_sequence if i == 0 else 0,
                )
            ),
        )
        for i, obj in enumerate(results)
    ]
    # The transport retry limit does not cap recompute continuation.
    response, calls = exchange(api, payload, replies, max_retries=1, reselect=True)
    assert response.status_code == 200
    assert response_text(api, response, streaming) == "你好！"
    assert [call["port"] for call in calls["prefill"]] == [8100, 8101, 8101]
    assert [call["port"] for call in calls["decode"]] == [8200, 8201, 8201]
    attempts = zip(calls["prefill"], calls["decode"], ("", "你", "你好"), (20, 18, 15), strict=True)
    for i, (prefill, attempt, prefix, budget) in enumerate(attempts, 1):
        expected: str | list[dict[str, str]]
        if isinstance(content, list):
            expected = content + ([{"type": content[0]["type"], "text": prefix}] if prefix else [])
        else:
            expected = content + prefix
        expected_history: list[dict[str, Any]] = [{"role": "user", "content": expected}, history[1]]
        expected_input = expected if input_form == "string-input" else expected_history
        assert prefill["body"][field] == attempt["body"][field] == expected_input
        assert prefill["body"][limit] == 1
        assert attempt["body"][limit] == budget
        assert attempt["body"]["kv_transfer_params"] == {"remote_block_ids": [i]}
    if streaming:
        events = sse_events(response.content)
        for sse_field in (b": keepalive\r\n", b"id: event-1\r\n", b"retry: 2000\r\n"):
            assert response.content.count(sse_field) == len(events)
        final = events[-2] if api == "/messages" else events[-1]["response"]
        if api == "/responses":
            assert events[0]["sequence_number"] == first_sequence
            assert final["id"] == events[0]["response"]["id"] == "reply-0"
    else:
        final = response.json()
    # The last decode attempt uses the newly selected prefiller's cache count (3).
    expected_usage: dict[str, Any] = dict(results[-1]["usage"])
    if api == "/messages":
        expected_usage.update(input_tokens=7, cache_read_input_tokens=3)
    else:
        expected_usage["input_tokens_details"] = {"cached_tokens": 3}
        assert [item["id"] for item in final["output"]] == ["item-0", "item-1", "item-2"]
    assert final["usage"] == expected_usage


@pytest.mark.parametrize("api", ["/messages", "/responses"])
@pytest.mark.parametrize("streaming", [False, True])
def test_recompute_leaves_budget_validation_to_backend(exchange, caplog, api, streaming):
    limit = "max_tokens" if api == "/messages" else "max_output_tokens"
    response, calls = exchange(
        api,
        {"stream": streaming, limit: 2},
        [
            httpx.Response(
                200,
                content=wire(
                    api, {"text": "done", "stop_reason": "recomputed", "usage": {"output_tokens": 2}}, streaming
                ),
            ),
            httpx.Response(400, json={"error": {"message": "output token limit must be positive"}}),
        ],
    )
    assert response.status_code == 200
    if streaming:
        assert b"message_stop" not in response.content
        assert b"response.completed" not in response.content
    else:
        assert response.content == b""
    assert len(calls["decode"]) == len(calls["prefill"]) == 2
    assert calls["decode"][1]["body"][limit] == 0
    assert "Error during streaming" in caplog.text


@pytest.mark.parametrize("api", ["/messages", "/responses"])
@pytest.mark.parametrize("streaming", [False, True])
def test_recompute_requires_usage_for_explicit_budget(exchange, caplog, api, streaming):
    limit = "max_tokens" if api == "/messages" else "max_output_tokens"
    result = {"text": "partial", "stop_reason": "recomputed", "usage": None}
    response, calls = exchange(
        api, {limit: 10, "stream": streaming}, [httpx.Response(200, content=wire(api, result, streaming))]
    )
    assert response.status_code == 200
    assert len(calls["prefill"]) == len(calls["decode"]) == 1
    assert "valid output_tokens count" in caplog.text
    if streaming:
        assert b"message_stop" not in response.content
        assert b"response.completed" not in response.content
    else:
        assert response.content == b""


@pytest.mark.parametrize("streaming", [False, True])
def test_responses_continuation_only_prefixes_first_text_block(exchange, streaming):
    first = {"text": "prefix", "stop_reason": "recomputed"}
    final = {
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "one"}, {"type": "output_text", "text": "two"}],
            }
        ]
    }
    response, _ = exchange(
        "/responses",
        {"stream": streaming},
        [httpx.Response(200, content=wire("/responses", obj, streaming)) for obj in (first, final)],
    )
    assert response.status_code == 200
    assert response_text("/responses", response, streaming) == "prefixonetwo"


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("response_input", ["", []])
def test_responses_recompute_with_empty_input(exchange, streaming, response_input):
    response, calls = exchange(
        "/responses",
        {"input": response_input, "stream": streaming},
        [
            httpx.Response(200, content=wire("/responses", obj, streaming))
            for obj in ({"text": "prefix", "stop_reason": "recomputed"}, {"text": ""})
        ],
    )
    assert response.status_code == 200
    assert calls["decode"][1]["body"]["input"] == "prefix"
    assert response_text("/responses", response, streaming) == "prefix"
    assert all("max_output_tokens" not in call["body"] for call in calls["decode"])


@pytest.mark.parametrize(
    "first_item",
    [
        {"type": "function_call_output", "call_id": "call", "output": "ok"},
        {
            "type": "reasoning",
            "id": "reasoning",
            "summary": [],
            "content": [{"type": "reasoning_text", "text": "context"}],
        },
    ],
)
def test_responses_recompute_does_not_append_text_to_non_message_input(exchange, caplog, first_item):
    response, calls = exchange(
        "/responses",
        {"input": [first_item]},
        [httpx.Response(200, content=wire("/responses", {"text": "prefix", "stop_reason": "recomputed"}, False))],
    )
    assert response.content == b""
    assert "first input message with content" in caplog.text
    assert len(calls["decode"]) == 1


@pytest.mark.parametrize("api", ["/messages", "/responses"])
@pytest.mark.parametrize("streaming", [False, True])
def test_recompute_stops_on_nontext_output(exchange, caplog, api, streaming):
    stopped: dict[str, Any] = {"stop_reason": "recomputed"}
    if api == "/messages":
        stopped["content"] = [{"type": "tool_use", "id": "tool", "name": "f", "input": {}}]
    else:
        stopped["output"] = [{"type": "function_call", "id": "tool", "name": "f", "arguments": "{}"}]
    response, calls = exchange(api, {"stream": streaming}, [httpx.Response(200, content=wire(api, stopped, streaming))])
    assert response.status_code == 200
    assert "text-only" in caplog.text
    if streaming:
        assert b"message_stop" not in response.content
        assert b"response.completed" not in response.content
    else:
        assert response.content == b""
    assert len(calls["decode"]) == 1


def test_recompute_stops_after_streamed_reasoning_with_empty_output(exchange, caplog):
    result = {
        "status": "incomplete",
        "stop_reason": "recomputed",
        "output": [{"type": "reasoning", "id": "reasoning", "summary": []}],
    }
    reply = httpx.Response(200, content=wire("/responses", result, True, omit_final_output=True))
    response, calls = exchange("/responses", {"stream": True}, [reply])
    assert len(calls["prefill"]) == len(calls["decode"]) == 1
    assert "text-only output" in caplog.text
    assert sse_events(response.content)[-1]["type"] == "response.output_item.done"


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize(
    "api, stopped, options, retry",
    [
        ("/messages", {"stop_reason": "recomputed"}, {}, True),
        ("/messages", {"stop_reason": "stop_sequence", "stop_sequence": "recomputed"}, {}, True),
        (
            "/messages",
            {"stop_reason": "stop_sequence", "stop_sequence": "recomputed"},
            {"stop_sequences": ["recomputed"]},
            False,
        ),
        ("/messages", {"stop_reason": "end_turn", "content": [{"text": "recomputed"}]}, {}, False),
        ("/responses", {"stop_reason": "recomputed"}, {}, True),
        ("/responses", {"status": "completed", "output": [{"stop_reason": "recomputed"}]}, {}, False),
        ("/responses", {"status": "incomplete", "incomplete_details": {"reason": "max_output_tokens"}}, {}, False),
    ],
)
def test_recompute_only_uses_protocol_stop_fields(exchange, api, stopped, options, retry, streaming):
    original, final = wire(api, stopped, streaming), wire(api, {"text": "done"}, streaming)
    response, calls = exchange(
        api,
        {"stream": streaming, **options},
        [httpx.Response(200, content=original), httpx.Response(200, content=final)],
    )
    if retry:
        assert response_text(api, response, streaming) == "done"
    else:
        assert response.content == original
    assert len(calls["decode"]) == len(calls["prefill"]) == (2 if retry else 1)


@pytest.mark.parametrize("streaming", [False, True])
def test_responses_incomplete_uses_prefill_cache_without_retry(exchange, streaming):
    result = {
        "id": "resp",
        "status": "incomplete",
        "output": [],
        "incomplete_details": {"reason": "max_output_tokens"},
        "usage": {"input_tokens_details": {"cached_tokens": 10, "cached_tokens_per_turn": [7, 3]}},
    }
    reply = httpx.Response(200, content=wire("/responses", result, streaming))
    response, calls = exchange("/responses", {"stream": streaming}, [reply])
    assert len(calls["prefill"]) == len(calls["decode"]) == 1
    final = sse_events(response.content)[-1]["response"] if streaming else response.json()
    assert final["status"] == "incomplete"
    assert final["incomplete_details"] == result["incomplete_details"]
    assert final["usage"]["input_tokens_details"] == {"cached_tokens": 5, "cached_tokens_per_turn": [2, 3]}


@pytest.mark.parametrize("api", ["/messages", "/responses"])
@pytest.mark.parametrize("status", [200, 400, 500])
def test_backend_error_uses_existing_retry_policy(exchange, api, status):
    streaming = status == 200
    raw = (
        b'event: error\ndata: {"type":"error","message":"failed"}\n\n'
        if streaming
        else b'{"error":{"message":"failed"}}'
    )
    attempts = 2 if status == 500 else 1
    response, calls = exchange(
        api, {"stream": streaming}, [httpx.Response(status, content=raw) for _ in range(attempts)]
    )
    assert response.status_code == status
    if streaming:
        assert response.content == raw
    else:
        assert response.json() == json.loads(raw)
    assert len(calls["prefill"]) == 1
    assert len(calls["decode"]) == attempts


@pytest.mark.parametrize("api", ["/messages", "/responses"])
@pytest.mark.parametrize("read_error", [False, True])
def test_stream_interruption_preserves_existing_close_behavior(exchange, api, read_error):
    raw = wire(api, {"text": "prefix", "stop_reason": "end_turn"}, True)
    cut = raw.index(b"\r\n\r\n", raw.index(b"prefix")) + 4
    body = Body(raw[:cut], httpx.ReadError("connection lost") if read_error else None)
    response, calls = exchange(api, {"stream": True}, [httpx.Response(200, stream=body)])
    assert response.status_code == 200
    assert b"prefix" in response.content
    assert response.content == raw[:cut]
    assert len(calls["decode"]) == 1
    assert body.closed


@pytest.mark.parametrize("api", ["/messages", "/responses"])
@pytest.mark.parametrize("streaming", [False, True])
def test_interrupted_decode_closes_backend_without_transport_retry(exchange, api, streaming):
    stream = Body(b'{"partial":', httpx.ReadError("connection lost"))
    response, calls = exchange(api, {"stream": streaming}, [httpx.Response(200, stream=stream)])
    assert response.status_code == 200
    assert response.content == b""
    assert stream.closed
    assert len(calls["decode"]) == 1


def test_incomplete_json_uses_existing_log_and_close_behavior(exchange, caplog):
    response, _ = exchange("/messages", {}, [httpx.Response(200, content=b'{"partial":')])
    assert response.status_code == 200
    assert response.content == b""
    assert "Error during streaming" in caplog.text


def test_releases_preopened_decode_on_client_disconnect(exchange):
    reply = httpx.Response(200, stream=Body(wire("/responses", {"text": "first"}, True)))
    with pytest.raises(ClientDisconnect):
        exchange("/responses", {"stream": True}, [reply], disconnect_at_headers=True)
    assert reply.is_closed


@pytest.mark.parametrize("stage", ["decode", "retry_prefill"])
@pytest.mark.parametrize("error", [asyncio.CancelledError, httpx.ReadError])
def test_recompute_failure_releases_load(exchange, stage, error):
    stream = Body(
        wire("/messages", {"stop_reason": "recomputed"}, False), error("interrupted") if stage == "decode" else None
    )
    kwargs = {"prefill_error": error("interrupted")} if stage == "retry_prefill" else {}
    args = ("/messages", {"stream": False}, [httpx.Response(200, stream=stream)])
    if error is asyncio.CancelledError:
        with pytest.raises(asyncio.CancelledError):
            exchange(*args, **kwargs)
    else:
        response, _ = exchange(*args, **kwargs)
        assert response.status_code == 200
        assert response.content == b""
    assert stream.closed


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize(
    "options, parameter",
    [
        ({"background": True}, "background"),
        ({"previous_response_id": "resp_other"}, "previous_response_id"),
        ({"previous_response_id": ""}, "previous_response_id"),
        ({"background": True, "previous_response_id": "resp_other"}, "background"),
    ],
)
def test_unsupported_response_state_never_reaches_backends(exchange, streaming, options, parameter):
    response, calls = exchange("/responses", {"stream": streaming, **options}, [])
    assert response.status_code == 400
    assert f"does not support {parameter}" in response.json()["detail"]
    assert calls == {"prefill": [], "decode": []}


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("affinity", [False, True])
@pytest.mark.parametrize(
    "header_id, body_id, expected", [("header", None, "header"), ("header", "body", "header"), (None, "body", "body")]
)
def test_response_request_identity_and_explicit_history(exchange, streaming, affinity, header_id, body_id, expected):
    headers = {"X-Session-Id": "session"}
    if header_id:
        headers["X-Request-Id"] = header_id
    payload = {
        "input": [{"role": "user", "content": "hi"}],
        "stream": streaming,
        "request_id": body_id,
        "user": "body-session",
        "background": False,
        "previous_response_id": None,
    }
    response, calls = exchange(
        "/responses",
        payload,
        [httpx.Response(200, content=wire("/responses", {}, streaming))],
        headers=headers,
        affinity=affinity,
    )
    assert response.status_code == 200
    for role in calls.values():
        assert role[0]["body"]["request_id"] == role[0]["headers"]["X-Request-Id"] == expected
        assert role[0]["body"]["user"] == ("session" if affinity else "body-session")
        assert role[0]["body"]["input"] == payload["input"]
