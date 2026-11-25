"""Proxy protocol contracts, exercised through HTTP with mock P/D backends."""

import argparse
import asyncio
import json

import httpx
import pytest

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


def wire(api, obj, streaming):
    if not streaming:
        return json.dumps(obj, ensure_ascii=False, indent=2).encode()
    event = (
        {"type": "message_delta", "delta": obj}
        if api == "/messages"
        else {
            "type": "response.completed",
            "response": obj,
        }
    )
    # Preserve comments, IDs, event names, multiline data, and CRLF framing.
    data = b"\r\n".join(
        b"data: " + line for line in json.dumps(event, ensure_ascii=False, indent=2).encode().splitlines()
    )
    return b": keepalive\r\nid: event-1\r\nevent: " + event["type"].encode() + b"\r\n" + data + b"\r\n\r\n"


@pytest.fixture
def exchange(monkeypatch):
    """One real httpx client per backend, with load cleanup checked on every path."""

    async def run(api, payload, scripts, *, headers=None, prefill_error=None, affinity=False):
        calls = {"prefill": [], "decode": []}
        scheduler = proxy.SharedProxyScheduler([("127.0.0.1", 8100)], [("127.0.0.1", 8200)])
        runtime = proxy.WorkerRuntime(scheduler)
        monkeypatch.setattr(proxy, "runtime", runtime)
        monkeypatch.setattr(
            proxy,
            "global_args",
            argparse.Namespace(
                max_retries=2,
                retry_delay=0,
                enable_prefill_affinity=affinity,
            ),
        )

        async def backend(request):
            role = request.url.host
            calls[role].append(
                {"body": json.loads(request.content), "headers": request.headers, "path": request.url.path}
            )
            if role == "prefill":
                if calls["decode"]:
                    # Close the abandoned decode before starting another P/D attempt.
                    assert scripts[len(calls["decode"]) - 1].is_closed
                if prefill_error and len(calls[role]) >= 2:
                    raise prefill_error
                return httpx.Response(
                    200,
                    json={
                        "kv_transfer_params": {"remote_block_ids": [len(calls[role])]},
                        "usage": {"cache_read_input_tokens": 32, "input_tokens_details": {"cached_tokens": 7}},
                    },
                )
            script = scripts[len(calls[role]) - 1]
            if isinstance(script, BaseException):
                raise script
            return script

        transport = httpx.MockTransport(backend)
        async with (
            httpx.AsyncClient(base_url="http://prefill/v1", transport=transport) as prefill,
            httpx.AsyncClient(base_url="http://decode/v1", transport=transport) as decode,
            httpx.AsyncClient(base_url="http://proxy", transport=httpx.ASGITransport(app=proxy.app)) as client,
        ):
            runtime._clients[proxy.ServerRole.PREFILL][proxy.server_key("127.0.0.1", 8100)] = prefill
            runtime._clients[proxy.ServerRole.DECODE][proxy.server_key("127.0.0.1", 8200)] = decode
            try:
                response = await client.post("/v1" + api, json=payload, headers=headers)
            finally:
                assert scheduler.request_num == 0
                assert all(entry.active_kv_cache == 0 for entry in scheduler.prefillers.values())
                assert all(entry.active_tokens == 0 for entry in scheduler.decoders.values())
        return response, calls

    return lambda *args, **kwargs: asyncio.run(run(*args, **kwargs))


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
        result = {"stop_reason": stop_reason}
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
    else:
        assert second == payload
        assert response.content == bodies[-1]
    for role in calls.values():
        assert all(call["path"] == "/v1" + api for call in role)


@pytest.mark.parametrize("api", ["/messages", "/responses"])
@pytest.mark.parametrize("streaming", [False, True])
def test_native_passthrough_and_prefill_payload(exchange, api, streaming):
    result = {
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
    assert response.content == raw  # No usage, tool, reasoning, or event rewrites.
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
@pytest.mark.parametrize("exhausted", [False, True])
def test_recompute_reuses_original_request_and_fresh_kv(exchange, api, streaming, exhausted):
    # Responses needs a backend extension to expose this marker; upstream omits it.
    stopped = {"stop_reason": "recomputed", "text": "discarded"}
    if api == "/messages":
        stopped.update(stop_reason="stop_sequence", stop_sequence="recomputed")
    final = {"stop_reason": "end_turn", "text": "done"}
    first = Body(wire(api, stopped, streaming))
    last = Body(wire(api, stopped if exhausted else final, streaming))
    payload = {
        "model": "m",
        "stream": streaming,
        "input": "hi",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 80,
        "max_output_tokens": 80,
    }
    response, calls = exchange(api, payload, [httpx.Response(200, stream=first), httpx.Response(200, stream=last)])
    assert response.status_code == (503 if exhausted else 200)
    if exhausted:
        assert "retry limit" in response.json()["detail"]
    else:
        assert response.content == wire(api, final, streaming)
    assert first.closed and last.closed
    assert len(calls["prefill"]) == len(calls["decode"]) == 2
    for index, attempt in enumerate(calls["decode"], 1):
        assert attempt["body"] == {**payload, "kv_transfer_params": {"remote_block_ids": [index]}}


@pytest.mark.parametrize("api", ["/messages", "/responses"])
def test_recompute_in_unterminated_final_event(exchange, api):
    stopped = wire(api, {"stop_reason": "recomputed"}, True).rstrip(b"\r\n")
    final = wire(api, {"text": "done"}, True)
    response, calls = exchange(
        api, {"stream": True}, [httpx.Response(200, content=stopped), httpx.Response(200, content=final)]
    )
    assert response.status_code == 200
    assert response.content == final
    assert len(calls["prefill"]) == len(calls["decode"]) == 2


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
    assert response.content == (final if retry else original)
    assert len(calls["decode"]) == len(calls["prefill"]) == (2 if retry else 1)


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
@pytest.mark.parametrize("streaming", [False, True])
def test_interrupted_decode_returns_502_and_closes_backend(exchange, api, streaming):
    stream = Body(b'{"partial":', httpx.ReadError("connection lost"))
    response, calls = exchange(api, {"stream": streaming}, [httpx.Response(200, stream=stream)])
    assert response.status_code == 502
    assert stream.closed
    assert len(calls["decode"]) == 1


def test_incomplete_json_returns_502(exchange):
    response, _ = exchange("/messages", {}, [httpx.Response(200, content=b'{"partial":')])
    assert response.status_code == 502


@pytest.mark.parametrize("stage", ["decode", "retry_prefill"])
@pytest.mark.parametrize("error", [asyncio.CancelledError, httpx.ReadError])
def test_recompute_failure_releases_load(exchange, stage, error):
    stream = Body(
        wire("/messages", {"stop_reason": "recomputed"}, True), error("interrupted") if stage == "decode" else None
    )
    kwargs = {"prefill_error": error("interrupted")} if stage == "retry_prefill" else {}
    args = ("/messages", {"stream": True}, [httpx.Response(200, stream=stream)])
    if error is asyncio.CancelledError:
        with pytest.raises(asyncio.CancelledError):
            exchange(*args, **kwargs)
    else:
        response, _ = exchange(*args, **kwargs)
        assert response.status_code == 502
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
    assert response.json()["error"]["param"] == parameter
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
