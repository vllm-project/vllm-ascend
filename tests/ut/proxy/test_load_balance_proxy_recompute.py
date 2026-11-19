"""Native API recompute tests with real HTTPX errors and the shared scheduler."""

import asyncio
import json

import httpx
import pytest

from examples.disaggregated_prefill_v1 import load_balance_proxy_server_example as proxy
from tests.ut.proxy.test_load_balance_proxy_protocol import _body, _request
from tests.ut.proxy.test_load_balance_proxy_protocol import installed_runtime as _runtime_fixture

installed_runtime = _runtime_fixture


class Chunks(httpx.AsyncByteStream):
    def __init__(self, chunks, *, fail=False):
        self.chunks = chunks
        self.fail = fail
        self.closed = False

    async def __aiter__(self):
        for chunk in self.chunks:
            yield chunk
        if self.fail:
            raise httpx.ReadError("decoder closed the connection")

    async def aclose(self):
        self.closed = True


def native_request(api, streaming):
    req = {"model": "m", "stream": streaming, "request_id": "client-request", "vllm_xargs": {"existing": 4}}
    if api == "/messages":
        req.update(messages=[{"role": "user", "content": "hello"}], max_tokens=117)
    else:
        req.update(input="hello", max_output_tokens=117)
    return req


def _event(payload):
    header = f"event: {payload['type']}\r\ndata: ".encode()
    return header + json.dumps(payload, ensure_ascii=False).encode() + b"\r\n\r\n"


def _start(api, attempt):
    if api == "/messages":
        return [{"type": "message_start", "message": {"id": attempt, "content": [], "usage": {"input_tokens": 50}}}]
    return [
        {"type": kind, "sequence_number": index, "response": {"id": attempt, "output": [], "usage": None}}
        for index, kind in enumerate(("response.created", "response.in_progress"))
    ]


def _content(api, text):
    if api == "/messages":
        return {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": text}}
    return {"type": "response.output_text.delta", "sequence_number": 2, "delta": text}


def _error(api, signal, *, streaming):
    if streaming and api == "/messages":
        return {"type": "error", "error": {"type": "api_error", "message": signal}}
    if streaming:
        return {"type": "error", "code": signal, "message": signal, "param": None, "sequence_number": 3}
    return {"error": {"type": "InternalServerError", "message": signal, "code": 500, "param": None}}


def _recompute_error(api, progress, *, streaming):
    if progress is None:
        return _error(api, proxy.PD_RECOMPUTE_RETRY, streaming=streaming)
    signal = proxy.PD_RECOMPUTE_INTERRUPTED
    event = _error(api, signal, streaming=streaming)
    payload = event if streaming and api == "/responses" else event["error"]
    payload["message"] = f"{signal}:{json.dumps(progress)}"
    return event


def _result(api):
    if api == "/messages":
        return {
            "type": "message",
            "id": "winning-attempt",
            "content": [{"type": "text", "text": "你好"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 49, "cache_read_input_tokens": 1, "output_tokens": 2},
        }
    return {
        "object": "response",
        "id": "winning-attempt",
        "status": "completed",
        "output": [{"type": "message", "content": [{"type": "output_text", "text": "你好"}]}],
        "usage": {"input_tokens": 50, "input_tokens_details": {"cached_tokens_per_turn": [1, 2]}, "output_tokens": 2},
    }


def _success_chunks(api, streaming):
    result = _result(api)
    if not streaming:
        return [json.dumps(result).encode()]
    terminal = (
        {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": result["usage"]}
        if api == "/messages"
        else {"type": "response.completed", "sequence_number": 3, "response": result}
    )
    events = [*_start(api, "winning-attempt"), _content(api, "你好"), terminal]
    if api == "/messages":
        events.append({"type": "message_stop"})
    data = b"".join(map(_event, events))
    cut = data.index("你".encode()) + 1
    return [data[:cut], data[cut:]]


def _text_result(api, attempt, text, output_tokens, *, input_tokens=50):
    result = _result(api)
    result["id"] = attempt
    result["usage"].update(input_tokens=input_tokens, output_tokens=output_tokens)
    if api == "/messages":
        result["content"][0]["text"] = text
    else:
        result["output"] = [
            {
                "type": "message",
                "role": "assistant",
                "id": f"item-{attempt}",
                "status": "completed",
                "content": [{"type": "output_text", "text": text, "annotations": [], "logprobs": []}] if text else [],
            }
        ]
        result["usage"]["total_tokens"] = input_tokens + output_tokens
    return result


def _text_stream(api, result, *, recompute=None):
    events = _start(api, result["id"])
    if api == "/messages":
        text = result["content"][0]["text"]
        if text:
            events.extend(
                [
                    {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
                    _content(api, text),
                    {"type": "content_block_stop", "index": 0},
                ]
            )
        if not recompute:
            events.extend(
                [
                    {
                        "type": "message_delta",
                        "delta": {"stop_reason": result["stop_reason"]},
                        "usage": result["usage"],
                    },
                    {"type": "message_stop"},
                ]
            )
    else:
        if result["output"] and result["output"][0]["content"]:
            item = result["output"][0]
            part = item["content"][0]
            indices = {"output_index": 0, "content_index": 0, "item_id": item["id"]}
            events.extend(
                [
                    {
                        "type": "response.output_item.added",
                        "output_index": 0,
                        "item": {**item, "status": "in_progress", "content": []},
                    },
                    {
                        "type": "response.content_part.added",
                        **indices,
                        "part": {**part, "text": ""},
                    },
                    {**_content(api, part["text"]), **indices, "logprobs": []},
                ]
            )
            if not recompute:
                events.extend(
                    [
                        {"type": "response.output_text.done", **indices, "text": part["text"], "logprobs": []},
                        {"type": "response.content_part.done", **indices, "part": part},
                        {"type": "response.output_item.done", "output_index": 0, "item": item},
                    ]
                )
        if not recompute:
            event = "response.incomplete" if result["status"] == "incomplete" else "response.completed"
            events.append({"type": event, "response": result})
    if recompute:
        events.append(_recompute_error(api, recompute, streaming=True))
    return _frame_chunks(api, events)


def _frame_chunks(api, events):
    if api == "/responses":
        for index, event in enumerate(events):
            event["sequence_number"] = index
    data = b"".join(map(_event, events))
    boundary = data.index(b"\r\n\r\n") + 3
    return [data[:boundary], data[boundary:]]


def _events(raw):
    return [json.loads(line[5:]) for line in raw.splitlines() if line.startswith(b"data:")]


def _structured_options(api, case):
    if case == "schema":
        schema = {"type": "object", "properties": {"key": {"type": "string"}}, "required": ["key"]}
        if api == "/messages":
            return {"output_config": {"format": {"type": "json_schema", "schema": schema}}}
        return {"text": {"format": {"type": "json_schema", "name": "result", "schema": schema}}}
    if case == "reasoning":
        return {"thinking": {"type": "adaptive"}} if api == "/messages" else {"reasoning": {"effort": "high"}}
    if case == "mcp":
        return {"tools": [{"type": "mcp", "server_label": "mock-tool", "server_url": "https://mock.invalid"}]}
    parameters = {"type": "object"}
    if api == "/messages":
        return {"tools": [{"name": "lookup", "input_schema": parameters}]}
    return {"tools": [{"type": "function", "name": "lookup", "parameters": parameters}]}


def _structured_attempt(api, case, attempt, *, interrupted):
    text = '{"key":' if interrupted else '{"key":"done"}'
    result = _text_result(api, attempt, text, 5 if interrupted else 2)
    if case == "schema":
        return result, _events(b"".join(_text_stream(api, result)))
    if api == "/messages":
        if case == "reasoning":
            block = {"type": "thinking", "thinking": f"reasoning-{attempt}", "signature": f"sig-{attempt}"}
            start = {**block, "thinking": "", "signature": ""}
            delta = {"type": "thinking_delta", "thinking": block["thinking"]}
        else:
            block = {"type": "tool_use", "id": f"call-{attempt}", "name": "lookup", "input": {"key": "done"}}
            start = {**block, "input": {}}
            delta = {"type": "input_json_delta", "partial_json": text}
            result["stop_reason"] = "tool_use"
        result["content"] = [block]
        events = [
            *_start(api, attempt),
            {"type": "content_block_start", "index": 0, "content_block": start},
            {"type": "content_block_delta", "index": 0, "delta": delta},
        ]
        if case == "reasoning":
            events.append(
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "signature_delta", "signature": block["signature"]},
                }
            )
        events.extend(
            [
                {"type": "content_block_stop", "index": 0},
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": result["stop_reason"]},
                    "usage": result["usage"],
                },
                {"type": "message_stop"},
            ]
        )
        return result, events

    if case == "reasoning":
        item = {
            "type": "reasoning",
            "id": f"rs-{attempt}",
            "summary": [],
            "content": [{"type": "reasoning_text", "text": f"reasoning-{attempt}"}],
        }
        start = {**item, "content": []}
        event = {
            "type": "response.reasoning_text.delta",
            "content_index": 0,
            "delta": item["content"][0]["text"],
        }
    else:
        tool_type = "mcp_call" if case == "mcp" else "function_call"
        item = {
            "type": tool_type,
            "id": f"tool-{attempt}",
            "name": "lookup",
            "arguments": text,
            "status": "completed",
        }
        if case == "mcp":
            item.update(server_label="mock-tool", output=f"mock-result-{attempt}")
        else:
            item["call_id"] = f"call-{attempt}"
        start = {**item, "arguments": "", "status": "in_progress"}
        event = {"type": f"response.{tool_type}_arguments.delta", "delta": text}
    result["output"] = [item]
    return result, [
        *_start(api, attempt),
        {"type": "response.output_item.added", "output_index": 0, "item": start},
        {**event, "output_index": 0, "item_id": item["id"]},
        {"type": "response.output_item.done", "output_index": 0, "item": item},
        {"type": "response.completed", "response": result},
    ]


def _assert_no_load(scheduler):
    assert scheduler.request_num == 0
    assert all(entry.active_kv_cache == pytest.approx(0) for entry in scheduler.prefillers.values())
    assert all(entry.active_tokens == pytest.approx(0) for entry in scheduler.decoders.values())


@pytest.mark.parametrize("api", ["/messages", "/responses"])
@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("after_output", [False, True])
def test_recompute_selects_new_pd_and_preserves_request(installed_runtime, monkeypatch, api, streaming, after_output):
    """Shift load before recompute and verify the second P/D pair sees the original request."""
    runtime = proxy.get_runtime()
    scheduler = proxy.SharedProxyScheduler(
        [("127.0.0.1", port) for port in (8100, 8101)], [("127.0.0.1", port) for port in (8200, 8201)]
    )
    runtime.scheduler = scheduler
    proxy.global_args.enable_prefill_affinity = True
    monkeypatch.setattr(proxy, "conversation_affinity_key", lambda request: "same-session")
    affinity_hosts = [proxy.server_key("127.0.0.1", port) for port in (8100, 8101)]
    first_prefill = scheduler.reserve_prefill_kv(1.0, "same-session")["key"]
    scheduler.release_prefill_kv(first_prefill, 1.0)
    other_prefill = next(key for key in affinity_hosts if key != first_prefill)
    routes = []
    old_stream = None
    original = native_request(api, streaming)
    progress = {"output_tokens": 5, "remaining_tokens": 112, "input_tokens": 50}
    partial = _text_result(api, "first-attempt", "第一段 ", 5)
    final = _text_result(api, "winning-attempt", "第二段", 2, input_tokens=59)
    other_load = 10000.0

    async def backend(request):
        nonlocal old_stream
        port = request.url.port
        payload = json.loads(request.content)
        routes.append((port, request.headers["X-Request-Id"], payload))
        assert scheduler.request_num == (1 if len(routes) <= 2 else 2)
        if port < 8200:
            key = proxy.server_key("127.0.0.1", port)
            is_retry = len(routes) > 1
            assert key == (other_prefill if is_retry else first_prefill)
            assert scheduler.prefillers[key].active_kv_cache > 0
            if is_retry:
                assert scheduler.decoders[proxy.server_key("127.0.0.1", 8200)].active_tokens == pytest.approx(
                    other_load
                )
                assert scheduler.decoders[proxy.server_key("127.0.0.1", 8201)].active_tokens == 0
                assert scheduler.prefillers[first_prefill].active_kv_cache == pytest.approx(other_load)
                assert old_stream is not None and old_stream.closed
            return httpx.Response(
                200,
                json={
                    "kv_transfer_params": {"do_remote_prefill": True, "remote_block_ids": [port]},
                    "usage": {
                        "cache_read_input_tokens": 7 if is_retry else 4,
                        "input_tokens_details": {"cached_tokens": 7 if is_retry else 4},
                    },
                },
            )
        assert scheduler.decoders[proxy.server_key("127.0.0.1", port)].active_tokens > 0
        if port == 8200:
            err = _error(api, proxy.PD_RECOMPUTE_RETRY, streaming=streaming)
            scheduler.begin_request(other_load, "same-session")
            busy_decoder = scheduler.pick_decoder(other_load)
            assert busy_decoder["port"] == 8201
            assert scheduler.pick_decoder(other_load)["port"] == 8200
            scheduler.release_decoder(busy_decoder["key"], other_load)
            if after_output and streaming:
                old_stream = Chunks(_text_stream(api, partial, recompute=progress))
            elif after_output:
                partial["kv_transfer_params"] = {proxy.PD_RECOMPUTE_ARG: proxy.PD_RECOMPUTE_INTERRUPTED, **progress}
                old_stream = Chunks([json.dumps(partial).encode()])
            elif streaming:
                if api == "/responses":
                    err["message"] = "Decode needs to restart."
                data = b": keepalive\r\n\r\n" + b"".join(map(_event, [*_start(api, "discarded-attempt"), err]))
                old_stream = Chunks([data[:-3], data[-3:] + _event({"type": "message_stop"})])
            else:
                old_stream = Chunks([json.dumps(err).encode()])
            return httpx.Response(200 if streaming or after_output else 500, stream=old_stream)
        if after_output:
            chunks = _text_stream(api, final) if streaming else [json.dumps(final).encode()]
            return httpx.Response(200, stream=Chunks(chunks))
        return httpx.Response(200, stream=Chunks(_success_chunks(api, streaming)))

    async def run():
        transport = httpx.MockTransport(backend)
        for role, ports in ((proxy.ServerRole.PREFILL, (8100, 8101)), (proxy.ServerRole.DECODE, (8200, 8201))):
            for port in ports:
                key = proxy.server_key("127.0.0.1", port)
                runtime._clients[role][key] = httpx.AsyncClient(
                    base_url=f"http://127.0.0.1:{port}/v1", transport=transport
                )
        try:
            result = await proxy.handle_completions_impl(api, _request(original))
            assert result.status_code == 200, (result.body, [port for port, _, _ in routes])
            return await _body(result)
        finally:
            await runtime.close()

    data = asyncio.run(run())
    assert [port for port, _, _ in routes] == [
        int(first_prefill.rsplit(":", 1)[1]),
        8200,
        int(other_prefill.rsplit(":", 1)[1]),
        8201,
    ]
    first_p, first_d, retry_p, retry_d = routes
    assert first_p[1] == first_d[1] == "client-request"
    assert retry_p[1] == retry_d[1] != first_d[1]
    budget_key = "max_tokens" if api == "/messages" else "max_output_tokens"
    history_key = "messages" if api == "/messages" else "input"
    for index, (prefill, decode) in enumerate(((first_p, first_d), (retry_p, retry_d))):
        assert prefill[2][budget_key] == 1
        assert prefill[2][history_key] == decode[2][history_key]
        continuing = index == 1 and after_output
        assert decode[2][budget_key] == (112 if continuing else 117)
        if continuing:
            history = decode[2][history_key]
            assert history[0] == {"role": "user", "content": "hello"}
            assert history[1]["role"] == "assistant"
            if api == "/messages":
                assert history[1]["content"] == "第一段 "
                assert prefill[2]["vllm_xargs"][proxy.PD_CONTINUE_MESSAGE] == proxy.PD_RECOMPUTE_VERSION
                assert decode[2]["vllm_xargs"][proxy.PD_CONTINUE_MESSAGE] == proxy.PD_RECOMPUTE_VERSION
            else:
                assert history[1]["status"] == "in_progress"
                assert history[1]["content"][0]["text"] == "第一段 "
        else:
            assert decode[2][history_key] == original[history_key]
        assert decode[2]["kv_transfer_params"]["remote_block_ids"] == [prefill[0]]
        assert decode[2]["vllm_xargs"]["existing"] == 4
        assert decode[2]["vllm_xargs"][proxy.PD_RECOMPUTE_ARG] == proxy.PD_RECOMPUTE_VERSION
        assert decode[2]["vllm_xargs"][proxy.PD_RECOMPUTE_RESUME] == proxy.PD_RECOMPUTE_VERSION
    assert b"discarded-attempt" not in data
    assert b"ascend_pd_recompute" not in data
    if after_output:
        events = _events(data) if streaming else None
        usage = _assert_resumed_response(api, events, data, "第一段 第二段", ("第一段 ", "第二段"))
        assert usage["output_tokens"] == 2
    elif streaming:
        events = _events(data)
        start_kind = "message_start" if api == "/messages" else "response.created"
        assert sum(event["type"] == start_kind for event in events) == 1
        delta_kind = "content_block_delta" if api == "/messages" else "response.output_text.delta"
        assert [event for event in events if event["type"] == delta_kind] == [_content(api, "你好")]
        usage = events[-2]["usage"] if api == "/messages" else events[-1]["response"]["usage"]
    else:
        result = json.loads(data)
        assert result["id"] == "winning-attempt"
        result_key = "content" if api == "/messages" else "output"
        assert result[result_key] == _result(api)[result_key]
        usage = result["usage"]
    if api == "/messages":
        assert usage["input_tokens"] + usage["cache_read_input_tokens"] == (60 if after_output else 50)
        assert usage["cache_read_input_tokens"] == 7
    else:
        assert usage["input_tokens"] == (59 if after_output else 50)
        assert usage["input_tokens_details"] == {"cached_tokens_per_turn": [7, 2], "cached_tokens": 9}
    assert scheduler.request_num == 1
    scheduler.finish_request(first_prefill, other_load, proxy.server_key("127.0.0.1", 8200), other_load, True)
    _assert_no_load(scheduler)


def _assert_resumed_response(api, events, raw, text, deltas, *, incomplete=False, response_id="first-attempt"):
    if events is None:
        result = json.loads(raw)
    elif api == "/messages":
        kinds = [event["type"] for event in events]
        assert kinds == [
            "message_start",
            "content_block_start",
            *["content_block_delta"] * len(deltas),
            "content_block_stop",
            "message_delta",
            "message_stop",
        ]
        assert events[0]["message"]["id"] == response_id
        assert [event["delta"]["text"] for event in events if event["type"] == "content_block_delta"] == list(deltas)
        return events[-2]["usage"]
    else:
        kinds = [event["type"] for event in events]
        assert kinds == [
            "response.created",
            "response.in_progress",
            "response.output_item.added",
            "response.content_part.added",
            *["response.output_text.delta"] * len(deltas),
            "response.output_text.done",
            "response.content_part.done",
            "response.output_item.done",
            "response.incomplete" if incomplete else "response.completed",
        ]
        assert [event["sequence_number"] for event in events] == list(range(len(events)))
        assert [event["delta"] for event in events if event["type"] == "response.output_text.delta"] == list(deltas)
        assert {event["item_id"] for event in events if "item_id" in event} == {f"item-{response_id}"}
        assert events[-4]["text"] == events[-3]["part"]["text"] == text
        result = events[-1]["response"]
        assert result["id"] == response_id
        assert result["output"][0]["id"] == events[-2]["item"]["id"] == f"item-{response_id}"
    assert proxy._native_text_block(api.removeprefix("/"), result)["text"] == text
    assert "kv_transfer_params" not in result
    if api == "/responses":
        usage = result["usage"]
        assert usage["total_tokens"] == usage["input_tokens"] + usage["output_tokens"]
    return result["usage"]


@pytest.mark.parametrize(
    "api,suffix",
    [(api, suffix) for api in ("/messages", "/responses") for suffix in ("", "third")]
    + [pytest.param("/responses", None, id="responses-before-output")],
)
@pytest.mark.parametrize("streaming", [False, True])
def test_two_recomputes_preserve_one_request_and_its_budget(installed_runtime, api, streaming, suffix):
    scheduler, prefill, _ = installed_runtime
    runtime = proxy.get_runtime()
    proxy.global_args.max_retries = 2
    request = native_request(api, streaming)
    if api == "/messages":
        request["tools"] = [{"name": "search", "input_schema": {"type": "object"}}]
    else:
        request["tools"] = [{"type": "function", "name": "search", "parameters": {"type": "object"}}]
    payloads = []
    attempt_ids = []
    opened = []
    partials = [
        ("first-attempt", "first ", {"output_tokens": 6, "remaining_tokens": 111, "input_tokens": 50}),
        ("second-attempt", "second ", {"output_tokens": 7, "remaining_tokens": 104, "input_tokens": 61}),
    ]

    async def backend(request):
        index = len(payloads)
        payloads.append(json.loads(request.content))
        attempt_ids.append(request.headers["X-Request-Id"])
        assert scheduler.request_num == 1
        assert all(stream.closed for stream in opened)
        status = 200
        if suffix is None:
            if index < len(partials):
                error = _recompute_error(api, None, streaming=streaming)
                chunks = [_event(error)] if streaming else [json.dumps(error).encode()]
                status = 200 if streaming else 500
            else:
                chunks = _success_chunks(api, streaming)
        else:
            progress = None
            if index < len(partials):
                attempt, text, progress = partials[index]
                result = _text_result(
                    api, attempt, text, progress["output_tokens"], input_tokens=progress["input_tokens"]
                )
                if not streaming:
                    result["kv_transfer_params"] = {proxy.PD_RECOMPUTE_ARG: proxy.PD_RECOMPUTE_INTERRUPTED, **progress}
            else:
                result = _text_result(api, "final-attempt", suffix, 2, input_tokens=72)
                if api == "/responses" and suffix:
                    result["status"] = "incomplete"
                    result["output"][0]["status"] = "incomplete"
            chunks = _text_stream(api, result, recompute=progress) if streaming else [json.dumps(result).encode()]
        stream = Chunks(chunks)
        opened.append(stream)
        return httpx.Response(status, stream=stream)

    async def run():
        async with httpx.AsyncClient(base_url="http://decoder/v1", transport=httpx.MockTransport(backend)) as client:
            runtime._clients[proxy.ServerRole.DECODE][proxy.server_key("127.0.0.1", 8200)] = client
            response = await proxy.handle_completions_impl(api, _request(request))
            assert response.status_code == 200
            return await _body(response)

    raw = asyncio.run(run())
    budget_key = "max_tokens" if api == "/messages" else "max_output_tokens"
    history_key = "messages" if api == "/messages" else "input"
    assert attempt_ids == [entry["headers"]["X-Request-Id"] for entry in prefill.posts]
    assert len(attempt_ids) == len(set(attempt_ids)) == 3
    if suffix is None:
        assert [payload[budget_key] for payload in payloads] == [117, 117, 117]
        assert all(payload[history_key] == request[history_key] for payload in payloads)
        assert b"ascend_pd_recompute" not in raw
        if streaming:
            assert _events(raw)[-1]["type"] == "response.completed"
        else:
            assert json.loads(raw)["status"] == "completed"
    else:
        deltas = ["first ", "second "] + ([suffix] if suffix else [])
        usage = _assert_resumed_response(
            api, _events(raw) if streaming else None, raw, "".join(deltas), deltas, incomplete=bool(suffix)
        )
        assert usage["output_tokens"] == 2
        assert [payload[budget_key] for payload in payloads] == [117, 111, 104]
        for payload, prefix in zip(payloads[1:], ("first ", "first second "), strict=True):
            assert len(payload[history_key]) == 2
            content = payload[history_key][-1]["content"]
            assert (content if api == "/messages" else content[0]["text"]) == prefix
    assert all(stream.closed for stream in opened)
    _assert_no_load(scheduler)


@pytest.mark.parametrize("api", ["/messages", "/responses"])
@pytest.mark.parametrize("case", ["schema", "reasoning", "client-tool"])
@pytest.mark.parametrize("streaming", [False, True])
def test_recompute_with_structured_output(installed_runtime, api, case, streaming):
    scheduler, prefill, _ = installed_runtime
    runtime = proxy.get_runtime()
    options = _structured_options(api, case)
    native = {**native_request(api, streaming), **options}
    progress = {"output_tokens": 5, "remaining_tokens": 112, "input_tokens": 50}
    first, first_events = _structured_attempt(api, case, "abandoned", interrupted=True)
    result, final_events = _structured_attempt(api, case, "winner", interrupted=False)
    resume_text = case == "schema"
    if resume_text:
        result = _text_result(api, "winner", '"done"}', 2)
        final_events = _events(b"".join(_text_stream(api, result)))
    reserved = []
    opened = []

    async def backend(request):
        payload = json.loads(request.content)
        reserved.append((request.headers["X-Request-Id"], payload))
        assert scheduler.request_num == 1
        assert all(stream.closed for stream in opened)
        status = 200
        if len(reserved) == 1:
            prefill.payload["kv_transfer_params"]["remote_block_ids"] = [2]
            error = _recompute_error(api, progress, streaming=streaming)
            if streaming:
                tail = 2 if api == "/messages" else 1
                emitted = first_events[:-tail]
                chunks = _frame_chunks(api, [*emitted, error])
            else:
                first["kv_transfer_params"] = {proxy.PD_RECOMPUTE_ARG: proxy.PD_RECOMPUTE_INTERRUPTED, **progress}
                chunks = [json.dumps(first).encode()]
        else:
            chunks = _frame_chunks(api, final_events) if streaming else [json.dumps(result).encode()]
        stream = Chunks(chunks)
        opened.append(stream)
        return httpx.Response(status, stream=stream)

    async def run():
        async with httpx.AsyncClient(base_url="http://decoder/v1", transport=httpx.MockTransport(backend)) as client:
            runtime._clients[proxy.ServerRole.DECODE][proxy.server_key("127.0.0.1", 8200)] = client
            response = await proxy.handle_completions_impl(api, _request(native))
            assert response.status_code == 200
            return await _body(response)

    raw = asyncio.run(run())
    if resume_text:
        _assert_resumed_response(
            api,
            _events(raw) if streaming else None,
            raw,
            '{"key":"done"}',
            ('{"key":', '"done"}'),
            response_id="abandoned",
        )
    else:
        assert b"abandoned" not in raw
        if streaming:
            assert [ev["type"] for ev in _events(raw)] == [ev["type"] for ev in final_events]
        else:
            assert (
                json.loads(raw)["content" if api == "/messages" else "output"]
                == result["content" if api == "/messages" else "output"]
            )
    assert b"ascend_pd_recompute" not in raw
    assert len(prefill.posts) == len(reserved) == 2
    assert [entry["headers"]["X-Request-Id"] for entry in prefill.posts] == [entry[0] for entry in reserved]
    budget_key, input_key = ("max_tokens", "messages") if api == "/messages" else ("max_output_tokens", "input")
    assert reserved[1][1]["kv_transfer_params"]["remote_block_ids"] == [2]
    for index, ((_, payload), pre) in enumerate(zip(reserved, prefill.posts, strict=True)):
        assert payload[input_key] == pre["json"][input_key]
        if index == 1 and resume_text:
            added = payload[input_key][-1]["content"]
            assert (added if api == "/messages" else added[0]["text"]) == '{"key":'
        else:
            assert payload[input_key] == native[input_key]
        assert pre["json"][budget_key] == 1
        assert {key: payload[key] for key in options} == {key: pre["json"][key] for key in options} == options
    assert [payload[budget_key] for _, payload in reserved] == [117, 112]
    assert all(stream.closed for stream in opened)
    _assert_no_load(scheduler)


@pytest.mark.parametrize("api", ["/messages", "/responses"])
@pytest.mark.parametrize("case", ["client-tool", "reasoning"])
@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("text_first", [True, False])
def test_mixed_recompute_keeps_only_text(installed_runtime, api, case, streaming, text_first):
    scheduler, prefill, _ = installed_runtime
    runtime = proxy.get_runtime()
    native = {**native_request(api, streaming), **_structured_options(api, case)}
    key = "content" if api == "/messages" else "output"
    index_key = "index" if api == "/messages" else "output_index"
    tail, starts = (2, 1) if api == "/messages" else (1, 2)
    progress = {"output_tokens": 5, "remaining_tokens": 112, "input_tokens": 50}
    first = _text_result(api, "first-attempt", "prefix ", 5)
    typed, structured = _structured_attempt(api, case, "discarded", interrupted=True)
    opening = _start(api, "first-attempt")
    text_events = _events(b"".join(_text_stream(api, first)))[starts:-tail]
    structured = structured[starts:-tail]
    for event in structured if text_first else text_events:
        if index_key in event:
            event[index_key] += 1
    events = opening + (text_events + structured if text_first else structured + text_events)
    first[key] = first[key] + typed[key] if text_first else typed[key] + first[key]
    first["kv_transfer_params"] = {proxy.PD_RECOMPUTE_ARG: proxy.PD_RECOMPUTE_INTERRUPTED, **progress}
    final, final_events = _structured_attempt(api, case, "winner", interrupted=False)
    requests = []

    async def backend(request):
        requests.append(json.loads(request.content))
        if len(requests) == 1:
            chunks = (
                _frame_chunks(api, events + [_recompute_error(api, progress, streaming=True)])
                if streaming
                else [json.dumps(first).encode()]
            )
        else:
            chunks = _frame_chunks(api, final_events) if streaming else [json.dumps(final).encode()]
        return httpx.Response(200, stream=Chunks(chunks))

    async def run():
        async with httpx.AsyncClient(base_url="http://decoder/v1", transport=httpx.MockTransport(backend)) as client:
            runtime._clients[proxy.ServerRole.DECODE][proxy.server_key("127.0.0.1", 8200)] = client
            response = await proxy.handle_completions_impl(api, _request(native))
            assert response.status_code == 200
            return await _body(response)

    raw = asyncio.run(run())
    assert len(requests) == len(prefill.posts) == 2
    budget, history = ("max_tokens", "messages") if api == "/messages" else ("max_output_tokens", "input")
    assert requests[1][budget] == 112
    content = requests[1][history][-1]["content"]
    assert (content if api == "/messages" else content[0]["text"]) == "prefix "
    assert b"discarded" not in raw and b"winner" in raw
    if streaming:
        received = _events(raw)
        if api == "/messages":
            assert received[-1]["type"] == "message_stop"
            assert [e["index"] for e in received if e["type"] == "content_block_stop"] == [0, 1]
            assert (
                "".join(e["delta"]["text"] for e in received if e.get("delta", {}).get("type") == "text_delta")
                == "prefix "
            )
        else:
            assert [e["sequence_number"] for e in received] == list(range(len(received)))
            assert received[-1]["type"] == "response.completed"
            assert received[-1]["response"]["output"][0]["content"][0]["text"] == "prefix "
            assert received[-1]["response"]["output"][1] == final[key][0]
    else:
        blocks = json.loads(raw)[key]
        assert blocks[1:] == final[key]
        assert (blocks[0]["text"] if api == "/messages" else blocks[0]["content"][0]["text"]) == "prefix "
    _assert_no_load(scheduler)


@pytest.mark.parametrize("api", ["/messages", "/responses"])
def test_plain_text_stream_sends_delta_before_backend_finishes(installed_runtime, api):
    scheduler, _, _ = installed_runtime
    runtime = proxy.get_runtime()
    result = _text_result(api, "winner", "live", 3)
    events = _events(b"".join(_text_stream(api, result)))
    delta_type = "content_block_delta" if api == "/messages" else "response.output_text.delta"
    split = next(index for index, event in enumerate(events) if event["type"] == delta_type) + 1
    tail = b"event: ping\r\ndata: [DONE]\r\n\r\n"

    async def run():
        release_tail = asyncio.Event()

        class PausedTail(Chunks):
            async def __aiter__(self):
                yield b"".join(map(_event, events[:split]))
                await release_tail.wait()
                yield b"".join(map(_event, events[split:])) + tail

        async def backend(request):
            return httpx.Response(200, stream=PausedTail([]))

        async def forward():
            response = await proxy.handle_completions_impl(api, _request(native_request(api, True)))
            frames = []
            async for raw in response.body_iterator:
                frames.append(raw)
                if raw != tail and _events(raw)[0]["type"] == delta_type:
                    release_tail.set()
            assert release_tail.is_set() and frames[-1] == tail

        async with httpx.AsyncClient(base_url="http://decoder/v1", transport=httpx.MockTransport(backend)) as client:
            runtime._clients[proxy.ServerRole.DECODE][proxy.server_key("127.0.0.1", 8200)] = client
            await asyncio.wait_for(forward(), timeout=2)

    asyncio.run(run())
    _assert_no_load(scheduler)


@pytest.mark.parametrize("api", ["/messages", "/responses"])
def test_text_retry_limit_ends_sse_after_visible_output(installed_runtime, api):
    scheduler, prefill, _ = installed_runtime
    runtime = proxy.get_runtime()
    calls = 0

    async def backend(request):
        nonlocal calls
        calls += 1
        partial = _text_result(api, f"attempt-{calls}", "partial ", 5)
        progress = {"output_tokens": 5, "remaining_tokens": 117 - 5 * calls, "input_tokens": 50}
        return httpx.Response(200, stream=Chunks(_text_stream(api, partial, recompute=progress)))

    async def run():
        request = native_request(api, True)
        async with httpx.AsyncClient(base_url="http://decoder/v1", transport=httpx.MockTransport(backend)) as client:
            runtime._clients[proxy.ServerRole.DECODE][proxy.server_key("127.0.0.1", 8200)] = client
            response = await proxy.handle_completions_impl(api, _request(request))
            assert response.status_code == 200
            return await _body(response)

    raw = asyncio.run(run())
    assert b"attempt-1" in raw and b"attempt-2" not in raw and b"remaining_tokens" not in raw
    assert _events(raw)[-1]["type"] == "error"
    assert len(prefill.posts) == calls == 2
    _assert_no_load(scheduler)


@pytest.mark.parametrize("api", ["/messages", "/responses"])
@pytest.mark.parametrize("case", ["retry-prefill", "retry-truncated", "budget-exhausted"])
def test_partial_recompute_stream_failure_is_terminal(installed_runtime, api, case):
    scheduler, prefill, _ = installed_runtime
    runtime = proxy.get_runtime()
    request = native_request(api, True)
    progress = {"output_tokens": 3, "remaining_tokens": 0 if case == "budget-exhausted" else 114, "input_tokens": 50}
    result = _text_result(api, "first-attempt", "partial", 3)
    opened = []

    async def backend(request):
        if opened:
            stream = Chunks([b"".join(map(_event, _start(api, "truncated-attempt")))])
        else:
            if case == "retry-prefill":
                prefill.payload = {"usage": {}}
            stream = Chunks(_text_stream(api, result, recompute=progress))
        opened.append(stream)
        return httpx.Response(200, stream=stream)

    async def run():
        async with httpx.AsyncClient(base_url="http://decoder/v1", transport=httpx.MockTransport(backend)) as client:
            runtime._clients[proxy.ServerRole.DECODE][proxy.server_key("127.0.0.1", 8200)] = client
            response = await proxy.handle_completions_impl(api, _request(request))
            assert response.status_code == 200
            return await _body(response)

    events = _events(asyncio.run(run()))
    assert events[-1]["type"] == "error"
    assert not any(event["type"] in ("message_stop", "response.completed") for event in events)
    if api == "/responses":
        assert [event["sequence_number"] for event in events] == list(range(len(events)))
    assert len(prefill.posts) == (2 if case.startswith("retry-") else 1)
    assert len(opened) == (2 if case == "retry-truncated" else 1)
    assert all(stream.closed for stream in opened)
    _assert_no_load(scheduler)


@pytest.mark.parametrize(
    "api,streaming,case",
    [
        (api, streaming, case)
        for api in ("/messages", "/responses")
        for streaming in (False, True)
        for case in ("after-tokens", "retry-limit", "missing-retry-kv", "ordinary-error")
    ]
    + [(api, True, "committed-before-retry") for api in ("/messages", "/responses")],
)
def test_recompute_failure_is_not_a_success(installed_runtime, api, streaming, case):
    scheduler, prefill, _ = installed_runtime
    runtime = proxy.get_runtime()
    native = native_request(api, streaming)
    calls = 0
    opened = []
    signal = proxy.PD_RECOMPUTE_INTERRUPTED if case == "after-tokens" else proxy.PD_RECOMPUTE_RETRY
    if case == "ordinary-error":
        signal = "upstream failed for another reason"

    async def backend(request):
        nonlocal calls
        calls += 1
        err = _error(api, signal, streaming=streaming)
        if case == "missing-retry-kv":
            prefill.payload = {"usage": {}}
        if streaming:
            events = _start(api, "discarded")
            if case in ("after-tokens", "committed-before-retry"):
                events.extend(
                    [
                        _content(api, "partial"),
                        err,
                        {"type": "message_stop" if api == "/messages" else "response.completed"},
                    ]
                )
            else:
                events.append(err)
            stream = Chunks([b"".join(map(_event, events))])
        else:
            stream = Chunks([json.dumps(err).encode()])
        opened.append(stream)
        return httpx.Response(200 if streaming else 500, stream=stream)

    async def run():
        async with httpx.AsyncClient(base_url="http://decoder/v1", transport=httpx.MockTransport(backend)) as client:
            runtime._clients[proxy.ServerRole.DECODE][proxy.server_key("127.0.0.1", 8200)] = client
            response = await proxy.handle_completions_impl(api, _request(native))
            return response.status_code, await _body(response)

    status, data = asyncio.run(run())
    if case in ("after-tokens", "committed-before-retry") and streaming:
        events = _events(data)
        assert status == 200
        assert events[-2] == _content(api, "partial")
        assert events[-1]["type"] == "error"
        assert b"cannot resume" in data
        assert not any(event["type"] in ("message_stop", "response.completed") for event in events)
    elif case == "ordinary-error":
        assert status == (200 if streaming else 500)
        assert signal.encode() in data
    else:
        assert status == (503 if case == "retry-limit" else 502)
        assert "error" in json.loads(data)
    assert calls == (2 if case == "retry-limit" else 1)
    assert len(prefill.posts) == (2 if case in ("retry-limit", "missing-retry-kv") else 1)
    assert all(stream.closed for stream in opened)
    _assert_no_load(scheduler)


@pytest.mark.parametrize("spec_version", ["2.3", "2.4"])
def test_disconnect_closes_native_stream_even_if_response_head_is_not_sent(installed_runtime, spec_version):
    scheduler, _, _ = installed_runtime
    runtime = proxy.get_runtime()
    stream = Chunks(_success_chunks("/responses", True))

    async def run():
        async with httpx.AsyncClient(
            base_url="http://decoder/v1", transport=httpx.MockTransport(lambda req: httpx.Response(200, stream=stream))
        ) as client:
            runtime._clients[proxy.ServerRole.DECODE][proxy.server_key("127.0.0.1", 8200)] = client
            response = await proxy.handle_completions_impl("/responses", _request(native_request("/responses", True)))

            async def receive():
                return {"type": "http.disconnect"}

            async def send(message):
                if spec_version == "2.4":
                    raise OSError("client disconnected")
                await asyncio.sleep(0)

            if spec_version == "2.4":
                with pytest.raises(Exception, match="client disconnected|^$"):
                    await response({"type": "http", "asgi": {"spec_version": spec_version}}, receive, send)
            else:
                await response({"type": "http", "asgi": {"spec_version": spec_version}}, receive, send)

    asyncio.run(run())
    assert stream.closed
    _assert_no_load(scheduler)


@pytest.mark.parametrize("recompute", [None, "before_output", "after_output"])
@pytest.mark.parametrize("streaming", [False, True])
def test_responses_server_tools_do_not_retry_recompute(installed_runtime, recompute, streaming):
    scheduler, prefill, _ = installed_runtime
    runtime = proxy.get_runtime()
    api = "/responses"
    options = _structured_options(api, "mcp")
    native = {**native_request(api, streaming), **options}
    result, events = _structured_attempt(api, "mcp", "tool-attempt", interrupted=False)
    if recompute is None:
        chunks = _frame_chunks(api, events) if streaming else [json.dumps(result).encode()]
    else:
        # Zero output in a later generation step can follow a completed tool call.
        progress = (
            {"output_tokens": 5, "remaining_tokens": 112, "input_tokens": 50} if recompute == "after_output" else None
        )
        error = _recompute_error(api, progress, streaming=streaming)
        chunks = _frame_chunks(api, [*_start(api, "interrupted"), error]) if streaming else [json.dumps(error).encode()]
    status = 200 if streaming or recompute is None else 500
    tool_calls = []
    opened = []

    def backend(request):
        payload = json.loads(request.content)
        assert payload["tools"] == options["tools"]
        assert scheduler.request_num == 1
        tool_calls.append(request.headers["X-Request-Id"])
        stream = Chunks(chunks)
        opened.append(stream)
        return httpx.Response(status, stream=stream)

    async def run():
        async with httpx.AsyncClient(base_url="http://decoder/v1", transport=httpx.MockTransport(backend)) as client:
            runtime._clients[proxy.ServerRole.DECODE][proxy.server_key("127.0.0.1", 8200)] = client
            response = await proxy.handle_completions_impl(api, _request(native))
            return response.status_code, await _body(response)

    status_code, raw = asyncio.run(run())
    assert len(tool_calls) == len(prefill.posts) == 1
    assert opened[0].closed
    _assert_no_load(scheduler)
    if recompute is None:
        assert status_code == 200
        response = _events(raw)[-1]["response"] if streaming else json.loads(raw)
        assert response["output"] == result["output"]
    else:
        assert status_code == 502
        error = json.loads(raw)["error"]
        assert error["type"] == "server_error"
        assert error["code"] == "ascend_pd_recompute_tool_replay_blocked"
        assert "server-side tool call" in error["message"]


@pytest.mark.parametrize("streaming", [False, True])
def test_cancellation_while_retrying_prefill_releases_load(installed_runtime, monkeypatch, streaming):
    scheduler, prefill, _ = installed_runtime
    runtime = proxy.get_runtime()
    err = _error("/responses", proxy.PD_RECOMPUTE_RETRY, streaming=streaming)
    stream = Chunks([_event(err) if streaming else json.dumps(err).encode()])
    initial_post = prefill.post

    async def run():
        retry_started = asyncio.Event()

        async def post(*args, **kwargs):
            if prefill.posts:
                retry_started.set()
                await asyncio.Event().wait()
            return await initial_post(*args, **kwargs)

        monkeypatch.setattr(prefill, "post", post)
        async with httpx.AsyncClient(
            base_url="http://decoder/v1",
            transport=httpx.MockTransport(lambda req: httpx.Response(200 if streaming else 500, stream=stream)),
        ) as client:
            runtime._clients[proxy.ServerRole.DECODE][proxy.server_key("127.0.0.1", 8200)] = client
            task = asyncio.create_task(
                proxy.handle_completions_impl("/responses", _request(native_request("/responses", streaming)))
            )
            try:
                await asyncio.wait_for(retry_started.wait(), timeout=2)
                assert stream.closed
                assert all(entry.active_tokens == 0 for entry in scheduler.decoders.values())
                assert scheduler.request_num == 1
            finally:
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task

    asyncio.run(run())
    _assert_no_load(scheduler)


@pytest.mark.parametrize("streaming", [False, True])
def test_native_decode_transport_truncation_before_commit_returns_502(installed_runtime, streaming):
    scheduler, prefill, _ = installed_runtime
    runtime = proxy.get_runtime()
    partial = b"event: response.created\ndata: {" if streaming else b'{"object":"response",'
    stream = Chunks([partial], fail=True)

    async def run():
        async with httpx.AsyncClient(
            base_url="http://decoder/v1", transport=httpx.MockTransport(lambda req: httpx.Response(200, stream=stream))
        ) as client:
            runtime._clients[proxy.ServerRole.DECODE][proxy.server_key("127.0.0.1", 8200)] = client
            original = _request(native_request("/responses", streaming))
            response = await proxy.handle_completions_impl("/responses", original)
            assert response.status_code == 502
            assert json.loads(await _body(response))["error"]["code"] == "backend_unavailable"

    asyncio.run(run())
    assert stream.closed
    assert len(prefill.posts) == 1
    _assert_no_load(scheduler)
