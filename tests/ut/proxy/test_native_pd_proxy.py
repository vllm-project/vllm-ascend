"""Native PD proxy retries and pass-through, using a real scheduler and mock endpoints."""

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


class BackendBody(httpx.AsyncByteStream):
    def __init__(self, data: bytes):
        self.data = data

    async def __aiter__(self):
        # Split UTF-8, SSE delimiters and JSON across network reads.
        for offset in range(0, len(self.data), 7):
            yield self.data[offset : offset + 7]


def _frame(event: dict) -> bytes:
    data = json.dumps(event, ensure_ascii=False, indent=2).encode().splitlines()
    return (
        b": upstream\r\nid: event-1\r\nretry: 2000\r\nevent: "
        + event["type"].encode()
        + b"\r\n"
        + b"".join(b"data: " + line + b"\r\n" for line in data)
        + b"\r\n"
    )


def _events(response: httpx.Response) -> list[dict]:
    frames = response.content.replace(b"\r\n", b"\n").split(b"\n\n")
    return [
        json.loads(b"\n".join(line[5:].lstrip() for line in frame.splitlines() if line.startswith(b"data:")))
        for frame in frames
        if b"data:" in frame
    ]


def _backend_response(api, streaming, result, *, omit_final_output=False, sequence_start=0):
    if not streaming:
        return httpx.Response(200, stream=BackendBody(json.dumps(result, ensure_ascii=False).encode()))
    if api == "messages":
        events: list[dict] = [{"type": "message_start", "message": {"id": result["id"], "content": []}}]
        for index, block in enumerate(result["content"]):
            events.append({"type": "content_block_start", "index": index, "content_block": {**block, "text": ""}})
            if block["type"] == "text":
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
                "delta": {key: result[key] for key in ("stop_reason", "stop_sequence") if key in result},
                "usage": result.get("usage"),
            }
        )
        events.append({"type": "message_stop"})
    else:
        events = [{"type": "response.created", "response": {"id": result["id"], "output": []}}]
        for index, item in enumerate(result["output"]):
            events.append(
                {"type": "response.output_item.added", "output_index": index, "item": {**item, "content": []}}
            )
            for part_index, part in enumerate(item.get("content", [])):
                events.append(
                    {
                        "type": "response.output_text.delta",
                        "output_index": index,
                        "item_id": item["id"],
                        "content_index": part_index,
                        "delta": part["text"],
                    }
                )
            events.append({"type": "response.output_item.done", "output_index": index, "item": item})
        final = {**result, "output": []} if omit_final_output else result
        events.append({"type": f"response.{result['status']}", "response": final})
        for number, event in enumerate(events, sequence_start):
            event["sequence_number"] = number
    return httpx.Response(200, stream=BackendBody(b"".join(_frame(event) for event in events)))


def _text_result(api, text, number, stop=None):
    result: dict[str, Any] = {"id": f"reply-{number}", "usage": {"input_tokens": 10, "output_tokens": 3}}
    if api == "messages":
        result.update(content=[{"type": "text", "text": text}], stop_reason=stop or "end_turn")
    else:
        result.update(
            status="completed",
            output=[
                {
                    "id": f"item-{number}",
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": text}],
                }
            ],
        )
        if stop:
            result["stop_reason"] = stop
    return result


@pytest.fixture
def exchange(monkeypatch):
    async def run(api, params, replies, *, reselect=False, disconnect_at_headers=False):
        scheduler = proxy.SharedProxyScheduler([("p", 8100)], [("d", 8200)])
        runtime = proxy.WorkerRuntime(scheduler)
        monkeypatch.setattr(proxy, "runtime", runtime)
        monkeypatch.setattr(
            proxy, "global_args", argparse.Namespace(enable_prefill_affinity=False, max_retries=1, retry_delay=0)
        )
        release = Mock(wraps=scheduler._release_load)
        monkeypatch.setattr(scheduler, "_release_load", release)
        calls: list[tuple[int, dict]] = []
        decoder_calls = 0

        def backend(request):
            nonlocal decoder_calls
            port = request.url.port
            calls.append((port, json.loads(request.content)))
            assert request.url.path == f"/v1/{api}"
            assert scheduler.request_num == 1 + int(reselect and decoder_calls > 0)
            if port < 8200:
                if decoder_calls:
                    assert replies[decoder_calls - 1].is_closed
                cached = 2 if port == 8100 else 3
                usage: dict[str, Any] = (
                    {"cache_read_input_tokens": cached}
                    if api == "messages"
                    else {"input_tokens_details": {"cached_tokens": cached}}
                )
                return httpx.Response(200, json={"kv_transfer_params": {"remote_block_ids": [port]}, "usage": usage})
            reply = replies[decoder_calls]
            decoder_calls += 1
            if reselect and decoder_calls == 1:
                # Keep the original pair busy with a separate request during reselection.
                scheduler.begin_request(1000)
                scheduler.pick_decoder(1000)
                scheduler.activate_waiting_instance(proxy.ServerRole.PREFILL, "p", 8101)
                scheduler.activate_waiting_instance(proxy.ServerRole.DECODE, "d", 8201)
            return reply

        async def disconnect_on_headers(scope, receive, send):
            scope["asgi"]["spec_version"] = "2.4"

            async def disconnect(message):
                raise OSError("client disconnected before the response body")

            await proxy.app(scope, receive, disconnect)

        async with AsyncExitStack() as stack:
            for role, host, ports in (
                (proxy.ServerRole.PREFILL, "p", (8100, 8101)),
                (proxy.ServerRole.DECODE, "d", (8200, 8201)),
            ):
                for port in ports:
                    client = httpx.AsyncClient(
                        base_url=f"http://{host}:{port}/v1", transport=httpx.MockTransport(backend)
                    )
                    runtime._clients[role][proxy.server_key(host, port)] = await stack.enter_async_context(client)
            client = await stack.enter_async_context(
                httpx.AsyncClient(
                    base_url="http://proxy",
                    transport=httpx.ASGITransport(app=disconnect_on_headers if disconnect_at_headers else proxy.app),
                )
            )
            response = None
            if disconnect_at_headers:
                with pytest.raises(ClientDisconnect):
                    await client.post(f"/v1/{api}", json={"model": "m", **params})
            else:
                response = await client.post(f"/v1/{api}", json={"model": "m", **params})

        assert [call.args[1] for call in release.call_args_list if call.args[1] is not None] == [
            proxy.server_key("p" if port < 8200 else "d", port) for port, _ in calls
        ]
        assert all(reply.is_closed for reply in replies[:decoder_calls])
        assert scheduler.request_num == int(reselect)
        if reselect:
            scheduler.finish_request("p:8100", 1000, "d:8200", 1000, release_prefill_kv=True)
        assert scheduler.request_num == 0
        assert all(entry.active_kv_cache == pytest.approx(0) for entry in scheduler.prefillers.values())
        assert all(entry.active_tokens == pytest.approx(0) for entry in scheduler.decoders.values())
        return response, calls

    return lambda *args, **kwargs: asyncio.run(run(*args, **kwargs))


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize(
    "api,array_content",
    [("messages", False), ("messages", True), ("responses", False), ("responses", True)],
)
def test_recompute_reselects_and_continues(exchange, api, array_content, streaming):
    messages = api == "messages"
    input_key, budget_key = ("messages", "max_tokens") if messages else ("input", "max_output_tokens")
    part_type = "text" if messages else "input_text"
    original = [{"type": part_type, "text": "hi "}] if array_content else "hi "
    history: list[dict[str, Any]] = [{"role": "user", "content": original}, {"role": "assistant", "content": "history"}]
    original_input = history if messages or array_content else original
    results = [_text_result(api, text, i, "recomputed" if i < 2 else None) for i, text in enumerate(("你", "好", "！"))]
    if messages:
        results[0].update(stop_reason="stop_sequence", stop_sequence="recomputed")
    else:
        results[0]["status"] = "incomplete"
    first_sequence = 10 if not messages and not array_content else 0
    response, calls = exchange(
        api,
        {"stream": streaming, input_key: original_input, budget_key: 10},
        [
            _backend_response(
                api,
                streaming,
                result,
                omit_final_output=not messages and i == 0 and array_content,
                sequence_start=first_sequence if i == 0 else 0,
            )
            for i, result in enumerate(results)
        ],
        reselect=True,
    )
    assert response.status_code == 200
    assert [port for port, _ in calls] == [8100, 8200, 8101, 8201, 8101, 8201]
    for (p_port, p_body), (_, d_body), prefix, budget in zip(calls[::2], calls[1::2], ("", "你", "你好"), (10, 7, 4)):
        content: list[dict[str, str]] | str
        if isinstance(original, list):
            content = original + ([{"type": part_type, "text": prefix}] if prefix else [])
        else:
            content = original + prefix
        expected_input = (
            [{**history[0], "content": content}, history[1]] if isinstance(original_input, list) else content
        )
        assert p_body[input_key] == d_body[input_key] == expected_input
        assert p_body[budget_key] == 1
        assert d_body[budget_key] == budget
        assert d_body["kv_transfer_params"]["remote_block_ids"] == [p_port]

    if not streaming:
        final = response.json()
        blocks = final["content"] if messages else [part for item in final["output"] for part in item["content"]]
        assert [block["text"] for block in blocks] == ["你", "好", "！"]
        assert final["usage"]["output_tokens"] == 3
        if messages:
            assert final["usage"]["cache_read_input_tokens"] == 3
            assert final["usage"]["input_tokens"] == 7
        else:
            assert final["usage"]["input_tokens_details"]["cached_tokens"] == 3
        return

    events = _events(response)
    assert response.content.count(b"id: event-1\r\n") == len(events)
    assert response.content.count(b"retry: 2000\r\n") == len(events)
    kinds = [event["type"] for event in events]
    if messages:
        assert kinds.count("message_start") == kinds.count("message_delta") == kinds.count("message_stop") == 1
        assert [e["index"] for e in events if e["type"] == "content_block_start"] == [0, 1, 2]
        assert [e["delta"]["text"] for e in events if e["type"] == "content_block_delta"] == ["你", "好", "！"]
        assert events[-2]["usage"]["output_tokens"] == 3
        assert events[-2]["usage"]["cache_read_input_tokens"] == 3
    else:
        assert kinds.count("response.created") == kinds.count("response.completed") == 1
        assert [e["sequence_number"] for e in events] == list(range(first_sequence, first_sequence + len(events)))
        assert [e["output_index"] for e in events if e["type"] == "response.output_item.added"] == [0, 1, 2]
        assert [e["delta"] for e in events if e["type"] == "response.output_text.delta"] == ["你", "好", "！"]
        final = events[-1]["response"]
        assert final["id"] == events[0]["response"]["id"] == "reply-0"
        assert [item["id"] for item in final["output"]] == ["item-0", "item-1", "item-2"]
        assert final["usage"]["output_tokens"] == 3
        assert final["usage"]["input_tokens_details"]["cached_tokens"] == 3


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("api", ["messages", "responses"])
def test_native_completion_without_recompute(exchange, api, streaming):
    if api == "messages":
        result = _text_result(api, "recomputed", 0, "stop_sequence")
        result["stop_sequence"] = "recomputed"
        params = {
            "messages": [{"role": "user", "content": "hello"}],
            "stop_sequences": ["recomputed"],
            "max_tokens": 10,
        }
    else:
        result = {
            "id": "resp",
            "status": "incomplete",
            "output": [],
            "incomplete_details": {"reason": "max_output_tokens"},
            "usage": {"input_tokens_details": {"cached_tokens": 10, "cached_tokens_per_turn": [7, 3]}},
        }
        params = {"input": "hello"}
    response, calls = exchange(api, {"stream": streaming, **params}, [_backend_response(api, streaming, result)])
    assert response.status_code == 200
    assert [port for port, _ in calls] == [8100, 8200]
    if api == "messages":
        final = _events(response)[-2]["delta"] if streaming else response.json()
        assert final["stop_reason"] == "stop_sequence" and final["stop_sequence"] == "recomputed"
    else:
        final = _events(response)[-1]["response"] if streaming else response.json()
        assert final["status"] == "incomplete"
        assert final["incomplete_details"] == result["incomplete_details"]
        assert final["usage"]["input_tokens_details"] == {"cached_tokens": 5, "cached_tokens_per_turn": [2, 3]}


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("api", ["messages", "responses"])
def test_structured_recompute_closes_without_retry(exchange, caplog, api, streaming):
    result = _text_result(api, "ignored", 0, "recomputed")
    if api == "messages":
        result["content"] = [{"type": "tool_use", "name": "lookup", "id": "call"}]
        params = {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 10}
    else:
        result["output"] = [{"type": "reasoning", "id": "reasoning", "summary": []}]
        params = {"input": "hi", "max_output_tokens": 10}
    stopped = _backend_response(api, streaming, result, omit_final_output=True)
    response, calls = exchange(api, {"stream": streaming, **params}, [stopped])
    assert [port for port, _ in calls] == [8100, 8200]
    assert "text-only output" in caplog.text
    assert response.status_code == 200
    if streaming:
        assert _events(response)[-1]["type"] in ("content_block_stop", "response.output_item.done")
    else:
        assert response.content == b""


def test_responses_retry_without_explicit_budget(exchange):
    response, calls = exchange(
        "responses",
        {"input": [], "stream": False},
        [
            _backend_response("responses", False, _text_result("responses", "first", 0, "recomputed")),
            _backend_response("responses", False, _text_result("responses", "last", 1)),
        ],
    )
    assert response.status_code == 200
    assert calls[-1][1]["input"] == "first"
    assert "max_output_tokens" not in calls[-1][1]


@pytest.mark.parametrize("streaming", [False, True])
def test_backend_rejects_exhausted_recompute_budget(exchange, caplog, streaming):
    first = _backend_response("messages", streaming, _text_result("messages", "first", 0, "recomputed"))
    response, calls = exchange(
        "messages",
        {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 3, "stream": streaming},
        [first, httpx.Response(400, json={"error": {"message": "max_tokens must be positive"}})],
    )
    assert response.status_code == 200
    assert [port for port, _ in calls] == [8100, 8200, 8100, 8200]
    assert calls[-1][1]["max_tokens"] == 0
    assert "Error during streaming" in caplog.text
    if streaming:
        assert _events(response)[-1]["type"] == "content_block_stop"
    else:
        assert response.content == b""


@pytest.mark.parametrize("parameter,value", [("background", True), ("previous_response_id", "resp-old")])
def test_responses_rejects_backend_local_state(exchange, parameter, value):
    response, calls = exchange("responses", {"input": "hi", parameter: value}, [])
    assert response.status_code == 400
    assert f"does not support {parameter}." in response.json()["detail"]
    assert calls == []


def test_recompute_requires_usage_for_explicit_budget(exchange, caplog):
    result = _text_result("responses", "first", 0, "recomputed")
    del result["usage"]
    response, calls = exchange(
        "responses",
        {"input": "hi", "max_output_tokens": 10},
        [_backend_response("responses", False, result)],
    )
    assert response.content == b""
    assert [port for port, _ in calls] == [8100, 8200]
    assert "valid output_tokens count" in caplog.text


def test_releases_preopened_decode_on_client_disconnect(exchange):
    result = _text_result("responses", "first", 0)
    _, calls = exchange(
        "responses",
        {"input": "hi", "stream": True},
        [_backend_response("responses", True, result)],
        disconnect_at_headers=True,
    )
    assert [port for port, _ in calls] == [8100, 8200]


def test_responses_does_not_append_text_to_reasoning_input(exchange, caplog):
    first_item = {
        "type": "reasoning",
        "id": "reasoning",
        "summary": [],
        "content": [{"type": "reasoning_text", "text": "context"}],
    }
    result = _text_result("responses", "first", 0, "recomputed")
    response, calls = exchange("responses", {"input": [first_item]}, [_backend_response("responses", False, result)])
    assert response.content == b""
    assert [port for port, _ in calls] == [8100, 8200]
    assert "first input message with content" in caplog.text
