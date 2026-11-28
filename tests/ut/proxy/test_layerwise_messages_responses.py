# SPDX-License-Identifier: Apache-2.0
import asyncio
import importlib.util
import json
import logging
import sys
from contextlib import AsyncExitStack, asynccontextmanager
from pathlib import Path
from types import ModuleType, SimpleNamespace

import httpx
import pytest

from examples.disaggregated_prefill_v1.layerwise_response_protocol import TextContinuation


@pytest.fixture
def proxy(monkeypatch):
    # The ASGI proxy only uses the vLLM logger and a connector wire key.
    # Isolate those imports so these HTTP tests also run on a CPU workstation.
    logger_module = ModuleType("vllm.logger")
    logger_module.init_logger = logging.getLogger
    protocol_name = "vllm_ascend.distributed.kv_transfer.kv_p2p.sfa_pd_rd2h.protocol"
    protocol = ModuleType(protocol_name)
    protocol.BATCH_KV_TRANSFER_PARAMS = "batch_kv_transfer_params"
    package = "examples.disaggregated_prefill_v1"
    name = "load_balance_proxy_layerwise_server_example"
    path = Path(__file__).resolve().parents[3] / "examples/disaggregated_prefill_v1" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"{package}.{name}", path)
    module = importlib.util.module_from_spec(spec)
    with monkeypatch.context() as patch:
        patch.setitem(sys.modules, logger_module.__name__, logger_module)
        patch.setitem(sys.modules, protocol_name, protocol)
        spec.loader.exec_module(module)
    module.global_args = SimpleNamespace(host="proxy", port=8000, max_retries=2, retry_delay=0)
    return module


@asynccontextmanager
async def backends(proxy, decode):
    prefill_requests = []
    decode_requests = []
    state = proxy.proxy_state = proxy.ProxyState([("p", 80)], [("d", 80)])
    async with AsyncExitStack() as stack:
        client = await stack.enter_async_context(httpx.AsyncClient(transport=httpx.ASGITransport(app=proxy.app)))

        async def prefill(request):
            prefill_requests.append((request.headers["x-request-id"], json.loads(request.content)))
            return httpx.Response(200, json={})

        async def decoder(request):
            body = json.loads(request.content)
            request_id = request.headers["x-request-id"]
            decode_requests.append((request_id, body))
            callback_id = (
                body["request_id"] + "_0" if request.url.path.endswith("/responses") else "chatcmpl-" + request_id
            )
            callback = await client.post(
                body["kv_transfer_params"]["metaserver"],
                json={"request_id": callback_id, "do_remote_decode": True, "remote_block_ids": [len(decode_requests)]},
            )
            assert callback.status_code == 200, callback.text
            return decode(body, len(decode_requests))

        for server, handler in [(state.prefillers[0], prefill), (state.decoders[0], decoder)]:
            await server.client.aclose()
            server.client = await stack.enter_async_context(
                httpx.AsyncClient(base_url=server.url, transport=httpx.MockTransport(handler))
            )
        yield client, prefill_requests, decode_requests
    assert not state.req_data_dict and not state.metaserver_params and not state.metaserver_dispatch_tasks
    assert (
        state.prefillers[0].active_tokens == state.decoders[0].active_tokens == state.prefillers[0].active_kv_cache == 0
    )


def output_item(text):
    return {"type": "message", "content": [{"type": "output_text", "text": text}]}


@pytest.mark.parametrize("api", ["messages", "responses"])
def test_json_recompute_with_fresh_pd_callback(proxy, api):
    messages = api == "messages"
    key = "messages" if messages else "input"
    limit = "max_tokens" if messages else "max_output_tokens"
    history = [{"role": "user", "content": []}, {"role": "assistant", "content": "history"}]

    def decode(body, attempt):
        text = ["a", "b", "c"][attempt - 1]
        payload = {"stop_reason": "recomputed" if attempt < 3 else "stop", "usage": {"output_tokens": 2}}
        if messages:
            payload["content"] = [{"type": "text", "text": text}]
        else:
            payload.update(id=body["request_id"], output=[output_item(text)])
        return httpx.Response(200, json=payload)

    async def run():
        async with backends(proxy, decode) as (client, prefill, decoder):
            response = await client.post(f"http://proxy/v1/{api}", json={key: history, limit: 12})
            assert response.status_code == 200
            result = response.json()
            texts = (
                [x["text"] for x in result["content"]]
                if messages
                else [x["content"][0]["text"] for x in result["output"]]
            )
            assert texts == ["a", "b", "c"]
            assert result["usage"] == {"output_tokens": 2}
            assert len({rid for rid, _ in decoder}) == 3
            assert [rid for rid, _ in decoder] == [rid for rid, _ in prefill]
            if not messages:
                assert result["id"] == decoder[0][1]["request_id"]
            for i, ((_, d), (_, p)) in enumerate(zip(decoder, prefill)):
                assert d[limit] == 12 - 2 * i
                assert d["kv_transfer_params"]["do_remote_prefill"] is True
                assert p[limit] == 1 and p["stream"] is False
                assert "min_tokens" not in p
                if not messages:
                    assert "max_tokens" not in p
                    assert p["request_id"] == d["request_id"]
                assert p["kv_transfer_params"]["remote_block_ids"] == [i + 1]
                assert p[key] == d[key]
                assert p[key][1] == history[1]
                if i:
                    assert p[key][0]["content"] == [{"type": "text" if messages else "input_text", "text": "ab"[:i]}]

    asyncio.run(run())


class SplitStream(httpx.AsyncByteStream):
    def __init__(self, events):
        self.closed = False
        self.data = b"".join(
            b"event: "
            + event["type"].encode()
            + b"\r\ndata: "
            + json.dumps(event, ensure_ascii=False).encode()
            + b"\r\n\r\n"
            for event in events
        )

    async def __aiter__(self):
        # Split a UTF-8 character; coalesce subsequent events in the last read.
        split = self.data.index("你".encode()) + 1
        yield self.data[:split]
        yield self.data[split:]

    async def aclose(self):
        self.closed = True


@pytest.mark.parametrize("api", ["messages", "responses"])
def test_sse_recompute(proxy, api):
    streams = []
    text = "你\u2028好"
    messages = api == "messages"

    def decode(body, attempt):
        if attempt == 2:
            assert streams[0].closed
            assert (body["messages"][0]["content"] if messages else body["input"]) == "Q" + text
        retry = attempt == 1
        if messages:
            events = [
                {"type": "message_start"},
                {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
                {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": text}},
                {"type": "content_block_stop", "index": 0},
                {
                    "type": "message_delta",
                    "delta": {
                        "stop_reason": "stop_sequence" if retry else "end_turn",
                        "stop_sequence": "recomputed" if retry else None,
                    },
                    "usage": {"output_tokens": 1},
                },
                {"type": "message_stop"},
            ]
        else:
            result = {"id": body["request_id"], "output": [output_item(text)], "usage": {"output_tokens": 1}}
            if retry:
                result["stop_reason"] = "recomputed"
            events = [
                {"type": "response.created", "response": {"id": body["request_id"]}},
                {"type": "response.in_progress", "response": {"id": body["request_id"]}},
                {"type": "response.output_text.delta", "output_index": 0, "delta": text},
                {"type": "response.completed", "response": result},
            ]
            for i, event in enumerate(events):
                event["sequence_number"] = i
        streams.append(SplitStream(events))
        return httpx.Response(200, stream=streams[-1], headers={"content-type": "text/event-stream"})

    async def run():
        async with backends(proxy, decode) as (client, _, decoder):
            body = {"messages": [{"role": "user", "content": "Q"}], "max_tokens": 3} if messages else {"input": "Q"}
            response = await client.post(f"http://proxy/v1/{api}", json={**body, "stream": True})
            assert response.status_code == 200 and response.headers["content-type"].startswith("text/event-stream")
            events = [json.loads(line[6:]) for line in response.content.split(b"\n") if line.startswith(b"data: ")]
            assert b"recomputed" not in response.content
            if messages:
                assert [e["type"] for e in events].count("message_start") == 1
                assert [e["type"] for e in events].count("message_stop") == 1
                assert [e["index"] for e in events if e["type"] == "content_block_delta"] == [0, 1]
            else:
                assert [e["sequence_number"] for e in events] == list(range(5))
                assert events[-1]["response"]["id"] == events[0]["response"]["id"]
                assert events[-1]["response"]["output"] == [output_item(text)] * 2
                assert [e["output_index"] for e in events if "output_index" in e] == [0, 1]
                assert "max_output_tokens" not in decoder[-1][1]
            assert len(streams) == 2 and all(stream.closed for stream in streams)

    asyncio.run(run())


@pytest.mark.parametrize(
    "option", [{"background": True}, {"previous_response_id": "old"}, {"tools": [{"type": "mcp"}]}]
)
def test_reject_before_pd_scheduling(proxy, option):
    async def run():
        async with backends(proxy, None) as (client, prefill, decoder):
            reply = await client.post("http://proxy/v1/responses", json={"input": "hi", **option})
            assert reply.status_code == 400
            assert not decoder and not prefill

    asyncio.run(run())


def test_explicit_stop_sequence_and_structured_output():
    result = {"content": [{"type": "tool_use"}], "stop_reason": "stop_sequence", "stop_sequence": "recomputed"}
    frame = json.dumps(result).encode()
    state = TextContinuation("/messages", {"stop_sequences": ["recomputed"]})
    state.begin_attempt()
    assert state.forward(frame) == frame
    state = TextContinuation("/messages", {})
    state.begin_attempt()
    with pytest.raises(ValueError, match="text-only"):
        state.forward(frame)


@pytest.mark.parametrize("status", [422, 503])
def test_backend_failure_and_transport_retry(proxy, status):
    def decode(body, attempt):
        if attempt == 1:
            return httpx.Response(status, json={"error": "backend error"})
        return httpx.Response(200, json={"id": body["request_id"], "output": []})

    async def run():
        async with backends(proxy, decode) as (client, _, decoder):
            reply = await client.post("http://proxy/v1/responses", json={"input": "Q"})
            if status == 422:
                assert reply.status_code == 422 and reply.json() == {"error": "backend error"}
                assert len(decoder) == 1
            else:
                assert reply.status_code == 200 and len(decoder) == 2
                assert decoder[0][0] != decoder[1][0]
                stale = await client.post(
                    "http://proxy/v1/metaserver", json={"request_id": decoder[0][1]["request_id"] + "_0"}
                )
                assert stale.status_code == 404

    asyncio.run(run())


def test_stream_delivered_before_backend_finishes_and_disconnect_releases_attempt(proxy):
    async def run():
        delivered = asyncio.Event()
        closed = asyncio.Event()

        class WaitingStream(httpx.AsyncByteStream):
            async def __aiter__(self):
                yield b'event: message_start\ndata: {"type":"message_start"}\n\n'
                await asyncio.Event().wait()

            async def aclose(self):
                closed.set()

        async with backends(proxy, lambda *_: httpx.Response(200, stream=WaitingStream())):
            scope = {"type": "http", "method": "POST", "path": "/v1/messages", "query_string": b"", "headers": []}
            pending_body = True

            async def receive():
                nonlocal pending_body
                if pending_body:
                    pending_body = False
                    return {"type": "http.request", "body": b'{"messages":[],"max_tokens":2,"stream":true}'}
                await delivered.wait()
                return {"type": "http.disconnect"}

            async def send(message):
                if message["type"] == "http.response.body" and b"message_start" in message["body"]:
                    delivered.set()

            await asyncio.wait_for(proxy.app(scope, receive, send), timeout=5)
            assert delivered.is_set() and closed.is_set()

    asyncio.run(run())


def test_responses_tool_result_cannot_be_extended():
    state = TextContinuation("/responses", {"input": [{"type": "function_call_output", "output": "done"}]})
    state.begin_attempt()
    with pytest.raises(ValueError, match="first message with content"):
        state.forward(json.dumps({"output": [output_item("prefix")], "stop_reason": "recomputed"}).encode())
