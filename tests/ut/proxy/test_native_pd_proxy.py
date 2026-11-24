"""Native PD proxy retries and pass-through, using a real scheduler and mock P/D endpoints."""

import argparse
import asyncio
import json
from contextlib import AsyncExitStack
from unittest.mock import Mock

import httpx
import pytest

from examples.disaggregated_prefill_v1 import load_balance_proxy_server_example as proxy


def _message(text, interrupted, streaming):
    reason = (
        {"stop_reason": "stop_sequence", "stop_sequence": "recomputed"} if interrupted else {"stop_reason": "end_turn"}
    )
    usage = {"output_tokens": 3}
    if not streaming:
        return httpx.Response(200, json={"content": [{"type": "text", "text": text}], "usage": usage, **reason})
    events = [
        {"type": "message_start", "message": {"id": text, "content": []}},
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": text}},
        {"type": "content_block_stop", "index": 0},
        {"type": "message_delta", "delta": reason, "usage": usage},
        {"type": "message_stop"},
    ]
    return httpx.Response(200, content=b"".join(_frame(e) for e in events))


def _frame(event):
    return f"event: {event['type']}\ndata: {json.dumps(event)}\n\n".encode()


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("case", ["recompute", "user-stop", "responses-incomplete"])
def test_native_decode(monkeypatch, case, streaming):
    scheduler = proxy.SharedProxyScheduler([("p", 8100)], [("d", 8200)])
    runtime = proxy.WorkerRuntime(scheduler)
    monkeypatch.setattr(proxy, "runtime", runtime)
    monkeypatch.setattr(
        proxy, "global_args", argparse.Namespace(enable_prefill_affinity=False, max_retries=1, retry_delay=0)
    )
    release = Mock(wraps=scheduler._release_load)
    monkeypatch.setattr(scheduler, "_release_load", release)
    routes: list[int] = []
    incomplete = {"id": "r", "status": "incomplete", "incomplete_details": {"reason": "max_output_tokens"}}
    event = {"type": "response.incomplete", "response": incomplete}

    def backend(request):
        port = request.url.port
        routes.append(port)
        body = json.loads(request.content)
        assert scheduler.request_num == (2 if port in (8101, 8201) else 1)
        if port in (8100, 8101):
            return httpx.Response(200, json={"kv_transfer_params": {"remote_block_ids": [port]}})
        if case == "responses-incomplete":
            return httpx.Response(200, content=_frame(event)) if streaming else httpx.Response(200, json=incomplete)
        if port == 8200:
            if case == "recompute":
                # An unrelated request keeps the original pair busy during reselection.
                scheduler.begin_request(1000)
                scheduler.pick_decoder(1000)
                scheduler.activate_waiting_instance(proxy.ServerRole.PREFILL, "p", 8101)
                scheduler.activate_waiting_instance(proxy.ServerRole.DECODE, "d", 8201)
            return _message("first", True, streaming)
        assert body["max_tokens"] == 7
        assert body["messages"][-1] == {"role": "assistant", "content": "first"}
        assert body["kv_transfer_params"]["remote_block_ids"] == [8101]
        return _message("second", False, streaming)

    async def run():
        async with AsyncExitStack() as stack:
            for role, host, ports in [
                (proxy.ServerRole.PREFILL, "p", (8100, 8101)),
                (proxy.ServerRole.DECODE, "d", (8200, 8201)),
            ]:
                for port in ports:
                    client = httpx.AsyncClient(
                        base_url=f"http://{host}:{port}/v1", transport=httpx.MockTransport(backend)
                    )
                    runtime._clients[role][proxy.server_key(host, port)] = await stack.enter_async_context(client)
            client = await stack.enter_async_context(
                httpx.AsyncClient(base_url="http://proxy", transport=httpx.ASGITransport(app=proxy.app))
            )
            if case == "responses-incomplete":
                return await client.post("/v1/responses", json={"model": "m", "input": "hello", "stream": streaming})
            request = {
                "model": "m",
                "messages": [{"role": "user", "content": "hello"}],
                "max_tokens": 10,
                "stream": streaming,
            }
            if case == "user-stop":
                request["stop_sequences"] = ["recomputed"]
            return await client.post("/v1/messages", json=request)

    response = asyncio.run(run())
    assert response.status_code == 200
    expected_ports = [8100, 8200, 8101, 8201] if case == "recompute" else [8100, 8200]
    assert routes == expected_ports
    assert [call.args[1] for call in release.call_args_list] == [
        proxy.server_key("p" if port < 8200 else "d", port) for port in expected_ports
    ]
    assert scheduler.request_num == int(case == "recompute")
    if case == "responses-incomplete":
        assert response.content == _frame(event) if streaming else response.json() == incomplete
    elif streaming:
        events = [json.loads(line[5:]) for line in response.content.splitlines() if line.startswith(b"data:")]
        texts = [e["delta"]["text"] for e in events if e["type"] == "content_block_delta"]
        assert texts == (["first", "second"] if case == "recompute" else ["first"])
        assert events[-1]["type"] == "message_stop"
    else:
        assert response.json()["content"][0]["text"] == ("firstsecond" if case == "recompute" else "first")
