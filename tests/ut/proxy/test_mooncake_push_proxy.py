# SPDX-License-Identifier: Apache-2.0

import asyncio
import importlib.util
import json
from pathlib import Path
from uuid import UUID

import httpx
import pytest

PROXY_PATH = Path(__file__).parents[3] / "examples/disaggregated_prefill_v1/mooncake_push_proxy.py"
SPEC = importlib.util.spec_from_file_location("mooncake_push_proxy", PROXY_PATH)
assert SPEC is not None and SPEC.loader is not None
PROXY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROXY)


def test_submit_both_legs_before_prefill_returns_and_relay_only_decode():
    async def run():
        prefill_started = asyncio.Event()
        decode_started = asyncio.Event()
        calls = []

        async def backend(request):
            body = json.loads(request.content)
            calls.append((request.url.host, body))
            if request.url.host == "prefill":
                prefill_started.set()
                await asyncio.wait_for(decode_started.wait(), 1)
                return httpx.Response(200, json={"choices": [{"text": "discard-this-p-token"}]})
            decode_started.set()
            await asyncio.wait_for(prefill_started.wait(), 1)
            return httpx.Response(200, json={"choices": [{"text": "decode-output"}]})

        app = PROXY.create_app("http://prefill", "http://decode", "p-kv", 5600, "p")
        async with (
            httpx.AsyncClient(transport=httpx.MockTransport(backend)) as backend_client,
            httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://proxy") as client,
        ):
            app.state.client = backend_client
            response = await client.post(
                "/v1/completions", json={"model": "model", "prompt": "prompt", "max_tokens": 3}
            )
            assert response.json()["choices"][0]["text"] == "decode-output"
            producer = next(body for host, body in calls if host == "prefill")
            consumer = next(body for host, body in calls if host == "decode")
            assert producer["max_tokens"] == 1
            assert consumer["max_tokens"] == 3
            assert (
                producer["kv_transfer_params"]["push_transfer_id"] == consumer["kv_transfer_params"]["push_transfer_id"]
            )
            UUID(consumer["kv_transfer_params"]["push_transfer_id"])
            assert "last_token_id" not in consumer["kv_transfer_params"]

    asyncio.run(run())


def test_prefill_http_failure_cancels_waiting_decode():
    async def run():
        decode_cancelled = asyncio.Event()

        async def backend(request):
            if request.url.host == "prefill":
                await asyncio.sleep(0.01)
                return httpx.Response(503, json={"error": "prefill unavailable"})
            try:
                await asyncio.Event().wait()
            finally:
                decode_cancelled.set()

        app = PROXY.create_app("http://prefill", "http://decode", "p-kv", 5600, "p")
        async with (
            httpx.AsyncClient(transport=httpx.MockTransport(backend)) as backend_client,
            httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://proxy") as client,
        ):
            app.state.client = backend_client
            response = await client.post("/v1/completions", json={"prompt": "prompt"})
            assert response.status_code == 503
            assert decode_cancelled.is_set()

    asyncio.run(run())


@pytest.mark.parametrize("body", [{"n": 2}, {"best_of": 2}, {"prompt": ["a", "b"]}, {"kv_transfer_params": {}}])
def test_reject_ambiguous_request_identity(body):
    with pytest.raises(ValueError):
        PROXY.build_requests(body, "id", "p", 5600, "p", 1)


def test_each_attempt_can_use_a_new_identity_without_changing_prompt():
    for key in ("attempt-a", "attempt-b"):
        p, d = PROXY.build_requests({"prompt": [1, 2, 3]}, key, "p", 5600, "p", 1)
        assert p["prompt"] == d["prompt"] == [1, 2, 3]
        assert d["kv_transfer_params"]["push_transfer_id"] == key


def test_disconnect_before_backend_headers_cancels_both_legs():
    async def run():
        started = {host: asyncio.Event() for host in ("prefill", "decode")}
        cancelled = set()

        async def backend(request):
            host = request.url.host
            started[host].set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.add(host)

        app = PROXY.create_app("http://prefill", "http://decode", "p-kv", 5600, "p")
        messages = asyncio.Queue()
        await messages.put(
            {
                "type": "http.request",
                "body": json.dumps({"prompt": "prompt"}).encode(),
                "more_body": False,
            }
        )

        async def send(_message):
            pass

        scope = {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": "/v1/completions",
            "raw_path": b"/v1/completions",
            "query_string": b"",
            "headers": [],
            "client": ("127.0.0.1", 1234),
            "server": ("proxy", 80),
        }
        async with httpx.AsyncClient(transport=httpx.MockTransport(backend)) as client:
            app.state.client = client
            task = asyncio.create_task(app(scope, messages.get, send))
            await asyncio.wait_for(asyncio.gather(*(event.wait() for event in started.values())), 1)
            await messages.put({"type": "http.disconnect"})
            result = await asyncio.wait_for(asyncio.gather(task, return_exceptions=True), 1)
            assert isinstance(result[0], asyncio.CancelledError)
            assert cancelled == {"prefill", "decode"}

    asyncio.run(run())


def test_decode_http_failure_cancels_prefill():
    async def run():
        cancelled = asyncio.Event()

        async def backend(request):
            if request.url.host == "decode":
                return httpx.Response(502, json={"error": "decode unavailable"})
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

        app = PROXY.create_app("http://prefill", "http://decode", "p-kv", 5600, "p")
        async with (
            httpx.AsyncClient(transport=httpx.MockTransport(backend)) as backend_client,
            httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://proxy") as client,
        ):
            app.state.client = backend_client
            response = await client.post("/v1/completions", json={"prompt": "prompt"})
            assert response.status_code == 502
            assert cancelled.is_set()

    asyncio.run(run())
