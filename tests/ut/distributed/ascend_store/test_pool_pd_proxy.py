# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib.util
import json
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def create_app():
    path = Path(__file__).resolve().parents[4] / "examples/pool_pd/proxy.py"
    spec = importlib.util.spec_from_file_location("pool_pd_proxy", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.create_app


def test_proxy_hands_ready_snapshot_to_decoder_and_preserves_generation(create_app):
    calls = []
    params = {"pool_pd": {"status": "ready", "transfer_id": "test"}}

    def backend(request):
        calls.append((request.url.host, json.loads(request.content)))
        if request.url.host == "prefill":
            return httpx.Response(200, json={"kv_transfer_params": params, "choices": [{"text": "discard"}]})
        return httpx.Response(200, json={"choices": [{"text": "decode result"}]})

    app = create_app("http://prefill", "http://decode", httpx.MockTransport(backend))
    with TestClient(app) as client:
        result = client.post("/v1/completions", json={"prompt": [1, 2, 3], "max_tokens": 32, "temperature": 0})
    assert result.status_code == 200
    assert result.json()["choices"][0]["text"] == "decode result"
    assert [host for host, _ in calls] == ["prefill", "decode"]
    assert calls[0][1]["max_tokens"] == 1
    assert calls[0][1]["ignore_eos"]
    assert calls[1][1] == {"prompt": [1, 2, 3], "max_tokens": 32, "temperature": 0, "kv_transfer_params": params}


@pytest.mark.parametrize("prefill_body", [{}, {"kv_transfer_params": {"pool_pd": {"status": "failed"}}}])
def test_proxy_does_not_decode_before_snapshot_is_ready(create_app, prefill_body):
    calls = []

    def backend(request):
        calls.append(request.url.host)
        return httpx.Response(200, json=prefill_body)

    app = create_app("http://prefill", "http://decode", httpx.MockTransport(backend))
    with TestClient(app) as client:
        result = client.post("/v1/completions", json={"prompt": "hello", "max_tokens": 8})
    assert result.status_code == 502
    assert calls == ["prefill"]


@pytest.mark.parametrize("status", [200, 503])
def test_streaming_handoff_preserves_decoder_events_and_errors(create_app, status):
    calls = []
    params = {"pool_pd": {"status": "ready"}}
    events = b'data: {"choices":[{"text":"hello"}]}\n\ndata: [DONE]\n\n'

    def backend(request):
        body = json.loads(request.content)
        calls.append(body)
        if request.url.host == "prefill":
            return httpx.Response(200, json={"kv_transfer_params": params})
        assert body["kv_transfer_params"] == params
        return httpx.Response(status, content=events if status == 200 else b'{"error":"busy"}')

    app = create_app("http://prefill", "http://decode", httpx.MockTransport(backend))
    with TestClient(app) as client:
        response = client.post(
            "/v1/completions",
            json={"prompt": "hello", "max_tokens": 8, "stream": True, "stream_options": {"include_usage": True}},
        )
    assert response.status_code == status
    assert calls[0]["stream"] is False
    assert "stream_options" not in calls[0]
    assert calls[1]["stream"] is True
    if status == 200:
        assert response.content == events
        assert response.headers["content-type"].startswith("text/event-stream")
    else:
        assert response.json() == {"error": "busy"}
