# SPDX-License-Identifier: Apache-2.0
"""Unit tests for load_balance_proxy_server_example.

Covers the fix for the issue where decode backend errors were swallowed into
an empty HTTP 200 response. Uses only stdlib + httpx + pytest, and stubs out
the runtime/scheduler so no real backend is required.

Run:
    pytest examples/disaggregated_prefill_v1/test_load_balance_proxy_server_example.py
"""

import asyncio
import importlib.util
import json
import os
from pathlib import Path
from unittest.mock import MagicMock

import httpx
import pytest

_HERE = Path(__file__).resolve().parent
_MODULE_PATH = _HERE / "load_balance_proxy_server_example.py"


def _load_module():
    """Load the example module from file (it is not on the normal import path)."""
    spec = importlib.util.spec_from_file_location("lb_proxy_example", _MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


lb = _load_module()

VLLM_400_BODY = json.dumps(
    {
        "error": {
            "message": (
                "This model's maximum context length is 131072 tokens. However, "
                "you requested 131064 output tokens and your prompt contains at "
                "least 9 input tokens, for a total of at least 131073 tokens."
            ),
            "type": "BadRequestError",
            "param": None,
            "code": 400,
        }
    }
).encode("utf-8")


# ---------------------------------------------------------------------------
# build_error_payload
# ---------------------------------------------------------------------------


def test_build_error_payload_upstream_4xx():
    exc = lb.DecodeUpstreamError(400, VLLM_400_BODY)
    status, obj = lb.build_error_payload(exc)
    assert status == 400
    assert obj["error"]["code"] == 400
    assert obj["error"]["type"] == "BadRequestError"
    assert "maximum context length is 131072" in obj["error"]["message"]


def test_build_error_payload_non_upstream_falls_back_to_502():
    status, obj = lb.build_error_payload(RuntimeError("connection reset"))
    assert status == 502
    assert obj["error"]["type"] == "proxy_error"
    assert obj["error"]["message"] == "connection reset"


# ---------------------------------------------------------------------------
# stream_service_response_with_retry
# ---------------------------------------------------------------------------


class _MockStreamResponse:
    def __init__(self, status_code, body=b""):
        self.status_code = status_code
        self._body = body

    async def aread(self):
        return self._body

    async def aiter_bytes(self):
        for chunk in [b'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n', b"data: [DONE]\n\n"]:
            yield chunk

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class _MockClient:
    base_url = "http://decoder:8000/v1"

    def __init__(self, status_code, body=b""):
        self._status = status_code
        self._body = body

    def stream(self, method, endpoint, json=None, headers=None):
        return _MockStreamResponse(self._status, self._body)


def _drain(async_gen):
    """Drive an async generator to completion, returning (chunks, exc)."""

    async def _run():
        chunks = []
        try:
            async for chunk in async_gen:
                chunks.append(chunk)
            return chunks, None
        except Exception as exc:  # noqa: BLE001
            return chunks, exc

    return asyncio.run(_run())


def test_stream_retry_200_yields_chunks():
    client = _MockClient(200)
    chunks, exc = _drain(
        lb.stream_service_response_with_retry(client, "/x", {}, "rid", max_retries=3, base_delay=0.0)
    )
    assert exc is None
    assert len(chunks) == 2
    assert chunks[-1] == b"data: [DONE]\n\n"


def test_stream_retry_4xx_raises_immediately_without_retry():
    client = _MockClient(400, VLLM_400_BODY)
    chunks, exc = _drain(
        lb.stream_service_response_with_retry(client, "/x", {}, "rid", max_retries=3, base_delay=0.0)
    )
    assert isinstance(exc, lb.DecodeUpstreamError)
    assert exc.status_code == 400
    assert exc.body == VLLM_400_BODY
    # No body was streamed before the error.
    assert chunks == []


def test_stream_retry_5xx_retries_then_raises():
    client = _MockClient(500, b"internal")
    chunks, exc = _drain(
        lb.stream_service_response_with_retry(client, "/x", {}, "rid", max_retries=2, base_delay=0.0)
    )
    assert isinstance(exc, lb.DecodeUpstreamError)
    assert exc.status_code == 500


# ---------------------------------------------------------------------------
# handle_completions_impl: end-to-end error forwarding
# ---------------------------------------------------------------------------


class _FakeSched:
    def get_snapshot(self):
        return {"prefill_instances": [], "decode_instances": []}


class _FakeRuntime:
    def __init__(self):
        self.scheduler = _FakeSched()

    async def schedule(self, method, *args, **kwargs):
        if method in ("begin_request", "reserve_prefill_kv"):
            return {"key": "p", "host": "h", "port": 1}
        if method == "pick_decoder":
            return {"key": "d", "host": "h", "port": 2}
        return None

    async def get_client(self, role, key):
        return None


class _FakeRequest:
    def __init__(self, payload):
        self._body = json.dumps(payload).encode("utf-8")
        self.client = ("test-client", 0)

    async def json(self):
        return json.loads(self._body)

    async def body(self):
        return self._body


def _make_instance_info():
    return lb.InstanceInfo(
        request_id="rid",
        prefiller_key="p",
        prefiller_score=1.0,
        decoder_key="d",
        decoder_score=1.0,
        decoder_host="h",
        decoder_port=2,
    )


def _install_stubs(monkeypatch, *, mode):
    """Replace runtime/args/assign_instances/decode-stream with stubs.

    mode="error": decode raises DecodeUpstreamError(400).
    mode="ok":    decode yields normal SSE chunks.
    """
    monkeypatch.setattr(lb, "get_runtime", lambda: _FakeRuntime())
    args = MagicMock()
    args.max_retries = 1
    args.retry_delay = 0.0
    monkeypatch.setattr(lb, "get_global_args", lambda: args)

    async def _fake_assign(api, req_data, request_length, *, is_initial_request):
        return _make_instance_info()

    monkeypatch.setattr(lb, "assign_instances", _fake_assign)

    if mode == "error":
        async def _fail(client, endpoint, req_data, request_id, **kwargs):
            raise lb.DecodeUpstreamError(400, VLLM_400_BODY)
            yield  # pragma: no cover - make it an async generator

        monkeypatch.setattr(lb, "stream_service_response_with_retry", _fail)
    else:
        async def _ok(client, endpoint, req_data, request_id, **kwargs):
            yield b'data: {"choices":[{"delta":{"content":"Hello!"}}]}\n\n'
            yield b"data: [DONE]\n\n"

        monkeypatch.setattr(lb, "stream_service_response_with_retry", _ok)


def _run_handler(payload):
    req = _FakeRequest(payload)

    async def _go():
        resp = await lb.handle_completions_impl("/chat/completions", req)
        body = b""
        async for chunk in resp.body_iterator:
            body += chunk if isinstance(chunk, bytes) else chunk.encode("utf-8")
        return resp.status_code, body

    return asyncio.run(_go())


def test_streaming_decode_error_forwards_sse_error_event(monkeypatch):
    _install_stubs(monkeypatch, mode="error")
    payload = {"model": "m", "messages": [{"role": "user", "content": "Hi"}], "max_tokens": 131064, "stream": True}
    status, body = _run_handler(payload)
    assert status == 200  # StreamingResponse already committed the head
    text = body.decode("utf-8")
    assert "data: " in text
    assert '"error"' in text
    assert "maximum context length is 131072" in text
    assert "data: [DONE]" in text
    assert len(body) > 0  # no longer an empty 200


def test_non_streaming_decode_error_forwards_json_error_body(monkeypatch):
    _install_stubs(monkeypatch, mode="error")
    payload = {"model": "m", "messages": [{"role": "user", "content": "Hi"}], "max_tokens": 131064, "stream": False}
    status, body = _run_handler(payload)
    text = body.decode("utf-8")
    assert '"error"' in text
    assert "maximum context length is 131072" in text
    assert len(body) > 0  # no longer an empty 200


def test_streaming_success_unchanged(monkeypatch):
    _install_stubs(monkeypatch, mode="ok")
    payload = {"model": "m", "messages": [{"role": "user", "content": "Hi"}], "max_tokens": 16, "stream": True}
    status, body = _run_handler(payload)
    assert status == 200
    assert b'"content"' in body
    assert b"data: [DONE]" in body


def test_non_streaming_success_unchanged(monkeypatch):
    _install_stubs(monkeypatch, mode="ok")
    payload = {"model": "m", "messages": [{"role": "user", "content": "Hi"}], "max_tokens": 16, "stream": False}
    status, body = _run_handler(payload)
    assert status == 200
    assert b'"content"' in body
