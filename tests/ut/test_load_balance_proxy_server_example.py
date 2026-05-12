#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#

import asyncio
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from fastapi import Request

REPO_ROOT = Path(__file__).resolve().parents[2]
PROXY_PATH = REPO_ROOT / "examples" / "disaggregated_prefill_v1" / "load_balance_proxy_server_example.py"
MODULE_NAME = "vllm_ascend_examples_load_balance_proxy_server_example"


def _load_proxy_module():
    sys.modules.pop(MODULE_NAME, None)
    spec = importlib.util.spec_from_file_location(MODULE_NAME, PROXY_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _build_request(payload: dict) -> Request:
    body = json.dumps(payload).encode("utf-8")
    sent = False

    async def receive():
        nonlocal sent
        if not sent:
            sent = True
            return {"type": "http.request", "body": body, "more_body": False}
        return {"type": "http.disconnect"}

    scope = {
        "type": "http",
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/v1/completions",
        "raw_path": b"/v1/completions",
        "query_string": b"",
        "headers": [(b"content-type", b"application/json")],
        "client": ("127.0.0.1", 12345),
        "server": ("testserver", 80),
    }
    return Request(scope, receive)


def test_handle_completions_returns_upstream_400_response(monkeypatch):
    module = _load_proxy_module()
    module.proxy_state = SimpleNamespace(request_num=0)

    req = httpx.Request("POST", "http://prefiller.example/v1/completions")
    upstream_response = httpx.Response(
        400,
        request=req,
        json={"error": {"message": "context length exceeded"}},
        headers={"content-type": "application/json"},
    )
    upstream_error = httpx.HTTPStatusError("Bad Request", request=req, response=upstream_response)

    async def raise_upstream_error(*args, **kwargs):
        raise upstream_error

    monkeypatch.setattr(module, "_handle_select_instance", raise_upstream_error)

    response = asyncio.run(module._handle_completions("/completions", _build_request({"prompt": "hello"})))

    assert response.status_code == 400
    assert json.loads(response.body) == {"error": {"message": "context length exceeded"}}
    assert module.proxy_state.request_num == 0


def test_stream_service_response_reads_http_error_body_before_response_closes():
    module = _load_proxy_module()

    class FakeStreamingResponse:
        def __init__(self):
            self.closed = False
            self.read_closed_states = []
            self.request = httpx.Request("POST", "http://decoder.example/v1/completions")

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            self.closed = True

        def raise_for_status(self):
            raise httpx.HTTPStatusError("Bad Gateway", request=self.request, response=self)

        async def aread(self):
            self.read_closed_states.append(self.closed)
            if self.closed:
                raise RuntimeError("response already closed")
            return b'{"error":"bad gateway"}'

    class FakeClient:
        def __init__(self, response):
            self.response = response

        def stream(self, *args, **kwargs):
            return self.response

    response = FakeStreamingResponse()
    client = FakeClient(response)

    async def consume_stream():
        async for _ in module.stream_service_response_with_retry(
            client,
            "/completions",
            {"prompt": "hello"},
            "request-id",
            max_retries=1,
        ):
            pytest.fail("expected an HTTPStatusError before any chunks are streamed")

    with pytest.raises(httpx.HTTPStatusError):
        asyncio.run(consume_stream())

    assert response.read_closed_states == [False]
