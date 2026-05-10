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
import logging
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import httpx

REPO_ROOT = Path(__file__).resolve().parents[2]
ERROR_BODY = b'{"error":{"message":"too long","type":"invalid_request_error"}}'
RECOMPUTE_BODY = (
    b'{"choices":[{"message":{"content":"partial"},"stop_reason":"recomputed"}],"usage":{"completion_tokens":1}}'
)


class FakeRequest:
    def __init__(self, payload):
        self.payload = payload

    async def json(self):
        return dict(self.payload)

    async def body(self):
        return b'{"messages":[{"role":"user","content":"hello"}]}'


class FakeBackendResponse:
    def __init__(self, status_code=400, body=ERROR_BODY):
        self.status_code = status_code
        self.body = body
        self.headers = {"content-type": "application/json"}
        self.request = httpx.Request("POST", "http://backend/v1/chat/completions")
        self.response = httpx.Response(status_code, request=self.request, content=body)

    async def aread(self):
        return self.body

    async def aiter_bytes(self):
        yield self.body

    def raise_for_status(self):
        self.response.raise_for_status()


class FakeStreamContext:
    def __init__(self, client):
        self.client = client

    async def __aenter__(self):
        self.client.enter_count += 1
        return self.client.response

    async def __aexit__(self, exc_type, exc, traceback):
        self.client.exit_count += 1


class FakeBackendClient:
    def __init__(self):
        self.response = FakeBackendResponse()
        self.stream_count = 0
        self.enter_count = 0
        self.exit_count = 0

    def stream(self, *args, **kwargs):
        self.stream_count += 1
        return FakeStreamContext(self)


async def _drain_streaming_response(response):
    async for _ in response.body_iterator:
        pass


def _load_example(relative_path: str):
    module_name = "test_" + relative_path.replace("/", "_").replace(".", "_")

    vllm_mod = types.ModuleType("vllm")
    logger_mod = types.ModuleType("vllm.logger")
    logger_mod.init_logger = logging.getLogger
    vllm_mod.logger = logger_mod

    pil_mod = types.ModuleType("PIL")
    image_mod = types.ModuleType("PIL.Image")
    pil_mod.Image = image_mod

    spec = importlib.util.spec_from_file_location(module_name, REPO_ROOT / relative_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    with patch.dict(
        sys.modules,
        {
            "vllm": vllm_mod,
            "vllm.logger": logger_mod,
            "PIL": pil_mod,
            "PIL.Image": image_mod,
        },
    ):
        spec.loader.exec_module(module)
    return module


def test_disaggregated_proxy_returns_decoder_http_error_before_streaming():
    mod = _load_example("examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py")
    client = FakeBackendClient()

    class ProxyState:
        request_num = 0

        def __init__(self):
            self.aborted = []
            self.released_kv = []
            self.released_decoder = []

        def abort_prefiller_request(self, prefiller_idx, request_id):
            self.aborted.append((prefiller_idx, request_id))

        def release_prefiller_kv(self, prefiller_idx, score):
            self.released_kv.append((prefiller_idx, score))

        def release_decoder(self, decoder_idx, score):
            self.released_decoder.append((decoder_idx, score))

    state = ProxyState()
    mod.proxy_state = state
    mod.global_args = SimpleNamespace(max_retries=1, retry_delay=0)

    async def select_instance(api, req_data, request_length):
        return SimpleNamespace(
            request_id="req-1",
            prefiller_idx=0,
            prefiller_score=11,
            decoder_idx=0,
            decoder_score=22,
            decoder=SimpleNamespace(client=client, url="http://decoder/v1"),
        )

    mod._handle_select_instance = select_instance

    response = asyncio.run(
        mod._handle_completions(
            "/chat/completions",
            FakeRequest({"messages": [{"role": "user", "content": "hello"}], "max_tokens": 4000}),
        )
    )

    assert response.status_code == 400
    assert response.body == ERROR_BODY
    assert state.request_num == 0
    assert state.aborted == [(0, "req-1")]
    assert state.released_kv == [(0, 11)]
    assert state.released_decoder == [(0, 22)]
    assert client.enter_count == 1
    assert client.exit_count == 1


def test_disaggregated_proxy_does_not_double_release_decoder_when_recompute_selection_fails():
    mod = _load_example("examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py")
    client = FakeBackendClient()
    client.response = FakeBackendResponse(200, RECOMPUTE_BODY)

    class ProxyState:
        request_num = 0

        def __init__(self):
            self.aborted = []
            self.released_kv = []
            self.released_decoder = []

        def abort_prefiller_request(self, prefiller_idx, request_id):
            self.aborted.append((prefiller_idx, request_id))

        def release_prefiller_kv(self, prefiller_idx, score):
            self.released_kv.append((prefiller_idx, score))

        def release_decoder(self, decoder_idx, score):
            self.released_decoder.append((decoder_idx, score))

    state = ProxyState()
    mod.proxy_state = state
    mod.global_args = SimpleNamespace(max_retries=1, retry_delay=0)
    select_calls = 0

    async def select_instance(api, req_data, request_length):
        nonlocal select_calls
        select_calls += 1
        if select_calls == 1:
            return SimpleNamespace(
                request_id="req-1",
                prefiller_idx=0,
                prefiller_score=11,
                decoder_idx=0,
                decoder_score=22,
                decoder=SimpleNamespace(client=client, url="http://decoder/v1"),
            )
        raise RuntimeError("no replacement decoder")

    mod._handle_select_instance = select_instance

    response = asyncio.run(
        mod._handle_completions(
            "/chat/completions",
            FakeRequest({"messages": [{"role": "user", "content": "hello"}], "max_tokens": 4000}),
        )
    )
    asyncio.run(_drain_streaming_response(response))

    assert select_calls == 2
    assert state.request_num == 0
    assert state.aborted == [(0, "req-1")]
    assert state.released_kv == [(0, 11)]
    assert state.released_decoder == [(0, 22)]


def test_layerwise_proxy_returns_decoder_http_error_before_streaming():
    mod = _load_example("examples/disaggregated_prefill_v1/load_balance_proxy_layerwise_server_example.py")
    client = FakeBackendClient()

    class ProxyState:
        def __init__(self):
            self.req_data_dict = {}
            self.decoders = [SimpleNamespace(client=client, url="http://decoder/v1")]
            self.released_decoder = []

        async def next_req_id(self):
            return "req-1"

        def calculate_decode_scores(self, request_length):
            return 22

        def select_decoder(self, decoder_score):
            return 0

        def release_decoder(self, decoder_idx, score):
            self.released_decoder.append((decoder_idx, score))

    state = ProxyState()
    mod.proxy_state = state
    mod.global_args = SimpleNamespace(host="127.0.0.1", port=9000, max_retries=1, retry_delay=0)

    response = asyncio.run(
        mod._handle_completions(
            "/chat/completions",
            FakeRequest({"messages": [{"role": "user", "content": "hello"}], "max_tokens": 4000}),
        )
    )

    assert response.status_code == 400
    assert response.body == ERROR_BODY
    assert state.released_decoder == [(0, 22)]
    assert client.enter_count == 1
    assert client.exit_count == 1


def test_epd_proxy_returns_pd_http_error_before_streaming():
    mod = _load_example("examples/epd_disaggregated/epd_load_balance_proxy_layerwise_server_example.py")
    client = FakeBackendClient()

    class ProxyState:
        def __init__(self):
            self.encoders = []
            self.pds = [SimpleNamespace(client=client, url="http://pd/v1")]
            self.aborted_pd = []
            self.released_pd = []

        async def next_req_id(self):
            return "req-1"

        def select_pd(self, token_score):
            return 0

        def release_pd(self, pd_idx, token_score):
            self.released_pd.append((pd_idx, token_score))

        def abort_pd_request(self, pd_idx, request_id):
            self.aborted_pd.append((pd_idx, request_id))

    state = ProxyState()
    mod.proxy_state = state
    mod.global_args = SimpleNamespace(max_retries=1, retry_delay=0)

    response = asyncio.run(
        mod._handle_completions(
            "/chat/completions",
            FakeRequest({"messages": [{"role": "user", "content": "hello"}], "max_tokens": 4000}),
        )
    )

    assert response.status_code == 400
    assert response.body == ERROR_BODY
    assert state.aborted_pd == [(0, "req-1")]
    assert state.released_pd == [(0, 0)]
    assert client.enter_count == 1
    assert client.exit_count == 1
