#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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

"""CPU-only unit tests for the EPD proxy robustness changes.

aiohttp sessions/responses are faked, so no real server process is started and
no network access is needed.
"""

import asyncio
import json
import re

import aiohttp
import pytest

from examples.disaggregated_encoder import disagg_epd_proxy as proxy

ENCODE_URL = "http://encode-a:19534"
DECODE_URL = "http://decode:19535"
PREFILL_URL = "http://prefill:19536"


class FakeContent:
    def __init__(self, chunks):
        self._chunks = list(chunks)

    async def iter_chunked(self, chunk_size):
        for chunk in self._chunks:
            if isinstance(chunk, BaseException):
                raise chunk
            yield chunk


class FakeResponse:
    """Async context manager mimicking aiohttp.ClientResponse."""

    def __init__(self, status=200, body=b"{}", *, enter_exc=None, read_exc=None, chunks=()):
        self.status = status
        self._body = body
        self._enter_exc = enter_exc
        self._read_exc = read_exc
        self.released = False
        self.content = FakeContent(chunks)

    async def __aenter__(self):
        if self._enter_exc is not None:
            raise self._enter_exc
        return self

    async def __aexit__(self, exc_type, exc, tb):
        self.released = True
        return False

    async def read(self):
        if self._read_exc is not None:
            raise self._read_exc
        return self._body


class FakeSession:
    """Records outbound calls and returns scripted FakeResponse objects."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def post(self, url, json=None, headers=None, timeout=None):
        self.calls.append({"url": url, "json": json, "headers": headers, "timeout": timeout})
        if not self.responses:
            raise AssertionError(f"unexpected POST to {url}")
        return self.responses.pop(0)


def _image_request(num_images=1):
    content = [{"type": "image_url", "image_url": {"url": f"http://img/{i}.png"}} for i in range(num_images)]
    return {"model": "m", "messages": [{"role": "user", "content": content}]}


def _set_timeouts(monkeypatch, **stages):
    monkeypatch.setattr(
        proxy.app.state,
        "stage_timeouts",
        {"encode": 0.0, "prefill": 0.0, "decode": 0.0, **stages},
        raising=False,
    )


@pytest.fixture(autouse=True)
def _isolated_app_state(monkeypatch):
    for name in ("encode_session", "prefill_session", "decode_session"):
        monkeypatch.setattr(proxy, name, None, raising=False)
    monkeypatch.setattr(proxy.app.state, "e_urls", [ENCODE_URL], raising=False)
    monkeypatch.setattr(proxy.app.state, "p_urls", [], raising=False)
    monkeypatch.setattr(proxy.app.state, "d_urls", [DECODE_URL], raising=False)
    monkeypatch.setattr(proxy.app.state, "encoder_dispatch_mode", proxy.EncoderDispatchMode.SINGLE, raising=False)
    monkeypatch.setattr(proxy.app.state, "encode_failures", {}, raising=False)
    _set_timeouts(monkeypatch)


# --------------------------------------------------------------------------- #
# Multimodal request helpers
# --------------------------------------------------------------------------- #
def test_extract_mm_items_cases():
    assert proxy.extract_mm_items({"messages": []}) == []
    assert proxy.extract_mm_items({"messages": [{"role": "user", "content": "plain text"}]}) == []

    single = _image_request()
    items = proxy.extract_mm_items(single)
    assert len(items) == 1
    assert items[0]["type"] == "image_url"

    multiple = _image_request(num_images=3)
    assert len(proxy.extract_mm_items(multiple)) == 3

    audio = {"messages": [{"role": "user", "content": [{"type": "audio_url", "audio_url": {"url": "a.wav"}}]}]}
    audio_items = proxy.extract_mm_items(audio)
    assert len(audio_items) == 1 and audio_items[0]["type"] == "audio_url"


def test_has_mm_input_cases():
    text_only = {"messages": [{"role": "user", "content": "hello"}]}
    assert proxy.has_mm_input(text_only) is False
    assert proxy.has_mm_input({"messages": []}) is False
    assert proxy.has_mm_input(_image_request()) is True


# --------------------------------------------------------------------------- #
# Encoder routing
# --------------------------------------------------------------------------- #
def test_encoder_single_routes_one_request_with_parent_id(monkeypatch):
    session = FakeSession([FakeResponse(200, b"{}")])
    monkeypatch.setattr(proxy, "encode_session", session, raising=False)

    asyncio.run(proxy._encode_single_request(_image_request(), ENCODE_URL, "parent-1"))

    assert len(session.calls) == 1
    call = session.calls[0]
    assert call["url"] == f"{ENCODE_URL}/v1/chat/completions"
    assert call["headers"] == {"x-request-id": "parent-1"}
    assert call["json"]["max_tokens"] == 1
    assert call["json"]["stream"] is False
    assert call["timeout"] is None


def test_encoder_fanout_one_request_per_item_with_child_ids(monkeypatch):
    session = FakeSession([FakeResponse(200, b"{}"), FakeResponse(200, b"{}")])
    monkeypatch.setattr(proxy, "encode_session", session, raising=False)

    asyncio.run(proxy._encode_fanout(_image_request(num_images=2), ["http://e1", "http://e2"], "parent"))

    assert len(session.calls) == 2
    child_ids = [call["headers"]["x-request-id"] for call in session.calls]
    assert all(re.fullmatch(r"parent:\d+:[0-9a-f]{6}", child_id) for child_id in child_ids)
    assert [call["url"] for call in session.calls] == [
        "http://e1/v1/chat/completions",
        "http://e2/v1/chat/completions",
    ]


# --------------------------------------------------------------------------- #
# Encoder failure classification
# --------------------------------------------------------------------------- #
def _run_single_encode(monkeypatch, response):
    session = FakeSession([response])
    monkeypatch.setattr(proxy, "encode_session", session, raising=False)
    with pytest.raises(proxy.HTTPException) as exc_info:
        asyncio.run(proxy._encode_single_request(_image_request(), ENCODE_URL, "req-1"))
    return exc_info.value, session


def test_encode_connection_error_maps_to_502(monkeypatch):
    error, session = _run_single_encode(monkeypatch, FakeResponse(enter_exc=aiohttp.ClientConnectionError("boom")))
    assert error.status_code == 502
    assert error.detail["stage"] == "encode"
    assert error.detail["url"] == f"{ENCODE_URL}/v1/chat/completions"
    assert proxy.app.state.encode_failures[ENCODE_URL] > 0
    assert len(session.calls) == 1


def test_encode_timeout_maps_to_504(monkeypatch):
    error, _ = _run_single_encode(monkeypatch, FakeResponse(enter_exc=asyncio.TimeoutError()))
    assert error.status_code == 504
    assert error.detail["stage"] == "encode"
    assert proxy.app.state.encode_failures[ENCODE_URL] > 0


def test_encode_downstream_4xx_propagates(monkeypatch):
    error, _ = _run_single_encode(monkeypatch, FakeResponse(400, b"bad request"))
    assert error.status_code == 400
    assert error.detail["stage"] == "encode"
    assert error.detail["status"] == 400


def test_encode_downstream_5xx_maps_to_502(monkeypatch):
    error, _ = _run_single_encode(monkeypatch, FakeResponse(500, b"server exploded"))
    assert error.status_code == 502
    assert error.detail["stage"] == "encode"
    assert error.detail["status"] == 500


# --------------------------------------------------------------------------- #
# Retry wrappers (counts and short-circuits unchanged)
# --------------------------------------------------------------------------- #
def test_non_stream_retry_stops_on_4xx():
    calls = []

    async def forward():
        calls.append(1)
        raise proxy.HTTPException(status_code=400, detail="bad request")

    with pytest.raises(proxy.HTTPException) as exc_info:
        asyncio.run(proxy.non_stream_retry_wrap(forward))
    assert exc_info.value.status_code == 400
    assert len(calls) == 1


@pytest.mark.parametrize(
    "exc_factory",
    [lambda: proxy.HTTPException(status_code=502, detail="bad"), asyncio.TimeoutError],
)
def test_non_stream_retry_retries_5xx_and_timeout(exc_factory):
    calls = []

    async def forward():
        calls.append(1)
        raise exc_factory()

    with pytest.raises(RuntimeError):
        asyncio.run(proxy.non_stream_retry_wrap(forward))
    assert len(calls) == 3


def test_stream_retry_stops_after_first_chunk():
    attempts = []

    async def forward():
        attempts.append(1)
        yield "chunk-1"
        raise proxy.HTTPException(status_code=502, detail="stream broke")

    async def consume():
        chunks = []
        async for chunk in proxy.stream_retry_wrap(forward):
            chunks.append(chunk)
        return chunks

    with pytest.raises(proxy.HTTPException):
        asyncio.run(consume())
    assert attempts == [1]


def test_stream_retry_retries_before_first_chunk():
    attempts = []

    async def forward():
        attempts.append(1)
        if len(attempts) == 1:
            raise proxy.HTTPException(status_code=502, detail="retry me")
        yield "chunk-1"

    async def consume():
        return [chunk async for chunk in proxy.stream_retry_wrap(forward)]

    assert asyncio.run(consume()) == ["chunk-1"]
    assert len(attempts) == 2


# --------------------------------------------------------------------------- #
# Stage timeouts and response release
# --------------------------------------------------------------------------- #
def test_stage_timeout_disabled_passes_none(monkeypatch):
    session = FakeSession([FakeResponse(200, b"{}")])
    status, body = asyncio.run(proxy._post_stage(session, f"{ENCODE_URL}/v1/chat/completions", {}, {}, "encode", "id"))
    assert (status, body) == (200, b"{}")
    assert session.calls[0]["timeout"] is None


def test_stage_timeout_enabled_passes_client_timeout(monkeypatch):
    _set_timeouts(monkeypatch, encode=12.5)
    session = FakeSession([FakeResponse(200, b"{}")])
    asyncio.run(proxy._post_stage(session, f"{ENCODE_URL}/v1/chat/completions", {}, {}, "encode", "id"))
    timeout = session.calls[0]["timeout"]
    assert isinstance(timeout, aiohttp.ClientTimeout)
    assert timeout.total == 12.5


def test_post_stage_releases_response(monkeypatch):
    response = FakeResponse(200, b"{}")
    session = FakeSession([response])
    asyncio.run(proxy._post_stage(session, f"{ENCODE_URL}/v1/chat/completions", {}, {}, "encode", "id"))
    assert response.released is True


# --------------------------------------------------------------------------- #
# Success paths (prefill kv transfer, decode JSON)
# --------------------------------------------------------------------------- #
def test_maybe_prefill_forwards_kv_transfer_params(monkeypatch):
    body = json.dumps({"kv_transfer_params": {"do_remote_prefill": True}}).encode()
    session = FakeSession([FakeResponse(200, body)])
    monkeypatch.setattr(proxy, "prefill_session", session, raising=False)

    req_data = {"model": "m", "messages": [{"role": "user", "content": "hi"}], "stream": True}
    result = asyncio.run(proxy.maybe_prefill(req_data, PREFILL_URL, "req-1"))

    assert result["kv_transfer_params"] == {"do_remote_prefill": True}
    call = session.calls[0]
    assert call["url"] == f"{PREFILL_URL}/v1/chat/completions"
    assert call["json"]["max_tokens"] == 1
    assert call["json"]["stream"] is False
    assert call["json"]["kv_transfer_params"]["do_remote_decode"] is True


def test_maybe_prefill_downstream_5xx_maps_to_502(monkeypatch):
    session = FakeSession([FakeResponse(500, b"prefill exploded")])
    monkeypatch.setattr(proxy, "prefill_session", session, raising=False)
    req_data = {"model": "m", "messages": [{"role": "user", "content": "hi"}], "stream": False}

    with pytest.raises(proxy.HTTPException) as exc_info:
        asyncio.run(proxy.maybe_prefill(req_data, PREFILL_URL, "req-1"))
    assert exc_info.value.status_code == 502
    assert exc_info.value.detail["stage"] == "prefill"


def test_forward_non_stream_returns_decode_json(monkeypatch):
    decode_body = b'{"choices": [{"message": {"content": "hi"}}]}'
    session = FakeSession([FakeResponse(200, decode_body)])
    monkeypatch.setattr(proxy, "decode_session", session, raising=False)
    req_data = {"model": "m", "messages": [{"role": "user", "content": "hello"}], "stream": False}

    result = asyncio.run(proxy.forward_non_stream(req_data, "req-1", None, DECODE_URL))
    assert result == {"choices": [{"message": {"content": "hi"}}]}
    assert session.calls[0]["timeout"] is None


def test_forward_non_stream_decode_4xx_propagates_without_retry(monkeypatch):
    session = FakeSession([FakeResponse(400, b"bad decode request")])
    monkeypatch.setattr(proxy, "decode_session", session, raising=False)
    req_data = {"model": "m", "messages": [{"role": "user", "content": "hello"}], "stream": False}

    with pytest.raises(proxy.HTTPException) as exc_info:
        asyncio.run(proxy.forward_non_stream(req_data, "req-1", None, DECODE_URL))
    assert exc_info.value.status_code == 400
    assert exc_info.value.detail["stage"] == "decode"
    assert len(session.calls) == 1


# --------------------------------------------------------------------------- #
# CLI parser / app state
# --------------------------------------------------------------------------- #
def test_build_arg_parser_defaults():
    parser = proxy.build_arg_parser()
    args = parser.parse_args(
        [
            "--encode-servers-urls",
            ENCODE_URL,
            "--prefill-servers-urls",
            "disable",
            "--decode-servers-urls",
            DECODE_URL,
        ]
    )
    assert args.encode_timeout_seconds == 0.0
    assert args.prefill_timeout_seconds == 0.0
    assert args.decode_timeout_seconds == 0.0
    assert args.encoder_dispatch_mode == "single"
    assert args.host == "0.0.0.0"
    assert args.port == 8000

    with pytest.raises(SystemExit):
        parser.parse_args([])


def test_configure_app_state_sets_stage_timeouts_and_failures():
    parser = proxy.build_arg_parser()
    args = parser.parse_args(
        [
            "--encode-servers-urls",
            f"{ENCODE_URL}, http://encode-b:19534",
            "--prefill-servers-urls",
            "none",
            "--decode-servers-urls",
            DECODE_URL,
            "--encode-timeout-seconds",
            "3.5",
            "--decode-timeout-seconds",
            "7",
            "--encoder-dispatch-mode",
            "fanout",
        ]
    )
    proxy.configure_app_state(args)

    assert proxy.app.state.e_urls == [ENCODE_URL, "http://encode-b:19534"]
    assert proxy.app.state.p_urls == []
    assert proxy.app.state.d_urls == [DECODE_URL]
    assert proxy.app.state.stage_timeouts == {"encode": 3.5, "prefill": 0.0, "decode": 7.0}
    assert proxy.app.state.encode_failures == {}
    assert proxy.app.state.encoder_dispatch_mode is proxy.EncoderDispatchMode.FANOUT

    prefill_args = parser.parse_args(
        [
            "--encode-servers-urls",
            ENCODE_URL,
            "--prefill-servers-urls",
            PREFILL_URL,
            "--decode-servers-urls",
            DECODE_URL,
        ]
    )
    proxy.configure_app_state(prefill_args)
    assert proxy.app.state.p_urls == [PREFILL_URL]
