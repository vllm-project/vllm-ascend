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
import time

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

    def raise_for_status(self):
        if self.status >= 400:
            raise aiohttp.ClientError(f"fake HTTP {self.status}")


class FakeSession:
    """Records outbound calls and returns scripted FakeResponse objects."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []
        self.closed = False

    def post(self, url, json=None, headers=None, **kwargs):
        self.calls.append({"method": "post", "url": url, "json": json, "headers": headers, "kwargs": kwargs})
        if not self.responses:
            raise AssertionError(f"unexpected POST to {url}")
        return self.responses.pop(0)

    def get(self, url, **kwargs):
        self.calls.append({"method": "get", "url": url, "kwargs": kwargs})
        if not self.responses:
            raise AssertionError(f"unexpected GET to {url}")
        return self.responses.pop(0)

    async def close(self):
        self.closed = True


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
    monkeypatch.setattr(proxy.app.state, "encoder_failure_cooldown_seconds", 30.0, raising=False)
    monkeypatch.setattr(proxy.app.state, "probe_encoders_on_startup", False, raising=False)
    monkeypatch.setattr(proxy.app.state, "shutdown_drain_seconds", 0.0, raising=False)
    monkeypatch.setattr(proxy.app.state, "stats", proxy.ProxyStats(), raising=False)
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
    assert "timeout" not in call["kwargs"]


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
    assert all("timeout" not in call["kwargs"] for call in session.calls)


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


def test_encode_downstream_redirect_maps_to_502(monkeypatch):
    error, _ = _run_single_encode(monkeypatch, FakeResponse(307, b"redirect"))
    assert error.status_code == 502
    assert error.detail["stage"] == "encode"
    assert error.detail["status"] == 307


def test_encoder_single_accepts_any_2xx(monkeypatch):
    session = FakeSession([FakeResponse(204, b"")])
    monkeypatch.setattr(proxy, "encode_session", session, raising=False)

    asyncio.run(proxy._encode_single_request(_image_request(), ENCODE_URL, "req-1"))

    assert len(session.calls) == 1
    assert proxy.app.state.encode_failures == {}


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
def test_stage_timeout_disabled_omits_timeout_kwarg(monkeypatch):
    assert proxy._stage_timeout("encode") is None
    session = FakeSession([FakeResponse(200, b"{}")])
    status, body = asyncio.run(proxy._post_stage(session, f"{ENCODE_URL}/v1/chat/completions", {}, {}, "encode", "id"))
    assert (status, body) == (200, b"{}")
    # Omitting the keyword keeps the session default total timeout alive; passing
    # timeout=None would disable it, so it must not be present at all.
    assert "timeout" not in session.calls[0]["kwargs"]
    assert proxy._stage_timeout_kwargs("encode") == {}


def test_stage_timeout_enabled_passes_client_timeout(monkeypatch):
    _set_timeouts(monkeypatch, encode=12.5)
    session = FakeSession([FakeResponse(200, b"{}")])
    asyncio.run(proxy._post_stage(session, f"{ENCODE_URL}/v1/chat/completions", {}, {}, "encode", "id"))
    timeout = session.calls[0]["kwargs"]["timeout"]
    assert isinstance(timeout, aiohttp.ClientTimeout)
    assert timeout.total == 12.5


def test_post_stage_releases_response(monkeypatch):
    response = FakeResponse(200, b"{}")
    session = FakeSession([response])
    asyncio.run(proxy._post_stage(session, f"{ENCODE_URL}/v1/chat/completions", {}, {}, "encode", "id"))
    assert response.released is True


def test_stream_decode_omits_timeout_kwarg_by_default(monkeypatch):
    session = FakeSession([FakeResponse(200, b"", chunks=[b"chunk-1"])])
    monkeypatch.setattr(proxy, "decode_session", session, raising=False)
    req_data = {"model": "m", "messages": [{"role": "user", "content": "hello"}], "stream": True}

    async def collect():
        return [chunk async for chunk in proxy.forward_stream(req_data, "req-1", None, DECODE_URL)]

    assert asyncio.run(collect()) == ["chunk-1"]
    assert "timeout" not in session.calls[0]["kwargs"]


def test_stream_decode_passes_timeout_when_configured(monkeypatch):
    _set_timeouts(monkeypatch, decode=5.0)
    session = FakeSession([FakeResponse(200, b"", chunks=[b"chunk-1"])])
    monkeypatch.setattr(proxy, "decode_session", session, raising=False)
    req_data = {"model": "m", "messages": [{"role": "user", "content": "hello"}], "stream": True}

    async def collect():
        return [chunk async for chunk in proxy.forward_stream(req_data, "req-1", None, DECODE_URL)]

    assert asyncio.run(collect()) == ["chunk-1"]
    timeout = session.calls[0]["kwargs"]["timeout"]
    assert isinstance(timeout, aiohttp.ClientTimeout)
    assert timeout.total == 5.0


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
    assert "timeout" not in session.calls[0]["kwargs"]


def test_forward_non_stream_accepts_201(monkeypatch):
    session = FakeSession([FakeResponse(201, b'{"choices": []}')])
    monkeypatch.setattr(proxy, "decode_session", session, raising=False)
    req_data = {"model": "m", "messages": [{"role": "user", "content": "hello"}], "stream": False}

    result = asyncio.run(proxy.forward_non_stream(req_data, "req-1", None, DECODE_URL))
    assert result == {"choices": []}


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


# --------------------------------------------------------------------------- #
# Health-aware encoder routing (N5)
# --------------------------------------------------------------------------- #
class _FakeAiohttpModule:
    """Replacement for the aiohttp module used by on_startup."""

    ClientTimeout = aiohttp.ClientTimeout

    def __init__(self, sessions):
        self.sessions = list(sessions)
        self.created = []

    def TCPConnector(self, **kwargs):
        return None

    def ClientSession(self, **kwargs):
        session = self.sessions.pop(0)
        self.created.append(session)
        return session


def _install_fake_aiohttp(monkeypatch, sessions):
    fake = _FakeAiohttpModule(sessions)
    monkeypatch.setattr(proxy, "aiohttp", fake)
    return fake


def test_record_encode_failure_classification(monkeypatch):
    # 4xx is a request-side problem and must not mark the encoder unhealthy.
    _run_single_encode(monkeypatch, FakeResponse(400, b"bad request"))
    assert proxy.app.state.encode_failures == {}

    # Downstream 5xx is a health-relevant encoder failure.
    _run_single_encode(monkeypatch, FakeResponse(500, b"server exploded"))
    assert ENCODE_URL in proxy.app.state.encode_failures

    proxy.app.state.encode_failures.clear()
    _run_single_encode(monkeypatch, FakeResponse(enter_exc=aiohttp.ClientConnectionError("boom")))
    assert ENCODE_URL in proxy.app.state.encode_failures

    # Stage timeouts count as encoder failures.
    proxy.app.state.encode_failures.clear()
    _run_single_encode(monkeypatch, FakeResponse(enter_exc=asyncio.TimeoutError()))
    assert ENCODE_URL in proxy.app.state.encode_failures

    # A non-connection client error (payload corruption) is not an unhealthy encoder.
    proxy.app.state.encode_failures.clear()
    payload_error = aiohttp.ClientPayloadError("bad payload")
    _run_single_encode(monkeypatch, FakeResponse(enter_exc=payload_error))
    assert proxy.app.state.encode_failures == {}

    # Generic exceptions are not encoder-health signals either.
    session = FakeSession([FakeResponse(enter_exc=RuntimeError("boom"))])
    monkeypatch.setattr(proxy, "encode_session", session, raising=False)
    with pytest.raises(RuntimeError, match="boom"):
        asyncio.run(proxy._encode_single_request(_image_request(), ENCODE_URL, "req-1"))
    assert proxy.app.state.encode_failures == {}

    # A successful encode clears a previous failure timestamp.
    proxy.app.state.encode_failures[ENCODE_URL] = time.time()
    session = FakeSession([FakeResponse(200, b"{}")])
    monkeypatch.setattr(proxy, "encode_session", session, raising=False)
    asyncio.run(proxy._encode_single_request(_image_request(), ENCODE_URL, "req-1"))
    assert ENCODE_URL not in proxy.app.state.encode_failures


def _health_state(monkeypatch, e_urls, failures, cooldown=30.0):
    monkeypatch.setattr(proxy.app.state, "e_urls", list(e_urls), raising=False)
    monkeypatch.setattr(proxy.app.state, "encode_failures", dict(failures), raising=False)
    monkeypatch.setattr(proxy.app.state, "encoder_failure_cooldown_seconds", cooldown, raising=False)


def test_failure_cooldown_returns_only_healthy_encoder_subset(monkeypatch):
    now = time.time()
    # A recently failed encoder is excluded while at least one healthy one exists.
    _health_state(monkeypatch, ["e1", "e2", "e3"], {"e2": now})
    assert proxy._encoder_urls_by_health() == ["e1", "e3"]

    # A stale failure outside the cooldown window is healthy again.
    _health_state(monkeypatch, ["e1", "e2"], {"e2": now - 120.0}, cooldown=30.0)
    assert proxy._encoder_urls_by_health() == ["e1", "e2"]

    # cooldown <= 0 disables health-aware filtering.
    _health_state(monkeypatch, ["e1", "e2"], {"e2": now}, cooldown=0.0)
    assert proxy._encoder_urls_by_health() == ["e1", "e2"]

    # All encoders failing falls back to the full list (best effort), original order.
    _health_state(monkeypatch, ["e1", "e2"], {"e1": now, "e2": now})
    assert proxy._encoder_urls_by_health() == ["e1", "e2"]

    # A single encoder is unaffected.
    _health_state(monkeypatch, ["e1"], {"e1": now})
    assert proxy._encoder_urls_by_health() == ["e1"]


def test_single_mode_picks_healthy_encoder(monkeypatch):
    _health_state(monkeypatch, ["http://e1:1", "http://e2:2"], {"http://e1:1": time.time()})
    monkeypatch.setattr(proxy.app.state, "encoder_dispatch_mode", proxy.EncoderDispatchMode.SINGLE, raising=False)
    session = FakeSession([FakeResponse(200, b"{}")])
    monkeypatch.setattr(proxy, "encode_session", session, raising=False)

    asyncio.run(proxy.fanout_encoder_primer(_image_request(), "req-1"))

    # The failing encoder is excluded from the candidate set, so the real
    # random choice can only dispatch to the healthy one.
    assert proxy._encoder_urls_by_health() == ["http://e2:2"]
    assert len(session.calls) == 1
    assert session.calls[0]["url"].startswith("http://e2:2/")


def test_fanout_mode_uses_only_healthy_encoder(monkeypatch):
    _health_state(monkeypatch, ["e1", "e2"], {"e1": time.time()})
    monkeypatch.setattr(proxy.app.state, "encoder_dispatch_mode", proxy.EncoderDispatchMode.FANOUT, raising=False)
    captured = {}

    async def fake_fanout(request, urls, req_id):
        captured["urls"] = list(urls)

    monkeypatch.setattr(proxy, "_encode_fanout", fake_fanout)
    asyncio.run(proxy.fanout_encoder_primer(_image_request(), "req-1"))
    assert captured["urls"] == ["e2"]


def test_fanout_requests_go_only_to_healthy_encoder(monkeypatch):
    _health_state(monkeypatch, ["http://e1:1", "http://e2:2"], {"http://e1:1": time.time()})
    session = FakeSession([FakeResponse(200, b"{}"), FakeResponse(200, b"{}")])
    monkeypatch.setattr(proxy, "encode_session", session, raising=False)

    asyncio.run(proxy._encode_fanout(_image_request(num_images=2), proxy._encoder_urls_by_health(), "req-1"))

    assert len(session.calls) == 2
    assert all(call["url"].startswith("http://e2:2/") for call in session.calls)


# --------------------------------------------------------------------------- #
# Startup probing
# --------------------------------------------------------------------------- #
def test_probe_encoders_on_startup_fails_fast(monkeypatch):
    fake = _install_fake_aiohttp(monkeypatch, [FakeSession([FakeResponse(500, b"down")]), FakeSession([])])
    monkeypatch.setattr(proxy.app.state, "e_urls", ["http://e1:1", "http://e2:2"], raising=False)
    monkeypatch.setattr(proxy.app.state, "probe_encoders_on_startup", True, raising=False)

    with pytest.raises(RuntimeError, match="Unreachable encoder instances") as exc_info:
        asyncio.run(proxy.on_startup())
    assert "http://e1:1" in str(exc_info.value)
    assert "http://e2:2" in str(exc_info.value)
    assert fake.created[0].calls[0]["url"] == "http://e1:1/health"
    assert all(session.closed for session in fake.created)


def test_probe_encoders_on_startup_passes_when_reachable(monkeypatch):
    fake = _install_fake_aiohttp(
        monkeypatch,
        [FakeSession([FakeResponse(200, b"{}"), FakeResponse(200, b"{}")]), FakeSession([])],
    )
    monkeypatch.setattr(proxy.app.state, "e_urls", ["http://e1:1", "http://e2:2"], raising=False)
    monkeypatch.setattr(proxy.app.state, "probe_encoders_on_startup", True, raising=False)

    asyncio.run(proxy.on_startup())
    urls = [call["url"] for call in fake.created[0].calls]
    assert urls == ["http://e1:1/health", "http://e2:2/health"]
    assert not any(session.closed for session in fake.created)


def test_probe_disabled_by_default(monkeypatch):
    fake = _install_fake_aiohttp(monkeypatch, [FakeSession([]), FakeSession([])])
    monkeypatch.setattr(proxy.app.state, "e_urls", ["http://e1:1"], raising=False)

    asyncio.run(proxy.on_startup())
    assert fake.created[0].calls == []


# --------------------------------------------------------------------------- #
# /metrics
# --------------------------------------------------------------------------- #
def test_metrics_endpoint_reports_stage_stats(monkeypatch):
    session = FakeSession([FakeResponse(200, b"{}")])
    asyncio.run(proxy._post_stage(session, f"{ENCODE_URL}/v1/chat/completions", {}, {}, "encode", "id"))
    failure = FakeSession([FakeResponse(400, b"bad")])
    asyncio.run(proxy._post_stage(failure, f"{ENCODE_URL}/v1/chat/completions", {}, {}, "encode", "id"))

    response = asyncio.run(proxy.metrics())
    text = response.body.decode()
    assert 'epd_requests_total{stage="encode",status="ok"} 1' in text
    assert 'epd_requests_total{stage="encode",status="http_4xx"} 1' in text
    for metric in (
        "epd_requests_total",
        "epd_stage_seconds_bucket",
        "epd_stage_seconds_sum",
        "epd_stage_seconds_count",
        "epd_retries_total",
        "epd_encoder_requests_total",
        "epd_mm_items_total",
        "epd_requests_inflight",
    ):
        assert metric in text
    assert "# HELP epd_requests_total" in text
    assert "# TYPE epd_requests_total counter" in text


def test_metrics_histogram_and_retries(monkeypatch):
    session = FakeSession([FakeResponse(200, b"{}")])
    asyncio.run(proxy._post_stage(session, f"{ENCODE_URL}/v1/chat/completions", {}, {}, "encode", "id"))
    text = asyncio.run(proxy.metrics()).body.decode()
    assert 'epd_stage_seconds_count{stage="encode"} 1' in text
    assert 'epd_stage_seconds_bucket{stage="encode",le="+Inf"} 1' in text

    async def always_fails():
        raise proxy.HTTPException(status_code=502, detail="bad gateway")

    with pytest.raises(RuntimeError):
        asyncio.run(proxy.non_stream_retry_wrap(always_fails, stage="encode"))
    text = asyncio.run(proxy.metrics()).body.decode()
    assert 'epd_retries_total{stage="encode"} 2' in text


def test_encoder_and_mm_item_counters(monkeypatch):
    session = FakeSession([FakeResponse(200, b"{}"), FakeResponse(200, b"{}")])
    monkeypatch.setattr(proxy, "encode_session", session, raising=False)
    asyncio.run(proxy._encode_fanout(_image_request(num_images=2), [ENCODE_URL], "req-1"))
    text = asyncio.run(proxy.metrics()).body.decode()
    assert "epd_encoder_requests_total 2" in text
    assert "epd_mm_items_total 2" in text


# --------------------------------------------------------------------------- #
# Shutdown draining
# --------------------------------------------------------------------------- #
def test_drain_waits_for_inflight(monkeypatch):
    session = FakeSession([])
    monkeypatch.setattr(proxy, "encode_session", session, raising=False)
    monkeypatch.setattr(proxy, "prefill_session", None, raising=False)
    monkeypatch.setattr(proxy, "decode_session", None, raising=False)
    monkeypatch.setattr(proxy.app.state, "shutdown_drain_seconds", 0.5, raising=False)
    stats = proxy.app.state.stats
    stats.inflight = 1

    async def release_later():
        await asyncio.sleep(0.05)
        stats.inflight = 0

    async def run_shutdown():
        started = time.monotonic()
        task = asyncio.ensure_future(release_later())
        await proxy.on_shutdown()
        await task
        return time.monotonic() - started

    elapsed = asyncio.run(run_shutdown())
    assert stats.inflight == 0
    assert session.closed
    assert 0.05 <= elapsed < 0.4


def test_drain_times_out_and_closes_with_remaining_inflight(monkeypatch):
    session = FakeSession([])
    monkeypatch.setattr(proxy, "encode_session", session, raising=False)
    monkeypatch.setattr(proxy, "prefill_session", None, raising=False)
    monkeypatch.setattr(proxy, "decode_session", None, raising=False)
    monkeypatch.setattr(proxy.app.state, "shutdown_drain_seconds", 0.2, raising=False)
    stats = proxy.app.state.stats
    stats.inflight = 1

    started = time.monotonic()
    asyncio.run(proxy.on_shutdown())
    elapsed = time.monotonic() - started
    assert 0.15 <= elapsed < 0.5
    assert stats.inflight == 1
    assert session.closed


def test_drain_zero_closes_immediately(monkeypatch):
    session = FakeSession([])
    monkeypatch.setattr(proxy, "encode_session", session, raising=False)
    monkeypatch.setattr(proxy, "prefill_session", None, raising=False)
    monkeypatch.setattr(proxy, "decode_session", None, raising=False)
    monkeypatch.setattr(proxy.app.state, "shutdown_drain_seconds", 0.0, raising=False)
    stats = proxy.app.state.stats
    stats.inflight = 1

    started = time.monotonic()
    asyncio.run(proxy.on_shutdown())
    elapsed = time.monotonic() - started
    assert elapsed < 0.2
    assert session.closed
    assert stats.inflight == 1


# --------------------------------------------------------------------------- #
# CLI flags for the new options
# --------------------------------------------------------------------------- #
def test_new_cli_defaults_and_flags():
    parser = proxy.build_arg_parser()
    required = [
        "--encode-servers-urls",
        ENCODE_URL,
        "--prefill-servers-urls",
        "disable",
        "--decode-servers-urls",
        DECODE_URL,
    ]
    args = parser.parse_args(required)
    assert args.encoder_failure_cooldown_seconds == 30.0
    assert args.probe_encoders_on_startup is False
    assert args.shutdown_drain_seconds == 0.0

    flagged = parser.parse_args(
        required
        + [
            "--encoder-failure-cooldown-seconds",
            "5",
            "--probe-encoders-on-startup",
            "--shutdown-drain-seconds",
            "12.5",
        ]
    )
    assert flagged.encoder_failure_cooldown_seconds == 5.0
    assert flagged.probe_encoders_on_startup is True
    assert flagged.shutdown_drain_seconds == 12.5

    proxy.configure_app_state(flagged)
    assert proxy.app.state.encoder_failure_cooldown_seconds == 5.0
    assert proxy.app.state.probe_encoders_on_startup is True
    assert proxy.app.state.shutdown_drain_seconds == 12.5
    assert isinstance(proxy.app.state.stats, proxy.ProxyStats)


# --------------------------------------------------------------------------- #
# /health contract
# --------------------------------------------------------------------------- #
def test_health_contract_unchanged(monkeypatch):
    encode = FakeSession([FakeResponse(200, b"{}")])
    decode = FakeSession([FakeResponse(200, b"{}")])
    monkeypatch.setattr(proxy, "encode_session", encode, raising=False)
    monkeypatch.setattr(proxy, "prefill_session", None, raising=False)
    monkeypatch.setattr(proxy, "decode_session", decode, raising=False)
    monkeypatch.setattr(proxy.app.state, "e_urls", [ENCODE_URL], raising=False)
    monkeypatch.setattr(proxy.app.state, "p_urls", [], raising=False)
    monkeypatch.setattr(proxy.app.state, "d_urls", [DECODE_URL], raising=False)

    response = asyncio.run(proxy.health_check())
    assert response.status_code == 200
    assert json.loads(response.body) == {
        "proxy": "healthy",
        "encode_cluster": "healthy",
        "prefill_cluster": "empty",
        "decode_cluster": "healthy",
    }

    encode.responses.append(FakeResponse(503, b""))
    response = asyncio.run(proxy.health_check())
    assert response.status_code == 503
    assert json.loads(response.body)["encode_cluster"] == "unhealthy"


# --------------------------------------------------------------------------- #
# Streaming in-flight gauge (N5-r2)
# --------------------------------------------------------------------------- #
def _asgi_scope(spec_version="2.4"):
    return {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": spec_version},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/v1/chat/completions",
        "raw_path": b"/v1/chat/completions",
        "query_string": b"",
        "headers": [],
        "client": ("127.0.0.1", 1234),
        "server": ("127.0.0.1", 8000),
    }


class _FakeRequest:
    def __init__(self, payload, req_id="req-1"):
        self._payload = payload
        self.headers = {"x-request-id": req_id}

    async def json(self):
        return self._payload


class _UnstartedStream:
    """Async iterator that records whether it was ever iterated or closed."""

    def __init__(self):
        self.started = False
        self.closed = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        self.started = True
        raise StopAsyncIteration

    async def aclose(self):
        self.closed = True


def test_tracked_streaming_response_normal_completion():
    stats = proxy.app.state.stats
    stats.inflight = 1
    events = []

    async def body():
        events.append("started")
        try:
            yield b"chunk-1"
            yield b"chunk-2"
        finally:
            events.append("finished")

    response = proxy._TrackedStreamingResponse(body(), media_type="text/event-stream", stats=stats)
    sent = []

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        sent.append(message)

    asyncio.run(response(_asgi_scope(), receive, send))

    assert stats.inflight == 0
    assert events == ["started", "finished"]
    bodies = [message["body"] for message in sent if message["type"] == "http.response.body"]
    assert b"chunk-1" in bodies and b"chunk-2" in bodies


def test_tracked_streaming_response_releases_when_not_started(monkeypatch):
    stats = proxy.app.state.stats
    stats.inflight = 1
    stream = _UnstartedStream()
    response = proxy._TrackedStreamingResponse(stream, media_type="text/event-stream", stats=stats)

    # Simulate the ASGI cancel path where the body iterator is never iterated.
    async def cancelled_call(self, scope, receive, send):
        return

    monkeypatch.setattr(proxy.StreamingResponse, "__call__", cancelled_call)
    asyncio.run(response(_asgi_scope(spec_version="2.3"), None, None))

    assert stats.inflight == 0
    assert stream.started is False
    assert stream.closed is True


def test_tracked_streaming_response_releases_on_exception():
    stats = proxy.app.state.stats
    stats.inflight = 1

    async def body():
        yield b"first"
        raise RuntimeError("stream broke")

    response = proxy._TrackedStreamingResponse(body(), media_type="text/event-stream", stats=stats)

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        pass

    with pytest.raises(RuntimeError, match="stream broke"):
        asyncio.run(response(_asgi_scope(), receive, send))
    assert stats.inflight == 0


def test_tracked_streaming_response_releases_on_cancellation():
    stats = proxy.app.state.stats
    stats.inflight = 1

    async def body():
        yield b"first"

    response = proxy._TrackedStreamingResponse(body(), media_type="text/event-stream", stats=stats)

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        raise asyncio.CancelledError()

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(response(_asgi_scope(), receive, send))
    assert stats.inflight == 0


def test_chat_handler_non_stream_releases_inflight(monkeypatch):
    stats = proxy.app.state.stats
    request = _FakeRequest({"model": "m", "messages": [], "stream": False})

    async def fake_forward(req_data, req_id, p_url, d_url):
        return {"choices": []}

    monkeypatch.setattr(proxy, "forward_non_stream", fake_forward)
    response = asyncio.run(proxy.chat_completions(request))
    assert response.status_code == 200
    assert stats.inflight == 0


def test_chat_handler_non_stream_exception_releases_inflight(monkeypatch):
    stats = proxy.app.state.stats
    request = _FakeRequest({"model": "m", "messages": [], "stream": False})

    async def failing_forward(req_data, req_id, p_url, d_url):
        raise proxy.HTTPException(status_code=400, detail="bad request")

    monkeypatch.setattr(proxy, "forward_non_stream", failing_forward)
    with pytest.raises(proxy.HTTPException):
        asyncio.run(proxy.chat_completions(request))
    assert stats.inflight == 0


def test_chat_handler_streaming_tracks_inflight_until_response(monkeypatch):
    stats = proxy.app.state.stats
    request = _FakeRequest({"model": "m", "messages": [], "stream": True})

    async def fake_stream(req_data, req_id, p_url, d_url):
        yield b"data: ok\n\n"

    monkeypatch.setattr(proxy, "forward_stream", fake_stream)
    response = asyncio.run(proxy.chat_completions(request))
    assert stats.inflight == 1
    assert isinstance(response, proxy._TrackedStreamingResponse)

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        pass

    asyncio.run(response(_asgi_scope(), receive, send))
    assert stats.inflight == 0
