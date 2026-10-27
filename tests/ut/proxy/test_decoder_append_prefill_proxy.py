# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""CPU HTTP contract tests for the standalone Decoder Append-Prefill proxy."""

import argparse
import asyncio
import copy
import json
import sys
import unittest
from typing import Any

import httpx
import pytest

from examples.disaggregated_prefill_v1 import decoder_append_prefill_proxy_server as proxy
from examples.disaggregated_prefill_v1 import load_balance_proxy_server_example as base_proxy


class FakeRequest:
    def __init__(self, headers=None, payload=None):
        self.headers = httpx.Headers(headers or {})
        self.payload = payload or {"model": "test", "prompt": "complete prompt", "max_tokens": 4}

    async def json(self):
        return dict(self.payload)

    async def body(self):
        return json.dumps(self.payload).encode()


def make_scheduler():
    return proxy.SharedProxyScheduler(
        prefiller_instances=[("127.0.0.1", 8100)],
        decoder_instances=[("127.0.0.1", 8200), ("127.0.0.1", 8201)],
    )


@pytest.mark.asyncio
async def test_entrypoints_keep_configuration_and_runtime_state_separate(monkeypatch):
    base_scheduler = base_proxy.SharedProxyScheduler(
        prefiller_instances=[("127.0.0.1", 8300), ("127.0.0.1", 8301)],
        decoder_instances=[("127.0.0.1", 8400)],
    )
    append_scheduler = make_dp4_scheduler()
    base_args = argparse.Namespace(workers=1, enable_prefill_affinity=True)
    append_args = argparse.Namespace(workers=1, enable_decoder_append_prefill=True, session_affinity_ttl=60.0)
    for module, args, scheduler in (
        (base_proxy, base_args, base_scheduler),
        (proxy, append_args, append_scheduler),
    ):
        monkeypatch.setattr(module, "global_args", args)
        monkeypatch.setattr(module, "shared_scheduler", scheduler)
        monkeypatch.setattr(module, "runtime", None)

    async with base_proxy.lifespan(base_proxy.app):
        base_runtime = base_proxy.get_runtime()
        async with proxy.lifespan(proxy.app):
            append_runtime = proxy.get_runtime()
            assert base_proxy.get_global_args() is base_args
            assert proxy.get_global_args() is append_args
            assert base_runtime is not append_runtime
            assert base_runtime.scheduler is base_scheduler
            assert append_runtime.scheduler is append_scheduler
            assert base_proxy.app is not proxy.app

            decoder_key = bind_and_finish_first_turn(append_scheduler)
            route = await proxy.plan_request_route(FakeRequest({"X-Session-ID": "session-1"}), {})
            assert route.pinned_decoder_key == decoder_key
            assert route.pinned_decoder_dp_rank == 0
            append_scheduler.finish_session("session-1", True)

            first = base_scheduler.begin_request(10.0, affinity_key="original-proxy-session")
            base_scheduler.finish_request(first["key"], 10.0, None, 0.0, release_prefill_kv=True)
            second = base_scheduler.begin_request(20.0, affinity_key="original-proxy-session")
            assert second["key"] == first["key"]
            assert base_scheduler.request_num == 1
            assert base_scheduler.prefillers[second["key"]].active_kv_cache == 20.0
            assert append_scheduler.request_num == 0
            base_scheduler.finish_request(second["key"], 20.0, None, 0.0, release_prefill_kv=True)
            base_clients = [client for clients in base_runtime._clients.values() for client in clients.values()]
            append_clients = [client for clients in append_runtime._clients.values() for client in clients.values()]

        assert proxy.runtime is None
        assert all(client.is_closed for client in append_clients)
        assert base_proxy.get_runtime() is base_runtime
        assert all(not client.is_closed for client in base_clients)

    assert base_proxy.runtime is None
    assert all(client.is_closed for client in base_clients)


def bind_and_finish_first_turn(scheduler, session_id="session-1"):
    reservation = scheduler.reserve_session(session_id, 60.0)
    assert reservation["status"] == "cold"
    decoder = scheduler.pick_decoder(10.0)
    scheduler.bind_session_decoder(session_id, decoder["key"])
    scheduler.finish_session(session_id, True)
    scheduler.release_decoder(decoder["key"], 10.0)
    return decoder["key"]


def test_session_affinity_lifecycle_and_busy_guard():
    scheduler = make_scheduler()

    cold = scheduler.reserve_session("session-1", 60.0)
    assert cold == {"status": "cold", "decoder_key": None, "turn_count": 0}
    assert scheduler.reserve_session("session-1", 60.0) == {"status": "busy"}

    decoder = scheduler.pick_decoder(10.0)
    scheduler.bind_session_decoder("session-1", decoder["key"])
    scheduler.finish_session("session-1", True)
    scheduler.release_decoder(decoder["key"], 10.0)

    warm = scheduler.reserve_session("session-1", 60.0)
    assert warm["status"] == "warm"
    assert warm["decoder_key"] == decoder["key"]
    assert warm["turn_count"] == 1
    scheduler.finish_session("session-1", False)


def test_decoder_removal_invalidates_affinity():
    scheduler = make_scheduler()
    decoder_key = bind_and_finish_first_turn(scheduler)
    decoder = scheduler.decoders[decoder_key]

    scheduler.remove_instances(proxy.ServerRole.DECODE, [(decoder.host, decoder.port)])

    assert "session-1" not in scheduler.session_affinities


def test_decoder_local_request_has_symmetric_load_accounting():
    scheduler = make_scheduler()
    decoder_key = bind_and_finish_first_turn(scheduler)

    scheduler.begin_decoder_request(decoder_key, 25.0)
    assert scheduler.request_num == 1
    assert scheduler.decoders[decoder_key].active_tokens == 25.0

    scheduler.finish_request(None, 0.0, decoder_key, 25.0, release_prefill_kv=False)
    assert scheduler.request_num == 0
    assert scheduler.decoders[decoder_key].active_tokens == 0.0


@pytest.mark.asyncio
async def test_enabled_routes_warm_session_to_pinned_decoder(monkeypatch):
    scheduler = make_scheduler()
    decoder_key = bind_and_finish_first_turn(scheduler)
    args = argparse.Namespace(enable_decoder_append_prefill=True, session_affinity_ttl=60.0)
    monkeypatch.setattr(proxy, "global_args", args)
    monkeypatch.setattr(proxy, "runtime", proxy.WorkerRuntime(scheduler))

    route = await proxy.plan_request_route(
        FakeRequest({proxy.SESSION_ID_HEADER: "session-1"}),
        {"prompt": "complete prompt"},
    )

    assert route.mode is proxy.RouteMode.DECODER_LOCAL
    assert route.pinned_decoder_key == decoder_key
    scheduler.finish_session("session-1", False)


@pytest.mark.asyncio
async def test_enabled_routes_body_session_to_pinned_decoder(monkeypatch):
    scheduler = make_scheduler()
    decoder_key = bind_and_finish_first_turn(scheduler)
    args = argparse.Namespace(enable_decoder_append_prefill=True, session_affinity_ttl=60.0)
    monkeypatch.setattr(proxy, "global_args", args)
    monkeypatch.setattr(proxy, "runtime", proxy.WorkerRuntime(scheduler))

    req_data = {"prompt": "complete prompt", "conversation_id": "session-1"}
    route = await proxy.plan_request_route(FakeRequest(), req_data)

    assert route.mode is proxy.RouteMode.DECODER_LOCAL
    assert route.reason == "affinity_hit"
    assert route.pinned_decoder_key == decoder_key
    assert "conversation_id" not in req_data
    scheduler.finish_session("session-1", False)


@pytest.mark.asyncio
async def test_disabled_ignores_legacy_decoder_local_header(monkeypatch):
    scheduler = make_scheduler()
    bind_and_finish_first_turn(scheduler)
    args = argparse.Namespace(enable_decoder_append_prefill=False, session_affinity_ttl=60.0)
    monkeypatch.setattr(proxy, "global_args", args)
    monkeypatch.setattr(proxy, "runtime", proxy.WorkerRuntime(scheduler))

    route = await proxy.plan_request_route(
        FakeRequest(
            {
                proxy.SESSION_ID_HEADER: "session-1",
                "X-vLLM-Request-Mode": "decoder-local",
            }
        ),
        {"prompt": "complete prompt"},
    )

    assert route.mode is proxy.RouteMode.PD
    assert route.pinned_decoder_key is None
    scheduler.finish_session("session-1", False)


def make_dp4_scheduler(epoch="decoder-test-epoch"):
    return proxy.SharedProxyScheduler(
        prefiller_instances=[("127.0.0.1", 8100)],
        decoder_instances=[("127.0.0.1", 8200)],
        decoder_data_parallel_size=4,
        decoder_engine_epoch=epoch,
    )


def test_dp_rank_round_robin_and_affinity_survives_other_sessions():
    scheduler = make_dp4_scheduler()
    for index in range(6):
        session = f"session-{index}"
        assert scheduler.reserve_session(session, 60)["status"] == "cold"
        decoder = scheduler.pick_decoder(10)
        assert decoder["dp_rank"] == index % 4
        scheduler.bind_session_decoder(session, decoder["key"], decoder["dp_rank"])
        scheduler.finish_session(session, True)
        scheduler.release_decoder(decoder["key"], 10)
    for index in range(6):
        reservation = scheduler.reserve_session(f"session-{index}", 60)
        assert reservation["decoder_dp_rank"] == index % 4
        assert reservation["decoder_engine_epoch"] == "decoder-test-epoch"
    assert scheduler.healthcheck()["affinity_kind"] == "proxy_hint_no_backend_lease"


def test_restart_same_epoch_is_cold_and_epoch_change_invalidates_affinity():
    scheduler = make_dp4_scheduler()
    bind_and_finish_first_turn(scheduler)
    restarted = make_dp4_scheduler()
    assert restarted.reserve_session("session-1", 60)["status"] == "cold"
    scheduler.decoder_engine_epoch = "replacement-engine"
    assert scheduler.reserve_session("session-1", 60)["status"] == "cold"


def test_rank_validation_has_no_load_side_effect():
    scheduler = make_dp4_scheduler()
    key = next(iter(scheduler.decoders))
    for invalid in (-1, 4):
        with pytest.raises(ValueError, match="outside configured DP size"):
            scheduler.begin_decoder_request(key, 10, invalid)
    assert scheduler.request_num == 0
    assert scheduler.decoders[key].active_tokens == 0
    with pytest.raises(ValueError):
        proxy.SharedProxyScheduler([], [], decoder_data_parallel_size=0)


@pytest.mark.asyncio
async def test_disabled_cleans_proxy_fields_and_stale_metadata(monkeypatch):
    monkeypatch.setattr(proxy, "global_args", argparse.Namespace(enable_decoder_append_prefill=False))
    body = {"conversation_id": "session", "kv_transfer_params": {"stale": True}, "prompt": "hello"}
    route = await proxy.plan_request_route(FakeRequest(), body)
    assert route.reason == "feature_disabled"
    assert body == {"prompt": "hello"}


def install_mock_backends(monkeypatch, scheduler, enabled=True):
    seen = []

    def backend(request):
        assert request.method == "POST"
        assert request.url.path in {"/v1/completions", "/v1/chat/completions"}
        body = json.loads(request.content)
        seen.append((request.url.port, dict(request.headers), body))
        result: dict[str, Any] = {
            "id": "backend-response",
            "choices": [{"text": " done", "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1024, "completion_tokens": 2, "prompt_tokens_details": {"cached_tokens": 512}},
        }
        if request.url.port == 8100:
            result["kv_transfer_params"] = {"remote_engine_id": "fresh-engine", "remote_block_ids": [1, 2]}
            result["usage"]["prompt_tokens_details"]["cached_tokens"] = 128
        return httpx.Response(200, json=result)

    runtime = proxy.WorkerRuntime(scheduler)
    transport = httpx.MockTransport(backend)
    for role, instances in [
        (proxy.ServerRole.PREFILL, scheduler.prefillers),
        (proxy.ServerRole.DECODE, scheduler.decoders),
    ]:
        for key, server in instances.items():
            runtime._clients[role][key] = httpx.AsyncClient(
                base_url=proxy.build_base_url(server.host, server.port),
                transport=transport,
            )
    monkeypatch.setattr(proxy, "runtime", runtime)
    monkeypatch.setattr(
        proxy,
        "global_args",
        argparse.Namespace(
            enable_decoder_append_prefill=enabled,
            session_affinity_ttl=60.0,
            max_retries=1,
            retry_delay=0.0,
        ),
    )
    return runtime, seen


async def consume_proxy_response(request):
    response = await proxy.handle_completions_impl("/completions", request)
    chunks = [chunk async for chunk in response.body_iterator]
    return response, json.loads(b"".join(chunks))


@pytest.mark.asyncio
async def test_cold_and_warm_local_send_only_trusted_decoder_rank(monkeypatch):
    scheduler = make_dp4_scheduler()
    runtime, seen = install_mock_backends(monkeypatch, scheduler)
    session_headers = {"X-Session-ID": "experiment-session", "X-data-parallel-rank": "99"}
    payload = {
        "prompt": "prefix and append",
        "conversation_id": "ignored-body-session",
        "kv_transfer_params": {"remote_engine_id": "expired-client-engine"},
    }
    try:
        cold, cold_body = await consume_proxy_response(FakeRequest(session_headers, payload))
        assert cold.headers["X-Proxy-Route"] == "pd"
        assert cold.headers["X-Decoder-DP-Rank"] == "0"
        assert cold.headers["X-Decoder-Engine-Epoch"] == "decoder-test-epoch"
        assert cold_body["usage"]["prompt_tokens_details"]["cached_tokens"] == 128
        warm_local, local_body = await consume_proxy_response(
            FakeRequest(
                {**session_headers},
                payload,
            )
        )
        assert warm_local.headers["X-Proxy-Route"] == "decoder_local"
        assert warm_local.headers["X-Decoder-DP-Rank"] == "0"
        assert local_body["usage"]["prompt_tokens_details"]["cached_tokens"] == 512
        assert local_body["proxy_metrics"]["final_route"] == "decoder_local"
        assert local_body["proxy_metrics"]["recovered"] is False
        assert local_body["proxy_metrics"]["cached_tokens_source"] == "decoder"
        assert [port for port, _, _ in seen] == [8100, 8200, 8200]
        for port, headers, body in seen:
            assert "conversation_id" not in body
            if port == 8100:
                assert "x-data-parallel-rank" not in headers
                assert body["kv_transfer_params"]["remote_engine_id"] is None
            else:
                assert headers["x-data-parallel-rank"] == "0"
                if body is not seen[-1][2]:
                    assert body["kv_transfer_params"]["remote_engine_id"] == "fresh-engine"
        assert "kv_transfer_params" not in seen[-1][2]
        assert seen[-1][2]["vllm_xargs"]["require_decoder_cache_hit"] == 1
        assert all("require_decoder_cache_hit" not in body.get("vllm_xargs", {}) for _, _, body in seen[:-1])
        assert scheduler.healthcheck()["request_num"] == 0
        assert scheduler.healthcheck()["in_flight_sessions"] == 0
    finally:
        await runtime.close()


@pytest.mark.asyncio
async def test_all_cold_and_no_session_decode_legs_receive_round_robin_rank(monkeypatch):
    scheduler = make_dp4_scheduler()
    runtime, seen = install_mock_backends(monkeypatch, scheduler)
    try:
        responses = [await consume_proxy_response(FakeRequest({"X-Session-ID": f"s-{i}"})) for i in range(4)]
        assert [response.headers["X-Decoder-DP-Rank"] for response, _ in responses] == ["0", "1", "2", "3"]
        response, _ = await consume_proxy_response(FakeRequest({"X-data-parallel-rank": "3"}))
        assert response.headers["X-Decoder-DP-Rank"] == "0"
        assert [h["x-data-parallel-rank"] for port, h, _ in seen if port == 8200] == ["0", "1", "2", "3", "0"]
    finally:
        await runtime.close()


@pytest.mark.asyncio
async def test_same_session_is_busy_while_first_response_unconsumed(monkeypatch):
    scheduler = make_dp4_scheduler()
    runtime, _ = install_mock_backends(monkeypatch, scheduler)
    try:
        request = FakeRequest({"X-Session-ID": "one-at-a-time"})
        first = await proxy.handle_completions_impl("/completions", request)
        second = await proxy.handle_completions_impl("/completions", request)
        assert second.status_code == 409
        assert json.loads(second.body)["error"]["code"] == "session_busy"
        _ = [chunk async for chunk in first.body_iterator]
        assert scheduler.healthcheck()["in_flight_sessions"] == 0
    finally:
        await runtime.close()


@pytest.mark.asyncio
async def test_decoder_local_stream_preserves_empty_choice_usage_chunk(monkeypatch):
    scheduler = make_dp4_scheduler()
    runtime, _ = install_mock_backends(monkeypatch, scheduler)
    try:
        await consume_proxy_response(FakeRequest({"X-Session-ID": "stream-session"}))

        async def decode_stream(*args, **kwargs):
            assert kwargs["decoder_dp_rank"] == 0
            yield b'data: {"choices":[{"text":"hello"}]}\n\n'
            yield b'data: {"choices":[],"usage":{"prompt_tokens_details":{"cached_tokens":512}}}\n\n'
            yield b"data: [DONE]\n\n"

        monkeypatch.setattr(proxy, "stream_service_response", decode_stream)
        response = await proxy.handle_completions_impl(
            "/completions",
            FakeRequest(
                {"X-Session-ID": "stream-session"},
                {"prompt": "prefix and append", "stream": True, "stream_options": {"include_usage": True}},
            ),
        )
        chunks = [chunk async for chunk in response.body_iterator]
        assert len(chunks) == 3
        usage = json.loads(chunks[1][len(b"data: ") :])
        assert usage["usage"]["prompt_tokens_details"]["cached_tokens"] == 512
        assert scheduler.session_affinities["stream-session"].turn_count == 2
    finally:
        await runtime.close()


@pytest.mark.asyncio
async def test_decoder_local_recovery_reports_final_pd_route_and_cache_source(monkeypatch):
    scheduler = make_dp4_scheduler()
    runtime, _ = install_mock_backends(monkeypatch, scheduler)
    try:
        await consume_proxy_response(FakeRequest({"X-Session-ID": "recover-session"}))
        decoder_key = next(iter(scheduler.decoders))
        decoder_client = runtime._clients[proxy.ServerRole.DECODE][decoder_key]
        await decoder_client.aclose()
        decoder_calls = []

        def backend(request):
            decoder_calls.append(request)
            recomputed = len(decoder_calls) == 1
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {"text": "" if recomputed else "done", "stop_reason": "recomputed" if recomputed else None}
                    ],
                    "usage": {
                        "completion_tokens": 0 if recomputed else 1,
                        "prompt_tokens_details": {"cached_tokens": 999},
                    },
                },
            )

        runtime._clients[proxy.ServerRole.DECODE][decoder_key] = httpx.AsyncClient(
            base_url="http://127.0.0.1:8200/v1",
            transport=httpx.MockTransport(backend),
        )
        response, body = await consume_proxy_response(
            FakeRequest(
                {"X-Session-ID": "recover-session"},
            )
        )
        assert response.headers["X-Proxy-Route"] == "decoder_local"
        assert body["proxy_metrics"]["initial_route"] == "decoder_local"
        assert body["proxy_metrics"]["final_route"] == "pd"
        assert body["proxy_metrics"]["recovered"] is True
        assert body["proxy_metrics"]["recovery_count"] == 1
        assert body["proxy_metrics"]["cached_tokens_source"] == "prefiller"
        assert body["usage"]["prompt_tokens_details"]["cached_tokens"] == 128
        assert [request.headers["x-data-parallel-rank"] for request in decoder_calls] == ["0", "0"]
        assert scheduler.healthcheck()["request_num"] == 0
    finally:
        await runtime.close()


@pytest.mark.asyncio
async def test_fail_closed_rejects_missing_prefiller_transfer_before_decoder(monkeypatch):
    scheduler = make_dp4_scheduler()
    runtime, seen = install_mock_backends(monkeypatch, scheduler)
    proxy.global_args.disable_recomputed_retry = True
    prefiller_key = next(iter(scheduler.prefillers))
    await runtime._clients[proxy.ServerRole.PREFILL][prefiller_key].aclose()
    runtime._clients[proxy.ServerRole.PREFILL][prefiller_key] = httpx.AsyncClient(
        base_url="http://127.0.0.1:8100/v1",
        transport=httpx.MockTransport(lambda _: httpx.Response(200, json={"choices": [{"text": "x"}]})),
    )
    try:
        response = await proxy.handle_completions_impl("/completions", FakeRequest({"X-Session-ID": "fail-session"}))
        assert response.status_code == 502
        assert json.loads(response.body)["error"]["code"] == "missing_prefiller_kv_transfer"
        assert not seen
        assert scheduler.healthcheck()["request_num"] == 0
        assert scheduler.healthcheck()["active_sessions"] == 0
    finally:
        await runtime.close()


@pytest.mark.asyncio
async def test_fail_closed_nonstream_recomputed_returns_502_without_p_retry(monkeypatch):
    scheduler = make_dp4_scheduler()
    runtime, seen = install_mock_backends(monkeypatch, scheduler)
    try:
        await consume_proxy_response(FakeRequest({"X-Session-ID": "fail-session"}))
        proxy.global_args.disable_recomputed_retry = True
        decoder_key = next(iter(scheduler.decoders))
        await runtime._clients[proxy.ServerRole.DECODE][decoder_key].aclose()
        runtime._clients[proxy.ServerRole.DECODE][decoder_key] = httpx.AsyncClient(
            base_url="http://127.0.0.1:8200/v1",
            transport=httpx.MockTransport(
                lambda request: httpx.Response(
                    200,
                    json={"choices": [{"text": "", "stop_reason": "recomputed"}]},
                )
            ),
        )
        response = await proxy.handle_completions_impl(
            "/completions",
            FakeRequest(
                {"X-Session-ID": "fail-session"},
            ),
        )
        assert response.status_code == 502
        assert json.loads(response.body)["error"]["code"] == "decoder_recomputed_retry_disabled"
        assert len(seen) == 2  # Only the initial cold P and D were called.
        assert scheduler.healthcheck()["request_num"] == 0
        assert scheduler.healthcheck()["in_flight_sessions"] == 0
        assert scheduler.session_affinities["fail-session"].turn_count == 1
    finally:
        await runtime.close()


def test_expired_session_loses_old_rank_binding(monkeypatch):
    scheduler = make_dp4_scheduler()
    bind_and_finish_first_turn(scheduler)
    affinity = scheduler.session_affinities["session-1"]
    monkeypatch.setattr(proxy.time, "monotonic", lambda: affinity.last_access_time + 61)
    assert scheduler.reserve_session("session-1", 60)["status"] == "cold"
    assert scheduler.session_affinities["session-1"].decoder_dp_rank is None


class InterruptedStream(httpx.AsyncByteStream):
    async def __aiter__(self):
        yield b'data: {"choices":[{"text":"hello"}]}\n\n'
        raise httpx.ReadError("injected interruption")


@pytest.mark.asyncio
async def test_partial_stream_failure_propagates_and_never_retries():
    requests = []

    def backend(request):
        requests.append(request)
        return httpx.Response(200, stream=InterruptedStream())

    async with httpx.AsyncClient(transport=httpx.MockTransport(backend)) as client:
        stream = proxy.stream_service_response(client, "http://decoder/completions", {"stream": True}, "r1", 2)
        assert json.loads((await anext(stream))[6:]) == {"choices": [{"text": "hello"}]}
        with pytest.raises(httpx.ReadError):
            await anext(stream)
    assert len(requests) == 1
    assert requests[0].headers["x-data-parallel-rank"] == "2"


class Frames(httpx.AsyncByteStream):
    def __init__(self, payloads):
        self.payloads = payloads
        self.closed = False

    async def __aiter__(self):
        wire = b"".join(b"data: " + json.dumps(payload).encode() + b"\n\n" for payload in self.payloads)
        wire += b"data: [DONE]\n\n"
        for offset in range(0, len(wire), 7):
            yield wire[offset : offset + 7]

    async def aclose(self):
        self.closed = True


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("chat", [False, True])
async def test_admission_miss_runs_d_p_d_with_original_request(monkeypatch, stream, chat):
    scheduler = make_dp4_scheduler()
    bind_and_finish_first_turn(scheduler)
    runtime, _ = install_mock_backends(monkeypatch, scheduler)
    proxy.global_args.disable_recomputed_retry = True
    seen = []
    streams: list[Frames] = []
    payload = {"max_tokens": 16, "cache_salt": "same-history", "stream": stream, "vllm_xargs": {"other_option": 3}}
    if chat:
        payload["messages"] = [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "previous"},
            {"role": "user", "content": "followup"},
        ]
    else:
        payload["prompt"] = [7, 9, 11, 13]
    original = copy.deepcopy(payload)

    def backend(request):
        body = json.loads(request.content)
        seen.append((request.url.port, dict(request.headers), body))
        miss = len(seen) == 1
        if request.url.port == 8100:
            assert all(s.closed for s in streams), "D stream must close before P starts"
            return httpx.Response(
                200,
                json={
                    "choices": [{"text": "ignored P token"}],
                    "kv_transfer_params": {"remote_engine_id": "fresh", "remote_block_ids": [1]},
                    "usage": {"prompt_tokens_details": {"cached_tokens": 128}},
                },
            )
        result = {
            "choices": [
                {
                    ("delta" if stream else "message") if chat else "text": {"content": "" if miss else "answer"}
                    if chat
                    else ""
                    if miss
                    else "answer",
                    "token_ids": [] if miss else [25],
                    "finish_reason": "stop",
                    "stop_reason": "decoder_cache_miss" if miss else None,
                }
            ],
            "usage": {"completion_tokens": 0 if miss else 1},
        }
        if stream:
            frames = Frames([result])
            streams.append(frames)
            return httpx.Response(200, stream=frames)
        return httpx.Response(200, json=result)

    for role, instances in [
        (proxy.ServerRole.PREFILL, scheduler.prefillers),
        (proxy.ServerRole.DECODE, scheduler.decoders),
    ]:
        for key, server in instances.items():
            await runtime._clients[role][key].aclose()
            runtime._clients[role][key] = httpx.AsyncClient(
                base_url=proxy.build_base_url(server.host, server.port), transport=httpx.MockTransport(backend)
            )
    try:
        response = await proxy.handle_completions_impl(
            "/chat/completions" if chat else "/completions",
            FakeRequest({"X-Session-ID": "session-1"}, payload),
        )
        chunks = [chunk async for chunk in response.body_iterator]
        result = json.loads(chunks[0][6:] if stream else chunks[0])
        assert [port for port, _, _ in seen] == [8200, 8100, 8200]
        for port, headers, body in seen:
            assert body.get("prompt") == original.get("prompt")
            assert body.get("messages") == original.get("messages")
            assert body["cache_salt"] == original["cache_salt"]
            assert body["vllm_xargs"]["other_option"] == 3
            if port == 8200:
                assert headers["x-data-parallel-rank"] == "0"
                assert body["max_tokens"] == 16
        assert seen[0][2]["vllm_xargs"]["require_decoder_cache_hit"] == 1
        assert all("require_decoder_cache_hit" not in row[2]["vllm_xargs"] for row in seen[1:])
        metrics = result["proxy_metrics"]
        assert metrics["initial_route"] == "decoder_local"
        assert metrics["final_route"] == "pd"
        assert metrics["admission_fallback"] is True
        assert metrics["fallback_reason"] == "decoder_cache_miss"
        assert metrics["recovery_count"] == 1
        assert metrics["decoder_engine_epoch"] == "decoder-test-epoch"
        assert metrics["cached_tokens_source"] == "prefiller"
        assert scheduler.request_num == 0
        assert all(s.active_tokens == 0 for s in scheduler.decoders.values())
        assert all(s.active_kv_cache == 0 for s in scheduler.prefillers.values())
        assert scheduler.session_affinities["session-1"].turn_count == 2
        assert all(s.closed for s in streams)
    finally:
        await runtime.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "prior",
    [
        {"text": "partial"},
        {"token_ids": [42]},
        {"delta": {"reasoning": "thinking"}},
        {"delta": {"tool_calls": [{"id": "call1"}]}},
    ],
)
async def test_miss_after_any_output_never_replays(monkeypatch, prior):
    scheduler = make_dp4_scheduler()
    bind_and_finish_first_turn(scheduler)
    runtime, seen = install_mock_backends(monkeypatch, scheduler)
    closed = []

    async def decode_stream(*args, **kwargs):
        try:
            yield b"data: " + json.dumps({"choices": [prior]}).encode() + b"\n\n"
            yield b'data: {"choices":[{"text":"","stop_reason":"decoder_cache_miss"}]}\n\n'
        finally:
            closed.append(True)

    monkeypatch.setattr(proxy, "stream_service_response", decode_stream)
    try:
        response = await proxy.handle_completions_impl(
            "/completions",
            FakeRequest(
                {"X-Session-ID": "session-1"},
                {"prompt": [1, 2], "stream": True},
            ),
        )
        chunks = [c async for c in response.body_iterator]
        assert json.loads(chunks[-1][6:])["error"]["code"] == "invalid_decoder_cache_miss"
        assert not seen  # No P call after the injected D stream.
        assert closed == [True]
        assert scheduler.request_num == 0
        assert scheduler.session_affinities["session-1"].turn_count == 1
    finally:
        await runtime.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("reason", ["decoder_cache_miss", "decoder_cache_lost_after_output"])
async def test_failed_fallback_never_sends_unprefilled_request_to_d(monkeypatch, reason):
    scheduler = make_dp4_scheduler()
    bind_and_finish_first_turn(scheduler)
    runtime, _ = install_mock_backends(monkeypatch, scheduler)
    proxy.global_args.disable_recomputed_retry = False
    seen = []

    def backend(request):
        seen.append(request.url.port)
        # P intentionally violates the transfer contract. D must not be called again.
        result = {"choices": [{"text": "", "stop_reason": reason}]}
        return httpx.Response(200, json=result)

    for role, instances in [
        (proxy.ServerRole.PREFILL, scheduler.prefillers),
        (proxy.ServerRole.DECODE, scheduler.decoders),
    ]:
        for key, server in instances.items():
            await runtime._clients[role][key].aclose()
            runtime._clients[role][key] = httpx.AsyncClient(
                base_url=proxy.build_base_url(server.host, server.port), transport=httpx.MockTransport(backend)
            )
    try:
        response = await proxy.handle_completions_impl("/completions", FakeRequest({"X-Session-ID": "session-1"}))
        chunks = [c async for c in response.body_iterator]
        error = json.loads(b"".join(chunks))["error"]["code"]
        if reason == "decoder_cache_miss":
            assert seen == [8200, 8100]
            assert error == "missing_prefiller_kv_transfer"
        else:
            assert seen == [8200]
            assert error == reason
        assert scheduler.request_num == 0
        assert all(s.active_tokens == 0 for s in scheduler.decoders.values())
        assert all(s.active_kv_cache == 0 for s in scheduler.prefillers.values())
        assert scheduler.session_affinities["session-1"].turn_count == 1
    finally:
        await runtime.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("failure", ["no_prefiller", "http_503"])
async def test_admission_fallback_failure_returns_error_and_releases_once(monkeypatch, stream, failure):
    scheduler = make_dp4_scheduler()
    bind_and_finish_first_turn(scheduler)
    runtime, _ = install_mock_backends(monkeypatch, scheduler)
    proxy.global_args.max_retries = 1
    seen = []

    def backend(request):
        seen.append(request.url.port)
        if request.url.port == 8100:
            return httpx.Response(503, json={"error": "injected P unavailable"})
        miss = {
            "choices": [{"text": "", "token_ids": [], "finish_reason": "stop", "stop_reason": "decoder_cache_miss"}]
        }
        return httpx.Response(200, stream=Frames([miss])) if stream else httpx.Response(200, json=miss)

    for role, instances in [
        (proxy.ServerRole.PREFILL, scheduler.prefillers),
        (proxy.ServerRole.DECODE, scheduler.decoders),
    ]:
        for key, server in instances.items():
            await runtime._clients[role][key].aclose()
            runtime._clients[role][key] = httpx.AsyncClient(
                base_url=proxy.build_base_url(server.host, server.port), transport=httpx.MockTransport(backend)
            )
    if failure == "no_prefiller":
        scheduler.remove_instances(proxy.ServerRole.PREFILL, [("127.0.0.1", 8100)])
    try:
        response = await proxy.handle_completions_impl(
            "/completions",
            FakeRequest(
                {"X-Session-ID": "session-1"},
                {"prompt": [7, 9, 11], "max_tokens": 16, "stream": stream},
            ),
        )
        raw = b"".join([chunk async for chunk in response.body_iterator]).decode()
        events = (
            [json.loads(line[6:]) for line in raw.splitlines() if line.startswith("data: {")]
            if stream
            else [json.loads(raw)]
        )
        assert events[0]["error"]["code"] == "decoder_cache_miss_fallback_failed"
        assert seen == ([8200] if failure == "no_prefiller" else [8200, 8100])
        assert scheduler.request_num == 0
        assert scheduler.healthcheck()["in_flight_sessions"] == 0
        assert all(server.active_tokens == 0 for server in scheduler.decoders.values())
        assert all(server.active_kv_cache == 0 for server in scheduler.prefillers.values())
    finally:
        await runtime.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["prefill", "decoder_open"])
async def test_disconnect_waits_for_assignment_cleanup(monkeypatch, stage):
    scheduler = make_dp4_scheduler()
    runtime, _ = install_mock_backends(monkeypatch, scheduler)
    entered = asyncio.Event()
    disconnect = asyncio.Event()
    closed = []

    class Request(FakeRequest):
        async def receive(self):
            await disconnect.wait()
            return {"type": "http.disconnect"}

    async def prefill(*args, **kwargs):
        try:
            entered.set()
            await asyncio.Event().wait()
        finally:
            closed.append("prefill")

    async def decode(*args, **kwargs):
        try:
            entered.set()
            await asyncio.Event().wait()
            yield b"unreachable"
        finally:
            closed.append("decoder_open")

    if stage == "prefill":
        monkeypatch.setattr(proxy, "send_request_to_service", prefill)
    else:
        monkeypatch.setattr(proxy, "stream_service_response", decode)
    request = Request(
        {"X-Session-ID": "cancel-while-assigning"}, {"prompt": [1, 2, 3], "stream": True, "max_tokens": 4}
    )
    wrapped = proxy.with_cancellation(lambda request: proxy.handle_completions_impl("/completions", request))
    task = asyncio.create_task(wrapped(request=request))
    try:
        await asyncio.wait_for(entered.wait(), 1)
        assert scheduler.healthcheck()["request_num"] == 1
        disconnect.set()
        await asyncio.wait_for(task, 1)
        assert closed == [stage]
        assert scheduler.healthcheck()["request_num"] == 0
        assert scheduler.healthcheck()["in_flight_sessions"] == 0
        assert all(x.active_tokens == 0 for x in scheduler.decoders.values())
        assert all(x.active_kv_cache == 0 for x in scheduler.prefillers.values())
    finally:
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await runtime.close()


class FrameTests(unittest.TestCase):
    def test_every_byte_boundary_and_coalescing(self):
        payload = 'data:{"content":"中文"}\r\n\r\ndata: {"usage":{"cached_tokens":1920}}\n\ndata: [DONE]\n\n'.encode()
        for size in [1, 2, 7, 65536]:
            p = proxy.SSEFramer()
            frames = []
            for offset in range(0, len(payload), size):
                frames.extend(p.feed(payload[offset : offset + size]))
            p.finish()
            self.assertEqual(len(frames), 3)
            self.assertEqual(json.loads(frames[0][6:]), {"content": "中文"})
            self.assertEqual(json.loads(frames[1][6:])["usage"]["cached_tokens"], 1920)

    def test_comments_and_multiline_json(self):
        p = proxy.SSEFramer()
        frames = p.feed(b': heartbeat\n\ndata: {"x":\ndata: 1}\n\ndata:[DONE]\n\n')
        p.finish()
        self.assertEqual(json.loads(frames[0][6:]), {"x": 1})

    def test_truncated_stream_rejected(self):
        for payload in [b'data: {"x":1}', b'data: {"x":1}\n\n', b"data: [DONE]\n"]:
            p = proxy.SSEFramer()
            p.feed(payload)
            with self.assertRaises(ValueError):
                p.finish()

    def test_malformed_or_after_done_rejected(self):
        for payload in [b"data: notjson\n\n", b"data:[DONE]\n\ndata:{}\n\n"]:
            with self.assertRaises(ValueError):
                proxy.SSEFramer().feed(payload)


def test_sse_size_limit_applies_per_event_not_http_chunk():
    parser = proxy.SSEFramer(limit=32)
    frames = parser.feed(b'data: {"x":1}\n\n' * 100 + b"data: [DONE]\n\n")
    parser.finish()
    assert len(frames) == 101
    with pytest.raises(ValueError, match="size limit"):
        proxy.SSEFramer(limit=32).feed(b"data: " + b"x" * 33)


@pytest.mark.asyncio
async def test_closing_framed_stream_closes_transport():
    transport = Frames([{"choices": [{"stop_reason": "decoder_cache_miss"}]}])
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, stream=transport))
    ) as client:
        stream = proxy.stream_service_response(client, "http://decoder/completions", {"stream": True}, "r1", 0)
        frame = await anext(stream)
        assert json.loads(frame[6:])["choices"][0]["stop_reason"] == "decoder_cache_miss"
        await stream.aclose()
        assert transport.closed


@pytest.mark.asyncio
async def test_incomplete_event_then_disconnect_never_retries():
    calls = []

    class InterruptedPartialEvent(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b'data: {"choices":'
            raise httpx.ReadError("injected interruption")

    def backend(request):
        calls.append(request)
        return httpx.Response(200, stream=InterruptedPartialEvent())

    async with httpx.AsyncClient(transport=httpx.MockTransport(backend)) as client:
        stream = proxy.stream_service_response(client, "http://decoder/completions", {"stream": True}, "r1", 0)
        with pytest.raises(httpx.ReadError):
            await anext(stream)
    assert len(calls) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("tail", [b"", b'data: {"choices":', b"data: invalid-json\n\n"])
async def test_broken_stream_reports_failure_and_releases_session(monkeypatch, tail):
    scheduler = make_dp4_scheduler()
    bind_and_finish_first_turn(scheduler)
    runtime, _ = install_mock_backends(monkeypatch, scheduler)
    closed = []
    calls = []

    class BrokenStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b'data: {"choices":[{"text":"hello"}]}\n\n'
            yield tail

        async def aclose(self):
            closed.append(True)

    def backend(request):
        calls.append(request)
        return httpx.Response(200, stream=BrokenStream())

    for key, server in scheduler.decoders.items():
        await runtime._clients[proxy.ServerRole.DECODE][key].aclose()
        runtime._clients[proxy.ServerRole.DECODE][key] = httpx.AsyncClient(
            base_url=proxy.build_base_url(server.host, server.port), transport=httpx.MockTransport(backend)
        )
    try:
        response = await proxy.handle_completions_impl(
            "/completions",
            FakeRequest(
                {"X-Session-ID": "session-1"},
                {"prompt": [1, 2, 3], "max_tokens": 4, "stream": True},
            ),
        )
        events = [json.loads(chunk[6:]) async for chunk in response.body_iterator]
        assert events[0]["choices"][0]["text"] == "hello"
        assert events[-1]["error"]["type"] == "proxy_stream_error"
        assert len(calls) == 1
        assert closed == [True]
        assert scheduler.healthcheck()["request_num"] == 0
        assert scheduler.healthcheck()["in_flight_sessions"] == 0
        assert scheduler.session_affinities["session-1"].turn_count == 1
        assert all(server.active_tokens == 0 for server in scheduler.decoders.values())
    finally:
        await runtime.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("enabled", [False, True])
async def test_startup_and_dispatch_only_send_inference_requests(monkeypatch, enabled):
    scheduler = make_dp4_scheduler()
    bind_and_finish_first_turn(scheduler)
    runtime, seen = install_mock_backends(monkeypatch, scheduler, enabled=enabled)
    proxy.global_args.workers = 1
    monkeypatch.setattr(proxy, "_ensure_scheduler", lambda _: scheduler)
    monkeypatch.setattr(proxy, "WorkerRuntime", lambda _: runtime)
    clients = [client for role_clients in runtime._clients.values() for client in role_clients.values()]

    async with proxy.lifespan(proxy.app):
        assert not seen
        response, body = await consume_proxy_response(FakeRequest({"X-Session-ID": "session-1"}))
        expected_route = "decoder_local" if enabled else "pd"
        assert response.headers["X-Proxy-Route"] == expected_route
        assert body["proxy_metrics"]["final_route"] == expected_route
        assert [port for port, _, _ in seen] == ([8200] if enabled else [8100, 8200])
        assert seen[-1][2].get("vllm_xargs", {}).get("require_decoder_cache_hit") == (1 if enabled else None)
        assert scheduler.healthcheck()["request_num"] == 0
        assert scheduler.healthcheck()["in_flight_sessions"] == 0
    assert proxy.runtime is None
    assert all(not clients for clients in runtime._clients.values())
    assert all(client.is_closed for client in clients)


def test_proxy_cli_has_only_boolean_append_switch(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["proxy"])
    assert proxy.parse_args().enable_decoder_append_prefill is False
    monkeypatch.setattr(sys, "argv", ["proxy", "--enable-decoder-append-prefill"])
    assert proxy.parse_args().enable_decoder_append_prefill is True
    monkeypatch.setattr(sys, "argv", ["proxy", "--decoder-append-routing-mode", "auto"])
    with pytest.raises(SystemExit):
        proxy.parse_args()
