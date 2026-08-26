import argparse
import asyncio
from typing import Any

import pytest

from examples.disaggregated_prefill_v1 import load_balance_proxy_server_example as proxy


def test_assign_instances_releases_prefill_reservation_on_cancelled_error(monkeypatch):
    scheduler = proxy.SharedProxyScheduler([("localhost", 8001)], [("localhost", 8002)])
    runtime = proxy.WorkerRuntime(scheduler)
    monkeypatch.setattr(proxy, "runtime", runtime)
    monkeypatch.setattr(proxy, "global_args", argparse.Namespace(max_retries=1, retry_delay=0))

    async def fail_prefill(*args, **kwargs):
        raise asyncio.CancelledError

    monkeypatch.setattr(proxy, "send_request_to_service", fail_prefill)

    try:
        with pytest.raises(asyncio.CancelledError):
            asyncio.run(proxy.assign_instances("/completions", {"prompt": "hello"}, 16, is_initial_request=True))

        prefiller = next(iter(scheduler.prefillers.values()))
        assert prefiller.active_kv_cache == 0
        assert scheduler.request_num == 0
    finally:
        asyncio.run(runtime.close())


def test_assign_instances_releases_decode_reservation_on_cancelled_error(monkeypatch):
    scheduler = proxy.SharedProxyScheduler([("localhost", 8001)], [("localhost", 8002)])
    runtime = proxy.WorkerRuntime(scheduler)
    monkeypatch.setattr(proxy, "runtime", runtime)
    monkeypatch.setattr(proxy, "global_args", argparse.Namespace(max_retries=1, retry_delay=0))

    class Response:
        def json(self):
            return {}

    async def finish_prefill(*args, **kwargs):
        return Response()

    monkeypatch.setattr(proxy, "send_request_to_service", finish_prefill)

    original_get_client = runtime.get_client

    async def fail_decoder_client(role, key):
        if role is proxy.ServerRole.DECODE:
            raise asyncio.CancelledError
        return await original_get_client(role, key)

    monkeypatch.setattr(runtime, "get_client", fail_decoder_client)

    try:
        with pytest.raises(asyncio.CancelledError):
            asyncio.run(proxy.assign_instances("/completions", {"prompt": "hello"}, 16, is_initial_request=True))

        prefiller = next(iter(scheduler.prefillers.values()))
        decoder = next(iter(scheduler.decoders.values()))
        assert prefiller.active_kv_cache == 0
        assert decoder.active_tokens == 0
        assert scheduler.request_num == 0
    finally:
        asyncio.run(runtime.close())


def test_reassign_instances_does_not_release_prefill_kv_twice(monkeypatch):
    scheduler = proxy.SharedProxyScheduler(
        [("localhost", 8001), ("localhost", 8002)],
        [("localhost", 8003)],
    )
    runtime = proxy.WorkerRuntime(scheduler)
    monkeypatch.setattr(proxy, "runtime", runtime)

    # Keep unrelated in-flight KV reservations on both prefillers. The request
    # being recomputed lands on the second prefiller with 50 tokens already
    # reserved by another request.
    scheduler.begin_request(50.0)
    scheduler.begin_request(50.0)
    scheduler.begin_request(50.0)
    previous_prefiller = scheduler.begin_request(120.0)
    previous_decoder = scheduler.pick_decoder(120.0)
    previous_instance = proxy.InstanceInfo(
        request_id="request-id",
        prefiller_key=previous_prefiller["key"],
        prefiller_score=120.0,
        decoder_key=previous_decoder["key"],
        decoder_score=120.0,
        decoder_host=previous_decoder["host"],
        decoder_port=previous_decoder["port"],
    )

    # The first decoder chunk has already released this request's KV score.
    scheduler.release_prefill_kv(previous_instance.prefiller_key, previous_instance.prefiller_score)

    sentinel = object()

    async def fake_assign_instances(*args: Any, **kwargs: Any) -> object:
        assert kwargs["is_initial_request"] is False
        return sentinel

    monkeypatch.setattr(proxy, "assign_instances", fake_assign_instances)

    try:
        result = asyncio.run(
            proxy.reassign_instances(
                "/completions",
                {"prompt": "hello"},
                16,
                previous_instance,
                previous_prefiller_kv_released=True,
            )
        )

        assert result is sentinel
        assert scheduler.prefillers[previous_instance.prefiller_key].active_kv_cache == 50.0
        assert scheduler.decoders[previous_instance.decoder_key].active_tokens == 0.0
    finally:
        asyncio.run(runtime.close())
