import argparse
import asyncio

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
