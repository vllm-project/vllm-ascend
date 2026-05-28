# SPDX-License-Identifier: Apache-2.0

import argparse
import asyncio
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from examples.disaggregated_prefill_v1 import load_balance_proxy_server_example as proxy


class FakeRequest:
    def __init__(self, payload):
        self._payload = payload
        self._body = json.dumps(payload).encode("utf-8")

    async def json(self):
        return self._payload.copy()

    async def body(self):
        return self._body


class FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload


class FakeRuntime:
    def __init__(self, scheduler):
        self.scheduler = scheduler
        self.prefiller_clients = {}
        self.decoder_clients = {}

    async def sync_clients(self):
        snapshot = self.scheduler.get_snapshot()
        self.prefiller_clients = {
            proxy.server_key(server["host"], server["port"]): SimpleNamespace(
                kind="prefill",
                base_url=proxy.build_base_url(server["host"], server["port"]),
                **server,
            )
            for server in snapshot["prefill_instances"]
        }
        self.decoder_clients = {
            proxy.server_key(server["host"], server["port"]): SimpleNamespace(
                kind="decode",
                base_url=proxy.build_base_url(server["host"], server["port"]),
                **server,
            )
            for server in snapshot["decode_instances"]
        }

    def get_prefiller_client(self, key):
        return self.prefiller_clients[key]

    def get_decoder_client(self, key):
        return self.decoder_clients[key]


@pytest.fixture(autouse=True)
def reset_proxy_globals(monkeypatch):
    scheduler = proxy.SharedProxyScheduler(
        [("127.0.0.1", 8100), ("127.0.0.1", 8101)],
        [("127.0.0.1", 8200), ("127.0.0.1", 8201)],
    )
    monkeypatch.setattr(proxy, "runtime", FakeRuntime(scheduler))
    monkeypatch.setattr(proxy, "global_args", argparse.Namespace(max_retries=1, retry_delay=0))


def test_proxy_basic_completion_flow(monkeypatch):
    asyncio.run(_test_proxy_basic_completion_flow(monkeypatch))


async def _test_proxy_basic_completion_flow(monkeypatch):
    calls = []

    async def fake_prefill_request(client, prefiller_key, endpoint, req_data, request_id, **_kwargs):
        calls.append(("prefill", client.port, prefiller_key, endpoint, req_data.copy(), request_id))
        return FakeResponse(
            {
                "kv_transfer_params": {
                    "do_remote_prefill": True,
                    "remote_engine_id": "engine-0",
                    "remote_block_ids": [1, 2],
                    "remote_host": "127.0.0.1",
                    "remote_port": client.port,
                }
            }
        )

    async def fake_decode_stream(client, endpoint, req_data, request_id, **_kwargs):
        calls.append(("decode", client.port, endpoint, req_data.copy(), request_id))
        yield json.dumps({"choices": [{"text": "ok"}], "usage": {"completion_tokens": 1}}).encode("utf-8")

    monkeypatch.setattr(proxy, "send_request_to_service", fake_prefill_request)
    monkeypatch.setattr(proxy, "stream_service_response_with_retry", fake_decode_stream)

    request = FakeRequest({"model": "test-model", "prompt": "hello", "max_tokens": 4})
    response = await proxy.handle_completions_impl("/completions", request)
    chunks = [chunk async for chunk in response.body_iterator]

    assert chunks == [b'{"choices": [{"text": "ok"}], "usage": {"completion_tokens": 1}}']
    assert [call[0] for call in calls] == ["prefill", "decode"]
    assert calls[1][3]["kv_transfer_params"]["remote_engine_id"] == "engine-0"
    assert proxy.runtime.scheduler.healthcheck()["request_num"] == 0


def test_proxy_releases_resources_on_cancel(monkeypatch):
    asyncio.run(_test_proxy_releases_resources_on_cancel(monkeypatch))


async def _test_proxy_releases_resources_on_cancel(monkeypatch):
    stream_started = asyncio.Event()

    async def fake_prefill_request(_client, _prefiller_key, _endpoint, _req_data, _request_id, **_kwargs):
        return FakeResponse({"kv_transfer_params": {"remote_engine_id": "engine-0"}})

    async def fake_decode_stream(_client, _endpoint, _req_data, request_id, **_kwargs):
        stream_started.set()
        yield b'{"choices": [{"text": "partial"}], "usage": {"completion_tokens": 1}}'
        await asyncio.Event().wait()

    monkeypatch.setattr(proxy, "send_request_to_service", fake_prefill_request)
    monkeypatch.setattr(proxy, "stream_service_response_with_retry", fake_decode_stream)

    request = FakeRequest({"model": "test-model", "prompt": "hello", "max_tokens": 4, "stream": True})
    response = await proxy.handle_completions_impl("/completions", request)

    async def consume_response():
        async for _chunk in response.body_iterator:
            pass

    task = asyncio.create_task(consume_response())
    await asyncio.wait_for(stream_started.wait(), timeout=1)
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    scheduler = proxy.runtime.scheduler
    prefiller_key = proxy.server_key("127.0.0.1", 8100)
    assert scheduler.healthcheck()["request_num"] == 0
    assert scheduler.prefillers[prefiller_key]["active_kv_cache"] == 0


def test_scheduler_balances_prefillers_and_decoders_under_high_concurrency():
    total_requests = 20000
    prefiller_instances = [("127.0.0.1", port) for port in range(8100, 8104)]
    decoder_instances = [("127.0.0.1", port) for port in range(8200, 8204)]
    scheduler = proxy.SharedProxyScheduler(prefiller_instances, decoder_instances)

    with ThreadPoolExecutor(max_workers=32) as executor:
        prefiller_keys = list(executor.map(lambda _: scheduler.select_prefiller(1.0)["key"], range(total_requests)))
        decoder_keys = list(executor.map(lambda _: scheduler.select_decoder(1.0)["key"], range(total_requests)))

    prefiller_counts = Counter(prefiller_keys)
    decoder_counts = Counter(decoder_keys)

    assert len(prefiller_counts) == len(prefiller_instances)
    assert len(decoder_counts) == len(decoder_instances)
    assert max(prefiller_counts.values()) - min(prefiller_counts.values()) <= 1
    assert max(decoder_counts.values()) - min(decoder_counts.values()) <= 1

    for key, count in prefiller_counts.items():
        scheduler.release_prefiller(key, float(count))
        scheduler.release_prefiller_kv(key, float(count))
    for key, count in decoder_counts.items():
        scheduler.release_decoder(key, float(count))

    assert all(state["active_tokens"] == 0 for state in scheduler.prefillers.values())
    assert all(state["active_kv_cache"] == 0 for state in scheduler.prefillers.values())
    assert all(state["active_tokens"] == 0 for state in scheduler.decoders.values())


def test_scheduler_taints_instances_while_requests_are_in_flight():
    scheduler = proxy.SharedProxyScheduler(
        [("127.0.0.1", 8100), ("127.0.0.1", 8101)],
        [("127.0.0.1", 8200), ("127.0.0.1", 8201)],
    )
    scheduler.request_started()

    assert scheduler.remove_prefillers([("127.0.0.1", 8100)])
    assert scheduler.remove_decoders([("127.0.0.1", 8200)])
    assert scheduler.select_prefiller(1.0)["key"] == proxy.server_key("127.0.0.1", 8101)
    assert scheduler.select_decoder(1.0)["key"] == proxy.server_key("127.0.0.1", 8201)

    scheduler.release_prefiller(proxy.server_key("127.0.0.1", 8101), 1.0)
    scheduler.release_prefiller_kv(proxy.server_key("127.0.0.1", 8101), 1.0)
    scheduler.release_decoder(proxy.server_key("127.0.0.1", 8201), 1.0)
    scheduler.request_finished()
    scheduler.finalize_tainted_instances()

    snapshot = scheduler.get_snapshot()
    assert [server["port"] for server in snapshot["prefill_instances"]] == [8101]
    assert [server["port"] for server in snapshot["decode_instances"]] == [8201]
