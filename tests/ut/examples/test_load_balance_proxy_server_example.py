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

from __future__ import annotations

import asyncio
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
PROXY_PATH = REPO_ROOT / "examples" / "disaggregated_prefill_v1" / "load_balance_proxy_server_example.py"
MODULE_NAME = "vllm_ascend_load_balance_proxy_server_example"


@pytest.fixture(scope="module")
def proxy_module():
    previous_policy = asyncio.get_event_loop_policy()
    previous_uvloop = sys.modules.pop("uvloop", None)
    try:
        sys.modules.pop(MODULE_NAME, None)
        spec = importlib.util.spec_from_file_location(MODULE_NAME, PROXY_PATH)
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        sys.modules[MODULE_NAME] = module
        spec.loader.exec_module(module)
        yield module
    finally:
        asyncio.set_event_loop_policy(previous_policy)
        sys.modules.pop(MODULE_NAME, None)
        sys.modules.pop("uvloop", None)
        if previous_uvloop is not None:
            sys.modules["uvloop"] = previous_uvloop


class RecordingRuntime:
    def __init__(self, scheduler: Any):
        self.scheduler = scheduler
        self.scheduled_methods: list[str] = []

    async def schedule(self, method: str, /, *args, **kwargs):
        self.scheduled_methods.append(method)
        return getattr(self.scheduler, method)(*args, **kwargs)

    async def get_client(self, _role, _key):
        return SimpleNamespace(base_url="http://prefiller/v1")


class FakeResponse:
    def json(self) -> dict:
        return {}


class InvalidJsonResponse:
    def json(self) -> dict:
        raise ValueError("invalid JSON")


def test_import_tolerates_missing_uvloop(monkeypatch):
    module_name = f"{MODULE_NAME}_without_uvloop"
    previous_policy = asyncio.get_event_loop_policy()
    monkeypatch.setitem(sys.modules, "uvloop", None)
    try:
        spec = importlib.util.spec_from_file_location(module_name, PROXY_PATH)
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    finally:
        asyncio.set_event_loop_policy(previous_policy)
        sys.modules.pop(module_name, None)


def test_finish_prefill_and_pick_decoder_transfers_active_pressure(proxy_module):
    scheduler = proxy_module.SharedProxyScheduler(
        [("127.0.0.1", 8100)],
        [("127.0.0.1", 8200)],
    )

    prefiller = scheduler.begin_request(10.0)
    prefiller_entry = scheduler.prefillers[prefiller["key"]]
    assert (prefiller_entry.active_tokens, prefiller_entry.active_kv_cache) == (10.0, 10.0)
    assert scheduler.request_num == 1

    decoder = scheduler.finish_prefill_and_pick_decoder(prefiller["key"], 10.0, 25.0)

    assert (prefiller_entry.active_tokens, prefiller_entry.active_kv_cache) == (0.0, 10.0)
    assert scheduler.decoders[decoder["key"]].active_tokens == 25.0
    assert scheduler.request_num == 1


def test_repeated_release_does_not_make_active_tokens_negative(proxy_module):
    scheduler = proxy_module.SharedProxyScheduler(
        [("127.0.0.1", 8100)],
        [("127.0.0.1", 8200)],
    )

    prefiller = scheduler.begin_request(10.0)
    decoder = scheduler.finish_prefill_and_pick_decoder(prefiller["key"], 10.0, 25.0)
    scheduler.release_decoder(decoder["key"], 25.0)
    scheduler.release_decoder(decoder["key"], 25.0)
    scheduler.abort_prefill(prefiller["key"], 10.0, is_initial_request=True)
    scheduler.abort_prefill(prefiller["key"], 10.0, is_initial_request=True)

    assert scheduler.prefillers[prefiller["key"]].active_tokens == 0.0
    assert scheduler.decoders[decoder["key"]].active_tokens == 0.0
    assert scheduler.request_num == 0


def test_prefill_routing_distinguishes_compute_from_kv_pressure(proxy_module):
    scheduler = proxy_module.SharedProxyScheduler(
        [("127.0.0.1", 8100), ("127.0.0.1", 8101)],
        [("127.0.0.1", 8200)],
    )

    busy_prefiller = scheduler.begin_request(10.0)
    kv_only_prefiller = scheduler.reserve_prefill(10.0)
    decoder = scheduler.finish_prefill_and_pick_decoder(kv_only_prefiller["key"], 10.0, 1.0)
    scheduler.release_decoder(decoder["key"], 1.0)

    next_prefiller = scheduler.reserve_prefill(10.0)

    assert next_prefiller["key"] == kv_only_prefiller["key"]
    assert next_prefiller["key"] != busy_prefiller["key"]


def test_assign_instances_does_not_add_a_success_path_scheduler_rpc(proxy_module, monkeypatch):
    scheduler = proxy_module.SharedProxyScheduler(
        [("127.0.0.1", 8100)],
        [("127.0.0.1", 8200)],
    )
    runtime = RecordingRuntime(scheduler)

    async def send_success(*_args, **_kwargs):
        return FakeResponse()

    monkeypatch.setattr(proxy_module, "runtime", runtime)
    monkeypatch.setattr(proxy_module, "global_args", SimpleNamespace(max_retries=1, retry_delay=0.0))
    monkeypatch.setattr(proxy_module, "send_request_to_service", send_success)

    info = asyncio.run(proxy_module.assign_instances("/completions", {}, 100, is_initial_request=True))

    assert runtime.scheduled_methods == ["begin_request", "finish_prefill_and_pick_decoder"]
    assert scheduler.prefillers[info.prefiller_key].active_tokens == 0.0
    assert scheduler.prefillers[info.prefiller_key].active_kv_cache > 0.0
    assert scheduler.decoders[info.decoder_key].active_tokens == 100.0


def test_assign_instances_releases_prefill_pressure_when_prefill_fails(proxy_module, monkeypatch):
    scheduler = proxy_module.SharedProxyScheduler(
        [("127.0.0.1", 8100)],
        [("127.0.0.1", 8200)],
    )
    runtime = RecordingRuntime(scheduler)

    async def send_failed(*_args, **_kwargs):
        raise RuntimeError("prefill failed")

    monkeypatch.setattr(proxy_module, "runtime", runtime)
    monkeypatch.setattr(proxy_module, "global_args", SimpleNamespace(max_retries=1, retry_delay=0.0))
    monkeypatch.setattr(proxy_module, "send_request_to_service", send_failed)

    with pytest.raises(RuntimeError, match="prefill failed"):
        asyncio.run(proxy_module.assign_instances("/completions", {}, 100, is_initial_request=True))

    prefiller = next(iter(scheduler.prefillers.values()))
    assert (prefiller.active_tokens, prefiller.active_kv_cache) == (0.0, 0.0)
    assert scheduler.request_num == 0
    assert runtime.scheduled_methods == ["begin_request", "abort_prefill"]


def test_assign_instances_releases_prefill_pressure_when_response_json_is_invalid(proxy_module, monkeypatch):
    scheduler = proxy_module.SharedProxyScheduler(
        [("127.0.0.1", 8100)],
        [("127.0.0.1", 8200)],
    )
    runtime = RecordingRuntime(scheduler)

    async def send_invalid_json(*_args, **_kwargs):
        return InvalidJsonResponse()

    monkeypatch.setattr(proxy_module, "runtime", runtime)
    monkeypatch.setattr(proxy_module, "global_args", SimpleNamespace(max_retries=1, retry_delay=0.0))
    monkeypatch.setattr(proxy_module, "send_request_to_service", send_invalid_json)

    with pytest.raises(ValueError, match="invalid JSON"):
        asyncio.run(proxy_module.assign_instances("/completions", {}, 100, is_initial_request=True))

    prefiller = next(iter(scheduler.prefillers.values()))
    assert (prefiller.active_tokens, prefiller.active_kv_cache) == (0.0, 0.0)
    assert scheduler.request_num == 0
    assert runtime.scheduled_methods == ["begin_request", "abort_prefill"]


def test_assign_instances_releases_prefill_pressure_when_no_decoder_exists(proxy_module, monkeypatch):
    scheduler = proxy_module.SharedProxyScheduler(
        [("127.0.0.1", 8100)],
        [],
    )
    runtime = RecordingRuntime(scheduler)

    async def send_success(*_args, **_kwargs):
        return FakeResponse()

    monkeypatch.setattr(proxy_module, "runtime", runtime)
    monkeypatch.setattr(proxy_module, "global_args", SimpleNamespace(max_retries=1, retry_delay=0.0))
    monkeypatch.setattr(proxy_module, "send_request_to_service", send_success)

    with pytest.raises(RuntimeError, match="No available decode servers"):
        asyncio.run(proxy_module.assign_instances("/completions", {}, 100, is_initial_request=True))

    prefiller = next(iter(scheduler.prefillers.values()))
    assert (prefiller.active_tokens, prefiller.active_kv_cache) == (0.0, 0.0)
    assert scheduler.request_num == 0
    assert runtime.scheduled_methods == [
        "begin_request",
        "finish_prefill_and_pick_decoder",
        "abort_prefill",
    ]


@pytest.mark.parametrize("is_initial_request", [True, False])
def test_assign_instances_releases_assignment_when_decoder_client_fails(
    proxy_module,
    monkeypatch,
    is_initial_request,
):
    scheduler = proxy_module.SharedProxyScheduler(
        [("127.0.0.1", 8100)],
        [("127.0.0.1", 8200)],
    )
    runtime = RecordingRuntime(scheduler)
    if not is_initial_request:
        scheduler.request_num = 1

    async def send_success(*_args, **_kwargs):
        return FakeResponse()

    async def get_client(role, _key):
        if role is proxy_module.ServerRole.DECODE:
            raise RuntimeError("decoder client unavailable")
        return SimpleNamespace(base_url="http://prefiller/v1")

    monkeypatch.setattr(runtime, "get_client", get_client)
    monkeypatch.setattr(proxy_module, "runtime", runtime)
    monkeypatch.setattr(proxy_module, "global_args", SimpleNamespace(max_retries=1, retry_delay=0.0))
    monkeypatch.setattr(proxy_module, "send_request_to_service", send_success)

    with pytest.raises(RuntimeError, match="decoder client unavailable"):
        asyncio.run(
            proxy_module.assign_instances(
                "/completions",
                {},
                100,
                is_initial_request=is_initial_request,
            )
        )

    prefiller = next(iter(scheduler.prefillers.values()))
    decoder = next(iter(scheduler.decoders.values()))
    assert (prefiller.active_tokens, prefiller.active_kv_cache) == (0.0, 0.0)
    assert decoder.active_tokens == 0.0
    assert scheduler.request_num == (0 if is_initial_request else 1)
    expected_pick = "begin_request" if is_initial_request else "reserve_prefill"
    assert runtime.scheduled_methods == [
        expected_pick,
        "finish_prefill_and_pick_decoder",
        "abort_assignment",
    ]


@pytest.mark.parametrize("is_initial_request", [True, False])
def test_assign_instances_releases_assignment_when_decoder_client_lookup_is_cancelled(
    proxy_module,
    monkeypatch,
    is_initial_request,
):
    scheduler = proxy_module.SharedProxyScheduler(
        [("127.0.0.1", 8100)],
        [("127.0.0.1", 8200)],
    )
    runtime = RecordingRuntime(scheduler)
    if not is_initial_request:
        scheduler.request_num = 1
    decoder_client_requested = asyncio.Event()

    async def send_success(*_args, **_kwargs):
        return FakeResponse()

    async def get_client(role, _key):
        if role is proxy_module.ServerRole.DECODE:
            decoder_client_requested.set()
            await asyncio.Future()
        return SimpleNamespace(base_url="http://prefiller/v1")

    monkeypatch.setattr(runtime, "get_client", get_client)
    monkeypatch.setattr(proxy_module, "runtime", runtime)
    monkeypatch.setattr(proxy_module, "global_args", SimpleNamespace(max_retries=1, retry_delay=0.0))
    monkeypatch.setattr(proxy_module, "send_request_to_service", send_success)

    async def cancel_assignment():
        task = asyncio.create_task(
            proxy_module.assign_instances(
                "/completions",
                {},
                100,
                is_initial_request=is_initial_request,
            )
        )
        await decoder_client_requested.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(cancel_assignment())

    prefiller = next(iter(scheduler.prefillers.values()))
    decoder = next(iter(scheduler.decoders.values()))
    assert (prefiller.active_tokens, prefiller.active_kv_cache) == (0.0, 0.0)
    assert decoder.active_tokens == 0.0
    assert scheduler.request_num == (0 if is_initial_request else 1)
    expected_pick = "begin_request" if is_initial_request else "reserve_prefill"
    assert runtime.scheduled_methods == [
        expected_pick,
        "finish_prefill_and_pick_decoder",
        "abort_assignment",
    ]


@pytest.mark.parametrize("is_initial_request", [True, False])
def test_assign_instances_releases_prefill_pressure_when_cancelled(
    proxy_module,
    monkeypatch,
    is_initial_request,
):
    scheduler = proxy_module.SharedProxyScheduler(
        [("127.0.0.1", 8100)],
        [("127.0.0.1", 8200)],
    )
    runtime = RecordingRuntime(scheduler)

    async def send_cancelled(*_args, **_kwargs):
        raise asyncio.CancelledError

    monkeypatch.setattr(proxy_module, "runtime", runtime)
    monkeypatch.setattr(proxy_module, "global_args", SimpleNamespace(max_retries=1, retry_delay=0.0))
    monkeypatch.setattr(proxy_module, "send_request_to_service", send_cancelled)

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(
            proxy_module.assign_instances(
                "/completions",
                {},
                100,
                is_initial_request=is_initial_request,
            )
        )

    prefiller = next(iter(scheduler.prefillers.values()))
    assert (prefiller.active_tokens, prefiller.active_kv_cache) == (0.0, 0.0)
    assert scheduler.request_num == 0
    expected_pick = "begin_request" if is_initial_request else "reserve_prefill"
    assert runtime.scheduled_methods == [expected_pick, "abort_prefill"]
