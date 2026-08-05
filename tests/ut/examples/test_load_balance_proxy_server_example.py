import argparse
import importlib.util
import sys
from pathlib import Path

import pytest
from fastapi import HTTPException

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = REPO_ROOT / "examples" / "disaggregated_prefill_v1" / "load_balance_proxy_server_example.py"
SPEC = importlib.util.spec_from_file_location("load_balance_proxy_server_example", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
proxy = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = proxy
SPEC.loader.exec_module(proxy)

PREFILLERS = [("10.0.0.1", 8100), ("10.0.0.2", 8100)]
DECODERS = [("10.0.1.1", 8200), ("10.0.1.2", 8200)]


def make_scheduler(prefill_policy="consistent_hash", decode_policy="least_loaded", **kwargs):
    return proxy.SharedProxyScheduler(
        PREFILLERS,
        DECODERS,
        prefill_policy=prefill_policy,
        decode_policy=decode_policy,
        **kwargs,
    )


def finish_prefill_only(scheduler, picked):
    scheduler.complete_prefill(picked["key"])
    scheduler.finish_request(None)


def test_policy_defaults_and_choices(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["load_balance_proxy_server_example.py"])
    args = proxy.parse_args()

    assert args.prefill_policy == "consistent_hash"
    assert args.decode_policy == "least_loaded"
    assert set(proxy.PREFILL_POLICIES) == {
        "round_robin",
        "random",
        "consistent_hash",
        "least_loaded",
        "cache_aware",
    }
    assert set(proxy.DECODE_POLICIES) == {"round_robin", "random", "least_loaded"}


def test_prefill_round_robin_is_independent_and_ordered():
    scheduler = make_scheduler(prefill_policy="round_robin")
    selected = []
    for i in range(4):
        picked = scheduler.begin_request(f"request-{i}", "prompt", "model")
        selected.append(picked["key"])
        finish_prefill_only(scheduler, picked)

    assert selected == [
        proxy.server_key(*PREFILLERS[0]),
        proxy.server_key(*PREFILLERS[1]),
        proxy.server_key(*PREFILLERS[0]),
        proxy.server_key(*PREFILLERS[1]),
    ]


def test_consistent_hash_keeps_same_key_on_same_prefill():
    scheduler = make_scheduler(prefill_policy="consistent_hash")
    selected = []
    for _ in range(5):
        picked = scheduler.begin_request("session:stable", "prompt", "model")
        selected.append(picked["key"])
        finish_prefill_only(scheduler, picked)

    assert len(set(selected)) == 1


def test_least_loaded_uses_only_inflight_counts_for_both_roles():
    scheduler = make_scheduler(prefill_policy="least_loaded", decode_policy="least_loaded")

    first_prefill = scheduler.begin_request("one", "prompt", "model")
    second_prefill = scheduler.begin_request("two", "prompt", "model")
    assert first_prefill["key"] != second_prefill["key"]

    first_decode = scheduler.pick_decoder()
    second_decode = scheduler.pick_decoder()
    assert first_decode["key"] != second_decode["key"]

    scheduler.complete_prefill(first_prefill["key"])
    scheduler.complete_prefill(second_prefill["key"])
    scheduler.release_decoder(first_decode["key"])
    scheduler.release_decoder(second_decode["key"])
    scheduler.finish_request(None)
    scheduler.finish_request(None)

    health = scheduler.healthcheck()
    assert set(health["prefill_loads"].values()) == {0}
    assert set(health["decode_loads"].values()) == {0}
    assert health["request_num"] == 0


def test_random_policy_selects_only_available_prefillers():
    scheduler = make_scheduler(prefill_policy="random")
    available = {proxy.server_key(*server) for server in PREFILLERS}

    for i in range(20):
        picked = scheduler.begin_request(f"request-{i}", "prompt", "model")
        assert picked["key"] in available
        finish_prefill_only(scheduler, picked)


def test_cache_aware_reuses_prefix_but_yields_to_inflight_imbalance():
    scheduler = make_scheduler(
        prefill_policy="cache_aware",
        cache_threshold=0.3,
        cache_balance_abs_threshold=2,
        cache_balance_rel_threshold=1.5,
    )
    prompt = "shared-prefix-" * 100

    first = scheduler.begin_request("first", prompt, "model-a")
    owner = first["key"]
    finish_prefill_only(scheduler, first)

    second = scheduler.begin_request("second", prompt + "tail", "model-a")
    assert second["key"] == owner
    finish_prefill_only(scheduler, second)

    scheduler.prefillers[owner].inflight_requests = 3
    overloaded = scheduler.begin_request("third", prompt + "another-tail", "model-a")
    assert overloaded["key"] != owner
    finish_prefill_only(scheduler, overloaded)


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


class FakeClient:
    def __init__(self, payloads):
        self.payloads = list(payloads)
        self.calls = []

    async def post(self, endpoint, *, json, headers):
        self.calls.append((endpoint, json, headers))
        return FakeResponse(self.payloads.pop(0))


class FakeRuntime:
    def __init__(self, scheduler, prefill_client):
        self.scheduler = scheduler
        self.prefill_client = prefill_client
        self.scheduled_methods = []

    async def schedule(self, method, /, *args, **kwargs):
        self.scheduled_methods.append(method)
        return getattr(self.scheduler, method)(*args, **kwargs)

    async def get_client(self, role, key):
        assert role is proxy.ServerRole.PREFILL
        return self.prefill_client


@pytest.mark.asyncio
async def test_missing_kv_retries_same_prefill_once_then_succeeds(monkeypatch):
    scheduler = make_scheduler()
    client = FakeClient(
        [
            {"id": "first-without-kv"},
            {
                "id": "second-with-kv",
                "kv_transfer_params": {"remote_engine_id": "engine-1"},
            },
        ]
    )
    fake_runtime = FakeRuntime(scheduler, client)
    monkeypatch.setattr(proxy, "runtime", fake_runtime)
    monkeypatch.setattr(proxy, "global_args", argparse.Namespace(retry_delay=0.0))

    instance = await proxy.assign_instances(
        "/chat/completions",
        {"model": "model-a", "messages": [{"role": "user", "content": "hello"}]},
        "session:test",
        is_initial_request=True,
    )

    assert len(client.calls) == 2
    assert client.calls[0][2]["X-Request-Id"] == client.calls[1][2]["X-Request-Id"]
    assert scheduler.prefillers[instance.prefiller_key].inflight_requests == 0
    assert scheduler.decoders[instance.decoder_key].inflight_requests == 1
    scheduler.finish_request(instance.decoder_key)


@pytest.mark.asyncio
async def test_missing_kv_twice_returns_502_without_selecting_decode(monkeypatch):
    scheduler = make_scheduler()
    client = FakeClient([{"id": "first"}, {"id": "second"}])
    fake_runtime = FakeRuntime(scheduler, client)
    monkeypatch.setattr(proxy, "runtime", fake_runtime)
    monkeypatch.setattr(proxy, "global_args", argparse.Namespace(retry_delay=0.0))

    with pytest.raises(HTTPException) as exc_info:
        await proxy.assign_instances(
            "/chat/completions",
            {"model": "model-a", "messages": [{"role": "user", "content": "hello"}]},
            "session:test",
            is_initial_request=True,
        )

    assert exc_info.value.status_code == 502
    assert len(client.calls) == 2
    assert "pick_decoder" not in fake_runtime.scheduled_methods
    health = scheduler.healthcheck()
    assert set(health["prefill_loads"].values()) == {0}
    assert set(health["decode_loads"].values()) == {0}
    assert health["request_num"] == 0


@pytest.mark.asyncio
async def test_reassign_failure_keeps_previous_decoder_reserved(monkeypatch):
    scheduler = make_scheduler()
    previous_decoder = scheduler.pick_decoder()
    previous_instance = proxy.InstanceInfo(
        request_id="request-id",
        prefiller_key=proxy.server_key(*PREFILLERS[0]),
        decoder_key=previous_decoder["key"],
        decoder_host=previous_decoder["host"],
        decoder_port=previous_decoder["port"],
    )
    fake_runtime = FakeRuntime(scheduler, prefill_client=None)
    monkeypatch.setattr(proxy, "runtime", fake_runtime)

    async def fail_assignment(*args, **kwargs):
        raise HTTPException(status_code=502, detail="prefill failed")

    monkeypatch.setattr(proxy, "assign_instances", fail_assignment)

    with pytest.raises(HTTPException, match="prefill failed"):
        await proxy.reassign_instances(
            "/chat/completions",
            {"model": "model-a"},
            "session:test",
            previous_instance,
        )

    assert scheduler.decoders[previous_instance.decoder_key].inflight_requests == 1
    scheduler.finish_request(previous_instance.decoder_key)
    assert scheduler.decoders[previous_instance.decoder_key].inflight_requests == 0
