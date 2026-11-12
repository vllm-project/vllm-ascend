# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from dataclasses import replace
from types import SimpleNamespace

import pytest
import requests
from starlette.requests import Request

from examples.rfork.rfork_planner import Scheduler, SeedRemovalResult, Settings, Store, _build_arg_parser, build_router


def test_lease_ttl_default_and_configuration_precedence(monkeypatch):
    monkeypatch.delenv("RFORK_MOCK_LEASE_TTL_SEC", raising=False)
    assert Settings().lease_ttl_sec == 60
    assert Settings.from_env().lease_ttl_sec == 60
    assert _build_arg_parser().parse_args([]).lease_ttl_sec == 60
    monkeypatch.setenv("RFORK_MOCK_LEASE_TTL_SEC", "120")
    assert Settings.from_env().lease_ttl_sec == 120
    assert _build_arg_parser().parse_args([]).lease_ttl_sec == 120
    assert _build_arg_parser().parse_args(["--lease-ttl-sec", "180"]).lease_ttl_sec == 180


def test_non_positive_lease_ttl_rejected(monkeypatch):
    value = 0
    with pytest.raises(ValueError, match="lease_ttl_sec must be > 0"):
        Settings(lease_ttl_sec=value)
    monkeypatch.setenv("RFORK_MOCK_LEASE_TTL_SEC", str(value))
    with pytest.raises(ValueError, match="lease_ttl_sec must be > 0"):
        Settings.from_env()


@pytest.fixture
def planner():
    clock = SimpleNamespace(now=0.0)
    store = Store(heartbeat_ttl_sec=60, default_resource_points=1, scheduler=Scheduler(), time_fn=lambda: clock.now)
    seed_args = dict(seed_key="model-key", seed_ip="127.0.0.1", seed_port=1234, seed_rank=0)
    seed = store.add_seed(**seed_args)
    _, lease = store.get_seed(seed_key="model-key")
    return SimpleNamespace(clock=clock, store=store, seed_args=seed_args, seed=seed, lease=lease)


def test_lease_expiry_boundary_and_capacity_reclaimed_once():
    ttl = 60
    clock = SimpleNamespace(now=0.0)
    store = Store(
        heartbeat_ttl_sec=ttl * 2,
        lease_ttl_sec=ttl,
        default_resource_points=1,
        scheduler=Scheduler(),
        time_fn=lambda: clock.now,
    )
    seed = store.add_seed(seed_key="k", seed_ip="127.0.0.1", seed_port=1234, seed_rank=0)
    _, lease = store.get_seed(seed_key="k")
    clock.now = ttl - 0.001
    assert store.gc_expired_leases() == 0
    assert store.get_seed(seed_key="k") is None
    clock.now = ttl
    assert store.gc_expired_leases() == 1
    assert seed.resource_used == 0
    assert store.gc_expired_leases() == 0
    _, next_lease = store.get_seed(seed_key="k")
    assert not store.put_seed(
        seed_ip=seed.seed_ip, seed_port=seed.seed_port, seed_rank=seed.seed_rank, user_id=lease.user_id
    )
    assert seed.resource_used == 1
    assert next_lease.user_id != lease.user_id


def test_seed_heartbeat_does_not_renew_lease(planner):
    planner.clock.now = 59
    planner.store.add_seed(**planner.seed_args)
    planner.clock.now = 60
    assert planner.store.gc() == (0, 1)
    assert planner.seed.resource_used == 0
    assert planner.store.debug_snapshot()["seed_count"] == 1


def test_stale_seed_drains_until_its_lease_is_released():
    clock = SimpleNamespace(now=0.0)
    store = Store(
        heartbeat_ttl_sec=60,
        lease_ttl_sec=120,
        default_resource_points=1,
        scheduler=Scheduler(),
        time_fn=lambda: clock.now,
    )
    store.add_seed(seed_key="k", seed_ip="127.0.0.1", seed_port=1234, seed_rank=0)
    assert store.get_seed(seed_key="k") is not None
    clock.now = 60
    assert store.gc() == (1, 0)
    snapshot = store.debug_snapshot()
    assert snapshot["lease_count"] == 1
    assert next(iter(snapshot["seeds"].values()))["draining"] is True
    assert store.get_seed(seed_key="k") is None
    assert store.renew_lease(seed_ip="127.0.0.1", seed_port=1234, seed_rank=0, user_id=next(iter(snapshot["leases"])))
    assert store.put_seed(seed_ip="127.0.0.1", seed_port=1234, seed_rank=0, user_id=next(iter(snapshot["leases"])))
    assert store.debug_snapshot()["seed_count"] == 0


def test_remove_seed_stops_new_allocations_until_active_lease_releases(planner):
    assert planner.store.remove_seed(**planner.seed_args) is SeedRemovalResult.DRAINING
    assert planner.store.get_seed(seed_key="model-key") is None
    assert planner.store.renew_lease(
        seed_ip=planner.seed.seed_ip,
        seed_port=planner.seed.seed_port,
        seed_rank=planner.seed.seed_rank,
        user_id=planner.lease.user_id,
    )
    assert planner.store.put_seed(
        seed_ip=planner.seed.seed_ip,
        seed_port=planner.seed.seed_port,
        seed_rank=planner.seed.seed_rank,
        user_id=planner.lease.user_id,
    )
    assert planner.store.remove_seed(**planner.seed_args) is SeedRemovalResult.NOT_FOUND


LEASE_TTL_SEC = 60


class _RoutedResponse:
    """Adapt a Starlette ``Response`` to the ``requests`` response surface."""

    def __init__(self, response):
        self.status_code = response.status_code
        self.headers = response.headers
        self.text = response.body.decode()


class _PlannerTransport:
    """Route client HTTP calls into the planner's real endpoints."""

    # The client catches requests.RequestException, so expose the real class.
    RequestException = requests.RequestException

    def __init__(self, store, planner_url="http://planner"):
        self.planner_url = planner_url
        # Match both the path and HTTP method to reject requests using the wrong verb.
        self.routes = {
            (route.path, verb): route.endpoint for route in build_router(store).routes for verb in route.methods
        }
        self.calls: list[tuple[str, str]] = []
        self.request_headers: list[dict[str, str]] = []

    def _dispatch(self, method, url, headers=None, timeout=None, allow_redirects=None):
        assert url.startswith(self.planner_url), url
        path = url[len(self.planner_url) :]
        assert timeout is not None, "the client must always bound its planner requests"
        self.calls.append((method, path))
        self.request_headers.append(dict(headers or {}))
        assert path in {route_path for route_path, _ in self.routes}, f"client called an unknown planner route: {path}"
        endpoint = self.routes.get((path, method))
        assert endpoint is not None, f"planner has no {method} handler for {path}"
        # ASGI carries header names lowercased; Starlette matches raw keys as-is.
        scope_headers = [(key.lower().encode(), value.encode()) for key, value in (headers or {}).items()]
        return _RoutedResponse(endpoint(Request({"type": "http", "headers": scope_headers})))

    def get(self, url, **kwargs):
        return self._dispatch("GET", url, **kwargs)

    def post(self, url, **kwargs):
        return self._dispatch("POST", url, **kwargs)


def _make_planner_http(
    runtime,
    monkeypatch,
    *,
    send_model_identity_headers: bool = False,
    is_draft_model: bool = False,
):
    clock = SimpleNamespace(now=0.0)
    store = Store(
        heartbeat_ttl_sec=LEASE_TTL_SEC * 2,
        lease_ttl_sec=LEASE_TTL_SEC,
        default_resource_points=1,
        scheduler=Scheduler(),
        time_fn=lambda: clock.now,
    )
    transport = _PlannerTransport(store)
    monkeypatch.setattr(runtime.client, "requests", transport)
    monkeypatch.setattr(runtime.client, "SEED_REMOVAL_RETRY_BACKOFF_SEC", 0)
    config = replace(runtime.config, send_model_identity_headers=send_model_identity_headers)
    identity = replace(runtime.identity, is_draft_model=is_draft_model)
    client = runtime.client.RForkPlannerClient(config, identity)
    client.bind_structural_digest("digest")
    return SimpleNamespace(
        clock=clock,
        store=store,
        transport=transport,
        client=client,
        removal_max_attempts=runtime.client.SEED_REMOVAL_MAX_ATTEMPTS,
    )


@pytest.fixture
def planner_http(runtime, monkeypatch):
    return _make_planner_http(runtime, monkeypatch)


def _advertise(planner_http, port=1234):
    result = planner_http.client.report_seed_once(port, seed_ip="127.0.0.1")
    assert result.status.name == "ACCEPTED"


@pytest.mark.parametrize("send_model_identity_headers, is_draft_model", [(False, False), (True, True)])
def test_full_seed_lifecycle_over_the_real_planner_routes(
    runtime, monkeypatch, send_model_identity_headers, is_draft_model
):
    planner_http = _make_planner_http(
        runtime,
        monkeypatch,
        send_model_identity_headers=send_model_identity_headers,
        is_draft_model=is_draft_model,
    )
    client = planner_http.client

    _advertise(planner_http)
    lease = client.acquire_seed()

    assert lease is not None
    assert (lease.seed_ip, lease.seed_port, lease.seed_rank) == ("127.0.0.1", 1234, 0)
    assert lease.seed_key == client.seed_key
    assert lease.lease_ttl_sec == LEASE_TTL_SEC
    assert lease.user_id

    assert client.renew_seed_once(lease)
    assert client.release_seed_once(lease).name == "RELEASED"
    assert client.remove_seed()

    assert [path for _, path in planner_http.transport.calls] == [
        "/add_seed",
        "/get_seed",
        "/renew_seed_lease",
        "/put_seed",
        "/remove_seed",
    ]
    expected_identity_headers = {
        "MODEL_URL": "model",
        "DEPLOY_STRATEGY": "strategy",
        "IS_DRAFT": str(is_draft_model).lower(),
    }
    expected_original_headers = {
        "/add_seed": {
            "SEED_KEY": client.seed_key,
            "SEED_IP": "127.0.0.1",
            "SEED_PORT": "1234",
            "SEED_RANK": "0",
            "SEED_REFCNT": "0",
        },
        "/get_seed": {"SEED_KEY": client.seed_key},
        "/renew_seed_lease": {
            "SEED_IP": "127.0.0.1",
            "SEED_PORT": "1234",
            "USER_ID": lease.user_id,
            "SEED_RANK": "0",
        },
        "/put_seed": {
            "SEED_IP": "127.0.0.1",
            "SEED_PORT": "1234",
            "USER_ID": lease.user_id,
            "SEED_RANK": "0",
        },
        "/remove_seed": {
            "SEED_KEY": client.seed_key,
            "SEED_IP": "127.0.0.1",
            "SEED_PORT": "1234",
            "SEED_RANK": "0",
        },
    }
    for (_, path), headers in zip(planner_http.transport.calls, planner_http.transport.request_headers):
        expected_headers = expected_original_headers[path]
        if send_model_identity_headers:
            expected_headers = {**expected_headers, **expected_identity_headers}
        assert headers == expected_headers
    assert planner_http.store.debug_snapshot()["seed_count"] == 0


def test_planner_does_not_hand_the_same_seed_to_two_concurrent_receivers(planner_http):
    _advertise(planner_http)

    # Default capacity is one point, so the second lease request finds nothing.
    first = planner_http.client.acquire_seed()
    assert first is not None
    assert planner_http.client.acquire_seed() is None

    assert planner_http.client.release_seed_once(first).name == "RELEASED"
    assert planner_http.client.acquire_seed() is not None


def test_removing_a_seed_with_an_active_lease_is_reported_as_failure(planner_http):
    client = planner_http.client
    _advertise(planner_http)
    assert client.acquire_seed() is not None

    # A 409 response keeps the advertisement available for a later withdrawal retry.
    assert not client.remove_seed()
    assert client.last_advertisement is not None
    removal_calls = [path for _, path in planner_http.transport.calls].count("/remove_seed")
    assert removal_calls == planner_http.removal_max_attempts
