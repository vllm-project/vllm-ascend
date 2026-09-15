# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from types import SimpleNamespace

import pytest
from starlette.requests import Request

from examples.rfork.rfork_planner import Scheduler, Settings, Store, _build_arg_parser, build_router


def test_lease_ttl_default_and_configuration_precedence(monkeypatch):
    monkeypatch.delenv("RFORK_MOCK_LEASE_TTL_SEC", raising=False)
    assert Settings().lease_ttl_sec == 60
    assert Settings.from_env().lease_ttl_sec == 60
    assert _build_arg_parser().parse_args([]).lease_ttl_sec == 60
    monkeypatch.setenv("RFORK_MOCK_LEASE_TTL_SEC", "120")
    assert Settings.from_env().lease_ttl_sec == 120
    assert _build_arg_parser().parse_args([]).lease_ttl_sec == 120
    assert _build_arg_parser().parse_args(["--lease-ttl-sec", "180"]).lease_ttl_sec == 180


@pytest.mark.parametrize("value", [0, -1])
def test_non_positive_lease_ttl_rejected(monkeypatch, value):
    with pytest.raises(ValueError, match="lease_ttl_sec must be > 0"):
        Settings(lease_ttl_sec=value)
    monkeypatch.setenv("RFORK_MOCK_LEASE_TTL_SEC", str(value))
    with pytest.raises(ValueError, match="lease_ttl_sec must be > 0"):
        Settings.from_env()


def test_non_integer_lease_ttl_rejected(monkeypatch):
    monkeypatch.setenv("RFORK_MOCK_LEASE_TTL_SEC", "invalid")
    with pytest.raises(ValueError):
        Settings.from_env()
    monkeypatch.delenv("RFORK_MOCK_LEASE_TTL_SEC")
    with pytest.raises(SystemExit):
        _build_arg_parser().parse_args(["--lease-ttl-sec", "1.5"])


@pytest.fixture
def planner():
    clock = SimpleNamespace(now=0.0)
    store = Store(heartbeat_ttl_sec=60, default_resource_points=1, scheduler=Scheduler(), time_fn=lambda: clock.now)
    seed_args = dict(seed_key="model-key", seed_ip="127.0.0.1", seed_port=1234, seed_rank=0)
    seed = store.add_seed(**seed_args)
    _, lease = store.get_seed(seed_key="model-key")
    return SimpleNamespace(clock=clock, store=store, seed_args=seed_args, seed=seed, lease=lease)


@pytest.mark.parametrize("ttl", [60, 120])
def test_lease_expiry_boundary_and_capacity_reclaimed_once(ttl):
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


def test_stale_seed_removes_lease_even_with_longer_lease_ttl():
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
    assert store.debug_snapshot()["lease_count"] == 0


@pytest.mark.parametrize("expired", [False, True])
def test_put_seed_http_status_and_duplicate_release(planner, expired):
    if expired:
        planner.clock.now = 59
        planner.store.add_seed(**planner.seed_args)
        planner.clock.now = 60
    headers = {
        "seed_ip": "127.0.0.1",
        "seed_port": "1234",
        "seed_rank": "0",
        "user_id": planner.lease.user_id,
    }
    request = Request({"type": "http", "headers": [(key.encode(), value.encode()) for key, value in headers.items()]})
    endpoint = next(route.endpoint for route in build_router(planner.store).routes if route.path == "/put_seed")
    response = endpoint(request)
    assert response.status_code == (404 if expired else 200)
    assert endpoint(request).status_code == 404
    assert planner.seed.resource_used == 0


def test_put_seed_missing_header_is_not_treated_as_expiry(planner):
    request = Request({"type": "http", "headers": []})
    endpoint = next(route.endpoint for route in build_router(planner.store).routes if route.path == "/put_seed")
    assert endpoint(request).status_code == 400
    assert planner.seed.resource_used == 1
