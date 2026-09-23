# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""Request/response contract tests between RForkPlannerClient and the example planner.

Other planner tests either stub ``requests`` (client side) or call ``Store``
directly (planner side), so a header rename on one side would not fail them. Here
the real client builds the requests and the real planner route handlers parse
them, which pins the header names, header values, status-code handling, and route
path plus verb for every route the client uses.

Scope: the route handlers are invoked directly rather than over a socket, because
``TestClient`` needs ``httpx``, which is not a test dependency here. Everything
the ASGI server itself would do -- real socket transport, middleware, content
negotiation, and its own 405/404 responses for unrouted requests -- is therefore
out of scope and must be covered by an e2e test.
"""

from types import SimpleNamespace

import pytest
import requests
from starlette.requests import Request

from examples.rfork.rfork_planner import Scheduler, Store, build_router

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
        # Key on (path, method) so calling a route with the wrong verb fails here,
        # the way a real server would answer 405.
        self.routes = {
            (route.path, verb): route.endpoint for route in build_router(store).routes for verb in route.methods
        }
        self.calls: list[tuple[str, str]] = []

    def _dispatch(self, method, url, headers=None, timeout=None, allow_redirects=None):
        assert url.startswith(self.planner_url), url
        path = url[len(self.planner_url) :]
        assert timeout is not None, "the client must always bound its planner requests"
        self.calls.append((method, path))
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


@pytest.fixture
def planner_http(runtime, monkeypatch):
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
    client = runtime.client.RForkPlannerClient(runtime.config, runtime.identity)
    client.bind_structural_digest("digest")
    return SimpleNamespace(
        clock=clock,
        store=store,
        transport=transport,
        client=client,
        removal_max_attempts=runtime.client.SEED_REMOVAL_MAX_ATTEMPTS,
    )


def _advertise(planner_http, port=1234):
    result = planner_http.client.report_seed_once(port, seed_ip="127.0.0.1")
    assert result.status.name == "ACCEPTED"


def test_full_seed_lifecycle_over_the_real_planner_routes(planner_http):
    client = planner_http.client

    # Advertise, then lease the seed back through the planner.
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
    assert planner_http.store.debug_snapshot()["seed_count"] == 0


def test_acquire_seed_returns_none_when_the_planner_has_no_seed(planner_http):
    assert planner_http.client.acquire_seed() is None
    assert planner_http.transport.calls == [("GET", "/get_seed")]


def test_planner_does_not_hand_the_same_seed_to_two_concurrent_receivers(planner_http):
    _advertise(planner_http)

    # Default capacity is one point, so the second lease request finds nothing.
    first = planner_http.client.acquire_seed()
    assert first is not None
    assert planner_http.client.acquire_seed() is None

    # Releasing the first lease returns the capacity to the pool.
    assert planner_http.client.release_seed_once(first).name == "RELEASED"
    assert planner_http.client.acquire_seed() is not None


def test_renew_and_release_are_rejected_after_the_lease_expires(planner_http):
    client = planner_http.client
    _advertise(planner_http)
    lease = client.acquire_seed()
    assert lease is not None

    planner_http.clock.now = LEASE_TTL_SEC + 1

    # An expired lease is reclaimed by the planner, so renewal fails and release is
    # reported as absent (404) rather than as a verified live release.
    assert not client.renew_seed_once(lease)
    assert client.release_seed_once(lease).name == "RELEASED"


def test_removing_a_seed_with_an_active_lease_is_reported_as_failure(planner_http):
    client = planner_http.client
    _advertise(planner_http)
    assert client.acquire_seed() is not None

    # /remove_seed answers 409 while the seed still has a live lease; the client
    # treats that as a failed withdrawal and keeps the advertisement for retry.
    assert not client.remove_seed()
    assert client.last_advertisement is not None
    removal_calls = [path for _, path in planner_http.transport.calls].count("/remove_seed")
    assert removal_calls == planner_http.removal_max_attempts
