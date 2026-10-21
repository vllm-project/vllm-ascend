# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""Compatibility coverage for a vLLM worker talking to a pre-15935 planner."""

from collections.abc import Mapping
from dataclasses import replace
from threading import Event
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import requests

_MISSING = object()


class _Response:
    def __init__(self, status_code: int, headers: dict[str, str] | None = None, text: str = "") -> None:
        self.status_code = status_code
        self.headers = headers or {}
        self.text = text


class _LegacyPlannerAdapter:
    """Small stateful adapter implementing the old add/get/put planner contract."""

    RequestException = requests.RequestException

    def __init__(
        self,
        *,
        lease_ttl_header: object = _MISSING,
        remove_supported: bool = False,
        put_statuses: list[int] | None = None,
        put_entered: Event | None = None,
        put_gate: Event | None = None,
        block_put_on_call: int | None = None,
    ) -> None:
        self.lease_ttl_header = lease_ttl_header
        self.remove_supported = remove_supported
        self.put_statuses = list(put_statuses or [])
        self.put_entered = put_entered
        self.put_gate = put_gate
        self.block_put_on_call = block_put_on_call
        self.put_attempt_count = 0
        self.calls: list[tuple[str, str]] = []
        self.request_headers: list[dict[str, str]] = []
        self.request_options: list[dict[str, object]] = []
        self.advertisements: dict[str, dict[str, object]] = {}
        self.active_leases: dict[str, dict[str, object]] = {}
        self._next_user_id = 0
        self._remove_conflict_seen = False
        self.clock = 0.0

    @property
    def active_count(self) -> int:
        return len(self.active_leases)

    def advance(self, seconds: float) -> None:
        self.clock += seconds

    def publish(self, seed_key: str, *, seed_ip: str = "127.0.0.1", seed_port: int = 1234, seed_rank: int = 0) -> None:
        self.advertisements[seed_key] = {
            "seed_ip": seed_ip,
            "seed_port": seed_port,
            "seed_rank": seed_rank,
            "user_id": None,
        }

    def force_lease(self, seed_key: str) -> dict[str, object]:
        advertisement = self.advertisements[seed_key]
        user_id = self._new_user_id()
        lease = {
            "seed_key": seed_key,
            "seed_ip": advertisement["seed_ip"],
            "seed_port": advertisement["seed_port"],
            "seed_rank": advertisement["seed_rank"],
            "user_id": user_id,
        }
        advertisement["user_id"] = user_id
        self.active_leases[user_id] = lease
        return lease

    def _new_user_id(self) -> str:
        self._next_user_id += 1
        return f"legacy-user-{self._next_user_id}"

    def _prepare_request(self, method: str, url: str, kwargs: dict[str, object]) -> tuple[str, dict[str, str]]:
        assert url.startswith("http://planner"), url
        path = url[len("http://planner") :]
        raw_headers = kwargs.get("headers")
        assert raw_headers is None or isinstance(raw_headers, Mapping)
        headers = dict(raw_headers or {})
        # Keep the same transport constraints as the real client contract.
        assert kwargs.get("timeout") is not None, "planner requests must have a timeout"
        assert kwargs.get("allow_redirects") is False, "planner requests must disable redirects"
        self.calls.append((method, path))
        self.request_headers.append(headers)
        self.request_options.append(
            {"timeout": kwargs.get("timeout"), "allow_redirects": kwargs.get("allow_redirects")}
        )
        return path, headers

    @staticmethod
    def _require(headers: dict[str, str], *names: str) -> None:
        missing = [name for name in names if name not in headers]
        assert not missing, f"missing old planner headers: {missing}"

    def get(self, url: str, **kwargs: object) -> _Response:
        path, headers = self._prepare_request("GET", url, kwargs)
        assert path == "/get_seed", path
        self._require(headers, "SEED_KEY")
        advertisement = self.advertisements.get(headers["SEED_KEY"])
        if advertisement is None or advertisement["user_id"] is not None:
            return _Response(404)

        user_id = self._new_user_id()
        advertisement["user_id"] = user_id
        lease = {
            "seed_key": headers["SEED_KEY"],
            "seed_ip": advertisement["seed_ip"],
            "seed_port": advertisement["seed_port"],
            "seed_rank": advertisement["seed_rank"],
            "user_id": user_id,
        }
        self.active_leases[user_id] = lease
        response_headers = {
            "SEED_IP": str(lease["seed_ip"]),
            "SEED_PORT": str(lease["seed_port"]),
            "USER_ID": str(user_id),
            "SEED_RANK": str(lease["seed_rank"]),
        }
        if self.lease_ttl_header is not _MISSING:
            response_headers["LEASE_TTL_SEC"] = str(self.lease_ttl_header)
        return _Response(200, response_headers)

    def post(self, url: str, **kwargs: object) -> _Response:
        path, headers = self._prepare_request("POST", url, kwargs)
        if path == "/add_seed":
            self._require(headers, "SEED_KEY", "SEED_IP", "SEED_PORT", "SEED_RANK", "SEED_REFCNT")
            key = headers["SEED_KEY"]
            current = self.advertisements.get(key)
            active_user = current["user_id"] if current is not None else None
            self.advertisements[key] = {
                "seed_ip": headers["SEED_IP"],
                "seed_port": int(headers["SEED_PORT"]),
                "seed_rank": int(headers["SEED_RANK"]),
                # Heartbeats update the advertisement but do not reset quota.
                "user_id": active_user,
            }
            return _Response(200)

        if path == "/put_seed":
            self._require(headers, "SEED_IP", "SEED_PORT", "USER_ID", "SEED_RANK")
            user_id = headers["USER_ID"]
            lease = self.active_leases.get(user_id)
            if lease is None:
                return _Response(404)
            assert str(lease["seed_ip"]) == headers["SEED_IP"]
            assert str(lease["seed_port"]) == headers["SEED_PORT"]
            assert str(lease["seed_rank"]) == headers["SEED_RANK"]
            self.put_attempt_count += 1
            if self.put_attempt_count == self.block_put_on_call:
                assert self.put_entered is not None
                assert self.put_gate is not None
                self.put_entered.set()
                assert self.put_gate.wait(5), "test did not release the blocked planner put"
            if self.put_statuses:
                status = self.put_statuses.pop(0)
                if status != 200:
                    return _Response(status)
            self.active_leases.pop(user_id)
            advertisement = self.advertisements.get(str(lease["seed_key"]))
            if advertisement is not None and advertisement["user_id"] == user_id:
                advertisement["user_id"] = None
            return _Response(200)

        if path == "/renew_seed_lease":
            # The legacy planner has no renewal endpoint.
            return _Response(404)

        if path == "/remove_seed":
            self._require(headers, "SEED_KEY", "SEED_IP", "SEED_PORT", "SEED_RANK")
            if not self.remove_supported:
                # A bare 404 is intentionally ambiguous: route support is unknown.
                return _Response(404)
            advertisement = self.advertisements.get(headers["SEED_KEY"])
            if advertisement is not None and advertisement["user_id"] is not None:
                self._remove_conflict_seen = True
                return _Response(409)
            if self._remove_conflict_seen:
                self.advertisements.pop(headers["SEED_KEY"], None)
                return _Response(404)
            self.advertisements.pop(headers["SEED_KEY"], None)
            return _Response(200)

        raise AssertionError(f"legacy planner received unknown route {path}")


def _make_client(runtime, monkeypatch, planner, *, identity=None, config=None):
    monkeypatch.setattr(runtime.client, "requests", planner)
    client = runtime.client.RForkPlannerClient(
        config if config is not None else runtime.config,
        identity if identity is not None else runtime.identity,
    )
    client.bind_structural_digest("digest")
    return client


def _advertise(client, planner, port: int = 1234) -> None:
    result = client.report_seed_once(port, seed_ip="127.0.0.1")
    assert result
    assert client.seed_key in planner.advertisements


def _make_legacy_session(runtime, monkeypatch, planner, *, identity=None):
    monkeypatch.setattr(runtime.client, "requests", planner)
    session = runtime.session.RForkSession(
        runtime.config,
        identity if identity is not None else runtime.identity,
    )
    backend = Mock()
    backend.register_memory_region.return_value = True
    backend.registered_structural_digest = "digest"
    backend.unregister_memory_region.return_value = True
    backend.finalize_transfer_engine.return_value = True
    backend.weight_formats = {}
    session.transfer_backend = backend
    assert session.register_destination(object(), True)
    planner.publish(session.planner.seed_key)
    return session, backend


def _wait_for_release(session) -> None:
    worker = session.lease_release_thread
    if worker is not None:
        worker.join(2)
    assert session.lease_release_thread is None or not session.lease_release_thread.is_alive()


@pytest.mark.parametrize("is_draft_model", [False, True])
def test_legacy_planner_supports_main_and_draft_leases_without_ttl(runtime, monkeypatch, is_draft_model):
    identity = replace(runtime.identity, is_draft_model=is_draft_model)
    planner = _LegacyPlannerAdapter()
    client = _make_client(runtime, monkeypatch, planner, identity=identity)

    _advertise(client, planner)
    lease = client.acquire_seed()
    assert lease is not None
    assert lease.lease_ttl_sec is None

    # A legacy heartbeat refreshes the advertisement but cannot reclaim the active quota.
    planner.advance(61)
    _advertise(client, planner)
    assert client.acquire_seed() is None

    assert client.release_seed_once(lease).name == "RELEASED"
    reusable_lease = client.acquire_seed()
    assert reusable_lease is not None
    assert client.release_seed_once(reusable_lease).name == "RELEASED"
    assert not any(path == "/renew_seed_lease" for _, path in planner.calls)
    assert all(option["timeout"] == runtime.config.request_timeout_sec for option in planner.request_options)
    assert all(option["allow_redirects"] is False for option in planner.request_options)


def test_legacy_session_acquire_does_not_start_lease_renewal(runtime, monkeypatch):
    planner = _LegacyPlannerAdapter()
    session, _ = _make_legacy_session(runtime, monkeypatch, planner)

    assert session.acquire_seed()
    assert session.seed_lease is not None
    assert session.seed_lease.lease_ttl_sec is None
    assert session.lease_renew_thread is None
    assert not any(path == "/renew_seed_lease" for _, path in planner.calls)

    session.prepare_for_fallback()
    _wait_for_release(session)
    assert planner.active_count == 0


@pytest.mark.parametrize("failure", ["metadata", "read"])
def test_legacy_planner_release_after_transfer_failure_restores_capacity(runtime, monkeypatch, failure):
    planner = _LegacyPlannerAdapter()
    session, backend = _make_legacy_session(runtime, monkeypatch, planner)
    assert session.acquire_seed()

    if failure == "metadata":
        monkeypatch.setattr(runtime.session, "fetch_seed_transfer_info", lambda *args: None)
    else:
        monkeypatch.setattr(
            runtime.session,
            "fetch_seed_transfer_info",
            lambda *args: SimpleNamespace(session_id="legacy-session"),
        )
        backend.read_weights_from_seed.return_value = False

    assert not session.transfer_from_seed(object(), True)
    cleanup = session.prepare_for_fallback()
    _wait_for_release(session)

    assert cleanup.memory_reset
    assert planner.active_count == 0
    assert "/put_seed" in [path for _, path in planner.calls]
    reusable_lease = session.planner.acquire_seed()
    assert reusable_lease is not None
    assert session.planner.release_seed_once(reusable_lease).name == "RELEASED"


@pytest.mark.parametrize("invalid_ttl", ["", 0, -1, "nan", "inf", "invalid"])
def test_identifiable_legacy_lease_with_invalid_ttl_is_returned_best_effort(runtime, monkeypatch, invalid_ttl):
    planner = _LegacyPlannerAdapter(lease_ttl_header=invalid_ttl)
    client = _make_client(runtime, monkeypatch, planner)
    _advertise(client, planner)

    assert client.acquire_seed() is None
    assert planner.active_count == 0
    assert "/put_seed" in [path for _, path in planner.calls]


@pytest.mark.parametrize("is_draft_model", [False, True])
def test_legacy_successful_transfer_releases_quota_in_background(runtime, monkeypatch, is_draft_model):
    identity = replace(runtime.identity, is_draft_model=is_draft_model)
    planner = _LegacyPlannerAdapter()
    session, backend = _make_legacy_session(runtime, monkeypatch, planner, identity=identity)
    assert session.acquire_seed()

    monkeypatch.setattr(
        runtime.session,
        "fetch_seed_transfer_info",
        lambda *args: SimpleNamespace(session_id="legacy-session"),
    )
    backend.read_weights_from_seed.return_value = True

    assert session.transfer_from_seed(object(), True)
    _wait_for_release(session)

    assert session.seed_lease is None
    assert planner.active_count == 0
    assert "/put_seed" in [path for _, path in planner.calls]
    reusable_lease = session.planner.acquire_seed()
    assert reusable_lease is not None
    assert session.planner.release_seed_once(reusable_lease).name == "RELEASED"


def test_session_keeps_rejected_lease_until_background_put_ack(runtime, monkeypatch):
    put_entered = Event()
    release_put = Event()
    planner = _LegacyPlannerAdapter(
        lease_ttl_header="invalid",
        put_statuses=[503, 200],
        put_entered=put_entered,
        put_gate=release_put,
        block_put_on_call=2,
    )
    session, backend = _make_legacy_session(runtime, monkeypatch, planner)

    assert session.acquire_seed() is False
    release_worker = session.lease_release_thread
    try:
        assert release_worker is not None
        assert put_entered.wait(2), "background lease release did not reach the planner"
        assert session.seed_lease is not None
        assert planner.active_count == 1
        assert backend.read_weights_from_seed.call_count == 0
        assert session.planner.take_rejected_lease() is None
    finally:
        release_put.set()
        if release_worker is not None:
            release_worker.join(2)
    assert not release_worker.is_alive()
    assert session.seed_lease is None
    assert planner.active_count == 0
    assert session.planner.take_rejected_lease() is None


def test_client_rejected_lease_blocks_followup_get_until_returned(runtime, monkeypatch):
    planner = _LegacyPlannerAdapter(
        lease_ttl_header="invalid",
        put_statuses=[503, 503, 200, 200],
    )
    client = _make_client(runtime, monkeypatch, planner)
    _advertise(client, planner)

    assert client.acquire_seed() is None
    assert planner.active_count == 1
    assert [path for _, path in planner.calls].count("/get_seed") == 1
    assert planner.put_attempt_count == 1

    # A failed best-effort put remains pending; the next acquisition must not
    # allocate a second lease before retrying that pending return.
    assert client.acquire_seed() is None
    assert [path for _, path in planner.calls].count("/get_seed") == 1
    assert planner.put_attempt_count == 2
    assert planner.active_count == 1

    planner.lease_ttl_header = _MISSING
    lease = client.acquire_seed()
    assert lease is not None
    assert lease.lease_ttl_sec is None
    assert [path for _, path in planner.calls].count("/get_seed") == 2
    assert planner.put_attempt_count == 3
    assert planner.active_count == 1
    assert client.take_rejected_lease() is None

    assert client.release_seed_once(lease).name == "RELEASED"
    assert planner.active_count == 0


def test_unknown_remove_route_does_not_confirm_seed_removal(runtime, monkeypatch):
    planner = _LegacyPlannerAdapter(remove_supported=False)
    client = _make_client(runtime, monkeypatch, planner)
    _advertise(client, planner)

    result = client.remove_seed()

    assert result is runtime.types.SeedRemovalResult.UNCONFIRMED
    assert client.last_advertisement is not None


def test_confirmed_remove_route_treats_post_release_404_as_absent(runtime, monkeypatch):
    planner = _LegacyPlannerAdapter(remove_supported=True)
    client = _make_client(runtime, monkeypatch, planner)
    _advertise(client, planner)
    raw_lease = planner.force_lease(client.seed_key)
    seed_port = raw_lease["seed_port"]
    seed_rank = raw_lease["seed_rank"]
    assert isinstance(seed_port, int)
    assert isinstance(seed_rank, int)
    lease = runtime.types.SeedLease(
        str(raw_lease["seed_ip"]),
        seed_port,
        str(raw_lease["user_id"]),
        seed_rank,
        client.seed_key,
        None,
    )

    assert client.remove_seed() is runtime.types.SeedRemovalResult.DRAINING
    assert client.last_advertisement is not None
    assert client.release_seed_once(lease).name == "RELEASED"

    result = client.remove_seed()
    assert result is runtime.types.SeedRemovalResult.ABSENT
    assert client.last_advertisement is None


@pytest.mark.parametrize("supported", [False, True])
def test_configured_seed_removal_capability(runtime, monkeypatch, supported):
    planner = _LegacyPlannerAdapter(remove_supported=False)
    config = type(runtime.config).from_extra_config({**_base_extra_config(), "rfork_seed_removal_supported": supported})
    client = _make_client(runtime, monkeypatch, planner, config=config)
    _advertise(client, planner)

    result = client.remove_seed()

    if supported:
        assert result is runtime.types.SeedRemovalResult.ABSENT
        assert client.last_advertisement is None
    else:
        assert result is runtime.types.SeedRemovalResult.UNSUPPORTED
        assert client.last_advertisement is not None
        assert "/remove_seed" not in [path for _, path in planner.calls]


def _base_extra_config():
    return {
        "model_url": "model",
        "model_deploy_strategy_name": "strategy",
        "rfork_scheduler_url": "http://planner",
    }


def test_seed_removal_capability_defaults_to_false_when_json_key_is_missing(runtime):
    parsed = type(runtime.config).from_extra_config(_base_extra_config())
    assert parsed.seed_removal_supported is False


def test_seed_removal_capability_accepts_explicit_json_null(runtime):
    raw_config = _base_extra_config()
    raw_config["rfork_seed_removal_supported"] = None
    parsed = type(runtime.config).from_extra_config(raw_config)
    assert parsed.seed_removal_supported is None


@pytest.mark.parametrize("value", [0, 1, "true", "false", [], {}])
def test_seed_removal_capability_rejects_non_boolean_configuration(runtime, value):
    raw_config = _base_extra_config()
    raw_config["rfork_seed_removal_supported"] = value
    with pytest.raises(ValueError, match="rfork_seed_removal_supported"):
        type(runtime.config).from_extra_config(raw_config)
