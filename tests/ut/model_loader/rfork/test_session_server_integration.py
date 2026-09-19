# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import importlib.util
import socket
import sys
import threading
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import requests


@pytest.fixture
def live_session(runtime, monkeypatch):
    """Real session, FastAPI/uvicorn listener and HTTP client; native NPU and planner injected."""
    path = Path(__file__).resolve().parents[4] / "vllm_ascend/model_loader/rfork/seed_server.py"
    name = "rfork_live_seed_server_test"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.setattr(runtime.session, "start_rfork_server", module.start_rfork_server)
    monkeypatch.setattr(runtime.session, "RForkSeedServerStartupError", module.RForkSeedServerStartupError)
    cfg = replace(
        runtime.config,
        seed_bind_host="127.0.0.1",
        seed_advertise_host="127.0.0.1",
        seed_timeout_sec=3.0,
        request_timeout_sec=1.0,
    )
    session = runtime.session.RForkSession(cfg, runtime.identity)
    session.planner = Mock(seed_key="live-key")
    session.planner.report_seed_once.return_value = True
    session.planner.remove_seed.return_value = True
    session.transfer_backend.transfer_session_id = "native-session"
    session.transfer_backend.weight_manifest = {"weight": [1234, 4, 4, [4], "float32"]}
    session.transfer_backend.weight_formats = {"weight": 2}
    handles = []
    original = module.start_rfork_server

    def start(*args, **kwargs):
        handle = original(*args, **kwargs)
        handles.append(handle)
        return handle

    monkeypatch.setattr(runtime.session, "start_rfork_server", start)
    client = sys.modules["vllm_ascend.model_loader.rfork.seed_client"]
    yield SimpleNamespace(session=session, handles=handles, client=client, runtime=runtime)
    session.planner.remove_seed.side_effect = None
    session.planner.remove_seed.return_value = True
    session.shutdown()
    for handle in handles:
        assert handle.stop(), "live seed server leaked a thread"


def test_session_serves_real_metadata_and_closes_listener(live_session):
    r = live_session
    assert r.session.start_seed_service(object(), True) is r.runtime.types.RForkSeedServiceStartResult.STARTED
    handle = r.session.seed_server
    assert handle.is_alive
    url = f"http://127.0.0.1:{handle.port}"
    info = r.client.fetch_seed_transfer_info(url, "live-key", 1.0)
    assert info.session_id == "native-session"
    assert info.weights == r.session.transfer_backend.weight_manifest
    assert info.formats == r.session.transfer_backend.weight_formats
    assert requests.get(f"{url}/health_check_with_key", params={"seed_key": "wrong"}, timeout=1).status_code == 400
    r.session.planner.report_seed_once.assert_called_once_with(handle.port, seed_ip="127.0.0.1")
    assert r.session.shutdown()
    assert not handle.is_alive
    assert handle.sock.fileno() == -1
    with pytest.raises(OSError):
        socket.create_connection(("127.0.0.1", handle.port), timeout=0.2)
    r.session.transfer_backend.finalize_transfer_engine.assert_called_once()
    assert r.session.shutdown()
    r.session.transfer_backend.finalize_transfer_engine.assert_called_once()


@pytest.mark.parametrize(
    ("global_rank", "is_draft_model", "expected_port"),
    [
        (0, False, 20000),
        (0, True, 20001),
        (3, False, 20006),
        (3, True, 20007),
    ],
)
def test_seed_server_port_reserves_main_and_draft_slots(runtime, global_rank, is_draft_model, expected_port):
    config = replace(runtime.config, seed_port_base=20000)
    identity = replace(runtime.identity, global_rank=global_rank, is_draft_model=is_draft_model)

    assert runtime.session._resolve_seed_server_port(config, identity) == expected_port


def test_seed_server_port_zero_keeps_dynamic_allocation(runtime):
    identity = replace(runtime.identity, global_rank=3, is_draft_model=True)

    assert runtime.session._resolve_seed_server_port(runtime.config, identity) == 0


def test_configured_seed_port_falls_back_when_already_in_use(live_session):
    r = live_session
    occupied = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    occupied.bind(("127.0.0.1", 0))
    occupied.listen(1)
    occupied_port = occupied.getsockname()[1]
    r.session.config = replace(r.session.config, seed_port_base=occupied_port)

    try:
        assert r.session.start_seed_service(object(), True) is r.runtime.types.RForkSeedServiceStartResult.STARTED
        assert r.session.seed_server.port != occupied_port
    finally:
        occupied.close()


def test_rejected_advertisement_cleans_real_server_before_unregister(live_session):
    r = live_session
    r.session.planner.report_seed_once.return_value = r.runtime.types.SeedReportResult(
        r.runtime.types.SeedReportStatus.REJECTED, "status=400"
    )

    def unregister():
        assert r.handles and not r.handles[0].is_alive
        assert r.handles[0].sock.fileno() == -1
        return True

    r.session.transfer_backend.unregister_memory_region.side_effect = unregister
    assert r.session.start_seed_service(object(), True) is r.runtime.types.RForkSeedServiceStartResult.FAILED
    assert r.session.seed_server is None
    r.session.transfer_backend.unregister_memory_region.assert_called_once()


def test_transient_initial_advertisement_recovers_without_restarting_instance(live_session):
    r = live_session
    r.session.config = replace(r.session.config, heartbeat_interval_sec=0.02)
    recovered = threading.Event()
    retryable = r.runtime.types.SeedReportResult(r.runtime.types.SeedReportStatus.RETRYABLE, "ConnectTimeout")
    accepted = r.runtime.types.SeedReportResult(r.runtime.types.SeedReportStatus.ACCEPTED)

    def report(*_args, **_kwargs):
        if r.session.planner.report_seed_once.call_count == 1:
            return retryable
        recovered.set()
        return accepted

    r.session.planner.report_seed_once.side_effect = report
    assert r.session.start_seed_service(object(), True) is r.runtime.types.RForkSeedServiceStartResult.STARTED
    handle = r.session.seed_server
    assert handle.is_alive
    assert recovered.wait(2)
    assert r.session.state is r.runtime.types.RForkLifecycleState.SERVING
    assert r.session.shutdown()
    assert not handle.is_alive


@pytest.mark.parametrize("removal_fails", [False, True])
def test_lost_publish_response_cleans_or_retains_live_resources(live_session, monkeypatch, removal_fails):
    r = live_session
    planner = r.runtime.client.RForkPlannerClient(r.session.config, r.runtime.identity)
    # Keep the fixture's Mock for teardown, but route planner calls through the real client.
    r.session.planner.report_seed_once.side_effect = planner.report_seed_once
    r.session.planner.remove_seed.side_effect = planner.remove_seed
    remote_seeds = set()

    def post(url, *, headers, **kwargs):
        identity = (headers["SEED_IP"], headers["SEED_PORT"], headers["SEED_RANK"])
        if url.endswith("/add_seed"):
            remote_seeds.add(identity)
            raise requests.Timeout("response lost after commit")
        assert url.endswith("/remove_seed")
        assert r.handles[0].is_alive
        r.session.transfer_backend.unregister_memory_region.assert_not_called()
        if removal_fails:
            return SimpleNamespace(status_code=503)
        remote_seeds.discard(identity)
        return SimpleNamespace(status_code=200)

    monkeypatch.setattr(requests, "post", post)
    assert r.session.start_seed_service(object(), True) is r.runtime.types.RForkSeedServiceStartResult.STARTED
    assert planner.last_advertisement is not None
    assert remote_seeds and r.session.seed_server.is_alive
    r.session.transfer_backend.unregister_memory_region.assert_not_called()
    if removal_fails:
        assert not r.session.prepare_for_fallback().can_schedule_seed
        assert r.session.seed_server.is_alive
        # A later acknowledged removal releases the resources in order.
        removal_fails = False
    assert r.session.prepare_for_fallback().can_schedule_seed
    assert not remote_seeds
    assert planner.last_advertisement is None
    assert r.session.seed_server is None
    assert not r.handles[0].is_alive
    r.session.transfer_backend.unregister_memory_region.assert_called_once()


@pytest.mark.parametrize("phase", ["health", "report"])
def test_start_network_work_does_not_hold_state_lock(live_session, monkeypatch, phase):
    r = live_session
    entered, resume = threading.Event(), threading.Event()
    if phase == "health":
        original = r.runtime.session.start_rfork_server

        def start(*args, **kwargs):
            entered.set()
            assert resume.wait(4)
            return original(*args, **kwargs)

        monkeypatch.setattr(r.runtime.session, "start_rfork_server", start)
    else:

        def report(*args, **kwargs):
            entered.set()
            assert resume.wait(4)
            return True

        r.session.planner.report_seed_once.side_effect = report
    results = []
    worker = threading.Thread(target=lambda: results.append(r.session.start_seed_service(object(), True)), daemon=True)
    try:
        worker.start()
        assert entered.wait(3)
        assert r.session._lock.acquire(timeout=0.5), f"{phase} I/O still holds session state lock"
        try:
            assert r.session.state is r.runtime.types.RForkLifecycleState.CLEANUP_REQUIRED
            assert not r.session.register_destination(object(), True)
            assert not r.session.acquire_seed()
            # Cancellation during startup must suppress or roll back publication.
            r.session.lease_release_stop_event.set()
        finally:
            r.session._lock.release()
    finally:
        resume.set()
        worker.join(5)
    assert not worker.is_alive()
    assert results == [r.runtime.types.RForkSeedServiceStartResult.FAILED]
    assert r.session.seed_server is None
    assert r.handles and all(not handle.is_alive for handle in r.handles)
    if phase == "health":
        r.session.planner.report_seed_once.assert_not_called()
