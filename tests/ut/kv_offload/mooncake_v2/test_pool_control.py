# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import threading
import time
from unittest.mock import MagicMock

import pytest
import zmq

from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.pool_control import (
    PoolControlServer,
    PoolManifestRegistry,
    pool_control_request,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.pool_metadata import PoolWorkerReport


def control(op="manifest", **kwargs):
    return {"op": op, "incarnation": "engine", "transfer_id": "transfer", "attempt_id": "attempt"} | kwargs


def test_manifest_channel_uses_real_json_sockets_and_idempotent_release():
    release = MagicMock()
    registry = PoolManifestRegistry("engine", release, retention_seconds=60)
    report = PoolWorkerReport("transfer", "attempt", "worker", True)
    server = PoolControlServer("127.0.0.1", registry.handle)
    try:
        with pytest.raises(RuntimeError, match="not committed"):
            pool_control_request(server.endpoint, control())
        registry.publish(report)
        wire = pool_control_request(server.endpoint, control())
        assert PoolWorkerReport.from_wire(wire) == report
        with pytest.raises(RuntimeError, match="incarnation"):
            pool_control_request(server.endpoint, control(incarnation="old-engine"))
        assert pool_control_request(server.endpoint, control("release")) == {"released": True}
        assert pool_control_request(server.endpoint, control("release")) == {"released": True}
        release.assert_called_once_with(report)
        with pytest.raises(RuntimeError, match="expired"):
            pool_control_request(server.endpoint, control())
    finally:
        server.close()
        registry.close()


def test_expiry_does_not_depend_on_engine_steps_and_republication_cannot_extend_it():
    now = [0.0]
    release = MagicMock()
    registry = PoolManifestRegistry("engine", release, retention_seconds=3, clock=lambda: now[0])
    report = PoolWorkerReport("transfer", "attempt", "worker", True)
    registry.publish(report)
    now[0] = 2
    registry.publish(report)
    now[0] = 3
    registry.expire()
    release.assert_called_once_with(report)
    with pytest.raises(ValueError, match="expired"):
        registry.handle(control())


def test_failed_release_keeps_ownership_for_retry():
    release = MagicMock(side_effect=[RuntimeError("native failure"), None])
    registry = PoolManifestRegistry("engine", release, retention_seconds=60)
    report = PoolWorkerReport("transfer", "attempt", "worker", True)
    registry.publish(report)
    with pytest.raises(RuntimeError, match="native failure"):
        registry.handle(control("release"))
    assert PoolWorkerReport.from_wire(registry.handle(control())) == report
    registry.handle(control("release"))
    assert release.call_count == 2


def test_acquisition_latency_consumes_manifest_retention():
    now = [2.0]
    release = MagicMock()
    registry = PoolManifestRegistry("engine", release, retention_seconds=3, clock=lambda: now[0])
    report = PoolWorkerReport("transfer", "attempt", "worker", True)
    registry.publish(report, retention_started_at=0.0)
    assert PoolWorkerReport.from_wire(registry.handle(control())) == report
    now[0] = 3.0
    with pytest.raises(ValueError, match="retention expired"):
        registry.handle(control())
    release.assert_called_once_with(report)


def test_expired_acquisition_cannot_publish_success():
    registry = PoolManifestRegistry("engine", MagicMock(), retention_seconds=3, clock=lambda: 3.0)
    report = PoolWorkerReport("transfer", "attempt", "worker", True)
    with pytest.raises(TimeoutError, match="before manifest publication"):
        registry.publish(report, retention_started_at=0.0)
    assert not registry._reports


def test_failed_worker_report_preserves_error_and_retries_partial_native_cleanup():
    release = MagicMock()
    registry = PoolManifestRegistry("engine", release, retention_seconds=60)
    report = PoolWorkerReport("transfer", "attempt", "worker", False, error="copy failed")
    registry.publish(report)
    assert PoolWorkerReport.from_wire(registry.handle(control())) == report
    registry.close()
    release.assert_called_once_with(report)


def test_expiry_failure_does_not_prevent_other_attempts_from_being_reclaimed():
    now = [0.0]
    released = []

    def release(report):
        if report.attempt_id == "failed":
            raise RuntimeError("native cleanup failed")
        released.append(report.attempt_id)

    registry = PoolManifestRegistry("engine", release, retention_seconds=1, clock=lambda: now[0])
    registry.publish(PoolWorkerReport("transfer", "failed", "worker", True))
    registry.publish(PoolWorkerReport("transfer", "good", "worker", True))
    now[0] = 2
    registry.expire()
    assert released == ["good"]
    assert ("transfer", "failed") in registry._reports
    assert ("transfer", "good") not in registry._reports


def test_bad_json_and_maintenance_failure_do_not_kill_control_service():
    ticks = []

    def maintenance():
        ticks.append(1)
        raise RuntimeError("transient maintenance failure")

    def handler(request):
        time.sleep(0.002)
        return {"pong": True}

    server = PoolControlServer("127.0.0.1", handler, maintenance=maintenance, maintenance_interval_s=0.001)
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.setsockopt(zmq.LINGER, 0)
    socket.setsockopt(zmq.RCVTIMEO, 2000)
    try:
        socket.connect(server.endpoint)
        socket.send(b"invalid json")
        assert socket.recv_json()["ok"] is False
        for _ in range(5):
            assert pool_control_request(server.endpoint, {}) == {"pong": True}
        assert ticks  # Maintenance runs even when valid requests keep arriving.
        server.check_health()
    finally:
        socket.close()
        context.term()
        server.close()


def test_control_close_waits_for_native_callback_to_finish():
    entered, release, closed = threading.Event(), threading.Event(), threading.Event()

    def handler(request):
        entered.set()
        assert release.wait(timeout=5)
        return {"released": True}

    server = PoolControlServer("127.0.0.1", handler)
    replies = []
    requester = threading.Thread(target=lambda: replies.append(pool_control_request(server.endpoint, {})))
    closer = threading.Thread(target=lambda: (server.close(), closed.set()))
    try:
        requester.start()
        assert entered.wait(timeout=2)
        closer.start()
        assert not closed.wait(timeout=0.05)
        release.set()
        requester.join(timeout=2)
        closer.join(timeout=2)
        assert closed.is_set()
        assert replies == [{"released": True}]
    finally:
        release.set()
        server.close()


def test_registry_close_attempts_all_cleanup_and_prevents_late_publication():
    release = MagicMock(side_effect=[RuntimeError("native failure"), None, None])
    registry = PoolManifestRegistry("engine", release, retention_seconds=10)
    first = PoolWorkerReport("transfer", "first", "worker", True)
    second = PoolWorkerReport("transfer", "second", "worker", True)
    registry.publish(first)
    registry.publish(second)
    with pytest.raises(RuntimeError, match="cleanup failed"):
        registry.close()
    assert release.call_count == 2
    assert len(registry._reports) == 1
    with pytest.raises(RuntimeError, match="closed"):
        registry.publish(second)
    registry.close()
    assert not registry._reports
