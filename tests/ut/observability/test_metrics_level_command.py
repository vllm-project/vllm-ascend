# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Curl command that sets the MS metric level on engine and workers."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from vllm_ascend.observability.ms_metrics.api import attach_router
from vllm_ascend.observability.ms_metrics.control import (
    apply_engine_metrics_level,
    apply_metrics_level_command,
)
from vllm_ascend.observability.ms_metrics.level import METRIC_LEVEL_DETAIL, get_metric_level, set_metric_level


class _Core:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple]] = []

    async def call_utility_all_async(self, method: str, *args):
        self.calls.append((method, args))
        return ["detail", "detail"]


def test_engine_method_sets_local_level_and_rpcs_workers():
    class _Engine:
        def __init__(self) -> None:
            self.rpc: tuple | None = None

        def collective_rpc(self, method, args=()):
            self.rpc = (method, args)
            return ["detail"]

    engine = _Engine()
    try:
        assert apply_engine_metrics_level(engine, "detail") == "detail"
        assert get_metric_level() == METRIC_LEVEL_DETAIL
        assert engine.rpc == ("set_metrics_level", ("detail",))
    finally:
        set_metric_level("basic")


def test_command_publishes_and_fans_out_to_every_engine(monkeypatch):
    core = _Core()
    engine_client = SimpleNamespace(engine_core=core)

    def _publish(level):
        assert level == "2"
        return "detail"

    monkeypatch.setattr(
        "vllm_ascend.observability.ms_metrics.control.publish_metrics_level",
        _publish,
    )
    try:
        result = asyncio.run(apply_metrics_level_command(engine_client, "2"))
    finally:
        set_metric_level("basic")
    assert result == "detail"
    assert core.calls == [("set_metrics_level", ("detail",))]


def test_command_rejects_unknown_level_before_rpc(monkeypatch):
    core = _Core()
    engine_client = SimpleNamespace(engine_core=core)

    def _publish(level):
        from vllm_ascend.observability.ms_metrics.level import canonical_metric_level

        return canonical_metric_level(level)

    monkeypatch.setattr(
        "vllm_ascend.observability.ms_metrics.control.publish_metrics_level",
        _publish,
    )
    with pytest.raises(ValueError, match="invalid metric level"):
        asyncio.run(apply_metrics_level_command(engine_client, "nope"))
    assert core.calls == []


def test_http_rejects_unknown_level():
    app = FastAPI()
    attach_router(app)
    response = TestClient(app).post("/set_metrics_level", params={"level": "nope"})
    assert response.status_code == 400
    assert "invalid metric level" in response.json()["detail"]


def test_http_sets_level(monkeypatch):
    app = FastAPI()
    attach_router(app)
    core = _Core()
    app.state.engine_client = SimpleNamespace(engine_core=core)
    monkeypatch.setattr(
        "vllm_ascend.observability.ms_metrics.control.publish_metrics_level",
        lambda level: level,
    )
    response = TestClient(app).post("/set_metrics_level", params={"level": "detail"})
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "level": "detail"}
    assert core.calls == [("set_metrics_level", ("detail",))]
