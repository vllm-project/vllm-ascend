# SPDX-License-Identifier: Apache-2.0
"""Behavior guards for the idle-KV-drain engine-step patch.

``patch_idle_kv_drain.py`` wraps ``EngineCoreProc._process_engine_step``:
after the original step, an idle step (no model executed) that still has
scheduler work and a KV connector drains the connector's finished-transfer
bookkeeping through ``scheduler._update_from_kv_xfer_finished``, so delayed
``finished_sending`` entries reach ``_free_blocks`` without waiting for a
new request to be scheduled. What is guarded here (everything reachable
from CPU UT):

* the drain fires exactly on idle steps that still have scheduler work and
  a connector, passing an empty ``KVConnectorOutput`` as the mutable carrier
  that ``update_connector_output`` fills in place;
* a step that executed the model never drains (the worker-output path owns
  that case);
* a fully idle engine (``has_requests() == False``) performs zero drain
  calls;
* a scheduler without a connector (or with ``connector=None``) never drains;
* a failing drain is logged and swallowed so the next step retries;
* ``_apply_patch`` is idempotent.
"""

import logging

import pytest
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorOutput

import vllm_ascend.patch.platform.patch_idle_kv_drain as idle_drain_patch


class FakeScheduler:
    def __init__(self, has_requests=True, connector="MISSING", drain_error=None):
        self.calls = []
        self.has_requests = lambda: has_requests
        if connector != "MISSING":
            self.connector = connector
        if drain_error is None:
            self._update_from_kv_xfer_finished = lambda out: self.calls.append(out)
        else:

            def _boom(out):
                raise drain_error

            self._update_from_kv_xfer_finished = _boom


class FakeEngineCore:
    def __init__(self, scheduler):
        self.scheduler = scheduler


@pytest.fixture
def wrapped_step(monkeypatch: pytest.MonkeyPatch):
    """Detach the wrapper from the real EngineCoreProc._process_engine_step."""
    monkeypatch.setattr(idle_drain_patch, "_OriginalProcessEngineStep", lambda self: False)
    return idle_drain_patch._process_engine_step_with_idle_kv_drain


def test_idle_step_with_scheduler_work_drains_connector(wrapped_step):
    scheduler = FakeScheduler(connector=object())
    engine = FakeEngineCore(scheduler)

    assert wrapped_step(engine) is False
    assert len(scheduler.calls) == 1
    assert isinstance(scheduler.calls[0], KVConnectorOutput)
    assert not scheduler.calls[0].finished_sending  # empty carrier, filled in place


def test_executed_step_does_not_drain(wrapped_step, monkeypatch):
    monkeypatch.setattr(idle_drain_patch, "_OriginalProcessEngineStep", lambda self: True)
    scheduler = FakeScheduler(connector=object())
    engine = FakeEngineCore(scheduler)

    assert wrapped_step(engine) is True
    assert not scheduler.calls


def test_fully_idle_engine_does_not_drain(wrapped_step):
    scheduler = FakeScheduler(has_requests=False, connector=object())
    engine = FakeEngineCore(scheduler)

    assert wrapped_step(engine) is False
    assert not scheduler.calls


def test_scheduler_without_connector_does_not_drain(wrapped_step):
    scheduler = FakeScheduler()  # connector attribute absent
    engine = FakeEngineCore(scheduler)
    assert wrapped_step(engine) is False
    assert not scheduler.calls

    scheduler = FakeScheduler(connector=None)
    engine = FakeEngineCore(scheduler)
    assert wrapped_step(engine) is False
    assert not scheduler.calls


def test_failing_drain_is_logged_and_swallowed(wrapped_step, caplog: pytest.LogCaptureFixture):
    scheduler = FakeScheduler(connector=object(), drain_error=RuntimeError("drain boom"))
    engine = FakeEngineCore(scheduler)

    with caplog.at_level(logging.ERROR, logger="vllm"):
        assert wrapped_step(engine) is False  # no raise; the next step retries
    assert any("Idle KV transfer drain failed" in r.message for r in caplog.records)


def test_apply_patch_is_idempotent():
    before = idle_drain_patch.EngineCoreProc._process_engine_step
    idle_drain_patch._apply_patch()
    assert idle_drain_patch.EngineCoreProc._process_engine_step is before
    assert getattr(before, "_vllm_ascend_idle_kv_drain", False)
