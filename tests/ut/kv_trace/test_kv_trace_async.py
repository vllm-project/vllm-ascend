# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Frozen plans retain step-local bindings and audits are consumed once."""

from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace as NS

import pytest

from .test_kv_trace_runtime import audit, message, plan, reasons
from .test_kv_trace_runtime import runtime_fixture as runtime_fixture
from .test_kv_trace_v2 import adapter_fixture as adapter_fixture
from .test_kv_trace_v2 import prepare


def test_snapshot_keeps_tensor_values_and_bindings(runtime):
    first = plan(runtime)
    frozen, ready = runtime.plan_subject.PlanSnapshotter().freeze(first)
    first.metadata[0].slot_mapping[0] = 256
    runtime.state.bind(message(bindings={"r": [[2]]}))
    assert ready is None
    audit(runtime, frozen)
    assert not reasons(runtime)


def test_pending_audit_is_completed_once_across_readers(runtime):
    calls = []
    pending = runtime.subject._PendingAudit(lambda: calls.append("audit"))
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda _: pending.finish(), range(8)))
    assert calls == ["audit"]


def test_pending_failure_is_preserved(runtime):
    error = ValueError("copy failed")

    def complete():
        raise error

    pending = runtime.subject._PendingAudit(complete)
    for _ in range(2):
        with pytest.raises(ValueError) as caught:
            pending.finish()
        assert caught.value is error


def test_immutable_output_drains_pending_without_calling_native_getter(runtime):
    calls = []

    class Output:
        __slots__ = ()

        def get_output(self):
            calls.append("getter")
            return "native"

    runtime._pending = runtime.subject._PendingAudit(lambda: calls.append("audit"))
    result = runtime._guard_output(Output())
    assert calls == ["audit"]
    assert result.get_output() == "native"
    assert calls == ["audit", "getter"]


def test_new_binding_does_not_force_old_audit_at_execute_input(adapter):
    runtime = adapter.runtime
    runtime._defer_audit = True
    output = NS(get_output=lambda: "native output")
    result = adapter.wrap_execute(lambda packet: prepare(adapter, ["r"], [128], [[1]], [1]) and output)(message())
    assert runtime._pending is not None
    assert not [event for event in runtime.events if event["event"] == "kv_trace.write"]
    with runtime.execute_scope(message(bindings={"r": [[2]]})):
        assert runtime.state.requests["r"] == ((2,),)
        assert not [event for event in runtime.events if event["event"] == "kv_trace.write"]
    assert result.get_output() == "native output"
    assert not reasons(runtime)


def test_raise_blocks_result_after_native_getter_without_early_get(adapter):
    runtime = adapter.runtime
    runtime.action = "raise"
    runtime._defer_audit = True
    calls = []

    def get_output():
        calls.append("native getter")
        return "result"

    output = NS(get_output=get_output)
    result = adapter.wrap_execute(lambda packet: prepare(adapter, ["r"], [256], [[1]], [1]) and output)(message())
    assert not calls
    with pytest.raises(runtime.subject.KVTraceError, match="write_target_mismatch"):
        result.get_output()
    assert calls == ["native getter"]
