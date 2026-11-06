# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Deferred evidence must remain exact and precede output/state consumption."""

from concurrent.futures import ThreadPoolExecutor
from threading import Event
from types import SimpleNamespace as NS

import pytest
import torch

from .test_kv_guard_runtime import bind, metadata
from .test_kv_guard_runtime import runtime_fixture as runtime_fixture


def deferred_forward(runtime, mode="FULL", slot=147):
    bind(runtime, computed=19)
    runtime._defer_audit = True
    runtime._scheduled = {"r": (19, 1)}
    slots, table, lengths = torch.tensor([slot]), torch.tensor([[1]]), torch.tensor([20])
    item = metadata(slots, table, lengths)
    runner = NS(input_batch=NS(req_ids=["r"]))
    context = NS(cudagraph_runtime_mode=mode, attn_metadata=dict.fromkeys(runtime.layer_groups, item))

    def forward(count):
        if mode != "FULL":
            for layer in runtime.layer_groups:
                runtime.write(layer, slots)
                runtime.read(layer, table, lengths)

    runtime.wrap_forward(runner, forward, lambda: context)(1)
    return slots, table, lengths


@pytest.mark.parametrize("mode", ["FULL", "NONE"])
@pytest.mark.parametrize("slot", [147, 148])
def test_deferred_audit_freezes_final_values_without_publishing_validity(runtime, mode, slot):
    slots, table, lengths = deferred_forward(runtime, mode, slot)
    assert runtime.state.blocks[0, 1].valid_end == 19
    slots[0], table[0, 0], lengths[0] = 0, 2, 99
    runtime._finish_pending()
    mismatch = any(event.get("reason") == "write_target_mismatch" for event in runtime.events)
    assert mismatch == (slot != 147)
    if slot == 147:
        assert runtime.state.blocks[0, 1].valid_end == 20
        assert not runtime.events


@pytest.mark.parametrize("mode", ["FULL", "NONE"])
def test_invalid_snapshot_cannot_escape_through_async_output(runtime, mode):
    runtime.action = "raise"
    runtime.state.emit = runtime.report
    deferred_forward(runtime, mode, 148)
    consumed = []
    output = NS(get_output=lambda: consumed.append(True))
    assert runtime._guard_output(output) is output
    with pytest.raises(runtime.subject.KVGuardError, match="write_target_mismatch"):
        output.get_output()
    assert not consumed
    with pytest.raises(runtime.subject.KVGuardError, match="write_target_mismatch"):
        output.get_output()
    assert not consumed


def test_direct_output_finishes_audit_before_return(runtime):
    deferred_forward(runtime)
    marker = object()
    assert runtime._guard_output(marker) is marker
    assert runtime.state.blocks[0, 1].valid_end == 20


def test_completed_output_still_honors_a_later_sticky_failure(runtime):
    runtime.action = "raise"
    deferred_forward(runtime)
    consumed = []
    output = runtime._guard_output(NS(get_output=lambda: consumed.append(True)))
    runtime._finish_pending()
    runtime._failure = "later_step_failed"
    with pytest.raises(runtime.subject.KVGuardError, match="later_step_failed"):
        output.get_output()
    assert not consumed


def test_readonly_output_fallback_cannot_bypass_a_sticky_failure(runtime):
    class Output:
        __slots__ = ()

        def get_output(self):
            return "result"

    runtime.action = "raise"
    deferred_forward(runtime)
    assert runtime._guard_output(Output()).get_output() == "result"
    runtime._failure = "later_step_failed"
    with pytest.raises(runtime.subject.KVGuardError, match="later_step_failed"):
        runtime._guard_output(Output())


def test_next_binding_and_clear_follow_the_predecessor_audit(runtime):
    deferred_forward(runtime)
    pending = runtime._pending
    consumed = []
    output = runtime._guard_output(NS(get_output=lambda: consumed.append(True)))
    original_bind = runtime.state.bind
    calls = []

    def checked_bind(context):
        assert runtime.state.blocks[0, 1].valid_end == 20
        calls.append("bind")
        original_bind(context)

    runtime.state.bind = checked_bind
    packet = NS(
        _ascend_kv_guard_context={
            "pool_id": "pool",
            "requests": {"next": [[(2, 1)]]},
            "released_request_ids": ["r"],
            "scheduled_requests": {"next": (0, 1)},
        }
    )
    with runtime.execute_scope(packet):
        runtime.zero_blocks([2])
        assert not calls and "r" in runtime.state.requests
        runtime._finish_pending()
        assert calls == ["bind"] and "r" not in runtime.state.requests
        assert runtime.state.blocks[0, 2].valid_end == 0
    # A late reader reuses the completed decision, not the next step's buffers/state.
    runtime.state.blocks[0, 1].epoch = 9
    output.get_output()
    assert consumed == [True] and pending is not None
    assert not runtime.events


def test_idle_step_drains_pending_validation_and_release(runtime):
    deferred_forward(runtime)
    packet = NS(
        _ascend_kv_guard_context={
            "pool_id": "pool",
            "requests": {},
            "released_request_ids": ["r"],
            "scheduled_requests": {},
        }
    )
    runtime.wrap_execute(lambda output: None)(packet)
    assert runtime._pending is None and not runtime._deferred_state
    assert not runtime.state.requests
    assert runtime.state.blocks[0, 1].valid_end == 20


@pytest.mark.parametrize("failure", [False, True])
def test_output_reader_and_next_forward_finish_only_once(runtime, failure):
    started, release = Event(), Event()
    calls = []

    def complete():
        calls.append(True)
        started.set()
        assert release.wait(2)
        if failure:
            raise RuntimeError("audit failed")

    pending = runtime.subject._PendingAudit(complete)
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(pending.finish) for _ in range(2)]
        assert started.wait(2)
        release.set()
        for future in futures:
            if failure:
                with pytest.raises(RuntimeError, match="audit failed"):
                    future.result()
            else:
                future.result()
    assert calls == [True] and pending._complete is None


def test_snapshot_keeps_missing_layer_coverage(runtime):
    bind(runtime)
    runtime._defer_audit = True
    runtime._scheduled = {"r": (0, 1)}
    item = metadata(torch.tensor([128]), torch.tensor([[1]]), torch.tensor([1]))
    runner = NS(input_batch=NS(req_ids=["r"]))
    context = NS(cudagraph_runtime_mode="FULL", attn_metadata={"layer0": item})
    runtime.wrap_forward(runner, lambda count: None, lambda: context)(1)
    runtime._finish_pending()
    assert runtime.state.blocks[0, 1].valid_end == 0
    assert any(event.get("reason") == "metadata_write_mapping_missing:layer1" for event in runtime.events)


@pytest.mark.parametrize("mode", ["FULL", "NONE"])
@pytest.mark.parametrize("field,reason", [("table", "read_target_mismatch"), ("length", "read_length_mismatch")])
def test_frozen_read_evidence_cannot_be_repaired_by_the_next_preparation(runtime, mode, field, reason):
    bind(runtime, computed=19)
    runtime._defer_audit = True
    runtime._scheduled = {"r": (19, 1)}
    slots = torch.tensor([147])
    table = torch.tensor([[2 if field == "table" else 1]])
    lengths = torch.tensor([21 if field == "length" else 20])
    item = metadata(slots, table, lengths)
    runner = NS(input_batch=NS(req_ids=["r"]))
    context = NS(cudagraph_runtime_mode=mode, attn_metadata=dict.fromkeys(runtime.layer_groups, item))

    def forward(count):
        if mode != "FULL":
            for layer in runtime.layer_groups:
                runtime.write(layer, slots)
                runtime.read(layer, table, lengths)

    runtime.wrap_forward(runner, forward, lambda: context)(1)
    table[0, 0], lengths[0] = 1, 20
    runtime._finish_pending()
    assert any(event.get("reason") == reason for event in runtime.events)
