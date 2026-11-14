# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Public block deltas, copied bindings and logical request removal."""

import pytest

from .test_kv_trace_runtime import message, reasons
from .test_kv_trace_runtime import runtime_fixture as runtime_fixture


def test_new_cached_and_resumed_messages(runtime):
    runtime.state.bind(message())
    runtime.state.bind(message(bindings={}, counts={"r": 1}, cached={"r": ([[2]], 128)}))
    assert runtime.state.requests["r"] == ((1, 2),)
    runtime.state.bind(message(bindings={}, counts={"r": 1}, cached={"r": ([[3]], 0)}, resumed={"r"}))
    assert runtime.state.requests["r"] == ((3,),)
    assert not reasons(runtime)


def test_unscheduled_native_binding_is_still_recorded(runtime):
    output = message(counts={})
    output.scheduled_new_reqs[0].replay_start = 16
    assert runtime.state.bind(output) == {}
    event = runtime.events[-1]
    assert event["requests"] == {"r": ((1,),)}
    assert event["replay_starts"] == {"r": 16}
    assert not reasons(runtime)


@pytest.mark.parametrize("kind", ["finished", "preempted"])
def test_finished_or_preempted_removes_only_local_request_binding(runtime, kind):
    runtime.state.bind(message(bindings={"A": [[1]], "B": [[1]]}))
    runtime.state.bind(message(bindings={}, counts={}, **{kind: ["A"]}))
    assert runtime.state.requests == {"B": ((1,),)}
    assert not hasattr(runtime.state, "blocks")


def test_caller_mutation_cannot_change_a_saved_binding(runtime):
    output = message()
    runtime.state.bind(output)
    output.scheduled_new_reqs[0].block_ids[0][0] = 7
    assert runtime.state.requests["r"] == ((1,),)


def test_cached_without_new_blocks_preserves_binding(runtime):
    runtime.state.bind(message())
    assert runtime.state.bind(message(bindings={}, counts={"r": 1}, cached={"r": (None, 1)})) == {"r": (1, 1)}
    assert runtime.state.requests["r"] == ((1,),)


def test_malformed_public_delta_is_a_violation(runtime):
    output = message()
    output.scheduled_cached_reqs.req_ids = ["other"]
    runtime.state.bind(output)
    assert "cached_request_array_length_mismatch" in reasons(runtime)


def test_missing_cached_binding_and_wrong_group_counts_are_violations(runtime):
    runtime.state.bind(message(bindings={}, counts={"r": 1}, cached={"r": ([[2]], 0)}))
    assert "scheduler_binding_missing" in reasons(runtime)
    runtime.state.bind(message())
    runtime.state.bind(message(bindings={}, counts={"r": 1}, cached={"r": ([[2], [3]], 1)}))
    assert "scheduler_group_count_mismatch" in reasons(runtime)


def test_invalid_scheduled_total_is_a_violation(runtime):
    output = message()
    output.total_num_scheduled_tokens = 2
    runtime.state.bind(output)
    assert "scheduled_token_total_mismatch" in reasons(runtime)
