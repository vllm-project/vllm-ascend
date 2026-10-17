# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MRV2 batch ordering, graph updates, dummy writes and lifecycle integration."""

import sys
from types import SimpleNamespace as NS
from typing import Any

import pytest
import torch

from .test_kv_guard_runtime import bind, load, metadata
from .test_kv_guard_runtime import runtime_fixture as runtime_fixture


@pytest.fixture
def adapter(runtime, monkeypatch):
    monkeypatch.setitem(sys.modules, "vllm_ascend.debug.kv_guard_runtime", runtime.subject)
    subject = load("guard_v2_subject", "vllm_ascend/debug/kv_guard_v2.py")
    impl = type("AscendAttentionBackendImpl", (), {})()
    runtime.config = NS(
        parallel_config=NS(),
        model_config=NS(),
        speculative_config=None,
        kv_transfer_config=None,
        compilation_config=NS(static_forward_context={"layer0": NS(impl=impl)}),
        _ascend_kv_guard=runtime,
    )
    runner: Any = NS(
        vllm_config=runtime.config,
        block_tables=NS(block_sizes=[128], kernel_block_sizes=[128]),
        kv_cache_config=NS(
            kv_cache_groups=[NS(layer_names=["layer0"], kv_cache_spec=NS(block_size=128, head_size=64))]
        ),
        model_state=NS(prepare_attn=lambda batch, mode: runner.next_metadata),
        next_metadata={},
    )
    runtime.layer_groups.clear()
    result = subject.KVGuardV2(runner, runtime)
    result.configure()
    result.subject = subject
    return result


def message(requests=None, scheduled=None, released=()):
    scheduled = {"r": (0, 1)} if scheduled is None else scheduled
    return NS(
        total_num_scheduled_tokens=sum(count for _, count in scheduled.values()),
        _ascend_kv_guard_context={
            "pool_id": "pool",
            "requests": {"r": [[(1, 1)]]} if requests is None else requests,
            "scheduled_requests": scheduled,
            "released_request_ids": list(released),
        },
    )


def prepare(adapter, ids, slots, tables, lengths, mode="NONE"):
    adapter.runner.next_metadata = {"layer0": metadata(torch.tensor(slots), torch.tensor(tables), lengths)}
    return adapter.runner.model_state.prepare_attn(NS(req_ids=ids, num_reqs=len(ids)), mode)


def reasons(runtime):
    return {event.get("reason") for event in runtime.events}


@pytest.mark.parametrize("mode", ["NONE", "FULL"])
def test_sorted_v2_batch_uses_batch_ids_instead_of_request_state_indices(adapter, mode):
    runtime = adapter.runtime
    packet = message(requests={"A": [[(1, 1)]], "B": [[(2, 1)]]}, scheduled={"A": (0, 1), "B": (0, 2)})
    adapter.runner.req_states = NS(req_id_to_index={"A": 9, "B": 3})

    def execute(output):
        prepare(adapter, ["B", "A"], [256, 257, 128], [[2], [1]], [2, 1], mode)
        if mode == "NONE":
            runtime.write("layer0", torch.tensor([256, 257, 128]))
            runtime.read("layer0", torch.tensor([[2], [1]]), [2, 1])
        return "unchanged"

    assert adapter.wrap_execute(execute)(packet) == "unchanged"
    assert not runtime.events
    assert runtime.state.blocks[0, 2].valid_end == 2
    assert runtime.state.blocks[0, 1].valid_end == 1
    assert runtime._phase is None and runtime._scheduled is None


@pytest.mark.parametrize("mode", ["NONE", "FULL"])
def test_v2_cross_request_swaps_are_not_legalized_by_allocated_blocks(adapter, mode):
    packet = message(requests={"A": [[(1, 1)]], "B": [[(2, 1)]]}, scheduled={"A": (0, 1), "B": (0, 1)})
    adapter.wrap_execute(lambda output: prepare(adapter, ["A", "B"], [256, 128], [[1], [2]], [1, 1], mode))(packet)
    assert "write_target_mismatch" in reasons(adapter.runtime)


def test_v2_full_replay_observes_current_metadata_after_parameter_update(adapter):
    runtime = adapter.runtime
    packet = message(requests={"r": [[(2, 1)]]})

    def execute(output):
        item = prepare(adapter, ["r"], [128, -1], [[1], [0]], [1, 0], "FULL")["layer0"]
        # Replay/parameter updates mutate the same buffers after collection.
        item.slot_mapping[0] = 256
        item.block_tables[0, 0] = 2
        runtime.write("layer0", torch.tensor([128]))
        assert runtime._accesses == []
        return "replayed"

    assert adapter.wrap_execute(execute)(packet) == "replayed"
    assert not runtime.events
    assert runtime.state.blocks[0, 2].valid_end == 1


def test_v2_full_cannot_reuse_last_steps_metadata(adapter):
    forward = adapter.wrap_execute(lambda output: prepare(adapter, ["r"], [128], [[1]], [1], "FULL"))
    forward(message())
    adapter.runner.model_state.prepare_attn = lambda batch, mode: None
    adapter.wrap_execute(lambda output: None)(message(requests={}, scheduled={"r": (1, 1)}))
    assert "v2_attention_metadata_missing" in reasons(adapter.runtime)
    assert adapter.runtime.state.blocks[0, 1].valid_end == 1


@pytest.mark.parametrize("ids", [["r", "r"], [], ["other"]])
def test_v2_missing_duplicate_or_unscheduled_batch_ids_are_coverage_gaps(adapter, ids):
    adapter.wrap_execute(lambda output: prepare(adapter, ids, [128], [[1]], [1]))(message())
    assert "v2_batch_request_mismatch" in reasons(adapter.runtime)
    assert adapter.runtime.state.blocks[0, 1].valid_end <= 0


def test_v2_receive_only_and_finished_steps_do_not_require_attention_metadata(adapter):
    sentinel = object()
    execute = adapter.wrap_execute(lambda output: sentinel)
    assert execute(message(scheduled={})) is sentinel
    assert "r" in adapter.runtime.state.requests
    assert execute(message(requests={}, scheduled={}, released=["r"])) is sentinel
    assert adapter.runtime.state.requests == {}
    assert not adapter.runtime.events


@pytest.mark.parametrize("profile", [False, True])
def test_v2_dummy_or_profile_writes_still_detect_case4_without_a_scheduler_binding(adapter, profile):
    runtime = adapter.runtime
    bind(runtime, computed=19)
    packet = NS(total_num_scheduled_tokens=1)
    adapter.wrap_execute(lambda output, **kwargs: prepare(adapter, ["dummy"], [142], [[1]], [19]))(
        packet, dummy_run=True, is_profile=profile
    )
    assert "dummy_write_to_live_block" in reasons(runtime)
    assert "scheduler_binding_missing" not in reasons(runtime)
    assert runtime.state.blocks[0, 1].valid_end < 0
    adapter.wrap_execute(lambda output: prepare(adapter, ["r"], [147], [[1]], [20]))(
        message(requests={}, scheduled={"r": (19, 1)})
    )
    assert "read_invalid_block" in reasons(runtime)


def test_v2_dummy_scratch_reads_are_ignored_but_null_writes_are_reported(adapter):
    packet = NS(total_num_scheduled_tokens=1)
    adapter.wrap_execute(lambda output, **kwargs: prepare(adapter, ["dummy"], [-1], [[0]], [1]))(packet, dummy_run=True)
    assert not adapter.runtime.events
    adapter.wrap_execute(lambda output, **kwargs: prepare(adapter, ["dummy"], [0], [[0]], [1]))(packet, dummy_run=True)
    assert "write_null_block" in reasons(adapter.runtime)


def test_v2_profile_without_attention_and_positional_flags_preserve_arguments(adapter):
    token = object()
    calls = []

    def execute(output, *args):
        calls.append((output, args))
        return token

    packet = NS(total_num_scheduled_tokens=16)
    assert adapter.wrap_execute(execute)(packet, token, True, True, True) is token
    assert calls == [(packet, (token, True, True, True))]
    assert not adapter.runtime.events


def test_v2_failed_forward_does_not_commit_writes_or_leak_scope(adapter):
    def execute(output):
        prepare(adapter, ["r"], [128], [[1]], [1])
        adapter.runtime.write("layer0", torch.tensor([128]))
        raise RuntimeError("original failure")

    with pytest.raises(RuntimeError, match="original failure"):
        adapter.wrap_execute(execute)(message())
    assert adapter.runtime.state.blocks[0, 1].valid_end == 0
    assert adapter.runtime._phase is None and adapter.runtime._scheduled is None


def test_v2_clears_then_copies_after_physical_update(adapter):
    runtime = adapter.runtime
    runtime.state.bind({"pool_id": "pool", "requests": {"r": [[(1, 1), (2, 1)]]}, "scheduled_requests": {"r": (4, 0)}})
    packet = NS(new_block_ids_to_zero=[2], kv_cache_block_copies=[NS(src_block_id=1, dst_block_id=2)])
    order = []
    clear, copy = runtime.zero_blocks, runtime.copy_blocks

    def cleared(blocks):
        order.append("clear")
        clear(blocks)

    def copied(blocks):
        order.append("copy")
        copy(blocks)

    runtime.zero_blocks = cleared
    runtime.copy_blocks = copied
    adapter.wrap_update(lambda output: order.append("physical"))(packet)
    assert order == ["physical", "clear", "copy"]
    assert runtime.state.blocks[0, 2].valid_end == 4
    adapter.wrap_update(lambda output: None)(NS(new_block_ids_to_zero=None, kv_cache_block_copies=None))


def test_v2_startup_capture_does_not_copy_metadata_inside_capture(adapter):
    calls = []
    adapter.runtime.observe_metadata = lambda metadata: calls.append(metadata)
    capture = adapter.wrap_capture(lambda: prepare(adapter, ["dummy"], [-1], [[0]], [1]))
    capture()
    assert calls == [] and not adapter.runtime.events


@pytest.mark.parametrize("released", [False, True])
def test_v2_live_or_prefix_preserving_recapture_is_rejected(adapter, released):
    bind(adapter.runtime, computed=19)
    if released:
        adapter.runtime.state.bind({"released_request_ids": ["r"]})
    with pytest.raises(ValueError, match="recapture is unsupported"):
        adapter.wrap_capture(lambda: pytest.fail("capture must not run"))()


def test_v2_kernel_allocator_coordinate_mismatch_is_rejected(adapter):
    adapter.runner.block_tables.kernel_block_sizes = [64]
    with pytest.raises(ValueError, match="allocator/kernel"):
        adapter.configure()


def test_v2_install_disabled_does_not_wrap_methods(adapter, monkeypatch):
    from vllm_ascend import envs

    monkeypatch.setattr(envs, "VLLM_ASCEND_KV_GUARD", False)
    assert adapter.subject.install_guard(NS(vllm_config=NS())) is None


def enable_raise(runtime):
    runtime.action = "raise"
    append = runtime.state.emit

    def report(event, **fields):
        append(event, **fields)
        runtime.report(event, **fields)

    runtime.state.emit = report
    runtime.gap = lambda reason, **fields: report("guard.coverage_gap", reason=reason, **fields)


def test_v2_raise_stops_output_and_latches_before_the_next_model_call(adapter):
    enable_raise(adapter.runtime)
    calls = []

    def execute(output):
        prepare(adapter, ["r"], [129], [[1]], [1])
        calls.append(True)
        return "must not escape"

    forward = adapter.wrap_execute(execute)
    for _ in range(2):
        with pytest.raises(adapter.runtime.subject.KVGuardError):
            forward(message())
    assert calls == [True]


def test_v2_missing_binding_and_batch_gap_stop_before_kernel_submission(adapter):
    enable_raise(adapter.runtime)
    calls = []
    with pytest.raises(adapter.runtime.subject.KVGuardError, match="scheduler_binding_missing"):
        adapter.wrap_execute(lambda output: calls.append(True))(NS(total_num_scheduled_tokens=1))
    assert calls == []


def test_v2_wrong_batch_ids_raise_before_the_model_runs(adapter):
    enable_raise(adapter.runtime)
    calls = []

    def execute(output):
        prepare(adapter, ["unknown"], [128], [[1]], [1])
        calls.append(True)

    with pytest.raises(adapter.runtime.subject.KVGuardError, match="v2_batch_request_mismatch"):
        adapter.wrap_execute(execute)(message())
    assert calls == []


def test_v2_install_wraps_real_runner_boundaries_and_reinitializes_model_state(adapter):
    runner = adapter.runner
    runner.execute_model = lambda output: prepare(adapter, ["r"], [128], [[1]], [1])
    runner.update_requests = lambda output: None
    runner.capture_model = lambda: "capture"
    installed = adapter.subject.install_guard(runner)
    assert installed is not None
    # Profiling can replace the model-state object before real cache setup.
    runner.model_state = NS(prepare_attn=lambda batch, mode: runner.next_metadata)
    installed.configure()
    wrapped = runner.model_state.prepare_attn
    installed.configure()
    assert runner.model_state.prepare_attn is wrapped
    assert runner.capture_model() == "capture"
    runner.execute_model(message())
    assert adapter.runtime.state.blocks[0, 1].valid_end == 1
    assert not adapter.runtime.events


def test_v2_keyword_calls_preserve_upstream_parameter_names(adapter):
    def execute(output):
        adapter.runner.next_metadata = {"layer0": metadata(torch.tensor([128]), torch.tensor([[1]]), [1])}
        return adapter.runner.model_state.prepare_attn(input_batch=NS(req_ids=["r"], num_reqs=1), cudagraph_mode="NONE")

    adapter.wrap_execute(execute)(scheduler_output=message())
    adapter.wrap_update(lambda output: "unchanged")(scheduler_output=NS())
    assert adapter.runtime.state.blocks[0, 1].valid_end == 1
    assert not adapter.runtime.events


def test_v2_pending_receive_also_blocks_graph_recapture(adapter):
    adapter.runtime.state.load({"request_id": "incoming", "local_block_ids": [[1]], "num_computed_tokens": 19})
    with pytest.raises(ValueError, match="recapture is unsupported"):
        adapter.wrap_capture(lambda: pytest.fail("capture must not run"))()
