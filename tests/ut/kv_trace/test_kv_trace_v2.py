# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MRV2-only installation, native argument forwarding and current FULL plans."""

from collections import namedtuple
from types import SimpleNamespace as NS

import pytest

from .test_kv_trace_runtime import message, metadata, reasons
from .test_kv_trace_runtime import runtime_fixture as runtime_fixture


@pytest.fixture(name="adapter")
def adapter_fixture(runtime):
    runner = NS(
        vllm_config=runtime.config,
        block_tables=NS(block_sizes=[128], kernel_block_sizes=[128], redirect_writes_to_null_block=False),
        kv_cache_config=NS(
            num_blocks=8, kv_cache_groups=[NS(layer_names=["layer0"], kv_cache_spec=NS(block_size=128, head_size=64))]
        ),
        next_metadata={},
        is_pooling_model=False,
        sample_tokens=lambda: None,
    )
    runner.model_state = NS(prepare_attn=lambda batch, mode: runner.next_metadata)
    adapter = runtime.runner_subject._V2RunnerAdapter(runner, runtime)
    runtime._configure_runner = adapter.configure
    runtime.configure(runner)
    return adapter


def prepare(adapter, ids, slots, tables, lengths, mode="NONE"):
    adapter.runner.next_metadata = {"layer0": metadata(slots, tables, lengths)}
    return adapter.runner.model_state.prepare_attn(NS(req_ids=ids, num_reqs=len(ids)), mode)


@pytest.mark.parametrize("mode", ["NONE", "FULL"])
def test_sorted_request_rows_are_independent_of_state_indices(adapter, mode):
    output = message(bindings={"A": [[1]], "B": [[2]]}, counts={"A": 1, "B": 2})
    adapter.runner.req_states = NS(req_id_to_index={"A": 9, "B": 3})
    execute = adapter.wrap_execute(
        lambda output: prepare(adapter, ["B", "A"], [256, 257, 128], [[2], [1]], [2, 1], mode)
    )
    execute(output)
    assert not reasons(adapter.runtime)


@pytest.mark.parametrize("mode", ["NONE", "FULL"])
def test_cross_request_slot_swap_is_checked(adapter, mode):
    output = message(bindings={"A": [[1]], "B": [[2]]})
    adapter.wrap_execute(lambda output: prepare(adapter, ["A", "B"], [256, 128], [[1], [2]], [1, 1], mode))(output)
    assert "write_target_mismatch" in reasons(adapter.runtime)


@pytest.mark.parametrize("ids", [["r", "r"], [], ["other"]])
def test_wrong_request_rows_are_violations(adapter, ids):
    adapter.wrap_execute(lambda output: prepare(adapter, ids, [128], [[1]], [1]))(message())
    assert "v2_batch_request_mismatch" in reasons(adapter.runtime)


def test_enabled_install_wraps_only_the_existing_mrv2_boundaries(runtime):
    runner = NS(
        vllm_config=runtime.config,
        block_tables=NS(block_sizes=[128], kernel_block_sizes=[128], redirect_writes_to_null_block=False),
        kv_cache_config=NS(
            num_blocks=8,
            kv_cache_groups=[NS(layer_names=["layer0"], kv_cache_spec=NS(block_size=128, head_size=64))],
        ),
        model_state=NS(prepare_attn=lambda batch, mode: {"layer0": metadata([128], [[1]], [1])}),
        update_requests=lambda output: "updated",
        capture_model=lambda: "captured",
        sample_tokens=lambda: "sampled",
        is_pooling_model=False,
    )

    def execute(output):
        runner.model_state.prepare_attn(NS(req_ids=["r"], num_reqs=1), "FULL")
        return "executed"

    runner.execute_model = execute
    assert runtime.runner_subject.install_trace(runner) is runtime
    runtime.configure(runner)
    assert runner.execute_model(message()) == "executed"
    assert runner.update_requests(message()) == "updated"
    assert runner.capture_model() == "captured"
    assert runner.sample_tokens() == "sampled"
    assert not reasons(runtime)


def test_full_metadata_is_frozen_after_native_parameter_update(adapter):
    def execute(output):
        item = prepare(adapter, ["r"], [128], [[1]], [1], "FULL")["layer0"]
        item.slot_mapping[0] = 256
        item.block_tables[0, 0] = 2

    adapter.wrap_execute(execute)(message(bindings={"r": [[2]]}))
    assert not reasons(adapter.runtime)


def test_dummy_without_scheduler_context_forwards_original_arguments(adapter):
    output = message(bindings={}, counts={})
    seen = []

    def execute(packet, intermediate, dummy_run, skip_attn_for_dummy_run, is_profile):
        seen.append((packet, intermediate, dummy_run, skip_attn_for_dummy_run, is_profile))
        prepare(adapter, [], [-1], [], [], "FULL")

    adapter.wrap_execute(execute)(output, "intermediate", True, False, False)
    assert seen == [(output, "intermediate", True, False, False)]
    assert not reasons(adapter.runtime)


def test_disabled_or_v1_install_does_not_wrap_methods(runtime):
    config = NS(use_v2_model_runner=False, additional_config={"kv_trace": {"enabled": True}})
    runner = NS(vllm_config=config, execute_model=lambda: None)
    original = runner.execute_model
    assert runtime.runner_subject.install_trace(runner) is None
    assert runner.execute_model is original and not hasattr(config, "_ascend_kv_trace")
    config.use_v2_model_runner = True
    config.additional_config["kv_trace"]["enabled"] = False
    assert runtime.runner_subject.install_trace(runner) is None


def test_zero_and_copy_are_records_without_lifecycle_side_effects(adapter):
    Copy = namedtuple("Copy", "src_block_id dst_block_id")
    output = message()
    output.new_block_ids_to_zero = [1]
    output.kv_cache_block_copies = [Copy(1, 2)]
    original = []

    def update(packet):
        original.append(packet)
        return "result"

    assert adapter.wrap_update(update)(output) == "result"
    assert original == [output]
    assert [event for event in adapter.runtime.events if event.get("zero_block_ids")] == [
        {"event": "kv_trace.boundary", "boundary": "model_runner.update_requests.return", "zero_block_ids": [1]}
    ]
    assert not hasattr(adapter.runtime.state, "blocks")
