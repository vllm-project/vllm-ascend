# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MRV2-only installation, native argument forwarding and current FULL plans."""

import sys
from collections import namedtuple
from types import ModuleType
from types import SimpleNamespace as NS

import pytest
import torch

from .test_kv_trace_runtime import message, metadata, reasons
from .test_kv_trace_runtime import runtime_fixture as runtime_fixture


@pytest.fixture(name="adapter")
def adapter_fixture(runtime):
    runner = NS(
        vllm_config=runtime.config,
        block_tables=NS(block_sizes=[128], redirect_writes_to_null_block=False),
        attn_groups=[[NS(kernel_block_size=128, kv_cache_spec=NS(block_size=128))]],
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


@pytest.mark.parametrize("pcp", [1, 2])
def test_mixed_kernel_block_sizes_use_each_attention_view(adapter, pcp):
    runtime, runner = adapter.runtime, adapter.runner
    runtime.config.parallel_config.prefill_context_parallel_size = pcp
    runner.kv_cache_config.kv_cache_groups[0].layer_names.append("split")
    backend = NS(get_name=lambda: "ASCEND")
    runner.attn_groups = [
        [
            NS(layer_names=["layer0"], backend=backend, kernel_block_size=None),
            NS(layer_names=["split"], backend=backend, kernel_block_size=64),
        ]
    ]
    runtime.configure(runner)
    batch = NS(
        req_ids=["r"],
        num_reqs=1,
        num_tokens=2,
        num_tokens_after_padding=2,
        positions=torch.tensor([63, 64]),
        query_start_loc=torch.tensor([0, 2]),
        seq_lens=torch.tensor([65]),
    )
    slots = [191, 192]
    if pcp > 1:
        runner.model_state.pcp_context = NS(
            global_batch=batch,
            global_block_tables=[torch.tensor([[1]])],
            global_slot_mappings=[torch.tensor(slots)],
            local_token_indices=[0, 1],
            padded_gather_idx=[0, 1],
        )
    runner.next_metadata = {"layer0": metadata(slots, [[1]], [65]), "split": metadata(slots, [[2, 3]], [65])}
    execute = adapter.wrap_execute(lambda output: runner.model_state.prepare_attn(batch, "NONE"))
    execute(message(computed={"r": 63}, counts={"r": 2}))
    assert not reasons(runtime)
    runner.next_metadata["split"].block_tables[0, 1] = 5
    execute(message(computed={"r": 63}, counts={"r": 2}))
    expected = "pcp_read_block_mapping_mismatch" if pcp > 1 else "read_target_mismatch"
    assert reasons(runtime) == {expected}


def test_configure_marks_mismatched_manager_blocks_observation_only(adapter):
    adapter.runner.block_tables.block_sizes = [64]
    adapter.runtime.configure(adapter.runner)
    assert adapter.runtime.layout[0]["reason"] == "block_coordinates_unadapted"


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
        block_tables=NS(block_sizes=[128], redirect_writes_to_null_block=False),
        attn_groups=[[NS(kernel_block_size=128, kv_cache_spec=NS(block_size=128))]],
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


@pytest.mark.parametrize("pcp,dcp", [(1, 1), (2, 1), (1, 2), (2, 2)])
def test_sfa_cp_views_preserve_observations_and_other_checks(adapter, monkeypatch, pcp, dcp):
    runtime, runner = adapter.runtime, adapter.runner
    runtime.action = "raise"
    runtime.config.parallel_config.prefill_context_parallel_size = pcp
    runtime.config.parallel_config.decode_context_parallel_size = dcp
    runner.dcp_rank = 0
    runner.kv_cache_config.kv_cache_groups[0].layer_names += ["sfa", "indexer"]
    runner.attn_groups = [
        [
            NS(layer_names=[layer], backend=NS(get_name=lambda name=name: name))
            for layer, name in [("layer0", "ASCEND"), ("sfa", "ASCEND_SFA"), ("indexer", "ASCEND_SFA_INDEXER")]
        ]
    ]
    if dcp > 1:
        common_cp = ModuleType("vllm_ascend.attention.context_parallel.common_cp")
        monkeypatch.setattr(common_cp, "use_history_current_split_decode", lambda *args, **kwargs: False, raising=False)
        monkeypatch.setitem(sys.modules, common_cp.__name__, common_cp)
    runtime.configure(runner)

    global_batch = NS(
        req_ids=["r"],
        num_reqs=1,
        num_tokens=2,
        num_tokens_after_padding=2,
        positions=torch.tensor([0, 1]),
        query_start_loc=torch.tensor([0, 2]),
        seq_lens=torch.tensor([2]),
    )
    slots = [128, -1] if dcp > 1 else [128, 129]
    batch = global_batch
    if pcp > 1:
        batch = NS(
            **{
                **vars(global_batch),
                "num_tokens": 1,
                "positions": torch.tensor([0]),
                "query_start_loc": torch.tensor([0, 1]),
                "seq_lens": torch.tensor([1]),
            }
        )
        runner.model_state.pcp_context = NS(
            global_batch=global_batch,
            global_block_tables=[torch.tensor([[1]])],
            global_slot_mappings=[torch.tensor(slots)],
            local_token_indices=[0],
            padded_gather_idx=[0, 1],
        )
    sfa_slots = slots[:1] if pcp > 1 else slots
    indexer_slots, sfa_tables = slots, [[1]]
    if dcp > 1:
        sfa_slots = indexer_slots = [256, 257]
        sfa_tables = [[2, 3]]
    runner.next_metadata = {
        "sfa": metadata(sfa_slots, sfa_tables, [2]),
        "layer0": metadata(slots, [[1]], [2]),
        "indexer": metadata(indexer_slots, sfa_tables, [2]),
    }
    execute = adapter.wrap_execute(lambda output: runner.model_state.prepare_attn(batch, "FULL"))
    execute(message(counts={"r": 2}))
    assert not reasons(runtime)
    observed = {
        (event["view_id"], event["event"])
        for event in runtime.events
        if event.get("role") == "target" and event["event"] in ("kv_trace.read", "kv_trace.write")
    }
    assert observed == {
        (view, event) for view in ("0", "(0, 1)", "(0, 2)") for event in ("kv_trace.read", "kv_trace.write")
    }
    skipped = {
        event["view_id"]
        for event in runtime.events
        if event.get("status") == "not_applicable" and event.get("reason") == "sfa_cp_metadata_unadapted"
    }
    assert skipped == ({"0", "(0, 2)"} if pcp > 1 or dcp > 1 else set())
    if pcp > 1:
        assert any(
            event.get("role") == "target.global" and event.get("status") == "checked" for event in runtime.events
        )

    runner.next_metadata["layer0"].slot_mapping[0] = 256
    expected = "pcp_write_frame_mismatch" if pcp > 1 else "write_target_mismatch"
    with pytest.raises(runtime.subject.KVTraceError, match=expected):
        execute(message(counts={"r": 2}))


def test_dummy_without_scheduler_context_forwards_original_arguments(adapter):
    output = message(bindings={}, counts={})
    seen = []

    def execute(packet, intermediate, dummy_run, skip_attn_for_dummy_run, is_profile):
        seen.append((packet, intermediate, dummy_run, skip_attn_for_dummy_run, is_profile))
        prepare(adapter, [], [-1], [], [], "FULL")

    adapter.wrap_execute(execute)(output, "intermediate", True, False, False)
    assert seen == [(output, "intermediate", True, False, False)]
    assert not reasons(adapter.runtime)


def test_disabled_install_does_not_wrap_methods(runtime):
    config = NS(additional_config={"kv_trace": {"enabled": False}})
    runner = NS(vllm_config=config, execute_model=lambda: None)
    original = runner.execute_model
    assert runtime.runner_subject.install_trace(runner) is None
    assert runner.execute_model is original and not hasattr(config, "_ascend_kv_trace")


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
