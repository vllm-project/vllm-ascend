# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Runner metadata per cache group, independent scheduler ranges and lifecycle."""

import importlib.util
import logging
import sys
from pathlib import Path
from types import ModuleType
from types import SimpleNamespace as NS
from typing import Any, cast

import pytest
import torch

from . import complete_load

ROOT = Path(__file__).parents[3]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(name="runtime")
def runtime_fixture(monkeypatch):
    package = ModuleType("vllm_ascend")
    package.__path__ = []
    debug = ModuleType("vllm_ascend.debug")
    debug.__path__ = []
    monkeypatch.setitem(sys.modules, "vllm_ascend", package)
    monkeypatch.setitem(sys.modules, "vllm_ascend.debug", debug)
    utils = load("guard_config_utils", "vllm_ascend/config_utils.py")
    monkeypatch.setitem(sys.modules, "vllm_ascend.config_utils", utils)
    options = load("guard_config_subject", "vllm_ascend/debug/kv_guard_config.py")
    monkeypatch.setitem(sys.modules, "vllm_ascend.debug.kv_guard_config", options)
    state = load("guard_runtime_state", "vllm_ascend/debug/kv_guard.py")
    monkeypatch.setitem(sys.modules, "vllm_ascend.debug.kv_guard", state)
    connector = load("guard_connector_subject", "vllm_ascend/debug/kv_guard_connector.py")
    monkeypatch.setitem(sys.modules, "vllm_ascend.debug.kv_guard_connector", connector)
    subject = load("guard_runtime_subject", "vllm_ascend/debug/kv_guard_runtime.py")
    logger = ModuleType("vllm.logger")
    cast(Any, logger).logger = logging.Logger("guard-test")
    vllm = ModuleType("vllm")
    vllm.__path__ = []
    monkeypatch.setitem(sys.modules, "vllm", vllm)
    monkeypatch.setitem(sys.modules, "vllm.logger", logger)
    config = NS(additional_config={"kv_guard_config": {"enabled": True}})
    runtime = subject.get_guard(config)
    assert subject.get_guard(config) is runtime
    runtime.subject = subject
    runtime.state.configure({0: 128})
    runtime.layer_groups = {"layer0": 0, "layer1": 0}
    runtime.events = []
    runtime.state.emit = lambda event, **fields: runtime.events.append({"event": event, **fields})
    runtime.gap = lambda reason, **fields: runtime.events.append(
        {"event": "guard.coverage_gap", "reason": reason, **fields}
    )
    return runtime


def bind(runtime, request="r", block=1, computed=0):
    runtime.state.bind(
        {"pool_id": "pool", "requests": {request: [[(block, 1)]]}, "scheduled_requests": {request: (computed, 0)}}
    )
    if computed:
        complete_load(runtime.state, request, computed)


def metadata(slots, table=None, lengths=None):
    return NS(slot_mapping=slots, block_tables=table, seq_lens=lengths, num_actual_tokens=len(slots))


def test_group_audit_shares_read_only_blocks_and_rejects_a_changed_write_allocation(runtime, monkeypatch):
    runtime.state.bind({"pool_id": "pool", "requests": {"r": [[(1, 1), (2, 1)]]}})
    complete_load(runtime.state, end=128)
    runtime._scheduled = {"r": (128, 1)}
    item = metadata(torch.tensor([256]), torch.tensor([[1, 2]]), torch.tensor([129]))
    runner = NS(input_batch=NS(req_ids=["r"]))
    context = NS(cudagraph_runtime_mode="FULL", attn_metadata={"layer0": item})
    guard_type = type(runtime.state)
    observe = guard_type.observe

    def observe_with_allocation_change(shadow, *args):
        assert shadow.blocks[0, 1] is runtime.state.blocks[0, 1]
        assert shadow.blocks[0, 2] is not runtime.state.blocks[0, 2]
        observe(shadow, *args)
        runtime.state.blocks[0, 2].epoch = 2

    monkeypatch.setattr(guard_type, "observe", observe_with_allocation_change)
    runtime.wrap_forward(runner, lambda count: None, lambda: context)(1)
    assert runtime.state.blocks[0, 2].valid_end == 0
    assert any(event.get("reason") == "allocation_changed_during_access_audit" for event in runtime.events)


def test_receive_only_step_binds_before_execute_and_cleans_finished(runtime):
    output = NS(
        _ascend_kv_guard_context={"pool_id": "pool", "requests": {"r": [[(1, 1)]]}, "scheduled_requests": {"r": (0, 0)}}
    )
    runtime.wrap_execute(lambda message: None)(output)
    complete_load(runtime.state)
    assert runtime.state.blocks[0, 1].valid_end == 19
    runtime.wrap_execute(lambda message: None)(
        NS(_ascend_kv_guard_context={"requests": {}, "released_request_ids": ["r"], "scheduled_requests": {}})
    )
    assert runtime.state.blocks[0, 1].owners == set()


def test_disabled_guard_does_not_create_runtime(runtime, monkeypatch):
    monkeypatch.setenv("VLLM_ASCEND_KV_GUARD", "1")
    monkeypatch.setenv("VLLM_ASCEND_KV_GUARD_ACTION", "raise")
    config = NS()
    assert runtime.subject.get_guard(config) is None
    assert not hasattr(config, "_ascend_kv_guard")


@pytest.mark.parametrize("use_v2", [False, True])
def test_guard_configuration_is_per_engine(runtime, use_v2, monkeypatch):
    monkeypatch.setenv("VLLM_ASCEND_KV_GUARD", "0")
    configurations = [
        NS(use_v2_model_runner=use_v2, additional_config={"kv_guard_config": options})
        for options in ({"enabled": True}, {"enabled": False}, {"enabled": True, "action": "raise"})
    ]
    log_guard, disabled_guard, raise_guard = [runtime.subject.get_guard(config) for config in configurations]
    assert log_guard is not None and log_guard.action == "log"
    assert disabled_guard is None
    assert raise_guard is not None and raise_guard.action == "raise"
    assert log_guard is not raise_guard
    assert runtime.subject.get_guard(configurations[0]) is log_guard
    assert runtime.subject.get_guard(configurations[2]) is raise_guard


@pytest.mark.parametrize(
    "options", [None, [], True, {"enabled": "false"}, {"enabled": 1}, {"action": "stop"}, {"enable": True}]
)
def test_guard_rejects_invalid_configuration(runtime, options):
    config = NS(additional_config={"kv_guard_config": options})
    with pytest.raises(ValueError):
        runtime.subject.get_guard(config)
    assert not hasattr(config, "_ascend_kv_guard")


@pytest.mark.parametrize("kernel_size, accepted", [(128, True), (16, False)])
def test_unadapted_kernel_block_coordinates_are_rejected(runtime, kernel_size, accepted):
    impl = type("AscendAttentionBackendImpl", (), {})()
    runtime.config = NS(
        parallel_config=NS(),
        kv_transfer_config=None,
        model_config=NS(),
        speculative_config=None,
        compilation_config=NS(static_forward_context={"layer0": NS(impl=impl)}),
    )
    runner = NS(
        input_batch=NS(block_table=NS(block_tables=[NS(block_size=kernel_size)])),
        kv_cache_config=NS(
            kv_cache_groups=[NS(kv_cache_spec=NS(block_size=128, head_size=64), layer_names=["layer0"])]
        ),
    )
    if accepted:
        runtime.configure(runner)
        assert not hasattr(impl, "_kv_guard")
    else:
        with pytest.raises(ValueError, match="unadapted"):
            runtime.configure(runner)


@pytest.mark.parametrize("use_v2", [False, True])
def test_scheduler_preserves_output_and_sends_only_changed_generations(runtime, use_v2):
    patch = load("guard_scheduler_subject", "vllm_ascend/patch/platform/patch_kv_guard.py")

    class Pool:
        blocks = [NS(block_id=i) for i in range(3)]
        null_block = blocks[0]

        def get_new_blocks(self, count):
            return [self.blocks[1]]

    class Scheduler:
        def __init__(self, vllm_config):
            self.vllm_config = vllm_config
            self.pool = Pool()
            self.mapping = {"r": [[1]]}
            self.kv_cache_manager = NS(
                block_pool=self.pool, get_block_ids=lambda request: self.mapping[request], free=lambda request: None
            )
            self.output = NS(
                scheduled_new_reqs=[NS(req_id="r", num_computed_tokens=0)],
                scheduled_cached_reqs=NS(req_ids=[], num_computed_tokens=[]),
                num_scheduled_tokens={"r": 1},
                finished_req_ids=set(),
            )

        def schedule(self):
            return self.output

    patch.install_scheduler_guard(Scheduler)
    patch.install_scheduler_guard(Scheduler)
    scheduler = Scheduler(NS(use_v2_model_runner=use_v2, additional_config={"kv_guard_config": {"enabled": True}}))
    for config in (NS(additional_config=None), NS(additional_config={"kv_guard_config": {"enabled": False}})):
        disabled = Scheduler(vllm_config=config)
        assert "schedule" not in vars(disabled)
        assert "get_new_blocks" not in vars(disabled.pool)
        assert disabled.schedule() is disabled.output
        assert not hasattr(disabled.output, "_ascend_kv_guard_context")
    scheduler.pool.get_new_blocks(1)
    assert scheduler.schedule() is scheduler.output
    binding = scheduler.output._ascend_kv_guard_context
    assert binding["requests"]["r"] == [[(1, 1)]]
    assert binding["scheduled_requests"] == {"r": (0, 1)}
    scheduler.schedule()
    assert scheduler.output._ascend_kv_guard_context["requests"] == {}
    scheduler.pool.get_new_blocks(1)
    scheduler.schedule()
    assert scheduler.output._ascend_kv_guard_context["requests"]["r"] == [[(1, 2)]]
    scheduler.kv_cache_manager.free(NS(request_id="r"))
    scheduler.schedule()
    assert scheduler.output._ascend_kv_guard_context["released_request_ids"] == ["r"]
    assert "r" in scheduler.output._ascend_kv_guard_context["requests"]
    scheduler.output.finished_req_ids = {"r"}
    scheduler.output.num_scheduled_tokens = {}
    scheduler.schedule()
    assert scheduler.output._ascend_kv_guard_context == {
        "pool_id": binding["pool_id"],
        "requests": {},
        "released_request_ids": ["r"],
        "scheduled_requests": {},
    }


@pytest.mark.parametrize("slots, mismatch", [([128, 129], False), ([129, 128], True)])
def test_full_graph_update_audits_final_metadata_without_guard_callbacks(runtime, monkeypatch, slots, mismatch):
    order: list[Any] = []
    monkeypatch.setattr(
        torch,
        "npu",
        NS(
            NPUGraph=object,
            graph_task_update_begin=lambda *args: order.append("begin"),
            graph_task_update_end=lambda *args: order.append("end"),
        ),
        raising=False,
    )
    monkeypatch.setitem(sys.modules, "torch_npu", ModuleType("torch_npu"))
    utils = ModuleType("vllm_ascend.utils")
    cast(Any, utils).weak_ref_tensors = lambda value: value
    monkeypatch.setitem(sys.modules, "vllm_ascend.utils", utils)
    graph = load("guard_graph_update_subject", "vllm_ascend/compilation/updatable_graph.py")
    bind(runtime)
    runtime.layer_groups = {"layer0": 0}
    runtime._scheduled = {"r": (0, 2)}
    item = metadata(torch.tensor([-1, -1]), torch.tensor([[1]]), torch.tensor([0]))
    runner = NS(input_batch=NS(req_ids=["r"]))
    context = NS(cudagraph_runtime_mode="FULL", attn_metadata={"layer0": item})

    def operation(**kwargs):
        order.append(("operation", kwargs["context_lens"].tolist()))
        item.slot_mapping[:] = torch.tensor(slots)
        item.seq_lens[:] = kwargs["context_lens"]

    task = graph.GraphUpdateTask(
        operation=operation,
        kwargs={"block_table": item.block_tables, "context_lens": torch.tensor([0])},
        provider=NS(layer_name="layer0"),
        provider_index=0,
        handle="handle",
        event=NS(record=lambda stream: order.append("event")),
    )
    bound = task.bind({"context_lens": torch.tensor([2])})

    def replay(count):
        bound.apply("stream")
        return "result"

    assert runtime.wrap_forward(runner, replay, lambda: context)(2) == "result"
    assert order == ["begin", ("operation", [2]), "end", "event"]
    assert any(event.get("reason") == "write_target_mismatch" for event in runtime.events) == mismatch
    assert not any(event["event"] == "guard.coverage_gap" for event in runtime.events)
    if not mismatch:
        assert runtime.state.blocks[0, 1].valid_end == 2


def test_v1_dummy_wrapper_preserves_decoder_replay_argument(runtime):
    calls = []
    runner = NS(input_batch=NS(req_ids=[]))
    context = NS(cudagraph_runtime_mode="NONE", attn_metadata=None)

    def forward(count, *, decoder_replay_dummy):
        calls.append((count, decoder_replay_dummy))
        return "unchanged"

    wrapped = runtime.wrap_forward(runner, forward, lambda: context)
    assert wrapped(8, decoder_replay_dummy=True, _kv_guard_phase="warmup") == "unchanged"
    assert calls == [(8, True)]
    assert not runtime.events


def test_repeated_violations_are_deduplicated_without_hiding_other_blocks(runtime):
    messages = []
    runtime.logger = NS(
        info=lambda *args: messages.append(("info", args)),
        warning=lambda *args: messages.append(("warning", args)),
        error=lambda *args: messages.append(("error", args)),
    )
    runtime.report("guard.violation", operation="read", reason="read_invalid_block", block_id=1)
    runtime.report("guard.violation", operation="read", reason="read_invalid_block", block_id=1)
    runtime.report("guard.violation", operation="read", reason="read_invalid_block", block_id=2)
    assert [level for level, args in messages].count("error") == 2


def test_runner_plan_detects_case4_without_any_operator_hook(runtime):
    bind(runtime, computed=19)
    item = metadata(torch.tensor([142]), torch.tensor([[1]]), torch.tensor([19]))
    runner = NS(input_batch=NS(req_ids=[]))
    context = NS(cudagraph_runtime_mode="NONE", attn_metadata={"layer0": item, "layer1": item})
    runtime.wrap_forward(runner, lambda n: "result", lambda: context)(1, _kv_guard_phase="dummy")
    violation = next(e for e in runtime.events if e.get("reason") == "dummy_write_to_live_block")
    assert violation["evidence"] == "execution_plan" and violation["offsets"] == [14]
    assert runtime.state.blocks[0, 1].valid_end == runtime.subject.INVALID_PREFIX


def test_final_metadata_is_read_after_forward_parameter_updates_and_copied_once(runtime, monkeypatch):
    bind(runtime)
    slots, table, lengths = torch.tensor([-1]), torch.tensor([[1]]), torch.tensor([0])
    item = metadata(slots, table, lengths)
    runner = NS(input_batch=NS(req_ids=["r"]), kv_caches=object())
    context = NS(cudagraph_runtime_mode="FULL", attn_metadata={"layer0": item, "layer1": item})
    copies = []
    original = runtime._list

    def record_copy(tensor):
        copies.append(tensor.data_ptr())
        return original(tensor)

    monkeypatch.setattr(runtime, "_list", record_copy)

    def replay(n):
        assert not copies  # No D2H before asynchronous graph parameter updates.
        assert runtime.state.blocks[0, 1].valid_end == 0
        slots[0], lengths[0] = 128, 1
        return "result"

    assert runtime.wrap_forward(runner, replay, lambda: context)(1) == "result"
    assert runtime.state.blocks[0, 1].valid_end == 1
    assert len(copies) == 3 and len(set(copies)) == 3
    assert not any(e["event"] in ("guard.violation", "guard.coverage_gap") for e in runtime.events)


def test_metadata_retains_complete_overwrite_diagnostic(runtime):
    bind(runtime, computed=1)
    runtime._scheduled = {"r": (0, 1)}
    item = metadata(torch.tensor([128]), torch.tensor([[1]]), torch.tensor([1]))
    context = NS(cudagraph_runtime_mode="FULL", attn_metadata=dict.fromkeys(runtime.layer_groups, item))
    runtime.wrap_forward(NS(input_batch=NS(req_ids=["r"])), lambda count: None, lambda: context)(1)
    event = next(event for event in runtime.events if event.get("reason") == "overwrite_valid_prefix")
    assert event["offsets"] == [0] and event["valid_end"] == 1
    assert event["evidence"] == "execution_plan" and event["group_id"] == 0


def test_unadapted_ring_cache_is_rejected_instead_of_silently_reducing_checks(runtime):
    spec = type("CircularBufferSpec", (), {"block_size": 128, "head_size": 64})()
    runtime.config = NS(
        parallel_config=NS(),
        kv_transfer_config=None,
        model_config=NS(is_hybrid=True),
        speculative_config=None,
        compilation_config=NS(static_forward_context={"ring": NS(kv_cache=object())}),
    )
    runner = NS(
        input_batch=NS(req_ids=[]), kv_cache_config=NS(kv_cache_groups=[NS(kv_cache_spec=spec, layer_names=["ring"])])
    )
    with pytest.raises(ValueError, match="unadapted cache validity"):
        runtime.configure(runner)


def test_failed_forward_preserves_exception_and_does_not_commit_planned_writes(runtime):
    bind(runtime)
    item = metadata(torch.tensor([128]), torch.tensor([[1]]), [1])
    context = NS(cudagraph_runtime_mode="NONE", attn_metadata={"layer0": item, "layer1": item})
    runner = NS(input_batch=NS(req_ids=["r"]))

    def model(n):
        raise RuntimeError("model failure")

    with pytest.raises(RuntimeError, match="model failure"):
        runtime.wrap_forward(runner, model, lambda: context)(1)
    assert runtime.state.blocks[0, 1].valid_end == 0 and runtime._phase is None


def test_only_needed_block_table_rows_and_columns_are_copied(runtime, monkeypatch):
    bind(runtime)
    table = torch.full((32, 1024), 99)
    table[0, 0] = 1
    item = metadata(torch.tensor([128]), table, [1])
    context = NS(cudagraph_runtime_mode="NONE", attn_metadata={"layer0": item, "layer1": item})
    runner = NS(input_batch=NS(req_ids=["r"]))
    shapes = []
    original = runtime._list

    def record_shape(value):
        shapes.append(tuple(getattr(value, "shape", ())))
        return original(value)

    monkeypatch.setattr(runtime, "_list", record_shape)
    runtime.wrap_forward(runner, lambda n: None, lambda: context)(1)
    assert (1, 1) in shapes and (32, 1024) not in shapes
    assert not any(e["event"] == "guard.violation" for e in runtime.events)


def test_unknown_backend_is_rejected_without_mutating_it(runtime):
    class OtherAttentionImpl:
        __slots__ = ()

    impl = OtherAttentionImpl()
    runtime.layer_groups.clear()
    runtime.config = NS(
        parallel_config=NS(),
        kv_transfer_config=None,
        model_config=NS(),
        speculative_config=None,
        compilation_config=NS(static_forward_context={"layer0": NS(impl=impl)}),
    )
    runner = NS(
        input_batch=NS(req_ids=["r"]),
        kv_cache_config=NS(
            kv_cache_groups=[NS(kv_cache_spec=NS(block_size=128, head_size=64), layer_names=["layer0"])]
        ),
    )
    with pytest.raises(ValueError, match="unadapted attention implementation"):
        runtime.configure(runner)
    assert not hasattr(impl, "_kv_guard")


def test_execute_and_update_bind_once_without_skipping_receive_only_or_next_step(runtime, monkeypatch):
    output = NS(_ascend_kv_guard_context={"requests": {}, "released_request_ids": []})
    runner = NS(requests={})
    bound = []
    original = runtime.state.bind

    def record(binding):
        bound.append(binding)
        return original(binding)

    monkeypatch.setattr(runtime.state, "bind", record)
    update = runtime.wrap_update(runner, lambda value: None)
    execute = runtime.wrap_execute(update)
    execute(output)
    assert len(bound) == 1 and runtime._scheduled is None
    execute(output)
    assert len(bound) == 2 and runtime._scheduled is None
    runtime.wrap_execute(lambda value: None)(output)
    assert len(bound) == 3 and runtime._scheduled is None


def test_shared_unsupported_metadata_reports_missing_read_coverage_for_each_group(runtime):
    runtime.layer_groups = {"layer0": 0, "layer1": 1}
    runtime.state.configure({0: 128, 1: 128})
    item = metadata(torch.tensor([-1]))
    context = NS(cudagraph_runtime_mode="FULL", attn_metadata={"layer0": item, "layer1": item})
    runner = NS(input_batch=NS(req_ids=[]))
    runtime.wrap_forward(runner, lambda n: None, lambda: context)(1)
    assert {
        event.get("group_id") for event in runtime.events if event.get("reason") == "metadata_read_range_missing"
    } == {0, 1}


def test_raise_mode_stops_on_coverage_gap_before_execution(runtime):
    runtime.action = "raise"
    runtime.gap = runtime.subject.GuardRuntime.gap.__get__(runtime)
    called = []
    with pytest.raises(runtime.subject.KVGuardError, match="scheduler_write_plan_missing"):
        runtime.wrap_execute(lambda message: called.append(True))(NS(_ascend_kv_guard_context={"requests": {}}))
    assert not called


def test_physical_zero_and_copy_preserve_only_proven_validity(runtime):
    bind(runtime, "source", 1, 19)
    output = NS(
        _ascend_kv_guard_context={
            "pool_id": "pool",
            "requests": {"dest": [[(2, 1)]]},
            "scheduled_requests": {"dest": (19, 0)},
        },
        new_block_ids_to_zero=[2],
        kv_cache_block_copies=[NS(src_block_id=1, dst_block_id=2)],
    )

    def execute(message):
        assert runtime.state.blocks[0, 2].valid_end == 0
        runtime.zero_blocks(message.new_block_ids_to_zero)
        assert runtime.state.blocks[0, 2].valid_end == 0
        runtime.copy_blocks(message.kv_cache_block_copies)
        assert runtime.state.blocks[0, 2].valid_end == 19

    runtime.wrap_execute(execute)(output)
    assert not any(e["event"] == "guard.violation" for e in runtime.events)
    runtime.zero_blocks([2])
    assert runtime.state.blocks[0, 2].valid_end == runtime.subject.INVALID_PREFIX
    assert any(e.get("reason") == "clear_live_block" for e in runtime.events)


def test_copy_to_null_or_unbound_block_cannot_silently_pass(runtime):
    bind(runtime, computed=19)
    runtime.copy_blocks([NS(src_block_id=1, dst_block_id=0), NS(src_block_id=1, dst_block_id=99)])
    assert any(e.get("reason") == "write_null_block" and e["operation"] == "copy" for e in runtime.events)
    assert any(e.get("reason") == "copy_destination_unbound" and e["block_id"] == 99 for e in runtime.events)


def test_scheduler_optional_zero_and_copy_fields_accept_none(runtime):
    output = NS(
        _ascend_kv_guard_context={"requests": {}, "scheduled_requests": {}},
        new_block_ids_to_zero=None,
        kv_cache_block_copies=None,
    )
    assert runtime.wrap_execute(lambda message: "executed")(output) == "executed"
    assert not any(e["event"] in ("guard.violation", "guard.coverage_gap") for e in runtime.events)


def run_plan(runtime, slots, table, lengths, *, ids=("r",), mode="FULL", phase="forward", forward=None):
    item = metadata(torch.tensor(slots), torch.tensor(table), torch.tensor(lengths))
    context = NS(cudagraph_runtime_mode=mode, attn_metadata=dict.fromkeys(runtime.layer_groups, item))
    runner = NS(input_batch=NS(req_ids=list(ids)))
    return runtime.wrap_forward(runner, forward or (lambda count: None), lambda: context)(
        len(slots), _kv_guard_phase=phase
    )


@pytest.mark.parametrize("mode", ["NONE", "FULL"])
def test_group_metadata_is_checked_once_without_attention_hooks(runtime, monkeypatch, mode):
    bind(runtime)
    runtime._scheduled = {"r": (0, 1)}
    calls = []
    original = runtime.subject.KVGuard.observe

    def checked(state, groups, requests, phase):
        calls.append((len(groups), list(requests)))
        return original(state, groups, requests, phase)

    monkeypatch.setattr(runtime.subject.KVGuard, "observe", checked)
    run_plan(runtime, [128], [[1]], [1], mode=mode)
    assert calls == [(1, ["r"])]
    assert runtime.state.blocks[0, 1].valid_end == 1
    assert not runtime.events
    assert set(runtime._metadata) == {0}
    assert not hasattr(runtime, "write") and not hasattr(runtime, "read")
    assert not hasattr(runtime, "_accesses")


@pytest.mark.parametrize("mode", ["NONE", "FULL"])
@pytest.mark.parametrize(
    "slots,table,lengths,reason",
    [
        ([128, 129], [[1]], [2], None),
        ([129, 128], [[1]], [2], "write_target_mismatch"),
        ([128, 128], [[1]], [2], "write_target_mismatch"),
        ([128, 129], [[2]], [2], "read_target_mismatch"),
        ([128, 129], [[1]], [1], None),
        ([128, 129], [[1]], [3], "read_length_mismatch"),
    ],
)
def test_metadata_constraints_share_one_checker(runtime, mode, slots, table, lengths, reason):
    bind(runtime)
    runtime._scheduled = {"r": (0, 2)}
    run_plan(runtime, slots, table, lengths, mode=mode)
    violations = [event for event in runtime.events if event["event"] == "guard.violation"]
    assert any(event.get("reason") == reason for event in violations) if reason else not violations
    assert all(event["evidence"] == "execution_plan" and "layer" not in event for event in violations)


@pytest.mark.parametrize("mode", ["NONE", "FULL"])
@pytest.mark.parametrize("swapped", [False, True])
def test_group_plan_checks_actual_request_order_and_cross_request_slots(runtime, mode, swapped):
    bind(runtime, "A", 1)
    bind(runtime, "B", 2)
    runtime._scheduled = {"A": (0, 1), "B": (0, 1)}
    run_plan(runtime, [128, 256] if swapped else [256, 128], [[2], [1]], [1, 1], ids=("B", "A"), mode=mode)
    assert any(event.get("reason") == "write_target_mismatch" for event in runtime.events) == swapped


@pytest.mark.parametrize("slots,mismatch", [([147, 148], False), ([148, 147], True), ([147, -1], True)])
def test_group_metadata_distinguishes_exact_token_positions_and_padding(runtime, slots, mismatch):
    bind(runtime, computed=19)
    runtime._scheduled = {"r": (19, 2)}
    run_plan(runtime, slots, [[1]], [21])
    assert any(event.get("reason") == "write_target_mismatch" for event in runtime.events) == mismatch


def test_truncated_group_write_cannot_advance_validity(runtime):
    bind(runtime, computed=19)
    runtime._scheduled = {"r": (19, 2)}
    run_plan(runtime, [147], [[1]], [19])
    assert any(event.get("reason") == "write_tokens_not_observed" for event in runtime.events)
    assert runtime.state.blocks[0, 1].valid_end == 19


def test_computed_range_does_not_initialize_unwritten_blocks(runtime):
    bind(runtime)
    runtime._scheduled = {"r": (19, 1)}
    run_plan(runtime, [147], [[1]], [20])
    assert any(event.get("reason") == "noncontiguous_write" for event in runtime.events)
    assert runtime.state.blocks[0, 1].valid_end == runtime.subject.INVALID_PREFIX


def test_read_block_order_is_checked_within_one_request(runtime):
    runtime.state.bind({"pool_id": "pool", "requests": {"r": [[(1, 1), (2, 1)]]}})
    complete_load(runtime.state, "r", 254)
    runtime._scheduled = {"r": (254, 1)}
    run_plan(runtime, [382], [[2, 1]], [255])
    assert any(event.get("reason") == "read_target_mismatch" for event in runtime.events)


def test_shared_prefix_and_distinct_tail_blocks_remain_legal(runtime):
    runtime.state.bind({"pool_id": "pool", "requests": {"A": [[(1, 1), (2, 1)]], "B": [[(1, 1), (3, 1)]]}})
    complete_load(runtime.state, "A", 128)
    runtime._scheduled = {"A": (128, 1), "B": (128, 1)}
    run_plan(runtime, [384, 256], [[1, 3], [1, 2]], [129, 129], ids=("B", "A"))
    assert not runtime.events


def test_dummy_metadata_write_still_taints_a_live_block(runtime):
    bind(runtime, computed=19)
    run_plan(runtime, [142], [[1]], [19], ids=(), mode="NONE", phase="dummy")
    assert any(event.get("reason") == "dummy_write_to_live_block" for event in runtime.events)
    run_plan(runtime, [-1], [[1]], [19], mode="NONE")
    assert any(event.get("reason") == "read_invalid_block" for event in runtime.events)


def test_only_group_coverage_is_required_not_every_layer(runtime):
    bind(runtime)
    item = metadata(torch.tensor([128]), torch.tensor([[1]]), [1])
    context = NS(cudagraph_runtime_mode="NONE", attn_metadata={"layer1": item})
    runtime.wrap_forward(NS(input_batch=NS(req_ids=["r"])), lambda count: None, lambda: context)(1)
    assert not runtime.events and runtime.state.blocks[0, 1].valid_end == 1


def test_missing_cache_group_is_an_explicit_coverage_gap(runtime):
    runtime.state.configure({0: 128, 1: 128})
    runtime.layer_groups = {"layer0": 0, "layer1": 1}
    item = metadata(torch.tensor([-1]), torch.tensor([[0]]), [0])
    context = NS(cudagraph_runtime_mode="FULL", attn_metadata={"layer0": item})
    runtime.wrap_forward(NS(input_batch=NS(req_ids=[])), lambda count: None, lambda: context)(1)
    assert any(
        event.get("reason") == "metadata_write_mapping_missing" and event.get("group_id") == 1
        for event in runtime.events
    )


def test_raise_mode_stops_results_and_latches_metadata_failure(runtime):
    runtime.action = "raise"
    runtime.state.emit = runtime.report
    bind(runtime, computed=19)
    runtime._scheduled = {"r": (19, 1)}
    with pytest.raises(runtime.subject.KVGuardError, match="write_target_mismatch"):
        run_plan(runtime, [148], [[1]], [20])
    with pytest.raises(runtime.subject.KVGuardError):
        runtime.wrap_execute(lambda output: None)(NS())
