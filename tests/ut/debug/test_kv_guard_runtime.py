# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Runner plans, ordered operator arguments, graph replay and allocation transport."""

import importlib.util
import logging
import sys
from pathlib import Path
from types import ModuleType
from types import SimpleNamespace as NS
from typing import Any, cast

import pytest
import torch

ROOT = Path(__file__).parents[3]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def runtime(monkeypatch):
    package = ModuleType("vllm_ascend")
    package.__path__ = []
    cast(Any, package).envs = NS(VLLM_ASCEND_KV_GUARD=True, VLLM_ASCEND_KV_GUARD_ACTION="log")
    debug = ModuleType("vllm_ascend.debug")
    debug.__path__ = []
    monkeypatch.setitem(sys.modules, "vllm_ascend", package)
    monkeypatch.setitem(sys.modules, "vllm_ascend.debug", debug)
    state = load("guard_runtime_state", "vllm_ascend/debug/kv_guard.py")
    monkeypatch.setitem(sys.modules, "vllm_ascend.debug.kv_guard", state)
    subject = load("guard_runtime_subject", "vllm_ascend/debug/kv_guard_runtime.py")
    logger = ModuleType("vllm.logger")
    cast(Any, logger).logger = logging.Logger("guard-test")
    vllm = ModuleType("vllm")
    vllm.__path__ = []
    monkeypatch.setitem(sys.modules, "vllm", vllm)
    monkeypatch.setitem(sys.modules, "vllm.logger", logger)
    config = NS()
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


def test_access_hooks_validate_case4_without_kv_copy_or_hash(runtime):
    bind(runtime, computed=19)
    slots, table = torch.tensor([142]), torch.tensor([[1]])
    runner, context = NS(input_batch=NS(req_ids=[])), NS(cudagraph_runtime_mode="NONE")

    def dummy(n):
        runtime.write("layer0", slots)
        runtime.read("layer0", table, [1])
        return "unchanged-model-result"

    model = runtime.wrap_forward(runner, dummy, lambda: context)
    assert model(1, _kv_guard_phase="dummy") == "unchanged-model-result"
    assert any(e.get("reason") == "dummy_write_to_live_block" and e["offsets"] == [14] for e in runtime.events)
    runner.input_batch.req_ids = ["r"]
    runtime.wrap_forward(runner, lambda n: runtime.read("layer0", table, [19]), lambda: context)(1)
    assert any(e.get("reason") == "read_invalid_block" for e in runtime.events)


def test_full_graph_without_current_metadata_reports_a_gap(runtime):
    bind(runtime, computed=19)
    runner, context = NS(input_batch=NS(req_ids=[])), NS(cudagraph_runtime_mode="FULL")

    def replay(n):
        runtime.write("layer0", torch.tensor([142]))
        runtime.read("layer0", torch.tensor([[1]]), [19])

    runtime.wrap_forward(runner, replay, lambda: context)(1, _kv_guard_phase="dummy")
    assert any(e.get("reason") == "full_graph_metadata_missing" for e in runtime.events)
    assert runtime.state.blocks[0, 1].valid_end == 19


def test_layer_hook_duplicates_do_not_look_like_repeated_writes(runtime):
    bind(runtime)
    slots, table = torch.arange(128, 147), torch.tensor([[1]])
    runner, context = NS(input_batch=NS(req_ids=["r"])), NS(cudagraph_runtime_mode="NONE")

    def model(n):
        for layer in ("layer0", "layer1"):
            runtime.write(layer, slots[:19])
            runtime.read(layer, table, [19])

    runtime.wrap_forward(runner, model, lambda: context)(19)
    assert not any(e["event"] == "guard.violation" for e in runtime.events)
    assert runtime.state.blocks[0, 1].valid_end == 19


def test_split_decode_prefill_write_ranges_and_read_request_offsets(runtime):
    bind(runtime, "decode", 1, 19)
    bind(runtime, "prefill", 2, 10)
    slots = torch.tensor([147, 266, 267])
    runner, context = NS(input_batch=NS(req_ids=["decode", "prefill"])), NS(cudagraph_runtime_mode="NONE")

    def model(n):
        runtime.write("layer0", slots[:1])
        runtime.write("layer0", slots[1:])
        runtime.read("layer0", torch.tensor([[1]]), [20])
        runtime.read("layer0", torch.tensor([[2]]), [12], request_offset=1)

    runtime.wrap_forward(runner, model, lambda: context)(3)
    assert not any(e["event"] == "guard.violation" for e in runtime.events)


def test_receive_only_step_binds_before_execute_and_cleans_finished(runtime):
    output = NS(
        _ascend_kv_guard_context={"pool_id": "pool", "requests": {"r": [[(1, 1)]]}, "scheduled_requests": {"r": (0, 0)}}
    )
    runtime.wrap_execute(lambda message: None)(output)
    runtime.transfer("load", {"request_id": "r", "local_block_ids": [[1]], "num_computed_tokens": 19})
    assert runtime.state.blocks[0, 1].valid_end == 19
    runtime.wrap_execute(lambda message: None)(
        NS(_ascend_kv_guard_context={"requests": {}, "released_request_ids": ["r"], "scheduled_requests": {}})
    )
    assert runtime.state.blocks[0, 1].owners == set()


def test_disabled_guard_does_not_create_runtime(runtime, monkeypatch):
    sys.modules["vllm_ascend"].envs.VLLM_ASCEND_KV_GUARD = False
    config = NS()
    assert runtime.subject.get_guard(config) is None
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
        assert impl._kv_guard is runtime
    else:
        with pytest.raises(ValueError, match="unadapted"):
            runtime.configure(runner)


def test_scheduler_preserves_output_and_sends_only_changed_generations(runtime):
    patch = load("guard_scheduler_subject", "vllm_ascend/patch/platform/patch_kv_guard.py")

    class Pool:
        blocks = [NS(block_id=i) for i in range(3)]
        null_block = blocks[0]

        def get_new_blocks(self, count):
            return [self.blocks[1]]

    class Scheduler:
        def __init__(self):
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
    scheduler = Scheduler()
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


def test_graph_update_uses_bound_guard_and_exact_operator_kwargs(runtime, monkeypatch):
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
    runtime.read = lambda layer, table, lengths: order.append(("read", layer, table.tolist(), lengths.tolist()))
    task = graph.GraphUpdateTask(
        operation=lambda **kwargs: order.append(("operation", kwargs["context_lens"].tolist())),
        kwargs={"block_table": torch.tensor([[1]]), "context_lens": torch.tensor([19])},
        provider=NS(layer_name="layer0"),
        provider_index=0,
        handle="handle",
        event=NS(record=lambda stream: order.append("event")),
        kv_guard=runtime,
    )
    bound = task.bind({"context_lens": torch.tensor([20])})
    assert bound.kv_guard is runtime
    bound.apply("stream")
    assert order == [("read", "layer0", [[1]], [20]), "begin", ("operation", [20]), "end", "event"]


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


def test_eager_raw_key_read_does_not_require_paged_cache_read_metadata(runtime):
    bind(runtime)
    runner, context = NS(input_batch=NS(req_ids=["r"])), NS(cudagraph_runtime_mode="NONE")

    def model(n):
        runtime.write("layer0", torch.tensor([128]))
        runtime.read("layer0", None, [1])

    runtime.wrap_forward(runner, model, lambda: context)(1)
    assert runtime.state.blocks[0, 1].valid_end == 1
    assert not any(e["event"] == "guard.coverage_gap" for e in runtime.events)


def test_same_layer_double_write_is_not_deduplicated(runtime):
    bind(runtime)
    slots = torch.tensor([128])
    runner, context = NS(input_batch=NS(req_ids=["r"])), NS(cudagraph_runtime_mode="NONE")

    def model(n):
        runtime.write("layer0", slots)
        runtime.write("layer0", slots)

    runtime.wrap_forward(runner, model, lambda: context)(1)
    assert any(e.get("reason") == "overwrite_valid_prefix" for e in runtime.events)


def test_equivalent_cloned_slots_across_layers_remain_one_logical_write(runtime):
    bind(runtime)
    slots = torch.tensor([128])
    runner, context = NS(input_batch=NS(req_ids=["r"])), NS(cudagraph_runtime_mode="NONE")

    def model(n):
        runtime.write("layer0", slots)
        runtime.write("layer1", slots.clone())

    runtime.wrap_forward(runner, model, lambda: context)(1)
    assert not any(e["event"] == "guard.violation" for e in runtime.events)


def test_v2_runner_does_not_silently_skip_the_guard(runtime):
    patch = load("guard_v2_gate_subject", "vllm_ascend/patch/platform/patch_kv_guard.py")

    class Scheduler:
        def __init__(self):
            self.vllm_config = NS(use_v2_model_runner=True)

    patch.install_scheduler_guard(Scheduler)
    with pytest.raises(ValueError, match="Model Runner V1"):
        Scheduler()


def metadata(slots, table=None, lengths=None):
    return NS(slot_mapping=slots, block_tables=table, seq_lens=lengths, num_actual_tokens=len(slots))


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


def test_current_full_graph_metadata_ignores_operator_callbacks(runtime):
    bind(runtime, computed=19)
    item = metadata(torch.tensor([147]), torch.tensor([[1]]), [20])
    context = NS(cudagraph_runtime_mode="FULL", attn_metadata={"layer0": item, "layer1": item})
    runner = NS(input_batch=NS(req_ids=["r"]))
    runtime.wrap_forward(runner, lambda n: runtime.write("layer0", torch.tensor([142])), lambda: context)(1)
    assert runtime.state.blocks[0, 1].valid_end == 20
    assert not any(e["event"] == "guard.violation" for e in runtime.events)


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


def test_missing_layer_metadata_does_not_advance_group_validity(runtime):
    bind(runtime)
    item = metadata(torch.tensor([128]), torch.tensor([[1]]), [1])
    context = NS(cudagraph_runtime_mode="NONE", attn_metadata={"layer0": item})
    runner = NS(input_batch=NS(req_ids=["r"]))
    runtime.wrap_forward(runner, lambda n: None, lambda: context)(1)
    assert any(e.get("reason") == "metadata_write_mapping_missing:layer1" for e in runtime.events)
    assert runtime.state.blocks[0, 1].valid_end == 0


def test_operator_read_before_write_cannot_be_legalized_by_later_write(runtime):
    bind(runtime)
    table = torch.tensor([[1]])
    runner, context = NS(input_batch=NS(req_ids=["r"])), NS(cudagraph_runtime_mode="NONE")

    def model(n):
        runtime.read("layer0", table, [1])
        runtime.write("layer0", torch.tensor([128]))

    runtime.wrap_forward(runner, model, lambda: context)(1)
    assert any(e.get("reason") == "read_unwritten_range" for e in runtime.events)


def test_later_illegal_write_does_not_retroactively_invalidate_earlier_read(runtime):
    bind(runtime, computed=19)
    runner, context = NS(input_batch=NS(req_ids=["r"])), NS(cudagraph_runtime_mode="NONE")

    def model(n):
        runtime.read("layer0", torch.tensor([[1]]), [19])
        runtime.write("layer0", torch.tensor([142]))

    runtime.wrap_forward(runner, model, lambda: context)(1)
    assert any(e.get("reason") == "overwrite_valid_prefix" for e in runtime.events)
    assert not any(e.get("reason") == "read_invalid_block" for e in runtime.events)
    assert runtime.state.blocks[0, 1].valid_end == runtime.subject.INVALID_PREFIX


def test_one_layers_write_cannot_hide_another_layers_missing_write(runtime):
    bind(runtime)
    slots, table = torch.tensor([128]), torch.tensor([[1]])
    item = metadata(slots, table, [1])
    context = NS(cudagraph_runtime_mode="NONE", attn_metadata={"layer0": item, "layer1": item})
    runner = NS(input_batch=NS(req_ids=["r"]))

    def model(n):
        runtime.write("layer0", slots)
        runtime.read("layer0", table, [1])
        runtime.read("layer1", table, [1])

    runtime.wrap_forward(runner, model, lambda: context)(1)
    violation = next(e for e in runtime.events if e.get("reason") == "read_unwritten_range")
    assert violation["layer"] == "layer1" and violation["evidence"] == "operator_arguments"
    assert runtime.state.blocks[0, 1].valid_end == 0


def test_missing_known_operator_layer_is_reported_and_not_committed(runtime):
    bind(runtime)
    item = metadata(torch.tensor([128]), torch.tensor([[1]]), [1])
    context = NS(cudagraph_runtime_mode="NONE", attn_metadata={"layer0": item, "layer1": item})
    runner = NS(input_batch=NS(req_ids=["r"]))
    runtime.wrap_forward(runner, lambda n: runtime.write("layer0", item.slot_mapping), lambda: context)(1)
    assert any(e.get("reason") == "operator_layer_not_observed:layer1" for e in runtime.events)
    assert runtime.state.blocks[0, 1].valid_end == 0


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


def test_eager_uses_operator_arguments_without_building_unused_plans(runtime, monkeypatch):
    bind(runtime)
    slots, table = torch.tensor([128]), torch.tensor([[1]])
    item = metadata(slots, table, [1])
    context = NS(cudagraph_runtime_mode="NONE", attn_metadata={"layer0": item, "layer1": item})
    runner = NS(input_batch=NS(req_ids=["r"]))

    def unused_plan(*args):
        raise AssertionError("eager must not materialize an unused metadata plan")

    monkeypatch.setattr(runtime, "_metadata_accesses", unused_plan)

    def model(n):
        for layer in ("layer0", "layer1"):
            runtime.write(layer, slots)
            runtime.read(layer, table, [1])

    runtime.wrap_forward(runner, model, lambda: context)(1)
    assert runtime.state.blocks[0, 1].valid_end == 1
    assert not any(e["event"] in ("guard.violation", "guard.coverage_gap") for e in runtime.events)


def test_full_metadata_skips_redundant_hooks_and_builds_one_shared_plan(runtime, monkeypatch):
    bind(runtime)
    slots, table = torch.tensor([128]), torch.tensor([[1]])
    item = metadata(slots, table, [1])
    context = NS(cudagraph_runtime_mode="FULL", attn_metadata={"layer0": item, "layer1": item})
    runner = NS(input_batch=NS(req_ids=["r"]))
    parsed = []
    original = runtime._metadata_accesses

    def parse(layer, value):
        parsed.append(layer)
        return original(layer, value)

    monkeypatch.setattr(runtime, "_metadata_accesses", parse)

    def replay(n):
        for layer in ("layer0", "layer1"):
            runtime.write(layer, slots)
            runtime.read(layer, table, [1])
        assert not runtime._accesses

    runtime.wrap_forward(runner, replay, lambda: context)(1)
    assert len(parsed) == 1
    assert runtime.state.blocks[0, 1].valid_end == 1
    assert not any(e["event"] in ("guard.violation", "guard.coverage_gap") for e in runtime.events)


def test_unique_layer_patterns_share_owner_snapshot_and_reuse_checker(runtime, monkeypatch):
    bind(runtime)
    slots, table = torch.tensor([128]), torch.tensor([[1]])
    context = NS(cudagraph_runtime_mode="NONE")
    runner = NS(input_batch=NS(req_ids=["r"]))
    owner_copies = []
    original = runtime.subject.replace
    scratch = runtime._scratch

    def copied(block, **fields):
        if "owners" in fields:
            owner_copies.append(block)
        return original(block, **fields)

    monkeypatch.setattr(runtime.subject, "replace", copied)

    def unexpected_checker(*args):
        raise AssertionError("the scratch checker must be reused")

    monkeypatch.setattr(runtime.subject, "KVGuard", unexpected_checker)

    def model(n):
        for layer in ("layer0", "layer1"):
            runtime.write(layer, slots.clone())
            runtime.read(layer, table, [1])

    runtime.wrap_forward(runner, model, lambda: context)(1)
    assert runtime._scratch is scratch and len(owner_copies) == 1
    assert runtime.state.blocks[0, 1].valid_end == 1
    assert not any(e["event"] in ("guard.violation", "guard.coverage_gap") for e in runtime.events)


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
    assert runtime._missing_groups == {0, 1}


@pytest.mark.parametrize("mode", ["NONE", "FULL"])
@pytest.mark.parametrize("swapped", [False, True])
def test_scheduler_plan_detects_cross_request_slot_swaps(runtime, mode, swapped):
    bind(runtime, "A", 1)
    bind(runtime, "B", 2)
    slots = torch.tensor([256, 128] if swapped else [128, 256])
    table = torch.tensor([[1], [2]])
    item = metadata(slots, table, [1, 1])
    context = NS(cudagraph_runtime_mode=mode, attn_metadata={"layer0": item, "layer1": item})
    runner = NS(input_batch=NS(req_ids=["A", "B"]))
    output = NS(_ascend_kv_guard_context={"requests": {}, "scheduled_requests": {"A": (0, 1), "B": (0, 1)}})

    def model(n):
        for layer in runtime.layer_groups:
            runtime.write(layer, slots)
            runtime.read(layer, table, [1, 1])

    forward = runtime.wrap_forward(runner, model, lambda: context)
    runtime.wrap_execute(lambda message: forward(2))(output)
    violations = [e for e in runtime.events if e["event"] == "guard.violation"]
    if swapped:
        mismatch = next(e for e in violations if e["reason"] == "write_target_mismatch")
        assert mismatch["request_id"] == "A" and mismatch["slot"] == 256 and mismatch["expected_slot"] == 128
        assert all(b.valid_end == runtime.subject.INVALID_PREFIX for b in runtime.state.blocks.values())
    else:
        assert not violations
    assert runtime._scheduled is None


def test_split_mixed_batch_uses_explicit_token_offset_and_runner_request_order(runtime):
    bind(runtime, "prefill", 2, 10)
    bind(runtime, "decode", 1, 19)
    runtime._scheduled = {"prefill": (10, 2), "decode": (19, 1)}
    runner = NS(input_batch=NS(req_ids=["decode", "prefill"]))
    context = NS(cudagraph_runtime_mode="NONE")

    def model(n):
        runtime.write("layer0", torch.tensor([266, 267]), token_offset=1)
        runtime.write("layer0", torch.tensor([147]))
        runtime.read("layer0", torch.tensor([[1], [2]]), [20, 12])

    runtime.wrap_forward(runner, model, lambda: context)(3)
    assert not any(e["event"] in ("guard.violation", "guard.coverage_gap") for e in runtime.events)


@pytest.mark.parametrize(
    "slots, mismatch", [([128, 129, -1], False), ([129, 128, -1], True), ([128, -1, -1], True), ([128, 129, 256], True)]
)
def test_exact_token_positions_distinguish_permutation_missing_write_and_padding(runtime, slots, mismatch):
    bind(runtime)
    runtime._scheduled = {"r": (0, 2)}
    runner = NS(input_batch=NS(req_ids=["r"]))
    context = NS(cudagraph_runtime_mode="NONE")
    runtime.wrap_forward(runner, lambda n: runtime.write("layer0", torch.tensor(slots)), lambda: context)(3)
    assert any(e.get("reason") == "write_target_mismatch" for e in runtime.events) == mismatch


def test_missing_scheduler_range_does_not_advance_validity(runtime):
    bind(runtime)
    runtime._scheduled = {"r": (None, 1)}
    runner = NS(input_batch=NS(req_ids=["r"]))
    context = NS(cudagraph_runtime_mode="NONE")
    runtime.wrap_forward(runner, lambda n: runtime.write("layer0", torch.tensor([128])), lambda: context)(1)
    assert any(e.get("reason") == "scheduler_write_range_missing" and e["request_id"] == "r" for e in runtime.events)
    assert runtime.state.blocks[0, 1].valid_end == 0


def test_raise_mode_stops_output_and_remains_latched_after_caller_catches(runtime):
    bind(runtime, computed=19)
    runtime.action = "raise"
    runtime.state.emit = runtime.report
    runner = NS(input_batch=NS(req_ids=[]))
    context = NS(cudagraph_runtime_mode="NONE")
    completed = []

    def model(n):
        runtime.write("layer0", torch.tensor([142]))
        completed.append("kernel submitted")
        return "must not escape"

    forward = runtime.wrap_forward(runner, model, lambda: context)
    with pytest.raises(runtime.subject.KVGuardError, match="dummy_write_to_live_block"):
        forward(1, _kv_guard_phase="dummy")
    assert completed == ["kernel submitted"]  # Auditing cannot undo submitted operations.
    with pytest.raises(runtime.subject.KVGuardError):
        forward(1, _kv_guard_phase="dummy")
    assert completed == ["kernel submitted"] and runtime._phase is None


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


def test_read_block_order_within_one_request_is_checked_against_scheduler(runtime):
    runtime.state.bind(
        {"pool_id": "pool", "requests": {"r": [[(1, 1), (2, 1), (3, 1)]]}, "scheduled_requests": {"r": (256, 1)}}
    )
    runtime._scheduled = {"r": (256, 1)}
    runner = NS(input_batch=NS(req_ids=["r"]))
    context = NS(cudagraph_runtime_mode="NONE")
    runtime.wrap_forward(runner, lambda n: runtime.read("layer0", torch.tensor([[2, 1]]), [256]), lambda: context)(1)
    assert any(e.get("reason") == "read_target_mismatch" for e in runtime.events)


def test_shared_prefix_read_and_tail_append_remain_legal(runtime):
    runtime.state.bind(
        {
            "pool_id": "pool",
            "requests": {"A": [[(1, 1), (2, 1)]], "B": [[(1, 1), (3, 1)]]},
            "scheduled_requests": {"A": (128, 1), "B": (128, 1)},
        }
    )
    runtime._scheduled = {"A": (128, 1), "B": (128, 1)}
    runner = NS(input_batch=NS(req_ids=["B", "A"]))
    context = NS(cudagraph_runtime_mode="NONE")

    def model(n):
        runtime.write("layer0", torch.tensor([384, 256]))
        runtime.read("layer0", torch.tensor([[1, 3], [1, 2]]), [129, 129])

    runtime.wrap_forward(runner, model, lambda: context)(2)
    assert not any(e["event"] in ("guard.violation", "guard.coverage_gap") for e in runtime.events)


def test_truncated_write_observation_is_incomplete_even_if_read_only_uses_old_prefix(runtime):
    bind(runtime, computed=19)
    runtime._scheduled = {"r": (19, 2)}
    runner = NS(input_batch=NS(req_ids=["r"]))
    context = NS(cudagraph_runtime_mode="NONE")

    def model(n):
        runtime.write("layer0", torch.tensor([147]))
        runtime.read("layer0", torch.tensor([[1]]), [19])

    runtime.wrap_forward(runner, model, lambda: context)(2)
    assert any(e.get("reason") == "write_tokens_not_observed" for e in runtime.events)
    assert runtime.state.blocks[0, 1].valid_end == 19


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
