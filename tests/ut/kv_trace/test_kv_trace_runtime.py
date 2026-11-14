# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MRV2 native messages and per-group address checks, without scheduler hooks."""

import importlib.util
import logging
import sys
from pathlib import Path
from types import ModuleType
from types import SimpleNamespace as NS

import pytest
import torch

ROOT = Path(__file__).parents[3]


@pytest.fixture(name="runtime")
def runtime_fixture(monkeypatch):
    for name in ("vllm_ascend", "vllm_ascend.kv_trace", "vllm"):
        package = ModuleType(name)
        package.__path__ = []
        monkeypatch.setitem(sys.modules, name, package)

    def load(name, path):
        spec = importlib.util.spec_from_file_location(name, ROOT / path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
        return module

    load("vllm_ascend.config_utils", "vllm_ascend/config_utils.py")
    modules = {}
    for name in ("config", "state", "connector", "plan", "runtime", "runner"):
        modules[name] = load(f"vllm_ascend.kv_trace.kv_trace_{name}", f"vllm_ascend/kv_trace/kv_trace_{name}.py")
    logger = ModuleType("vllm.logger")
    monkeypatch.setattr(logger, "logger", logging.Logger("kv-trace-test"), raising=False)
    monkeypatch.setitem(sys.modules, "vllm.logger", logger)
    config = NS(
        use_v2_model_runner=True,
        additional_config={"kv_trace": {"enabled": True}},
        parallel_config=NS(tensor_parallel_size=1, pipeline_parallel_size=1),
        scheduler_config=NS(async_scheduling=False),
        model_config=NS(),
        speculative_config=None,
        kv_transfer_config=None,
    )
    runtime = modules["runtime"].get_trace(config)
    runtime.subject = modules["runtime"]
    runtime.runner_subject = modules["runner"]
    runtime.plan_subject = modules["plan"]
    runtime.connector_subject = modules["connector"]
    runtime.layer_groups = {"layer0": 0, "layer1": 0}
    runtime.state.configure({0: 128}, 8)
    runtime.events = []
    original = runtime.report

    def report(event, **fields):
        runtime.events.append({"event": event, **fields})
        original(event, **fields)

    runtime.report = runtime.state.emit = report
    return runtime


def message(bindings=None, computed=None, counts=None, cached=None, finished=(), preempted=(), resumed=()):
    bindings = {"r": [[1]]} if bindings is None else bindings
    computed = computed or {}
    counts = {request: 1 for request in bindings} if counts is None else counts
    cached = cached or {}
    return NS(
        scheduled_new_reqs=[
            NS(req_id=req, block_ids=groups, num_computed_tokens=computed.get(req, 0), replay_start=0)
            for req, groups in bindings.items()
        ],
        scheduled_cached_reqs=NS(
            req_ids=list(cached),
            new_block_ids=[value[0] for value in cached.values()],
            num_computed_tokens=[value[1] for value in cached.values()],
            resumed_req_ids=set(resumed),
        ),
        num_scheduled_tokens=counts,
        total_num_scheduled_tokens=sum(counts.values()),
        finished_req_ids=set(finished),
        preempted_req_ids=set(preempted),
        kv_connector_metadata=None,
        new_block_ids_to_zero=None,
        kv_cache_block_copies=None,
    )


def metadata(slots, tables, lengths):
    return NS(
        slot_mapping=torch.tensor(slots),
        block_tables=torch.tensor(tables),
        seq_lens=torch.tensor(lengths),
        num_actual_tokens=len(slots),
    )


def reasons(runtime):
    return {event.get("reason") for event in runtime.events if event["event"] == "kv_trace.violation"}


def plan(runtime, output=None, slots=(128,), tables=((1,),), lengths=(1,), ids=("r",), phase="forward"):
    output = message() if output is None else output
    scheduled = runtime.state.bind(output)
    return runtime.plan_subject.AccessPlan(
        phase=phase,
        request_ids=list(ids),
        scheduled=scheduled,
        metadata={0: metadata(slots, tables, lengths)},
        full_graph=False,
        bindings=dict(runtime.state.requests),
    )


def audit(runtime, item):
    runtime.plan_subject.audit_access_plan(runtime.state, item, runtime.gap)


def test_public_scheduler_output_needs_no_private_context(runtime):
    output = message()
    assert not hasattr(output, "_ascend_kv_trace_context")
    with runtime.execute_scope(output):
        assert runtime._scheduled == {"r": (0, 1)}
    assert runtime.state.requests == {"r": ((1,),)}
    assert runtime._scheduled is None


@pytest.mark.parametrize("options", [{"enabled": "false"}, {"enabled": 1}, {"action": "stop"}, {"extra": True}])
def test_invalid_config_is_rejected(runtime, options):
    with pytest.raises((TypeError, ValueError)):
        runtime.subject.get_trace(NS(additional_config={"kv_trace": options}))


@pytest.mark.parametrize("options", [None, {"enabled": False}])
def test_disabled_configuration_creates_no_trace(runtime, options):
    config = NS(additional_config={} if options is None else {"kv_trace": options})
    assert runtime.subject.get_trace(config) is None
    assert not hasattr(config, "_ascend_kv_trace")


def test_only_kv_trace_key_enables_the_feature(runtime):
    assert runtime.subject.get_trace(NS(additional_config={"kv_guard": {"enabled": True}})) is None
    assert runtime.subject.get_trace(runtime.config) is runtime


@pytest.mark.parametrize(
    "slots,tables,lengths,reason",
    [
        ([129], [[1]], [1], "write_target_mismatch"),
        ([256], [[1]], [1], "write_target_mismatch"),
        ([128], [[2]], [1], "read_target_mismatch"),
        ([128], [[1]], [2], "read_length_mismatch"),
        ([128, 256], [[1]], [1], "write_target_mismatch"),
        ([], [[1]], [1], "write_target_mismatch"),
        ([1024], [[1]], [1], "write_outside_cache"),
        ([0], [[1]], [1], "write_null_block"),
        ([-2], [[1]], [1], "write_outside_cache"),
    ],
)
def test_address_plan_violations(runtime, slots, tables, lengths, reason):
    audit(runtime, plan(runtime, slots=slots, tables=tables, lengths=lengths))
    assert reason in reasons(runtime)


def test_shared_prefix_is_valid_without_inventing_allocation_epochs(runtime):
    output = message(bindings={"A": [[1, 2]], "B": [[1, 3]]}, computed={"A": 128, "B": 128}, counts={"A": 1, "B": 1})
    audit(runtime, plan(runtime, output, slots=[256, 384], tables=[[1, 2], [1, 3]], lengths=[129, 129], ids=["A", "B"]))
    assert not reasons(runtime)
    assert not hasattr(runtime.state, "blocks")
    assert {event["event"] for event in runtime.events} >= {"kv_trace.bind", "kv_trace.write", "kv_trace.read"}


def test_snapshot_uses_its_original_bindings_after_next_message(runtime):
    first = plan(runtime)
    runtime.state.bind(message(bindings={"r": [[2]]}))
    audit(runtime, first)
    assert not reasons(runtime)
    assert first.bindings == {"r": ((1,),)}


def test_graph_padding_rows_and_pad_slots_are_not_real_requests(runtime):
    audit(runtime, plan(runtime, slots=[128, -1], tables=[[1], [0]], lengths=[1, 0]))
    assert not reasons(runtime)


@pytest.mark.parametrize(
    "slot,allow,invalid", [(-1, False, False), (128, False, True), (0, True, False), (128, True, True)]
)
def test_dummy_pad_rule(runtime, slot, allow, invalid):
    item = plan(runtime, slots=[slot], phase="dummy")
    item.allow_null_block = allow
    audit(runtime, item)
    assert ("dummy_write_target" in reasons(runtime)) is invalid


def test_missing_metadata_does_not_become_a_violation(runtime):
    item = plan(runtime)
    item.metadata.clear()
    runtime.action = "raise"
    audit(runtime, item)
    runtime._check_failure()
    assert not reasons(runtime)


def test_raise_latches_a_real_plan_violation(runtime):
    runtime.action = "raise"
    audit(runtime, plan(runtime, slots=[256]))
    with pytest.raises(runtime.subject.KVTraceError, match="write_target_mismatch"):
        runtime._check_failure()
