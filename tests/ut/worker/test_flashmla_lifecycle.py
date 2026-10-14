# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU control-flow regressions; framework parents and device work are mocked.

Run with --noconftest. These execute production wrappers and
scope bodies, not full worker startup, ACL capture, or NPU kernels.
"""

import ast
import sys
from collections import namedtuple
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import Mock

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
ATTN = "vllm_ascend/worker/v2/attn_utils.py"
RUNNER = "vllm_ascend/worker/v2/model_runner.py"
DRAFT = "vllm_ascend/worker/v2/spec_decode/dspark/speculator.py"


def source_node(path, name, owner=None):
    body = ast.parse((ROOT / path).read_text()).body
    if owner:
        body = next(n for n in body if isinstance(n, ast.ClassDef) and n.name == owner).body
    return next(n for n in body if isinstance(n, ast.FunctionDef) and n.name == name)


def compile_nodes(nodes, scope):
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    exec(
        compile(ast.fix_missing_locations(ast.Module(body=[future, *nodes], type_ignores=[])), str(ROOT), "exec"), scope
    )


def class_node(name: str, bases: list[ast.expr], body: list[ast.stmt]) -> ast.ClassDef:
    if sys.version_info >= (3, 12):
        return ast.ClassDef(name=name, bases=bases, keywords=[], body=body, decorator_list=[], type_params=[])
    return ast.ClassDef(name=name, bases=bases, keywords=[], body=body, decorator_list=[])


@pytest.fixture
def binding():
    scope = {"contextmanager": contextmanager}
    compile_nodes([source_node(ATTN, "flashmla_metadata_scope")], scope)
    return scope["flashmla_metadata_scope"]


def groups_for(state):
    return [[SimpleNamespace(get_metadata_builder=lambda _: SimpleNamespace(flashmla_state=state))]]


def test_disabled_scope_does_not_read_uninitialized_groups(binding):
    with binding(None, None):
        pass


@pytest.mark.parametrize("fail", [False, True])
def test_scope_releases_and_restores_after_forward(binding, fail):
    old = object()
    state = SimpleNamespace(executor=old, defer=False)
    executor = SimpleNamespace(submission_in_flight=True, release=Mock())
    with pytest.raises(ValueError) if fail else nullcontext(), binding(groups_for(state), executor):
        assert state.executor is executor and state.defer
        if fail:
            raise ValueError("forward failed")
    executor.release.assert_called_once()
    assert state.executor is old and not state.defer


def test_nested_same_executor_releases_only_at_owner_exit(binding):
    state = SimpleNamespace(executor=None, defer=False)
    executor = SimpleNamespace(submission_in_flight=True, release=Mock())
    groups = groups_for(state)
    with binding(groups, executor):
        with binding(groups, executor):
            assert state.executor is executor
        executor.release.assert_not_called()
        assert state.executor is executor and state.defer
    executor.release.assert_called_once()
    assert state.executor is None and not state.defer


@pytest.mark.parametrize("failure", ["builder", "release"])
def test_scope_restores_after_setup_or_release_failure(binding, failure):
    old = object()
    state = SimpleNamespace(executor=old, defer=False)
    executor = SimpleNamespace(submission_in_flight=True, release=Mock())
    groups = groups_for(state)
    if failure == "builder":
        groups[0].append(SimpleNamespace(get_metadata_builder=Mock(side_effect=ValueError("builder failed"))))
    else:
        executor.release.side_effect = ValueError("release failed")
    with pytest.raises(ValueError, match=failure), binding(groups, executor):
        pass
    assert state.executor is old and not state.defer


@pytest.fixture
def wrapper(binding):
    events = []

    @contextmanager
    def dp_context():
        events.append("dp_enter")
        try:
            yield
        finally:
            events.append("dp_exit")

    class Parent:
        parent_call: Mock

        def execute_model(self, *args, **kwargs):
            return self.parent_call(*args, **kwargs)

        def propose(self, *args, **kwargs):
            return self.parent_call(*args, **kwargs)

        def capture(self, *args, **kwargs):
            return self.parent_call(*args, **kwargs)

        def set_attn(self, model_state, cache, *args):
            self.attn_groups = cache.groups
            self._context_slot_mappings = SimpleNamespace(to=lambda _: object())

    scope = {
        "Parent": Parent,
        "ModelWithContext": lambda model, **kwargs: model,
        "communicator_switch": nullcontext,
        "Any": Any,
        "IntermediateTensors": type("IntermediateTensors", (), {}),
        "cast": cast,
        "AttentionLayerBase": object,
        "set_current_vllm_config": lambda _: nullcontext(),
        "get_layers_from_vllm_config": lambda *args: {},
        "_get_graph_update_backend": lambda _: object,
        "AscendMLABackend": type("MLA", (), {}),
        "AscendAttentionBackend": type("GQA", (), {}),
        "dflash_speculator": SimpleNamespace(),
        "prepare_dflash_inputs_factory": Mock(),
        "DeviceMetadataExecutor": lambda: SimpleNamespace(submission_in_flight=False, release=Mock()),
        "flashmla_metadata_scope": binding,
        "nullcontext": nullcontext,
        "_start_profiling_chunk_timing": lambda *args: None,
        "_finish_profiling_chunk_timing": lambda *args: None,
        "has_kv_transfer_group": lambda: False,
        "should_skip_allreduce_across_dp_group": lambda _: True,
        "skip_dp_coordination": dp_context,
        "lmhead_tp_enable": lambda: False,
        "torch": SimpleNamespace(from_numpy=lambda x: x, int32="int32"),
        "build_attn_metadata_wrapper": nullcontext,
        "build_attn_metadata_factory": lambda *args, **kwargs: nullcontext(),
    }
    for name, path, owner, methods in (
        ("Runner", RUNNER, "NPUModelRunner", ["execute_model"]),
        ("Capture", "vllm_ascend/worker/v2/aclgraph_utils.py", "ModelAclGraphManager", ["capture"]),
        ("Draft", DRAFT, "AscendDSparkSpeculator", ["propose", "capture", "set_attn"]),
    ):
        nodes = [source_node(path, method, owner) for method in methods]
        for node in nodes:
            node.decorator_list = []
        cls = class_node(name, [ast.Name(id="Parent", ctx=ast.Load())], nodes)
        compile_nodes([cls], scope)

    def make(kind, executor):
        obj = scope[kind]()
        obj.flashmla_executor = executor
        obj.parent_call = Mock(return_value="output")
        obj.ascend_config = SimpleNamespace(scheduler_config=SimpleNamespace(profiling_chunk_config=None))
        obj.vllm_config = SimpleNamespace(parallel_config=object(), cache_config=SimpleNamespace(block_size=128))
        obj.draft_attn_layer_names = set()
        obj.attn_vllm_config = obj.vllm_config
        obj.model_state = SimpleNamespace(kvpp_is_dummy_run=False, finish_execution=Mock())
        obj.kvpp = SimpleNamespace(complete_forward=Mock())
        obj._maybe_send_draft_kv = Mock()
        obj._start_dump_data = Mock()
        obj._finalize_dump_data = Mock()
        obj.use_dcp = False
        obj.replicated_pcp = False
        obj.enable_adaptive_verification = False
        obj.attn_architecture = None
        obj.input_buffers = SimpleNamespace(positions=None)
        obj.max_num_tokens = 32
        return obj

    def call(obj, kind, **flags):
        if kind == "Runner":
            return obj.execute_model(SimpleNamespace(total_num_scheduled_tokens=1, num_scheduled_tokens={}), **flags)
        batch = SimpleNamespace(is_prefilling_np=[False])
        return obj.propose(batch, *([None] * 10), dp_sync="target-dp", **flags)

    return SimpleNamespace(make=make, call=call, events=events)


@pytest.mark.parametrize("kind", ["Runner", "Draft"])
@pytest.mark.parametrize("active_executor", [False, True])
def test_initial_profile_before_set_attn(wrapper, kind, active_executor):
    executor = SimpleNamespace(submission_in_flight=False) if active_executor else None
    obj = wrapper.make(kind, executor)
    assert not hasattr(obj, "attn_groups")
    assert wrapper.call(obj, kind, dummy_run=True, is_profile=True, skip_attn_for_dummy_run=True) == "output"
    obj.parent_call.assert_called_once()
    assert not hasattr(obj, "attn_groups")
    if kind == "Draft":
        assert obj.parent_call.call_args.args[11] is None  # Profiling discards target DP counts.
    else:
        obj.model_state.finish_execution.assert_called_once_with(failed=False)
        assert wrapper.events == ["dp_enter", "dp_exit"]


@pytest.mark.parametrize("kind", ["Runner", "Draft"])
@pytest.mark.parametrize("missing_flag", ["dummy_run", "is_profile", "skip_attn_for_dummy_run"])
def test_active_executor_rejects_missing_groups_outside_initial_profile(wrapper, kind, missing_flag):
    obj = wrapper.make(kind, SimpleNamespace(submission_in_flight=False))
    flags = dict(dummy_run=True, is_profile=True, skip_attn_for_dummy_run=True)
    flags[missing_flag] = False
    with pytest.raises(RuntimeError, match="outside initial memory profiling"):
        wrapper.call(obj, kind, **flags)
    obj.parent_call.assert_not_called()


@pytest.mark.parametrize("kind", ["Runner", "Draft"])
@pytest.mark.parametrize("fail", [False, True])
def test_initialized_forward_binds_and_restores_with_cleanup(wrapper, kind, fail):
    state = SimpleNamespace(executor=None, defer=False)
    executor = SimpleNamespace(submission_in_flight=True, release=Mock())
    obj = wrapper.make(kind, executor)
    obj.attn_groups = groups_for(state)

    def forward(*args, **kwargs):
        assert state.executor is executor and state.defer
        if fail:
            raise ValueError("forward failed")
        return "output"

    obj.parent_call.side_effect = forward
    with pytest.raises(ValueError) if fail else nullcontext():
        wrapper.call(obj, kind)
    assert state.executor is None and not state.defer
    executor.release.assert_called_once()
    if kind == "Runner":
        obj.model_state.finish_execution.assert_called_once_with(failed=fail)
        assert wrapper.events == ["dp_enter", "dp_exit"]
        assert obj._maybe_send_draft_kv.call_count == (0 if fail else 1)


@pytest.mark.parametrize("fail", [False, True])
def test_draft_capture_binds_and_restores(wrapper, fail):
    state = SimpleNamespace(executor=None, defer=False)
    executor = SimpleNamespace(submission_in_flight=True, release=Mock())
    obj = wrapper.make("Draft", executor)
    obj.attn_groups = groups_for(state)

    def capture():
        assert state.executor is executor and state.defer
        if fail:
            raise ValueError("capture failed")

    obj.parent_call.side_effect = capture
    with pytest.raises(ValueError) if fail else nullcontext():
        obj.capture()
    assert state.executor is None and not state.defer
    executor.release.assert_called_once()


@pytest.mark.parametrize("external", [False, True])
def test_profile_then_set_attn_then_capture_and_propose(wrapper, external):
    obj = wrapper.make("Draft", None)
    wrapper.call(obj, "Draft", dummy_run=True, is_profile=True, skip_attn_for_dummy_run=True)
    assert not hasattr(obj, "attn_groups")
    state = SimpleNamespace(executor=None, defer=False) if external else None
    cache = SimpleNamespace(groups=groups_for(state), kv_cache_groups=[])
    obj.set_attn(None, cache, None, None, None)
    assert (obj.flashmla_executor is not None) is external

    def consume(*args, **kwargs):
        if external:
            assert state is not None
            assert state.executor is obj.flashmla_executor and state.defer
        return "output"

    obj.parent_call.side_effect = consume
    obj.capture()
    wrapper.call(obj, "Draft")
    if external:
        assert state is not None
        assert state.executor is None and not state.defer


@pytest.mark.parametrize("fail", [False, True])
def test_target_capture_keeps_dependency_cleanup(wrapper, fail):
    state = SimpleNamespace(executor=None, defer=False)
    executor = SimpleNamespace(submission_in_flight=True, release=Mock())
    manager = wrapper.make("Capture", executor)
    manager.model_runner = SimpleNamespace(flashmla_executor=executor, pcp_manager=None)
    device_metadata = SimpleNamespace(finish=Mock())

    def capture(*args, **kwargs):
        assert state.executor is executor and state.defer
        if fail:
            raise ValueError("capture failed")

    manager.parent_call.side_effect = capture
    with pytest.raises(ValueError) if fail else nullcontext():
        manager.capture(
            None, SimpleNamespace(device_metadata=device_metadata), None, None, None, groups_for(state), None
        )
    device_metadata.finish.assert_called_once()
    executor.release.assert_called_once()
    assert state.executor is None and not state.defer


@pytest.mark.parametrize("replicated", [False, True])
@pytest.mark.parametrize("adaptive", [False, True])
@pytest.mark.parametrize("profile", [False, True])
def test_draft_preserves_dependency_dp_coordination(wrapper, replicated, adaptive, profile):
    obj = wrapper.make("Draft", None)
    obj.replicated_pcp = replicated
    obj.enable_adaptive_verification = adaptive
    wrapper.call(obj, "Draft", is_profile=profile)
    expected = None if replicated or (adaptive and not profile) else "target-dp"
    assert obj.parent_call.call_args.args[11] == expected


@pytest.mark.parametrize("contract", ["legacy", "main", "main_without_optional_arrays"])
@pytest.mark.parametrize("external", [False, True])
def test_request_order_keeps_new_batch_fields_aligned(contract, external):
    fields = (
        "req_ids num_scheduled_tokens idx_mapping_np prefill_len_np "
        "num_computed_prefill_tokens_np is_prefilling_np has_prefill num_tokens"
    )
    values = [
        ["prefill-a", "decode", "prefill-b"],
        np.array([1, 4, 1]),
        np.array([10, 20, 30]),
        np.array([100, 200, 300]),
        np.array([11, 22, 33]),
        np.array([True, False, True]),
        True,
        6,
    ]
    if contract != "legacy":
        fields += " num_draft_tokens_np prefill_runs_as_decode_np decode_graph_eligible"
        values += [
            np.array([0, 3, 0]) if contract == "main" else None,
            np.array([True, False, True]) if contract == "main" else None,
            True,
        ]
    state: Any = namedtuple("BatchReqState", fields)(*values)

    class Parent:
        def gather_batch_req_state(self, scheduler_output, dummy_run):
            return state, 4

    method = source_node(RUNNER, "gather_batch_req_state", "NPUModelRunner")
    cls = class_node("Runner", [ast.Name(id="Parent", ctx=ast.Load())], [method])
    scope: dict[str, Any] = {"np": np, "Parent": Parent}
    compile_nodes([cls], scope)
    runner = scope["Runner"]()
    runner.vllm_config = SimpleNamespace(kv_transfer_config=None)
    runner.flashmla_executor = object() if external else None
    result, uniform_count = runner.gather_batch_req_state(None, False)
    order = [1, 0, 2] if external else [0, 1, 2]
    assert result.req_ids == [state.req_ids[index] for index in order]
    for field in state._fields:
        expected = getattr(state, field)
        actual = getattr(result, field)
        if isinstance(expected, np.ndarray):
            np.testing.assert_array_equal(actual, expected[order])
        elif field == "decode_graph_eligible" and external:
            assert actual is False
        elif field != "req_ids":
            assert actual == expected
    assert uniform_count == (None if external else 4)
