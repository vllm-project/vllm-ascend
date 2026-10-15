# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU checks of metadata preparation and the surrounding worker control flow.

Run with --noconftest. Production builders run on CPU with emulated external
operators; framework parents are mocked. This is not ACL capture or NPU evidence.
"""

import ast
import sys
from collections import namedtuple
from contextlib import contextmanager, nullcontext
from copy import copy
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from tests.ut.attention.test_flashmla_dcp import addresses, common

pytest_plugins = ("tests.ut.attention.test_flashmla_dcp",)

ROOT = Path(__file__).resolve().parents[3]
ATTN = "vllm_ascend/worker/v2/attn_utils.py"
RUNNER = "vllm_ascend/worker/v2/model_runner.py"
DRAFT = "vllm_ascend/worker/v2/spec_decode/dspark/speculator.py"
MLA = "vllm_ascend/attention/mla_v1.py"


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


def groups_for(state):
    return [[SimpleNamespace(get_metadata_builder=Mock(return_value=state), layer_names=["layer"])]]


@pytest.fixture
def wrapper():
    """Execute the production wrappers with observable framework parent calls."""
    events = []

    @contextmanager
    def context(name):
        events.append(f"{name}_enter")
        try:
            yield
        finally:
            events.append(f"{name}_exit")

    class Parent:
        parent_call: Mock

        def execute_model(self, *args, **kwargs):
            return self.parent_call(*args, **kwargs)

        def propose(self, *args, **kwargs):
            return self.parent_call(*args, **kwargs)

        def capture(self, *args, **kwargs):
            return self.parent_call(*args, **kwargs)

        def set_attn(self, model_state, cache, *args):
            events.append("parent_set_attn")
            self.attn_groups = cache.groups
            self._context_slot_mappings = SimpleNamespace(to=lambda _: object())

    scope = {
        "Parent": Parent,
        "ModelWithContext": lambda model, **kwargs: model,
        "communicator_switch": lambda: context("communicator"),
        "Any": Any,
        "IntermediateTensors": type("IntermediateTensors", (), {}),
        "cast": cast,
        "AttentionLayerBase": object,
        "set_current_vllm_config": lambda _: context("config"),
        "get_layers_from_vllm_config": lambda *args: {},
        "_get_graph_update_backend": lambda _: object,
        "AscendMLABackend": type("MLA", (), {}),
        "AscendAttentionBackend": type("GQA", (), {}),
        "dflash_speculator": SimpleNamespace(),
        "prepare_dflash_inputs_factory": Mock(),
        "nullcontext": nullcontext,
        "_start_profiling_chunk_timing": lambda *args: None,
        "_finish_profiling_chunk_timing": lambda *args: None,
        "has_kv_transfer_group": lambda: False,
        "should_skip_allreduce_across_dp_group": lambda _: True,
        "skip_dp_coordination": lambda: context("dp"),
        "lmhead_tp_enable": lambda: False,
        "torch": torch,
        "build_attn_metadata_wrapper": lambda: context("metadata"),
        "build_attn_metadata_factory": lambda *args, **kwargs: context("factory"),
    }
    for name, path, owner, methods in (
        ("Runner", RUNNER, "NPUModelRunner", ["execute_model"]),
        ("Capture", "vllm_ascend/worker/v2/aclgraph_utils.py", "ModelAclGraphManager", ["capture"]),
        ("Draft", DRAFT, "AscendDSparkSpeculator", ["propose", "set_attn"]),
    ):
        nodes = [source_node(path, method, owner) for method in methods]
        for node in nodes:
            node.decorator_list = []
        compile_nodes([class_node(name, [ast.Name(id="Parent", ctx=ast.Load())], nodes)], scope)

    return SimpleNamespace(classes=scope, events=events)


def make_wrapper(wrapper, kind):
    obj = wrapper.classes[kind]()
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


def call_wrapper(obj, kind, **flags):
    if kind == "Runner":
        return obj.execute_model(SimpleNamespace(total_num_scheduled_tokens=1, num_scheduled_tokens={}), **flags)
    batch = SimpleNamespace(is_prefilling_np=np.array([False]))
    return obj.propose(batch, *([None] * 10), dp_sync="target-dp", **flags)


@pytest.mark.parametrize("kind", ["Runner", "Draft"])
def test_initial_profile_before_set_attn(wrapper, kind):
    obj = make_wrapper(wrapper, kind)
    assert not hasattr(obj, "attn_groups")
    assert call_wrapper(obj, kind, dummy_run=True, is_profile=True, skip_attn_for_dummy_run=True) == "output"
    obj.parent_call.assert_called_once()
    assert not hasattr(obj, "attn_groups")
    if kind == "Draft":
        assert obj.parent_call.call_args.args[11] is None  # Profiling discards target DP counts.
    else:
        obj.model_state.finish_execution.assert_called_once_with(failed=False)
        assert wrapper.events == ["dp_enter", "dp_exit"]


@pytest.mark.parametrize("kind", ["Runner", "Draft"])
@pytest.mark.parametrize("external", [False, True])
@pytest.mark.parametrize("fail", [False, True])
def test_forward_keeps_backend_state_and_dependency_cleanup(wrapper, runtime, kind, external, fail):
    state = runtime.builder(size=1) if external else None
    obj = make_wrapper(wrapper, kind)
    obj.attn_groups = groups_for(state)

    def forward(*args, **kwargs):
        if state is not None:
            flash = state.build(common([11], [1]), 1, 1, False)
            assert flash.cache_lens[0] == 11
            assert not state.defer and state.take_tasks() == ()
        if fail:
            raise ValueError("forward failed")
        return "output"

    obj.parent_call.side_effect = forward
    with pytest.raises(ValueError, match="forward failed") if fail else nullcontext():
        call_wrapper(obj, kind)
    # Generic execution does not rediscover or temporarily mutate builders.
    obj.attn_groups[0][0].get_metadata_builder.assert_not_called()
    if state is not None:
        assert not state.defer and state.take_tasks() == ()
    if kind == "Runner":
        obj.model_state.finish_execution.assert_called_once_with(failed=fail)
        assert wrapper.events == ["dp_enter", "dp_exit"]
        assert obj._maybe_send_draft_kv.call_count == (0 if fail else 1)


@pytest.mark.parametrize("external", [False, True])
def test_profile_set_attn_capture_propose_keep_target_and_draft_independent(wrapper, runtime, external):
    target = runtime.builder(size=1)
    target_flash = target.build(common([17], [1], tokens=4), 1, 1, False, retain_for_graph=True)
    target_snapshot = target_flash.schedule.clone()
    draft = runtime.builder(size=1) if external else None
    obj = make_wrapper(wrapper, "Draft")
    call_wrapper(obj, "Draft", dummy_run=True, is_profile=True, skip_attn_for_dummy_run=True)
    assert not hasattr(obj, "attn_groups")
    wrapper.events.clear()
    cache = SimpleNamespace(groups=groups_for(draft), kv_cache_groups=[])
    obj.set_attn(None, cache, None, None, None)
    assert obj.attn_groups is cache.groups
    assert wrapper.events == ["config_enter", "parent_set_attn", "config_exit"]
    seen = []

    def consume(*args, **kwargs):
        if draft is not None:
            flash = draft.build(
                common([23 + len(seen)], [1], tokens=4, causal=False), 1, 1, False, retain_for_graph=True
            )
            seen.append(flash)
            assert flash.cache_lens[0] == 22 + len(seen)
            assert not draft.defer and draft.take_tasks() == ()
        return "output"

    obj.parent_call.side_effect = consume
    obj.capture()  # The framework parent owns capture after set_attn.
    call_wrapper(obj, "Draft")
    cache.groups[0][0].get_metadata_builder.assert_not_called()
    torch.testing.assert_close(target_flash.schedule, target_snapshot)
    if external:
        assert seen[0] is seen[1]
        assert seen[0].schedule.data_ptr() != target_flash.schedule.data_ptr()
    assert wrapper.events[-4:] == ["metadata_enter", "factory_enter", "factory_exit", "metadata_exit"]


@pytest.mark.parametrize("fail", [False, True])
def test_target_capture_keeps_dependency_cleanup(wrapper, runtime, fail):
    state = runtime.builder(size=1)
    groups = groups_for(state)
    manager = make_wrapper(wrapper, "Capture")
    manager.model_runner = SimpleNamespace(pcp_manager=None)
    device_metadata = SimpleNamespace(finish=Mock())

    def capture(*args, **kwargs):
        flash = state.build(common([9], [1]), 1, 1, False, retain_for_graph=True)
        assert flash.cache_lens[0] == 9
        assert not state.defer and state.take_tasks() == ()
        if fail:
            raise ValueError("capture failed")

    manager.parent_call.side_effect = capture
    with pytest.raises(ValueError, match="capture failed") if fail else nullcontext():
        manager.capture(None, SimpleNamespace(device_metadata=device_metadata), None, None, None, groups, None)
    device_metadata.finish.assert_called_once()
    groups[0][0].get_metadata_builder.assert_not_called()
    assert wrapper.events == ["communicator_enter", "communicator_exit"]


@pytest.mark.parametrize("replicated", [False, True])
@pytest.mark.parametrize("adaptive", [False, True])
@pytest.mark.parametrize("profile", [False, True])
def test_draft_preserves_dependency_dp_coordination(wrapper, replicated, adaptive, profile):
    obj = make_wrapper(wrapper, "Draft")
    obj.replicated_pcp = replicated
    obj.enable_adaptive_verification = adaptive
    call_wrapper(obj, "Draft", is_profile=profile)
    expected = None if replicated or (adaptive and not profile) else "target-dp"
    assert obj.parent_call.call_args.args[11] == expected


@pytest.fixture
def metadata_build(runtime):
    """Route the public per-group build through the real external MLA branch."""
    ignored = type("OtherBackend", (), {})
    scope = {
        "torch": torch,
        "np": np,
        "copy": copy,
        "ring_state_update_skipped": lambda: False,
        "uses_turboquant_groups": lambda _: False,
        "AscendCommonAttentionMetadata": lambda **kw: SimpleNamespace(_seq_lens_cpu=kw["seq_lens_cpu"], **kw),
        "split_flashmla_requests": runtime.api.split_flashmla_requests,
        "AscendAttentionState": SimpleNamespace(ChunkedPrefill="chunked"),
    }
    for name in (
        "AscendDSAMetadataBuilder",
        "AscendDSACPMetadataBuilder",
        "AscendDSAV41MetadataBuilder",
        "AscendSFAMetadataBuilder",
        "GDNAttentionMetadataBuilder",
    ):
        scope[name] = ignored
    methods = [
        source_node(MLA, name, "AscendMLAMetadataBuilder")
        for name in ("build", "set_num_actual_tokens", "_build_external_flashmla", "build_for_cudagraph_capture")
    ]
    compile_nodes([class_node("Builder", [], methods), source_node(ATTN, "build_attn_metadata")], scope)
    builder = scope["Builder"]()
    builder.flashmla_state = runtime.builder(size=1)
    builder.metadata_cls = SimpleNamespace
    builder.decode_metadata_cls = SimpleNamespace
    builder.reorder_batch_threshold = 4
    builder.dcp_enabled = False
    return SimpleNamespace(build=scope["build_attn_metadata"], builder=builder)


@pytest.mark.parametrize("capture", [False, True])
@pytest.mark.parametrize("external", [False, True])
def test_public_build_prepares_once_per_group_before_return(metadata_build, capture, external):
    builder = (
        metadata_build.builder
        if external
        else SimpleNamespace(build=Mock(return_value="native"), build_for_cudagraph_capture=Mock(return_value="native"))
    )
    groups = groups_for(builder)
    inputs = common([11, 7], [4, 3], tokens=16)
    kwargs = dict(
        attn_groups=groups,
        num_reqs=2,
        num_tokens=16,
        num_actual_tokens=7,
        query_start_loc_gpu=inputs.query_start_loc,
        query_start_loc_cpu=inputs.query_start_loc,
        max_query_len=4,
        seq_lens=inputs.seq_lens,
        max_seq_len=11,
        block_tables=[inputs.block_table_tensor],
        slot_mappings=inputs.slot_mapping.unsqueeze(0),
        positions=inputs.positions,
        is_prefilling=torch.tensor([False, False]),
        kv_cache_config=SimpleNamespace(kv_cache_groups=[object()]),
        for_cudagraph_capture=capture,
    )
    result = metadata_build.build(**kwargs)
    groups[0][0].get_metadata_builder.assert_called_once_with(0)
    if external:
        flash = result["layer"].external_flashmla
        pointers = addresses(flash)
        assert flash.cache_lens.tolist() == [11, 7] + [0] * 7
        assert builder.flashmla_state.take_tasks() == ()
        torch.testing.assert_close(flash.schedule, torch.cat((flash.cache_lens, flash.cu, flash.used_q)))
    else:
        assert result == {"layer": "native"}
        (builder.build_for_cudagraph_capture if capture else builder.build).assert_called_once()
    # A second build in the same execution needs neither a release scope nor a
    # task collector. A captured capacity must already contain its latest input.
    inputs.seq_lens.add_(8)
    latest = metadata_build.build(**dict(kwargs, for_cudagraph_capture=False))
    assert groups[0][0].get_metadata_builder.call_count == 2
    if external:
        refreshed = latest["layer"].external_flashmla
        assert refreshed.cache_lens.tolist() == [19, 15] + [0] * 7
        assert builder.flashmla_state.take_tasks() == ()
        if capture:
            assert refreshed is flash and addresses(refreshed) == pointers
    else:
        assert latest == result


@pytest.mark.parametrize("external", [False, True])
def test_request_order_keeps_release_batch_fields_aligned(external):
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
    state: Any = namedtuple("BatchReqState", fields)(*values)

    class Parent:
        def gather_batch_req_state(self, scheduler_output, dummy_run):
            return state, 4

    method = source_node(RUNNER, "gather_batch_req_state", "NPUModelRunner")
    cls = class_node("Runner", [ast.Name(id="Parent", ctx=ast.Load())], [method])
    scope: dict[str, Any] = {"np": np, "Parent": Parent}
    compile_nodes([source_node("vllm_ascend/attention/flashmla.py", "prepare_flashmla_batch_req_state"), cls], scope)
    runner = scope["Runner"]()
    runner.vllm_config = SimpleNamespace(kv_transfer_config=None)
    runner._uses_flashmla = external
    result, uniform_count = runner.gather_batch_req_state(None, False)
    order = [1, 0, 2] if external else [0, 1, 2]
    assert result.req_ids == [state.req_ids[index] for index in order]
    for field in state._fields:
        expected = getattr(state, field)
        actual = getattr(result, field)
        if isinstance(expected, np.ndarray):
            np.testing.assert_array_equal(actual, expected[order])
        elif field != "req_ids":
            assert actual == expected
    assert uniform_count == (None if external else 4)
