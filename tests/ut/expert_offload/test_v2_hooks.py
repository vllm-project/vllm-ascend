# SPDX-License-Identifier: Apache-2.0
"""Exercise actual hook bodies in isolation from optional vLLM/NPU imports.

This verifies lifecycle composition and argument forwarding, not GPUModelRunner
or NPU kernel execution. Full runner tests remain part of the hardware gate.
"""

import ast
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from vllm_ascend.expert_offload.config import ExpertOffloadConfig
from vllm_ascend.expert_offload.expert_offload_manager import ExpertOffloadManager, OffloadState

ROOT = Path(__file__).resolve().parents[3]


def load_hook_class(path, class_name, methods, parent, namespace):
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    node = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    selected = [node for node in node.body if isinstance(node, ast.FunctionDef) and node.name in methods]
    assert len(selected) == len(methods)
    for method in selected:
        method.decorator_list = []
    class_body: list[ast.stmt] = list(selected)
    # Parsing initializes version-specific AST fields, including 3.12 type_params.
    isolated_class = ast.parse(f"class {class_name}(Parent):\n    pass\n").body[0]
    assert isinstance(isolated_class, ast.ClassDef)
    isolated_class.body = class_body
    isolated = ast.Module(
        body=[
            ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
            isolated_class,
        ],
        type_ignores=[],
    )
    namespace = {**namespace, "Parent": parent}
    exec(compile(ast.fix_missing_locations(isolated), str(ROOT / path), "exec"), namespace)
    return namespace[class_name]


def test_graph_manager_filters_padded_candidates_and_subscribes_actual_npu_stream():
    from vllm.config.compilation import CUDAGraphMode

    descs = [SimpleNamespace(num_tokens=n, uniform_token_count=1) for n in (1, 2, 4)]
    cached_stream, actual_stream = object(), object()

    class Parent:
        def __init__(self, *args, **kwargs):
            self.device = args[1]
            self._capture_descs = {CUDAGraphMode.FULL: descs}
            self._candidates = {(n, 0): [desc] for n, desc in zip((1, 2, 3), descs)}

        def needs_capture(self):
            return any(self._capture_descs.values())

        def _capture_stream(self, desc):
            return cached_stream

    manager = ExpertOffloadManager()
    manager.bindings["layer"] = SimpleNamespace(top_k=6, num_slots=16)
    manager.subscribe_graph_stream = Mock()
    runner = SimpleNamespace(expert_offload_manager=manager, update_stream=object())
    graph_params = Mock()
    cls = load_hook_class(
        "vllm_ascend/worker/v2/aclgraph_utils.py",
        "ModelAclGraphManager",
        {"__init__", "_capture_stream"},
        Parent,
        {
            "torch": SimpleNamespace(
                npu=SimpleNamespace(Stream=Mock(), current_stream=Mock(return_value=actual_stream))
            ),
            "collect_sorted_captured_token_sizes": lambda groups: sorted(
                d.num_tokens for ds in groups.values() for d in ds
            ),
            "set_graph_params": graph_params,
        },
    )
    graph = cls(object(), "npu:0", CUDAGraphMode.FULL_DECODE_ONLY, 1, runner)
    assert graph.capture_sizes == [1, 2]
    assert graph._candidates[(3, 0)] == []  # Three actual tokens round up to four.
    graph_params.assert_called_once_with([1, 2])
    assert graph._capture_stream(descs[0]) is actual_stream
    manager.subscribe_graph_stream.assert_called_once_with(actual_stream)


def make_runner(*, manager=None, fail_load=False, fail_forward=False, has_prefill=False):
    events: list[object] = []

    class Parent:
        def load_model(self, load_dummy_weights=False, *args, **kwargs):
            events.append(("parent_load", load_dummy_weights, args, kwargs))
            if fail_load:
                raise RuntimeError("load failed")
            self.model = object()
            self.model_memory_usage = 1000

        def execute_model(self, scheduler_output, **kwargs):
            if manager is not None:
                assert manager.executing
            self.gather_batch_req_state(scheduler_output, kwargs["dummy_run"])
            if fail_forward:
                raise RuntimeError("forward failed")
            return "result"

        def gather_batch_req_state(self, scheduler_output, dummy_run):
            return SimpleNamespace(has_prefill=has_prefill), 1

        def shutdown(self):
            events.append("parent_shutdown")

    runner_class = load_hook_class(
        "vllm_ascend/worker/v2/model_runner.py",
        "NPUModelRunner",
        {"load_model", "execute_model", "gather_batch_req_state", "shutdown"},
        Parent,
        {
            "torch": SimpleNamespace(npu=SimpleNamespace(memory_allocated=Mock(side_effect=[100, 164]))),
            "CUDAGraphMode": SimpleNamespace(NONE=0),
            "IntermediateTensors": type("IntermediateTensors", (), {}),
            "get_pd_dspark_aux_layer_ids": lambda _vc: [9],
            "has_kv_transfer_group": lambda: False,
            "should_skip_allreduce_across_dp_group": lambda _vc: False,
            "nullcontext": nullcontext,
            "_start_profiling_chunk_timing": lambda *_args: None,
            "_finish_profiling_chunk_timing": lambda *_args: None,
            "lmhead_tp_enable": lambda: False,
            "is_pd_decode_recompute_scheduler_enabled": lambda _vc: False,
        },
    )
    runner = runner_class()
    runner.expert_offload_manager = manager
    runner.vllm_config = SimpleNamespace(kv_transfer_config=None)
    runner.device = "cpu"
    runner.compilation_config = SimpleNamespace(cudagraph_mode=0)
    runner._start_dump_data = Mock()
    runner._finalize_dump_data = Mock()
    runner.ascend_config = SimpleNamespace(
        scheduler_config=SimpleNamespace(profiling_chunk_config=SimpleNamespace(need_timing=False))
    )
    runner.model_state = SimpleNamespace(kvpp_is_dummy_run=False, finish_execution=Mock())
    runner._maybe_send_draft_kv = Mock()
    runner.kvpp = SimpleNamespace(complete_forward=Mock())
    runner.adaptive_verification = None
    runner.speculator = None
    return runner, events


def test_load_preserves_parent_contract_and_accounts_pool_memory():
    manager = Mock()
    runner, events = make_runner(manager=manager)
    runner.load_model(False, "extra", reload=False)
    assert events == [("parent_load", False, ("extra",), {"reload": False})]
    manager.finalize_loading.assert_called_once_with(runner.model, None)
    assert runner.model_memory_usage == 1064
    assert runner.pd_dspark_aux_layer_ids == [9]


def test_load_adopts_draft_layers_and_shares_runner_owned_manager():
    manager = Mock()
    runner, _ = make_runner(manager=manager)
    runner.speculator = SimpleNamespace(model=object())
    runner.load_model()
    manager.finalize_loading.assert_called_once_with(runner.model, runner.speculator.model)
    assert runner.speculator.expert_offload_manager is manager


@pytest.mark.parametrize("profile,failed", [(False, False), (True, False), (False, True)])
def test_dspark_propose_owns_draft_lifecycle_and_preserves_parent_arguments(profile, failed):
    manager = ExpertOffloadManager()
    manager.state = OffloadState.READY

    class Parent:
        def propose(self, *args, **kwargs):
            assert manager.executing and not manager.has_prefill
            assert manager.is_profile is profile
            assert kwargs["is_profile"] is profile
            if failed:
                raise RuntimeError("draft failed")
            return args

    cls = load_hook_class(
        "vllm_ascend/worker/v2/spec_decode/dspark/speculator.py",
        "AscendDSparkSpeculator",
        {"propose"},
        Parent,
        {
            "torch": torch,
            "nullcontext": nullcontext,
            "build_attn_metadata_wrapper": nullcontext,
            "build_attn_metadata_factory": lambda *_a, **_k: nullcontext(),
        },
    )
    draft = cls()
    draft.expert_offload_manager = manager
    draft.replicated_pcp = False
    draft.enable_adaptive_verification = False
    draft.use_dcp = False
    draft.input_buffers = SimpleNamespace(positions=object())
    draft.max_num_tokens = 16
    draft.attn_vllm_config = SimpleNamespace(parallel_config=object())
    batch = SimpleNamespace(is_prefilling_np=np.array([True]))
    args = (batch, {}, {}, *[object() for _ in range(8)])
    if failed:
        with pytest.raises(RuntimeError, match="draft failed"):
            draft.propose(*args, is_profile=profile)
    else:
        result = draft.propose(*args, is_profile=profile)
        assert result[:11] == args
    assert not manager.executing and manager.has_prefill
    assert manager.state is (OffloadState.FAILED if failed else OffloadState.READY)


@pytest.mark.parametrize("sizes", [(128, 128), (128, 256)])
def test_dspark_input_preparation_uses_actual_draft_physical_kv_blocks(sizes):
    class Parent:
        def set_attn(self, *_args):
            self.draft_kv_cache_group_ids = [1, 2]

    backend = type("Backend", (), {})
    factory = Mock(return_value=object())
    patched_module = SimpleNamespace()
    cls = load_hook_class(
        "vllm_ascend/worker/v2/spec_decode/dspark/speculator.py",
        "AscendDSparkSpeculator",
        {"set_attn"},
        Parent,
        {
            "set_current_vllm_config": lambda *_a: nullcontext(),
            "Any": object,
            "cast": lambda _t, v: v,
            "AttentionLayerBase": object,
            "get_layers_from_vllm_config": lambda *_a: {},
            "_get_graph_update_backend": lambda *_a: backend,
            "AscendMLABackend": backend,
            "AscendAttentionBackend": backend,
            "dflash_speculator": patched_module,
            "prepare_dflash_inputs_factory": factory,
            "torch": torch,
        },
    )
    draft = cls()
    draft.vllm_config = SimpleNamespace(cache_config=SimpleNamespace(block_size=32))
    draft.attn_vllm_config = object()
    draft._context_slot_mappings = torch.zeros(1, dtype=torch.long)
    draft.draft_attn_layer_names = set()
    draft.attn_groups = []
    cache = SimpleNamespace(
        kv_cache_groups=[
            SimpleNamespace(layer_names=[], kv_cache_spec=SimpleNamespace(block_size=size)) for size in (32, *sizes)
        ]
    )
    if sizes[0] != sizes[1]:
        with pytest.raises(ValueError, match="physical KV block size"):
            draft.set_attn(None, cache, None, None, None)
        factory.assert_not_called()
    else:
        draft.set_attn(None, cache, None, None, None)
        factory.assert_called_once_with(128)
        assert patched_module.prepare_dflash_inputs is factory.return_value


def test_disabled_load_does_not_allocate_or_finalize_offload():
    runner, events = make_runner()
    runner.load_model(True)
    assert events == [("parent_load", True, (), {})]
    assert runner.model_memory_usage == 1000


def test_dummy_load_rejected_before_parent_and_loading_failure_closes_manager():
    manager = Mock()
    runner, events = make_runner(manager=manager)
    with pytest.raises(ValueError, match="dummy"):
        runner.load_model(True)
    assert not events
    runner, _ = make_runner(manager=manager, fail_load=True)
    with pytest.raises(RuntimeError, match="load failed"):
        runner.load_model()
    manager.close.assert_called_once()
    manager.finalize_loading.assert_not_called()


@pytest.mark.parametrize(
    "prefill,profile,dummy", [(False, False, False), (True, False, False), (False, True, False), (False, False, True)]
)
def test_execute_uses_real_batch_phase_and_retires_lifecycle(prefill, profile, dummy):
    manager = ExpertOffloadManager()
    manager.state = OffloadState.READY
    runner, _ = make_runner(manager=manager, has_prefill=prefill)
    scheduler_output = SimpleNamespace(total_num_scheduled_tokens=1, num_scheduled_tokens={"request": 1})
    assert runner.execute_model(scheduler_output, is_profile=profile, dummy_run=dummy) == "result"
    assert not manager.executing and manager.state is OffloadState.READY
    assert manager.has_prefill and not manager.is_profile
    runner.model_state.finish_execution.assert_called_once_with(failed=False)
    runner.kvpp.complete_forward.assert_called_once()


def test_execute_failure_retires_offload_and_existing_metadata():
    manager = ExpertOffloadManager()
    manager.state = OffloadState.READY
    runner, _ = make_runner(manager=manager, fail_forward=True)
    with pytest.raises(RuntimeError, match="forward failed"):
        runner.execute_model(SimpleNamespace(total_num_scheduled_tokens=1, num_scheduled_tokens={"request": 1}))
    assert not manager.executing and manager.state is OffloadState.FAILED
    runner.model_state.finish_execution.assert_called_once_with(failed=True)
    runner._maybe_send_draft_kv.assert_not_called()


def test_shutdown_preserves_parent_even_on_offload_cleanup_failure():
    manager = Mock()
    manager.close.side_effect = RuntimeError("cleanup failed")
    runner, events = make_runner(manager=manager)
    with pytest.raises(RuntimeError, match="cleanup failed"):
        runner.shutdown()
    assert events == ["parent_shutdown"]


@pytest.mark.parametrize("enabled", [False, True])
def test_quant_wrapper_preserves_disabled_allocation_and_bounds_enabled_loading(enabled):
    class Parent:
        requires_device_loading = True

        def __init__(self, moe_config):
            pass

    config = ExpertOffloadConfig(enabled=enabled, num_device_experts=2 if enabled else None)
    binding_factory = Mock()

    def set_attrs(param, attrs):
        for key, value in attrs.items():
            setattr(param, key, value)

    scheme = SimpleNamespace(
        quant_type="mxfp4",
        group_size=32,
        get_weight=lambda *args: {"w13_weight": torch.empty(6, 4, 3), "w2_weight": torch.empty(6, 4, 3)},
        get_dynamic_quant_param=lambda *args: {
            "w13_weight_scale": torch.empty(6, 4, 3),
            "w2_weight_scale": torch.empty(6, 4, 3),
        },
        process_weights_after_loading=Mock(),
    )
    cls = load_hook_class(
        "vllm_ascend/quantization/method_adapters.py",
        "AscendFusedMoEMethod",
        {"__init__", "configure_expert_offload", "create_weights", "_create_weights", "process_weights_after_loading"},
        Parent,
        {
            "torch": torch,
            "nullcontext": nullcontext,
            "set_weight_attrs": set_attrs,
            "ExpertOffloadConfig": ExpertOffloadConfig,
            "LayerOffloadBinding": binding_factory,
            "QuantType": SimpleNamespace(W4A8MXFP="mxfp4"),
            "FusedMoeWeightScaleSupported": SimpleNamespace(
                CHANNEL=SimpleNamespace(value="channel"),
                GROUP=SimpleNamespace(value="group"),
            ),
        },
    )
    method = cls(scheme, object())
    method.configure_expert_offload(config)
    layer = torch.nn.Module()
    layer.top_k = 2
    with torch.device("meta"):
        method.create_weights(layer, 6, 3, 2, torch.float32, weight_loader=object())
    assert layer.w13_weight.device.type == ("cpu" if enabled else "meta")
    assert method.requires_device_loading is not enabled
    if enabled:
        binding_factory.assert_called_once_with(layer, scheme, config)
        method.process_weights_after_loading(layer)
        binding_factory.return_value.process_weights_after_loading.assert_called_once_with(layer)
        scheme.process_weights_after_loading.assert_not_called()
    else:
        binding_factory.assert_not_called()
        method.process_weights_after_loading(layer)
        scheme.process_weights_after_loading.assert_called_once_with(layer)


@pytest.mark.parametrize("enabled", [False, True])
def test_routed_owner_injects_offload_before_weight_creation(enabled):
    offload_config = ExpertOffloadConfig(enabled=enabled, num_device_experts=2 if enabled else None)
    method = Mock()
    quant_config = SimpleNamespace(get_quant_method=Mock(return_value=method))
    cls = load_hook_class(
        "vllm_ascend/ops/fused_moe/routed_experts.py",
        "AscendRoutedExperts",
        {"_get_quant_method"},
        object,
        {"get_ascend_config": lambda: SimpleNamespace(expert_offload_config=offload_config)},
    )
    layer = cls()
    layer.tid2eid = None
    assert layer._get_quant_method("model.layers.0.mlp.experts", quant_config, "moe_config") is method
    quant_config.get_quant_method.assert_called_once_with(layer, "model.layers.0.mlp.experts", tid2eid=None)
    if enabled:
        method.configure_expert_offload.assert_called_once_with(offload_config)
    else:
        method.configure_expert_offload.assert_not_called()


def test_routed_owner_rejects_unquantized_or_unsupported_adapter():
    offload_config = ExpertOffloadConfig(enabled=True, num_device_experts=2)
    cls = load_hook_class(
        "vllm_ascend/ops/fused_moe/routed_experts.py",
        "AscendRoutedExperts",
        {"_get_quant_method"},
        object,
        {"get_ascend_config": lambda: SimpleNamespace(expert_offload_config=offload_config)},
    )
    layer = cls()
    layer.tid2eid = None
    with pytest.raises(ValueError, match="quantized"):
        layer._get_quant_method("experts", None, object())
    quant_config = SimpleNamespace(get_quant_method=lambda *_args, **_kwargs: object())
    with pytest.raises(ValueError, match="does not support"):
        layer._get_quant_method("experts", quant_config, object())


@pytest.mark.parametrize("offload,fail", [(False, False), (True, False), (True, True)])
def test_routed_hook_passes_consistent_view_and_restores_comm(offload, fail):
    original_comm, slot_comm = Mock(), Mock()
    context = SimpleNamespace(in_profile_run=False, moe_comm_type="allgather", moe_comm_method=None)
    cls = load_hook_class(
        "vllm_ascend/ops/fused_moe/routed_experts.py",
        "AscendRoutedExperts",
        {"forward_impl"},
        object,
        {
            "torch": torch,
            "_EXTRA_CTX": context,
            "get_forward_context": lambda: SimpleNamespace(all_moe_layers=None),
            "get_moe_comm_method": lambda *_args: original_comm,
            "_record_v2_eplb_load": Mock(),
            "AllGatherCommImpl": Mock,
        },
    )
    layer = cls()
    layer.enable_npugraph_ex_static_kernel = False
    layer.moe_config = SimpleNamespace(is_sequence_parallel=False)
    layer.quant_type = "mxfp4"
    layer.router = object()
    layer.dynamic_eplb = False
    layer._use_v2_model_runner = True
    layer.return_with_event = False
    logical = torch.tensor([[4, 5]])
    physical = torch.tensor([[0, 1]])
    view = object()
    layer._select_experts = Mock(return_value=(torch.ones(1, 2), logical))
    layer._expert_offload_binding = Mock() if offload else None
    if offload:
        assert layer._expert_offload_binding is not None
        layer._expert_offload_binding.prepare.return_value = view, physical, slot_comm
    original_comm.prepare.return_value = SimpleNamespace(
        hidden_states=torch.ones(1, 4),
        router_logits=torch.ones(1, 6),
        mc2_mask=None,
        padded_hidden_states_shape=None,
        pertoken_scale=None,
    )

    def apply(**kwargs):
        assert kwargs["layer"] is (view if offload else layer)
        assert torch.equal(kwargs["topk_ids"], physical if offload else logical)
        assert context.moe_comm_method is (slot_comm if offload else original_comm)
        if fail:
            raise RuntimeError("kernel failed")
        return SimpleNamespace(routed_out="output")

    layer.quant_method = SimpleNamespace(apply=apply)
    if fail:
        with pytest.raises(RuntimeError, match="kernel failed"):
            layer.forward_impl(hidden_states=torch.ones(1, 4), router_logits=torch.ones(1, 6))
        original_comm.finalize.assert_not_called()
    else:
        layer.forward_impl(hidden_states=torch.ones(1, 4), router_logits=torch.ones(1, 6))
        original_comm.finalize.assert_called_once()
    assert context.moe_comm_method is original_comm
    assert layer.ascend_mc2_mask is None and layer.ascend_pertoken_scale is None
