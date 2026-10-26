# SPDX-License-Identifier: Apache-2.0
"""Exercise actual hook bodies in isolation from optional vLLM/NPU imports.

This verifies lifecycle composition and argument forwarding, not GPUModelRunner
or NPU kernel execution. Full runner tests remain part of the hardware gate.
"""

import ast
import sys
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from vllm_ascend.expert_offload.config import ExpertOffloadConfig
from vllm_ascend.expert_offload.expert_offload_manager import ExpertOffloadManager, OffloadState
from vllm_ascend.expert_offload.lifecycle import (
    close_expert_offload_on_shutdown,
    finalize_expert_offload_loading,
    guard_expert_offload_loading,
)
from vllm_ascend.expert_offload.quantization import MXFP4ExpertAdapter, maybe_offload_quant_method

ROOT = Path(__file__).resolve().parents[3]


def load_hook_class(path, class_name, methods, parent, namespace):
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    node = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    selected = [node for node in node.body if isinstance(node, ast.FunctionDef) and node.name in methods]
    for branch in node.body:
        if isinstance(branch, ast.If) and isinstance(branch.test, ast.Call):
            if isinstance(branch.test.func, ast.Name) and branch.test.func.id == "vllm_version_is":
                version = branch.test.args[0]
                assert isinstance(version, ast.Constant) and isinstance(version.value, str)
                release = namespace["vllm_version_is"](version.value)
                selected.extend(
                    method
                    for method in (branch.body if release else branch.orelse)
                    if isinstance(method, ast.FunctionDef) and method.name in methods
                )
    assert len(selected) == len(methods)
    for method in selected:
        # Exercise offload lifecycle decorators while excluding unrelated runtime wrappers.
        method.decorator_list = [
            decorator
            for decorator in method.decorator_list
            if isinstance(decorator, ast.Name)
            and decorator.id in ("guard_expert_offload_loading", "close_expert_offload_on_shutdown")
        ]
    class_body: list[ast.stmt] = [
        field for field in node.body if isinstance(field, ast.AnnAssign) and isinstance(field.value, ast.Constant)
    ] + list(selected)
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


@pytest.fixture(autouse=True)
def loading_memory(monkeypatch):
    memory_allocated = Mock(side_effect=[100, 164])
    monkeypatch.setattr(
        "vllm_ascend.expert_offload.lifecycle.torch",
        SimpleNamespace(npu=SimpleNamespace(memory_allocated=memory_allocated)),
    )
    return memory_allocated


@pytest.fixture(autouse=True)
def warmups(monkeypatch):
    modules = {}
    for name, functions in {
        "nz_warmup": ("start_nz_warm_thread", "join_nz_warm_thread"),
        "early_kernel_warmup": ("start_early_kernel_warmup", "join_early_kernel_warmup"),
    }.items():
        qualified = f"vllm_ascend.model_executor.warmup.{name}"
        module = ModuleType(qualified)
        for function in functions:
            setattr(module, function, Mock())
        monkeypatch.setitem(sys.modules, qualified, module)
        modules[name] = module
    return SimpleNamespace(**modules)


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
    fake_torch = SimpleNamespace(npu=SimpleNamespace(Stream=Mock(), current_stream=Mock(return_value=actual_stream)))
    cls = load_hook_class(
        "vllm_ascend/worker/v2/aclgraph_utils.py",
        "ModelAclGraphManager",
        {"__init__", "_capture_stream"},
        Parent,
        {
            "torch": fake_torch,
            "collect_sorted_captured_token_sizes": lambda groups: sorted(
                d.num_tokens for ds in groups.values() for d in ds
            ),
            "set_graph_params": graph_params,
        },
    )
    with patch("vllm_ascend.expert_offload.expert_offload_manager.torch", fake_torch):
        graph = cls(object(), "npu:0", CUDAGraphMode.FULL_DECODE_ONLY, 1, runner)
    fake_torch.npu.Stream.assert_called_once_with(device="npu:0")
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
            "torch": torch,
            "guard_expert_offload_loading": guard_expert_offload_loading,
            "finalize_expert_offload_loading": finalize_expert_offload_loading,
            "close_expert_offload_on_shutdown": close_expert_offload_on_shutdown,
            "CUDAGraphMode": SimpleNamespace(NONE=0),
            "C8KVBlockZeroer": type("C8KVBlockZeroer", (), {}),
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
    runner._uses_flashmla = False
    runner.vllm_config = SimpleNamespace(kv_transfer_config=None)
    runner.device = "cpu"
    runner.compilation_config = SimpleNamespace(cudagraph_mode=0)
    runner._start_dump_data = Mock()
    runner._finalize_dump_data = Mock()
    runner.ascend_config = SimpleNamespace(
        scheduler_config=SimpleNamespace(profiling_chunk_config=SimpleNamespace(need_timing=False))
    )
    runner.model_state = SimpleNamespace(kvpp_is_dummy_run=False, finish_execution=Mock())
    runner.prompt_logprobs_worker = None
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
    manager.finalize_loading.assert_called_once_with(runner.model)
    assert runner.model_memory_usage == 1064
    assert runner.pd_dspark_aux_layer_ids == [9]


def test_load_only_adopts_target_and_preserves_native_draft():
    manager = Mock()
    runner, _ = make_runner(manager=manager)
    runner.speculator = SimpleNamespace(model=object())
    runner.load_model()
    manager.finalize_loading.assert_called_once_with(runner.model)
    assert not hasattr(runner.speculator, "expert_offload_manager")


def test_disabled_load_does_not_allocate_or_finalize_offload(loading_memory):
    runner, events = make_runner()
    runner.load_model(True)
    assert events == [("parent_load", True, (), {})]
    assert runner.model_memory_usage == 1000
    loading_memory.assert_not_called()


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
    "enabled,failure",
    [(enabled, failure) for enabled in (False, True) for failure in ("start_nz", "start", "load", "join", "join_early")]
    + [(True, "finalize")],
)
def test_loading_failure_preserves_native_warmup_behavior_and_closes_offload(warmups, enabled, failure):
    manager = Mock() if enabled else None
    runner, _ = make_runner(manager=manager, fail_load=failure == "load")
    if failure == "start_nz":
        warmups.nz_warmup.start_nz_warm_thread.side_effect = RuntimeError("start_nz failed")
    elif failure == "start":
        warmups.early_kernel_warmup.start_early_kernel_warmup.side_effect = RuntimeError("start failed")
    elif failure == "join":
        warmups.nz_warmup.join_nz_warm_thread.side_effect = RuntimeError("join failed")
    elif failure == "join_early":
        warmups.early_kernel_warmup.join_early_kernel_warmup.side_effect = RuntimeError("join_early failed")
    elif failure == "finalize":
        assert manager is not None
        manager.finalize_loading.side_effect = RuntimeError("finalize failed")
    with pytest.raises(RuntimeError, match=f"{failure} failed"):
        runner.load_model()
    if failure in ("start_nz", "start"):
        warmups.nz_warmup.join_nz_warm_thread.assert_not_called()
    else:
        warmups.nz_warmup.join_nz_warm_thread.assert_called_once_with()
    if failure in ("start_nz", "start", "join"):
        warmups.early_kernel_warmup.join_early_kernel_warmup.assert_not_called()
    else:
        warmups.early_kernel_warmup.join_early_kernel_warmup.assert_called_once_with("load_model")
    if manager is not None:
        manager.close.assert_called_once_with()
        if failure != "finalize":
            manager.finalize_loading.assert_not_called()


def test_loading_cleanup_failure_preserves_the_original_error():
    manager = Mock()
    manager.close.side_effect = RuntimeError("cleanup failed")
    runner, _ = make_runner(manager=manager, fail_load=True)
    with (
        patch("vllm_ascend.expert_offload.lifecycle.logger.exception") as log,
        pytest.raises(RuntimeError, match="load failed"),
    ):
        runner.load_model()
    manager.close.assert_called_once_with()
    log.assert_called_once_with("Failed to clean up expert offload after model loading failure")


def test_pool_accounting_starts_after_both_warmups(warmups, loading_memory):
    manager = Mock()
    runner, _ = make_runner(manager=manager)
    runner.compilation_config.cudagraph_mode = 1
    order = Mock()
    order.attach_mock(warmups.nz_warmup.join_nz_warm_thread, "join_nz")
    order.attach_mock(warmups.early_kernel_warmup.join_early_kernel_warmup, "join_early")
    order.attach_mock(manager.finalize_loading, "finalize")
    order.attach_mock(loading_memory, "memory")
    order.attach_mock(runner._start_dump_data, "dump")
    runner.load_model()
    assert [entry[0] for entry in order.mock_calls] == ["join_nz", "join_early", "memory", "finalize", "memory", "dump"]
    assert runner.model_memory_usage == 1064
    assert runner.model_state.prompt_logprobs_worker is runner.prompt_logprobs_worker


@pytest.mark.parametrize("enabled", [False, True])
def test_dumper_failure_preserves_error_and_releases_offload(enabled, loading_memory):
    manager = Mock() if enabled else None
    runner, _ = make_runner(manager=manager)
    runner.compilation_config.cudagraph_mode = 1
    runner._start_dump_data.side_effect = RuntimeError("dumper failed")
    with pytest.raises(RuntimeError, match="dumper failed"):
        runner.load_model()
    if manager is not None:
        manager.finalize_loading.assert_called_once_with(runner.model)
        manager.close.assert_called_once_with()
    else:
        loading_memory.assert_not_called()


def test_load_preserves_keyword_argument_forwarding():
    manager = Mock()
    runner, events = make_runner(manager=manager)
    runner.load_model(load_dummy_weights=False, reload=False)
    assert events == [("parent_load", False, (), {"reload": False})]
    manager.finalize_loading.assert_called_once_with(runner.model)


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
    runner._c8_mxfp_v_scale_caches = [object()]
    with pytest.raises(RuntimeError, match="cleanup failed"):
        runner.shutdown()
    assert events == ["parent_shutdown"]
    assert runner._c8_mxfp_v_scale_caches == []


def test_shutdown_drains_offload_before_native_cleanup():
    manager = Mock()
    runner, events = make_runner(manager=manager)
    runner._c8_mxfp_v_scale_caches = [object()]

    def close():
        assert runner._c8_mxfp_v_scale_caches
        events.append("offload_close")

    manager.close.side_effect = close
    runner.shutdown()
    assert events == ["offload_close", "parent_shutdown"]
    assert runner._c8_mxfp_v_scale_caches == []


def test_shutdown_preserves_c8_cleanup_without_an_initialized_offload_manager():
    runner, events = make_runner()
    del runner.expert_offload_manager
    runner._c8_mxfp_v_scale_caches = [object()]
    runner.shutdown()
    assert runner._c8_mxfp_v_scale_caches == []
    assert events == ["parent_shutdown"]


def test_runner_defaults_to_no_offload_even_when_init_is_skipped():
    runner, _ = make_runner()
    del runner.expert_offload_manager
    assert runner.expert_offload_manager is None
    runner.load_model()
    scheduler = SimpleNamespace(total_num_scheduled_tokens=1, num_scheduled_tokens={"request": 1})
    assert runner.execute_model(scheduler, dummy_run=False) == "result"
    assert "expert_offload_manager" not in vars(runner)


@pytest.fixture
def quant_methods(monkeypatch):
    class Parent:
        requires_device_loading = True

        def __init__(self, moe_config):
            pass

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
    native_cls = load_hook_class(
        "vllm_ascend/quantization/method_adapters.py",
        "AscendFusedMoEMethod",
        {"__init__", "create_weights", "apply", "process_weights_after_loading"},
        Parent,
        {
            "torch": torch,
            "set_weight_attrs": set_attrs,
            "FusedMoeWeightScaleSupported": SimpleNamespace(
                CHANNEL=SimpleNamespace(value="channel"),
                GROUP=SimpleNamespace(value="group"),
            ),
        },
    )
    offload_cls = load_hook_class(
        "vllm_ascend/expert_offload/method.py",
        "ExpertOffloadMoEMethod",
        {"__init__", "create_weights", "apply", "process_weights_after_loading"},
        native_cls,
        {
            "AscendFusedMoEMethod": native_cls,
            "torch": torch,
            "LayerOffloadBinding": binding_factory,
            "MXFP4ExpertAdapter": MXFP4ExpertAdapter,
            "QuantType": SimpleNamespace(W4A8MXFP="mxfp4"),
        },
    )
    state = SimpleNamespace(config=None)
    config_module = ModuleType("vllm_ascend.ascend_config")
    monkeypatch.setattr(
        config_module, "get_ascend_config", lambda: SimpleNamespace(expert_offload_config=state.config), raising=False
    )
    monkeypatch.setitem(sys.modules, config_module.__name__, config_module)
    method_module = ModuleType("vllm_ascend.expert_offload.method")
    monkeypatch.setattr(method_module, "ExpertOffloadMoEMethod", offload_cls, raising=False)
    monkeypatch.setitem(sys.modules, method_module.__name__, method_module)

    class OwnerParent(torch.nn.Module):
        def _get_quant_method(self, prefix, quant_config, moe_config):
            return object()

    owner_cls = load_hook_class(
        "vllm_ascend/ops/fused_moe/routed_experts.py",
        "AscendRoutedExperts",
        {"_get_quant_method"},
        OwnerParent,
        {"maybe_offload_quant_method": maybe_offload_quant_method},
    )
    layer = owner_cls()
    layer.tid2eid = None
    layer.moe_config = object()
    return SimpleNamespace(
        native_cls=native_cls, offload_cls=offload_cls, layer=layer, state=state, scheme=scheme, binding=binding_factory
    )


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("is_draft", [False, True])
def test_quant_wrapper_only_offloads_target_and_preserves_native_draft(quant_methods, enabled, is_draft):
    q = quant_methods
    config = ExpertOffloadConfig(enabled=enabled, num_device_experts=2 if enabled else None)
    q.state.config = config
    layer = q.layer
    layer._expert_offload_enabled = not is_draft
    native = q.native_cls(q.scheme, layer.moe_config)
    quant_config = SimpleNamespace(get_quant_method=Mock(return_value=native))
    method = layer._get_quant_method("experts", quant_config, layer.moe_config)
    quant_config.get_quant_method.assert_called_once_with(layer, "experts", tid2eid=None)
    offloaded = enabled and not is_draft
    assert isinstance(method, q.offload_cls) is offloaded
    if not offloaded:
        assert method is native
    layer.top_k = 2
    with torch.device("meta"):
        method.create_weights(layer, 6, 3, 2, torch.float32, weight_loader=object())
    assert layer.w13_weight.device.type == ("cpu" if offloaded else "meta")
    assert method.requires_device_loading is not offloaded
    assert layer.w13_weight.shape[0] == 6
    if offloaded:
        q.binding.assert_called_once_with(layer, q.scheme, config)
        method.process_weights_after_loading(layer)
        q.binding.return_value.process_weights_after_loading.assert_called_once_with(layer)
        q.scheme.process_weights_after_loading.assert_not_called()
    else:
        q.binding.assert_not_called()
        assert not hasattr(layer, "_expert_offload_binding")
        method.process_weights_after_loading(layer)
        q.scheme.process_weights_after_loading.assert_called_once_with(layer)


def test_missing_offload_config_preserves_native_method(quant_methods):
    layer = quant_methods.layer
    layer._expert_offload_enabled = True
    native = object()
    quant_config = SimpleNamespace(get_quant_method=lambda *_a, **_k: native)
    assert layer._get_quant_method("experts", quant_config, None) is native


def test_routed_owner_rejects_unquantized_or_unsupported_adapter(quant_methods):
    quant_methods.state.config = ExpertOffloadConfig(enabled=True, num_device_experts=2)
    layer = quant_methods.layer
    layer._expert_offload_enabled = True
    with pytest.raises(ValueError, match="quantized"):
        layer._get_quant_method("experts", None, object())
    quant_config = SimpleNamespace(get_quant_method=lambda *_args, **_kwargs: object())
    with pytest.raises(ValueError, match="does not support"):
        layer._get_quant_method("experts", quant_config, object())
    layer._expert_offload_enabled = False
    assert layer._get_quant_method("experts", quant_config, object()) is not None


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
    if offload:

        class Parent:
            def apply(self, **kwargs):
                return apply(**kwargs)

        offload_cls = load_hook_class(
            "vllm_ascend/expert_offload/method.py",
            "ExpertOffloadMoEMethod",
            {"apply"},
            Parent,
            {"_EXTRA_CTX": context},
        )
        layer.quant_method = offload_cls()
    if fail:
        with pytest.raises(RuntimeError, match="kernel failed"):
            layer.forward_impl(hidden_states=torch.ones(1, 4), router_logits=torch.ones(1, 6))
        original_comm.finalize.assert_not_called()
    else:
        layer.forward_impl(hidden_states=torch.ones(1, 4), router_logits=torch.ones(1, 6))
        original_comm.finalize.assert_called_once()
    assert context.moe_comm_method is original_comm
    assert layer.ascend_mc2_mask is None and layer.ascend_pertoken_scale is None
