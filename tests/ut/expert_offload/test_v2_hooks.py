# SPDX-License-Identifier: Apache-2.0
"""Exercise actual hook bodies in isolation from optional vLLM/NPU imports.

This verifies lifecycle composition and argument forwarding, not GPUModelRunner
or NPU kernel execution. Full runner tests remain part of the hardware gate.
"""

import sys
from contextlib import nullcontext
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from vllm_ascend.expert_offload.config import ExpertOffloadConfig
from vllm_ascend.expert_offload.expert_offload_manager import ExpertOffloadManager, ExpertOffloadRoutedExperts
from vllm_ascend.expert_offload.method import ExpertOffloadMoEMethod
from vllm_ascend.expert_offload.utils import initialize_expert_offload
from vllm_ascend.ops.fused_moe.routed_experts import AscendRoutedExperts
from vllm_ascend.quantization.method_adapters import AscendFusedMoEMethod
from vllm_ascend.quantization.quant_type import QuantType


@pytest.fixture(autouse=True)
def loading_memory(monkeypatch):
    memory_allocated = Mock(side_effect=[100, 164])
    monkeypatch.setattr(
        "vllm_ascend.expert_offload.utils.torch",
        SimpleNamespace(npu=SimpleNamespace(memory_allocated=memory_allocated)),
    )
    return memory_allocated


@pytest.fixture(autouse=True)
def warmups(monkeypatch):
    modules = {}
    for name, functions in {
        "nz_warmup": ("start_nz_warm_thread", "join_nz_warm_thread"),
    }.items():
        qualified = f"vllm_ascend.model_executor.warmup.{name}"
        module = ModuleType(qualified)
        for function in functions:
            setattr(module, function, Mock())
        monkeypatch.setitem(sys.modules, qualified, module)
        modules[name] = module
    return SimpleNamespace(**modules)


def make_runner(load_hook_class, *, manager=None, has_prefill=False):
    events: list[object] = []

    class Parent:
        def load_model(self, load_dummy_weights=False, *args, **kwargs):
            events.append(("parent_load", load_dummy_weights, args, kwargs))
            self.model = torch.nn.Module()
            self.model_memory_usage = 1000

        def execute_model(self, scheduler_output, **kwargs):
            self.gather_batch_req_state(scheduler_output, kwargs["dummy_run"])
            events.append(("parent_execute", kwargs["dummy_run"], kwargs["is_profile"]))
            return "result"

        def gather_batch_req_state(self, scheduler_output, dummy_run):
            if dummy_run:
                return None, 1
            return SimpleNamespace(has_prefill=getattr(scheduler_output, "has_prefill", has_prefill)), 1

        def shutdown(self):
            events.append("parent_shutdown")

    runner_class = load_hook_class(
        "vllm_ascend/worker/v2/model_runner.py",
        "NPUModelRunner",
        {"load_model", "execute_model", "gather_batch_req_state", "shutdown"},
        Parent,
        {
            "torch": torch,
            "initialize_expert_offload": initialize_expert_offload,
            "logger": Mock(),
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
    runner.vllm_config = SimpleNamespace(kv_transfer_config=None, model_config=object())
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


def test_load_preserves_parent_contract_and_accounts_pool_memory(load_hook_class):
    manager = Mock()
    runner, events = make_runner(load_hook_class, manager=manager)
    runner.load_model(False, "extra", reload=False)
    assert events == [("parent_load", False, ("extra",), {"reload": False})]
    manager.initialize.assert_called_once_with(runner.model)
    assert runner.model_memory_usage == 1064
    assert runner.pd_dspark_aux_layer_ids == [9]
    manager.unsubscribe_graph_streams.side_effect = lambda: events.append("unsubscribe_reports")
    runner.shutdown()
    assert events[-2:] == ["unsubscribe_reports", "parent_shutdown"]
    assert runner.expert_offload_manager is None


@pytest.mark.parametrize("load_dummy_weights", [False, True])
def test_disabled_load_does_not_initialize_offload(load_hook_class, loading_memory, load_dummy_weights):
    runner, events = make_runner(load_hook_class)
    del runner.expert_offload_manager
    assert runner.expert_offload_manager is None
    initialize = Mock()
    with patch.dict(runner.load_model.__globals__, initialize_expert_offload=initialize):
        runner.load_model(load_dummy_weights)
    assert events == [("parent_load", load_dummy_weights, (), {})]
    initialize.assert_not_called()
    assert runner.model_memory_usage == 1000
    loading_memory.assert_not_called()


def test_execute_preserves_dummy_pool_and_restores_decode_phase(load_hook_class):
    manager = ExpertOffloadManager()
    runner, events = make_runner(load_hook_class, manager=manager)
    scheduler_output = SimpleNamespace(
        total_num_scheduled_tokens=1, num_scheduled_tokens={"request": 1}, has_prefill=False
    )
    for profile, dummy, prefill in (
        (True, True, False),
        (False, False, False),
        (False, True, False),
        (False, False, True),
        (False, False, False),
    ):
        scheduler_output.has_prefill = prefill
        runner.execute_model(scheduler_output, is_profile=profile, dummy_run=dummy)
        assert manager.has_prefill is (dummy or prefill)
        assert events[-1] == ("parent_execute", dummy, profile)


@pytest.mark.parametrize("enabled", [None, False, True])
def test_registration_selects_offload_layer_only_when_enabled(monkeypatch, enabled):
    from vllm.model_executor.custom_op import op_registry_oot
    from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts

    from vllm_ascend import utils

    monkeypatch.setattr(utils, "_ASCEND_CUSTOMOP_IS_REIGISTERED", False)
    monkeypatch.setattr(utils, "REGISTERED_ASCEND_OPS", {})
    additional_config = {} if enabled is None else {"expert_offload_config": {"enabled": enabled}}
    with patch.dict(op_registry_oot, {}, clear=True):
        utils.register_ascend_customop(SimpleNamespace(additional_config=additional_config))
        expected = ExpertOffloadRoutedExperts if enabled else AscendRoutedExperts
        assert utils.REGISTERED_ASCEND_OPS["RoutedExperts"] is expected
        assert type(RoutedExperts.__new__(RoutedExperts)) is expected


@pytest.fixture(params=[QuantType.W4A8MXFP, QuantType.W8A8MXFP])
def quant_methods(monkeypatch, request):
    scheme = SimpleNamespace(
        quant_type=request.param,
        group_size=32,
        get_weight=Mock(
            side_effect=lambda *args: {"w13_weight": torch.empty(6, 4, 3), "w2_weight": torch.empty(6, 4, 3)}
        ),
        get_dynamic_quant_param=Mock(
            side_effect=lambda *args: {
                "w13_weight_scale": torch.empty(6, 4, 3),
                "w2_weight_scale": torch.empty(6, 4, 3),
            }
        ),
        process_weights_after_loading=Mock(),
        validate_weight_shapes=Mock(),
    )
    moe_config = SimpleNamespace()

    def make_layer(layer_cls):
        layer = layer_cls.__new__(layer_cls)
        torch.nn.Module.__init__(layer)
        layer.moe_config, layer.top_k = moe_config, 2
        return layer

    binding = Mock()
    state = SimpleNamespace(config=None)
    get_config = Mock(side_effect=lambda: SimpleNamespace(expert_offload_config=state.config))
    monkeypatch.setattr("vllm_ascend.expert_offload.expert_offload_manager.get_ascend_config", get_config)
    monkeypatch.setattr("vllm_ascend.expert_offload.method.LayerOffloadBinding", binding)
    monkeypatch.setattr("vllm.compilation.backends.model_tag", "backbone")
    native = AscendFusedMoEMethod(scheme, moe_config)
    quant_config = SimpleNamespace(online_quantization_config=None, get_quant_method=Mock(return_value=native))
    return SimpleNamespace(
        make_layer=make_layer,
        state=state,
        get_config=get_config,
        scheme=scheme,
        binding=binding,
        native=native,
        quant_config=quant_config,
    )


def test_offload_layer_rejects_non_mxfp_before_allocating(quant_methods):
    q = quant_methods
    q.scheme.quant_type = QuantType.W4A8
    q.state.config = ExpertOffloadConfig(enabled=True, num_device_experts=2)
    layer = q.make_layer(ExpertOffloadRoutedExperts)
    with pytest.raises(ValueError, match="supports only W4A8-MXFP and W8A8-MXFP"):
        layer._get_quant_method("experts", q.quant_config, layer.moe_config)
    q.scheme.get_weight.assert_not_called()
    q.binding.assert_not_called()


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("model_tag", ["backbone", "dspark_head", "draft_model", "eagle_head"])
def test_offload_layer_preserves_native_resolution_and_draft(quant_methods, monkeypatch, enabled, model_tag):
    q = quant_methods
    config = ExpertOffloadConfig(enabled=enabled, num_device_experts=2 if enabled else None)
    q.state.config = config
    layer = q.make_layer(ExpertOffloadRoutedExperts if enabled else AscendRoutedExperts)
    monkeypatch.setattr("vllm.compilation.backends.model_tag", model_tag)
    method = layer._get_quant_method("experts", q.quant_config, layer.moe_config)
    q.quant_config.get_quant_method.assert_called_once_with(layer, "experts")
    offloaded = enabled and model_tag == "backbone"
    assert isinstance(method, ExpertOffloadMoEMethod) is offloaded
    assert method.quant_method is q.scheme
    if not offloaded:
        assert method is q.native
    if not enabled:
        q.get_config.assert_not_called()
    weight_loader = object()
    with torch.device("meta"):
        method.create_weights(layer, 6, 3, 2, torch.float32, weight_loader=weight_loader)
        assert torch.empty(1).device.type == "meta"
    assert all(param.device.type == ("cpu" if offloaded else "meta") for param in layer.parameters())
    assert all(param.weight_loader is weight_loader for param in layer.parameters())
    assert method.requires_device_loading is not offloaded
    assert layer.w13_weight.shape[0] == 6
    q.scheme.get_weight.assert_called_once_with(6, 2, 3, torch.float32)
    q.scheme.get_dynamic_quant_param.assert_called_once_with(6, 2, 3, torch.float32)
    q.scheme.validate_weight_shapes.assert_called_once_with(layer, 2)
    if offloaded:
        q.binding.assert_called_once_with(layer, q.scheme, config)
        method.process_weights_after_loading(layer)
        q.binding.return_value.initialize_storage.assert_called_once_with(layer)
        q.scheme.process_weights_after_loading.assert_not_called()
    else:
        q.binding.assert_not_called()
        assert not hasattr(layer, "_expert_offload_binding")
        method.process_weights_after_loading(layer)
        q.scheme.process_weights_after_loading.assert_called_once_with(layer)


def test_quantized_execution_uses_slot_weights_ids_and_comm(load_hook_class):
    original_comm, slot_comm, view = object(), object(), object()
    context = SimpleNamespace(moe_comm_method=original_comm)
    logical, physical = torch.tensor([[4, 5]]), torch.tensor([[0, 1]])
    x = torch.ones(1, 4)

    class Parent:
        def apply(self, **kwargs):
            assert kwargs["layer"] is view
            assert torch.equal(kwargs["topk_ids"], physical)
            assert context.moe_comm_method is slot_comm
            return kwargs["x"]

    cls = load_hook_class(
        "vllm_ascend/expert_offload/method.py", "ExpertOffloadMoEMethod", {"apply"}, Parent, {"_EXTRA_CTX": context}
    )
    binding = Mock()
    binding.prepare.return_value = view, physical, slot_comm
    layer = SimpleNamespace(_expert_offload_binding=binding)
    assert cls().apply(layer, x, torch.ones(1, 2), logical) is x
    binding.prepare.assert_called_once_with(layer, logical)
    assert context.moe_comm_method is original_comm
