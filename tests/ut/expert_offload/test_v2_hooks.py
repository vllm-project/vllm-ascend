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
from vllm import config as vllm_config

from vllm_ascend.expert_offload.config import ExpertOffloadConfig
from vllm_ascend.expert_offload.expert_offload_manager import ExpertOffloadManager
from vllm_ascend.expert_offload.quantization import expert_adapter_type, expert_offload_quant_method
from vllm_ascend.expert_offload.utils import finalize_expert_offload_loading
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
            self.model = object()
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
            "finalize_expert_offload_loading": finalize_expert_offload_loading,
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
    manager.finalize_loading.assert_called_once_with(runner.model)
    assert runner.model_memory_usage == 1064
    assert runner.pd_dspark_aux_layer_ids == [9]
    manager.unsubscribe_graph_streams.side_effect = lambda: events.append("unsubscribe_reports")
    runner.shutdown()
    assert events[-2:] == ["unsubscribe_reports", "parent_shutdown"]
    assert runner.expert_offload_manager is None


@pytest.mark.parametrize("load_dummy_weights", [False, True])
def test_disabled_load_does_not_allocate_or_finalize_offload(load_hook_class, loading_memory, load_dummy_weights):
    runner, events = make_runner(load_hook_class)
    del runner.expert_offload_manager
    assert runner.expert_offload_manager is None
    finalize = Mock()
    with patch.dict(runner.load_model.__globals__, finalize_expert_offload_loading=finalize):
        runner.load_model(load_dummy_weights)
    assert events == [("parent_load", load_dummy_weights, (), {})]
    finalize.assert_not_called()
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


@pytest.fixture
def quant_methods(monkeypatch, load_hook_class):
    class Parent:
        requires_device_loading = True

        def __init__(self, moe_config):
            pass

    binding_factory = Mock()

    def set_attrs(param, attrs):
        for key, value in attrs.items():
            setattr(param, key, value)

    scheme = SimpleNamespace(
        quant_type=QuantType.W4A8MXFP,
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
            "expert_adapter_type": expert_adapter_type,
        },
    )
    state = SimpleNamespace(config=None)
    monkeypatch.setattr("vllm.compilation.backends.model_tag", "backbone")
    monkeypatch.setattr("vllm.config.get_current_vllm_config_or_none", lambda: object())
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
        {
            "expert_offload_quant_method": expert_offload_quant_method,
            "get_current_vllm_config_or_none": lambda: vllm_config.get_current_vllm_config_or_none(),
            "get_ascend_config": lambda: config_module.get_ascend_config(),
        },
    )
    layer = owner_cls()
    layer.tid2eid = None
    layer.moe_config = object()
    return SimpleNamespace(
        native_cls=native_cls,
        offload_cls=offload_cls,
        layer=layer,
        state=state,
        scheme=scheme,
        binding=binding_factory,
    )


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("model_tag", ["backbone", "dspark_head", "draft_model", "eagle_head"])
def test_quant_wrapper_only_offloads_target_and_preserves_native_draft(quant_methods, monkeypatch, enabled, model_tag):
    is_draft = model_tag != "backbone"
    q = quant_methods
    config = ExpertOffloadConfig(enabled=enabled, num_device_experts=2 if enabled else None)
    q.state.config = config
    layer = q.layer
    native = q.native_cls(q.scheme, layer.moe_config)
    quant_config = SimpleNamespace(get_quant_method=Mock(return_value=native))
    monkeypatch.setattr("vllm.compilation.backends.model_tag", model_tag)
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
