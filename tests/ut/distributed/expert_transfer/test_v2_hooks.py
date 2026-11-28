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

from vllm_ascend.ascend_config import ExpertOffloadConfig
from vllm_ascend.distributed.expert_transfer import expert_offload_utils as expert_offload
from vllm_ascend.distributed.expert_transfer.expert_offload_manager import ExpertOffloadManager
from vllm_ascend.distributed.expert_transfer.expert_offload_method import ExpertOffloadMoEMethod
from vllm_ascend.ops.fused_moe import routed_experts
from vllm_ascend.ops.fused_moe.routed_experts import AscendRoutedExperts
from vllm_ascend.quantization.method_adapters import AscendFusedMoEMethod
from vllm_ascend.quantization.quant_type import QuantType


@pytest.fixture(autouse=True)
def loading_memory(monkeypatch):
    memory_allocated = Mock(side_effect=[100, 164])
    monkeypatch.setattr(torch.npu, "memory_allocated", memory_allocated)
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
        {
            "load_model",
            "execute_model",
            "gather_batch_req_state",
            "shutdown",
            "_maybe_send_draft_kv",
        },
        Parent,
        {
            "torch": torch,
            "expert_offload": expert_offload,
            "logger": Mock(),
            "CUDAGraphMode": SimpleNamespace(NONE=0),
            "C8KVBlockZeroer": type("C8KVBlockZeroer", (), {}),
            "HostMirrorConnector": type("HostMirrorConnector", (), {}),
            "IntermediateTensors": type("IntermediateTensors", (), {}),
            "get_pd_dspark_aux_layer_ids": lambda _vc: [9],
            # The stub parent creates no host-offload owners.
            "track_host_offload_owners": lambda: nullcontext([]),
            "has_kv_transfer_group": lambda: False,
            # D2H runner hooks: stubs so the isolated module resolves the import.
            "d2h_retire": lambda runner, ids=None: 0,
            "d2h_record_source_ready": lambda runner: False,
            "should_skip_allreduce_across_dp_group": lambda _vc: False,
            "nullcontext": nullcontext,
            "lmhead_tp_enable": lambda: False,
            "is_pd_decode_recompute_scheduler_enabled": lambda _vc: False,
        },
    )
    runner = runner_class()
    runner.pcp_manager = None
    runner.model = torch.nn.Module()
    if manager is not None:
        runner.model._expert_offload_manager = manager
    runner._target_uses_flashmla = False
    runner.vllm_config = SimpleNamespace(kv_transfer_config=None, model_config=object())
    runner.device = "cpu"
    runner.compilation_config = SimpleNamespace(cudagraph_mode=0)
    runner._start_dump_data = Mock()
    runner._finalize_dump_data = Mock()
    runner.ascend_config = SimpleNamespace(
        ascend_compilation_config=SimpleNamespace(enable_static_kernel=False),
        scheduler_config=SimpleNamespace(profiling_chunk_config=SimpleNamespace(need_timing=False)),
        expert_offload_config=ExpertOffloadConfig(enabled=manager is not None, num_device_experts=2),
    )
    runner.model_state = SimpleNamespace(kvpp_is_dummy_run=False, finish_execution=Mock())
    runner.prompt_logprobs_worker = None
    runner._maybe_send_draft_kv = Mock()
    runner.kvpp = SimpleNamespace(complete_forward=Mock())
    runner.adaptive_verification = None
    runner.speculator = None
    return runner, events


def test_load_preserves_parent_contract_and_accounts_pool_memory(load_hook_class, warmups):
    manager = Mock()
    manager.initialize.side_effect = lambda _model: warmups.nz_warmup.join_nz_warm_thread.assert_called_once()
    runner, events = make_runner(load_hook_class, manager=manager)
    with patch(
        "vllm_ascend.distributed.expert_transfer.expert_offload_manager.ExpertOffloadManager", return_value=manager
    ):
        runner.load_model(False, "extra", reload=False)
    assert events == [("parent_load", False, ("extra",), {"reload": False})]
    manager.initialize.assert_called_once_with(runner.model)
    assert runner.model_memory_usage == 1064
    assert runner.pd_dspark_aux_layer_ids == [9]
    manager.unsubscribe_graph_streams.side_effect = lambda: events.append("unsubscribe_reports")
    runner.shutdown()
    assert events[-2:] == ["unsubscribe_reports", "parent_shutdown"]
    assert expert_offload.get_manager(runner.model) is None
    assert not hasattr(runner, "expert_offload_manager")


@pytest.mark.parametrize("load_dummy_weights", [False, True])
def test_disabled_load_does_not_initialize_offload(load_hook_class, loading_memory, load_dummy_weights):
    runner, events = make_runner(load_hook_class)
    with patch(
        "vllm_ascend.distributed.expert_transfer.expert_offload_manager.ExpertOffloadManager"
    ) as manager_factory:
        runner.load_model(load_dummy_weights)
    assert events == [("parent_load", load_dummy_weights, (), {})]
    manager_factory.assert_not_called()
    assert expert_offload.get_manager(runner.model) is None
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
        (True, False, False),
        (False, False, False),
        (False, True, False),
        (False, False, True),
        (False, False, False),
    ):
        scheduler_output.has_prefill = prefill
        runner.execute_model(scheduler_output, is_profile=profile, dummy_run=dummy)
        assert manager.has_prefill is (dummy or prefill)
        assert events[-1] == ("parent_execute", dummy, profile)


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
    weight_loader = object()
    native = AscendFusedMoEMethod(scheme, moe_config)
    quant_config = SimpleNamespace(online_quantization_config=None, get_quant_method=Mock(return_value=native))

    def parent_init(layer, *args, **kwargs):
        torch.nn.Module.__init__(layer)
        layer.moe_config, layer.top_k = moe_config, 2
        layer.layer_name, layer.quant_config = "experts", quant_config
        layer.quant_method = layer._get_quant_method("experts", quant_config, moe_config)
        layer.quant_method.create_weights(layer, 6, 3, 2, torch.float32, weight_loader=weight_loader)

    binding = Mock()
    state = SimpleNamespace(config=None)
    monkeypatch.setattr(routed_experts.RoutedExperts, "__init__", parent_init)
    monkeypatch.setattr(
        routed_experts,
        "get_ascend_config",
        lambda: SimpleNamespace(
            expert_offload_config=state.config, ascend_compilation_config=SimpleNamespace(enable_static_kernel=False)
        ),
    )
    monkeypatch.setattr(
        routed_experts,
        "get_current_vllm_config",
        lambda: SimpleNamespace(model_config=SimpleNamespace(runner_type="generate")),
    )
    monkeypatch.setattr("vllm_ascend.distributed.expert_transfer.expert_offload_manager.LayerOffloadBinding", binding)
    monkeypatch.setattr("vllm.compilation.backends.model_tag", "backbone")
    return SimpleNamespace(
        make_layer=AscendRoutedExperts,
        state=state,
        scheme=scheme,
        binding=binding,
        native=native,
        quant_config=quant_config,
        weight_loader=weight_loader,
    )


@pytest.mark.parametrize(
    "enabled,model_tag", [(False, "backbone"), (True, "backbone"), (True, "dspark_head"), (True, "draft_model")]
)
def test_offload_layer_preserves_native_resolution_and_draft(quant_methods, monkeypatch, enabled, model_tag):
    q = quant_methods
    config = ExpertOffloadConfig(enabled=enabled, num_device_experts=2 if enabled else None)
    q.state.config = config
    monkeypatch.setattr("vllm.compilation.backends.model_tag", model_tag)
    with torch.device("meta"):
        layer = q.make_layer()
        assert torch.empty(1).device.type == "meta"
    method = layer.quant_method
    q.quant_config.get_quant_method.assert_called_once_with(layer, "experts")
    offloaded = enabled and model_tag == "backbone"
    assert isinstance(method, ExpertOffloadMoEMethod) is offloaded
    assert method.quant_method is q.scheme
    if not offloaded:
        assert method is q.native
    assert all(param.device.type == ("cpu" if offloaded else "meta") for param in layer.parameters())
    assert all(param.weight_loader is q.weight_loader for param in layer.parameters())
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
        "vllm_ascend/distributed/expert_transfer/expert_offload_method.py",
        "ExpertOffloadMoEMethod",
        {"apply"},
        Parent,
        {"_EXTRA_CTX": context},
    )
    binding = Mock()
    binding.prepare.return_value = view, physical, slot_comm
    layer = SimpleNamespace(_expert_offload_binding=binding)
    assert cls().apply(layer, x, torch.ones(1, 2), logical) is x
    binding.prepare.assert_called_once_with(layer, logical)
    assert context.moe_comm_method is original_comm
