# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Check dedicated selection, local routing and graph-boundary ownership."""

import ast
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch


@pytest.fixture
def module():
    root = Path(__file__).resolve().parents[4] / "vllm_ascend/ops/fused_moe"
    loaded = {}
    for suffix, file in (
        ("deepgemm_v41", "deepgemm_v41.py"),
        ("router.fused_topk_router", "router/fused_topk_router.py"),
    ):
        name = "vllm_ascend.ops.fused_moe." + suffix
        spec = importlib.util.spec_from_file_location(name, root / file)
        value = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(value)
        loaded[name] = value
    with patch.dict(sys.modules, loaded):
        spec = importlib.util.spec_from_file_location("mega_module_test", root / "deepgemm_v41_module.py")
        value = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(value)
        yield value


@pytest.fixture
def selection(module, monkeypatch):
    config = SimpleNamespace(
        model_config=SimpleNamespace(hf_text_config=SimpleNamespace(model_type="deepseek_v41")),
        parallel_config=SimpleNamespace(enable_eplb=False, enable_dbo=False, expert_placement_strategy="linear"),
        lora_config=None,
    )
    ascend = SimpleNamespace(enable_fused_mc2=0, mix_placement=False)
    hardware = SimpleNamespace(supports=lambda capability: True)
    monkeypatch.setattr(module, "get_current_vllm_config", lambda: config)
    monkeypatch.setattr(module, "get_ascend_config", lambda: ascend)
    monkeypatch.setattr(module, "get_current_hardware_profile", lambda: hardware)
    monkeypatch.setattr(module, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(module, "get_ep_group", lambda: SimpleNamespace(world_size=8))
    factory = MagicMock()
    monkeypatch.setattr(module, "FusedMoEFactory", factory)
    kwargs = dict(
        top_k=6,
        num_experts=384,
        renormalize=True,
        scoring_func="sqrtsoftplus",
        routed_scaling_factor=1.5,
        image_sentinel_lo=129257,
    )
    return config, ascend, hardware, factory, kwargs


def test_factory_selects_explicit_classes_and_scales_only_in_backend(module, selection):
    _, _, _, factory, kwargs = selection
    module.create_deepgemm_v41_moe(**kwargs)
    args = factory.call_args.kwargs
    assert args["runner_cls"] is module.DeepGemmV41MoE
    assert args["routed_experts_cls"] is module.DeepGemmV41RoutedExperts
    assert args["router"].local_input_ids
    assert args["router"].routed_scaling_factor == 1
    assert args["routed_scaling_factor"] == 1.5
    assert args["apply_routed_scale_to_output"]


@pytest.mark.parametrize(
    "bad",
    ["model", "hardware", "eplb", "dynamic_eplb", "expert_map", "dbo", "placement", "mc2", "mixed", "lora", "scale"],
)
def test_reject_before_factory_can_mutate_shared_weights(module, selection, bad):
    config, ascend, hardware, factory, kwargs = selection
    if bad == "model":
        config.model_config.hf_text_config.model_type = "other"
    elif bad == "hardware":
        hardware.supports = lambda capability: False
    elif bad == "eplb":
        config.parallel_config.enable_eplb = True
    elif bad == "dynamic_eplb":
        ascend.eplb_config = SimpleNamespace(dynamic_eplb=True)
    elif bad == "expert_map":
        ascend.eplb_config = SimpleNamespace(expert_map_path="custom-map.json")
    elif bad == "dbo":
        config.parallel_config.enable_dbo = True
    elif bad == "placement":
        config.parallel_config.expert_placement_strategy = "round_robin"
    elif bad == "mc2":
        ascend.enable_fused_mc2 = 1
    elif bad == "mixed":
        ascend.mix_placement = True
    elif bad == "lora":
        config.lora_config = object()
    else:
        kwargs["routed_scaling_factor"] = float("nan")
    with pytest.raises(ValueError):
        module.create_deepgemm_v41_moe(**kwargs)
    factory.assert_not_called()


def test_forward_uses_existing_graph_op_without_prepare_or_reduction(module):
    runner = object.__new__(module.DeepGemmV41MoE)
    torch.nn.Module.__init__(runner)
    runner._forward_entry = MagicMock(return_value=torch.ones(2, 4))
    runner._encode_layer_name = lambda: "layer"
    x, logits, ids = torch.zeros(2, 4), torch.zeros(2, 384), torch.arange(2)
    assert runner(x, logits, input_ids=ids).shape == x.shape
    runner._forward_entry.assert_called_once_with(x, logits, None, ids, "layer", 0)
    with pytest.raises(ValueError):
        runner(x, logits, shared_experts_input=x.clone())


def test_forward_impl_calls_one_backend_with_local_rows(module):
    runner = object.__new__(module.DeepGemmV41MoE)
    torch.nn.Module.__init__(runner)
    x, logits, ids = torch.zeros(2, 4), torch.zeros(2, 384), torch.arange(2)
    weights, topk = torch.ones(2, 6), torch.zeros(2, 6, dtype=torch.int32)
    runner.router = MagicMock()
    runner.router.select_experts.return_value = weights, topk
    backend = MagicMock()
    backend.forward.return_value = x
    runner.routed_experts = SimpleNamespace(mega_experts=backend)
    assert runner._forward_impl(x, logits, None, ids) is x
    backend.forward.assert_called_once_with(runner.routed_experts, x, weights, topk)
    assert runner.router.select_experts.call_args.kwargs["input_ids"] is ids


def test_dedicated_weight_lifecycle_packs_once_and_rejects_generic_apply(module):
    method = object.__new__(module.DeepGemmV41WeightMethod)
    layer = torch.nn.Module()
    with patch.object(module, "DeepGemmV41Experts", return_value=object()) as constructor:
        method.process_weights_after_loading(layer)
        method.process_weights_after_loading(layer)
        constructor.assert_called_once_with(layer)
    with pytest.raises(RuntimeError, match="DeepGemmV41MoE"):
        method.apply(layer, torch.zeros(1, 4), None, None)


def test_unsupported_weight_method_rejected_before_runner(module):
    layer = object.__new__(module.DeepGemmV41RoutedExperts)
    torch.nn.Module.__init__(layer)
    with (
        patch.object(module.AscendRoutedExperts, "_get_quant_method", return_value=object()),
        pytest.raises(ValueError, match="W4A8 MXFP4"),
    ):
        layer._get_quant_method("layer", None, None)


def test_model_external_gate_preserves_precast_fp32_weight():
    # Execute the production forward method without importing the full model's
    # attention/Engram dependencies. The actual torch matmul catches BF16/FP32
    # mismatches when moving gate computation out of the ordinary runner.
    path = Path(__file__).resolve().parents[4] / "vllm_ascend/models/deepseek_v41/model.py"
    tree = ast.parse(path.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "DeepseekV41MoE")
    forward = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "forward")
    namespace = {"torch": torch, "F": torch.nn.functional}
    exec(compile(ast.Module(body=[forward], type_ignores=[]), str(path), "exec"), namespace)
    experts = MagicMock(side_effect=lambda **kwargs: kwargs["hidden_states"])
    experts.is_internal_router = False
    gate = SimpleNamespace(weight=torch.ones(4, 4, dtype=torch.bfloat16), weight_fp32=torch.eye(4))
    model = SimpleNamespace(is_sequence_parallel=False, tp_size=1, experts=experts, gate=gate)
    x = torch.arange(8).reshape(2, 4).bfloat16()
    output = namespace["forward"](model, x)
    assert torch.equal(output, x)
    assert torch.equal(experts.call_args.kwargs["router_logits"], x.float())


def test_local_hash_router_never_reads_comm_object(module):
    router = module.AscendFusedTopKRouter(
        top_k=6,
        global_num_experts=384,
        scoring_func="sqrtsoftplus",
        tid2eid=torch.zeros(4, 6, dtype=torch.int32),
        local_input_ids=True,
    )
    source = sys.modules[router.__class__.__module__]
    weights, topk = torch.ones(2, 6), torch.zeros(2, 6, dtype=torch.int32)
    with (
        patch.object(source, "_EXTRA_CTX", SimpleNamespace()),
        patch.object(torch.ops._C_ascend, "moe_gating_top_k_hash", return_value=(weights, topk, None), create=True),
    ):
        actual = router._compute_routing(
            torch.zeros(2, 5120), torch.zeros(2, 384), torch.int32, input_ids=torch.tensor([1, -1])
        )
        assert torch.equal(actual[0], weights)
        with pytest.raises(ValueError, match="local query rows"):
            router._compute_routing(torch.zeros(2, 5120), torch.zeros(2, 384), torch.int32, input_ids=torch.tensor([1]))
