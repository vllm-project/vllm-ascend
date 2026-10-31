# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""CPU checks for adapter guards, raw scale packing, and target/draft ownership."""

import importlib.util
import sys
import weakref
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch


@pytest.fixture
def context():
    spec = importlib.util.spec_from_file_location(
        "deepgemm_v41_backend",
        Path(__file__).resolve().parents[4] / "vllm_ascend/ops/fused_moe/deepgemm_v41.py",
    )
    backend = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(backend)
    config = SimpleNamespace(
        additional_config={"deepgemm_mega_moe": True},
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(model_type="deepseek_v41", n_routed_experts=384, n_shared_experts=0)
        ),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=128),
        parallel_config=SimpleNamespace(enable_eplb=False),
    )
    group = SimpleNamespace(world_size=8, device_group=object())
    ascend = SimpleNamespace(enable_fused_mc2=0)
    backend.get_current_vllm_config = lambda: config
    backend.get_ep_group = lambda: group
    backend.get_tensor_model_parallel_world_size = lambda: 1
    backend.get_ascend_config = lambda: ascend
    dependency = MagicMock()
    dependency.transform_weights_for_mega_moe.side_effect = lambda a, b: (a, b)

    def buffer(group, experts, capacity, topk, hidden, intermediate, num_shared_experts, base=None):
        return SimpleNamespace(
            num_bytes=1234,
            base=base,
            num_shared_experts=num_shared_experts,
            x=torch.empty((capacity, hidden), dtype=torch.float8_e4m3fn),
            x_sf=torch.empty((capacity, hidden // 64), dtype=torch.int16),
            topk_idx=torch.empty((capacity, topk), dtype=torch.int32),
            topk_weights=torch.empty((capacity, topk), dtype=torch.float32),
        )

    dependency.SymmBuffer.side_effect = buffer
    with patch.dict(sys.modules, {"deep_gemm": dependency}):
        yield backend, config, group, ascend, dependency


def layer(experts=384, topk=6):
    local = experts // 8
    result = torch.nn.Module()
    result.moe_config = SimpleNamespace(
        activation="silu", num_experts=experts, experts_per_token=topk, swiglu_limit=10.0
    )
    result.global_redundant_expert_num = 0
    result.apply_router_weight_on_input = False
    for name, shape in {
        "w13_weight": (local, 512, 128),
        "w2_weight": (local, 256, 128),
        "w13_weight_scale": (local, 512, 8),
        "w2_weight_scale": (local, 256, 8),
    }.items():
        data = torch.arange(torch.tensor(shape).prod().item(), dtype=torch.int64).to(torch.uint8).reshape(shape)
        result.register_parameter(name, torch.nn.Parameter(data, requires_grad=False))
    return result


def shared_module(count=1, block=32, fp32_scale=False):
    shared = torch.nn.Module()
    for name, shape in {"gate_up_proj": (512 * count, 256), "down_proj": (256, 256 * count)}.items():
        linear = torch.nn.Module()
        linear.weight = torch.nn.Parameter(torch.zeros(shape, dtype=torch.float8_e4m3fn), requires_grad=False)
        scale_shape = (shape[0] // block, shape[1] // block) if block else (shape[0], shape[1] // 32)
        exponents = (torch.arange(scale_shape[0] * scale_shape[1]).reshape(scale_shape) % 5 + 125).to(torch.uint8)
        scale = torch.pow(2.0, exponents.float() - 127) if fp32_scale else exponents
        linear.weight_scale = torch.nn.Parameter(scale, requires_grad=False)
        setattr(shared, name, linear)
    return shared


def test_target_draft_share_one_allocation_and_keep_layouts(context):
    backend, _, _, _, dependency = context
    main = backend.DeepGemmV41Experts(layer())
    second_layer = backend.DeepGemmV41Experts(layer())
    draft = backend.DeepGemmV41Experts(layer(128, 3))
    assert main.buffer is second_layer.buffer
    assert draft.buffer is not main.buffer
    assert draft.buffer.base is main.buffer
    assert dependency.SymmBuffer.call_count == 2


def test_checkpoint_scales_are_paired_along_k_without_transpose(context):
    backend, _, _, _, dependency = context
    item = layer()
    original = item.w13_weight_scale.detach().clone()
    backend.DeepGemmV41Experts(item)
    pair = dependency.transform_weights_for_mega_moe.call_args.args[0]
    assert pair[0].dtype == torch.int8
    assert pair[1].dtype == torch.int16
    assert pair[1].shape == (48, 512, 4)
    assert torch.equal(pair[1].view(torch.uint8), original)


@pytest.mark.parametrize("mode", ["model", "tp", "ep", "cann", "eplb", "replicas", "input_weight", "activation"])
def test_unsupported_config_fails_before_weight_conversion(context, mode):
    backend, config, group, ascend, dependency = context
    item = layer()
    if mode == "model":
        config.model_config.hf_text_config.model_type = "other"
    if mode == "tp":
        backend.get_tensor_model_parallel_world_size = lambda: 2
    if mode == "ep":
        group.world_size = 4
    if mode == "cann":
        ascend.enable_fused_mc2 = 1
    if mode == "eplb":
        config.parallel_config.enable_eplb = True
    if mode == "replicas":
        item.global_redundant_expert_num = 8
    if mode == "input_weight":
        item.apply_router_weight_on_input = True
    if mode == "activation":
        item.moe_config.activation = "relu"
    with pytest.raises(ValueError):
        backend.DeepGemmV41Experts(item)
    dependency.transform_weights_for_mega_moe.assert_not_called()


def test_draft_cannot_own_initial_allocation(context):
    backend, _, _, _, _ = context
    with pytest.raises(RuntimeError, match="target before"):
        backend.DeepGemmV41Experts(layer(128, 3))


@pytest.mark.parametrize("block,fp32_scale", [(0, False), (32, False), (32, True), (128, True)])
def test_shared_checkpoint_scales_and_kernel_weight_ownership(context, block, fp32_scale):
    backend, config, _, _, dependency = context
    config.model_config.hf_text_config.n_shared_experts = 1
    item, shared = layer(), shared_module(block=block, fp32_scale=fp32_scale)
    original = shared.gate_up_proj.weight_scale.detach().clone()
    gate_weight_parameter = shared.gate_up_proj.weight
    scale_parameter = shared.gate_up_proj.weight_scale
    item._deepgemm_v41_shared_experts = weakref.ref(shared)
    adapter = backend.DeepGemmV41Experts(item)
    shared_l1 = dependency.transform_weights_for_mega_moe.call_args_list[0].args[0]
    expected = ((original.view(torch.int32) >> 23) & 0xFF).to(torch.uint8) if fp32_scale else original
    if block:
        expected = expected.repeat_interleave(block, dim=0).repeat_interleave(block // 32, dim=1)
    assert shared_l1[0].dtype == torch.float8_e4m3fn
    assert shared_l1[1].shape == (512, 4)
    assert torch.equal(shared_l1[1].view(torch.uint8), expected)
    assert adapter.shared_l1_weights[0] is gate_weight_parameter
    assert adapter.shared_l1_weights[1] is scale_parameter
    assert adapter.shared_l2_weights[0] is shared.down_proj.weight
    assert adapter.num_shared_experts == adapter.buffer.num_shared_experts == 1
    assert dependency.transform_weights_for_mega_moe.call_count == 2


def test_main_draft_shared_experts_reuse_base_but_keep_different_shared_layouts(context):
    backend, config, _, _, dependency = context
    config.model_config.hf_text_config.n_shared_experts = 1
    main_layer, main_shared = layer(), shared_module()
    main_layer._deepgemm_v41_shared_experts = weakref.ref(main_shared)
    main = backend.DeepGemmV41Experts(main_layer)
    draft_layer, draft_shared = layer(128, 3), shared_module()
    draft_layer._deepgemm_v41_shared_experts = weakref.ref(draft_shared)
    draft = backend.DeepGemmV41Experts(draft_layer)
    assert draft.buffer.base is main.buffer
    assert draft.shared_l1_weights[0] is not main.shared_l1_weights[0]
    config.model_config.hf_text_config.n_shared_experts = 0
    routed_only = backend.DeepGemmV41Experts(layer())
    assert routed_only.buffer is not main.buffer
    assert routed_only.buffer.base is main.buffer
    assert dependency.SymmBuffer.call_count == 3


def test_two_shared_experts_keep_routed_expert_count_and_topk(context):
    backend, config, _, _, dependency = context
    config.model_config.hf_text_config.n_shared_experts = 2
    item, shared = layer(), shared_module(count=2)
    item._deepgemm_v41_shared_experts = weakref.ref(shared)
    adapter = backend.DeepGemmV41Experts(item)
    assert adapter.num_shared_experts == adapter.buffer.num_shared_experts == 2
    assert adapter.num_experts == 384
    assert adapter.topk == 6
    assert dependency.SymmBuffer.call_args.kwargs["num_shared_experts"] == 2


def test_same_shared_weights_cannot_be_converted_for_another_layer(context):
    backend, config, _, _, dependency = context
    config.model_config.hf_text_config.n_shared_experts = 1
    first, second, shared = layer(), layer(), shared_module()
    first._deepgemm_v41_shared_experts = weakref.ref(shared)
    second._deepgemm_v41_shared_experts = weakref.ref(shared)
    backend.DeepGemmV41Experts(first)
    dependency.transform_weights_for_mega_moe.reset_mock()
    with pytest.raises(RuntimeError, match="already been converted"):
        backend.DeepGemmV41Experts(second)
    dependency.transform_weights_for_mega_moe.assert_not_called()


@pytest.mark.parametrize("mode", ["missing", "expired", "wrong_count", "wrong_shape", "wrong_dtype", "converted"])
def test_configured_shared_experts_cannot_fall_back_or_transform_twice(context, mode):
    backend, config, _, _, dependency = context
    config.model_config.hf_text_config.n_shared_experts = 1
    item, shared = layer(), shared_module()
    if mode != "missing":
        item._deepgemm_v41_shared_experts = weakref.ref(shared)
    if mode == "expired":
        del shared
    elif mode == "wrong_count":
        config.model_config.hf_text_config.n_shared_experts = 2
    elif mode == "wrong_shape":
        shared.gate_up_proj.weight.data = shared.gate_up_proj.weight.data[:256]
    elif mode == "wrong_dtype":
        shared.gate_up_proj.weight.data = shared.gate_up_proj.weight.data.to(torch.bfloat16)
    elif mode == "converted":
        shared._deepgemm_v41_weights_transformed = True
    with pytest.raises((RuntimeError, ValueError)):
        backend.DeepGemmV41Experts(item)
    dependency.transform_weights_for_mega_moe.assert_not_called()


def test_apply_fuses_shared_once_and_scales_only_routed_weights(context):
    backend, config, _, _, dependency = context
    config.model_config.hf_text_config.n_shared_experts = 1
    item, shared = layer(), shared_module()
    item._deepgemm_v41_shared_experts = weakref.ref(shared)
    item._deepgemm_v41_routed_scaling_factor = 1.5
    shared.forward = MagicMock(side_effect=AssertionError("Shared experts must only execute inside MegaMoE"))
    adapter = backend.DeepGemmV41Experts(item)
    weights = torch.full((2, 6), 0.25)
    indices = torch.tensor([[0, 48, 96, 144, 192, 240]] * 2, dtype=torch.int32)
    x = torch.ones((2, 256), dtype=torch.bfloat16)
    quantized = x.to(torch.float8_e4m3fn)
    scales = torch.full((2, 8), 127, dtype=torch.uint8)
    stream = SimpleNamespace(record_event=lambda: object())
    result_module = SimpleNamespace(FusedExpertsResult=lambda **kwargs: SimpleNamespace(**kwargs))
    dependency.fp8_fp4_mega_moe.side_effect = lambda out, *args, **kwargs: out.fill_(2)
    with (
        patch.object(torch, "npu", SimpleNamespace(current_stream=lambda: stream), create=True),
        patch.object(backend.torch_npu, "npu_dynamic_mx_quant", return_value=(quantized, scales)),
        patch.dict(sys.modules, {"vllm_ascend.ops.fused_moe.moe_comm_method": result_module}),
    ):
        result = adapter.forward(item, x, weights, indices)
    assert torch.equal(adapter.buffer.topk_weights[:2], weights * 1.5)
    assert torch.equal(adapter.buffer.topk_idx[:2], indices)
    assert torch.equal(weights, torch.full((2, 6), 0.25))
    assert torch.equal(result, torch.full_like(x, 2))
    kwargs = dependency.fp8_fp4_mega_moe.call_args.kwargs
    assert kwargs["shared_l1_weights"] is adapter.shared_l1_weights
    assert kwargs["shared_l2_weights"] is adapter.shared_l2_weights
    shared.forward.assert_not_called()
    dependency.fp8_fp4_mega_moe.assert_called_once()


@pytest.mark.parametrize("tokens", [0, 1, 513, 1024])
def test_large_prefill_does_not_synchronize_host(context, tokens):
    backend, config, _, _, dependency = context
    config.scheduler_config.max_num_batched_tokens = 1024
    adapter = backend.DeepGemmV41Experts(layer())
    stream = MagicMock()

    def quant(x, **kwargs):
        return x.to(torch.float8_e4m3fn), torch.full((tokens, 8), 127, dtype=torch.uint8)

    with (
        patch.object(torch, "npu", SimpleNamespace(current_stream=lambda: stream), create=True),
        patch.object(backend.torch_npu, "npu_dynamic_mx_quant", side_effect=quant, create=True),
        patch.dict(
            sys.modules,
            {"vllm_ascend.ops.fused_moe.moe_comm_method": SimpleNamespace(FusedExpertsResult=SimpleNamespace)},
        ),
    ):
        adapter.forward(
            layer(),
            torch.ones((tokens, 256), dtype=torch.bfloat16),
            torch.ones((tokens, 6)),
            torch.zeros((tokens, 6), dtype=torch.int32),
        )
    stream.synchronize.assert_not_called()
    dependency.fp8_fp4_mega_moe.assert_called_once()


@pytest.mark.parametrize("name,value", [("expert_placement_strategy", "round_robin"), ("enable_dbo", True)])
def test_rejects_incompatible_symmetric_buffer_scheduling(context, name, value):
    backend, config, _, _, dependency = context
    setattr(config.parallel_config, name, value)
    with pytest.raises(ValueError):
        backend.DeepGemmV41Experts(layer())
    dependency.SymmBuffer.assert_not_called()
    dependency.transform_weights_for_mega_moe.assert_not_called()
