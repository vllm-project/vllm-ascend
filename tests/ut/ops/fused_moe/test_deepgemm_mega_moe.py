# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""CPU checks for adapter guards, raw scale packing, and target/draft ownership."""

import importlib.util
import sys
import weakref
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, call, patch

import pytest
import torch
from vllm.model_executor.layers.activation import SiluAndMulWithClamp


@pytest.fixture
def context():
    spec = importlib.util.spec_from_file_location(
        "deepgemm_mega_moe_backend",
        Path(__file__).resolve().parents[4] / "vllm_ascend/ops/fused_moe/deepgemm_mega_moe.py",
    )
    assert spec is not None and spec.loader is not None
    backend: Any = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(backend)
    config = SimpleNamespace(
        additional_config={"enable_fused_mc2": 3, "deepgemm_mega_moe": True},
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(model_type="deepseek_v41", n_routed_experts=384, n_shared_experts=0)
        ),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=128),
        parallel_config=SimpleNamespace(enable_eplb=False),
        lora_config=None,
    )
    group = SimpleNamespace(world_size=8, device_group=object(), cpu_group=object())
    backend.get_current_vllm_config = lambda: config
    backend.get_ep_group = lambda: group
    backend.get_tensor_model_parallel_world_size = lambda: 1
    dependency = MagicMock()
    dependency._C.get_dry_run.return_value = False
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
    with (
        patch("vllm_ascend.ops.fused_moe.deepgemm_loader.load_deepgemm", return_value=dependency),
        patch.object(torch, "npu", SimpleNamespace(synchronize=MagicMock()), create=True),
        patch.object(torch.distributed, "barrier"),
    ):
        yield backend, config, group, dependency


def layer(experts=384, topk=6, ep_size=8):
    local = experts // ep_size
    result = torch.nn.Module()
    result.moe_config = SimpleNamespace(
        activation="silu",
        num_experts=experts,
        num_logical_experts=experts,
        num_local_experts=local,
        experts_per_token=topk,
        swiglu_limit=10.0,
    )
    result.router = None
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
    shared.act_fn = MagicMock(spec=SiluAndMulWithClamp, swiglu_limit=10.0, alpha=1.0, beta=0.0)
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
    backend, _, group, dependency = context
    main = backend.DeepGemmExperts(layer())
    second_layer = backend.DeepGemmExperts(layer())
    draft = backend.DeepGemmExperts(layer(128, 3))
    assert main.buffer is second_layer.buffer
    assert draft.buffer is not main.buffer
    assert draft.buffer.base is main.buffer
    assert dependency.SymmBuffer.call_count == 2
    assert dependency.fp8_fp4_mega_moe.call_count == 2
    assert all(c.args[0].shape[0] == 0 for c in dependency.fp8_fp4_mega_moe.call_args_list)
    assert dependency._C.set_dry_run.call_args_list == [call(True), call(False)] * 2
    assert torch.distributed.barrier.call_args_list == [call(group=group.cpu_group)] * 2


@pytest.mark.parametrize("previous_dry_run", [False, True])
def test_failed_cold_compile_restores_mode_without_publishing_buffer(context, previous_dry_run):
    backend, _, group, dependency = context
    dependency._C.get_dry_run.return_value = previous_dry_run
    dependency.fp8_fp4_mega_moe.side_effect = RuntimeError("cold compile failed")
    with pytest.raises(RuntimeError, match="cold compile failed"):
        backend.DeepGemmExperts(layer())
    assert dependency._C.set_dry_run.call_args_list == [call(True), call(previous_dry_run)]
    assert group._deepgemm_mega_moe_buffers == {}
    torch.distributed.barrier.assert_not_called()


def test_strided_checkpoint_weights_and_scales_pack_identically(context):
    backend, _, _, dependency = context
    item = layer()
    original = {}
    for name in ("w13_weight", "w2_weight", "w13_weight_scale", "w2_weight_scale"):
        data = getattr(item, name).data
        storage = torch.empty((*data.shape[:-1], data.shape[-1] * 2), dtype=data.dtype)
        view = storage[..., ::2]
        view.copy_(data)
        getattr(item, name).data = view
        original[name] = data.clone()
        assert not view.is_contiguous()
    backend.DeepGemmExperts(item)
    for pair, prefix in zip(dependency.transform_weights_for_mega_moe.call_args.args, ("w13", "w2")):
        assert pair[0].dtype == torch.int8 and pair[1].dtype == torch.int16
        assert pair[0].is_contiguous() and pair[1].is_contiguous()
        torch.testing.assert_close(pair[0].view(torch.uint8), original[prefix + "_weight"])
        torch.testing.assert_close(pair[1].view(torch.uint8), original[prefix + "_weight_scale"])


@pytest.mark.parametrize("mode", ["model", "tp", "eplb", "input_weight", "activation", "lora"])
def test_unsupported_config_fails_before_weight_conversion(context, mode):
    backend, config, group, dependency = context
    item = layer()
    if mode == "model":
        config.model_config.hf_text_config.model_type = "other"
    if mode == "tp":
        backend.get_tensor_model_parallel_world_size = lambda: 2
    if mode == "eplb":
        item.router = SimpleNamespace(eplb_state=object())
    if mode == "input_weight":
        item.apply_router_weight_on_input = True
    if mode == "activation":
        item.moe_config.activation = "relu"
    if mode == "lora":
        config.lora_config = object()
    with (
        patch("vllm_ascend.ops.fused_moe.deepgemm_loader.load_deepgemm") as load,
        pytest.raises(ValueError),
    ):
        backend.DeepGemmExperts(item)
    load.assert_not_called()
    dependency.transform_weights_for_mega_moe.assert_not_called()


@pytest.mark.parametrize("model_type", ["deepseek_v4", "deepseek_v41", "deepseek_v41_text"])
def test_v4_family_uses_same_kernel(context, model_type):
    backend, config, _, dependency = context
    config.model_config.hf_text_config.model_type = model_type
    backend.DeepGemmExperts(layer())
    dependency.fp8_fp4_mega_moe.assert_called_once()


@pytest.mark.parametrize(
    ("counts", "fallback"),
    [([32, 32], False), ([32, None], False), ([None, None], False), ([32, 24], True), ([40, 32], True)],
)
def test_device_support_is_resolved_for_the_ep_group(context, counts, fallback):
    backend, _, group, _ = context
    config = SimpleNamespace(additional_config={"enable_fused_mc2": 3, "deepgemm_mega_moe": True})
    group.world_size = len(counts)

    def collect(peers, local, group):
        assert local == counts[0]
        peers[:] = counts

    with (
        patch.object(
            torch.npu, "get_device_properties", return_value=SimpleNamespace(cube_core_num=counts[0]), create=True
        ),
        patch.object(torch.distributed, "all_gather_object", side_effect=collect),
        patch("vllm_ascend.ascend_config.clear_ascend_config"),
        patch("vllm_ascend.ascend_config.init_ascend_config") as resolve,
    ):
        backend.resolve_deepgemm_device_support(config, 0, group)
    assert config.additional_config == {"enable_fused_mc2": 3, "deepgemm_mega_moe": not fallback}
    assert resolve.called == fallback


@pytest.mark.parametrize("ep_size", [4, 16])
def test_backend_uses_ep_group_size(context, ep_size):
    backend, _, group, dependency = context
    group.world_size = ep_size
    backend.DeepGemmExperts(layer(ep_size=ep_size))
    assert dependency.SymmBuffer.call_args.args[0] is group.device_group


def test_draft_cannot_own_initial_allocation(context):
    backend, _, _, _ = context
    with pytest.raises(RuntimeError, match="target before"):
        backend.DeepGemmExperts(layer(128, 3))


@pytest.mark.parametrize("block,fp32_scale", [(0, False), (32, False), (32, True), (128, True)])
def test_shared_checkpoint_scales_and_kernel_weight_ownership(context, block, fp32_scale):
    backend, config, _, dependency = context
    config.model_config.hf_text_config.n_shared_experts = 1
    item, shared = layer(), shared_module(block=block, fp32_scale=fp32_scale)
    original = shared.gate_up_proj.weight_scale.detach().clone()
    gate_weight_parameter = shared.gate_up_proj.weight
    scale_parameter = shared.gate_up_proj.weight_scale
    item._deepgemm_mega_moe_shared_experts = weakref.ref(shared)
    adapter = backend.DeepGemmExperts(item)
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
    backend, config, _, dependency = context
    config.model_config.hf_text_config.n_shared_experts = 1
    main_layer, main_shared = layer(), shared_module()
    main_layer._deepgemm_mega_moe_shared_experts = weakref.ref(main_shared)
    main = backend.DeepGemmExperts(main_layer)
    draft_layer, draft_shared = layer(128, 3), shared_module()
    draft_layer._deepgemm_mega_moe_shared_experts = weakref.ref(draft_shared)
    draft = backend.DeepGemmExperts(draft_layer)
    assert draft.buffer.base is main.buffer
    assert draft.shared_l1_weights[0] is not main.shared_l1_weights[0]
    config.model_config.hf_text_config.n_shared_experts = 0
    routed_only = backend.DeepGemmExperts(layer())
    assert routed_only.buffer is not main.buffer
    assert routed_only.buffer.base is main.buffer
    assert dependency.SymmBuffer.call_count == 3


def test_two_shared_experts_keep_routed_expert_count_and_topk(context):
    backend, config, _, dependency = context
    config.model_config.hf_text_config.n_shared_experts = 2
    item, shared = layer(), shared_module(count=2)
    item._deepgemm_mega_moe_shared_experts = weakref.ref(shared)
    adapter = backend.DeepGemmExperts(item)
    assert adapter.num_shared_experts == adapter.buffer.num_shared_experts == 2
    assert adapter.num_experts == 384
    assert adapter.topk == 6
    assert dependency.SymmBuffer.call_args.kwargs["num_shared_experts"] == 2


@pytest.mark.parametrize(
    "mode",
    ["missing", "expired", "wrong_count", "wrong_shape", "wrong_dtype", "converted", "activation", "bias", "gate"],
)
def test_shared_experts_must_be_compatible_and_unconverted(context, mode):
    backend, config, _, dependency = context
    config.model_config.hf_text_config.n_shared_experts = 1
    item, shared = layer(), shared_module()
    if mode != "missing":
        item._deepgemm_mega_moe_shared_experts = weakref.ref(shared)
    if mode == "expired":
        del shared
    elif mode == "wrong_count":
        config.model_config.hf_text_config.n_shared_experts = 2
    elif mode == "wrong_shape":
        shared.gate_up_proj.weight.data = shared.gate_up_proj.weight.data[:256]
    elif mode == "wrong_dtype":
        shared.gate_up_proj.weight.data = shared.gate_up_proj.weight.data.to(torch.bfloat16)
    elif mode == "converted":
        shared._deepgemm_mega_moe_weights_transformed = True
    elif mode == "activation":
        shared.act_fn.swiglu_limit = 8.0
    elif mode == "bias":
        shared.down_proj.bias = torch.nn.Parameter(torch.zeros(256))
    elif mode == "gate":
        shared.expert_gate = torch.nn.Identity()
    with pytest.raises((RuntimeError, ValueError)):
        backend.DeepGemmExperts(item)
    dependency.transform_weights_for_mega_moe.assert_not_called()


@pytest.mark.parametrize("scaling_factor", [1.0, 1.5])
def test_apply_fuses_shared_once_and_scales_only_routed_weights(context, scaling_factor):
    backend, config, _, dependency = context
    config.model_config.hf_text_config.n_shared_experts = 1
    item, shared = layer(), shared_module()
    item._deepgemm_mega_moe_shared_experts = weakref.ref(shared)
    item.ascend_megamoe_fold_scale = scaling_factor
    shared.forward = MagicMock(side_effect=AssertionError("Shared experts must only execute inside MegaMoE"))
    adapter = backend.DeepGemmExperts(item)
    dependency.fp8_fp4_mega_moe.reset_mock()
    weights = torch.full((2, 6), 0.25)
    indices = torch.tensor([[0, 48, 96, 144, 192, 240]] * 2, dtype=torch.int32)
    x = torch.ones((2, 256), dtype=torch.bfloat16)
    quantized = x.to(torch.float8_e4m3fn)
    scales = torch.full((2, 8), 127, dtype=torch.uint8)
    stream = SimpleNamespace(record_event=lambda: object())
    result_module = SimpleNamespace(FusedExpertsResult=lambda **kwargs: SimpleNamespace(**kwargs))
    dependency.fp8_fp4_mega_moe.side_effect = lambda out, *args, **kwargs: out.fill_(2)

    def quant_out(value, out, scale):
        assert out.data_ptr() == adapter.buffer.x.data_ptr()
        assert scale.data_ptr() == adapter.buffer.x_sf.data_ptr()
        out.copy_(quantized)
        scale.view(torch.uint8).copy_(scales.reshape(scale.shape))

    with (
        patch.object(torch, "npu", SimpleNamespace(current_stream=lambda: stream), create=True),
        patch.object(torch.ops._C_ascend, "dynamic_mx_quant_out", side_effect=quant_out, create=True),
        patch.dict(sys.modules, {"vllm_ascend.ops.fused_moe.moe_comm_method": result_module}),
    ):
        result = adapter.apply(item, x, weights, indices)
    assert torch.equal(adapter.buffer.topk_weights[:2], weights * scaling_factor)
    assert torch.equal(adapter.buffer.topk_idx[:2], indices)
    assert torch.equal(weights, torch.full((2, 6), 0.25))
    assert torch.equal(result.routed_out, torch.full_like(x, 2))
    kwargs = dependency.fp8_fp4_mega_moe.call_args.kwargs
    assert kwargs["shared_l1_weights"] is adapter.shared_l1_weights
    assert kwargs["shared_l2_weights"] is adapter.shared_l2_weights
    shared.forward.assert_not_called()
    dependency.fp8_fp4_mega_moe.assert_called_once()


@pytest.mark.parametrize("tokens", [0, 1024])
def test_large_prefill_does_not_synchronize_host(context, tokens):
    backend, config, _, dependency = context
    config.scheduler_config.max_num_batched_tokens = 1024
    adapter = backend.DeepGemmExperts(layer())
    dependency.fp8_fp4_mega_moe.reset_mock()
    stream = MagicMock()

    def quant(x, out, scale):
        out.copy_(x.to(torch.float8_e4m3fn))
        scale.view(torch.uint8).fill_(127)

    with (
        patch.object(torch, "npu", SimpleNamespace(current_stream=lambda: stream), create=True),
        patch.object(torch.ops._C_ascend, "dynamic_mx_quant_out", side_effect=quant, create=True),
        patch.dict(
            sys.modules,
            {"vllm_ascend.ops.fused_moe.moe_comm_method": SimpleNamespace(FusedExpertsResult=SimpleNamespace)},
        ),
    ):
        adapter.apply(
            layer(),
            torch.ones((tokens, 256), dtype=torch.bfloat16),
            torch.ones((tokens, 6)),
            torch.zeros((tokens, 6), dtype=torch.int32),
        )
    stream.synchronize.assert_not_called()
    stream.record_event.assert_not_called()
    dependency.fp8_fp4_mega_moe.assert_called_once()


@pytest.mark.parametrize("name,value", [("expert_placement_strategy", "round_robin"), ("enable_dbo", True)])
def test_rejects_incompatible_symmetric_buffer_scheduling(context, name, value):
    backend, config, _, dependency = context
    setattr(config.parallel_config, name, value)
    with pytest.raises(ValueError):
        backend.DeepGemmExperts(layer())
    dependency.SymmBuffer.assert_not_called()
    dependency.transform_weights_for_mega_moe.assert_not_called()
