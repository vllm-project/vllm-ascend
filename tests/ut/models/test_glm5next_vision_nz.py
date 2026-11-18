# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from torch import nn
from vllm.model_executor.layers.linear import UnquantizedLinearMethod

from vllm_ascend.device.hardware import AscendDeviceType
from vllm_ascend.device.hardware_profile import get_hardware_profile
from vllm_ascend.models.glm5next import multimodal as vision
from vllm_ascend.ops.linear import unquantized_gemm


@pytest.mark.parametrize("device", [AscendDeviceType.A2, AscendDeviceType.A3, AscendDeviceType.A5])
@pytest.mark.parametrize("mode", [0, 1, 2])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("unquantized", [False, True])
def test_vision_nz_policy(monkeypatch, device, mode, dtype, unquantized):
    monkeypatch.setattr(vision, "get_current_hardware_profile", lambda: get_hardware_profile(device))
    monkeypatch.setattr(vision, "get_ascend_config", lambda: SimpleNamespace(weight_nz_mode=mode))
    layer = nn.Linear(32, 64, dtype=dtype)
    layer.quant_method = UnquantizedLinearMethod() if unquantized else object()
    expected = device == AscendDeviceType.A5 and mode == 2 and unquantized and dtype != torch.float32
    assert vision._requires_2d_nz_input(layer) == expected


@pytest.mark.parametrize("use_2d", [False, True])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("has_bias", [False, True])
@pytest.mark.parametrize("noncontiguous", [False, True])
@pytest.mark.parametrize("shape", [(32,), (7, 32), (2, 3, 32), (2, 2, 3, 32), (0, 1, 32)])
def test_vision_projection_preserves_values_and_scope(use_2d, dtype, has_bias, noncontiguous, shape):
    generator = torch.Generator().manual_seed(42)
    x = torch.randn(shape, generator=generator, dtype=dtype)
    if noncontiguous:
        x = torch.stack((x, x), dim=-1)[..., 0]
    weight = nn.Parameter(torch.randn(64, 32, generator=generator, dtype=dtype), requires_grad=False)
    bias = torch.randn(64, generator=generator, dtype=dtype) if has_bias else None
    calls = []

    def project(input_):
        calls.append(input_)
        return torch.nn.functional.linear(input_, weight, bias), bias

    expected = torch.nn.functional.linear(x.float(), weight.float(), bias.float() if bias is not None else None).to(
        dtype
    )
    actual, returned_bias = vision._vision_projection(project, x, use_2d)
    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=2e-2)
    assert returned_bias is bias
    assert len(calls) == 1
    if use_2d and x.ndim != 2:
        assert calls[0].ndim == 2
        if x.is_contiguous() and x.numel():
            assert calls[0].data_ptr() == x.data_ptr()
    else:
        assert calls[0] is x


@pytest.mark.parametrize("use_2d", [False, True])
def test_vision_mlp_keeps_activation_layout(monkeypatch, use_2d):
    model = object.__new__(vision.Glm5NextVisionMLP)
    nn.Module.__init__(model)
    inputs: list[torch.Tensor] = []

    def project(x: torch.Tensor) -> tuple[torch.Tensor, None]:
        inputs.append(x)
        return x, None

    model.gate_up_proj = Mock(side_effect=project)
    model.down_proj = Mock(side_effect=project)
    model._gate_up_2d = model._down_2d = use_2d
    model.act_fn = Mock(side_effect=lambda x: x + 1)
    x = torch.randn(3, 1, 8)
    projection = Mock(wraps=vision._vision_projection)
    monkeypatch.setattr(vision, "_vision_projection", projection)
    torch.testing.assert_close(model(x), x + 1)
    assert [v.ndim for v in inputs] == ([2, 2] if use_2d else [3, 3])
    assert model.act_fn.call_args.args[0].shape == x.shape
    assert projection.call_count == (2 if use_2d else 0)


@pytest.mark.parametrize("use_2d", [False, True])
def test_vision_attention_scopes_output_projection(monkeypatch, use_2d):
    model = object.__new__(vision.Glm5NextVisionAttention)
    nn.Module.__init__(model)
    model.head_dim = model.hidden_size_per_attention_head = 4
    model.num_attention_heads_per_partition = 2
    qkv = torch.randn(3, 1, 24)
    model.qkv = Mock(return_value=(qkv, None))
    model.q_norm = model.k_norm = SimpleNamespace(weight=torch.ones(4), variance_epsilon=1e-5)
    model.attn = Mock(side_effect=lambda **kwargs: kwargs["query"])
    model.proj = Mock(side_effect=lambda x: (x, None))
    model._proj_2d = use_2d
    monkeypatch.setattr(vision, "fused_q_kv_rmsnorm", lambda q, k, *args: (q, k))
    projection = Mock(wraps=vision._vision_projection)
    monkeypatch.setattr(vision, "_vision_projection", projection)
    actual = model(torch.zeros(3, 1, 8), torch.tensor([0, 3]), None, None)
    torch.testing.assert_close(actual, qkv[..., :8])
    assert model.proj.call_args.args[0].ndim == (2 if use_2d else 3)
    assert projection.call_count == (1 if use_2d else 0)


def test_shared_linear_keeps_native_dispatch(monkeypatch):
    x = torch.randn(3, 2, 32).transpose(0, 1)
    weight = torch.randn(64, 32)
    bias = torch.randn(64)
    sentinel = object()
    linear = Mock(return_value=sentinel)
    monkeypatch.setattr(torch.nn.functional, "linear", linear)
    assert unquantized_gemm(x, weight, bias) is sentinel
    assert linear.call_args.args[0] is x
    assert linear.call_args.args[1] is weight
    assert linear.call_args.args[2] is bias
