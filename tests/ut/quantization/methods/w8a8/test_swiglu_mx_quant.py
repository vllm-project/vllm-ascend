# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
from vllm.model_executor.layers.fused_moe.activation import MoEActivation

from vllm_ascend.ops.fused_moe.moe_mlp import apply_moe_mlp
from vllm_ascend.quantization.methods.w8a8 import w8a8_mxfp8 as module
from vllm_ascend.quantization.utils import DynamicMXQuantScaleAlg


def make_method(scale_alg):
    config = SimpleNamespace(quant_config=SimpleNamespace(), use_v2_model_runner=True)
    with (
        patch.object(module, "get_current_vllm_config", return_value=config),
        patch.object(module, "get_ascend_config"),
        patch.object(module, "get_dynamic_mx_quant_scale_alg", return_value=scale_alg) as get_scale_alg,
    ):
        method = module.AscendW8A8MXFP8DynamicFusedMoEMethod()
    get_scale_alg.assert_called_once_with(config)
    return method


@pytest.mark.parametrize("scale_alg", list(DynamicMXQuantScaleAlg))
@pytest.mark.parametrize("activation", list(MoEActivation))
def test_fused_activation_dispatch(scale_alg, activation):
    method = make_method(scale_alg)
    expected = activation == MoEActivation.SILU or (
        scale_alg == DynamicMXQuantScaleAlg.CEIL and activation == MoEActivation.SWIGLUOAI_UNINTERLEAVE
    )
    assert method.supports_fused_activation(activation) == expected


def test_enabling_one_model_does_not_change_other_instances():
    enabled = make_method(1)
    other_model = make_method(0)
    assert enabled.supports_fused_activation(MoEActivation.SWIGLUOAI_UNINTERLEAVE)
    assert not other_model.supports_fused_activation(MoEActivation.SWIGLUOAI_UNINTERLEAVE)
    assert "fused_activations" not in vars(enabled)
    assert "fused_activations" not in vars(other_model)
    assert module.AscendW8A8MXFP8DynamicFusedMoEMethod.fused_activations == frozenset({"silu"})


@pytest.mark.parametrize("scale_alg", [0, 1])
def test_activation_quantization_reuses_cached_scale_algorithm(scale_alg):
    method = make_method(scale_alg)
    hidden_states = torch.randn(4, 64, dtype=torch.bfloat16)
    quantized = torch.zeros(4, 64, dtype=torch.float8_e4m3fn)
    scales = torch.ones(4, 1, 2, dtype=torch.uint8)
    with (
        patch.object(module, "get_dynamic_mx_quant_scale_alg", side_effect=AssertionError("unexpected scale lookup")),
        patch.object(module.torch_npu, "npu_dynamic_mx_quant", return_value=(quantized, scales)) as quant,
    ):
        assert method.supports_fused_activation(MoEActivation.SWIGLUOAI_UNINTERLEAVE) == (scale_alg == 1)
        result, result_scale = method.apply_act_quant(None, hidden_states)
    quant.assert_called_once_with(hidden_states, dst_type=torch.float8_e4m3fn, scale_alg=scale_alg)
    assert result is quantized
    assert result_scale is scales


@pytest.mark.parametrize("scale_alg", [0, 1])
def test_mlp_dispatch_preserves_fallback_and_gmm2_scales(scale_alg):
    method = make_method(scale_alg)
    packed = torch.randn(4, 128, dtype=torch.bfloat16)
    quantized = torch.zeros(4, 64, dtype=torch.float8_e4m3fn)
    scales = torch.ones(4, 1, 2, dtype=torch.uint8)
    method.apply_gmm1 = Mock(return_value=packed)
    method.apply_act_quant = Mock(return_value=(quantized, scales))
    method.apply_gmm2 = Mock(return_value="result")
    inputs = SimpleNamespace(
        activation=MoEActivation.SWIGLUOAI_UNINTERLEAVE,
        quant=SimpleNamespace(is_quant=True),
        lora_context=None,
        swiglu_limit=5.0,
        swiglu_alpha=1.5,
        swiglu_beta=0.5,
    )
    with (
        patch.object(module, "get_dynamic_mx_quant_scale_alg", side_effect=AssertionError("unexpected scale lookup")),
        patch.object(module.torch_npu, "npu_swiglu_mx_quant", return_value=(quantized, scales), create=True) as fused,
        patch("vllm_ascend.ops.fused_moe.moe_mlp.DeviceOperator.clipped_swiglu", return_value="activated") as legacy,
        patch("vllm_ascend.ops.fused_moe.moe_mlp.maybe_record_event", return_value=None),
    ):
        assert apply_moe_mlp(inputs, method) == ("result", None)
    method.apply_gmm1.assert_called_once_with(inputs)
    method.apply_gmm2.assert_called_once_with(inputs, quantized, scales)
    if scale_alg == 1:
        fused.assert_called_once_with(
            packed,
            activate_left=True,
            swiglu_mode=2,
            clamp_limit=5.0,
            glu_alpha=1.5,
            glu_bias=0.5,
            dst_type=torch.float8_e4m3fn,
            scale_alg=1,
        )
        legacy.assert_not_called()
        method.apply_act_quant.assert_not_called()
    else:
        fused.assert_not_called()
        legacy.assert_called_once_with(packed, swiglu_limit=5.0, swiglu_alpha=1.5, swiglu_beta=0.5)
        method.apply_act_quant.assert_called_once_with(inputs, "activated")
