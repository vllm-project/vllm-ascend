from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from vllm.model_executor.layers.fused_moe.activation import MoEActivation

from vllm_ascend.quantization.methods.w4a8 import w4a8_mxfp4 as mxfp
from vllm_ascend.utils import AscendDeviceType


@pytest.mark.parametrize("group_list_type", [0, 1])
@pytest.mark.parametrize("prequantized", [False, True])
def test_gmm_situ_quant_reuses_input_quantization_and_gmm2(group_list_type, prequantized):
    method = object.__new__(mxfp.AscendW4A8MXFPDynamicFusedMoEMethod)
    method.group_size = 32
    hidden_states = torch.empty(4, 128, dtype=torch.bfloat16)
    quantized = torch.empty(4, 128, dtype=torch.float8_e4m3fn)
    dynamic_scale = torch.empty(4, 2, 2, dtype=torch.uint8)
    activated = torch.empty(4, 64, dtype=torch.float8_e4m3fn)
    act_scale = torch.empty(4, 1, 2, dtype=torch.uint8)
    final = torch.empty_like(hidden_states)
    layer = SimpleNamespace(
        w13_weight=torch.empty(2, 64, 128, dtype=torch.uint8),
        w13_weight_scale=torch.empty(2, 2, 128, 2, dtype=torch.uint8),
        w2_weight=torch.empty(2, 32, 128, dtype=torch.uint8),
        w2_weight_scale=torch.empty(2, 1, 128, 2, dtype=torch.uint8),
    )
    inputs = SimpleNamespace(
        hidden_states=quantized if prequantized else hidden_states,
        dynamic_scale=dynamic_scale if prequantized else None,
        layer=layer,
        group_list=torch.tensor([1, 4] if group_list_type == 0 else [1, 3], dtype=torch.int64),
        group_list_type=group_list_type,
        activation=MoEActivation.SITU,
        activation_situ_beta=4.0,
        activation_situ_linear_beta=25.0,
    )
    with (
        patch.object(mxfp, "get_ascend_device_type", return_value=AscendDeviceType.A5),
        patch.object(method, "_quant_hidden_states", return_value=(quantized, dynamic_scale)) as quantize,
        patch.object(
            mxfp.DeviceOperator, "npu_grouped_matmul_situ_quant", return_value=(activated, act_scale, None)
        ) as fused,
        patch.object(mxfp, "maybe_normalize_mxfp_scale_layout", side_effect=lambda x: x),
        patch.object(mxfp, "dispose_tensor") as dispose,
        patch.object(mxfp.torch_npu, "npu_grouped_matmul", return_value=[final], create=True) as gmm,
    ):
        out, scale = method.apply_gmm1_act_quant(inputs)
        result = method.apply_gmm2(inputs, out, scale)

    quantize.assert_called_once_with(inputs.hidden_states, inputs.dynamic_scale)
    dispose.assert_called_once_with(inputs.hidden_states)
    assert out is activated and scale is act_scale and result is final
    assert fused.call_args.kwargs["x"] is quantized
    assert fused.call_args.kwargs["x_scale"] is dynamic_scale
    assert fused.call_args.kwargs["weight"] is layer.w13_weight
    assert fused.call_args.kwargs["group_list"] is inputs.group_list
    assert fused.call_args.kwargs["group_list_type"] == group_list_type
    assert fused.call_args.kwargs["beta"] == 4.0
    assert fused.call_args.kwargs["linear_beta"] == 25.0
    gmm.assert_called_once()
    assert gmm.call_args.kwargs["weight"][0] is layer.w2_weight
    assert gmm.call_args.kwargs["x"][0] is activated
    assert gmm.call_args.kwargs["per_token_scale"][0] is act_scale


@pytest.mark.parametrize("linear_beta,group_list_type", [(None, 0), (0.0, 1), (25.0, 2)])
def test_unsupported_gmm_situ_quant_parameters_keep_split_path(linear_beta, group_list_type):
    method = object.__new__(mxfp.AscendW4A8MXFPDynamicFusedMoEMethod)
    method.group_size = 32
    hidden_states = torch.empty(4, 128, dtype=torch.bfloat16)
    inputs = SimpleNamespace(
        hidden_states=hidden_states,
        dynamic_scale=None,
        layer=SimpleNamespace(w13_weight=object(), w13_weight_scale=object()),
        group_list=torch.tensor([1, 3], dtype=torch.int64),
        group_list_type=group_list_type,
        activation=MoEActivation.SITU,
        activation_situ_beta=4.0,
        activation_situ_linear_beta=linear_beta,
    )
    split_out, split_scale = object(), object()
    with (
        patch.object(mxfp, "get_ascend_device_type", return_value=AscendDeviceType.A5),
        patch.object(method, "_quant_hidden_states", return_value=(hidden_states, object())),
        patch.object(mxfp.DeviceOperator, "npu_grouped_matmul_situ_quant") as fused,
        patch.object(mxfp, "dispose_tensor"),
        patch.object(mxfp, "maybe_normalize_mxfp_scale_layout", side_effect=lambda x: x),
        patch.object(mxfp.torch_npu, "npu_grouped_matmul", return_value=[hidden_states], create=True) as gmm,
        patch.object(torch.ops._C_ascend, "situ_mx_quant", return_value=(split_out, split_scale), create=True) as situ,
    ):
        out, scale = method.apply_gmm1_act_quant(inputs)

    assert out is split_out and scale is split_scale
    fused.assert_not_called()
    gmm.assert_called_once()
    situ.assert_called_once()


@pytest.mark.parametrize("activation", ["situ", "silu"])
def test_gmm_situ_quant_w13_loader_preserves_native_fp4_metadata(activation):
    method = object.__new__(mxfp.AscendW4A8MXFPDynamicFusedMoEMethod)
    layer = torch.nn.Module()
    layer.activation = activation
    for name, shape in (
        ("w13_weight", (2, 128, 64)),
        ("w2_weight", (2, 128, 32)),
        ("w13_weight_scale", (2, 128, 4)),
        ("w2_weight_scale", (2, 128, 2)),
    ):
        setattr(layer, name, torch.nn.Parameter(torch.zeros(shape, dtype=torch.uint8), requires_grad=False))
    original_w13_bytes = layer.w13_weight.data_ptr()
    with patch.object(mxfp.torch_npu, "npu_format_cast", side_effect=lambda x, *a, **kw: x) as cast:
        method.process_weights_after_loading(layer)

    expected_dtype = torch.float4_e2m1fn_x2 if activation == "situ" else torch.uint8
    assert layer.w13_weight.dtype == expected_dtype
    assert layer.w13_weight.data_ptr() == original_w13_bytes
    assert layer.w2_weight.dtype == torch.uint8
    assert cast.call_args_list[0].kwargs["input_dtype"] == (
        torch.float4_e2m1fn_x2 if activation == "situ" else mxfp.torch_npu.float4_e2m1fn_x2
    )
    assert cast.call_args_list[1].kwargs["input_dtype"] == mxfp.torch_npu.float4_e2m1fn_x2
