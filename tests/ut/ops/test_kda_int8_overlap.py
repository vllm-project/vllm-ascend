# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from vllm_ascend.device.hardware import AscendDeviceType
from vllm_ascend.device.hardware_profile import get_hardware_profile
from vllm_ascend.ops.kimi_kda import AscendKimiK3DeltaAttention
from vllm_ascend.quantization.methods.w8a8.w8a8_dynamic import AscendW8A8DynamicLinearMethod
from vllm_ascend.quantization.prepared_linear_input import PreparedLinearInput


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("wrapped", [False, True])
def test_quantize_once_and_reuse_scales(dtype, wrapped):
    scheme = AscendW8A8DynamicLinearMethod()
    method = SimpleNamespace(quant_method=scheme, apply=scheme.apply) if wrapped else scheme
    layer = SimpleNamespace(quant_method=method, weight=object(), weight_scale=object())
    attention = SimpleNamespace(in_proj_qkvgfab=layer)
    x = torch.randn(4, 16, dtype=dtype)
    q, scale, result = torch.zeros_like(x, dtype=torch.int8), torch.ones(4), object()
    with (
        patch(
            "vllm_ascend.ops.kimi_kda.get_current_hardware_profile",
            return_value=get_hardware_profile(AscendDeviceType.A3),
        ),
        patch("torch_npu.npu_dynamic_quant", return_value=(q, scale)) as quant,
        patch("torch_npu.npu_quant_matmul", return_value=result) as matmul,
    ):
        prepared = AscendKimiK3DeltaAttention._quantize_fused_qkv(attention, x)
        assert isinstance(prepared, PreparedLinearInput)
        assert prepared.quantized is q and prepared.scale is scale
        output = AscendKimiK3DeltaAttention._matmul_fused_qkv(attention, prepared)
    assert output is result
    quant.assert_called_once_with(x, dst_type=torch.int8)
    assert matmul.call_args.args[0] is q
    assert matmul.call_args.kwargs["pertoken_scale"] is scale
    assert matmul.call_args.kwargs["output_dtype"] == dtype


@pytest.mark.parametrize(
    "enabled,dtype,shape",
    [
        (False, torch.bfloat16, (4, 16)),  # A5/A2 keep their original late INT8 quantization.
        (True, torch.float32, (4, 16)),
        (True, torch.bfloat16, (1, 4, 16)),
    ],
)
def test_unsupported_inputs_keep_original_path(enabled, dtype, shape):
    scheme = AscendW8A8DynamicLinearMethod()
    attention = SimpleNamespace(in_proj_qkvgfab=SimpleNamespace(quant_method=scheme))
    x = torch.randn(shape, dtype=dtype)
    with (
        patch(
            "vllm_ascend.ops.kimi_kda.get_current_hardware_profile",
            return_value=get_hardware_profile(AscendDeviceType.A3 if enabled else AscendDeviceType.A5),
        ),
        patch("torch_npu.npu_dynamic_quant") as quant,
    ):
        assert AscendKimiK3DeltaAttention._quantize_fused_qkv(attention, x) is x
    quant.assert_not_called()


def test_prepare_decline_falls_back():
    scheme = AscendW8A8DynamicLinearMethod()
    attention = SimpleNamespace(in_proj_qkvgfab=SimpleNamespace(quant_method=scheme))
    x = torch.randn(4, 16, dtype=torch.bfloat16)
    with (
        patch(
            "vllm_ascend.ops.kimi_kda.get_current_hardware_profile",
            return_value=get_hardware_profile(AscendDeviceType.A3),
        ),
        patch.object(scheme, "prepare_input_for_overlap", return_value=None),
    ):
        assert AscendKimiK3DeltaAttention._quantize_fused_qkv(attention, x) is x
