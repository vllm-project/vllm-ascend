# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from vllm_ascend.device.hardware_profile import WeightLayoutPolicy
from vllm_ascend.ops import linear


def make_profile(is_310p=False):
    policy = WeightLayoutPolicy.FORCE_NZ if is_310p else WeightLayoutPolicy.CONFIGURABLE
    return SimpleNamespace(weight_layout_policy=policy)


def make_layer(weight, prefix="model.layers.0.linear_attn.conv1d"):
    layer = torch.nn.Module()
    layer.prefix = prefix
    layer.weight = torch.nn.Parameter(weight, requires_grad=False)
    return layer


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("channels", [2048, 4096, 8192])
def test_gdn_conv_weight_layout_preserves_values_and_loader(dtype, channels):
    expected = torch.arange(channels * 4, dtype=torch.float32).reshape(channels, 1, 4).to(dtype)
    layer = make_layer(expected.clone())
    parameter = layer.weight
    loader = object()
    parameter.weight_loader = loader
    parameter.input_dim = 1
    parameter.output_dim = 0
    method = linear.AscendUnquantizedLinearMethod()

    with patch.object(linear, "get_current_hardware_profile", return_value=make_profile()):
        method.process_weights_after_loading(layer)
        kernel_weight = layer.weight.view(channels, 4).transpose(0, 1)
        assert layer.weight is parameter
        assert layer.weight.weight_loader is loader
        assert (parameter.input_dim, parameter.output_dim) == (1, 0)
        assert parameter.shape == expected.shape
        assert torch.equal(parameter, expected)
        assert kernel_weight.is_contiguous()
        assert kernel_weight.data_ptr() == parameter.data_ptr()
        assert parameter.untyped_storage().nbytes() == parameter.numel() * parameter.element_size()

        pointer = parameter.data_ptr()
        method.process_weights_after_loading(layer)
        assert parameter.data_ptr() == pointer
        assert torch.equal(parameter, expected)

        # A loader copying new checkpoint values must respect strided storage.
        with torch.no_grad():
            parameter.copy_(expected + 1)
        method.process_weights_after_loading(layer)
        assert parameter.data_ptr() == pointer
        assert torch.equal(parameter, expected + 1)


@pytest.mark.parametrize(
    "prefix,shape,dtype,is_310p",
    [
        ("model.layers.0.mamba.conv1d", (64, 1, 4), torch.float32, False),
        ("model.layers.0.linear_attn.conv1d_extra", (64, 1, 4), torch.float32, False),
        ("model.layers.0.linear_attn.conv1d", (64, 4), torch.float32, False),
        ("model.layers.0.linear_attn.conv1d", (64, 2, 4), torch.float32, False),
        ("model.layers.0.linear_attn.conv1d", (64, 1, 4), torch.int8, False),
        ("model.layers.0.linear_attn.conv1d", (64, 1, 4), torch.float32, True),
    ],
)
def test_gdn_conv_weight_layout_fallback(prefix, shape, dtype, is_310p):
    layer = make_layer(torch.zeros(shape, dtype=dtype), prefix)
    pointer, stride = layer.weight.data_ptr(), layer.weight.stride()
    with patch.object(linear, "get_current_hardware_profile", return_value=make_profile(is_310p)):
        linear.AscendUnquantizedLinearMethod().process_weights_after_loading(layer)
    assert layer.weight.data_ptr() == pointer
    assert layer.weight.stride() == stride


def test_non_conv_weight_still_uses_original_format_conversion():
    layer = make_layer(torch.zeros(64, 4), prefix="model.layers.0.mlp.down_proj")
    with (
        patch.object(linear, "get_current_hardware_profile", return_value=make_profile()),
        patch.object(linear, "maybe_trans_nz", side_effect=lambda weight: weight) as conversion,
    ):
        linear.AscendUnquantizedLinearMethod().process_weights_after_loading(layer)
    conversion.assert_called_once()

