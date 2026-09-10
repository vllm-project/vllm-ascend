# SPDX-License-Identifier: Apache-2.0
# Copyright contributors to the vllm-ascend project

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from vllm_ascend.ops import rearrange_qkv as rearrange_qkv_module
from vllm_ascend.utils import AscendDeviceType


def make_layer(**kwargs):
    attributes = {
        "key_dim": 2048,
        "value_dim": 6144,
        "tp_size": 2,
        "head_k_dim": 128,
        "head_v_dim": 128,
    }
    attributes.update(kwargs)
    return SimpleNamespace(
        **attributes,
        rearrange_mixed_qkv=Mock(return_value=(None, None, None)),
    )


def make_input(tokens=17, dtype=torch.bfloat16, contiguous=True):
    return SimpleNamespace(dtype=dtype, shape=(tokens, 5120), is_contiguous=lambda: contiguous)


@pytest.mark.parametrize("device_type", [AscendDeviceType.A2, AscendDeviceType.A3])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_supported_layout_uses_custom_op(device_type, dtype):
    layer = make_layer()
    mixed_qkv = make_input(dtype=dtype)
    source = torch.arange(17 * 5120, dtype=torch.float32).reshape(17, 5120)
    expected_parts = source.split([1024, 1024, 3072], dim=-1)
    packed_qkv = torch.cat([part.flatten() for part in expected_parts]).to(dtype)
    custom_op = Mock(return_value=packed_qkv)

    with (
        patch.object(
            rearrange_qkv_module,
            "SUPPORTS_REARRANGE_QKV",
            device_type in (AscendDeviceType.A2, AscendDeviceType.A3),
        ),
        patch.object(torch.ops, "_C_ascend", SimpleNamespace(npu_rearrange_qkv=custom_op)),
    ):
        outputs = rearrange_qkv_module.rearrange_mixed_qkv(layer, mixed_qkv)

    custom_op.assert_called_once_with(mixed_qkv, 1024, 1024, 3072)
    layer.rearrange_mixed_qkv.assert_not_called()
    for output, expected, heads in zip(outputs, expected_parts, (8, 8, 24)):
        assert output.shape == (1, 17, heads, 128)
        torch.testing.assert_close(output.reshape(17, -1), expected.to(dtype), rtol=0, atol=0)


@pytest.mark.parametrize("device_type", [AscendDeviceType._310P, AscendDeviceType.A5])
def test_unsupported_device_uses_original_implementation(device_type):
    layer = make_layer()
    mixed_qkv = make_input()
    with patch.object(
        rearrange_qkv_module,
        "SUPPORTS_REARRANGE_QKV",
        device_type in (AscendDeviceType.A2, AscendDeviceType.A3),
    ):
        result = rearrange_qkv_module.rearrange_mixed_qkv(layer, mixed_qkv)

    assert result == (None, None, None)
    layer.rearrange_mixed_qkv.assert_called_once_with(mixed_qkv)


@pytest.mark.parametrize(
    ("layer_attributes", "dtype", "contiguous"),
    [
        ({}, torch.float32, True),
        ({}, torch.bfloat16, False),
        ({"key_dim": 2032}, torch.bfloat16, True),
        ({"value_dim": 6128}, torch.bfloat16, True),
    ],
)
def test_unsupported_layout_uses_original_implementation(layer_attributes, dtype, contiguous):
    layer = make_layer(**layer_attributes)
    mixed_qkv = make_input(dtype=dtype, contiguous=contiguous)
    with patch.object(rearrange_qkv_module, "SUPPORTS_REARRANGE_QKV", True):
        result = rearrange_qkv_module.rearrange_mixed_qkv(layer, mixed_qkv)

    assert result == (None, None, None)
    layer.rearrange_mixed_qkv.assert_called_once_with(mixed_qkv)


def test_none_uses_original_implementation():
    layer = make_layer()
    assert rearrange_qkv_module.rearrange_mixed_qkv(layer, None) == (None, None, None)
    layer.rearrange_mixed_qkv.assert_called_once_with(None)
