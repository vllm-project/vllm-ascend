# SPDX-License-Identifier: Apache-2.0
# Copyright contributors to the vllm-ascend project

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from vllm_ascend.ops import gdn as gdn_module
from vllm_ascend.ops.gdn import AscendGatedDeltaNetAttention


def make_layer(**kwargs):
    attributes = {
        "key_dim": 2048,
        "value_dim": 6144,
        "tp_size": 2,
        "head_k_dim": 128,
        "head_v_dim": 128,
    }
    attributes.update(kwargs)
    return SimpleNamespace(**attributes)


def make_input(tokens=17, dtype=torch.bfloat16, contiguous=True):
    return SimpleNamespace(dtype=dtype, shape=(tokens, 5120), is_contiguous=lambda: contiguous)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_supported_layout_uses_custom_op(dtype):
    layer = make_layer()
    mixed_qkv = make_input(dtype=dtype)
    source = torch.arange(17 * 5120, dtype=torch.float32).reshape(17, 5120)
    expected_parts = source.split([1024, 1024, 3072], dim=-1)
    packed_qkv = torch.cat([part.flatten() for part in expected_parts]).to(dtype)
    custom_op = Mock(return_value=packed_qkv)
    fallback = Mock()

    with (
        patch.object(gdn_module, "SUPPORTS_REARRANGE_QKV_DMA", True),
        patch.object(gdn_module, "_ORIGINAL_REARRANGE_MIXED_QKV", fallback),
        patch.object(torch.ops, "_C_ascend", SimpleNamespace(npu_rearrange_qkv=custom_op)),
    ):
        outputs = AscendGatedDeltaNetAttention.rearrange_mixed_qkv(layer, mixed_qkv)

    custom_op.assert_called_once_with(mixed_qkv, 1024, 1024, 3072)
    fallback.assert_not_called()
    for output, expected, heads in zip(outputs, expected_parts, (8, 8, 24)):
        assert output.shape == (1, 17, heads, 128)
        torch.testing.assert_close(output.reshape(17, -1), expected.to(dtype), rtol=0, atol=0)


def test_profile_without_capability_uses_original_implementation():
    layer = make_layer()
    mixed_qkv = make_input()
    fallback = Mock(return_value=(None, None, None))
    with (
        patch.object(gdn_module, "SUPPORTS_REARRANGE_QKV_DMA", False),
        patch.object(gdn_module, "_ORIGINAL_REARRANGE_MIXED_QKV", fallback),
    ):
        result = AscendGatedDeltaNetAttention.rearrange_mixed_qkv(layer, mixed_qkv)

    assert result == (None, None, None)
    fallback.assert_called_once_with(layer, mixed_qkv)


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
    fallback = Mock(return_value=(None, None, None))
    with (
        patch.object(gdn_module, "SUPPORTS_REARRANGE_QKV_DMA", True),
        patch.object(gdn_module, "_ORIGINAL_REARRANGE_MIXED_QKV", fallback),
    ):
        result = AscendGatedDeltaNetAttention.rearrange_mixed_qkv(layer, mixed_qkv)

    assert result == (None, None, None)
    fallback.assert_called_once_with(layer, mixed_qkv)


def test_none_uses_original_implementation():
    layer = make_layer()
    fallback = Mock(return_value=(None, None, None))
    with patch.object(gdn_module, "_ORIGINAL_REARRANGE_MIXED_QKV", fallback):
        assert AscendGatedDeltaNetAttention.rearrange_mixed_qkv(layer, None) == (None, None, None)
    fallback.assert_called_once_with(layer, None)
