# SPDX-License-Identifier: Apache-2.0
# Copyright contributors to the vllm-ascend project

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from vllm_ascend.ops.gdn import AscendGatedDeltaNetAttention


def make_layer():
    return SimpleNamespace(
        key_dim=2048,
        value_dim=6144,
        tp_size=2,
        head_k_dim=128,
        head_v_dim=128,
    )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_supported_layout_uses_custom_op(dtype):
    layer = make_layer()
    source = torch.arange(17 * 5120, dtype=torch.float32).reshape(17, 5120)
    mixed_qkv = source.to(dtype)
    expected_parts = source.split([1024, 1024, 3072], dim=-1)
    packed_qkv = torch.cat([part.flatten() for part in expected_parts]).to(dtype)
    custom_op = Mock(return_value=packed_qkv)

    with patch.object(torch.ops, "_C_ascend", SimpleNamespace(npu_rearrange_qkv=custom_op)):
        outputs = AscendGatedDeltaNetAttention.rearrange_mixed_qkv(layer, mixed_qkv)

    custom_op.assert_called_once_with(mixed_qkv, 1024, 1024, 3072)
    for output, expected, heads in zip(outputs, expected_parts, (8, 8, 24)):
        assert output.shape == (1, 17, heads, 128)
        torch.testing.assert_close(output.reshape(17, -1), expected.to(dtype), rtol=0, atol=0)


def test_none_input_returns_empty_outputs():
    layer = make_layer()
    custom_op = Mock()
    with patch.object(torch.ops, "_C_ascend", SimpleNamespace(npu_rearrange_qkv=custom_op)):
        outputs = AscendGatedDeltaNetAttention.rearrange_mixed_qkv(layer, None)

    assert outputs == (None, None, None)
    custom_op.assert_not_called()
