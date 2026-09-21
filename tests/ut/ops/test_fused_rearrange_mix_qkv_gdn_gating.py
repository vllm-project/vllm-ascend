# SPDX-License-Identifier: Apache-2.0
# Copyright contributors to the vllm-ascend project

from unittest.mock import Mock, patch

import torch

from vllm_ascend.ops import gdn as gdn_module
from vllm_ascend.ops.gdn import AscendGatedDeltaNetAttention


class LayerStub:
    _view_packed_qkv = AscendGatedDeltaNetAttention._view_packed_qkv
    rearrange_mixed_qkv_and_fused_gdn_gating = AscendGatedDeltaNetAttention.rearrange_mixed_qkv_and_fused_gdn_gating

    def rearrange_mixed_qkv(self, mixed_qkv):
        raise AssertionError("unexpected fallback")


def make_layer(**kwargs):
    attributes = {
        "key_dim": 2048,
        "value_dim": 6144,
        "tp_size": 2,
        "head_k_dim": 128,
        "head_v_dim": 128,
    }
    attributes.update(kwargs)
    layer = LayerStub()
    for name, value in attributes.items():
        setattr(layer, name, value)
    return layer


def test_supported_fused_layout_uses_device_adaptor():
    layer = make_layer()
    tokens = 17
    mixed_qkv = torch.randn(tokens, 5120, dtype=torch.bfloat16)
    a = torch.randn(tokens, 24, dtype=torch.bfloat16)
    b = torch.randn_like(a)
    A_log = torch.randn(24, dtype=torch.float32)
    dt_bias = torch.randn(24, dtype=torch.bfloat16)
    packed_qkv = torch.arange(tokens * 5120, dtype=torch.float32).to(torch.bfloat16)
    expected_g = torch.randn(tokens, 24, dtype=torch.float32)
    expected_beta = torch.randn(tokens, 24, dtype=torch.bfloat16)
    fused_op = Mock(return_value=(packed_qkv, expected_g, expected_beta))
    gating = Mock()

    with (
        patch.object(gdn_module.DeviceOperator, "fused_rearrange_mix_qkv_gdn_gating", fused_op),
        patch.object(gdn_module.DeviceOperator, "fused_gdn_gating", gating),
    ):
        query, key, value, g, beta = layer.rearrange_mixed_qkv_and_fused_gdn_gating(
            mixed_qkv,
            A_log,
            a,
            b,
            dt_bias,
        )

    fused_op.assert_called_once_with(mixed_qkv, a, b, A_log, dt_bias, 1024, 1024, 3072)
    gating.assert_not_called()
    assert query.shape == (1, tokens, 8, 128)
    assert key.shape == (1, tokens, 8, 128)
    assert value.shape == (1, tokens, 24, 128)
    torch.testing.assert_close(g, expected_g.unsqueeze(0))
    torch.testing.assert_close(beta, expected_beta.unsqueeze(0))


def test_adaptor_without_fused_implementation_uses_separate_paths():
    layer = make_layer()
    tokens = 3
    mixed_qkv = torch.randn(tokens, 5120, dtype=torch.bfloat16)
    a = torch.randn(tokens, 24, dtype=torch.bfloat16)
    b = torch.randn_like(a)
    A_log = torch.randn(24, dtype=torch.float32)
    dt_bias = torch.randn_like(A_log)
    expected_qkv = (Mock(), Mock(), Mock())
    expected_gating = (Mock(), Mock())

    with (
        patch.object(gdn_module.DeviceOperator, "fused_rearrange_mix_qkv_gdn_gating", return_value=None),
        patch.object(layer, "rearrange_mixed_qkv", return_value=expected_qkv) as rearrange,
        patch.object(gdn_module.DeviceOperator, "fused_gdn_gating", return_value=expected_gating) as gating,
    ):
        result = layer.rearrange_mixed_qkv_and_fused_gdn_gating(mixed_qkv, A_log, a, b, dt_bias)

    assert result == (*expected_qkv, *expected_gating)
    rearrange.assert_called_once_with(mixed_qkv)
    gating.assert_called_once_with(A_log, a, b, dt_bias)
