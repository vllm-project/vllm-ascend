from unittest.mock import MagicMock, patch

import pytest
import torch
from vllm.config import set_current_vllm_config
from vllm.model_executor.layers.layernorm import RMSNormGated

import vllm_ascend._310p.ops.layernorm as layernorm_310
from vllm_ascend._310p.ops.layernorm import AscendRMSNorm310, AscendRMSNormGated310


@pytest.fixture(autouse=True)
def default_vllm_config():
    mock_config = MagicMock()
    mock_config.compilation_config.custom_ops = ["all"]
    mock_config.quant_config = None
    with set_current_vllm_config(mock_config):
        yield mock_config


@patch("torch.nn.functional.silu", side_effect=lambda tensor: tensor + 1)
@patch("torch_npu.npu_rms_norm")
def test_rmsnorm_gated_310_forward_oot_uses_rmsnorm_activation_mul(mock_rms_norm, mock_silu):
    layer = AscendRMSNormGated310(hidden_size=8, eps=1e-5, norm_before_gate=True)
    x = torch.randn(2, 8, dtype=torch.float32)
    z = torch.randn(2, 8, dtype=torch.float32)
    normed = torch.randn(2, 8, dtype=torch.float32)
    mock_rms_norm.return_value = (normed, None)

    with patch.object(RMSNormGated, "forward_native", autospec=True) as mock_forward_native:
        out = layer.forward_oot(x, z)

    mock_forward_native.assert_not_called()
    mock_rms_norm.assert_called_once()
    rms_norm_args = mock_rms_norm.call_args.args
    assert rms_norm_args[0] is x
    assert rms_norm_args[1] is layer.weight
    assert rms_norm_args[2] == layer.eps
    mock_silu.assert_called_once_with(z)
    assert torch.allclose(out, normed * (z + 1))


@patch("torch_npu.npu_rms_norm")
def test_rmsnorm_gated_310_forward_oot_uses_rmsnorm_without_gate(mock_rms_norm):
    layer = AscendRMSNormGated310(hidden_size=8, eps=1e-5)
    x = torch.randn(2, 8, dtype=torch.float32)
    expected = torch.randn(2, 8, dtype=torch.float32)
    mock_rms_norm.return_value = (expected, None)

    with patch.object(RMSNormGated, "forward_native", autospec=True, return_value=expected) as mock_forward_native:
        out = layer.forward_oot(x, None)

    mock_forward_native.assert_not_called()
    mock_rms_norm.assert_called_once()
    rms_norm_args = mock_rms_norm.call_args.args
    assert rms_norm_args[0] is x
    assert rms_norm_args[1] is layer.weight
    assert rms_norm_args[2] == layer.eps
    assert out is expected


def test_rmsnorm_gated_310_forward_oot_keeps_native_for_group_norm():
    layer = AscendRMSNormGated310(hidden_size=8, eps=1e-5, group_size=4)
    x = torch.randn(2, 8, dtype=torch.float32)
    z = torch.randn(2, 8, dtype=torch.float32)
    expected = torch.randn(2, 8, dtype=torch.float32)

    with patch.object(RMSNormGated, "forward_native", autospec=True, return_value=expected) as mock_forward_native:
        out = layer.forward_oot(x, z)

    mock_forward_native.assert_called_once_with(layer, x, z)
    assert out is expected


def test_rmsnorm_gated_310_uses_adn_dispatch_for_fp16():
    layer = AscendRMSNormGated310(hidden_size=256, eps=1e-5).to(dtype=torch.float16)
    x = torch.randn(2, 256, dtype=torch.float16)
    expected = torch.randn_like(x)
    fallback = torch.randn_like(x)

    with (
        patch.object(
            layernorm_310,
            "adn_rms_norm_or_fallback",
            return_value=expected,
        ) as experimental_dispatch,
        patch("torch_npu.npu_rms_norm", return_value=(fallback, None)) as baseline,
    ):
        out = layer.forward_oot(x)

    experimental_dispatch.assert_called_once_with(x, layer.weight, layer.eps)
    baseline.assert_not_called()
    assert out is expected


def test_rmsnorm_310_uses_adn_dispatch_without_residual():
    layer = AscendRMSNorm310(hidden_size=256, eps=1e-5, dtype=torch.float16)
    x = torch.randn(480, 256, dtype=torch.float16)
    expected = torch.randn_like(x)

    with (
        patch.object(
            layernorm_310,
            "adn_rms_norm_or_fallback",
            return_value=expected,
        ) as experimental_dispatch,
        patch("torch_npu.npu_rms_norm") as baseline,
    ):
        out = layer.forward_oot(x)

    experimental_dispatch.assert_called_once_with(
        x,
        layer.weight,
        layer.variance_epsilon,
    )
    baseline.assert_not_called()
    assert out is expected


def test_rmsnorm_310_keeps_fused_native_path_with_residual():
    layer = AscendRMSNorm310(hidden_size=256, eps=1e-5, dtype=torch.float16)
    x = torch.randn(2, 256, dtype=torch.float16)
    residual = torch.randn_like(x)
    expected = torch.randn_like(x)
    updated_residual = torch.randn_like(x)

    with (
        patch.object(layernorm_310, "adn_rms_norm_or_fallback") as experimental_dispatch,
        patch(
            "torch_npu.npu_add_rms_norm",
            return_value=(expected, None, updated_residual),
        ) as fused_baseline,
    ):
        out, out_residual = layer.forward_oot(x, residual)

    experimental_dispatch.assert_not_called()
    fused_baseline.assert_called_once_with(x, residual, layer.weight, layer.variance_epsilon)
    assert out is expected
    assert out_residual is updated_residual
