from unittest.mock import MagicMock, patch

import pytest
import torch
from vllm.config import set_current_vllm_config
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.third_party.flash_linear_attention.ops.kda import FusedRMSNormGated

from vllm_ascend.ops.layernorm import AscendFusedRMSNormGated, AscendRMSNorm
from vllm_ascend.utils import enable_custom_op
from vllm_ascend.utils import is_310p as is_310p_hw

enable_custom_op()


@pytest.fixture
def dummy_tensor():
    return torch.randn(4, 8, dtype=torch.float16)


def mock_rms_norm(x, weight, eps):
    return x + 1, None


def mock_add_rms_norm(x, residual, weight, eps):
    return 2 * x, None, 2 * residual


def mock_add_rms_norm_bias(x, residual, weight, bias, eps):
    if bias is None:
        return 2 * x, None, 2 * residual
    else:
        return 2 * x + bias, None, 2 * residual


@pytest.fixture(autouse=True)
def default_vllm_config():
    mock_config = MagicMock()
    mock_config.compilation_config.custom_ops = ["all"]

    with set_current_vllm_config(mock_config):
        yield mock_config


@pytest.mark.skip("Skip as register_kernels has NPU SocName checking in CANN 8.5.0.")
@pytest.mark.parametrize("residual", [None, torch.randn(4, 8, dtype=torch.float32)])
@patch("torch_npu.npu_rms_norm", side_effect=mock_rms_norm)
@patch("torch_npu.npu_add_rms_norm", side_effect=mock_add_rms_norm)
@patch("torch.ops._C_ascend.npu_add_rms_norm_bias", side_effect=mock_add_rms_norm_bias)
def test_RMSNorm_forward(
    mock_add_rms_norm_bias, mock_add_rmsnorm, mock_rmsnorm, residual, dummy_tensor, default_vllm_config
):
    layer = RMSNorm(hidden_size=8, eps=1e-05)
    if residual is not None:
        out_x, out_residual = layer.forward_oot(dummy_tensor, residual)
        expected_out_x = 2 * dummy_tensor
        expected_out_residual = 2 * residual
        mock_add_rms_norm_bias.assert_called_once()
        assert torch.allclose(out_x, expected_out_x)
        assert torch.allclose(out_residual, expected_out_residual)
    else:
        out_x = layer.forward_oot(dummy_tensor, residual)
        expected_out_x = dummy_tensor + 1

        mock_rmsnorm.assert_called_once()
        assert torch.allclose(out_x, expected_out_x)


def test_RMSNorm_supports_quant_config_without_quant_description(default_vllm_config):
    default_vllm_config.quant_config = object()

    layer = RMSNorm(hidden_size=8, eps=1e-05)

    assert layer.bias is None


def test_RMSNorm_creates_bias_from_quant_description(default_vllm_config):
    quant_config = MagicMock()
    quant_config.quant_description = {"model.layers.0.input_layernorm.bias": "W8A8"}
    default_vllm_config.quant_config = quant_config

    layer = RMSNorm(hidden_size=8, eps=1e-05)

    assert layer.bias is not None
    assert not layer.bias.requires_grad


def test_FusedRMSNormGated_dispatches_to_ascend_kernel(default_vllm_config):
    layer = FusedRMSNormGated(hidden_size=8, eps=1e-6, activation="sigmoid")
    x = torch.randn(1, 4, 2, 8)
    gate = torch.randn(4, 2, 8)
    residual = torch.randn_like(x)
    expected = (torch.empty_like(x), torch.empty_like(x))

    with patch("vllm_ascend.ops.layernorm.rms_norm_gated", return_value=expected) as fused_norm_gate:
        actual = layer(x, gate, residual=residual, prenorm=True, residual_in_fp32=True)

    assert isinstance(layer, AscendFusedRMSNormGated)
    assert actual is expected
    fused_norm_gate.assert_called_once_with(
        x,
        gate,
        layer.weight,
        layer.bias,
        "sigmoid",
        residual=residual,
        eps=1e-6,
        prenorm=True,
        residual_in_fp32=True,
    )


@pytest.mark.skipif(not is_310p_hw(), reason="310P device unittest case.")
@pytest.mark.parametrize("residual", [None, torch.randn(4, 8, dtype=torch.float16)])
@patch("torch_npu.npu_rms_norm", side_effect=mock_rms_norm)
@patch("torch_npu.npu_add_rms_norm", side_effect=mock_add_rms_norm)
def test_RMSNorm_forward_310p(mock_add_rmsnorm, mock_rmsnorm, residual, dummy_tensor, default_vllm_config):
    layer = RMSNorm(hidden_size=8, eps=1e-05)
    if residual is not None:
        out_x, out_residual = layer.forward_oot(dummy_tensor, residual)
        expected_out_x = 2 * dummy_tensor
        expected_out_residual = 2 * residual
        mock_add_rmsnorm.assert_called_once()
        assert torch.allclose(out_x, expected_out_x)
        assert torch.allclose(out_residual, expected_out_residual)
    else:
        out_x = layer.forward_oot(dummy_tensor, residual)
        expected_out_x = dummy_tensor + 1
        mock_rmsnorm.assert_called_once()
        assert torch.allclose(out_x, expected_out_x)


@patch("torch_npu.npu_add_rms_norm", side_effect=mock_add_rms_norm)
@patch("torch.ops._C_ascend.npu_add_rms_norm_bias", side_effect=mock_add_rms_norm_bias)
def test_mistral4_uses_native_add_rms_norm(
    mock_add_rms_norm_bias,
    mock_add_rmsnorm,
    dummy_tensor,
):
    mock_config = MagicMock()
    mock_config.model_config.hf_config.model_type = "mistral3"
    mock_config.model_config.hf_config.text_config.model_type = "mistral4"
    mock_config.quant_config = None
    mock_config.compilation_config.custom_ops = ["all"]
    residual = torch.randn_like(dummy_tensor)

    with (
        set_current_vllm_config(mock_config),
        patch("vllm_ascend.ops.layernorm.enable_custom_op", return_value=True),
    ):
        layer = AscendRMSNorm(hidden_size=8, eps=1e-5)
        out_x, out_residual = layer.forward_oot(dummy_tensor, residual)

    mock_add_rmsnorm.assert_called_once()
    mock_add_rms_norm_bias.assert_not_called()
    assert torch.allclose(out_x, 2 * dummy_tensor)
    assert torch.allclose(out_residual, 2 * residual)
