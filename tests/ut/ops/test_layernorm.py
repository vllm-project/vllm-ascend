import sys
from unittest.mock import MagicMock, patch

import pytest
import torch
from vllm.config import set_current_vllm_config
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.third_party.flash_linear_attention.ops.kda import FusedRMSNormGated

from vllm_ascend.ops.layernorm import (
    AscendFusedRMSNormGated,
    AscendRMSNorm,
    _add_bias_with_axpy,
    _enable_a5_add_rms_norm_bias,
)
from vllm_ascend.utils import AscendDeviceType, enable_custom_op
from vllm_ascend.utils import is_310p as is_310p_hw

enable_custom_op()


@pytest.mark.parametrize("width", [16, 128, 1024, 6144])
def test_a5_add_rms_norm_bias_loads_extension_only_when_supported(width):
    tensor = MagicMock(shape=(3, width))
    with (
        patch("vllm_ascend.envs.VLLM_ASCEND_ENABLE_ADD_RMS_NORM_BIAS", True),
        patch("vllm_ascend.ops.layernorm.get_ascend_device_type", return_value=AscendDeviceType.A5),
        patch("vllm.envs.VLLM_BATCH_INVARIANT", False),
        patch("vllm_ascend.ops.layernorm.bootstrap_custom_op_env") as bootstrap,
        patch.dict(sys.modules, {"vllm_ascend.vllm_ascend_C": MagicMock()}),
    ):
        assert _enable_a5_add_rms_norm_bias(tensor)
        bootstrap.assert_called_once_with(include_vendor_lib=True)


@pytest.mark.parametrize(
    "enabled, device, invariant, width",
    [
        (False, AscendDeviceType.A5, False, 6144),
        (True, AscendDeviceType.A2, False, 6144),
        (True, AscendDeviceType.A5, True, 6144),
        (True, AscendDeviceType.A5, False, 0),
        (True, AscendDeviceType.A5, False, 15),
        (True, AscendDeviceType.A5, False, 6145),
        (True, AscendDeviceType.A5, False, 8192),
    ],
)
def test_a5_add_rms_norm_bias_keeps_existing_path(enabled, device, invariant, width):
    tensor = MagicMock(shape=(3, width))
    with (
        patch("vllm_ascend.envs.VLLM_ASCEND_ENABLE_ADD_RMS_NORM_BIAS", enabled),
        patch("vllm_ascend.ops.layernorm.get_ascend_device_type", return_value=device),
        patch("vllm.envs.VLLM_BATCH_INVARIANT", invariant),
        patch("vllm_ascend.ops.layernorm.bootstrap_custom_op_env") as bootstrap,
    ):
        assert not _enable_a5_add_rms_norm_bias(tensor)
        bootstrap.assert_not_called()


def test_a5_add_rms_norm_bias_reports_missing_extension():
    tensor = MagicMock(shape=(3, 6144))
    with (
        patch("vllm_ascend.envs.VLLM_ASCEND_ENABLE_ADD_RMS_NORM_BIAS", True),
        patch("vllm_ascend.ops.layernorm.get_ascend_device_type", return_value=AscendDeviceType.A5),
        patch("vllm.envs.VLLM_BATCH_INVARIANT", False),
        patch("vllm_ascend.ops.layernorm.bootstrap_custom_op_env"),
        patch.dict(sys.modules, {"vllm_ascend.vllm_ascend_C": None}),
        pytest.raises(ImportError),
    ):
        _enable_a5_add_rms_norm_bias(tensor)


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


def test_add_bias_with_axpy_is_exact_and_uses_negative_alpha():
    x = torch.randn(4, 8, dtype=torch.float16)
    bias = torch.randn(8, dtype=torch.float16)
    torch.testing.assert_close(_add_bias_with_axpy(x.clone(), -bias), x + bias, atol=0, rtol=0)

    target = MagicMock()
    negative_bias = MagicMock()
    _add_bias_with_axpy(target, negative_bias)
    target.add_.assert_called_once_with(negative_bias, alpha=-1.0)


def test_bias_weight_loader_caches_negative_bias():
    layer = MagicMock()
    param = torch.nn.Parameter(torch.zeros(8), requires_grad=False)
    loaded_weight = torch.randn(8)

    AscendRMSNorm._bias_weight_loader(layer, param, loaded_weight)

    torch.testing.assert_close(param, loaded_weight)
    torch.testing.assert_close(layer._negative_bias, -loaded_weight)
    assert layer.bias_loaded is True


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
