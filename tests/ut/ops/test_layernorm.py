import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from vllm.config import set_current_vllm_config
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.third_party.flash_linear_attention.ops.kda import FusedRMSNormGated

from vllm_ascend.device.hardware_profile import HardwareCapability, get_current_hardware_profile
from vllm_ascend.ops import layernorm as ascend_layernorm
from vllm_ascend.ops.layernorm import AscendFusedRMSNormGated
from vllm_ascend.utils import enable_custom_op

enable_custom_op()


@pytest.fixture
def dummy_tensor():
    return torch.randn(4, 8, dtype=torch.float16)


def mock_rms_norm(x, weight, eps):
    return x + 1, None


def mock_add_rms_norm(x, residual, weight, eps):
    return 2 * x, None, 2 * residual


def mock_add_rms_norm_bias(x, residual, weight, bias, eps):
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
        mock_add_rmsnorm.assert_called_once()
        mock_add_rms_norm_bias.assert_not_called()
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


@pytest.mark.parametrize("has_bias", [False, True])
@pytest.mark.parametrize("bias_loaded", [False, True])
def test_rms_norm_residual_routes_by_bias_presence(has_bias, bias_loaded):
    # Exercise routing without CustomOp registration or an NPU kernel launch.
    x = torch.empty(4, 8)
    residual = torch.empty_like(x)
    layer = SimpleNamespace(
        weight=torch.ones(8),
        bias=torch.zeros(8) if has_bias else None,
        bias_loaded=bias_loaded,
        variance_epsilon=1e-6,
    )
    expected = (torch.empty_like(x), None, torch.empty_like(residual))
    with (
        patch.dict(sys.modules, {"vllm_ascend.vllm_ascend_C": MagicMock()}),
        patch.object(ascend_layernorm, "enable_custom_op") as enable_custom,
        patch("torch_npu.npu_add_rms_norm", return_value=expected) as native,
        patch.object(torch.ops._C_ascend, "npu_add_rms_norm_bias", return_value=expected, create=True) as custom,
    ):
        actual = ascend_layernorm.AscendRMSNorm.forward_oot(layer, x, residual)

    assert actual[0] is expected[0]
    assert actual[1] is expected[2]
    if has_bias:
        custom.assert_called_once_with(x, residual, layer.weight, layer.bias, layer.variance_epsilon)
        enable_custom.assert_called_once()
        native.assert_not_called()
    else:
        native.assert_called_once_with(x, residual, layer.weight, layer.variance_epsilon)
        custom.assert_not_called()
        enable_custom.assert_not_called()


@pytest.mark.parametrize("has_residual", [False, True])
@pytest.mark.parametrize("custom_ops_enabled", [False, True])
def test_gemma_rms_norm_always_uses_cann(has_residual, custom_ops_enabled):
    x = torch.empty(4, 8)
    residual = torch.empty_like(x) if has_residual else None
    layer = SimpleNamespace(weight=torch.arange(8, dtype=torch.float32), variance_epsilon=1e-6)
    y = torch.empty_like(x)
    residual_out = torch.empty_like(x)
    with (
        patch.object(ascend_layernorm, "enable_custom_op", return_value=custom_ops_enabled) as enable_custom,
        patch("torch_npu.npu_rms_norm", return_value=(y, None)) as rms,
        patch("torch_npu.npu_add_rms_norm", return_value=(y, None, residual_out)) as add_rms,
        patch.object(torch.ops._C_ascend, "npu_add_rms_norm_bias", create=True) as custom,
    ):
        actual = ascend_layernorm.AscendGemmaRMSNorm.forward_oot(layer, x, residual)

    custom.assert_not_called()
    enable_custom.assert_not_called()
    if has_residual:
        assert actual[0] is y
        assert actual[1] is residual_out
        add_rms.assert_called_once()
        assert add_rms.call_args.args[1] is residual
        args = add_rms.call_args.args
        gamma = args[2]
        rms.assert_not_called()
    else:
        assert actual is y
        rms.assert_called_once()
        args = rms.call_args.args
        gamma = args[1]
        add_rms.assert_not_called()
    assert args[0] is x
    assert args[-1] == layer.variance_epsilon
    torch.testing.assert_close(gamma, 1.0 + layer.weight)


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


@pytest.mark.skipif(
    get_current_hardware_profile().supports(HardwareCapability.STANDARD_WORKER_PATCHES),
    reason="310P device unittest case.",
)
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


class _CountingQuantDescription(dict):
    """Counts full scans so the cache can be asserted on."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.scans = 0

    def __iter__(self):
        self.scans += 1
        return super().__iter__()


@pytest.fixture
def clean_norm_bias_cache():
    ascend_layernorm._NORM_BIAS_IN_QUANT_DESCRIPTION.clear()
    yield
    ascend_layernorm._NORM_BIAS_IN_QUANT_DESCRIPTION.clear()


def test_norm_bias_lookup_scans_quant_description_once(clean_norm_bias_cache):
    """Every RMSNorm used to rescan the whole quant_description."""
    quant_description = _CountingQuantDescription(
        {
            "model.layers.0.self_attn.q_proj.weight": "W8A8",
            "model.layers.0.input_layernorm.norm.bias": "FLOAT",
        }
    )

    assert ascend_layernorm._quant_description_has_norm_bias(quant_description) is True
    assert ascend_layernorm._quant_description_has_norm_bias(quant_description) is True

    assert quant_description.scans == 1


def test_norm_bias_lookup_reports_missing_bias(clean_norm_bias_cache):
    quant_description = {"model.layers.0.self_attn.q_proj.weight": "W8A8"}

    assert ascend_layernorm._quant_description_has_norm_bias(quant_description) is False


def test_norm_bias_lookup_ignores_a_stale_entry(clean_norm_bias_cache):
    """An id can only be reused after the old dict is gone; never trust it blindly."""
    quant_description = {"model.layers.0.input_layernorm.norm.bias": "FLOAT"}
    ascend_layernorm._NORM_BIAS_IN_QUANT_DESCRIPTION[id(quant_description)] = ({}, False)

    assert ascend_layernorm._quant_description_has_norm_bias(quant_description) is True


def test_norm_bias_lookup_handles_empty_description(clean_norm_bias_cache):
    assert ascend_layernorm._quant_description_has_norm_bias({}) is False
    assert ascend_layernorm._quant_description_has_norm_bias(None) is False
    assert not ascend_layernorm._NORM_BIAS_IN_QUANT_DESCRIPTION
