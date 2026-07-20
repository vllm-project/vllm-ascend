from unittest.mock import MagicMock, patch

import pytest
import torch
from vllm.config import set_current_vllm_config
from vllm.model_executor.layers.layernorm import RMSNorm

from vllm_ascend.ops.layernorm import AscendGemmaRMSNorm
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


def mock_gemma_rms_norm(x, weight, eps):
    return x + weight, None


def mock_gemma_add_rms_norm(x, residual, weight, eps):
    return x + weight, None, residual + weight


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


def test_gemma_rms_norm_weight_loader_updates_weight_plus_one():
    layer = AscendGemmaRMSNorm(hidden_size=4, eps=1e-05)
    loaded_weight = torch.tensor([0.25, 0.5, 0.75, 1.0])

    layer.weight.weight_loader(layer.weight, loaded_weight)

    assert torch.allclose(layer.weight, loaded_weight)
    assert torch.allclose(layer.weight_plus_one, loaded_weight + 1.0)


@pytest.mark.parametrize("residual", [None, torch.randn(2, 4, dtype=torch.float32)])
@patch("torch_npu.npu_rms_norm", side_effect=mock_gemma_rms_norm, create=True)
@patch("torch_npu.npu_add_rms_norm", side_effect=mock_gemma_add_rms_norm, create=True)
@patch("torch.ops.vllm.maybe_chunk_residual", side_effect=lambda x, residual: residual)
def test_gemma_rms_norm_forward_uses_cached_weight_plus_one(
    mock_maybe_chunk_residual,
    mock_add_rmsnorm,
    mock_rmsnorm,
    residual,
):
    layer = AscendGemmaRMSNorm(hidden_size=4, eps=1e-05)
    loaded_weight = torch.tensor([0.25, 0.5, 0.75, 1.0])
    layer.weight.weight_loader(layer.weight, loaded_weight)
    x = torch.randn(2, 4, dtype=torch.float32)

    if residual is None:
        out_x = layer.forward_oot(x)
        mock_rmsnorm.assert_called_once()
        assert mock_rmsnorm.call_args.args[1] is layer.weight_plus_one
        assert torch.allclose(out_x, x + loaded_weight + 1.0)
        mock_add_rmsnorm.assert_not_called()
        mock_maybe_chunk_residual.assert_not_called()
    else:
        out_x, out_residual = layer.forward_oot(x, residual)
        mock_add_rmsnorm.assert_called_once()
        assert mock_add_rmsnorm.call_args.args[2] is layer.weight_plus_one
        assert torch.allclose(out_x, x + loaded_weight + 1.0)
        assert torch.allclose(out_residual, residual + loaded_weight + 1.0)
        mock_rmsnorm.assert_not_called()
        mock_maybe_chunk_residual.assert_called_once_with(x, residual)


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
