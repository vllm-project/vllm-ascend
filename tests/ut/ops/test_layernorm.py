from unittest.mock import patch

import pytest
import torch
from vllm.model_executor.layers.layernorm import RMSNorm

from vllm_ascend.utils import AscendDeviceType


@pytest.fixture
def dummy_tensor():
    return torch.randn(4, 8, dtype=torch.float16)


def mock_npu_rms_norm(x, weight, eps):
    return x + 1, None


def mock_add_rmsnorm_bias(input, residual, norm_weight, norm_bias, eps):
    return 2 * input, 2 * residual


@pytest.mark.parametrize("is_310p", [True, False])
@pytest.mark.parametrize(
    "residual",
    [None, torch.randn(4, 8, dtype=torch.float32, device="cpu")])
@patch("torch.ops.vllm.add_rmsnorm_bias", side_effect=mock_add_rmsnorm_bias)
@patch("torch_npu.npu_rms_norm", side_effect=mock_npu_rms_norm)
def test_RMSNorm_forward(mock_add_rmsnorm, mock_rmsnorm, is_310p, residual,
                         dummy_tensor):

    with patch("vllm_ascend.utils.get_ascend_device_type",
               return_value=AscendDeviceType._310P
               if is_310p else AscendDeviceType.A3):
        layer = RMSNorm(hidden_size=8, eps=1e-05)
        output = layer.forward_oot(dummy_tensor, residual)

        if residual is not None:
            out_x, out_residual = output

            if is_310p:
                expected_arg_x = dummy_tensor + residual.to(dummy_tensor.dtype)
                expected_out_x = expected_arg_x + 1
                expected_out_residual = expected_arg_x.to(residual.dtype)

                mock_add_rmsnorm.assert_called_once()
                assert torch.allclose(out_x, expected_out_x, atol=1e-5)
                assert torch.allclose(out_residual,
                                      expected_out_residual,
                                      atol=1e-5)
            else:
                expected_out_x = 2 * dummy_tensor
                expected_out_residual = 2 * residual
                mock_rmsnorm.assert_called_once()
                assert torch.allclose(out_x, expected_out_x, atol=1e-5)
                assert torch.allclose(
                    out_residual, 
                    expected_out_residual, 
                    atol=1e-5)
        else:
            out_x = output
            expected_out_x = 2 * dummy_tensor
            mock_rmsnorm.assert_called_once()

            assert torch.allclose(out_x, expected_out_x, atol=1e-5)
