from unittest.mock import patch

import pytest
import torch
from vllm.model_executor.layers.layernorm import RMSNorm

from vllm_ascend.utils import AscendDeviceType


@pytest.fixture
def dummy_tensor():
    return torch.randn(4, 8, dtype=torch.float16)


def mock_rms_norm(x, weight, eps):
    return x + 1, None


def mock_add_rms_norm(x, residual, weight, eps):
    return 2 * x, None, 2 * residual


@pytest.mark.parametrize("is_310p", [True, False])
@pytest.mark.parametrize("residual",
                         [None, torch.randn(4, 8, dtype=torch.float32)])
@patch("torch_npu.npu_rms_norm", side_effect=mock_rms_norm)
@patch("torch_npu.npu_add_rms_norm", side_effect=mock_add_rms_norm)
def test_RMSNorm_forward(mock_add_rmsnorm, mock_rmsnorm, is_310p, residual,
                         dummy_tensor):

    with patch("vllm_ascend.utils.get_ascend_device_type",
               return_value=AscendDeviceType._310P
               if is_310p else AscendDeviceType._910_93):
        layer = RMSNorm(hidden_size=8, eps=1e-05)
        if residual is not None:
            out_x, out_residual = layer.forward_oot(dummy_tensor, residual)

            if is_310p:
                expected_arg_x = dummy_tensor + residual.to(dummy_tensor.dtype)
                expected_out_x = expected_arg_x + 1
                expected_out_residual = expected_arg_x.to(residual.dtype)

                mock_rmsnorm.assert_called_once()
                assert torch.allclose(out_x, expected_out_x)
                assert torch.allclose(out_residual, expected_out_residual)
            else:
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
