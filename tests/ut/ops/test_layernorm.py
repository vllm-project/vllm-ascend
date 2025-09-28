from unittest.mock import patch

import pytest
import torch
from vllm.model_executor.layers.layernorm import RMSNorm


@pytest.fixture
def dummy_tensor():
    return torch.randn(4, 8, dtype=torch.float16)


def mock_maybe_chunk_residual(x, residual):
    if x.size(0) != residual.size(0):
        return residual[:4]

    return residual


def mock_rms_norm(x, weight, eps):
    return x + 1, None


def mock_add_rms_norm(x, residual, weight, eps):
    return 2 * x, None, 2 * residual


@pytest.mark.parametrize("is_310p_return", [True, False])
@pytest.mark.parametrize("residual",
                         [None, torch.randn(4, 8, dtype=torch.float32)])
@patch("torch_npu.npu_rms_norm", side_effect=mock_rms_norm)
@patch("torch_npu.npu_add_rms_norm", side_effect=mock_add_rms_norm)
@patch("torch.ops.vllm.maybe_wait_prefetch_done", side_effect=lambda x: None)
@patch("torch.ops.vllm.maybe_chunk_residual",
       side_effect=mock_maybe_chunk_residual)
def test_RMSNorm_forward(mock_maybe_chunk_residual,
                         mock_maybe_wait_prefetch_done, mock_add_rmsnorm,
                         mock_rmsnorm, is_310p_return, residual, dummy_tensor):

    with patch("vllm_ascend.utils.is_310p", return_value=is_310p_return):
        layer = RMSNorm(hidden_size=8, eps=1e-05)
        if residual is not None:
            out_x, out_residual = layer.forward_oot(dummy_tensor, residual)

            if is_310p_return:
                expected_arg_x = dummy_tensor + residual.to(dummy_tensor.dtype)
                expected_out_x = expected_arg_x + 1
                expected_out_residual = expected_arg_x.to(residual.dtype)

                mock_maybe_chunk_residual.assert_called_once()
                mock_rmsnorm.assert_called_once()
                mock_maybe_wait_prefetch_done.assert_called_once()
                assert torch.allclose(out_x, expected_out_x)
                assert torch.allclose(out_residual, expected_out_residual)
            else:
                expected_out_x = 2 * dummy_tensor
                expected_out_residual = 2 * residual
                mock_maybe_chunk_residual.assert_called_once()
                mock_add_rmsnorm.assert_called_once()
                mock_maybe_wait_prefetch_done.assert_called_once()
                assert torch.allclose(out_x, expected_out_x)
                assert torch.allclose(out_residual, expected_out_residual)
        else:
            out_x = layer.forward(dummy_tensor, residual)
            expected_out_x = dummy_tensor + 1

            mock_rmsnorm.assert_called_once()
            assert torch.allclose(out_x, expected_out_x)


@patch("vllm_ascend.utils.is_310p", return_value=False)
@patch("torch_npu.npu_add_rms_norm", side_effect=mock_add_rms_norm)
@patch("torch.ops.vllm.maybe_wait_prefetch_done", side_effect=lambda x: None)
@patch("torch.ops.vllm.maybe_chunk_residual",
       side_effect=mock_maybe_chunk_residual)
def test_RMSNorm_forward_with_flashcomm_v1(mock_maybe_chunk_residual,
                                           mock_maybe_wait_prefetch_done,
                                           mock_add_rms_norm, mock_is310p):
    x = torch.randn(4, 512, dtype=torch.bfloat16)
    residual = torch.randn(16, 512, dtype=torch.bfloat16)
    layer = RMSNorm(hidden_size=512, eps=1e-05)

    out_x, out_residual = layer.forward_oot(x, residual)

    expected_out_x = 2 * x
    expected_out_residual = 2 * residual[:4]

    mock_maybe_chunk_residual.assert_called_once()
    mock_add_rms_norm.assert_called_once()
    mock_maybe_wait_prefetch_done.assert_called_once()
    assert out_residual.size(0) == 4
    assert torch.allclose(out_x, expected_out_x)
    assert torch.allclose(out_residual, expected_out_residual)
