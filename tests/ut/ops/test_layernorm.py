import unittest

import pytest
import torch
from pytest_mock import MockerFixture
from vllm.model_executor.layers.layernorm import RMSNorm

from tests.ut.base import PytestBase
from vllm_ascend.quantization.w8a8 import AscendW8A8LinearMethod
from vllm_ascend.utils import version_check


def mock_rms_norm(x, weight, eps):
    return x + 1, None


def mock_add_rms_norm(x, residual, weight, eps):
    return 2 * x, None, 2 * residual


def mock_add_rms_norm_quant(x, residual, weight, quant_scale, quant_offset,
                            epsilon):
    x_out = 2 * x
    residual_out = 2 * residual
    x_out_quant = x_out.to(torch.int8)
    residual_out_quant = residual_out.to(torch.int8)
    return x_out_quant, None, residual_out_quant


def mock_add_rms_norm_quant_with_bias(x, residual, weight, quant_scale,
                                      quant_offset, beta, epsilon):
    x_out = 2 * x
    residual_out = 2 * residual
    x_out_quant = x_out.to(torch.int8)
    residual_out_quant = residual_out.to(torch.int8)
    return x_out_quant, None, residual_out_quant


class TestAscendRMSNorm(PytestBase):

    @pytest.fixture(autouse=True)
    def context(self, mocker: MockerFixture):
        mocker.patch("torch_npu.npu_rms_norm", side_effect=mock_rms_norm)
        mocker.patch("torch_npu.npu_add_rms_norm",
                     side_effect=mock_add_rms_norm)
        torch_npu_check = version_check()
        arnq_side_effect = mock_add_rms_norm_quant_with_bias if torch_npu_check else mock_add_rms_norm_quant
        mocker.patch("torch_npu.npu_add_rms_norm_quant",
                     side_effect=arnq_side_effect)
        mocker.patch("torch.ops.vllm.maybe_wait_prefetch_done",
                     side_effect=lambda x: None)

    # Test case for the most common and basic scenario
    @pytest.mark.parametrize(
        "residual", [None, torch.randn(4, 8, dtype=torch.float16)])
    def test_forward_oot_basic(self, residual):
        layer = RMSNorm(hidden_size=8, eps=1e-05)
        x = torch.randn(4, 8, dtype=torch.float16)
        if residual is not None:
            x_out, residual_out = layer.forward_oot(x, residual)

            x_out_expected = 2 * x
            residual_out_expected = 2 * residual

            assert torch.allclose(x_out, x_out_expected)
            assert torch.allclose(residual_out, residual_out_expected)
        else:
            x_out = layer.forward(x, residual)
            x_out_expected = x + 1

            assert torch.allclose(x_out, x_out_expected)

    # Test case for addrmsnorm + w8a8 quant fusion
    def test_forward_oot_with_quant_fusion(self, mocker: MockerFixture):
        mock_is_310p = mocker.patch("vllm_ascend.utils.is_310p")
        mock_is_310p.return_value = False
        mock_get_forward_context = mocker.patch(
            "vllm_ascend.ops.layernorm.get_forward_context")

        # Simulating a scenario with quant_fusion enabled
        mock_forward_context = mocker.MagicMock()

        mock_model_instance = mocker.MagicMock()
        mock_forward_context.model_instance = mock_model_instance
        torch_npu_check = version_check()
        num_hidden_layers = 3 if torch_npu_check else 2
        mock_model_instance.model.layers = [
            mocker.MagicMock() for _ in range(num_hidden_layers)
        ]

        mock_layer_0 = mock_model_instance.model.layers[0]
        mock_layer_0.self_attn.qkv_proj = mocker.MagicMock()
        mock_layer_0.mlp.gate_up_proj = mocker.MagicMock()

        mock_layer_1 = mock_model_instance.model.layers[1]
        mock_layer_1.self_attn.qkv_proj = mocker.MagicMock()
        mock_layer_1.mlp.gate_up_proj = mocker.MagicMock()

        mock_quant_method_0_qkv = mocker.MagicMock()
        mock_quant_method_0_qkv.quant_method = AscendW8A8LinearMethod()
        mock_quant_method_0_gate_up = mocker.MagicMock()
        mock_quant_method_0_gate_up.quant_method = AscendW8A8LinearMethod()
        mock_layer_0.self_attn.qkv_proj.quant_method = mock_quant_method_0_qkv
        mock_layer_0.mlp.gate_up_proj.quant_method = mock_quant_method_0_gate_up

        mock_quant_method_1_qkv = mocker.MagicMock()
        mock_quant_method_1_qkv.quant_method = AscendW8A8LinearMethod()
        mock_quant_method_1_gate_up = mocker.MagicMock()
        mock_quant_method_1_gate_up.quant_method = AscendW8A8LinearMethod()
        mock_layer_1.self_attn.qkv_proj.quant_method = mock_quant_method_1_qkv
        mock_layer_1.mlp.gate_up_proj.quant_method = mock_quant_method_1_gate_up

        mock_get_forward_context.return_value = mock_forward_context

        mock_forward_context.addrmsnorm_quant_fusion_enabled = True
        mock_forward_context.prefetch_mlp_enabled = False
        mock_forward_context.layer_idx = 0
        mock_forward_context.num_hidden_layers = num_hidden_layers
        mock_forward_context.fusion_linear = "gate_up_dense"
        mock_forward_context.weight_prefetch_method = None

        # Ensure fusion and layer_idx increment are handled correctly
        x = torch.randn(4, 8, dtype=torch.float16)
        residual = torch.randn(4, 8, dtype=torch.float16)
        layer = RMSNorm(hidden_size=8, eps=1e-05)

        x_out, residual_out = layer.forward_oot(x, residual)

        assert mock_get_forward_context.call_count == 2
        assert mock_forward_context.fusion_linear == "qkv_dense"
        assert mock_forward_context.layer_idx == 1

        x_out, residual_out = layer.forward_oot(x, residual)

        assert mock_get_forward_context.call_count == 4
        assert mock_forward_context.fusion_linear == "gate_up_dense"
        assert mock_forward_context.layer_idx == 1

        if torch_npu_check:
            mock_forward_context.fusion_linear = "gate_moe"
        x_out, residual_out = layer.forward_oot(x, residual)

        assert mock_get_forward_context.call_count == 6
        fusion_linear_expected = "qkv_moe" if torch_npu_check else "qkv_dense"
        assert mock_forward_context.fusion_linear == fusion_linear_expected
        assert mock_forward_context.layer_idx == 2

        x_out, residual_out = layer.forward_oot(x, residual)

        assert mock_get_forward_context.call_count == 7
        fusion_linear_expected = "gate_moe" if torch_npu_check else "qkv_dense"
        assert mock_forward_context.fusion_linear == fusion_linear_expected
        assert mock_forward_context.layer_idx == 2

        if not torch_npu_check:
            return
        # last layer returned directly
        x_out, residual_out = layer.forward_oot(x, residual)

        assert mock_get_forward_context.call_count == 8
        assert mock_forward_context.fusion_linear == "qkv_moe"
        assert mock_forward_context.layer_idx == 3

        x_out, residual_out = layer.forward_oot(x, residual)

        assert mock_get_forward_context.call_count == 9
        assert mock_forward_context.fusion_linear == "qkv_moe"
        assert mock_forward_context.layer_idx == 3


if __name__ == '__main__':
    unittest.main()
