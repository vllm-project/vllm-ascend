from unittest.mock import Mock, patch

import pytest
import torch
import torch.nn as nn

from tests.ut.base import TestBase
from tests.ut.quantization.conftest_quantization import create_mock_ascend_config, create_mock_vllm_config
from vllm_ascend.quantization.methods.w4a16_mxfp4 import AscendW4A16MXFP4FusedMoEMethod


class TestAscendW4A16MXFP4MoEMethod(TestBase):
    num_experts = 8
    hidden_size = 128
    intermediate_size = 256

    @patch("vllm_ascend.quantization.methods.w4a16_mxfp4.ensure_mxfp4_moe_available")
    @patch("vllm_ascend.quantization.methods.w4a16_mxfp4.get_current_vllm_config")
    @patch("vllm_ascend.quantization.methods.w4a16_mxfp4.get_ascend_config")
    @patch("vllm_ascend.quantization.methods.w4a16_mxfp4.get_ep_group")
    def setUp(self, mock_ep_group, mock_ascend, mock_vllm, mock_ensure):
        mock_vllm.return_value = create_mock_vllm_config()
        mock_ascend.return_value = create_mock_ascend_config()
        mock_ensure.return_value = None
        mock_ep_group.return_value = Mock()
        self.scheme = AscendW4A16MXFP4FusedMoEMethod()

    @pytest.mark.skip("Execute after the issue is fixed")
    def test_get_weight_static_method(self):
        result = self.scheme.get_weight(self.num_experts, self.intermediate_size, self.hidden_size, torch.bfloat16)
        self.assertEqual(result["w13_weight"].dtype, torch.uint8)
        self.assertEqual(result["w2_weight"].dtype, torch.uint8)
        self.assertEqual(
            result["w13_weight"].shape, (self.num_experts, 2 * self.intermediate_size, self.hidden_size // 2)
        )
        self.assertEqual(result["w2_weight"].shape, (self.num_experts, self.hidden_size, self.intermediate_size // 2))

    @pytest.mark.skip("Execute after the issue is fixed")
    def test_get_dynamic_quant_param_based_on_group_size(self):
        group_sizes = [16, 32, 64]
        for gs in group_sizes:
            self.scheme.group_size = gs
            result = self.scheme.get_dynamic_quant_param(
                self.num_experts, self.intermediate_size, self.hidden_size, torch.bfloat16
            )
            self.assertEqual(result["w13_weight_scale"].shape[2], self.hidden_size // gs)
            self.assertEqual(result["w13_weight_scale"].dtype, torch.uint8)
            self.assertEqual(result["w2_weight_scale"].dtype, torch.uint8)

    @pytest.mark.skip("Execute after the issue is fixed")
    def test_process_weights_transposes_weights(self):
        layer = nn.Module()
        layer.w13_weight = nn.Parameter(torch.randint(0, 255, (8, 256, 64), dtype=torch.uint8), requires_grad=False)
        layer.w2_weight = nn.Parameter(torch.randint(0, 255, (8, 128, 128), dtype=torch.uint8), requires_grad=False)
        layer.w13_weight_scale = nn.Parameter(
            torch.randint(0, 255, (8, 256, 4), dtype=torch.uint8), requires_grad=False
        )
        layer.w2_weight_scale = nn.Parameter(torch.randint(0, 255, (8, 128, 8), dtype=torch.uint8), requires_grad=False)
        self.scheme.process_weights_after_loading(layer)
        self.assertEqual(layer.w13_weight.shape, (8, 128, 32))
        self.assertEqual(layer.w13_weight_scale.shape, (8, 4, 256))
        self.assertEqual(layer.w2_weight.shape, (8, 256, 16))
        self.assertEqual(layer.w2_weight_scale.shape, (8, 8, 128))
