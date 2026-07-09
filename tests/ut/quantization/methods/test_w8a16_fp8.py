from unittest.mock import MagicMock, patch

import torch
import torch.nn as nn

from tests.ut.base import TestBase
from tests.ut.quantization.conftest_quantization import create_mock_ascend_config, create_mock_vllm_config
from vllm_ascend.quantization.methods.w8a16_fp8 import AscendW8A16FP8FusedMoEMethod


class TestAscendW8A16FP8MoEMethod(TestBase):
    num_experts = 8
    hidden_size = 128
    intermediate_size = 256

    @patch("vllm_ascend.quantization.methods.w8a16_mxfp8.get_ep_group")
    @patch("vllm_ascend.quantization.methods.w8a16_mxfp8.get_current_vllm_config")
    @patch("vllm_ascend.quantization.methods.w8a16_mxfp8.get_ascend_config")
    def setUp(self, mock_ascend, mock_vllm, mock_ep_group):
        mock_vllm.return_value = create_mock_vllm_config()
        mock_ascend.return_value = create_mock_ascend_config()
        mock_ep_group.return_value = MagicMock()
        self.scheme = AscendW8A16FP8FusedMoEMethod()

    def test_get_weight(self):
        result = self.scheme.get_weight(self.num_experts, self.intermediate_size, self.hidden_size, torch.bfloat16)
        self.assertEqual(result["w13_weight"].dtype, torch.float8_e4m3fn)
        self.assertEqual(result["w2_weight"].dtype, torch.float8_e4m3fn)
        self.assertEqual(
            result["w13_weight"].shape,
            (self.num_experts, 2 * self.intermediate_size, self.hidden_size),
        )
        self.assertEqual(
            result["w2_weight"].shape,
            (self.num_experts, self.hidden_size, self.intermediate_size),
        )

    def test_get_dynamic_quant_param(self):
        result = self.scheme.get_dynamic_quant_param(
            self.num_experts, self.intermediate_size, self.hidden_size, torch.bfloat16
        )
        self.assertEqual(result["w13_weight_scale"].shape, (self.num_experts, 2 * self.intermediate_size))
        self.assertEqual(result["w2_weight_scale"].shape, (self.num_experts, self.hidden_size))
        self.assertEqual(result["w13_weight_scale"].dtype, torch.bfloat16)
        self.assertEqual(result["w2_weight_scale"].dtype, torch.bfloat16)

    def test_process_weights_transposes_weights(self):
        layer = nn.Module()
        layer.w13_weight = nn.Parameter(
            torch.empty(
                self.num_experts,
                2 * self.intermediate_size,
                self.hidden_size,
                dtype=torch.float8_e4m3fn,
            ),
            requires_grad=False,
        )
        layer.w2_weight = nn.Parameter(
            torch.empty(
                self.num_experts,
                self.hidden_size,
                self.intermediate_size,
                dtype=torch.float8_e4m3fn,
            ),
            requires_grad=False,
        )
        layer.w13_weight_scale = nn.Parameter(
            torch.empty(self.num_experts, 2 * self.intermediate_size, dtype=torch.bfloat16),
            requires_grad=False,
        )
        layer.w2_weight_scale = nn.Parameter(
            torch.empty(self.num_experts, self.hidden_size, dtype=torch.bfloat16),
            requires_grad=False,
        )

        self.scheme.process_weights_after_loading(layer)

        self.assertEqual(
            layer.w13_weight.shape,
            (self.num_experts, self.hidden_size, 2 * self.intermediate_size),
        )
        self.assertEqual(
            layer.w2_weight.shape,
            (self.num_experts, self.intermediate_size, self.hidden_size),
        )
        self.assertEqual(layer.w13_weight_scale.shape, (self.num_experts, 2 * self.intermediate_size))
        self.assertEqual(layer.w2_weight_scale.shape, (self.num_experts, self.hidden_size))
        self.assertTrue(layer.w13_weight_scale.is_contiguous())
        self.assertTrue(layer.w2_weight_scale.is_contiguous())
