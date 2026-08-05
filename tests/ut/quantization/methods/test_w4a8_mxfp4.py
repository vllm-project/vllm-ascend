from unittest.mock import MagicMock, Mock, patch

import torch
import torch.nn as nn

from tests.ut.base import TestBase
from tests.ut.quantization.conftest_quantization import create_mock_ascend_config, create_mock_vllm_config
from vllm_ascend.quantization.methods.w4a8_mxfp4 import (
    AscendW4A8MXFPDynamicFusedMoEMethod,
    AscendW4A8MXFPDynamicLinearMethod,
)


class TestAscendW4A8MXFP4LinearMethod(TestBase):
    @patch("vllm_ascend.quantization.methods.w4a8_mxfp4.ensure_mxfp4_linear_available")
    @patch("vllm_ascend.quantization.methods.w4a8_mxfp4.get_current_vllm_config")
    def setUp(self, mock_vllm, mock_ensure):
        mock_vllm.return_value = create_mock_vllm_config()
        mock_ensure.return_value = None
        self.scheme = AscendW4A8MXFPDynamicLinearMethod()

    def test_get_weight_various_input_sizes(self):
        for input_size in [64, 128, 256, 512]:
            result = self.scheme.get_weight(input_size, 128, torch.bfloat16)
            self.assertEqual(result["weight"].shape, (128, input_size // 2))
            self.assertEqual(result["weight"].dtype, torch.uint8)

    def test_get_pergroup_param_based_on_group_size(self):
        group_sizes = [16, 32, 64]
        for gs in group_sizes:
            self.scheme.group_size = gs
            result = self.scheme.get_pergroup_param(256, 128, torch.bfloat16)
            self.assertEqual(result["weight_scale"].shape, (128, 256 // gs))
            self.assertEqual(result["weight_scale"].dtype, torch.uint8)

    @patch("vllm_ascend.quantization.methods.w4a8_mxfp4.torch_npu")
    def test_process_weights_after_loading_transposes(self, mock_npu):
        # npu_format_cast returns the input tensor (mocked as identity)
        mock_npu.npu_format_cast.side_effect = lambda x, *a, **kw: x
        layer = nn.Module()
        layer.weight = nn.Parameter(torch.randint(0, 255, (128, 128), dtype=torch.uint8), requires_grad=False)
        layer.weight_scale = nn.Parameter(torch.randint(0, 255, (128, 8), dtype=torch.uint8), requires_grad=False)
        self.scheme.process_weights_after_loading(layer)
        self.assertEqual(layer.weight.shape, (128, 128))
        self.assertEqual(layer.weight_scale.shape, (4, 128, 2))

    @patch("vllm_ascend.quantization.methods.w4a8_mxfp4.torch_npu")
    def test_apply_with_prequantized_input(self, mock_npu):
        mock_npu.npu_quant_matmul.return_value = torch.randn(32, 128)
        layer = MagicMock()
        layer.weight = MagicMock(data=torch.randint(0, 255, (128, 128), dtype=torch.uint8))
        layer.weight_scale = MagicMock(data=torch.randint(0, 255, (4, 128, 2), dtype=torch.uint8))
        quantized_x = torch.randint(0, 255, (32, 128), dtype=torch.uint8)
        dynamic_scale = torch.randint(0, 255, (32, 4), dtype=torch.uint8)
        x = (quantized_x, dynamic_scale)
        with patch.object(self.scheme, "group_size", 32):
            output = self.scheme.apply(layer, x)
        self.assertEqual(output.shape[0], 32)

    @patch("vllm_ascend.quantization.methods.w4a8_mxfp4.torch_npu")
    def test_apply_with_bf16_input(self, mock_npu):
        mock_npu.npu_dynamic_mx_quant.return_value = (
            torch.randint(0, 255, (32, 128), dtype=torch.uint8),
            torch.randint(0, 255, (32, 4), dtype=torch.uint8),
        )
        mock_npu.npu_quant_matmul.return_value = torch.randn(32, 128)
        layer = MagicMock()
        layer.weight = MagicMock(data=torch.randint(0, 255, (128, 128), dtype=torch.uint8))
        layer.weight_scale = MagicMock(data=torch.randint(0, 255, (4, 128, 2), dtype=torch.uint8))
        x = torch.randn(32, 256, dtype=torch.bfloat16)
        with patch.object(self.scheme, "group_size", 32):
            output = self.scheme.apply(layer, x)
        self.assertEqual(output.shape[0], 32)


class TestAscendW4A8MXFP4MoEMethod(TestBase):
    num_experts = 8
    hidden_size = 128
    intermediate_size = 256

    @patch("vllm_ascend.quantization.methods.w4a8_mxfp4.get_ep_group")
    @patch("vllm_ascend.quantization.methods.w4a8_mxfp4.ensure_mxfp4_linear_available")
    @patch("vllm_ascend.quantization.methods.w4a8_mxfp4.get_current_vllm_config")
    @patch("vllm_ascend.quantization.methods.w4a8_mxfp4.get_ascend_config")
    def setUp(self, mock_ascend, mock_vllm, mock_ensure, mock_ep_group):
        mock_vllm.return_value = create_mock_vllm_config()
        mock_ascend.return_value = create_mock_ascend_config()
        mock_ensure.return_value = None
        mock_ep_group.return_value = Mock()
        self.scheme = AscendW4A8MXFPDynamicFusedMoEMethod()

    def test_get_weight_static_method(self):
        result = self.scheme.get_weight(self.num_experts, self.intermediate_size, self.hidden_size, torch.bfloat16)
        self.assertEqual(result["w13_weight"].dtype, torch.uint8)
        self.assertEqual(result["w2_weight"].dtype, torch.uint8)
        self.assertEqual(
            result["w13_weight"].shape, (self.num_experts, 2 * self.intermediate_size, self.hidden_size // 2)
        )
        self.assertEqual(result["w2_weight"].shape, (self.num_experts, self.hidden_size, self.intermediate_size // 2))

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

    @patch("vllm_ascend.quantization.methods.w4a8_mxfp4.torch_npu")
    def test_process_weights_transposes_weights(self, mock_npu):
        # npu_format_cast returns the input tensor (mocked as identity)
        mock_npu.npu_format_cast.side_effect = lambda x, *a, **kw: x
        layer = nn.Module()
        layer.w13_weight = nn.Parameter(torch.randint(0, 255, (8, 256, 64), dtype=torch.uint8), requires_grad=False)
        layer.w2_weight = nn.Parameter(torch.randint(0, 255, (8, 128, 128), dtype=torch.uint8), requires_grad=False)
        layer.w13_weight_scale = nn.Parameter(
            torch.randint(0, 255, (8, 256, 4), dtype=torch.uint8), requires_grad=False
        )
        layer.w2_weight_scale = nn.Parameter(torch.randint(0, 255, (8, 128, 8), dtype=torch.uint8), requires_grad=False)
        self.scheme.process_weights_after_loading(layer)
        self.assertEqual(layer.w13_weight.shape, (8, 64, 256))
        self.assertEqual(layer.w2_weight.shape, (8, 128, 128))
        self.assertEqual(layer.w13_weight_scale.shape, (8, 2, 256, 2))
        self.assertEqual(layer.w2_weight_scale.shape, (8, 4, 128, 2))

    @patch("vllm_ascend.quantization.methods.w4a8_mxfp4.torch_npu")
    @patch("vllm_ascend.quantization.methods.w4a8_mxfp4._EXTRA_CTX")
    def test_apply_full_params(self, mock_ctx, mock_npu):
        tokens = 4
        layer = nn.Module()
        layer.w13_weight = nn.Parameter(torch.randint(0, 255, (8, 64, 256), dtype=torch.uint8), requires_grad=False)
        layer.w2_weight = nn.Parameter(torch.randint(0, 255, (8, 128, 128), dtype=torch.uint8), requires_grad=False)
        layer.w13_weight_scale = nn.Parameter(
            torch.randint(0, 255, (8, 2, 256, 2), dtype=torch.uint8), requires_grad=False
        )
        layer.w2_weight_scale = nn.Parameter(
            torch.randint(0, 255, (8, 4, 128, 2), dtype=torch.uint8), requires_grad=False
        )
        layer.swiglu_limit = 0.0
        layer.activation = "silu"
        layer._ascend_pertoken_scale = torch.randn(tokens)
        layer.apply_router_weight_on_input = True
        layer.ascend_expert_map = None
        layer.global_redundant_expert_num = 0
        layer.log2phy = None
        layer._ascend_mc2_mask = None
        x = torch.randn(tokens, self.hidden_size, dtype=torch.bfloat16)
        topk_weights = torch.randn(tokens, 2)
        topk_ids = torch.randint(0, self.num_experts, (tokens, 2))
        mock_comm = Mock()
        mock_comm.fused_experts.return_value = torch.randn(tokens, self.hidden_size)
        mock_ctx.moe_comm_method = mock_comm
        mock_ctx.moe_comm_type = Mock()
        self.scheme.apply(
            layer,
            x,
            topk_weights,
            topk_ids,
            shared_experts=None,
            shared_experts_input=None,
        )
        mock_comm.fused_experts.assert_called_once()
