from unittest.mock import MagicMock, Mock, patch
import torch
import torch.nn as nn
from tests.ut.base import TestBase
from tests.ut.quantization.conftest_quantization import create_mock_vllm_config, create_mock_ascend_config
from vllm_ascend.device.mxfp_compat import FLOAT8_E8M0FNU_DTYPE
from vllm_ascend.quantization.methods.w4a4_mxfp4 import (
    AscendW4A4MXFP4DynamicLinearMethod,
    AscendW4A4MXFP4DynamicFusedMoEMethod,
)
from vllm_ascend.quantization.methods.base import QuantType


class TestAscendW4A4MXFP4LinearMethod(TestBase):

    @patch("vllm_ascend.quantization.methods.w4a4_mxfp4.ensure_mxfp4_linear_available")
    @patch("vllm_ascend.quantization.methods.w4a4_mxfp4.get_current_vllm_config")
    def setUp(self, mock_vllm, mock_ensure):
        mock_vllm.return_value = create_mock_vllm_config()
        mock_ensure.return_value = None
        self.scheme = AscendW4A4MXFP4DynamicLinearMethod()

    def test_model_dtype_is_none(self):
        self.assertIsNone(self.scheme.model_dtype)

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

    def test_get_pertensor_param_returns_empty(self):
        result = self.scheme.get_pertensor_param(torch.bfloat16)
        self.assertEqual(result, {})

    def test_get_perchannel_param_returns_empty(self):
        result = self.scheme.get_perchannel_param(128, torch.bfloat16)
        self.assertEqual(result, {})

    def test_process_weights_after_loading_transposes(self):
        layer = nn.Module()
        layer.weight = nn.Parameter(torch.randint(0, 255, (128, 128), dtype=torch.uint8), requires_grad=False)
        layer.weight_scale = nn.Parameter(torch.randint(0, 255, (128, 8), dtype=torch.uint8), requires_grad=False)
        self.scheme.process_weights_after_loading(layer)
        self.assertEqual(layer.weight.shape, (128, 128))
        self.assertEqual(layer.weight_scale.shape[0], 4)

    @patch("vllm_ascend.quantization.methods.w4a4_mxfp4.ensure_mxfp4_linear_available")
    @patch("vllm_ascend.quantization.methods.w4a4_mxfp4.get_current_vllm_config")
    def test_group_size_from_config(self, mock_vllm_config, mock_ensure):
        mock_vllm_config.return_value = create_mock_vllm_config(quant_description={"group_size": 64})
        scheme = AscendW4A4MXFP4DynamicLinearMethod()
        self.assertEqual(scheme.group_size, 64)

    @patch("vllm_ascend.quantization.methods.w4a4_mxfp4.torch_npu")
    def test_apply_3d_input(self, mock_npu):
        mock_npu.npu_dynamic_mx_quant.return_value = (
            torch.randint(0, 255, (32, 128), dtype=torch.uint8),
            torch.randint(0, 255, (32, 4), dtype=torch.uint8)
        )
        mock_npu.npu_quant_matmul.return_value = torch.randn(32, 1, 128)
        layer = MagicMock()
        layer.weight = MagicMock(data=torch.randint(0, 255, (128, 128), dtype=torch.uint8))
        layer.weight_scale = MagicMock(data=torch.randint(0, 255, (4, 128, 2), dtype=torch.uint8))
        x = torch.randn(32, 1, 256, dtype=torch.bfloat16)
        with patch.object(self.scheme, 'group_size', 32):
            output = self.scheme.apply(layer, x)
        self.assertEqual(output.shape[0], 32)


class TestAscendW4A4MXFP4MoEMethod(TestBase):
    num_experts = 8
    hidden_size = 128
    intermediate_size = 256

    @patch("vllm_ascend.quantization.methods.w4a4_mxfp4.ensure_mxfp4_moe_available")
    @patch("vllm_ascend.quantization.methods.w4a4_mxfp4.get_ep_group")
    @patch("vllm_ascend.quantization.methods.w4a4_mxfp4.get_current_vllm_config")
    @patch("vllm_ascend.quantization.methods.w4a4_mxfp4.get_ascend_config")
    def setUp(self, mock_ascend, mock_vllm, mock_ep, mock_ensure):
        mock_vllm.return_value = create_mock_vllm_config()
        mock_ascend.return_value = create_mock_ascend_config()
        mock_ep.return_value = Mock()
        mock_ensure.return_value = None
        self.scheme = AscendW4A4MXFP4DynamicFusedMoEMethod()

    def test_init(self):
        self.assertEqual(self.scheme.quant_type, QuantType.MXFP4)
        self.assertIsNone(self.scheme.model_dtype)

    def test_get_weight_static_method(self):
        result = self.scheme.get_weight(self.num_experts, self.intermediate_size, self.hidden_size, torch.bfloat16)
        self.assertEqual(result["w13_weight"].dtype, torch.uint8)
        self.assertEqual(result["w2_weight"].dtype, torch.uint8)
        self.assertEqual(result["w13_weight"].shape,
                         (self.num_experts, 2 * self.intermediate_size, self.hidden_size // 2))
        self.assertEqual(result["w2_weight"].shape, (self.num_experts, self.hidden_size, self.intermediate_size // 2))

    def test_get_dynamic_quant_param_based_on_group_size(self):
        group_sizes = [16, 32, 64]
        for gs in group_sizes:
            self.scheme.group_size = gs
            result = self.scheme.get_dynamic_quant_param(self.num_experts, self.intermediate_size, self.hidden_size,
                                                         torch.bfloat16)
            self.assertEqual(result["w13_weight_scale"].shape[2], self.hidden_size // gs)
            self.assertEqual(result["w13_weight_scale"].dtype, torch.uint8)
            self.assertEqual(result["w2_weight_scale"].dtype, torch.uint8)

    def test_process_weights_transposes_weights(self):
        layer = nn.Module()
        layer.w13_weight = nn.Parameter(torch.randint(0, 255, (8, 256, 64), dtype=torch.uint8), requires_grad=False)
        layer.w2_weight = nn.Parameter(torch.randint(0, 255, (8, 128, 128), dtype=torch.uint8), requires_grad=False)
        layer.w13_weight_scale = nn.Parameter(torch.randint(0, 255, (8, 256, 4), dtype=torch.uint8),
                                              requires_grad=False)
        layer.w2_weight_scale = nn.Parameter(torch.randint(0, 255, (8, 128, 8), dtype=torch.uint8), requires_grad=False)
        self.scheme.process_weights_after_loading(layer)
        self.assertEqual(layer.w13_weight.shape, (8, 64, 256))
        self.assertEqual(layer.w2_weight.shape, (8, 128, 128))
        self.assertEqual(layer.w13_weight_scale.shape, (8, 2, 256, 2))
        self.assertEqual(layer.w2_weight_scale.shape, (8, 4, 128, 2))

    @patch("vllm_ascend.quantization.methods.w4a4_mxfp4.torch_npu")
    @patch("vllm_ascend.quantization.methods.w4a4_mxfp4._EXTRA_CTX")
    @patch("vllm_ascend.quantization.methods.w4a4_mxfp4.select_experts")
    def test_apply_full_params(self, mock_select, mock_ctx, mock_npu):
        tokens = 4
        layer = nn.Module()
        layer.w13_weight = nn.Parameter(torch.randint(0, 255, (8, 64, 256), dtype=torch.uint8), requires_grad=False)
        layer.w2_weight = nn.Parameter(torch.randint(0, 255, (8, 128, 128), dtype=torch.uint8), requires_grad=False)
        layer.w13_weight_scale = nn.Parameter(torch.randint(0, 255, (8, 64, 128, 2), dtype=torch.uint8),
                                              requires_grad=False)
        layer.w2_weight_scale = nn.Parameter(torch.randint(0, 255, (8, 128, 64, 2), dtype=torch.uint8),
                                             requires_grad=False)
        x = torch.randn(tokens, self.hidden_size, dtype=torch.bfloat16)
        router_logits = torch.randn(tokens, self.num_experts, dtype=torch.float32)
        topk_weights = torch.randn(tokens, 2)
        topk_ids = torch.randint(0, self.num_experts, (tokens, 2))
        mock_select.return_value = (topk_weights, topk_ids)
        mock_comm = Mock()
        mock_comm.fused_experts.return_value = torch.randn(tokens, self.hidden_size)
        mock_ctx.moe_comm_method = mock_comm
        mock_ctx.moe_comm_type = Mock()
        result = self.scheme.apply(
            layer, x, router_logits, top_k=2, renormalize=True,
            global_num_experts=self.num_experts,
            activation="silu", pertoken_scale=torch.randn(tokens),
            apply_router_weight_on_input=True
        )
        mock_comm.fused_experts.assert_called_once()
