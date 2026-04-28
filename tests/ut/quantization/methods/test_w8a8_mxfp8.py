from unittest.mock import Mock, patch

import torch
import torch.nn as nn

from tests.ut.base import TestBase
from tests.ut.quantization.conftest_quantization import (
    create_mock_ascend_config,
    create_mock_vllm_config,
    create_mxfp_moe_layer,
)
from vllm_ascend.quantization.methods.w8a8_mxfp8 import (
    AscendW8A8MXFP8DynamicFusedMoEMethod,
    AscendW8A8MXFP8DynamicLinearMethod,
)


class TestAscendW8A8MXFP8LinearMethod(TestBase):
    @patch("vllm_ascend.quantization.methods.w8a8_mxfp8.ensure_mxfp8_linear_available")
    @patch("vllm_ascend.quantization.methods.w8a8_mxfp8.get_current_vllm_config")
    def setUp(self, mock_vllm, mock_ensure):
        mock_vllm.return_value = create_mock_vllm_config()
        mock_ensure.return_value = None
        self.scheme = AscendW8A8MXFP8DynamicLinearMethod()

    def test_get_weight_various_input_sizes(self):
        sizes = [(128, 64), (512, 256), (1024, 512)]
        for input_size, output_size in sizes:
            result = self.scheme.get_weight(input_size, output_size, torch.bfloat16)
            self.assertEqual(result["weight"].shape, (output_size, input_size))
            self.assertEqual(result["weight"].dtype, torch.float8_e4m3fn)

    def test_get_pergroup_param_group_size_variations(self):
        group_sizes = [16, 32, 64, 128]
        for gs in group_sizes:
            self.scheme.group_size = gs
            result = self.scheme.get_pergroup_param(256, 128, torch.bfloat16)
            self.assertEqual(result["weight_scale"].shape, (128, 256 // gs))
            self.assertEqual(result["weight_scale"].dtype, torch.uint8)

    def test_process_weights_stores_original_shapes(self):
        layer = nn.Module()
        layer.weight = nn.Parameter(torch.randn(128, 256).to(torch.float8_e4m3fn), requires_grad=False)
        layer.weight_scale = nn.Parameter(torch.randint(0, 255, (128, 8), dtype=torch.uint8), requires_grad=False)
        self.scheme.process_weights_after_loading(layer)
        self.assertTrue(hasattr(layer, "_mxfp8_original_shapes"))
        self.assertEqual(layer._mxfp8_original_shapes["weight"], (128, 256))
        self.assertTrue(layer._mxfp8_transformed)
        self.assertEqual(layer.weight_scale.shape, (4, 128, 2))

    def test_restore_after_process_returns_original_shape(self):
        layer = nn.Module()
        layer.weight = nn.Parameter(torch.randn(128, 256).to(torch.float8_e4m3fn), requires_grad=False)
        layer.weight_scale = nn.Parameter(torch.randint(0, 255, (128, 8), dtype=torch.uint8), requires_grad=False)
        original_weight_shape = layer.weight.shape
        original_scale_shape = layer.weight_scale.shape
        self.scheme.process_weights_after_loading(layer)
        self.scheme.restore_weights_for_rl_loading(layer)
        self.assertEqual(layer.weight.shape, original_weight_shape)
        self.assertEqual(layer.weight_scale.shape, original_scale_shape)
        self.assertFalse(layer._mxfp8_transformed)

    @patch("vllm_ascend.quantization.methods.w8a8_mxfp8.torch_npu")
    def test_apply(self, mock_torch_npu):
        from vllm_ascend.device.mxfp_compat import FLOAT8_E8M0FNU_DTYPE

        dynamic_scale = torch.randint(0, 255, (32, 8), dtype=torch.uint8)
        mock_torch_npu.npu_dynamic_mx_quant.return_value = (
            torch.randint(0, 255, (32, 256), dtype=torch.uint8),
            dynamic_scale,
        )
        mock_torch_npu.npu_quant_matmul.return_value = torch.randn(32, 128, dtype=torch.float16)
        layer = nn.Module()
        layer.weight = nn.Parameter(torch.randn(256, 128).to(torch.float8_e4m3fn), requires_grad=False)
        layer.weight_scale = nn.Parameter(torch.randint(0, 255, (4, 128, 2), dtype=torch.uint8), requires_grad=False)
        x = torch.randn(32, 1, 256, dtype=torch.float16)
        bias = torch.randn(128, dtype=torch.float16)
        output = self.scheme.apply(layer, x, bias)
        self.assertEqual(output.shape, (32, 1, 128))
        call_kwargs = mock_torch_npu.npu_quant_matmul.call_args.kwargs
        self.assertEqual(call_kwargs["bias"].dtype, torch.float32)
        self.assertEqual(call_kwargs["group_sizes"], [1, 1, self.scheme.group_size])
        self.assertEqual(call_kwargs["scale_dtype"], FLOAT8_E8M0FNU_DTYPE)
        self.assertEqual(call_kwargs["output_dtype"], torch.float16)


class TestAscendW8A8MXFP8MoEMethod(TestBase):
    num_experts = 8
    hidden_size = 128
    intermediate_size = 256

    @patch("vllm_ascend.quantization.methods.w8a8_mxfp8.ensure_mxfp8_moe_available")
    @patch("vllm_ascend.quantization.methods.w8a8_mxfp8.get_ep_group")
    @patch("vllm_ascend.quantization.methods.w8a8_mxfp8.get_current_vllm_config")
    @patch("vllm_ascend.quantization.methods.w8a8_mxfp8.get_ascend_config")
    def setUp(self, mock_ascend, mock_vllm, mock_ep, mock_ensure):
        mock_vllm.return_value = create_mock_vllm_config()
        mock_ascend.return_value = create_mock_ascend_config()
        mock_ep.return_value = Mock()
        mock_ensure.return_value = None
        self.scheme = AscendW8A8MXFP8DynamicFusedMoEMethod()

    def test_get_weight_various_expert_counts(self):
        for num_experts in [4, 8, 16]:
            result = self.scheme.get_weight(num_experts, self.intermediate_size, self.hidden_size, torch.bfloat16)
            self.assertEqual(result["w13_weight"].shape[0], num_experts)
            self.assertEqual(result["w2_weight"].dtype, torch.float8_e4m3fn)

    def test_get_dynamic_quant_param_dtype_uint8(self):
        result = self.scheme.get_dynamic_quant_param(
            self.num_experts, self.intermediate_size, self.hidden_size, torch.bfloat16
        )
        self.assertEqual(result["w13_weight_scale"].shape, (8, 512, 4))
        self.assertEqual(result["w2_weight_scale"].dtype, torch.uint8)

    def test_process_weights_stores_original_shapes(self):
        layer = create_mxfp_moe_layer(
            num_experts=self.num_experts, hidden_size=self.hidden_size, intermediate_size=self.intermediate_size
        )
        original_shape = layer.w13_weight.shape
        self.scheme.process_weights_after_loading(layer)
        self.assertTrue(hasattr(layer, "_mxfp8_original_shapes"))
        self.assertIn("w13_weight", layer._mxfp8_original_shapes)
        self.assertEqual(layer.w13_weight.shape, (original_shape[0], original_shape[2], original_shape[1]))

    def test_restore_weights_for_rl_loading(self):
        layer = create_mxfp_moe_layer(
            num_experts=self.num_experts, hidden_size=self.hidden_size, intermediate_size=self.intermediate_size
        )
        original_w13_shape = layer.w13_weight.shape
        self.scheme.process_weights_after_loading(layer)
        self.assertNotEqual(layer.w13_weight.shape, original_w13_shape)
        self.scheme.restore_weights_for_rl_loading(layer)
        self.assertEqual(layer.w13_weight.shape, original_w13_shape)

    @patch("vllm_ascend.quantization.methods.w8a8_mxfp8._EXTRA_CTX")
    @patch("vllm_ascend.quantization.methods.w8a8_mxfp8.select_experts")
    def test_apply_full_params(self, mock_select, mock_ctx):
        tokens = 4
        layer = create_mxfp_moe_layer(
            num_experts=self.num_experts, hidden_size=self.hidden_size, intermediate_size=self.intermediate_size
        )
        self.scheme.process_weights_after_loading(layer)
        x = torch.randn(tokens, self.hidden_size, dtype=torch.bfloat16)
        router_logits = torch.randn(tokens, self.num_experts, dtype=torch.float32)
        topk_weights = torch.randn(tokens, 2)
        topk_ids = torch.randint(0, self.num_experts, (tokens, 2))
        mock_select.return_value = (topk_weights, topk_ids)
        mock_comm = Mock()
        mock_comm.fused_experts.return_value = torch.randn(tokens, self.hidden_size)
        mock_ctx.moe_comm_method = mock_comm
        mock_ctx.moe_comm_type = Mock()
        self.scheme.apply(
            layer,
            x,
            router_logits,
            top_k=2,
            renormalize=True,
            global_num_experts=self.num_experts,
            activation="silu",
            pertoken_scale=torch.randn(tokens),
        )
        mock_select.assert_called_once()
        mock_comm.fused_experts.assert_called_once()
