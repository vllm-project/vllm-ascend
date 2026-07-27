from unittest.mock import Mock, patch

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
    @patch("vllm_ascend.quantization.methods.w4a16_mxfp4.get_ep_group")
    @patch("vllm_ascend.quantization.methods.w4a16_mxfp4.get_current_vllm_config")
    @patch("vllm_ascend.quantization.methods.w4a16_mxfp4.get_ascend_config")
    def setUp(self, mock_ascend, mock_vllm, mock_ep, mock_ensure):
        mock_vllm.return_value = create_mock_vllm_config()
        mock_ascend.return_value = create_mock_ascend_config()
        mock_ep.return_value = Mock()
        mock_ensure.return_value = None
        self.scheme = AscendW4A16MXFP4FusedMoEMethod()

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

    @patch("vllm_ascend.quantization.methods.w4a16_mxfp4.torch_npu")
    def test_process_weights_transposes_weights(self, mock_torch_npu):
        mock_torch_npu.npu_format_cast.side_effect = lambda x, *args, **kwargs: x
        mock_torch_npu.npu_convert_weight_to_int4pack.side_effect = lambda x: x

        layer = nn.Module()
        layer.w13_weight = nn.Parameter(torch.randint(0, 255, (8, 256, 64), dtype=torch.uint8), requires_grad=False)
        layer.w2_weight = nn.Parameter(torch.randint(0, 255, (8, 128, 128), dtype=torch.uint8), requires_grad=False)
        layer.w13_weight_scale = nn.Parameter(
            torch.randint(0, 255, (8, 256, 4), dtype=torch.uint8), requires_grad=False
        )
        layer.w2_weight_scale = nn.Parameter(torch.randint(0, 255, (8, 128, 8), dtype=torch.uint8), requires_grad=False)
        self.scheme.process_weights_after_loading(layer)
        self.assertEqual(layer.w13_weight.shape, (8, 128, 256))
        self.assertEqual(layer.w13_weight_scale.shape, (8, 4, 256))
        self.assertEqual(layer.w2_weight.shape, (8, 256, 128))
        self.assertEqual(layer.w2_weight_scale.shape, (8, 8, 128))

    @patch("vllm_ascend.quantization.methods.w4a16_mxfp4.torch_npu")
    @patch("vllm_ascend.quantization.methods.w4a16_mxfp4._EXTRA_CTX")
    @patch("vllm_ascend.quantization.methods.w4a16_mxfp4.select_experts")
    def test_apply_full_params(self, mock_select, mock_ctx, mock_npu):
        tokens = 4
        layer = nn.Module()
        layer.swiglu_limit = 0.0
        layer.w13_weight = nn.Parameter(torch.randint(0, 255, (8, 64, 256), dtype=torch.uint8), requires_grad=False)
        layer.w2_weight = nn.Parameter(torch.randint(0, 255, (8, 128, 128), dtype=torch.uint8), requires_grad=False)
        layer.w13_weight_scale = nn.Parameter(
            torch.randint(0, 255, (8, 64, 128, 2), dtype=torch.uint8), requires_grad=False
        )
        layer.w2_weight_scale = nn.Parameter(
            torch.randint(0, 255, (8, 128, 64, 2), dtype=torch.uint8), requires_grad=False
        )
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
            num_experts=self.num_experts,
            activation="silu",
            pertoken_scale=torch.randn(tokens),
            apply_router_weight_on_input=True,
        )
        mock_comm.fused_experts.assert_called_once()
