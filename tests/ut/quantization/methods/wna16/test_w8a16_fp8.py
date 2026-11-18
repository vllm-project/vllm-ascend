from unittest.mock import Mock, patch

import torch
import torch.nn as nn

from tests.ut.base import TestBase
from tests.ut.quantization.conftest_quantization import create_mock_ascend_config, create_mock_vllm_config
from vllm_ascend.quantization.methods.wna16.w8a16_fp8 import AscendW8A16FP8FusedMoEMethod, AscendW8A16FP8LinearMethod


class TestAscendW8A16FP8LinearMethod(TestBase):
    input_size = 128
    output_size = 256

    @patch("vllm_ascend.quantization.methods.wna16.w8a16_fp8.get_current_vllm_config")
    def setUp(self, mock_vllm):
        mock_vllm.return_value = create_mock_vllm_config()
        self.scheme = AscendW8A16FP8LinearMethod()

    def test_get_weight(self):
        result = self.scheme.get_weight(self.input_size, self.output_size, torch.bfloat16)
        self.assertEqual(result["weight"].dtype, torch.float8_e4m3fn)
        self.assertEqual(result["weight"].shape, (self.output_size, self.input_size))
        self.assertEqual(len(result), 1)

    def test_get_perchannel_param_matches_modelslim_contract(self):
        for output_size in (128, 256):
            result = self.scheme.get_perchannel_param(output_size, torch.bfloat16)
            self.assertEqual(result["weight_scale"].shape, (output_size, 1))
            self.assertEqual(result["weight_scale"].dtype, torch.float32)
            self.assertEqual(len(result), 1)

    def test_process_weights_transposes_and_flattens_scales(self):
        layer = nn.Module()
        layer.weight = nn.Parameter(
            torch.empty(self.output_size, self.input_size, dtype=torch.float8_e4m3fn),
            requires_grad=False,
        )
        layer.weight_scale = nn.Parameter(
            torch.ones(self.output_size, 1, dtype=torch.float32),
            requires_grad=False,
        )

        self.scheme.process_weights_after_loading(layer)

        self.assertEqual(layer.weight.shape, (self.input_size, self.output_size))
        self.assertTrue(layer.weight.data.is_contiguous())
        self.assertEqual(layer.weight_scale.shape, (self.output_size,))
        self.assertEqual(layer.weight_scale.dtype, torch.bfloat16)

    @patch("vllm_ascend.quantization.methods.wna16.w8a16_fp8.torch_npu")
    def test_apply_antiquant_matmul_without_offset(self, mock_npu):
        layer = nn.Module()
        layer.weight = nn.Parameter(
            torch.empty(self.input_size, self.output_size, dtype=torch.float8_e4m3fn),
            requires_grad=False,
        )
        layer.weight_scale = nn.Parameter(torch.ones(self.output_size, dtype=torch.bfloat16), requires_grad=False)
        x = torch.randn(4, self.input_size, dtype=torch.bfloat16)
        expected = torch.randn(4, self.output_size, dtype=torch.bfloat16)
        mock_npu.npu_weight_quant_batchmatmul.return_value = expected

        output = self.scheme.apply(layer, x)

        self.assertTrue(torch.equal(output, expected))
        mock_npu.npu_weight_quant_batchmatmul.assert_called_once()
        call_kwargs = mock_npu.npu_weight_quant_batchmatmul.call_args.kwargs
        self.assertNotIn("antiquant_offset", call_kwargs)
        self.assertIsNone(call_kwargs["bias"])


class TestAscendW8A16FP8MoEMethod(TestBase):
    num_experts = 8
    hidden_size = 128
    intermediate_size = 256

    @patch("vllm_ascend.quantization.methods.wna16.w8a16_fp8.get_current_vllm_config")
    @patch("vllm_ascend.quantization.methods.wna16.w8a16_fp8.get_ascend_config")
    def setUp(self, mock_ascend, mock_vllm):
        mock_vllm.return_value = create_mock_vllm_config()
        mock_ascend.return_value = create_mock_ascend_config()
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
        self.assertEqual(result["w13_weight_scale"].shape, (self.num_experts, 2 * self.intermediate_size, 1))
        self.assertEqual(result["w2_weight_scale"].shape, (self.num_experts, self.hidden_size, 1))
        self.assertEqual(result["w13_weight_scale"].dtype, torch.float32)
        self.assertEqual(result["w2_weight_scale"].dtype, torch.float32)

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
            torch.ones(self.num_experts, 2 * self.intermediate_size, 1, dtype=torch.float32),
            requires_grad=False,
        )
        layer.w2_weight_scale = nn.Parameter(
            torch.ones(self.num_experts, self.hidden_size, 1, dtype=torch.float32),
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
        self.assertEqual(layer.w13_weight_scale.dtype, torch.bfloat16)
        self.assertEqual(layer.w2_weight_scale.dtype, torch.bfloat16)
        self.assertTrue(layer.w13_weight_scale.is_contiguous())
        self.assertTrue(layer.w2_weight_scale.is_contiguous())

    @patch("vllm_ascend.quantization.methods.wna16.w8a16_fp8._EXTRA_CTX")
    def test_apply_full_params(self, mock_ctx):
        tokens = 4
        layer = nn.Module()
        layer.w13_weight = nn.Parameter(
            torch.empty(
                self.num_experts,
                self.hidden_size,
                2 * self.intermediate_size,
                dtype=torch.float8_e4m3fn,
            ),
            requires_grad=False,
        )
        layer.w2_weight = nn.Parameter(
            torch.empty(
                self.num_experts,
                self.intermediate_size,
                self.hidden_size,
                dtype=torch.float8_e4m3fn,
            ),
            requires_grad=False,
        )
        layer.w13_weight_scale = nn.Parameter(
            torch.ones(self.num_experts, 2 * self.intermediate_size, dtype=torch.bfloat16),
            requires_grad=False,
        )
        layer.w2_weight_scale = nn.Parameter(
            torch.ones(self.num_experts, self.hidden_size, dtype=torch.bfloat16),
            requires_grad=False,
        )
        x = torch.randn(tokens, self.hidden_size, dtype=torch.bfloat16)
        topk_weights = torch.randn(tokens, 2)
        topk_ids = torch.randint(0, self.num_experts, (tokens, 2))
        layer.activation = "silu"
        layer.ascend_pertoken_scale = None
        layer.apply_router_weight_on_input = False
        layer.ascend_expert_map = None
        layer.global_redundant_expert_num = 0
        layer.ascend_mc2_mask = None
        mock_comm = Mock()
        mock_comm.fused_experts.return_value = torch.randn(tokens, self.hidden_size)
        mock_ctx.moe_comm_method = mock_comm

        self.scheme.apply(
            layer,
            x,
            topk_weights,
            topk_ids,
            shared_experts=None,
            shared_experts_input=None,
        )

        mock_comm.fused_experts.assert_called_once()
        fused_experts_input = mock_comm.fused_experts.call_args.kwargs["fused_experts_input"]
        self.assertEqual(fused_experts_input.quant.quant_type, self.scheme.quant_type)
        self.assertIs(fused_experts_input.layer, layer)
        quant_method = mock_comm.fused_experts.call_args.kwargs["quant_method"]
        self.assertIs(quant_method, self.scheme)
