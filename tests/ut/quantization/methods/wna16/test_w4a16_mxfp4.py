from unittest.mock import MagicMock, Mock, patch

import torch
import torch.nn as nn
from vllm.model_executor.layers.linear import RowParallelLinear

from tests.ut.base import TestBase
from tests.ut.quantization.conftest_quantization import create_mock_ascend_config, create_mock_vllm_config
from vllm_ascend.quantization.methods.wna16.w4a16_mxfp4 import (
    AscendW4A16MXFP4FusedMoEMethod,
    AscendW4A16MXFP4LinearMethod,
    unpack_uint8_to_fp4_return_float32,
)


class TestAscendW4A16MXFP4LinearMethod(TestBase):
    input_size = 128
    output_size = 256

    @patch("vllm_ascend.quantization.methods.wna16.w4a16_mxfp4.get_current_vllm_config")
    def setUp(self, mock_vllm):
        mock_vllm.return_value = create_mock_vllm_config()
        self.scheme = AscendW4A16MXFP4LinearMethod()

    def test_get_weight(self):
        result = self.scheme.get_weight(self.input_size, self.output_size, torch.bfloat16)
        self.assertEqual(result["weight"].dtype, torch.uint8)
        self.assertEqual(result["weight"].shape, (self.output_size, self.input_size // 2))
        self.assertEqual(len(result), 1)

    def test_get_pergroup_param_based_on_group_size(self):
        for gs in (16, 32, 64):
            self.scheme.group_size = gs
            result = self.scheme.get_pergroup_param(self.input_size, self.output_size, torch.bfloat16)
            self.assertEqual(result["weight_scale"].shape, (self.output_size, self.input_size // gs))
            self.assertEqual(result["weight_scale"].dtype, torch.uint8)

    @patch("vllm_ascend.quantization.methods.wna16.w4a16_mxfp4.torch_npu")
    def test_process_weights_unpacks_and_transposes(self, mock_npu):
        mock_npu.npu_format_cast.side_effect = lambda x, *args, **kwargs: x
        mock_npu.npu_convert_weight_to_int4pack.side_effect = lambda x: x

        layer = nn.Module()
        layer.weight = nn.Parameter(torch.randint(0, 255, (256, 64), dtype=torch.uint8), requires_grad=False)
        layer.weight_scale = nn.Parameter(torch.randint(0, 255, (256, 4), dtype=torch.uint8), requires_grad=False)

        self.scheme.process_weights_after_loading(layer)

        self.assertEqual(layer.weight.shape, (128, 256))
        self.assertEqual(layer.weight_scale.shape, (4, 256))
        self.assertEqual(layer.mxfp4_tp_padding, (0, 0))
        self.assertTrue(layer.weight.data.is_contiguous())
        self.assertTrue(layer.weight_scale.data.is_contiguous())

    @patch("vllm_ascend.quantization.methods.wna16.w4a16_mxfp4.torch_npu")
    def test_process_weights_preserves_unaligned_tp_group_phase(self, mock_npu):
        mock_npu.npu_format_cast.side_effect = lambda x, *args, **kwargs: x
        mock_npu.npu_convert_weight_to_int4pack.side_effect = lambda x: x
        layer = RowParallelLinear.__new__(RowParallelLinear)
        nn.Module.__init__(layer)
        original_weight = torch.randint(1, 255, (2, 264), dtype=torch.uint8)
        layer.weight = nn.Parameter(original_weight.clone(), requires_grad=False)
        layer.weight_scale = nn.Parameter(torch.randint(1, 255, (2, 17), dtype=torch.uint8), requires_grad=False)
        layer.tp_rank = 3
        layer.input_size_per_partition = 528

        self.scheme.process_weights_after_loading(layer)

        self.assertEqual(layer.mxfp4_tp_padding, (16, 0))
        # packed K: 264 + 16/2 = 272 bytes -> unpacked logical K = 544
        self.assertEqual(layer.weight.shape, (544, 2))
        self.assertEqual(layer.weight_scale.shape, (17, 2))
        torch.testing.assert_close(layer.weight.data[:16], torch.zeros(16, 2))
        torch.testing.assert_close(
            layer.weight.data[16:],
            unpack_uint8_to_fp4_return_float32(original_weight).transpose(0, 1),
        )

    @patch("vllm_ascend.quantization.methods.wna16.w4a16_mxfp4.torch_npu")
    def test_apply_pads_activation_with_weight_group_phase(self, mock_npu):
        mock_npu.npu_weight_quant_batchmatmul.return_value = torch.empty(2, 4)
        layer = MagicMock()
        layer.mxfp4_tp_padding = (16, 0)
        x = torch.randn(2, 528)

        self.scheme.apply(layer, x)

        padded_x = mock_npu.npu_weight_quant_batchmatmul.call_args.kwargs["x"]
        self.assertEqual(padded_x.shape, (2, 544))
        torch.testing.assert_close(padded_x[:, :16], torch.zeros(2, 16))
        torch.testing.assert_close(padded_x[:, 16:], x)


class TestAscendW4A16MXFP4MoEMethod(TestBase):
    num_experts = 8
    hidden_size = 128
    intermediate_size = 256

    @patch("vllm_ascend.quantization.methods.wna16.w4a16_mxfp4.get_current_vllm_config")
    @patch("vllm_ascend.quantization.methods.wna16.w4a16_mxfp4.get_ascend_config")
    def setUp(self, mock_ascend, mock_vllm):
        mock_vllm.return_value = create_mock_vllm_config()
        mock_ascend.return_value = create_mock_ascend_config()
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

    @patch("vllm_ascend.quantization.methods.wna16.w4a16_mxfp4.torch_npu")
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

    @patch("vllm_ascend.quantization.methods.wna16.w4a16_mxfp4.torch_npu")
    @patch("vllm_ascend.quantization.methods.wna16.w4a16_mxfp4._EXTRA_CTX")
    def test_apply_full_params(self, mock_ctx, mock_npu):
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
        topk_weights = torch.randn(tokens, 2)
        topk_ids = torch.randint(0, self.num_experts, (tokens, 2))
        layer.activation = "silu"
        layer.ascend_pertoken_scale = torch.randn(tokens)
        layer.apply_router_weight_on_input = True
        layer.ascend_expert_map = None
        layer.global_redundant_expert_num = 0
        layer.log2phy = None
        layer.ascend_mc2_mask = None
        layer.swiglu_alpha = 1.0
        layer.swiglu_beta = 0.0
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
