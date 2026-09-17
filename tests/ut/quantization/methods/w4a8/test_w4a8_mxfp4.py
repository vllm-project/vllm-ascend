from unittest.mock import MagicMock, Mock, patch

import torch
import torch.nn as nn

from tests.ut.base import TestBase
from tests.ut.quantization.conftest_quantization import create_mock_ascend_config, create_mock_vllm_config
from vllm_ascend.quantization.methods.w4a8.w4a8_mxfp4 import (
    AscendW4A8MXFPDynamicFusedMoEMethod,
    AscendW4A8MXFPDynamicLinearMethod,
)


class TestAscendW4A8MXFP4LinearMethod(TestBase):
    @patch("vllm_ascend.quantization.methods.w4a8.w4a8_mxfp4.get_current_vllm_config")
    def setUp(self, mock_vllm):
        mock_vllm.return_value = create_mock_vllm_config()
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

    @patch("vllm_ascend.quantization.methods.w4a8.w4a8_mxfp4.torch_npu")
    def test_process_weights_after_loading_transposes(self, mock_npu):
        # npu_format_cast returns the input tensor (mocked as identity)
        mock_npu.npu_format_cast.side_effect = lambda x, *a, **kw: x
        layer = nn.Module()
        layer.weight = nn.Parameter(torch.randint(0, 255, (128, 128), dtype=torch.uint8), requires_grad=False)
        layer.weight_scale = nn.Parameter(torch.randint(0, 255, (128, 8), dtype=torch.uint8), requires_grad=False)
        self.scheme.process_weights_after_loading(layer)
        self.assertEqual(layer.weight.shape, (128, 128))
        self.assertEqual(layer.weight_scale.shape, (4, 128, 2))

    @staticmethod
    def _make_linear_layer():
        layer = nn.Module()
        layer.weight = nn.Parameter(torch.randint(0, 255, (128, 128), dtype=torch.uint8), requires_grad=False)
        layer.weight_scale = nn.Parameter(torch.randint(0, 255, (128, 8), dtype=torch.uint8), requires_grad=False)
        return layer

    @patch("vllm_ascend.quantization.methods.w4a8.w4a8_mxfp4.torch_npu")
    def test_process_records_original_shapes_and_marks_transformed(self, mock_npu):
        mock_npu.npu_format_cast.side_effect = lambda x, *a, **kw: x
        layer = self._make_linear_layer()
        self.scheme.process_weights_after_loading(layer)
        self.assertTrue(hasattr(layer, "_mxfp4_original_shapes"))
        self.assertEqual(layer._mxfp4_original_shapes["weight"], (128, 128))
        self.assertEqual(layer._mxfp4_original_shapes["weight_scale"], (128, 8))
        self.assertTrue(layer._mxfp4_transformed)

    @patch("vllm_ascend.quantization.methods.w4a8.w4a8_mxfp4.torch_npu")
    def test_process_is_idempotent(self, mock_npu):
        # The transform is not idempotent (a second transpose/reshape corrupts
        # it), so a repeated call must be a no-op. veRL calls process again
        # after load_weights(), which is exactly this path.
        mock_npu.npu_format_cast.side_effect = lambda x, *a, **kw: x
        layer = self._make_linear_layer()
        self.scheme.process_weights_after_loading(layer)
        weight_shape = layer.weight.shape
        scale_shape = layer.weight_scale.shape
        recorded = dict(layer._mxfp4_original_shapes)
        self.scheme.process_weights_after_loading(layer)
        self.assertEqual(layer.weight.shape, weight_shape)
        self.assertEqual(layer.weight_scale.shape, scale_shape)
        self.assertEqual(layer._mxfp4_original_shapes, recorded)

    @patch("vllm_ascend.quantization.methods.w4a8.w4a8_mxfp4.torch_npu")
    def test_restore_after_process_returns_original_shape(self, mock_npu):
        mock_npu.npu_format_cast.side_effect = lambda x, *a, **kw: x
        layer = self._make_linear_layer()
        original_weight_shape = layer.weight.shape
        original_scale_shape = layer.weight_scale.shape
        self.scheme.process_weights_after_loading(layer)
        self.scheme.restore_weights_for_rl_loading(layer)
        self.assertEqual(layer.weight.shape, original_weight_shape)
        self.assertEqual(layer.weight_scale.shape, original_scale_shape)
        self.assertEqual(layer.weight.dtype, torch.uint8)
        self.assertFalse(layer._mxfp4_transformed)

    def test_restore_without_process_is_noop(self):
        layer = self._make_linear_layer()
        original_weight_shape = layer.weight.shape
        original_scale_shape = layer.weight_scale.shape
        self.scheme.restore_weights_for_rl_loading(layer)
        self.assertEqual(layer.weight.shape, original_weight_shape)
        self.assertEqual(layer.weight_scale.shape, original_scale_shape)
        self.assertFalse(hasattr(layer, "_mxfp4_transformed"))

    @patch("vllm_ascend.quantization.methods.w4a8.w4a8_mxfp4.torch_npu")
    def test_refit_cycle_restore_then_process_round_trips(self, mock_npu):
        # restore -> load_weights -> process must land back in the inference
        # layout, and the recorded shapes must survive the cycle.
        mock_npu.npu_format_cast.side_effect = lambda x, *a, **kw: x
        layer = self._make_linear_layer()
        self.scheme.process_weights_after_loading(layer)
        inference_weight_shape = layer.weight.shape
        inference_scale_shape = layer.weight_scale.shape
        self.scheme.restore_weights_for_rl_loading(layer)
        self.scheme.process_weights_after_loading(layer)
        self.assertEqual(layer.weight.shape, inference_weight_shape)
        self.assertEqual(layer.weight_scale.shape, inference_scale_shape)
        self.assertTrue(layer._mxfp4_transformed)

    @patch("vllm_ascend.quantization.methods.w4a8.w4a8_mxfp4.torch_npu")
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

    @patch("vllm_ascend.quantization.methods.w4a8.w4a8_mxfp4.torch_npu")
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

    @patch("vllm_ascend.quantization.methods.w4a8.w4a8_mxfp4.get_current_vllm_config")
    @patch("vllm_ascend.quantization.methods.w4a8.w4a8_mxfp4.get_ascend_config")
    def setUp(self, mock_ascend, mock_vllm):
        mock_vllm.return_value = create_mock_vllm_config()
        mock_ascend.return_value = create_mock_ascend_config()
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

    @patch("vllm_ascend.quantization.methods.w4a8.w4a8_mxfp4.get_current_vllm_config")
    @patch("vllm_ascend.quantization.methods.w4a8.w4a8_mxfp4.use_cann_megamoe", return_value=False)
    @patch("vllm_ascend.quantization.methods.w4a8.w4a8_mxfp4.torch_npu")
    def test_process_weights_transposes_weights(self, mock_npu, mock_use_cann_megamoe, mock_vllm):
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

    @staticmethod
    def _make_moe_layer():
        layer = nn.Module()
        layer.w13_weight = nn.Parameter(torch.randint(0, 255, (8, 256, 64), dtype=torch.uint8), requires_grad=False)
        layer.w2_weight = nn.Parameter(torch.randint(0, 255, (8, 128, 128), dtype=torch.uint8), requires_grad=False)
        layer.w13_weight_scale = nn.Parameter(
            torch.randint(0, 255, (8, 256, 4), dtype=torch.uint8), requires_grad=False
        )
        layer.w2_weight_scale = nn.Parameter(torch.randint(0, 255, (8, 128, 8), dtype=torch.uint8), requires_grad=False)
        return layer

    @patch("vllm_ascend.quantization.methods.w4a8.w4a8_mxfp4.get_current_vllm_config")
    @patch("vllm_ascend.quantization.methods.w4a8.w4a8_mxfp4.use_cann_megamoe", return_value=False)
    @patch("vllm_ascend.quantization.methods.w4a8.w4a8_mxfp4.torch_npu")
    def test_process_records_original_shapes_and_marks_transformed(self, mock_npu, mock_use_cann_megamoe, mock_vllm):
        mock_npu.npu_format_cast.side_effect = lambda x, *a, **kw: x
        layer = self._make_moe_layer()
        self.scheme.process_weights_after_loading(layer)
        self.assertTrue(hasattr(layer, "_mxfp4_original_shapes"))
        self.assertEqual(layer._mxfp4_original_shapes["w13_weight"], (8, 256, 64))
        self.assertEqual(layer._mxfp4_original_shapes["w13_weight_scale"], (8, 256, 4))
        self.assertEqual(layer._mxfp4_original_shapes["w2_weight"], (8, 128, 128))
        self.assertEqual(layer._mxfp4_original_shapes["w2_weight_scale"], (8, 128, 8))
        self.assertTrue(layer._mxfp4_transformed)

    @patch("vllm_ascend.quantization.methods.w4a8.w4a8_mxfp4.get_current_vllm_config")
    @patch("vllm_ascend.quantization.methods.w4a8.w4a8_mxfp4.use_cann_megamoe", return_value=False)
    @patch("vllm_ascend.quantization.methods.w4a8.w4a8_mxfp4.torch_npu")
    def test_process_is_idempotent(self, mock_npu, mock_use_cann_megamoe, mock_vllm):
        # The transform is not idempotent (a second transpose/reshape corrupts
        # it), so a repeated call must be a no-op. veRL calls process again
        # after load_weights(), which is exactly this path.
        mock_npu.npu_format_cast.side_effect = lambda x, *a, **kw: x
        layer = self._make_moe_layer()
        self.scheme.process_weights_after_loading(layer)
        shapes = {
            k: tuple(getattr(layer, k).shape)
            for k in ("w13_weight", "w2_weight", "w13_weight_scale", "w2_weight_scale")
        }
        recorded = dict(layer._mxfp4_original_shapes)
        self.scheme.process_weights_after_loading(layer)
        for k, shape in shapes.items():
            self.assertEqual(tuple(getattr(layer, k).shape), shape)
        self.assertEqual(layer._mxfp4_original_shapes, recorded)

    @patch("vllm_ascend.quantization.methods.w4a8.w4a8_mxfp4.get_current_vllm_config")
    @patch("vllm_ascend.quantization.methods.w4a8.w4a8_mxfp4.use_cann_megamoe", return_value=False)
    @patch("vllm_ascend.quantization.methods.w4a8.w4a8_mxfp4.torch_npu")
    def test_restore_after_process_returns_original_shape(self, mock_npu, mock_use_cann_megamoe, mock_vllm):
        mock_npu.npu_format_cast.side_effect = lambda x, *a, **kw: x
        layer = self._make_moe_layer()
        original_weight_shapes = tuple(layer.w13_weight.shape), tuple(layer.w2_weight.shape)
        original_scale_shapes = tuple(layer.w13_weight_scale.shape), tuple(layer.w2_weight_scale.shape)
        self.scheme.process_weights_after_loading(layer)
        # Sanity: process really did change the layout.
        self.assertNotEqual(tuple(layer.w13_weight.shape), original_weight_shapes[0])
        self.scheme.restore_weights_for_rl_loading(layer)
        self.assertEqual(tuple(layer.w13_weight.shape), original_weight_shapes[0])
        self.assertEqual(tuple(layer.w2_weight.shape), original_weight_shapes[1])
        self.assertEqual(tuple(layer.w13_weight_scale.shape), original_scale_shapes[0])
        self.assertEqual(tuple(layer.w2_weight_scale.shape), original_scale_shapes[1])
        self.assertEqual(layer.w13_weight.dtype, torch.uint8)
        self.assertFalse(layer._mxfp4_transformed)

    def test_restore_without_process_is_noop(self):
        layer = self._make_moe_layer()
        original_shape = tuple(layer.w13_weight.shape)
        self.scheme.restore_weights_for_rl_loading(layer)
        self.assertEqual(tuple(layer.w13_weight.shape), original_shape)
        self.assertFalse(hasattr(layer, "_mxfp4_transformed"))

    @patch("vllm_ascend.quantization.methods.w4a8.w4a8_mxfp4.get_current_vllm_config")
    @patch("vllm_ascend.quantization.methods.w4a8.w4a8_mxfp4.use_cann_megamoe", return_value=False)
    @patch("vllm_ascend.quantization.methods.w4a8.w4a8_mxfp4.torch_npu")
    def test_refit_cycle_restore_then_process_round_trips(self, mock_npu, mock_use_cann_megamoe, mock_vllm):
        # restore -> load_weights -> process must land back in the inference
        # layout, and the recorded shapes must survive the cycle.
        mock_npu.npu_format_cast.side_effect = lambda x, *a, **kw: x
        layer = self._make_moe_layer()
        self.scheme.process_weights_after_loading(layer)
        inference_shapes = {
            k: tuple(getattr(layer, k).shape)
            for k in ("w13_weight", "w2_weight", "w13_weight_scale", "w2_weight_scale")
        }
        self.scheme.restore_weights_for_rl_loading(layer)
        self.scheme.process_weights_after_loading(layer)
        for k, shape in inference_shapes.items():
            self.assertEqual(tuple(getattr(layer, k).shape), shape)
        self.assertTrue(layer._mxfp4_transformed)

    @patch("vllm_ascend.quantization.methods.w4a8.w4a8_mxfp4.torch_npu")
    @patch("vllm_ascend.quantization.methods.w4a8.w4a8_mxfp4._EXTRA_CTX")
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
        layer.ascend_pertoken_scale = torch.randn(tokens)
        layer.apply_router_weight_on_input = True
        layer.ascend_expert_map = None
        layer.global_redundant_expert_num = 0
        layer.log2phy = None
        layer.ascend_mc2_mask = None
        layer.swiglu_alpha = 1.0
        layer.swiglu_beta = 0.0
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
