from unittest.mock import MagicMock, Mock, patch

import torch
import torch.nn as nn
from vllm.model_executor.layers.linear import RowParallelLinear

from tests.ut.base import TestBase
from tests.ut.quantization.conftest_quantization import create_mock_ascend_config, create_mock_vllm_config
from vllm_ascend.quantization.method_adapters import AscendLinearMethod
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

    def test_get_weight_rejects_split_fp4_byte(self):
        with self.assertRaisesRegex(ValueError, "even input size"):
            self.scheme.get_weight(1077, 128, torch.bfloat16)

    def test_get_pergroup_param_based_on_group_size(self):
        group_sizes = [16, 32, 64]
        for gs in group_sizes:
            self.scheme.group_size = gs
            result = self.scheme.get_pergroup_param(256, 128, torch.bfloat16)
            self.assertEqual(result["weight_scale"].shape, (128, 256 // gs))
            self.assertEqual(result["weight_scale"].dtype, torch.uint8)

    def test_get_pergroup_param_preserves_partial_group(self):
        self.scheme.group_size = 32
        result = self.scheme.get_pergroup_param(4304, 128, torch.bfloat16)
        self.assertEqual(result["weight_scale"].shape, (128, 135))

    @patch("vllm_ascend.quantization.methods.w4a8.w4a8_mxfp4.torch_npu")
    def test_partial_scale_group_padding(self, mock_npu):
        mock_npu.npu_format_cast.side_effect = lambda x, *a, **kw: x
        layer = nn.Module()
        layer.weight = nn.Parameter(torch.zeros(8, 2152, dtype=torch.uint8), requires_grad=False)
        scales = torch.randint(0, 255, (8, 135), dtype=torch.uint8)
        layer.weight_scale = nn.Parameter(scales.clone(), requires_grad=False)
        for _ in range(2):
            self.scheme.process_weights_after_loading(layer)
            unpacked = layer.weight_scale.transpose(-3, -2).reshape(8, 136)
            self.assertTrue(torch.equal(unpacked[:, :135], scales))
            self.assertTrue(torch.equal(unpacked[:, 135], torch.zeros(8, dtype=torch.uint8)))

    @patch("vllm_ascend.quantization.methods.w4a8.w4a8_mxfp4.torch_npu")
    def test_apply_restores_activation_shape_and_expands_bias(self, mock_npu):
        for shape in [(128,), (6, 128), (2, 3, 128), (2, 2, 3, 128)]:
            with self.subTest(shape=shape):
                x = torch.randn(*shape, dtype=torch.bfloat16)
                flat_x = x.reshape(-1, shape[-1])
                quantized = torch.zeros_like(flat_x, dtype=torch.float8_e4m3fn)
                scale = torch.zeros(flat_x.shape[0], 2, 2, dtype=torch.uint8)
                mock_npu.npu_dynamic_mx_quant.return_value = (quantized, scale)
                projected = torch.randn(flat_x.shape[0], 16, dtype=x.dtype)
                mock_npu.npu_quant_matmul.return_value = projected
                bias = torch.randn(16, dtype=x.dtype)
                output = self.scheme.apply(MagicMock(spec=["weight", "weight_scale"]), x, bias)
                quant_input = mock_npu.npu_dynamic_mx_quant.call_args.args[0]
                self.assertTrue(torch.equal(quant_input, flat_x))
                matmul_bias = mock_npu.npu_quant_matmul.call_args.kwargs["bias"]
                self.assertEqual(matmul_bias.shape, (1, 16))
                self.assertTrue(torch.equal(matmul_bias[0], bias))
                self.assertEqual(output.shape, (*shape[:-1], 16))
                self.assertTrue(torch.equal(output.reshape(-1, 16), projected))

    def _make_row_layer(self, input_size, tp_size, tp_rank):
        layer = RowParallelLinear.__new__(RowParallelLinear)
        nn.Module.__init__(layer)
        layer.tp_size = tp_size
        layer.tp_rank = tp_rank
        layer.input_size = input_size
        layer.input_size_per_partition = input_size // tp_size
        layer.output_size = layer.output_size_per_partition = 8
        AscendLinearMethod(self.scheme).create_weights(
            layer,
            input_size_per_partition=layer.input_size_per_partition,
            output_partition_sizes=[8],
            input_size=input_size,
            output_size=8,
            params_dtype=torch.bfloat16,
            weight_loader=layer.weight_loader,
        )
        return layer

    @staticmethod
    def _dequantize_weight(packed_weight, scales):
        values = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6])
        codes = torch.stack((packed_weight & 15, packed_weight >> 4), dim=-1).flatten(-2).long()
        expanded_scales = torch.exp2(scales.float() - 127).repeat_interleave(32, dim=-1)
        return values[codes] * expanded_scales[:, : codes.shape[-1]]

    @patch("vllm_ascend.quantization.methods.w4a8.w4a8_mxfp4.torch_npu")
    def test_checkpoint_tp_shards_preserve_dequantized_projection(self, mock_npu):
        mock_npu.npu_format_cast.side_effect = lambda x, *a, **kw: x
        input_size = 4304
        checkpoint_weight = torch.randint(0, 256, (8, input_size // 2), dtype=torch.uint8)
        checkpoint_scales = torch.randint(125, 130, (8, 135), dtype=torch.uint8)
        x = torch.randint(-2, 3, (2, input_size), dtype=torch.int32).float()
        expected = x @ self._dequantize_weight(checkpoint_weight, checkpoint_scales).T
        for tp_size in (1, 2, 4, 8):
            actual = torch.zeros_like(expected)
            for tp_rank in range(tp_size):
                with self.subTest(tp_size=tp_size, tp_rank=tp_rank):
                    layer = self._make_row_layer(input_size, tp_size, tp_rank)
                    layer.weight.weight_loader(layer.weight, checkpoint_weight)
                    layer.weight_scale.weight_loader(layer.weight_scale, checkpoint_scales)
                    local_scales = layer.weight_scale.clone()
                    start = tp_rank * layer.input_size_per_partition
                    end = start + layer.input_size_per_partition
                    torch.testing.assert_close(local_scales, checkpoint_scales[:, start // 32 : (end + 31) // 32])
                    for _ in range(2):
                        self.scheme.process_weights_after_loading(layer)
                        prefix, suffix = layer.mxfp4_tp_padding
                        padded_x = torch.nn.functional.pad(x[:, start:end], (prefix, suffix))
                        weight = layer.weight.T
                        scales = layer.weight_scale.transpose(0, 1).reshape(8, -1)
                        projection = padded_x @ self._dequantize_weight(weight, scales).T
                    actual += projection
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    @patch("vllm_ascend.quantization.methods.w4a8.w4a8_mxfp4.torch_npu")
    def test_tp_padding_reaches_plain_and_overlap_quantization(self, mock_npu):
        mock_npu.npu_format_cast.side_effect = lambda x, *a, **kw: x
        layer = self._make_row_layer(4304, 4, 3)
        self.scheme.process_weights_after_loading(layer)
        self.assertEqual(layer.mxfp4_tp_padding, (28, 16))
        x = torch.randn(2, 1076, dtype=torch.bfloat16)
        padded_x = torch.nn.functional.pad(x, (28, 16))
        quantized = torch.zeros(2, 1120, dtype=torch.float8_e4m3fn)
        scale = torch.zeros(2, 18, 2, dtype=torch.uint8)
        mock_npu.npu_dynamic_mx_quant.return_value = (quantized, scale)
        mock_npu.npu_quant_matmul.return_value = torch.zeros(2, 8, dtype=x.dtype)
        self.scheme.apply(layer, x)
        torch.testing.assert_close(mock_npu.npu_dynamic_mx_quant.call_args.args[0], padded_x)
        prepared = self.scheme.prepare_input_for_overlap(layer, x)
        torch.testing.assert_close(mock_npu.npu_dynamic_mx_quant.call_args.args[0], padded_x)
        mock_npu.npu_dynamic_mx_quant.reset_mock()
        self.scheme.apply(layer, prepared)
        mock_npu.npu_dynamic_mx_quant.assert_not_called()
        with self.assertRaisesRegex(ValueError, "TP group padding"):
            self.scheme.apply(layer, (torch.zeros(2, 1076, dtype=torch.float8_e4m3fn), scale))

    @patch("vllm_ascend.quantization.methods.w4a8.w4a8_mxfp4.torch_npu")
    def test_tp_padding_is_delegated_and_independent_of_weight_layout(self, mock_npu):
        class LinearWrapper(torch.nn.Module):
            def __init__(self, inner):
                super().__init__()
                self.inner = inner

            def __getattr__(self, name):
                try:
                    return super().__getattr__(name)
                except AttributeError:
                    return getattr(self.inner, name)

        mock_npu.npu_format_cast.side_effect = lambda x, *a, **kw: x
        layer = self._make_row_layer(4304, 4, 3)
        self.scheme.process_weights_after_loading(layer)
        # Simulate an alternate storage view; input validation must use the
        # logical partition width rather than infer it from weight storage.
        layer.weight.data = layer.weight.data.reshape(35, 16, 8)
        wrapped = LinearWrapper(layer)
        x = torch.randn(2, 1076, dtype=torch.bfloat16)
        quantized = torch.zeros(2, 1120, dtype=torch.float8_e4m3fn)
        scale = torch.zeros(2, 18, 2, dtype=torch.uint8)
        mock_npu.npu_dynamic_mx_quant.return_value = (quantized, scale)
        mock_npu.npu_quant_matmul.return_value = torch.zeros(2, 8, dtype=x.dtype)
        prepared = self.scheme.prepare_input_for_overlap(wrapped, x)
        torch.testing.assert_close(
            mock_npu.npu_dynamic_mx_quant.call_args.args[0], torch.nn.functional.pad(x, (28, 16))
        )
        self.scheme.apply(wrapped, prepared)
        self.scheme.apply(wrapped, x)
        with self.assertRaisesRegex(ValueError, "TP group padding"):
            self.scheme.apply(wrapped, (torch.zeros(2, 1076, dtype=torch.float8_e4m3fn), scale))

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
    def test_process_marks_transformed(self, mock_npu):
        mock_npu.npu_format_cast.side_effect = lambda x, *a, **kw: x
        layer = self._make_linear_layer()
        self.scheme.process_weights_after_loading(layer)
        self.assertTrue(layer._mxfp4_transformed)

    @patch("vllm_ascend.quantization.methods.w4a8.w4a8_mxfp4.torch_npu")
    def test_process_is_idempotent(self, mock_npu):
        # The transform is not idempotent (a second transpose/reshape corrupts
        # it), so a repeated call must be a no-op.
        mock_npu.npu_format_cast.side_effect = lambda x, *a, **kw: x
        layer = self._make_linear_layer()
        self.scheme.process_weights_after_loading(layer)
        weight_shape = layer.weight.shape
        scale_shape = layer.weight_scale.shape
        self.scheme.process_weights_after_loading(layer)
        self.assertEqual(layer.weight.shape, weight_shape)
        self.assertEqual(layer.weight_scale.shape, scale_shape)

    @patch("vllm_ascend.quantization.methods.w4a8.w4a8_mxfp4.torch_npu")
    def test_apply_with_prequantized_input(self, mock_npu):
        mock_npu.npu_quant_matmul.return_value = torch.randn(32, 128)
        layer = MagicMock()
        layer.mxfp4_tp_padding = (0, 0)
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
        layer.mxfp4_tp_padding = (0, 0)
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
    @patch("vllm_ascend.quantization.methods.w4a8.w4a8_mxfp4.use_cann_megamoe", return_value=True)
    @patch("vllm_ascend.quantization.methods.w4a8.w4a8_mxfp4.torch_npu")
    def test_megamoe_scales_preserve_e8m0_bytes(self, mock_npu, mock_use_cann_megamoe, mock_vllm):
        mock_npu.npu_format_cast.side_effect = lambda x, *a, **kw: x
        layer = self._make_moe_layer()
        original_scales = {}
        for name in ("w13_weight_scale", "w2_weight_scale"):
            scale = getattr(layer, name).data
            # Cover every byte encoding to detect accidental numeric conversion.
            scale.copy_(torch.arange(scale.numel()).to(torch.uint8).reshape(scale.shape))
            original_scales[name] = scale.clone()

        self.scheme.process_weights_after_loading(layer)

        for name, original in original_scales.items():
            scales = getattr(layer, f"cann_mega_moe_{name}_list")
            self.assertEqual(len(scales), original.shape[0])
            for expert, scale in enumerate(scales):
                with self.subTest(name=name, expert=expert):
                    self.assertEqual(scale.dtype, torch.float8_e8m0fnu)
                    self.assertEqual(scale.shape, (original.shape[1], original.shape[2] // 2, 2))
                    torch.testing.assert_close(scale.view(torch.uint8).flatten(), original[expert].flatten())

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
