import unittest
from unittest.mock import MagicMock, Mock, patch

import torch

from tests.ut.base import TestBase
from vllm_ascend.quantization.methods.w4a4_mxfp4_flatquant import (
    MAX_SUPPORT_DIM,
    AscendW4A4MXFP4FlatQuantDynamicLinearMethod,
    get_decompose_dim,
)


class TestGetDecomposeDim(TestBase):
    """Unit tests for the get_decompose_dim helper."""

    def test_perfect_square_decomposition(self):
        self.assertEqual(get_decompose_dim(1024, 1), (32, 32))

    def test_non_square_decomposition(self):
        left, right = get_decompose_dim(32, 1)
        self.assertEqual((left, right), (4, 8))
        self.assertEqual(left * right, 32)

    def test_decomposition_product_equals_n(self):
        left, right = get_decompose_dim(256, 1)
        self.assertEqual(left * right, 256)

    def test_raises_when_dim_sum_exceeds_max(self):
        n = (MAX_SUPPORT_DIM + 1) ** 2
        with self.assertRaisesRegex(ValueError, "should be less than"):
            get_decompose_dim(n, 1)

    def test_fallback_when_left_times_m_exceeds_max(self):
        n = MAX_SUPPORT_DIM * MAX_SUPPORT_DIM
        left, right = get_decompose_dim(n, 2)
        self.assertEqual(left, MAX_SUPPORT_DIM)
        self.assertEqual(right, 2 * n // MAX_SUPPORT_DIM)


class TestAscendW4A4MXFP4FlatQuantDynamicLinearMethod(TestBase):
    """Unit tests for AscendW4A4MXFP4FlatQuantDynamicLinearMethod."""

    input_size = 1024
    output_size = 64
    group_size = 32
    max_supported_tp = 4

    def _build_method(self, tp_size=1, max_supported_tp=None, group_size=None):
        max_supported_tp = self.max_supported_tp if max_supported_tp is None else max_supported_tp
        group_size = self.group_size if group_size is None else group_size
        mock_vllm_config = Mock()
        mock_vllm_config.quant_config = Mock(
            quant_description={"group_size": group_size, "max_supported_tp": max_supported_tp}
        )
        with (
            patch("vllm_ascend.quantization.methods.w4a4_mxfp4_flatquant.ensure_mxfp4_flatquant_linear_available"),
            patch(
                "vllm_ascend.quantization.methods.w4a4_mxfp4_flatquant.get_current_vllm_config",
                return_value=mock_vllm_config,
            ),
            patch(
                "vllm_ascend.quantization.methods.w4a4_mxfp4_flatquant.get_tensor_model_parallel_world_size",
                return_value=tp_size,
            ),
        ):
            return AscendW4A4MXFP4FlatQuantDynamicLinearMethod()

    def setUp(self):
        self.method = self._build_method()

    def test_init_default(self):
        self.assertEqual(self.method.group_size, self.group_size)
        self.assertEqual(self.method.max_supported_tp, self.max_supported_tp)
        self.assertEqual(self.method.tp_size, 1)

    def test_init_raises_on_oversized_tp(self):
        with self.assertRaisesRegex(ValueError, "is not supported"):
            self._build_method(tp_size=8, max_supported_tp=4)

    def test_get_weight(self):
        params = self.method.get_weight(self.input_size, self.output_size, torch.bfloat16)
        self.assertIn("weight", params)
        self.assertEqual(params["weight"].dtype, torch.uint8)
        self.assertEqual(params["weight"].shape, (self.output_size, self.input_size // 2))
        self.assertEqual(self.method.input_size, self.input_size)

    def test_get_weight_raises_on_odd_input(self):
        with self.assertRaisesRegex(ValueError, "must be divisible by 2"):
            self.method.get_weight(127, self.output_size, torch.bfloat16)

    def test_get_pertensor_param_non_row(self):
        self.method.get_weight(self.input_size, self.output_size, torch.bfloat16)
        params = self.method.get_pertensor_param(torch.bfloat16, layer_type="others")
        left_dim, right_dim = get_decompose_dim(self.input_size, 1)
        self.assertEqual(params["left_trans"].shape, (left_dim, left_dim))
        self.assertEqual(params["right_trans"].shape, (right_dim, right_dim))
        self.assertEqual(params["clip_ratio"].shape, (1,))
        self.assertEqual(params["left_trans"].dtype, torch.bfloat16)
        self.assertEqual(params["right_trans"].dtype, torch.bfloat16)
        self.assertEqual(params["clip_ratio"].dtype, torch.float32)

    def test_get_pertensor_param_row(self):
        self.method.get_weight(self.input_size, self.output_size, torch.bfloat16)
        params = self.method.get_pertensor_param(torch.bfloat16, layer_type="row")
        origin_size = self.input_size * self.method.tp_size
        _, right_trans_dim = get_decompose_dim(
            origin_size // self.method.max_supported_tp, self.method.max_supported_tp
        )
        left_trans_dim = origin_size // right_trans_dim
        self.assertEqual(params["left_trans"].shape, (left_trans_dim, left_trans_dim))
        self.assertEqual(params["right_trans"].shape, (right_trans_dim, right_trans_dim))

    def test_get_pergroup_param(self):
        params = self.method.get_pergroup_param(self.input_size, self.output_size, torch.bfloat16)
        self.assertIn("weight_scale", params)
        self.assertEqual(params["weight_scale"].dtype, torch.uint8)
        self.assertEqual(
            params["weight_scale"].shape,
            (self.output_size, self.input_size // self.group_size),
        )

    @patch("vllm_ascend.quantization.methods.w4a4_mxfp4_flatquant.torch_npu")
    def test_apply(self, mock_torch_npu):
        layer = MagicMock()
        layer.left_trans = torch.randn(32, 32)
        layer.right_trans = torch.randn(32, 32)
        layer.aclnn_clip_ratio = 0.9
        layer.weight = MagicMock()
        layer.weight_scale = MagicMock()

        batch = 8
        x = torch.randn(batch, self.input_size, dtype=torch.bfloat16)
        bias = torch.randn(self.output_size, dtype=torch.bfloat16)

        mock_torch_npu.npu_kronecker_quant.return_value = (MagicMock(), MagicMock())
        expected_output = torch.randn(batch, self.output_size, dtype=torch.bfloat16)
        mock_torch_npu.npu_quant_matmul.return_value = expected_output

        output = self.method.apply(layer, x, bias=bias)

        mock_torch_npu.npu_kronecker_quant.assert_called_once()
        mock_torch_npu.npu_quant_matmul.assert_called_once()
        call_kwargs = mock_torch_npu.npu_quant_matmul.call_args.kwargs
        self.assertIs(call_kwargs["bias"], bias)
        self.assertEqual(call_kwargs["output_dtype"], torch.bfloat16)
        self.assertEqual(call_kwargs["group_sizes"], [1, 1, self.method.group_size])
        self.assertEqual(output.shape, (batch, self.output_size))

    @patch("vllm_ascend.quantization.methods.w4a4_mxfp4_flatquant.torch_npu")
    def test_apply_preserves_input_shape(self, mock_torch_npu):
        layer = MagicMock()
        layer.left_trans = torch.randn(32, 32)
        layer.right_trans = torch.randn(32, 32)
        layer.aclnn_clip_ratio = 0.9
        x = torch.randn(2, 4, self.input_size, dtype=torch.bfloat16)
        mock_torch_npu.npu_kronecker_quant.return_value = (MagicMock(), MagicMock())
        mock_torch_npu.npu_quant_matmul.return_value = torch.randn(8, self.output_size, dtype=torch.bfloat16)
        output = self.method.apply(layer, x)
        self.assertEqual(output.shape, (2, 4, self.output_size))

    def test_apply_dimension_mismatch_raises(self):
        layer = MagicMock()
        layer.left_trans = torch.randn(16, 16)
        layer.right_trans = torch.randn(16, 16)
        x = torch.randn(4, self.input_size)
        with self.assertRaisesRegex(ValueError, "dimension mismatch"):
            self.method.apply(layer, x)

    def test_process_weights_after_loading(self):
        layer = MagicMock()
        layer.weight.data = torch.randint(0, 255, (self.output_size, self.input_size // 2), dtype=torch.uint8)

        weight_scale_data = torch.randint(
            0, 255, (self.output_size, self.input_size // self.group_size), dtype=torch.uint8
        )
        layer.weight_scale.data = weight_scale_data
        layer.weight_scale.shape = weight_scale_data.shape
        layer.left_trans.data = torch.randn(32, 32, dtype=torch.bfloat16)
        layer.right_trans.data = torch.randn(32, 32, dtype=torch.bfloat16)
        layer.clip_ratio.data = torch.tensor([0.95])

        self.method.process_weights_after_loading(layer)

        # weight transposed: (output, input/2) -> (input/2, output)
        self.assertEqual(layer.weight.data.shape, (self.input_size // 2, self.output_size))
        # weight_scale view+transpose: (out, in/group) -> (in/group/2, out, 2)
        self.assertEqual(
            layer.weight_scale.data.shape,
            (self.input_size // self.group_size // 2, self.output_size, 2),
        )
        # left_trans is parameterized after a t().contiguous(); shape remains (32, 32)
        self.assertIsInstance(layer.left_trans, torch.nn.Parameter)
        self.assertEqual(layer.left_trans.shape, (32, 32))
        self.assertTrue(layer.left_trans.data.is_contiguous())
        # clip_ratio cast to float32, aclnn_clip_ratio set to its scalar value
        self.assertEqual(layer.clip_ratio.dtype, torch.float32)
        self.assertAlmostEqual(layer.aclnn_clip_ratio, 0.95, places=5)


if __name__ == "__main__":
    unittest.main(argv=["first-arg-is-ignored"], exit=False)
