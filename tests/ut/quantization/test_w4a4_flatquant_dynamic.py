import unittest
from unittest.mock import MagicMock, patch

import torch
import torch.nn as nn

from vllm_ascend.quantization.w4a4_flatquant_dynamic import (
    AscendW4A4FlatQuantDynamicLinearMethod, get_decompose_dim,
    pack_int4_weights)


class TestW4A4FlatQuantDynamic(unittest.TestCase):
    """
    Unit test suite for AscendW4A4FlatQuantDynamicLinearMethod and its helper functions.
    """

    def setUp(self):
        """Set up the test environment before each test."""
        self.method = AscendW4A4FlatQuantDynamicLinearMethod()
        self.output_size = 64
        self.input_size = 768  # 768 = 24 * 32, divisible by 8
        self.params_dtype = torch.float16

    ## Test Helper Functions
    ## --------------------

    def test_get_decompose_dim(self):
        """
        Tests the get_decompose_dim function with various inputs.
        """
        self.assertEqual(get_decompose_dim(1024), (32, 32))
        self.assertEqual(get_decompose_dim(768), (24, 32))
        self.assertEqual(get_decompose_dim(100), (10, 10))
        self.assertEqual(get_decompose_dim(99), (9, 11))

    @patch('vllm_ascend.quantization.w4a4_flatquant_dynamic.torch_npu')
    def test_pack_int4_weights_npu_success(self, mock_torch_npu):
        """
        Tests weight packing using the mocked NPU kernel.
        """
        weight_tensor = torch.randn(self.output_size, self.input_size)
        mock_packed_tensor = torch.randint(
            0,
            100, (self.output_size, self.input_size // 8),
            dtype=torch.int32)
        mock_npu_tensor = MagicMock()
        mock_npu_tensor.to.return_value = mock_packed_tensor
        mock_torch_npu.npu_convert_weight_to_int4pack.return_value = mock_npu_tensor
        with patch('torch.Tensor.npu', return_value=weight_tensor):
            result = pack_int4_weights(weight_tensor)

        mock_torch_npu.npu_convert_weight_to_int4pack.assert_called_once()
        self.assertTrue(torch.equal(result, mock_packed_tensor))

    ## Test AscendW4A4FlatQuantDynamicLinearMethod Class
    ## --------------------------------------------------

    def test_get_weight(self):
        """Tests the get_weight static method for correct output."""
        params = self.method.get_weight(self.input_size, self.output_size,
                                        self.params_dtype)
        self.assertIn("weight", params)
        self.assertEqual(params["weight"].shape,
                         (self.output_size, self.input_size))
        self.assertEqual(params["weight"].dtype, torch.int8)
        self.assertEqual(AscendW4A4FlatQuantDynamicLinearMethod.input_size,
                         self.input_size)

    def test_get_weight_value_error(self):
        """Tests that get_weight raises ValueError for invalid input_size."""
        with self.assertRaisesRegex(ValueError, "must be divisible by 8"):
            self.method.get_weight(127, self.output_size, self.params_dtype)

    def test_get_pertensor_param(self):
        """Tests the get_pertensor_param static method."""
        self.method.get_weight(self.input_size, self.output_size,
                               self.params_dtype)
        params = self.method.get_pertensor_param(self.params_dtype)
        left_dim, right_dim = get_decompose_dim(self.input_size)
        self.assertIn("left_trans", params)
        self.assertIn("right_trans", params)
        self.assertIn("clip_ratio", params)
        self.assertEqual(params["left_trans"].shape, (left_dim, left_dim))
        self.assertEqual(params["right_trans"].shape, (right_dim, right_dim))
        self.assertEqual(params["clip_ratio"].shape, (1, ))
        self.assertEqual(params["left_trans"].dtype, self.params_dtype)
        self.assertEqual(params["clip_ratio"].dtype, torch.float32)

    def test_get_perchannel_param(self):
        """Tests the get_perchannel_param static method."""
        params = self.method.get_perchannel_param(self.output_size,
                                                  self.params_dtype)
        self.assertIn("weight_scale", params)
        self.assertIn("weight_offset", params)
        self.assertEqual(params["weight_scale"].shape, (self.output_size, 1))
        self.assertEqual(params["weight_offset"].shape, (self.output_size, 1))
        self.assertEqual(params["weight_scale"].dtype, torch.float32)
        self.assertEqual(params["weight_offset"].dtype, torch.float32)

    def test_get_pergroup_param(self):
        """Tests the get_pergroup_param method."""
        params = self.method.get_pergroup_param(self.input_size,
                                                self.output_size,
                                                self.params_dtype)
        self.assertEqual(params, {})

    def _prepare_apply_mocks_and_layer(self, batch_size):
        """Helper to create a mock layer and input tensor for apply tests."""
        layer = nn.Module()
        m, n = get_decompose_dim(self.input_size)
        layer.left_trans = torch.randn(m, m, dtype=self.params_dtype)
        layer.right_trans = torch.randn(n, n, dtype=self.params_dtype)
        layer.aclnn_clip_ratio = 0.95
        layer.weight_packed = torch.randint(
            -8, 7, (self.output_size, self.input_size // 8), dtype=torch.int32)
        layer.weight_scale = torch.randn(self.output_size,
                                         1,
                                         dtype=torch.float32)
        x = torch.randn(batch_size, self.input_size, dtype=self.params_dtype)
        return layer, x, m, n

    @patch('vllm_ascend.quantization.w4a4_flatquant_dynamic.torch_npu')
    def test_apply_small_batch(self, mock_torch_npu):
        """Tests the apply method with a batch size smaller than MAX_BATCH_SIZE."""
        batch_size = 128
        layer, x, m, n = self._prepare_apply_mocks_and_layer(batch_size)
        mock_quant_x = torch.randint(0,
                                     255, (batch_size, self.input_size // 8),
                                     dtype=torch.int32)
        mock_act_scale = torch.randn(batch_size, 1, dtype=torch.float32)
        mock_torch_npu.npu_kronecker_quant.return_value = (mock_quant_x.view(
            batch_size, m, n // 8), mock_act_scale)
        mock_output = torch.randn(batch_size,
                                  self.output_size,
                                  dtype=self.params_dtype)
        mock_torch_npu.npu_quant_matmul.return_value = mock_output
        bias = torch.randn(self.output_size, dtype=self.params_dtype)
        output = self.method.apply(layer, x, bias=bias)
        mock_torch_npu.npu_kronecker_quant.assert_called_once()
        mock_torch_npu.npu_quant_matmul.assert_called_once()
        self.assertTrue(
            torch.allclose(output, mock_output + bias.to(self.params_dtype)))
        self.assertEqual(output.shape, (batch_size, self.output_size))

    @patch(
        'vllm_ascend.quantization.w4a4_flatquant_dynamic.KRONECKER_QUANT_MAX_BATCH_SIZE',
        10)
    @patch('vllm_ascend.quantization.w4a4_flatquant_dynamic.torch_npu')
    def test_apply_large_batch(self, mock_torch_npu):
        """Tests the apply method with a batch size larger than MAX_BATCH_SIZE."""
        batch_size = 25
        layer, x, m, n = self._prepare_apply_mocks_and_layer(batch_size)
        mock_quant_x = torch.randint(0,
                                     255, (batch_size, self.input_size // 8),
                                     dtype=torch.int32)
        mock_act_scale = torch.randn(batch_size, 1, dtype=torch.float32)
        mock_torch_npu.npu_kronecker_quant.side_effect = [
            (mock_quant_x[:10].view(10, m, n // 8), mock_act_scale[:10]),
            (mock_quant_x[10:20].view(10, m, n // 8), mock_act_scale[10:20]),
            (mock_quant_x[20:].view(5, m, n // 8), mock_act_scale[20:]),
        ]
        mock_output = torch.randn(batch_size,
                                  self.output_size,
                                  dtype=self.params_dtype)
        mock_torch_npu.npu_quant_matmul.return_value = mock_output
        output = self.method.apply(layer, x, bias=None)
        self.assertEqual(mock_torch_npu.npu_kronecker_quant.call_count, 3)
        mock_torch_npu.npu_quant_matmul.assert_called_once()
        self.assertTrue(torch.equal(output, mock_output))
        self.assertEqual(output.shape, (batch_size, self.output_size))

    def test_apply_dimension_mismatch_error(self):
        """Tests that apply raises ValueError on transform matrix dimension mismatch."""
        layer, x, _, _ = self._prepare_apply_mocks_and_layer(16)
        layer.left_trans = torch.randn(20, 20)
        layer.right_trans = torch.randn(30, 30)  # 20 * 30 != 768
        with self.assertRaisesRegex(
                ValueError, "FlatQuant transform matrices dimension mismatch"):
            self.method.apply(layer, x)

    @patch('vllm_ascend.quantization.w4a4_flatquant_dynamic.pack_int4_weights')
    def test_process_weights_after_loading(self, mock_pack_weights):
        """Tests weight processing after loading, without transpose."""
        layer = nn.Module()
        layer.weight = torch.randint(-8,
                                     7, (self.output_size, self.input_size),
                                     dtype=torch.int8)
        layer.weight_scale = torch.randn(self.output_size,
                                         1,
                                         dtype=torch.bfloat16)
        layer.weight_offset = torch.randn(self.output_size,
                                          1,
                                          dtype=torch.bfloat16)
        layer.left_trans = torch.randn(24, 24)
        layer.right_trans = torch.randn(32, 32)
        layer.clip_ratio = torch.tensor([0.9])
        mock_packed = torch.randint(0,
                                    100,
                                    (self.output_size, self.input_size // 8),
                                    dtype=torch.int32)
        mock_pack_weights.return_value = mock_packed
        self.method.transpose_weight = False
        self.method.process_weights_after_loading(layer)
        mock_pack_weights.assert_called_once()
        self.assertFalse(hasattr(layer, 'weight'))
        self.assertTrue(hasattr(layer, 'weight_packed'))
        self.assertTrue(torch.equal(layer.weight_packed.data, mock_packed))
        self.assertEqual(layer.weight_scale.dtype, torch.float32)
        self.assertEqual(layer.weight_offset.dtype, torch.float32)
        self.assertEqual(layer.clip_ratio.dtype, torch.float32)
        self.assertTrue(layer.aclnn_clip_ratio - 0.9 < 0.01)
        self.assertEqual(layer.left_trans.shape, (24, 24))
        self.assertTrue(layer.left_trans.is_contiguous())

    @patch('vllm_ascend.quantization.w4a4_flatquant_dynamic.pack_int4_weights')
    def test_process_weights_after_loading_with_transpose(
            self, mock_pack_weights):
        """Tests weight processing after loading, with transpose."""
        layer = nn.Module()
        layer.weight = torch.randint(-8,
                                     7, (self.output_size, self.input_size),
                                     dtype=torch.int8)
        layer.weight_scale = torch.randn(self.output_size,
                                         1,
                                         dtype=torch.bfloat16)
        layer.weight_offset = torch.randn(self.output_size,
                                          1,
                                          dtype=torch.bfloat16)
        layer.left_trans = torch.randn(24, 24)
        layer.right_trans = torch.randn(32, 32)
        layer.clip_ratio = torch.tensor([0.9])
        mock_packed = torch.randint(0,
                                    100,
                                    (self.output_size, self.input_size // 8),
                                    dtype=torch.int32)
        mock_pack_weights.return_value = mock_packed
        self.method.transpose_weight = True
        self.method.process_weights_after_loading(layer)
        self.assertTrue(hasattr(layer, 'weight_packed'))
        self.assertEqual(layer.weight_packed.shape,
                         (self.input_size // 8, self.output_size))
        self.assertTrue(layer.weight_packed.is_contiguous())


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
