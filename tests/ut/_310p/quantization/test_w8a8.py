from unittest.mock import MagicMock, patch

import torch

from tests.ut.base import TestBase
from vllm_ascend._310p.quantization.methods.w8a8_static import AscendW8A8LinearMethod310


class TestAscendW8A8LinearMethod310(TestBase):
    def setUp(self):
        self.method = AscendW8A8LinearMethod310()

    def test_get_weight_310(self):
        weight = self.method.get_weight(10, 20)
        self.assertEqual(weight["weight"].dtype, torch.int8)
        self.assertEqual(weight["weight"].shape, (20, 10))

    def test_get_pertensor_param_310(self):
        params = self.method.get_pertensor_param(torch.bfloat16)
        self.assertEqual(params["input_scale"].dtype, torch.bfloat16)
        self.assertEqual(params["input_offset"].dtype, torch.int8)
        self.assertEqual(params["input_scale"].shape, (1,))
        self.assertEqual(params["input_offset"].shape, (1,))

    def test_get_perchannel_param_310(self):
        params = self.method.get_perchannel_param(10, torch.bfloat16)

        self.assertEqual(params["quant_bias"].dtype, torch.int32)
        self.assertEqual(params["deq_scale"].dtype, torch.float32)
        self.assertEqual(params["weight_scale"].dtype, torch.bfloat16)
        self.assertEqual(params["weight_offset"].dtype, torch.bfloat16)
        self.assertEqual(params["quant_bias"].shape, (10,))
        self.assertEqual(params["deq_scale"].shape, (10,))
        self.assertEqual(params["weight_scale"].shape, (10, 1))
        self.assertEqual(params["weight_offset"].shape, (10, 1))

    @patch("vllm_ascend.quantization.methods.w8a8_static.get_weight_prefetch_method")
    @patch("torch.ops.vllm.quantize")
    @patch("torch_npu.npu_quant_matmul")
    def test_apply_with_x_not_int8_310(self, mock_npu_quant_matmul, mock_quantize, mock_get_weight_prefetch_method):
        layer = MagicMock()
        layer.aclnn_input_scale = 0.1
        layer.aclnn_input_offset = 0.2
        layer.weight = torch.randn(128, 256)
        layer.deq_scale = 0.3

        mock_get_weight_prefetch_method.return_value = MagicMock()

        x = torch.randn(32, 128)
        bias = torch.randn(256)
        mock_quantize.return_value = torch.randint(-128, 127, x.shape, dtype=torch.int8)

        expected_y_output = torch.randn(32, 256)
        mock_npu_quant_matmul.return_value = expected_y_output

        output = self.method.apply(layer, x, bias)

        expected_y_output += bias
        self.assertTrue(torch.equal(output, expected_y_output))

    @patch("torch_npu.npu_quant_matmul")
    def test_apply_with_x_is_int8_310(self, mock_npu_quant_matmul):
        layer = MagicMock()
        layer.aclnn_input_scale = 0.1
        layer.aclnn_input_offset = 0.2
        layer.weight = torch.randn(128, 256)
        layer.deq_scale = 0.3

        x = torch.randint(-128, 127, (32, 128), dtype=torch.int8)
        bias = torch.randn(256)

        expected_y_output = torch.randn(32, 256)
        mock_npu_quant_matmul.return_value = expected_y_output

        output = self.method.apply(layer, x, bias)
        expected_y_output += bias
        self.assertTrue(torch.equal(output, expected_y_output))
