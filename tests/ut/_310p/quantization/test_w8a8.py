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

    @patch("torch.ops.vllm.quantize")
    @patch("torch_npu.npu_quant_matmul")
    def test_apply_with_x_not_int8_310(self, mock_npu_quant_matmul, mock_quantize):
        layer = MagicMock()
        layer.aclnn_input_scale = torch.randn(256)
        layer.aclnn_input_scale_reciprocal = 1.0 / layer.aclnn_input_scale
        layer.aclnn_input_offset = torch.randint(-128, 127, (256,), dtype=torch.int8)
        layer.weight = torch.randn(128, 256)
        layer.deq_scale = torch.randn(128)
        layer.quant_bias = torch.randint(-128, 127, (256,))
        layer.params_dtype = torch.float16

        x = torch.randn(32, 128)
        expect_x_output = torch.randint(-128, 127, x.shape, dtype=torch.int8)
        mock_quantize.return_value = expect_x_output

        expected_y_output = torch.randn(32, 256)
        mock_npu_quant_matmul.return_value = expected_y_output

        output = self.method.apply(layer, x, tp_rank=0)

        mock_quantize.assert_called_with(
            x, layer.aclnn_input_scale, layer.aclnn_input_scale_reciprocal, layer.aclnn_input_offset
        )
        mock_npu_quant_matmul.assert_called_with(
            expect_x_output, layer.weight, layer.deq_scale, bias=layer.quant_bias, output_dtype=layer.params_dtype
        )
        # The bias is added by the linear layer's forward pass, not the quant method.
        self.assertTrue(torch.equal(output, expected_y_output))

    @patch("torch.ops.vllm.quantize")
    @patch("torch_npu.npu_quant_matmul")
    def test_apply_with_x_is_int8_310(self, mock_npu_quant_matmul, mock_quantize):
        layer = MagicMock()
        layer.aclnn_input_scale = torch.randn(256)
        layer.aclnn_input_offset = torch.randint(-128, 127, (256,))
        layer.weight = torch.randn(128, 256)
        layer.deq_scale = torch.randn(128)
        layer.quant_bias = torch.randint(-128, 127, (256,))
        layer.params_dtype = torch.float16

        x = torch.randint(-128, 127, (32, 128), dtype=torch.int8)

        expected_y_output = torch.randn(32, 256)
        mock_npu_quant_matmul.return_value = expected_y_output

        output = self.method.apply(layer, x, tp_rank=0)

        mock_quantize.assert_not_called()
        mock_npu_quant_matmul.assert_called_with(
            x, layer.weight, layer.deq_scale, bias=layer.quant_bias, output_dtype=layer.params_dtype
        )
        # The bias is added by the linear layer's forward pass, not the quant method.
        self.assertTrue(torch.equal(output, expected_y_output))
