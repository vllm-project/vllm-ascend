import os
from unittest.mock import MagicMock, patch

import torch

from tests.ut.base import TestBase
from vllm_ascend.quantization.methods.w8a8_static import AscendW8A8LinearMethod
from vllm_ascend.utils import AscendDeviceType


class TestAscendW8A8LinearMethod(TestBase):

    def setUp(self):
        self.method = AscendW8A8LinearMethod()

    def test_get_weight(self):
        weight = self.method.get_weight(10, 20)
        self.assertEqual(weight['weight'].dtype, torch.int8)
        self.assertEqual(weight['weight'].shape, (20, 10))

    def test_get_pertensor_param(self):
        params = self.method.get_pertensor_param(torch.bfloat16)
        self.assertEqual(params['input_scale'].dtype, torch.bfloat16)
        self.assertEqual(params['input_offset'].dtype, torch.int8)
        self.assertEqual(params['input_scale'].shape, (1, ))
        self.assertEqual(params['input_offset'].shape, (1, ))

    def test_get_perchannel_param(self):
        params = self.method.get_perchannel_param(10, torch.bfloat16)

        self.assertEqual(params['quant_bias'].dtype, torch.int32)
        self.assertEqual(params['deq_scale'].dtype, torch.float32)
        self.assertEqual(params['weight_scale'].dtype, torch.bfloat16)
        self.assertEqual(params['weight_offset'].dtype, torch.bfloat16)
        self.assertEqual(params['quant_bias'].shape, (10, ))
        self.assertEqual(params['deq_scale'].shape, (10, ))
        self.assertEqual(params['weight_scale'].shape, (10, 1))
        self.assertEqual(params['weight_offset'].shape, (10, 1))

    @patch(
        "vllm_ascend.quantization.methods.w8a8_static.get_weight_prefetch_method"
    )
    @patch("torch.ops.vllm.quantize")
    @patch("torch_npu.npu_quant_matmul")
    def test_apply_with_x_not_int8(self, mock_npu_quant_matmul, mock_quantize,
                                   mock_get_weight_prefetch_method):
        layer = MagicMock()
        layer.aclnn_input_scale = 0.1
        layer.aclnn_input_offset = 0.2
        layer.weight = torch.randn(128, 256)
        layer.deq_scale = 0.3

        mock_get_weight_prefetch_method.return_value = MagicMock()

        x = torch.randn(32, 128)
        bias = torch.randn(256)
        mock_quantize.return_value = torch.randint(-128,
                                                   127,
                                                   x.shape,
                                                   dtype=torch.int8)

        expected_y_output = torch.randn(32, 256)
        mock_npu_quant_matmul.return_value = expected_y_output

        output = self.method.apply(layer, x, bias)

        expected_y_output += bias
        self.assertTrue(torch.equal(output, expected_y_output))

    @patch("torch_npu.npu_quant_matmul")
    def test_apply_with_x_is_int8(self, mock_npu_quant_matmul):
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

    @patch('vllm_ascend.utils.get_ascend_device_type',
           return_value=AscendDeviceType._310P)
    @patch("torch_npu.npu_quant_matmul")
    def test_apply_with_x_is_310p(self, mock_npu_quant_matmul,
                                  mock_soc_version):
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

    @patch.dict(os.environ, {"VLLM_ASCEND_ENABLE_NZ": "0"})
    @patch('torch_npu.npu_format_cast')
    def test_process_weights_after_loading_with_nz0(self,
                                                    mock_npu_format_cast):
        layer = MagicMock()

        layer.weight.data = torch.randint(-127,
                                          128, (128, 256),
                                          dtype=torch.int8)
        layer.input_scale.data = torch.tensor([0.1])
        layer.input_offset.data = torch.tensor([0])
        layer.deq_scale = torch.tensor([0.5])
        layer.weight_scale.data = torch.randn(128, 1)
        layer.weight_offset.data = torch.randn(128, 1)

        mock_npu_format_cast.return_value = MagicMock
        self.method.process_weights_after_loading(layer)

        expected_offset = torch.tensor([0]).repeat(256).to(torch.int8)
        self.assertTrue(
            torch.equal(layer.aclnn_input_offset.data, expected_offset))
        self.assertFalse(layer.aclnn_input_offset.requires_grad)

        self.assertFalse(layer.deq_scale.requires_grad)

        self.assertEqual(layer.weight_scale.data.shape, (128, ))
        self.assertEqual(layer.weight_offset.data.shape, (128, ))
        mock_npu_format_cast.assert_not_called()

    @patch.dict(os.environ, {"VLLM_ASCEND_ENABLE_NZ": "1"})
    @patch('torch_npu.npu_format_cast')
    def test_process_weights_after_loading_with_nz1(self,
                                                    mock_npu_format_cast):
        layer = MagicMock()

        layer.weight.data = torch.randint(-127,
                                          128, (128, 256),
                                          dtype=torch.int8)
        layer.input_scale.data = torch.tensor([0.1])
        layer.input_offset.data = torch.tensor([0])
        layer.deq_scale = torch.tensor([0.5])
        layer.weight_scale.data = torch.randn(128, 1)
        layer.weight_offset.data = torch.randn(128, 1)

        mock_npu_format_cast.return_value = MagicMock
        self.method.process_weights_after_loading(layer)

        expected_offset = torch.tensor([0]).repeat(256).to(torch.int8)
        self.assertTrue(
            torch.equal(layer.aclnn_input_offset.data, expected_offset))
        self.assertFalse(layer.aclnn_input_offset.requires_grad)

        self.assertFalse(layer.deq_scale.requires_grad)

        self.assertEqual(layer.weight_scale.data.shape, (128, ))
        self.assertEqual(layer.weight_offset.data.shape, (128, ))
        mock_npu_format_cast.assert_called_once()

    @patch.dict(os.environ, {"VLLM_ASCEND_ENABLE_NZ": "2"})
    @patch('torch_npu.npu_format_cast')
    def test_process_weights_after_loading_with_nz2(self,
                                                    mock_npu_format_cast):
        layer = MagicMock()

        layer.weight.data = torch.randint(-127,
                                          128, (128, 256),
                                          dtype=torch.int8)
        layer.input_scale.data = torch.tensor([0.1])
        layer.input_offset.data = torch.tensor([0])
        layer.deq_scale = torch.tensor([0.5])
        layer.weight_scale.data = torch.randn(128, 1)
        layer.weight_offset.data = torch.randn(128, 1)

        mock_npu_format_cast.return_value = MagicMock
        self.method.process_weights_after_loading(layer)

        expected_offset = torch.tensor([0]).repeat(256).to(torch.int8)
        self.assertTrue(
            torch.equal(layer.aclnn_input_offset.data, expected_offset))
        self.assertFalse(layer.aclnn_input_offset.requires_grad)

        self.assertFalse(layer.deq_scale.requires_grad)

        self.assertEqual(layer.weight_scale.data.shape, (128, ))
        self.assertEqual(layer.weight_offset.data.shape, (128, ))
        mock_npu_format_cast.assert_called_once()
