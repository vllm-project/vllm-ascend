import os
from unittest.mock import MagicMock, patch

import torch

from tests.ut.base import TestBase
from tests.ut.quantization.conftest_quantization import identity
from vllm_ascend.quantization.methods.w8a16 import AscendW8A16LinearMethod


class TestAscendW8A16LinearMethod(TestBase):
    def setUp(self):
        self.method = AscendW8A16LinearMethod()

    def test_get_weight(self):
        sizes = [(64, 128), (256, 512), (1024, 2048), (1, 1)]
        for input_size, output_size in sizes:
            weight = self.method.get_weight(input_size, output_size)
            self.assertEqual(weight["weight"].dtype, torch.int8)
            self.assertEqual(weight["weight"].shape, (output_size, input_size))
            self.assertEqual(len(weight), 1)

        weight = self.method.get_weight(256, 128, torch.float16)
        self.assertEqual(weight["weight"].dtype, torch.int8)

    def test_get_per_channel_param(self):
        for output_size, dtype in [(128, torch.bfloat16), (256, torch.float16)]:
            per_channel_params = self.method.get_perchannel_param(output_size, dtype)
            self.assertEqual(per_channel_params["weight_scale"].dtype, dtype)
            self.assertEqual(per_channel_params["weight_scale"].shape, (output_size, 1))
            self.assertEqual(per_channel_params["weight_offset"].dtype, dtype)
            self.assertEqual(per_channel_params["weight_offset"].shape, (output_size, 1))
            self.assertEqual(len(per_channel_params), 2)

    def test_get_pertensor_param_returns_empty(self):
        result = self.method.get_pertensor_param(torch.bfloat16)
        self.assertEqual(result, {})

    def test_get_pergroup_param_returns_empty(self):
        result = self.method.get_pergroup_param(256, 128, torch.bfloat16)
        self.assertEqual(result, {})

    @patch("torch_npu.npu_weight_quant_batchmatmul")
    def test_apply_with_x_is_int8(self, mock_npu_weight_quant_batchmatmul):
        layer = MagicMock()
        layer.weight.data = torch.randn(128, 256)
        layer.weight_scale.data = torch.randn(128, 1)
        layer.weight_offset.data = torch.randn(128, 1)

        x = torch.randn(32, 128)
        bias = torch.randn(256)

        expected_y_output = torch.randn(32, 256)
        mock_npu_weight_quant_batchmatmul.return_value = expected_y_output

        output = self.method.apply(layer, x, bias)
        self.assertTrue(torch.equal(output, expected_y_output))
        mock_npu_weight_quant_batchmatmul.assert_called_once()

    @patch.dict(os.environ, {"VLLM_ASCEND_ENABLE_NZ": "0"})
    @patch("torch_npu.npu_format_cast")
    def test_process_weights_after_loading_with_nz0(self, mock_npu_format_cast):
        layer = MagicMock()
        layer.weight.data = torch.randint(-128, 127, (128, 256), dtype=torch.int8)
        layer.weight_scale.data = torch.randn(128, 1)
        layer.weight_offset.data = torch.randn(128, 1)

        mock_npu_format_cast.side_effect = identity
        self.method.process_weights_after_loading(layer)

        self.assertEqual(layer.weight.data.shape, (256, 128))
        self.assertEqual(layer.weight_scale.data.shape, (128,))
        self.assertEqual(layer.weight_offset.data.shape, (128,))
        mock_npu_format_cast.assert_not_called()

    @patch.dict(os.environ, {"VLLM_ASCEND_ENABLE_NZ": "1"})
    @patch("torch_npu.npu_format_cast")
    def test_process_weights_after_loading_with_nz1(self, mock_npu_format_cast):
        layer = MagicMock()

        layer.weight.data = torch.randint(-128, 127, (128, 256), dtype=torch.int8)
        layer.weight_scale.data = torch.randn(128, 1)
        layer.weight_offset.data = torch.randn(128, 1)

        mock_npu_format_cast.side_effect = identity
        self.method.process_weights_after_loading(layer)

        self.assertEqual(layer.weight.data.shape, (256, 128))
        self.assertEqual(layer.weight_scale.data.shape, (128,))
        self.assertEqual(layer.weight_offset.data.shape, (128,))
        mock_npu_format_cast.assert_called_once()

    @patch.dict(os.environ, {"VLLM_ASCEND_ENABLE_NZ": "2"})
    @patch("torch_npu.npu_format_cast")
    def test_process_weights_after_loading_with_nz2(self, mock_npu_format_cast):
        layer = MagicMock()

        layer.weight.data = torch.randint(-128, 127, (128, 256), dtype=torch.int8)
        layer.weight_scale.data = torch.randn(128, 1)
        layer.weight_offset.data = torch.randn(128, 1)

        mock_npu_format_cast.side_effect = identity
        self.method.process_weights_after_loading(layer)

        self.assertEqual(layer.weight.data.shape, (256, 128))
        self.assertEqual(layer.weight_scale.data.shape, (128,))
        self.assertEqual(layer.weight_offset.data.shape, (128,))
        mock_npu_format_cast.assert_called_once()
