import os
from unittest.mock import MagicMock, patch

import torch

from tests.ut.base import TestBase
from tests.ut.conftest import npu_test
from tests.ut.quantization.conftest_quantization import create_linear_layer, identity
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


@npu_test(num_npus=1, npu_type="a2")
class TestAscendW8A16LinearMethodWithNpu(TestBase):
    def setUp(self):
        self.method = AscendW8A16LinearMethod()

    def test_apply_with_npu(self):
        input_size, output_size = 128, 256
        params_dtype = torch.bfloat16
        layer = create_linear_layer(self.method, input_size, output_size, params_dtype)
        self.method.process_weights_after_loading(layer)

        x = torch.randn(32, input_size, dtype=params_dtype).npu()
        bias = torch.randn(output_size, dtype=torch.float32).npu()

        output = self.method.apply(layer, x, bias)
        self.assertEqual(output.shape, (32, output_size))
