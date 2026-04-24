from unittest.mock import MagicMock, patch

import torch
import torch.nn as nn

from tests.ut.base import TestBase
from vllm_ascend.quantization.methods.w4a4_laos_dynamic import AscendW4A4LaosDynamicLinearMethod


class TestAscendW4A4LaosDynamicLinearMethod(TestBase):
    def setUp(self):
        self.method = AscendW4A4LaosDynamicLinearMethod()

    def test_get_weight_various_sizes(self):
        sizes = [(64, 128), (256, 512), (1024, 2048)]
        for input_size, output_size in sizes:
            result = self.method.get_weight(input_size, output_size, torch.bfloat16)
            self.assertEqual(result["weight"].shape, (output_size, input_size))
            self.assertEqual(result["weight"].dtype, torch.int8)

    def test_get_perchannel_param_various_output_sizes(self):
        output_sizes = [1, 64, 128, 512]
        for output_size in output_sizes:
            result = self.method.get_perchannel_param(output_size, torch.bfloat16)
            self.assertEqual(result["weight_scale"].shape, (output_size, 1))
            self.assertEqual(result["weight_offset"].shape, (output_size, 1))
            self.assertEqual(result["weight_scale"].dtype, torch.float32)
            self.assertEqual(result["weight_offset"].dtype, torch.float32)

    @patch("torch_npu.npu_quant_matmul")
    @patch("torch_npu.npu_dynamic_quant")
    def test_apply_with_bias(self, mock_dyn_quant, mock_matmul):
        mock_dyn_quant.return_value = (
            torch.randint(0, 15, (32, 128), dtype=torch.int32),
            torch.randn(32, dtype=torch.float32),
        )
        expected_output = torch.randn(32, 256, dtype=torch.bfloat16)
        mock_matmul.return_value = expected_output
        layer = MagicMock()
        layer.weight = MagicMock(data=torch.randint(-8, 7, (256, 128), dtype=torch.int8))
        layer.weight_scale = MagicMock(data=torch.randn(256, dtype=torch.float32))
        x = torch.randn(32, 128, dtype=torch.bfloat16)
        bias = torch.randn(256, dtype=torch.bfloat16)
        output = self.method.apply(layer, x, bias)
        expected_output = expected_output + bias
        self.assertTrue(torch.equal(output, expected_output))

    @patch("torch_npu.npu_convert_weight_to_int4pack")
    def test_process_weights_various_input_sizes(self, mock_convert):
        for input_size, output_size in [(64, 128), (256, 512)]:
            mock_convert.return_value = torch.randint(0, 15, (output_size, input_size // 8), dtype=torch.int32)
            layer = nn.Module()
            layer.weight = nn.Parameter(
                torch.randint(-8, 7, (output_size, input_size), dtype=torch.int8), requires_grad=False
            )
            layer.weight_scale = nn.Parameter(torch.randn(output_size, 1, dtype=torch.float32), requires_grad=False)
            self.method.process_weights_after_loading(layer)
            mock_convert.assert_called()
            self.assertEqual(layer.weight_scale.data.dtype, torch.float32)
            self.assertEqual(layer.weight.shape, (input_size // 8, output_size))
