import os
from unittest.mock import MagicMock, patch

import torch

from tests.ut.base import TestBase
from tests.ut.conftest import npu_test
from tests.ut.quantization.conftest_quantization import create_linear_layer, identity
from vllm_ascend.quantization.methods.w8a8_static import AscendW8A8LinearMethod
from vllm_ascend.utils import COMPRESSED_TENSORS_METHOD


class TestAscendW8A8LinearMethod(TestBase):
    def setUp(self):
        self.method = AscendW8A8LinearMethod()

    def test_get_weight(self):
        sizes = [(64, 128), (256, 512), (1024, 2048), (1, 1)]
        for input_size, output_size in sizes:
            weight = self.method.get_weight(input_size, output_size)
            self.assertEqual(weight["weight"].dtype, torch.int8)
            self.assertEqual(weight["weight"].shape, (output_size, input_size))
            self.assertEqual(len(weight), 1)

        weight = self.method.get_weight(256, 128, torch.float16)
        self.assertEqual(weight["weight"].dtype, torch.int8)

    def test_get_pertensor_param(self):
        dtypes = [torch.bfloat16, torch.float16, torch.float32]
        for dtype in dtypes:
            params = self.method.get_pertensor_param(dtype)
            self.assertEqual(params["input_scale"].dtype, dtype)
            self.assertEqual(params["input_offset"].dtype, torch.int8)
            self.assertEqual(params["input_scale"].shape, (1,))
            self.assertEqual(params["input_offset"].shape, (1,))

    def test_get_perchannel_param(self):
        for output_size, dtype in [(128, torch.bfloat16), (256, torch.float16)]:
            params = self.method.get_perchannel_param(output_size, dtype)
            self.assertEqual(params["quant_bias"].shape, (output_size,))
            self.assertEqual(params["quant_bias"].dtype, torch.int32)
            self.assertEqual(params["weight_scale"].shape, (output_size, 1))
            self.assertEqual(params["weight_scale"].dtype, dtype)
            self.assertEqual(params["weight_offset"].shape, (output_size, 1))
            self.assertEqual(params["weight_offset"].dtype, dtype)
            self.assertEqual(params["deq_scale"].shape, (output_size,))
            if dtype == torch.bfloat16:
                self.assertEqual(params["deq_scale"].dtype, torch.float32)
            elif dtype == torch.float16:
                self.assertEqual(params["deq_scale"].dtype, torch.int64)

    @patch("vllm_ascend.quantization.methods.w8a8_static.get_weight_prefetch_method")
    @patch("torch.ops.vllm.quantize")
    @patch("torch_npu.npu_quant_matmul")
    def test_apply_with_x_not_int8(self, mock_npu_quant_matmul, mock_quantize, mock_get_weight_prefetch_method):
        layer = MagicMock()
        layer.aclnn_input_scale = 0.1
        layer.aclnn_input_offset = 0.2
        layer.weight = torch.randn(128, 256)
        layer.deq_scale = 0.3
        quant_bias = torch.zeros(256)
        layer.quant_bias = quant_bias

        mock_get_weight_prefetch_method.return_value = MagicMock()

        x = torch.randn(32, 128)
        bias = torch.randn(256)
        mock_quantize.return_value = torch.randint(-128, 127, x.shape, dtype=torch.int8)

        expected_y_output = torch.randn(32, 256)
        mock_npu_quant_matmul.return_value = expected_y_output

        output = self.method.apply(layer, x, bias)

        self.assertTrue(torch.equal(output, expected_y_output))
        mock_quantize.assert_called_once()
        mock_npu_quant_matmul.assert_called_once()
        call_kwargs = mock_npu_quant_matmul.call_args.kwargs
        self.assertTrue(torch.equal(call_kwargs["bias"], quant_bias))

    @patch("torch.ops.vllm.quantize")
    @patch("torch_npu.npu_quant_matmul")
    def test_apply_with_x_is_int8(self, mock_npu_quant_matmul, mock_quantize):
        layer = MagicMock()
        layer.aclnn_input_scale = 0.1
        layer.aclnn_input_offset = 0.2
        layer.weight = torch.randn(128, 256)
        layer.deq_scale = 0.3
        layer.ascend_quant_method = COMPRESSED_TENSORS_METHOD

        x = torch.randint(-128, 127, (32, 128), dtype=torch.int8)
        bias = torch.randn(256)

        expected_y_output = torch.randn(32, 256)
        mock_npu_quant_matmul.return_value = expected_y_output

        output = self.method.apply(layer, x, bias)
        self.assertTrue(torch.equal(output, expected_y_output))
        mock_quantize.assert_not_called()
        mock_npu_quant_matmul.assert_called_once()
        call_kwargs = mock_npu_quant_matmul.call_args.kwargs
        self.assertTrue(torch.equal(call_kwargs["bias"], bias))

    @patch.dict(os.environ, {"VLLM_ASCEND_ENABLE_NZ": "1"})
    @patch("torch_npu.npu_format_cast")
    def test_process_weights_after_loading_with_nz1(self, mock_npu_format_cast):
        layer = MagicMock()

        layer.weight.data = torch.randint(-128, 127, (128, 256), dtype=torch.int8)
        layer.input_scale.data = torch.tensor([0.1])
        layer.input_offset.data = torch.tensor([0])
        layer.weight_scale.data = torch.randn(128, 1)
        layer.weight_offset.data = torch.randn(128, 1)

        mock_npu_format_cast.side_effect = identity
        self.method.process_weights_after_loading(layer)

        expected_offset = torch.tensor([0]).repeat(256).to(torch.int8)
        self.assertTrue(torch.equal(layer.aclnn_input_offset.data, expected_offset))
        self.assertFalse(layer.aclnn_input_offset.requires_grad)

        self.assertEqual(layer.weight.data.shape, (256, 128))
        self.assertEqual(layer.weight_scale.data.shape, (128,))
        self.assertEqual(layer.weight_offset.data.shape, (128,))
        mock_npu_format_cast.assert_called_once()
        self.assertTrue(isinstance(layer.deq_scale, MagicMock))

    @patch.dict(os.environ, {"VLLM_ASCEND_ENABLE_NZ": "2"})
    @patch("torch_npu.npu_format_cast")
    def test_process_weights_after_loading_with_nz2_and_compressed_tensors(self, mock_npu_format_cast):
        layer = MagicMock()

        layer.weight.data = torch.randint(-128, 127, (128, 256), dtype=torch.int8)
        layer.input_scale.data = torch.tensor([0.1])
        layer.input_offset.data = torch.tensor([0])
        layer.weight_scale.data = torch.randn(128, 1)
        layer.weight_offset.data = torch.randn(128, 1)
        layer.ascend_quant_method = COMPRESSED_TENSORS_METHOD

        mock_npu_format_cast.side_effect = identity
        self.method.process_weights_after_loading(layer)

        expected_offset = torch.tensor([0]).repeat(256).to(torch.int8)
        self.assertTrue(torch.equal(layer.aclnn_input_offset.data, expected_offset))
        self.assertFalse(layer.aclnn_input_offset.requires_grad)

        self.assertEqual(layer.weight.data.shape, (256, 128))
        self.assertEqual(layer.weight_scale.data.shape, (128,))
        self.assertEqual(layer.weight_offset.data.shape, (128,))
        mock_npu_format_cast.assert_called_once()
        self.assertFalse(isinstance(layer.deq_scale, MagicMock))


@npu_test(num_npus=1, npu_type="a2")
class TestAscendW8A8LinearMethodWithNpu(TestBase):
    def setUp(self):
        self.method = AscendW8A8LinearMethod()

    @patch("vllm_ascend.quantization.methods.w8a8_static.get_weight_prefetch_method")
    def test_apply_with_npu(self, mock_get_weight_prefetch_method):
        mock_get_weight_prefetch_method.return_value = MagicMock()

        input_size, output_size = 128, 256
        params_dtype = torch.bfloat16
        layer = create_linear_layer(self.method, input_size, output_size, params_dtype)
        layer.params_dtype = params_dtype
        self.method.process_weights_after_loading(layer)

        x = torch.randn(32, input_size, dtype=params_dtype).npu()
        bias = torch.randn(output_size, dtype=torch.float32).npu()

        output = self.method.apply(layer, x, bias)
        self.assertEqual(output.shape, (32, output_size))
