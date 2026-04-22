#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
        params = self.method.get_pertensor_param(torch.float16)
        self.assertEqual(params["input_scale"].dtype, torch.float16)
        self.assertEqual(params["input_offset"].dtype, torch.int8)
        self.assertEqual(params["input_scale"].shape, (1,))
        self.assertEqual(params["input_offset"].shape, (1,))

    def test_get_perchannel_param_310(self):
        params = self.method.get_perchannel_param(10, torch.float16)

        self.assertEqual(params["quant_bias"].dtype, torch.int32)
        self.assertEqual(params["deq_scale"].dtype, torch.int64)
        self.assertEqual(params["weight_scale"].dtype, torch.float16)
        self.assertEqual(params["weight_offset"].dtype, torch.float16)

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
            x,
            layer.aclnn_input_scale,
            layer.aclnn_input_scale_reciprocal,
            layer.aclnn_input_offset,
        )
        mock_npu_quant_matmul.assert_called_once()
        (args, kwargs) = mock_npu_quant_matmul.call_args

        # positional args
        self.assertTrue(torch.equal(args[0], expect_x_output))
        self.assertTrue(torch.equal(args[1], layer.weight.data))
        self.assertTrue(torch.equal(args[2], layer.deq_scale))

        # kwargs
        self.assertTrue(torch.equal(kwargs["bias"], layer.quant_bias))
        self.assertEqual(kwargs["output_dtype"], layer.params_dtype)

        self.assertTrue(torch.equal(output, expected_y_output))

    @patch("torch.ops.vllm.quantize")
    @patch("torch_npu.npu_quant_matmul")
    def test_apply_with_x_is_int8_310(self, mock_npu_quant_matmul, mock_quantize):
        layer = MagicMock()
        layer.aclnn_input_scale = torch.randn(256)
        layer.aclnn_input_offset = torch.randint(-128, 127, (256,), dtype=torch.int8)
        layer.weight = torch.randn(128, 256)
        layer.deq_scale = torch.randn(128)
        layer.quant_bias = torch.randint(-128, 127, (256,))
        layer.params_dtype = torch.float16

        x = torch.randint(-128, 127, (32, 128), dtype=torch.int8)

        expected_y_output = torch.randn(32, 256)
        mock_npu_quant_matmul.return_value = expected_y_output

        output = self.method.apply(layer, x, tp_rank=0)

        mock_quantize.assert_not_called()
        mock_npu_quant_matmul.assert_called_once()
        (args, kwargs) = mock_npu_quant_matmul.call_args

        self.assertTrue(torch.equal(args[0], x))
        self.assertTrue(torch.equal(args[1], layer.weight.data))
        self.assertTrue(torch.equal(args[2], layer.deq_scale))

        self.assertTrue(torch.equal(kwargs["bias"], layer.quant_bias))
        self.assertEqual(kwargs["output_dtype"], layer.params_dtype)

        self.assertTrue(torch.equal(output, expected_y_output))

    @patch("torch_npu.npu_format_cast")
    def test_process_weights_after_loading_calls_nz_format_cast_310p(self, mock_npu_format_cast):
        mock_npu_format_cast.side_effect = lambda x, fmt: x

        layer = MagicMock()

        # Attributes used by process_weights_after_loading()
        layer.weight = MagicMock()
        layer.input_scale = MagicMock()
        layer.input_offset = MagicMock()
        layer.weight_scale = MagicMock()
        layer.weight_offset = MagicMock()
        layer.w2_weight_offset = MagicMock()

        layer.weight.data = torch.randint(-127, 128, (128, 256), dtype=torch.int8)
        layer.input_scale.data = torch.tensor([0.1], dtype=torch.float16)
        layer.input_offset.data = torch.tensor([0], dtype=torch.int8)

        layer.weight_scale.data = torch.randn(128, 1, dtype=torch.bfloat16)
        layer.weight_offset.data = torch.randn(128, 1, dtype=torch.bfloat16)
        # w2_weight_offset is reshaped to (N, -1); any (N, 1) is fine
        layer.w2_weight_offset.data = torch.randn(128, 1, dtype=torch.bfloat16)

        self.method.process_weights_after_loading(layer)

        mock_npu_format_cast.assert_called_once()
