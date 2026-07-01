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

import math
from unittest.mock import MagicMock, patch

import pytest
import torch

from tests.ut.base import TestBase
from vllm_ascend._310p.quantization.methods.w8a8sc import AscendW8A8SCLinearMethod310


class TestAscendW8A8SCLinearMethod310(TestBase):

    def setUp(self):
        self.method = AscendW8A8SCLinearMethod310()

    def test_get_weight_310(self):
        weight = self.method.get_weight(10, 20)
        self.assertEqual(weight["weight"].dtype, torch.int8)
        self.assertEqual(weight["weight"].shape, (10 * 20, ))
        self.assertEqual(weight["index"].dtype, torch.int8)
        index_len = math.ceil(10 / 256) * math.ceil(20 / 128) * 8
        self.assertEqual(weight["index"].shape, (index_len, ))
        self.assertEqual(weight["info"].dtype, torch.int64)
        self.assertEqual(weight["info"].shape, (5, ))

    def test_get_pertensor_param_310(self):
        params = self.method.get_pertensor_param(torch.float16)
        self.assertEqual(params["input_scale"].dtype, torch.float16)
        self.assertEqual(params["input_offset"].dtype, torch.int8)
        self.assertEqual(params["input_scale"].shape, (1, ))
        self.assertEqual(params["input_offset"].shape, (1, ))

    def test_get_perchannel_param_310(self):
        params = self.method.get_perchannel_param(10, torch.float16)

        self.assertEqual(params["quant_bias"].dtype, torch.int32)
        self.assertEqual(params["deq_scale"].dtype, torch.int64)
        self.assertEqual(params["quant_bias"].shape, (10, ))
        self.assertEqual(params["deq_scale"].shape, (10, ))

    @pytest.mark.skip(
        "Skip as npu_matmul_compress_dequant will be supported in PTA 26.0.0.")
    @patch("torch.ops.vllm.quantize")
    @patch("torch_npu.npu_matmul_compress_dequant")
    def test_apply_with_x_not_int8_310(self, mock_matmul_compress_dequant,
                                       mock_quantize):
        layer = MagicMock()
        layer.aclnn_input_scale = torch.randn(256)
        layer.aclnn_input_scale_reciprocal = 1.0 / layer.aclnn_input_scale
        layer.aclnn_input_offset = torch.randint(-128,
                                                 127, (256, ),
                                                 dtype=torch.int8)
        layer.weight = torch.randint(-128,
                                     127, (256 * 128, ),
                                     dtype=torch.int8)
        layer.index = torch.randint(-128, 127, (8, ), dtype=torch.int8)
        layer.deq_scale = torch.randn(128)
        layer.quant_bias = torch.randint(-128, 127, (256, ))
        layer.params_dtype = torch.float16

        x = torch.randn(32, 128)
        expect_x_output = torch.randint(-128, 127, x.shape, dtype=torch.int8)
        mock_quantize.return_value = expect_x_output

        expected_y_output = torch.randn(32, 256)
        mock_matmul_compress_dequant.return_value = expected_y_output

        output = self.method.apply(layer, x, tp_rank=0)

        mock_quantize.assert_called_with(x, layer.aclnn_input_scale,
                                         layer.aclnn_input_scale_reciprocal,
                                         layer.aclnn_input_offset)
        mock_matmul_compress_dequant.assert_called_with(
            expect_x_output, layer.weight, layer.index, layer.quant_bias,
            layer.deq_scale)
        self.assertTrue(torch.equal(output, expected_y_output))

    @pytest.mark.skip(
        "Skip as npu_matmul_compress_dequant will be supported in PTA 26.0.0.")
    @patch("torch.ops.vllm.quantize")
    @patch("torch_npu.npu_matmul_compress_dequant")
    def test_apply_with_x_is_int8_310(self, mock_matmul_compress_dequant,
                                      mock_quantize):
        layer = MagicMock()
        layer.aclnn_input_scale = torch.randn(256)
        layer.aclnn_input_offset = torch.randint(-128,
                                                 127, (256, ),
                                                 dtype=torch.int8)
        layer.weight = torch.randint(-128,
                                     127, (256 * 128, ),
                                     dtype=torch.int8)
        layer.index = torch.randint(-128, 127, (8, ), dtype=torch.int8)
        layer.deq_scale = torch.randn(128)
        layer.quant_bias = torch.randint(-128, 127, (256, ))
        layer.params_dtype = torch.float16

        x = torch.randint(-128, 127, (32, 128), dtype=torch.int8)

        expected_y_output = torch.randn(32, 256)
        mock_matmul_compress_dequant.return_value = expected_y_output

        output = self.method.apply(layer, x, tp_rank=0)

        mock_quantize.assert_not_called()
        mock_matmul_compress_dequant.assert_called_with(
            x, layer.weight, layer.index, layer.quant_bias, layer.deq_scale)
        self.assertTrue(torch.equal(output, expected_y_output))
