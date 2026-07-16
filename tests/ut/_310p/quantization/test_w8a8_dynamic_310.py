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

from unittest.mock import MagicMock, Mock, patch

import torch

from tests.ut.base import TestBase
from vllm_ascend._310p.quantization.methods.w8a8_dynamic import (
    AscendW8A8DynamicFusedMoEMethod310,
    AscendW8A8DynamicLinearMethod310,
)


class TestAscendW8A8FusedMoEMethod310(TestBase):
    num_experts = 8
    hidden_size = 128
    intermediate_size = 128

    @patch("vllm_ascend._310p.quantization.methods.w8a8_dynamic.get_ep_group")
    def setUp(self, mock_get_ep_group):
        with patch(
            "vllm_ascend._310p.quantization.methods.w8a8_dynamic.get_current_vllm_config"
        ) as mock_get_current_vllm_config:
            mock_vllm_config = Mock()
            mock_vllm_config.quant_config = Mock(quant_description={"group_size": 0})
            mock_vllm_config.scheduler_config = Mock(
                max_num_batched_tokens=2048, max_model_len=2048, enable_chunked_prefill=False
            )
            mock_get_current_vllm_config.return_value = mock_vllm_config
            mock_ep_group = Mock()
            mock_get_ep_group.return_value = mock_ep_group
            mock_ascend_config = Mock()

            mock_ascend_config.enable_chunked_prefill = False

            self.quant_method = AscendW8A8DynamicFusedMoEMethod310()

    def test_get_weight_310(self):
        param_dict = self.quant_method.get_weight(
            self.num_experts, self.intermediate_size, self.hidden_size, torch.float16
        )
        self.assertEqual(param_dict["w13_weight"].dtype, torch.int8)
        self.assertEqual(
            param_dict["w13_weight"].shape, (self.num_experts, 2 * self.intermediate_size, self.hidden_size)
        )
        self.assertEqual(param_dict["w2_weight"].dtype, torch.int8)
        self.assertEqual(param_dict["w2_weight"].shape, (self.num_experts, self.hidden_size, self.intermediate_size))

    def test_get_dynamic_quant_param_310(self):
        param_dict = self.quant_method.get_dynamic_quant_param(
            self.num_experts, self.intermediate_size, self.hidden_size, torch.float16
        )
        self.assertEqual(param_dict["w13_weight_scale"].dtype, torch.float32)
        self.assertEqual(param_dict["w13_weight_scale"].shape, (self.num_experts, 2 * self.intermediate_size, 1))
        self.assertEqual(param_dict["w2_weight_scale"].dtype, torch.float32)
        self.assertEqual(param_dict["w2_weight_scale"].shape, (self.num_experts, self.hidden_size, 1))


class TestAscendW8A8DynamicLinearMethod310(TestBase):
    def setUp(self):
        self.method = AscendW8A8DynamicLinearMethod310()

    def test_get_weight_310(self):
        weight = self.method.get_weight(10, 20)
        self.assertEqual(weight["weight"].dtype, torch.int8)
        self.assertEqual(weight["weight"].shape, (20, 10))

    def test_get_perchannel_param_310(self):
        params = self.method.get_perchannel_param(10, torch.float32)

        self.assertEqual(params["weight_scale"].dtype, torch.float32)
        self.assertEqual(params["weight_offset"].dtype, torch.float32)

        self.assertEqual(params["weight_scale"].shape, (10, 1))
        self.assertEqual(params["weight_offset"].shape, (10, 1))

    @patch("torch_npu.npu_quant_matmul_dequant")
    @patch("torch_npu.npu_quant_matmul")
    @patch("torch_npu.npu_dynamic_quant")
    def test_apply_310_uses_dequant_path(
        self,
        mock_npu_dynamic_quant,
        mock_npu_quant_matmul,
        mock_npu_quant_matmul_dequant,
    ):
        layer = MagicMock()
        layer.weight = torch.randint(-127, 128, (256, 128), dtype=torch.int8)
        layer.weight_scale = torch.randn(256, dtype=torch.float32)
        x = torch.randn(32, 128, dtype=torch.float16)
        expected_output = torch.randn(32, 256)
        mock_npu_quant_matmul_dequant.return_value = expected_output

        output = self.method.apply(layer, x, tp_rank=0)

        mock_npu_dynamic_quant.assert_not_called()
        mock_npu_quant_matmul.assert_not_called()
        mock_npu_quant_matmul_dequant.assert_called_once()
        args, kwargs = mock_npu_quant_matmul_dequant.call_args
        self.assertTrue(torch.equal(args[0], x))
        self.assertTrue(torch.equal(args[1], layer.weight.data))
        self.assertTrue(torch.equal(args[2], layer.weight_scale))
        self.assertIsNone(kwargs["bias"])
        self.assertEqual(kwargs["quant_mode"], "pertoken")
        self.assertTrue(torch.equal(output, expected_output))

    @patch("torch_npu.npu_quant_matmul_dequant")
    @patch("torch_npu.npu_quant_matmul")
    @patch("torch_npu.npu_dynamic_quant")
    def test_apply_310_passes_bias_to_dequant(
        self,
        mock_npu_dynamic_quant,
        mock_npu_quant_matmul,
        mock_npu_quant_matmul_dequant,
    ):
        layer = MagicMock()
        layer.weight = torch.randint(-127, 128, (256, 128), dtype=torch.int8)
        layer.weight_scale = torch.randn(256, dtype=torch.float32)
        x = torch.randn(32, 128, dtype=torch.float16)
        bias = torch.randn(256, dtype=torch.float16)
        expected_output = torch.randn(32, 256)
        mock_npu_quant_matmul_dequant.return_value = expected_output

        output = self.method.apply(layer, x, bias=bias, tp_rank=0)

        mock_npu_dynamic_quant.assert_not_called()
        mock_npu_quant_matmul.assert_not_called()
        mock_npu_quant_matmul_dequant.assert_called_once()
        _args, kwargs = mock_npu_quant_matmul_dequant.call_args
        self.assertTrue(torch.equal(kwargs["bias"], bias))
        self.assertEqual(kwargs["quant_mode"], "pertoken")
        self.assertTrue(torch.equal(output, expected_output))

    @patch("torch_npu.npu_quant_matmul_dequant")
    @patch("torch_npu.npu_quant_matmul")
    @patch("torch_npu.npu_dynamic_quant")
    def test_apply_310_flattens_non_2d_input(
        self,
        mock_npu_dynamic_quant,
        mock_npu_quant_matmul,
        mock_npu_quant_matmul_dequant,
    ):
        layer = MagicMock()
        layer.weight = torch.randint(-127, 128, (256, 128), dtype=torch.int8)
        layer.weight_scale = torch.randn(256, dtype=torch.float32)
        x = torch.randn(2, 16, 128, dtype=torch.float16)
        expected_output = torch.randn(32, 256)
        mock_npu_quant_matmul_dequant.return_value = expected_output

        output = self.method.apply(layer, x, tp_rank=0)

        mock_npu_dynamic_quant.assert_not_called()
        mock_npu_quant_matmul.assert_not_called()
        mock_npu_quant_matmul_dequant.assert_called_once()
        args, kwargs = mock_npu_quant_matmul_dequant.call_args
        self.assertEqual(args[0].shape, (32, 128))
        self.assertEqual(kwargs["quant_mode"], "pertoken")
        self.assertEqual(output.shape, (2, 16, 256))

    def test_process_weights_after_loading_keeps_weight_contiguous_310p(self):
        layer = MagicMock()
        layer.weight = MagicMock()
        layer.weight_scale = MagicMock()
        layer.weight_offset = MagicMock()

        layer.weight.data = torch.randint(-127, 128, (256, 128), dtype=torch.int8)
        original_weight = layer.weight.data.clone()
        layer.weight_scale.data = torch.randn(256, 1, dtype=torch.bfloat16)
        layer.weight_offset.data = torch.randn(256, 1, dtype=torch.bfloat16)

        self.method.process_weights_after_loading(layer)

        self.assertTrue(layer.weight.data.is_contiguous())
        self.assertTrue(torch.equal(layer.weight.data, original_weight))
        self.assertEqual(layer.weight_scale.data.shape, (256,))
        self.assertEqual(layer.weight_offset.data.shape, (256,))
