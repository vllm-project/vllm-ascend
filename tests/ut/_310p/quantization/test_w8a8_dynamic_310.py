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
    _MIN_NZ_QUANT_MATMUL_N,
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

    def test_apply_310(self):
        layer = MagicMock()
        layer.weight = torch.randint(-8, 8, (256, 128), dtype=torch.int8)
        layer.weight_scale = torch.ones(256, dtype=torch.float32)
        x = torch.randn(32, 128, dtype=torch.float16)

        output = self.method.apply(layer, x, tp_rank=0)

        self.assertEqual(output.shape, (32, 256))
        self.assertEqual(output.dtype, torch.float16)

    def test_apply_fp16_fallback_skips_quant_matmul_310(self):
        layer = MagicMock()
        layer.weight = torch.randint(-8, 8, (256, 2048), dtype=torch.int8)
        layer.weight_scale = torch.ones(256, dtype=torch.float32)
        x = torch.randn(4, 2048, dtype=torch.float16)

        output = self.method.apply(layer, x, tp_rank=0)

        self.assertEqual(output.shape, (4, 256))
        self.assertEqual(output.dtype, torch.float16)

    @patch("vllm_ascend.utils.is_310p", return_value=True)
    @patch("torch_npu.npu_format_cast")
    def test_process_weights_keeps_nd_for_small_n_310p(self, mock_npu_format_cast, _mock_is_310p):
        mock_npu_format_cast.side_effect = lambda x, fmt: x
        layer = MagicMock()
        layer.weight = MagicMock()
        layer.weight_scale = MagicMock()
        layer.weight_offset = MagicMock()
        layer.weight.data = torch.randint(-127, 128, (256, 2048), dtype=torch.int8)
        layer.weight_scale.data = torch.randn(256, 1, dtype=torch.float32)
        layer.weight_offset.data = torch.randn(256, 1, dtype=torch.float32)

        self.method.process_weights_after_loading(layer)

        mock_npu_format_cast.assert_not_called()
        self.assertEqual(layer.weight.data.shape, (256, 2048))
        self.assertTrue(layer._310p_w8a8_dynamic_fp16_fallback)
        self.assertEqual(layer.weight_scale.data.shape, (256,))
        self.assertEqual(layer.weight_offset.data.shape, (256,))

    @patch("vllm_ascend.utils.is_310p", return_value=True)
    @patch("torch_npu.npu_format_cast")
    def test_process_weights_keeps_nd_for_fused_qkv_small_kv_310p(self, mock_npu_format_cast, _mock_is_310p):
        mock_npu_format_cast.side_effect = lambda x, fmt: x
        layer = MagicMock()
        layer.num_kv_heads = 1
        layer.head_size = 256
        layer.weight = MagicMock()
        layer.weight_scale = MagicMock()
        layer.weight_offset = MagicMock()
        layer.weight.data = torch.randint(-127, 128, (2560, 2048), dtype=torch.int8)
        layer.weight_scale.data = torch.randn(2560, 1, dtype=torch.float32)
        layer.weight_offset.data = torch.randn(2560, 1, dtype=torch.float32)

        self.method.process_weights_after_loading(layer)

        mock_npu_format_cast.assert_not_called()
        self.assertEqual(layer.weight.data.shape, (2560, 2048))
        self.assertTrue(layer._310p_w8a8_dynamic_fp16_fallback)

    @patch("vllm_ascend.utils.is_310p", return_value=True)
    @patch("torch_npu.npu_format_cast")
    def test_process_weights_after_loading_keeps_nd_even_for_large_n_310p(self, mock_npu_format_cast, _mock_is_310p):
        mock_npu_format_cast.side_effect = lambda x, fmt: x

        layer = MagicMock()
        layer.weight = MagicMock()
        layer.weight_scale = MagicMock()
        layer.weight_offset = MagicMock()

        n = _MIN_NZ_QUANT_MATMUL_N
        layer.weight.data = torch.randint(-127, 128, (n, 256), dtype=torch.int8)
        layer.weight_scale.data = torch.randn(n, 1, dtype=torch.bfloat16)
        layer.weight_offset.data = torch.randn(n, 1, dtype=torch.bfloat16)

        self.method.process_weights_after_loading(layer)

        mock_npu_format_cast.assert_not_called()
        self.assertEqual(layer.weight.data.shape, (n, 256))
        self.assertTrue(layer._310p_w8a8_dynamic_fp16_fallback)
        self.assertEqual(layer.weight_scale.data.shape, (n,))
        self.assertEqual(layer.weight_offset.data.shape, (n,))

    @patch("vllm_ascend.utils.is_310p", return_value=True)
    @patch("torch_npu.npu_format_cast")
    def test_moe_process_weights_transposes_output_dimension_before_nz(self, mock_npu_format_cast, _mock_is_310p):
        mock_npu_format_cast.side_effect = lambda x, fmt: x
        method = AscendW8A8DynamicFusedMoEMethod310.__new__(AscendW8A8DynamicFusedMoEMethod310)
        layer = MagicMock()
        layer.w13_weight = MagicMock()
        layer.w2_weight = MagicMock()
        layer.w13_weight_scale = MagicMock()
        layer.w13_weight_offset = MagicMock()
        layer.w2_weight_scale = MagicMock()
        layer.w2_weight_offset = MagicMock()

        num_experts, intermediate_size, hidden_size = 8, 96, 256
        layer.w13_weight.data = torch.randint(
            -127,
            128,
            (num_experts, 2 * intermediate_size, hidden_size),
            dtype=torch.int8,
        )
        layer.w2_weight.data = torch.randint(
            -127,
            128,
            (num_experts, hidden_size, intermediate_size),
            dtype=torch.int8,
        )
        layer.w13_weight_scale.data = torch.randn(num_experts, 2 * intermediate_size, 1)
        layer.w13_weight_offset.data = torch.zeros(num_experts, 2 * intermediate_size, 1)
        layer.w2_weight_scale.data = torch.randn(num_experts, hidden_size, 1)
        layer.w2_weight_offset.data = torch.zeros(num_experts, hidden_size, 1)

        method.process_weights_after_loading(layer)

        self.assertEqual(
            layer.w13_weight.data.shape,
            (num_experts, hidden_size, 2 * intermediate_size),
        )
        self.assertEqual(
            layer.w2_weight.data.shape,
            (num_experts, intermediate_size, hidden_size),
        )
        self.assertEqual(layer.w13_weight_scale.data.shape, (num_experts, 2 * intermediate_size))
        self.assertEqual(layer.w2_weight_scale.data.shape, (num_experts, hidden_size))
        self.assertEqual(mock_npu_format_cast.call_count, 2)
