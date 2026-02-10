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

from unittest.mock import call, patch

import torch

from tests.ut.base import TestBase
from vllm_ascend._310p.fused_moe.moe_mlp import unified_apply_mlp


class TestUnifiedApplyMLP310(TestBase):
    @patch("torch_npu.npu_grouped_matmul")
    @patch("torch_npu.npu_swiglu")
    def test_unified_apply_mlp_without_quantization_310(self, mock_npu_swiglu, mock_npu_grouped_matmul):
        mock_gmm1_out = torch.randn(10, 40, dtype=torch.float16)
        mock_gmm2_out = torch.randn(10, 20, dtype=torch.float16)
        mock_npu_grouped_matmul.side_effect = [[mock_gmm1_out], [mock_gmm2_out]]

        mock_npu_swiglu_output = torch.randn(10, 40, dtype=torch.float16)
        mock_npu_swiglu.return_value = mock_npu_swiglu_output

        hidden_states = torch.randn(10, 20, dtype=torch.float16)
        w1 = torch.randn(5, 20, 40, dtype=torch.float16)
        w2 = torch.randn(5, 40, 20, dtype=torch.float16)
        group_list = torch.tensor([2, 4, 6, 8, 10], dtype=torch.int64)

        result = unified_apply_mlp(
            hidden_states=hidden_states,
            w1=w1,
            w1_scale=None,
            w2=w2,
            w2_scale=None,
            group_list=group_list,
            group_list_type=1,
            with_quant=False,
        )

        self.assertEqual(mock_npu_grouped_matmul.call_count, 2)
        mock_npu_grouped_matmul.assert_has_calls(
            [
                call(
                    x=[hidden_states], weight=[w1], split_item=2, group_list_type=1, group_type=0, group_list=group_list
                ),
                call(
                    x=[mock_npu_swiglu_output],
                    weight=[w2],
                    split_item=2,
                    group_list_type=1,
                    group_type=0,
                    group_list=group_list,
                ),
            ],
            any_order=True,
        )
        mock_npu_swiglu.assert_called_once()
        mock_npu_swiglu.assert_called_with(mock_gmm1_out)

        self.assertEqual(result.shape, hidden_states.shape)
        self.assertEqual(result.dtype, torch.float16)

    @patch("torch.cumsum")
    @patch("torch_npu.npu_quant_grouped_matmul_dequant")
    @patch("torch_npu.npu_swiglu")
    def test_unified_apply_mlp_with_quantization_310(
        self, mock_npu_swiglu, mock_npu_quant_grouped_matmul_dequant, mock_cumsum
    ):
        mock_cumsum_out = torch.arange(0, 10, dtype=torch.int64)
        mock_cumsum.return_value = mock_cumsum_out
        mock_gmm1_out = torch.randn(10, 40, dtype=torch.float16)
        mock_gmm2_out = torch.randn(10, 20, dtype=torch.float16)
        mock_npu_quant_grouped_matmul_dequant.side_effect = [mock_gmm1_out, mock_gmm2_out]

        mock_npu_swiglu_output = torch.randn(10, 40, dtype=torch.float16)
        mock_npu_swiglu.return_value = mock_npu_swiglu_output

        hidden_states = torch.randn(10, 20, dtype=torch.float16)
        w1 = torch.randn(5, 20, 40, dtype=torch.float16)
        w1_scale = torch.rand(5, 40, dtype=torch.float32)
        w2 = torch.randn(5, 40, 20, dtype=torch.float16)
        w2_scale = torch.rand(5, 40, dtype=torch.float32)
        group_list = torch.tensor([2, 4, 6, 8, 10], dtype=torch.int64)

        result = unified_apply_mlp(
            hidden_states=hidden_states,
            w1=w1,
            w1_scale=w1_scale,
            w2=w2,
            w2_scale=w2_scale,
            group_list=group_list,
            group_list_type=1,
            with_quant=True,
        )

        mock_cumsum.assert_called_once()
        self.assertEqual(mock_npu_quant_grouped_matmul_dequant.call_count, 2)
        mock_npu_quant_grouped_matmul_dequant.assert_has_calls(
            [
                call(
                    x=hidden_states,
                    quantized_weight=w1,
                    weight_scale=w1_scale,
                    group_list=mock_cumsum_out,
                    quant_mode="pertoken",
                ),
                call(
                    x=mock_npu_swiglu_output,
                    quantized_weight=w2,
                    weight_scale=w2_scale,
                    group_list=mock_cumsum_out,
                    quant_mode="pertoken",
                ),
            ],
            any_order=True,
        )
        mock_npu_swiglu.assert_called_once()
        mock_npu_swiglu.assert_called_with(mock_gmm1_out)

        self.assertEqual(result.shape, hidden_states.shape)
        self.assertEqual(result.dtype, torch.float16)
