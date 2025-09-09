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
# This file is a part of the vllm-ascend project.
#
from unittest.mock import patch

import torch

from tests.ut.base import TestBase
from vllm_ascend.ops.common_fused_moe import AscendFusedMoE, fused_experts_moge


class TestFusedExpertsMoGE(TestBase):

    def test_fused_experts_moge(self):
        with patch('torch_npu.npu_grouped_matmul') as mock_grouped_matmul, \
             patch('torch_npu.npu_swiglu') as mock_swiglu, \
             patch('vllm_ascend.utils.is_310p') as mock_is_310p:

            mock_is_310p.return_value = False

            mock_grouped_matmul.side_effect = lambda x, weight, **kwargs: [
                torch.randn(x[0].shape[0], weight[0].shape[1])
            ]

            mock_swiglu.side_effect = lambda x: x

            hidden_states = torch.randn(4, 128)
            w1 = torch.randn(4, 256, 128)
            w2 = torch.randn(4, 128, 128)
            topk_weights = torch.rand(4, 1)
            topk_ids = torch.tensor([[0], [1], [2], [3]], dtype=torch.long)
            top_k = 1
            global_num_experts = 4

            moe_parallel_config = type(
                'MockConfig', (), {
                    'ep_size': 1,
                    'tp_size': 1,
                    'dp_size': 1,
                    'tp_rank': 0,
                    'dp_rank': 0,
                    'ep_rank': 0,
                    'use_ep': True
                })()

            output = fused_experts_moge(
                hidden_states=hidden_states,
                w1=w1,
                w2=w2,
                moe_parallel_config=moe_parallel_config,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                top_k=top_k,
                global_num_experts=global_num_experts,
                apply_router_weight_on_input=True,
            )

            self.assertEqual(output.shape, (4, 128))


class TestLoadWeight(TestBase):

    def test_load_w13_transpose(self):
        with patch.object(AscendFusedMoE, "__init__",
                          lambda self, *args, **kwargs: None):
            moe = AscendFusedMoE(num_experts=4, top_k=2, hidden_size=8)
            moe.hidden_size = 8
            expert_data = torch.randn(128, 8)
            loaded_weight = torch.randn(128, 4)
            moe._load_w13(expert_data, 1, "w1", loaded_weight, 0)

            expert_data = torch.randn(8, 128)
            loaded_weight = torch.randn(128, 4)
            moe._load_w13(expert_data, 1, "w1", loaded_weight, 0)

            expert_data = torch.randn(128, 8)
            loaded_weight = torch.randn(128, 4)
            moe._load_w13(expert_data, 1, "w3", loaded_weight, 0)

            expert_data = torch.randn(8, 128)
            loaded_weight = torch.randn(128, 4)
            moe._load_w13(expert_data, 1, "w3", loaded_weight, 0)

    def test_load_w2_transpose(self):
        with patch.object(AscendFusedMoE, "__init__",
                          lambda self, *args, **kwargs: None):
            moe = AscendFusedMoE(num_experts=4, top_k=2, hidden_size=8)
            expert_data = torch.randn(128, 4)
            loaded_weight = torch.randn(128, 8)
            moe._load_w2(expert_data, 1, loaded_weight, 0)

            expert_data = torch.randn(4, 128)
            loaded_weight = torch.randn(128, 8)
            moe._load_w2(expert_data, 1, loaded_weight, 0)
