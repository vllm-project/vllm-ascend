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
from vllm_ascend.ops.common_fused_moe import AscendFusedMoE


class TestLoadWeight(TestBase):

    def test_load_w13_transpose(self):
        with patch.object(AscendFusedMoE, "__init__",
                          lambda self, *args, **kwargs: None):
            moe = AscendFusedMoE(num_experts=4, top_k=2, hidden_size=8)

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
