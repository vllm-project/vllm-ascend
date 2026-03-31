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

import pytest
import torch

from vllm_ascend._310p.fused_moe.experts_selector import select_experts


class TestExpertsSelector310:
    @pytest.mark.parametrize("global_num_experts", [256, 128])
    def test_select_experts(self, global_num_experts):
        x = torch.randn(8, 2)
        router_logits = torch.randn(8, 2)
        topk_weights, topk_ids = select_experts(
            hidden_states=x,
            router_logits=router_logits,
            top_k=2,
            use_grouped_topk=False,
            renormalize=True,
            topk_group=None,
            num_expert_group=None,
            custom_routing_function=None,
            scoring_func="softmax",
            e_score_correction_bias=None,
            global_num_experts=global_num_experts,
        )

        assert topk_weights.shape == (8, 2)
        assert topk_ids.shape == (8, 2)
