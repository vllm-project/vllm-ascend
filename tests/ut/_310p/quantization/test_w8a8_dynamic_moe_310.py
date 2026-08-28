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

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from vllm_ascend._310p.quantization.methods import w8a8_dynamic as w8a8_dynamic_310_module
from vllm_ascend._310p.quantization.methods.w8a8_dynamic import AscendW8A8DynamicFusedMoEMethod310


def test_w8a8_dynamic_moe_apply_310_uses_preselected_experts():
    method = AscendW8A8DynamicFusedMoEMethod310.__new__(AscendW8A8DynamicFusedMoEMethod310)
    method.in_dtype = torch.float16
    expert_map = torch.tensor([0, 1], dtype=torch.int32)
    layer = SimpleNamespace(
        w13_weight=torch.randint(-8, 8, (2, 4, 6), dtype=torch.int8),
        w2_weight=torch.randint(-8, 8, (2, 6, 4), dtype=torch.int8),
        w13_weight_scale=torch.ones(2, 4),
        w2_weight_scale=torch.ones(2, 6),
        ascend_expert_map=expert_map,
        global_redundant_expert_num=0,
        ascend_mc2_mask=None,
        apply_router_weight_on_input=True,
        log2phy=None,
        ascend_pertoken_scale=None,
        activation="silu",
        swiglu_limit=0.0,
        swiglu_alpha=1.0,
        swiglu_beta=0.0,
    )
    hidden_states = torch.randn(3, 6, dtype=torch.float16)
    topk_weights = torch.rand(3, 2, dtype=torch.float32)
    topk_ids = torch.tensor([[0, 1], [1, 0], [0, 1]], dtype=torch.int32)
    expected_output = object()
    comm_method = MagicMock()
    comm_method.fused_experts.return_value = expected_output

    with patch.object(
        w8a8_dynamic_310_module,
        "_EXTRA_CTX",
        SimpleNamespace(moe_comm_method=comm_method),
    ):
        output = method.apply(
            layer=layer,
            x=hidden_states,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            shared_experts=None,
            shared_experts_input=None,
        )

    assert output is expected_output
    fused_experts_input = comm_method.fused_experts.call_args.kwargs["fused_experts_input"]
    torch.testing.assert_close(fused_experts_input.topk_weights, topk_weights.to(torch.float16))
    assert fused_experts_input.topk_ids is topk_ids
    assert fused_experts_input.routing.expert_map is expert_map
    assert fused_experts_input.routing.apply_router_weight_on_input is True
