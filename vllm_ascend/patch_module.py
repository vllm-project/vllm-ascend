#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
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
#
import sys
import types
from typing import Any, cast

import torch  # type: ignore[import-not-found]


def fused_topk(hidden_states, gating_output, topk, renormalize):
    assert hidden_states.shape[0] == gating_output.shape[0], "Number of tokens mismatch"

    scores = torch.softmax(gating_output, dim=-1)
    topk_weights, topk_ids = torch.topk(scores, k=topk, dim=-1, sorted=False)

    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

    return topk_weights.to(torch.float32), topk_ids.to(torch.int32)


def grouped_topk(
    hidden_states,
    gating_output,
    topk,
    renormalize,
    num_expert_group=0,
    topk_group=0,
    scoring_func="softmax",
    e_score_correction_bias=None,
):
    from vllm_ascend.ops.fused_moe import group_topk as ascend_group_topk

    return ascend_group_topk(
        hidden_states=hidden_states,
        gating_output=gating_output,
        topk=topk,
        renormalize=renormalize,
        num_expert_group=num_expert_group,
        topk_group=topk_group,
        scoring_func=scoring_func,
        e_score_correction_bias=e_score_correction_bias,
    )


def fused_experts(
    hidden_states, w1, w2, topk_weights, topk_ids, inplace=False, **kwargs
):
    from vllm_ascend.ops.fused_moe import fused_experts as ascend_fused_experts

    del inplace, kwargs
    return ascend_fused_experts(
        hidden_states=hidden_states,
        w1=w1,
        w2=w2,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        top_k=topk_ids.shape[1],
    )


# prevent errors caused by triton not supported
fused_moe_module = types.ModuleType("fused_moe_module")
fused_moe_module_any = cast(Any, fused_moe_module)
fused_moe_module_any.fused_topk = fused_topk
fused_moe_module_any.grouped_topk = grouped_topk
fused_moe_module_any.fused_experts = fused_experts

sys.modules["vllm.model_executor.layers.fused_moe.fused_moe"] = fused_moe_module
