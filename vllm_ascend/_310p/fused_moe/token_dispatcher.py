# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024; NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
# Copyright 2023 DeepSeek-AI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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

from typing import Optional

import torch
import torch_npu
from vllm.distributed.parallel_state import get_ep_group


from vllm_ascend.ops.fused_moe.token_dispatcher import TokenDispatcherWithAllGather, TokenDispatchResult


class TokenDispatcherWithAllGather310(TokenDispatcherWithAllGather):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def token_dispatch(self,
                       hidden_states: torch.Tensor,
                       topk_weights: torch.Tensor,
                       topk_ids: torch.Tensor,
                       expert_map: Optional[torch.Tensor] = None,
                       global_redundant_expert_num: int = 0,
                       mc2_mask: Optional[torch.Tensor] = None,
                       apply_router_weight_on_input: bool = False,
                       with_quant: bool = False,
                       dynamic_eplb: bool = False,
                       pertoken_scale: Optional[torch.Tensor] = None):
        if with_quant:
            raise RuntimeError("Quant is not supported for 310P currently.")
        self.original_shape = hidden_states.shape

        num_tokens = hidden_states.shape[:-1].numel()
        self.apply_router_weight_on_input = apply_router_weight_on_input
        if self.apply_router_weight_on_input:
            assert (topk_weights.dim() == 2
                    ), "`topk_weights` should be in shape (num_tokens, topk)"
            _, topk = topk_weights.shape
            assert (
                topk == 1
            ), "Only support topk=1 when `apply_router_weight_on_input` is True"
            hidden_states = hidden_states * \
                topk_weights.to(hidden_states.dtype)
        if expert_map is not None:
            global_num_experts = len(expert_map) + global_redundant_expert_num
            mask = (expert_map[topk_ids] != -1)
            topk_weights = topk_weights * mask
            first_expert_idx = get_ep_group(
            ).rank_in_group * self.num_experts_local
            last_expert_idx = first_expert_idx + self.num_experts_local
        else:
            first_expert_idx = 0
            last_expert_idx = self.num_experts_local
            global_num_experts = self.num_experts_local

        sorted_hidden_states, expanded_row_idx, expert_tokens = (
            self.moe_init_routing(
                hidden_states,
                topk_ids,
                active_num=num_tokens * self.top_k,
                active_expert_range=[first_expert_idx, last_expert_idx]
            ))
        expert_tokens = expert_tokens.to(torch.int64)
        group_list_type = 1  # `count` mode
        context_metadata = {
            "topk_weights": topk_weights,
            "expanded_row_idx": expanded_row_idx
        }

        return TokenDispatchResult(
            hidden_states=sorted_hidden_states,
            dynamic_scale=None,
            group_list=expert_tokens,
            group_list_type=group_list_type,
            context_metadata=context_metadata,
        )

    def moe_init_routing(x, expert_idx, active_num, active_expert_range):
        # 常量定义（避免重复计算）
        MAX_INT32 = torch.iinfo(torch.int32).max
        expert_start, expert_end = active_expert_range
        
        # 输入转换（PyTorch 张量化）
        x = torch.tensor(x, dtype=torch.float32)  # 确保浮点运算
        expert_idx = torch.tensor(expert_idx, dtype=torch.int32)
        
        # 1. 预处理 expert_idx
        num_rows= x.shape[0]
        k = expert_idx.shape[-1]
        
        # 展平专家索引并处理边界
        expert_idx_flat = expert_idx.flatten()
        mask = (expert_idx_flat >= expert_start) & (expert_idx_flat < expert_end)
        actual_expert_total_num = mask.sum().item()
        
        # 用 MAX_INT32 替换无效索引（稳定排序关键）
        expert_idx_flat = torch.where(~mask, 
                                    torch.full_like(expert_idx_flat, MAX_INT32, dtype=torch.int32),
                                    expert_idx_flat)
        
        # 2. 稳定排序（关键优化点）
        sorted_idx = torch.argsort(expert_idx_flat, stable=True)
        sorted_expert_idx = expert_idx_flat[sorted_idx]
        
        # 3. 构建 expanded_row_idx
        expanded_row_idx = torch.full((num_rows * k,), -1, dtype=torch.int32)
        expanded_row_idx[sorted_idx[:actual_expert_total_num]] = torch.arange(actual_expert_total_num, dtype=torch.int32)
        
        # 4. 处理 expert_tokens_count
        counts = torch.bincount(
            sorted_expert_idx[:actual_expert_total_num] - expert_start,
            minlength=expert_end - expert_start
        )
        expert_tokens_count = counts
        
        # 5. 处理 active_num
        active_num = min(active_num or actual_expert_total_num, actual_expert_total_num)
        
        # 6. 构建 expanded_x
        expanded_x = x[sorted_idx[:active_num] // k]
        
        # 7. 转换为 NumPy 格式（与原函数输出一致）
        return (
            expanded_x,
            expanded_row_idx,
            expert_tokens_count
        )