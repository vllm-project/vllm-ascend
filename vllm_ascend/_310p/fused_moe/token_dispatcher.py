# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024; NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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


import torch
import torch_npu
from vllm.distributed.parallel_state import get_ep_group

from vllm_ascend.ops.fused_moe.moe_runtime_args import MoEAllGatherCombineMetadata, MoETokenDispatchInput
from vllm_ascend.ops.fused_moe.token_dispatcher import MoETokenDispatchOutput, TokenDispatcherWithAllGather


class TokenDispatcherWithAllGather310(TokenDispatcherWithAllGather):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def token_dispatch(
        self,
        token_dispatch_input: MoETokenDispatchInput,
    ):
        hidden_states = token_dispatch_input.hidden_states
        topk_weights = token_dispatch_input.topk_weights
        topk_ids = token_dispatch_input.topk_ids
        expert_map = token_dispatch_input.routing.expert_map
        apply_router_weight_on_input = token_dispatch_input.routing.apply_router_weight_on_input
        restore_shape = hidden_states.shape

        num_tokens = hidden_states.shape[:-1].numel()
        if apply_router_weight_on_input:
            assert topk_weights.dim() == 2, "`topk_weights` should be in shape (num_tokens, topk)"
            _, topk = topk_weights.shape
            assert topk == 1, "Only support topk=1 when `apply_router_weight_on_input` is True"
            hidden_states = hidden_states * topk_weights.to(hidden_states.dtype)
        if expert_map is not None:
            mask = expert_map[topk_ids] != -1
            topk_weights = topk_weights * mask
            first_expert_idx = get_ep_group().rank_in_group * self.num_experts_local
            last_expert_idx = first_expert_idx + self.num_experts_local
        else:
            first_expert_idx = 0
            last_expert_idx = self.num_experts_local

        assert hidden_states.shape[-1] % 16 == 0, (
            f"The last dim of hidden_states {hidden_states.shape[-1]} should be aligned with 16."
        )
        sorted_hidden_states, expanded_row_idx, expert_tokens, _ = torch_npu.npu_moe_init_routing_v2(
            hidden_states,
            topk_ids,
            active_num=num_tokens * self.top_k,
            expert_num=self.num_experts_local,
            drop_pad_mode=0,
            active_expert_range=[first_expert_idx, last_expert_idx],
            quant_mode=-1,
            row_idx_type=0,
        )
        expert_tokens = expert_tokens.to(torch.int64)
        group_list_type = 0  # `cumsum` mode

        return MoETokenDispatchOutput(
            hidden_states=sorted_hidden_states,
            group_list=expert_tokens,
            group_list_type=group_list_type,
            combine_metadata=MoEAllGatherCombineMetadata(
                topk_weights=topk_weights,
                expanded_row_idx=expanded_row_idx,
                restore_shape=restore_shape,
            ),
        )
