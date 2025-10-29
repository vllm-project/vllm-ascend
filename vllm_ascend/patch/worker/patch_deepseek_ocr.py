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
"""
Patch for DeepseekMoE to use NPU-optimized operations.

This patch replaces DeepseekMoE.forward with an NPU-optimized version that uses
torch_npu.npu_moe_gating_top_k_softmax for better performance on Ascend NPUs.
"""

import torch
import torch_npu
import vllm.model_executor.models.deepseek
from vllm.distributed import tensor_model_parallel_all_reduce


def ascend_deepseek_moe_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    """
    NPU-optimized forward pass for DeepSeek MoE.

    This replaces the default MoE forward to use torch_npu.npu_moe_gating_top_k_softmax
    for better performance on Ascend NPUs.

    Args:
        self: DeepseekMoE instance
        hidden_states: Input tensor [num_tokens, hidden_dim]

    Returns:
        Output tensor [num_tokens, hidden_dim]
    """
    num_tokens, hidden_dim = hidden_states.shape
    hidden_states = hidden_states.view(-1, hidden_dim)

    # Compute shared experts output if available
    if self.config.n_shared_experts is not None:
        shared_output = self.shared_experts(hidden_states)

    # Compute router logits: (num_tokens, n_experts)
    router_logits, _ = self.gate(hidden_states)

    # Use NPU-optimized top-k gating with softmax
    topk_weights, topk_ids, _ = torch_npu.npu_moe_gating_top_k_softmax(
        router_logits, finished=None, k=self.top_k
    )

    # Renormalize topk weights if configured
    if self.config.norm_topk_prob:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

    # Run MoE inference: route tokens to experts
    final_hidden_states = ascend_deepseek_moe_infer(
        self, hidden_states, topk_ids, topk_weights
    )

    # Add shared experts output if available
    if self.config.n_shared_experts is not None:
        final_hidden_states = final_hidden_states + shared_output

    # All-reduce across tensor parallel ranks
    final_hidden_states = tensor_model_parallel_all_reduce(final_hidden_states)

    return final_hidden_states.view(num_tokens, hidden_dim)


def ascend_deepseek_moe_infer(
    self,
    x: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weight: torch.Tensor,
) -> torch.Tensor:
    """
    NPU-optimized MoE inference using expert routing.

    This implementation routes tokens to experts efficiently by:
    1. Counting tokens per expert
    2. Sorting tokens by expert ID
    3. Processing all tokens for each expert in a batch
    4. Scattering results back to original positions
    5. Applying router weights

    Args:
        self: DeepseekMoE instance
        x: Input tensor [num_tokens, hidden_dim]
        topk_ids: Expert IDs for each token [num_tokens, top_k]
        topk_weight: Router weights [num_tokens, top_k]

    Returns:
        Weighted expert outputs [num_tokens, hidden_dim]
    """
    # Count how many tokens go to each expert
    # cnts: [num_tokens, num_experts] with 1s at selected expert positions
    cnts = topk_ids.new_zeros((topk_ids.shape[0], len(self.experts)))
    cnts.scatter_(1, topk_ids, 1)
    tokens_per_expert = cnts.sum(dim=0)  # [num_experts]

    # Sort all (token, expert) pairs by expert ID for efficient batching
    # idxs: indices that would sort the flattened topk_ids
    idxs = topk_ids.to(torch.float32).view(-1).argsort()

    # Gather tokens in sorted order by expert
    # Each token appears top_k times (once for each selected expert)
    sorted_tokens = x[idxs // topk_ids.shape[1]]
    tokens_per_expert = tokens_per_expert.cpu().numpy()

    # Process tokens through each expert
    outputs = []
    start_idx = 0
    for i, num_tokens in enumerate(tokens_per_expert):
        end_idx = start_idx + num_tokens
        if num_tokens == 0:
            continue

        expert = self.experts[i]
        tokens_for_this_expert = sorted_tokens[start_idx:end_idx]
        expert_out = expert(tokens_for_this_expert)
        outputs.append(expert_out)
        start_idx = end_idx

    # Concatenate all expert outputs
    outs = torch.cat(outputs, dim=0) if len(outputs) else sorted_tokens.new_empty(0)

    # Scatter outputs back to original positions
    new_x = torch.empty_like(outs)
    new_x[idxs] = outs

    # Reshape to [num_tokens, top_k, hidden_dim] and apply router weights
    final_out = (
        new_x.view(*topk_ids.shape, -1)
        .type(topk_weight.dtype)
        .mul_(topk_weight.unsqueeze(dim=-1))
        .sum(dim=1)
        .type(new_x.dtype)
    )
    return final_out


# Patch DeepseekMoE class with NPU-optimized forward method
vllm.model_executor.models.deepseek.DeepseekMoE.forward = ascend_deepseek_moe_forward
