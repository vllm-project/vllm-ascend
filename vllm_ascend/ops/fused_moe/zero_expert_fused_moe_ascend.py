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
Ascend-optimized extension for vllm's ZeroExpertFusedMoE.
This module provides support for the select_experts method required by ZeroExpertFusedMoE.
"""

from vllm.logger import logger
from vllm.model_executor.layers.fused_moe.zero_expert_fused_moe import (
    ZeroExpertFusedMoE as VLLMZeroExpertFusedMoE,
)

from vllm_ascend.ops.fused_moe.experts_selector import select_experts


def _zero_expert_fused_moe_select_experts(self):
    """
    Select the top-k experts for each token using Ascend-optimized routing.

    This method is added to vllm's ZeroExpertFusedMoE via monkey-patching to provide
    Ascend-optimized expert selection. It will be called by the forward method after
    capturing the necessary context.

    Returns:
        tuple: (topk_weights, topk_ids) where
            - topk_weights: Expert weights of shape (num_tokens, top_k)
            - topk_ids: Selected expert IDs of shape (num_tokens, top_k)
    """
    # Get configuration attributes from self
    top_k = self.top_k
    use_grouped_topk = self.use_grouped_topk
    renormalize = self.renormalize
    topk_group = self.topk_group
    num_expert_group = self.num_expert_group
    custom_routing_function = self.custom_routing_function
    scoring_func = self.scoring_func
    routed_scaling_factor = self.routed_scaling_factor
    e_score_correction_bias = self.e_score_correction_bias
    global_num_experts = self.global_num_experts

    # Get hidden_states and router_logits from the captured context
    # These are set by the forward method before calling select_experts
    hidden_states = self._select_experts_hidden_states
    router_logits = self._select_experts_router_logits

    topk_weights, topk_ids = select_experts(
        hidden_states=hidden_states,
        router_logits=router_logits,
        top_k=top_k,
        use_grouped_topk=use_grouped_topk,
        renormalize=renormalize,
        topk_group=topk_group,
        num_expert_group=num_expert_group,
        custom_routing_function=custom_routing_function,
        scoring_func=scoring_func,
        routed_scaling_factor=routed_scaling_factor,
        e_score_correction_bias=e_score_correction_bias,
        global_num_experts=global_num_experts,
    )
    return topk_weights, topk_ids


# Store the original forward method
_original_zero_expert_fused_moe_forward = None


def _zero_expert_fused_moe_forward_wrapper(original_forward):
    """
    Wrap the original ZeroExpertFusedMoE.forward to capture context for select_experts.
    """

    def forward(self, *args, **kwargs):
        # Capture the context for select_experts to use
        # The first argument is typically hidden_states, the second is router_logits
        if len(args) >= 2:
            self._select_experts_hidden_states = args[0]
            self._select_experts_router_logits = args[1]
        elif 'hidden_states' in kwargs and 'router_logits' in kwargs:
            self._select_experts_hidden_states = kwargs['hidden_states']
            self._select_experts_router_logits = kwargs['router_logits']
        else:
            logger.warning(
                'ZeroExpertFusedMoE.forward wrapper could not capture hidden_states and '
                'router_logits. select_experts may fail.'
            )

        # Call the original forward method
        return original_forward(self, *args, **kwargs)

    return forward


def patch_zero_expert_fused_moe():
    """
    Monkey-patch vllm's ZeroExpertFusedMoE to add the select_experts method.
    This ensures that the upstream vllm's ZeroExpertFusedMoE can work with Ascend optimization.
    """
    # Only apply the patch if select_experts is not already defined
    if hasattr(VLLMZeroExpertFusedMoE, 'select_experts') and callable(
        getattr(VLLMZeroExpertFusedMoE, 'select_experts')
    ):
        return

    # Add the select_experts method
    VLLMZeroExpertFusedMoE.select_experts = _zero_expert_fused_moe_select_experts

    # Wrap the forward method to capture context
    if hasattr(VLLMZeroExpertFusedMoE, 'forward'):
        original_forward = VLLMZeroExpertFusedMoE.forward
        VLLMZeroExpertFusedMoE.forward = _zero_expert_fused_moe_forward_wrapper(original_forward)

    logger.debug('Patched ZeroExpertFusedMoE with Ascend optimizations')


# Apply the patch when this module is imported
patch_zero_expert_fused_moe()


