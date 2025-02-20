from typing import Callable, Optional

import torch
import torch.nn.functional as F
from vllm.model_executor.layers.fused_moe.layer import UnquantizedFusedMoEMethod


def fused_moe(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    global_num_experts: int,
    expert_map: torch.Tensor = None,
    renormalize: bool = False,
) -> torch.Tensor:
    """
    Args:
        hidden_states: [*, hidden_size]
        w1: [num_experts, intermediate_size * 2, hidden_size]
        w2: [num_experts, hidden_size, intermediate_size]
        gating_output: [*, num_experts]
        expert_map: [num_experts]
    """
    orig_shape = hidden_states.shape
    hidden_size = hidden_states.shape[-1]
    num_tokens = hidden_states.shape[:-1].numel()
    num_experts = w1.shape[0]
    intermediate_size = w2.shape[-1]
    dtype = hidden_states.dtype

    hidden_states = hidden_states.view(num_tokens, hidden_size)
    gating_output = gating_output.view(num_tokens, global_num_experts)
    topk_weights = gating_output.softmax(dim=-1, dtype=torch.float)
    topk_weights, selected_experts = topk_weights.topk(topk, dim=-1)
    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    topk_weights = topk_weights.to(dtype)

    if expert_map is not None:
        selected_experts = expert_map[selected_experts]

    final_hidden_states = None
    for expert_idx in range(num_experts):
        expert_w1 = w1[expert_idx]
        expert_w2 = w2[expert_idx]
        expert_mask = selected_experts == expert_idx
        expert_weights = (topk_weights * expert_mask).sum(dim=-1, keepdim=True)
        x = F.linear(hidden_states, expert_w1)
        gate = F.silu(x[:, :intermediate_size])
        x = x[:, intermediate_size:] * gate
        x = F.linear(x, expert_w2)
        current_hidden_states = x * expert_weights
        if final_hidden_states is None:
            final_hidden_states = current_hidden_states
        else:
            final_hidden_states = final_hidden_states + current_hidden_states

    return final_hidden_states.view(orig_shape)  # type: ignore


def forward_oot(
    self,
    layer: torch.nn.Module,
    x: torch.Tensor,
    use_grouped_topk: bool,
    top_k: int,
    router_logits: torch.Tensor,
    renormalize: bool,
    topk_group: Optional[int] = None,
    num_expert_group: Optional[int] = None,
    global_num_experts: int = -1,
    expert_map: Optional[torch.Tensor] = None,
    custom_routing_function: Optional[Callable] = None,
    scoring_func: str = "softmax",
    e_score_correction_bias: Optional[torch.Tensor] = None,
):
    return fused_moe(
        hidden_states=x,
        w1=layer.w13_weight,
        w2=layer.w2_weight,
        topk=top_k,
        gating_output=router_logits,
        global_num_experts=global_num_experts,
        expert_map=expert_map,
        renormalize=renormalize,
    )


UnquantizedFusedMoEMethod.forward_oot = forward_oot
