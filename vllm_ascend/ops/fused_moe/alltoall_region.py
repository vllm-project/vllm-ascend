# SPDX-License-Identifier: Apache-2.0
"""Keep ragged AllToAll receive buffers inside the routed-expert region.

Dispatch alone cannot hide sum(output_splits): its output feeds the expert MLP.
The first shape-preserving boundary ends after reverse exchange/unpermutation.
Router, shared experts and prepare/finalize remain outside this region.
"""

import torch


@torch.library.custom_op("vllm_ascend::fxrt_alltoall_routed_experts", mutates_args=())
def alltoall_routed_experts(
    hidden_states: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    w1: list[torch.Tensor],
    w2: list[torch.Tensor],
    w1_scale: list[torch.Tensor],
    w2_scale: list[torch.Tensor],
    w1_bias: torch.Tensor | None,
    w2_bias: torch.Tensor | None,
    expert_map: torch.Tensor | None,
    log2phy: torch.Tensor | None,
    pertoken_scale: torch.Tensor | None,
    mc2_mask: torch.Tensor | None,
    global_redundant_expert_num: int,
    apply_router_weight_on_input: bool,
    activation: str,
    need_trans: bool,
    comm_quant_mode: int | None,
    is_per_channel_weight: bool,
    swiglu_limit: float,
    num_local_experts: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    # Lazy imports avoid the comm-method/custom-op registration cycle.
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation

    from vllm_ascend.ascend_forward_context import MoECommType
    from vllm_ascend.ops.fused_moe.moe_comm_method import MoECommMethod, get_moe_comm_method
    from vllm_ascend.ops.fused_moe.moe_runtime_args import build_fused_experts_input
    from vllm_ascend.quantization.quant_type import QuantType

    method = get_moe_comm_method(MoECommType.ALLTOALL)
    if method is None:
        raise RuntimeError("AllToAll communication method has not been initialized")
    payload = build_fused_experts_input(
        hidden_states=hidden_states,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        w1=w1,
        w2=w2,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        w1_bias=w1_bias,
        w2_bias=w2_bias,
        quant_type=QuantType.W8A8,
        dynamic_eplb=False,
        expert_map=expert_map,
        log2phy=log2phy,
        pertoken_scale=pertoken_scale,
        mc2_mask=mc2_mask,
        global_redundant_expert_num=global_redundant_expert_num,
        apply_router_weight_on_input=apply_router_weight_on_input,
        activation=MoEActivation(activation),
        need_trans=need_trans,
        comm_quant_mode=comm_quant_mode,
        is_per_channel_weight=is_per_channel_weight,
        swiglu_limit=swiglu_limit,
    )
    # Explicit base call prevents recursion through AlltoAllCommImpl's wrapper.
    # Original async_op=True collectives and Work.wait() remain inside, without
    # exposing Work, CPU split lists or value-dependent allocations to Dynamo.
    result = MoECommMethod.fused_experts(method, payload)
    return result.routed_out, result.expert_tokens.to(torch.int64)


@alltoall_routed_experts.register_fake
def _alltoall_routed_experts_fake(
    hidden_states,
    topk_weights,
    topk_ids,
    w1,
    w2,
    w1_scale,
    w2_scale,
    w1_bias,
    w2_bias,
    expert_map,
    log2phy,
    pertoken_scale,
    mc2_mask,
    global_redundant_expert_num,
    apply_router_weight_on_input,
    activation,
    need_trans,
    comm_quant_mode,
    is_per_channel_weight,
    swiglu_limit,
    num_local_experts,
):
    return torch.empty_like(hidden_states), topk_ids.new_empty((num_local_experts,), dtype=torch.int64)


def run_alltoall_routed_region(method, payload):
    """Flatten the supported DSV4 W8A8 contract, keeping weights as graph inputs."""
    from vllm_ascend.ops.fused_moe.moe_comm_method import FusedExpertsResult
    from vllm_ascend.quantization.quant_type import QuantType

    weights = payload.weights
    if payload.quant.quant_type != QuantType.W8A8 or payload.dynamic_eplb or payload.lora_context is not None:
        raise NotImplementedError("AllToAll region supports static W8A8 experts without LoRA")
    if any(v is not None for v in (weights.w1_scale_bias, weights.w2_scale_bias, weights.w1_offset, weights.w2_offset)):
        raise NotImplementedError("AllToAll W8A8 region does not support fused scale-bias or weight offsets")

    def tensors(value):
        if isinstance(value, torch.Tensor):
            return [value]
        assert value is not None
        return value

    routed, counts = alltoall_routed_experts(
        payload.hidden_states,
        payload.topk_weights,
        payload.topk_ids,
        tensors(weights.w1),
        tensors(weights.w2),
        tensors(weights.w1_scale),
        tensors(weights.w2_scale),
        weights.w1_bias,
        weights.w2_bias,
        payload.routing.expert_map,
        payload.routing.log2phy,
        payload.routing.pertoken_scale,
        payload.routing.mc2_mask,
        payload.routing.global_redundant_expert_num,
        payload.routing.apply_router_weight_on_input,
        getattr(payload.activation, "value", payload.activation),
        payload.need_trans,
        payload.quant.comm_quant_mode,
        payload.quant.is_per_channel_weight,
        payload.swiglu_limit,
        method.moe_config.num_local_experts,
    )
    return FusedExpertsResult(
        routed_out=routed,
        expert_tokens=counts,
        group_list_type=1,
        swiglu_limit=payload.swiglu_limit,
    )
