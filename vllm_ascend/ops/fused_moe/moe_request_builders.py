from __future__ import annotations

from typing import TypeVar

import torch

from vllm_ascend.ops.fused_moe.moe_runtime_args import (
    MoEFusedExpertsInput,
    MoEMlpComputeInput,
    MoEMxfpParams,
    MoEQuantParams,
    MoERoutingParams,
    MoETokenDispatchInput,
    MoETokenDispatchOutput,
    MoEWeights,
)
from vllm_ascend.quantization.quant_type import QuantType

TMoERoutingMetadata = TypeVar("TMoERoutingMetadata")


def build_fused_experts_input(
    *,
    hidden_states: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    w1: torch.Tensor | list[torch.Tensor],
    w2: torch.Tensor | list[torch.Tensor],
    quant_type: QuantType,
    dynamic_eplb: bool,
    expert_map: torch.Tensor | None = None,
    global_redundant_expert_num: int = 0,
    mc2_mask: torch.Tensor | None = None,
    apply_router_weight_on_input: bool = False,
    log2phy: torch.Tensor | None = None,
    pertoken_scale: torch.Tensor | None = None,
    activation: str = "silu",
    need_trans: bool = False,
    w1_bias: torch.Tensor | None = None,
    w2_bias: torch.Tensor | None = None,
    comm_quant_mode: int | None = None,
    mxfp: MoEMxfpParams | None = None,
    w1_scale: list[torch.Tensor] | torch.Tensor | None = None,
    w2_scale: list[torch.Tensor] | torch.Tensor | None = None,
    w1_scale_bias: torch.Tensor | None = None,
    w2_scale_bias: torch.Tensor | None = None,
    w1_offset: torch.Tensor | None = None,
    w2_offset: torch.Tensor | None = None,
) -> MoEFusedExpertsInput:
    if quant_type == QuantType.MXFP8 and mxfp is None:
        raise ValueError("mxfp params are required when quant_type is QuantType.MXFP8.")

    return MoEFusedExpertsInput(
        hidden_states=hidden_states,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        weights=MoEWeights(
            w1=w1,
            w2=w2,
            w1_bias=w1_bias,
            w2_bias=w2_bias,
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            w1_scale_bias=w1_scale_bias,
            w2_scale_bias=w2_scale_bias,
            w1_offset=w1_offset,
            w2_offset=w2_offset,
        ),
        routing=MoERoutingParams(
            expert_map=expert_map,
            global_redundant_expert_num=global_redundant_expert_num,
            mc2_mask=mc2_mask,
            apply_router_weight_on_input=apply_router_weight_on_input,
            log2phy=log2phy,
            pertoken_scale=pertoken_scale,
        ),
        activation=activation,
        need_trans=need_trans,
        dynamic_eplb=dynamic_eplb,
        quant=MoEQuantParams(
            quant_type=quant_type,
            comm_quant_mode=comm_quant_mode,
            mxfp=mxfp,
        ),
    )


def build_token_dispatch_input(
    *,
    request: MoEFusedExpertsInput,
    topk_ids: torch.Tensor | None = None,
) -> MoETokenDispatchInput:
    return MoETokenDispatchInput(
        hidden_states=request.hidden_states,
        topk_weights=request.topk_weights,
        topk_ids=request.topk_ids if topk_ids is None else topk_ids,
        routing=request.routing,
        quant=request.quant,
    )


def build_mlp_compute_input(
    *,
    request: MoEFusedExpertsInput,
    dispatch_result: MoETokenDispatchOutput[TMoERoutingMetadata],
    use_fusion_ops: bool,
) -> MoEMlpComputeInput:
    if request.quant.is_mxfp and request.quant.mxfp is None:
        raise ValueError("request.quant.mxfp is required when quant_type is QuantType.MXFP8.")

    return MoEMlpComputeInput(
        hidden_states=dispatch_result.hidden_states,
        group_list=dispatch_result.group_list,
        group_list_type=dispatch_result.group_list_type,
        dynamic_scale=dispatch_result.dynamic_scale,
        topk_scales=dispatch_result.topk_scales,
        weights=request.weights,
        quant=request.quant,
        fusion=request.quant.quant_type in (QuantType.W8A8, QuantType.MXFP8) and use_fusion_ops,
        activation=request.activation,
        need_trans=request.need_trans,
        dynamic_eplb=request.dynamic_eplb,
    )
