from collections.abc import Callable

import torch
from vllm.distributed.eplb.eplb_state import EplbLayerState
from vllm.model_executor.layers.fused_moe import FusedMoERouter
from vllm.model_executor.layers.fused_moe.router.custom_routing_router import CustomRoutingRouter

from vllm_ascend.device.hardware_profile import HardwareCapability, get_current_hardware_profile
from vllm_ascend.ops.fused_moe.router.fused_topk_router import (
    AscendFusedTopKRouter as AscendFusedMoERouter,
)
from vllm_ascend.ops.fused_moe.router.grouped_topk_router import AscendGroupedTopKRouter


def check_npu_moe_gating_top_k(
    top_k: int,
    renormalize: bool,
    topk_group: int | None = None,
    num_expert_group: int | None = None,
    scoring_func: str = "softmax",
    custom_routing_function: Callable | None = None,
):
    if scoring_func == "sigmoid" and not renormalize:  # sigmoid + renorm=0 is not supported in current branch
        return False
    if custom_routing_function is not None:
        return False
    if scoring_func != "softmax" and scoring_func != "sigmoid" and scoring_func != "sqrtsoftplus":
        return False
    topk_group = topk_group if topk_group is not None else 1
    num_expert_group = num_expert_group if num_expert_group is not None else 1
    if top_k < 1:
        return False
    return 1 <= topk_group <= num_expert_group


def create_ascend_fused_moe_router(
    top_k: int,
    global_num_experts: int,
    renormalize: bool = True,
    use_grouped_topk: bool = False,
    num_expert_group: int | None = None,
    topk_group: int | None = None,
    scoring_func: str = "softmax",
    num_fused_shared_experts: int = 0,
    shared_expert_weight: float = 1.0,
    routed_scaling_factor: float = 1.0,
    e_score_correction_bias: torch.Tensor | None = None,
    custom_routing_function: Callable | None = None,
    eplb_state: EplbLayerState | None = None,
    num_logical_experts: int | None = None,
    hash_indices_table: torch.Tensor | None = None,
    tid2eid: torch.Tensor | None = None,
    bias_vl: torch.Tensor | None = None,
    image_sentinel_lo: int = 129257,
) -> FusedMoERouter:
    if custom_routing_function is not None:
        return CustomRoutingRouter(
            top_k=top_k,
            global_num_experts=global_num_experts,
            eplb_state=eplb_state,
            custom_routing_function=custom_routing_function,
            renormalize=renormalize,
        )
    if get_current_hardware_profile().supports(HardwareCapability.FUSED_MOE_COMPATIBILITY):
        from vllm_ascend._310p.fused_moe.grouped_topk_router import AscendGroupedTopKRouter310

        return AscendGroupedTopKRouter310(
            top_k=top_k,
            global_num_experts=global_num_experts,
            num_expert_group=num_expert_group,
            topk_group=topk_group,
            use_grouped_topk=use_grouped_topk,
            renormalize=renormalize,
            scoring_func=scoring_func,
            routed_scaling_factor=routed_scaling_factor,
            e_score_correction_bias=e_score_correction_bias,
            num_fused_shared_experts=num_fused_shared_experts,
            eplb_state=eplb_state,
        )
    is_support_npu_moe_gating_top_k = check_npu_moe_gating_top_k(
        top_k=top_k,
        renormalize=renormalize,
        topk_group=topk_group,
        num_expert_group=num_expert_group,
        scoring_func=scoring_func,
        custom_routing_function=custom_routing_function,
    )
    if is_support_npu_moe_gating_top_k:
        return AscendFusedMoERouter(
            top_k=top_k,
            global_num_experts=global_num_experts,
            eplb_state=eplb_state,
            renormalize=renormalize,
            use_grouped_topk=use_grouped_topk,
            num_expert_group=num_expert_group,
            topk_group=topk_group,
            custom_routing_function=custom_routing_function,
            scoring_func=scoring_func,
            routed_scaling_factor=routed_scaling_factor,
            e_score_correction_bias=e_score_correction_bias,
            num_logical_experts=num_logical_experts,
            tid2eid=tid2eid,
            bias_vl=bias_vl,
            image_sentinel_lo=image_sentinel_lo,
        )
    return AscendGroupedTopKRouter(
        top_k=top_k,
        global_num_experts=global_num_experts,
        num_expert_group=num_expert_group,
        topk_group=topk_group,
        use_grouped_topk=use_grouped_topk,
        renormalize=renormalize,
        scoring_func=scoring_func,
        routed_scaling_factor=routed_scaling_factor,
        e_score_correction_bias=e_score_correction_bias,
        num_fused_shared_experts=num_fused_shared_experts,
        eplb_state=eplb_state,
    )
