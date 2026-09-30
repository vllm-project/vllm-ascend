import torch
import torch.nn.functional as F
from vllm.distributed.eplb.eplb_state import EplbLayerState
from vllm.model_executor.layers.fused_moe.config import (
    RoutingMethodType,
    get_routing_method_type,
)
from vllm.model_executor.layers.fused_moe.router.base_router import BaseRouter


class AscendGroupedTopKRouter(BaseRouter):
    def __init__(
        self,
        top_k: int,
        global_num_experts: int,
        num_expert_group: int | None,
        topk_group: int | None,
        use_grouped_topk: bool = False,
        renormalize: bool = True,
        scoring_func: str = "softmax",
        routed_scaling_factor: float = 1.0,
        e_score_correction_bias: torch.Tensor | None = None,
        num_fused_shared_experts: int = 0,
        eplb_state: EplbLayerState | None = None,
    ):
        super().__init__(
            top_k=top_k,
            global_num_experts=global_num_experts,
            eplb_state=eplb_state,
        )
        self.num_expert_group = num_expert_group
        self.topk_group = topk_group
        self.renormalize = renormalize
        self.scoring_func = scoring_func
        self.routed_scaling_factor = routed_scaling_factor
        self.e_score_correction_bias = e_score_correction_bias
        self.num_fused_shared_experts = num_fused_shared_experts
        self.use_grouped_topk = use_grouped_topk

    @property
    def routing_method_type(self) -> RoutingMethodType:
        return get_routing_method_type(
            scoring_func=self.scoring_func,
            top_k=self.top_k,
            renormalize=self.renormalize,
            num_expert_group=self.num_expert_group if self.use_grouped_topk else None,
            has_e_score_bias=self.e_score_correction_bias is not None,
            routed_scaling_factor=self.routed_scaling_factor,
        )

    def _renormalize_topk_weights(
        self,
        topk_weights: torch.Tensor,
    ):
        if self.renormalize:
            topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
        return topk_weights

    def _native_grouped_topk(
        self,
        topk_weights: torch.Tensor,
    ):
        topk_group = 0 if self.topk_group is None else self.topk_group
        num_expert_group = 0 if self.num_expert_group is None else self.num_expert_group

        num_token = topk_weights.shape[0]
        grouped_weights = topk_weights.view(num_token, num_expert_group, -1).max(dim=-1).values
        topk_group_indices = torch.topk(grouped_weights.to(torch.float32), k=topk_group, dim=-1, sorted=False)[1]
        topk_group_mask = torch.zeros_like(grouped_weights)
        topk_group_mask.scatter_(1, topk_group_indices, 1)
        topk_weight_mask = (
            topk_group_mask.unsqueeze(-1)
            .expand(num_token, num_expert_group, topk_weights.shape[-1] // num_expert_group)
            .reshape(num_token, -1)
        )
        topk_weights = topk_weights.masked_fill(~topk_weight_mask.bool(), 0.0)

        return topk_weights

    def _compute_routing(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        indices_type: torch.dtype | None,
        *,
        input_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.scoring_func == "softmax":
            topk_weights = router_logits.softmax(dim=-1)
        elif self.scoring_func == "sigmoid":
            topk_weights = router_logits.sigmoid()
        elif self.scoring_func == "sqrtsoftplus":
            topk_weights = F.softplus(router_logits).sqrt()
        else:
            raise ValueError(f"Unsupported scoring function: {self.scoring_func}")

        if self.use_grouped_topk:
            if self.e_score_correction_bias is not None:
                # Store original scores before applying correction bias. We use biased
                # scores for expert selection but original scores for routing weights
                original_weights = topk_weights
                topk_weights = topk_weights + self.e_score_correction_bias.unsqueeze(0)

            # TODO: Change to npu_group_topk when the latest CANN and NNAL is available
            # >>> torch_npu._npu_group_topk(topk_weights, group_num=num_expert_group, k=topk_group)
            topk_weights = self._native_grouped_topk(topk_weights)
            # TODO bfloat16 is not supported in torch.topk with ge graph.
            if self.e_score_correction_bias is not None:
                topk_ids = torch.topk(topk_weights.to(torch.float32), k=self.top_k, dim=-1, sorted=False)[1]
                # Use original unbiased scores for the routing weights
                topk_weights = original_weights.gather(1, topk_ids)
            else:
                topk_weights, topk_ids = torch.topk(topk_weights.to(torch.float32), k=self.top_k, dim=-1, sorted=False)
            topk_ids = topk_ids.to(torch.int32)
            topk_weights = self._renormalize_topk_weights(topk_weights)
            return topk_weights * self.routed_scaling_factor, topk_ids

        if self.e_score_correction_bias is not None:
            topk_weights = topk_weights + self.e_score_correction_bias

        topk_weights, topk_ids = topk_weights.topk(self.top_k, dim=-1)

        # Required by npu_moe_init_routing
        topk_ids = topk_ids.to(torch.int32)
        topk_weights = self._renormalize_topk_weights(topk_weights)
        topk_weights = topk_weights * self.routed_scaling_factor

        return topk_weights, topk_ids
