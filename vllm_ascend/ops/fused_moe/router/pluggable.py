# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

"""Keep upstream concrete constructors and metadata with Ascend routing kernels."""

from vllm.model_executor.layers.fused_moe.router.custom_routing_router import CustomRoutingRouter
from vllm.model_executor.layers.fused_moe.router.fused_topk_bias_router import FusedTopKBiasRouter
from vllm.model_executor.layers.fused_moe.router.fused_topk_router import FusedTopKRouter
from vllm.model_executor.layers.fused_moe.router.grouped_topk_router import GroupedTopKRouter
from vllm.model_executor.layers.fused_moe.router.routing_simulator_router import RoutingSimulatorRouter

from vllm_ascend.ops.fused_moe.router.eplb_router import AscendEplbRouter
from vllm_ascend.ops.fused_moe.router.fused_topk_router import AscendFusedTopKRouter
from vllm_ascend.ops.fused_moe.router.grouped_topk_router import AscendGroupedTopKRouter
from vllm_ascend.ops.fused_moe.router.router_factory import check_npu_moe_gating_top_k


def _compute_routing(self, hidden_states, router_logits, indices_type, *, input_ids=None):
    if not check_npu_moe_gating_top_k(
        self.top_k, self.renormalize, self.topk_group, self.num_expert_group, self.scoring_func
    ):
        return AscendGroupedTopKRouter._compute_routing(
            self, hidden_states, router_logits, indices_type, input_ids=input_ids
        )
    return AscendFusedTopKRouter._compute_routing(self, hidden_states, router_logits, indices_type, input_ids=input_ids)


class AscendTopKRouter(FusedTopKRouter, AscendFusedTopKRouter):
    _compute_routing = _compute_routing


class AscendGroupedRouter(GroupedTopKRouter, AscendFusedTopKRouter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_grouped_topk = True

    _compute_routing = _compute_routing


class AscendBiasRouter(FusedTopKBiasRouter, AscendFusedTopKRouter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tid2eid = self._hash_indices_table
        if self.e_score_correction_bias is not None:
            self.num_logical_experts = self.e_score_correction_bias.numel()

    _compute_routing = _compute_routing


class AscendCustomRouter(AscendEplbRouter, CustomRoutingRouter):
    pass


class AscendSimulatorRouter(AscendEplbRouter, RoutingSimulatorRouter):
    pass
