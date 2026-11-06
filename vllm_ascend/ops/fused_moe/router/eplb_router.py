# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import torch
from vllm.model_executor.layers.fused_moe.router.base_router import BaseRouter

from vllm_ascend.distributed.eplb.eplb_state import AscendEplbLayerState
from vllm_ascend.ops.fused_moe.eplb import use_allgather_eplb


class AscendEplbRouter(BaseRouter):
    """Keep Ascend EPLB state and mapping with the router that consumes them."""

    eplb_state: AscendEplbLayerState | None

    def __init__(self, *args, eplb_state=None, **kwargs):
        if eplb_state is not None and not isinstance(eplb_state, AscendEplbLayerState):
            eplb_state = AscendEplbLayerState.from_upstream(eplb_state)
        super().__init__(*args, eplb_state=eplb_state, **kwargs)

    def _apply_eplb_mapping(self, topk_ids: torch.Tensor) -> torch.Tensor:
        if self.eplb_state is None:
            return topk_ids
        self._validate_eplb_state()
        routing_table = self.eplb_state.expert_replica_routing_table
        if use_allgather_eplb():
            routing_table = self.eplb_state.allgather_expert_replica_routing_table
        if routing_table is None:
            raise RuntimeError("Ascend EPLB expert replica routing table is not initialized.")
        return torch.ops.vllm.ascend_eplb_map_to_physical(topk_ids, routing_table)
