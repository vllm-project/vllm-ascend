# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

"""Ascend-owned EPLB state extensions."""

from dataclasses import fields
from typing import Any

import torch
from vllm.distributed import get_ep_group
from vllm.distributed.eplb import eplb_state as _eplb_state

from vllm_ascend.ops.fused_moe import eplb as _eplb_ops


class AscendEplbLayerState(_eplb_state.EplbLayerState):
    """EPLB layer state with a graph-stable Ascend mapping lookup."""

    def __init__(self) -> None:
        super().__init__()
        self.physical_id_lookup: torch.Tensor | None = None

    @classmethod
    def from_upstream(cls, state: _eplb_state.EplbLayerState) -> "AscendEplbLayerState":
        ascend_state = cls()
        for field in fields(_eplb_state.EplbLayerState):
            setattr(ascend_state, field.name, getattr(state, field.name))
        return ascend_state

    def set_layer_state(
        self,
        moe_layer_idx: int,
        expert_load_view: torch.Tensor,
        logical_to_physical_map: torch.Tensor,
        logical_replica_count: torch.Tensor,
    ) -> None:
        super().set_layer_state(
            moe_layer_idx,
            expert_load_view,
            logical_to_physical_map,
            logical_replica_count,
        )
        self.refresh_physical_id_lookup()

    def refresh_physical_id_lookup(self) -> None:
        logical_to_physical_map = self.logical_to_physical_map
        logical_replica_count = self.logical_replica_count
        if logical_to_physical_map is None or logical_replica_count is None:
            raise RuntimeError("Cannot build Ascend EPLB lookup before layer state is initialized.")

        new_lookup = _eplb_ops.build_physical_id_lookup(
            logical_to_physical_map,
            logical_replica_count,
            get_ep_group().rank_in_group,
        )
        if self.physical_id_lookup is not None and self.physical_id_lookup.shape == new_lookup.shape:
            self.physical_id_lookup.copy_(new_lookup, non_blocking=True)
        else:
            self.physical_id_lookup = new_lookup


def refresh_model_lookups(model_state: Any, layer_idx: int | None = None) -> None:
    """Refresh all lookups, or one layer after an async map commit."""
    layers = list(model_state.model.moe_layers)
    selected_layers = enumerate(layers) if layer_idx is None else ((layer_idx, layers[layer_idx]),)
    for _, layer in selected_layers:
        layer_state = layer.eplb_state
        if isinstance(layer_state, AscendEplbLayerState):
            layer_state.refresh_physical_id_lookup()


class AscendEplbState(_eplb_state.EplbState):
    """Own Ascend lookup refreshes without patching upstream commit helpers."""

    def rearrange(
        self,
        is_profile: bool = False,
        rank_mapping: dict[int, int] | None = None,
    ) -> torch.Tensor | None:
        result = super().rearrange(is_profile=is_profile, rank_mapping=rank_mapping)
        if not is_profile and not self.is_async:
            for model_state in self.model_states.values():
                refresh_model_lookups(model_state)
        return result

    @classmethod
    def from_mapping(
        cls,
        model,
        model_config,
        device: torch.device,
        parallel_config,
        expanded_physical_to_logical: torch.Tensor,
        num_valid_physical_experts: int,
    ) -> "AscendEplbState":
        state = super().from_mapping(
            model=model,
            model_config=model_config,
            device=device,
            parallel_config=parallel_config,
            expanded_physical_to_logical=expanded_physical_to_logical,
            num_valid_physical_experts=num_valid_physical_experts,
        )
        for model_state in state.model_states.values():
            refresh_model_lookups(model_state)
        return state
