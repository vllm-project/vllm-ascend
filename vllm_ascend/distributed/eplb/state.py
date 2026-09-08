# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

"""Ascend-owned extensions for the upstream EPLB state."""

import inspect
from dataclasses import fields
from typing import Any

import numpy as np
import torch
from torch.distributed import all_reduce
from vllm.distributed import get_ep_group
from vllm.distributed.eplb import eplb_state as _eplb_state
from vllm.distributed.parallel_state import get_node_count

from vllm_ascend.ascend_config import StairConfig
from vllm_ascend.ops.fused_moe import eplb as _eplb_ops

ASYNC_EPLB_CYCLE_COMMITTED_LOG = "Ascend async EPLB cycle committed"


def _upstream_from_mapping_accepts_valid_expert_count() -> bool:
    """Return whether the selected vLLM uses the release mapping contract."""
    return "num_valid_physical_experts" in inspect.signature(_eplb_state.EplbState.from_mapping).parameters


class AscendEplbLayerState(_eplb_state.EplbLayerState):
    """EPLB layer state with a graph-stable replica routing table."""

    def __init__(self) -> None:
        super().__init__()
        self.expert_replica_routing_table: torch.Tensor | None = None

    @classmethod
    def from_upstream(
        cls,
        state: _eplb_state.EplbLayerState,
    ) -> "AscendEplbLayerState":
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
        self.refresh_expert_replica_routing_table()

    def refresh_expert_replica_routing_table(self) -> None:
        logical_to_physical_map = self.logical_to_physical_map
        logical_replica_count = self.logical_replica_count
        if logical_to_physical_map is None or logical_replica_count is None:
            raise RuntimeError("Cannot build the replica routing table before EPLB layer state is initialized.")

        new_routing_table = _eplb_ops.build_expert_replica_routing_table(
            logical_to_physical_map,
            logical_replica_count,
            get_ep_group().rank_in_group,
        )
        if (
            self.expert_replica_routing_table is not None
            and self.expert_replica_routing_table.shape == new_routing_table.shape
        ):
            self.expert_replica_routing_table.copy_(
                new_routing_table,
                non_blocking=True,
            )
        else:
            self.expert_replica_routing_table = new_routing_table


def refresh_model_routing_tables(
    model_state: Any,
    layer_idx: int | None = None,
) -> None:
    """Refresh every routing table, or one table after an async commit."""
    layers = list(model_state.model.moe_layers)
    selected_layers = enumerate(layers) if layer_idx is None else ((layer_idx, layers[layer_idx]),)
    for _, layer in selected_layers:
        layer_state = layer.eplb_state
        if isinstance(layer_state, AscendEplbLayerState):
            layer_state.refresh_expert_replica_routing_table()


class AscendEplbState(_eplb_state.EplbState):
    """Keep Ascend routing and load-recording state around upstream EPLB."""

    cuda_device_index: int | None

    def __init__(
        self,
        parallel_config,
        device: torch.device,
        stair_config: StairConfig | None = None,
    ) -> None:
        super().__init__(parallel_config, device)
        self._has_fresh_recorded_load = False
        self._stair_config = stair_config
        if self.cuda_device_index is None:
            self.cuda_device_index = torch.accelerator.current_device_index()

    def _initialize_stair_model(self, model_state: Any) -> None:
        model = model_state.model
        model_state._stair_load_window = torch.zeros(
            self.expert_load_window_size,
            model.num_moe_layers,
            model.num_logical_experts,
            dtype=torch.int64,
            device=self.device,
        )
        model_state._stair_accepted_scores = np.full(model.num_moe_layers, np.nan)

    def add_model(self, model, model_config) -> None:
        super().add_model(model, model_config)
        if self._stair_config is not None:
            self._initialize_stair_model(self.model_states[model_config.compute_hash()])

    def step(
        self,
        is_dummy: bool = False,
        is_profile: bool = False,
        log_stats: bool = False,
    ) -> None:
        if (
            self._stair_config is not None
            and not is_dummy
            and not is_profile
            and self._should_record_current_step(log_stats=log_stats)
        ):
            for model_state in self.model_states.values():
                target = model_state._stair_load_window[self.expert_load_window_step]
                target.zero_()
                target.scatter_add_(
                    1,
                    model_state.physical_to_logical_map.long(),
                    model_state.expert_load_pass.to(torch.int64),
                )
        super().step(is_dummy=is_dummy, is_profile=is_profile, log_stats=log_stats)

    def _has_global_fresh_recorded_load(self) -> bool:
        """Synchronize whether any EP rank recorded load since rearranging."""
        ep_group = get_ep_group()
        cpu_group = getattr(ep_group, "cpu_group", None)
        if cpu_group is not None:
            if cpu_group.size() <= 1:
                return self._has_fresh_recorded_load
            flag = torch.tensor(
                (self._has_fresh_recorded_load,),
                dtype=torch.int32,
                device="cpu",
            )
            all_reduce(flag, group=cpu_group)
            return bool(flag.item())

        device_group = ep_group.device_group
        if device_group.size() <= 1:
            return self._has_fresh_recorded_load
        flag = torch.tensor(
            (self._has_fresh_recorded_load,),
            dtype=torch.int32,
            device=self.device,
        )
        all_reduce(flag, group=device_group)
        return bool(flag.item())

    def _rearrange_stair(self) -> None:
        ep_group = get_ep_group().device_group
        num_ranks = ep_group.size()
        num_nodes = get_node_count()
        if num_ranks % num_nodes:
            num_nodes = 1
        for model_state in self.model_states.values():
            logical_load = torch.roll(
                model_state._stair_load_window,
                shifts=-self.expert_load_window_step,
                dims=0,
            ).clone()
            all_reduce(logical_load, group=ep_group)
            model = model_state.model
            model_state.eplb_stats = _eplb_state.EplbStats(
                global_expert_load_window=logical_load,
                num_replicas=model.num_physical_experts,
                num_groups=model.num_expert_groups,
                num_nodes=num_nodes,
                num_gpus=num_ranks,
            )
            model_state.rebalanced = True
        self.rearrange_event.record()

    def rearrange(
        self,
        is_profile: bool = False,
        rank_mapping: dict[int, int] | None = None,
    ) -> torch.Tensor | None:
        should_gate = (
            hasattr(self, "_has_fresh_recorded_load")
            and not is_profile
            and rank_mapping is None
            and not self.parallel_config.enable_elastic_ep
        )
        if should_gate and not self._has_global_fresh_recorded_load():
            return None

        if getattr(self, "_stair_config", None) is not None and not is_profile:
            if rank_mapping is not None:
                raise ValueError("STAIR does not support elastic expert parallelism")
            self._rearrange_stair()
            result = None
        else:
            result = super().rearrange(
                is_profile=is_profile,
                rank_mapping=rank_mapping,
            )
        if not is_profile and not self.is_async:
            for model_state in self.model_states.values():
                refresh_model_routing_tables(model_state)
        if not is_profile:
            self._has_fresh_recorded_load = False
        return result

    @classmethod
    def from_mapping(
        cls,
        model,
        model_config,
        device: torch.device,
        parallel_config,
        expanded_physical_to_logical: torch.Tensor,
        num_valid_physical_experts: int | None = None,
    ) -> "AscendEplbState":
        from_mapping_kwargs: dict[str, Any] = {
            "model": model,
            "model_config": model_config,
            "device": device,
            "parallel_config": parallel_config,
            "expanded_physical_to_logical": expanded_physical_to_logical,
        }
        if _upstream_from_mapping_accepts_valid_expert_count():
            if num_valid_physical_experts is None:
                raise TypeError("num_valid_physical_experts is required by the selected vLLM release mapping contract")
            from_mapping_kwargs["num_valid_physical_experts"] = num_valid_physical_experts
        state = super().from_mapping(**from_mapping_kwargs)
        for model_state in state.model_states.values():
            refresh_model_routing_tables(model_state)
        return state
