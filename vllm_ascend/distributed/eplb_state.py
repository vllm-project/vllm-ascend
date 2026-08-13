# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

"""Ascend EPLB state for Statistical Temporal-Aware Incremental Rebalancing (STAIR)."""

from dataclasses import fields
from typing import Any

import torch
from torch.distributed import all_reduce
from vllm.distributed import get_ep_group
from vllm.distributed.eplb import eplb_state as _eplb_state
from vllm.logger import logger

from vllm_ascend.distributed.stair_policy import StairEplbPolicy
from vllm_ascend.ops.fused_moe import eplb as _eplb_ops


class AscendEplbLayerState(_eplb_state.EplbLayerState):
    """EPLB layer state with a graph-stable replica routing table."""

    def __init__(self) -> None:
        super().__init__()
        self.expert_replica_routing_table: torch.Tensor | None = None

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
            self.expert_replica_routing_table.copy_(new_routing_table, non_blocking=True)
        else:
            self.expert_replica_routing_table = new_routing_table


def refresh_model_routing_tables(model_state: Any, layer_idx: int | None = None) -> None:
    """Refresh all routing tables, or one table after an async map commit."""
    layers = list(model_state.model.moe_layers)
    selected_layers = enumerate(layers) if layer_idx is None else ((layer_idx, layers[layer_idx]),)
    for _, layer in selected_layers:
        layer_state = layer.eplb_state
        if isinstance(layer_state, AscendEplbLayerState):
            layer_state.refresh_expert_replica_routing_table()


class AscendEplbState(_eplb_state.EplbState):
    """STAIR state, load-window preservation, and Ascend async lifecycle."""

    policy: Any
    cuda_device_index: int | None
    async_worker: Any

    def __init__(self, parallel_config, device: torch.device) -> None:
        super().__init__(parallel_config, device)
        self.stair_policy = StairEplbPolicy()
        self.policy = self.stair_policy
        self._has_fresh_recorded_load = False
        self._preserve_expert_load_time_series = False
        if self.cuda_device_index is None:
            self.cuda_device_index = torch.accelerator.current_device_index()

    def add_model(self, model, model_config) -> None:
        super().add_model(model, model_config)
        self.is_async = True
        # Upstream initializes its configured policy in add_model. Ascend MRv2
        # intentionally exposes one placement policy so state cannot diverge
        # between model registrations.
        self.policy = self.stair_policy
        model_state = self.model_states[model_config.compute_hash()]
        model_state_any: Any = model_state
        model_state_any._ascend_eplb_state = self
        logger.info("Selected Ascend EPLB placement policy: Statistical Temporal-Aware Incremental Rebalancing (STAIR)")

    def start_async_loop(
        self,
        rank_mapping: dict[int, int] | None = None,
        is_profile: bool = False,
    ) -> None:
        del rank_mapping
        if self.async_worker is not None:
            return
        from vllm_ascend.distributed.eplb_async_worker import start_async_worker

        self.async_worker = start_async_worker(self, is_profile=is_profile)

    def step(
        self,
        is_dummy: bool = False,
        is_profile: bool = False,
        log_stats: bool = False,
    ) -> None:
        if not is_dummy and not is_profile and self._should_record_current_step(log_stats=log_stats):
            self._has_fresh_recorded_load = True
        super().step(is_dummy=is_dummy, is_profile=is_profile, log_stats=log_stats)

    def _has_global_fresh_recorded_load(self) -> bool:
        """Synchronize whether any EP rank recorded load since rearranging."""
        ep_group = get_ep_group()
        cpu_group = getattr(ep_group, "cpu_group", None)
        if cpu_group is not None:
            if cpu_group.size() <= 1:
                return self._has_fresh_recorded_load
            flag = torch.tensor((self._has_fresh_recorded_load,), dtype=torch.int32, device="cpu")
            all_reduce(flag, group=cpu_group)
            return bool(flag.item())

        device_group = ep_group.device_group
        if device_group.size() <= 1:
            return self._has_fresh_recorded_load
        flag = torch.tensor((self._has_fresh_recorded_load,), dtype=torch.int32, device=self.device)
        all_reduce(flag, group=device_group)
        return bool(flag.item())

    def _build_logical_expert_load_time_series(self) -> list[torch.Tensor]:
        logical_load_windows = []
        for model_state in self.model_states.values():
            physical_load_window = model_state.expert_load_window[:, :, : self.num_valid_physical_experts]
            logical_load_window = torch.zeros(
                physical_load_window.shape[0],
                model_state.model.num_moe_layers,
                model_state.model.num_logical_experts,
                dtype=physical_load_window.dtype,
                device=physical_load_window.device,
            )
            logical_load_window.scatter_add_(
                dim=-1,
                index=model_state.physical_to_logical_map[:, : self.num_valid_physical_experts]
                .unsqueeze(0)
                .expand_as(physical_load_window)
                .long(),
                src=physical_load_window,
            )
            logical_load_windows.append(logical_load_window)
        return logical_load_windows

    def _allreduce_list(self, tensor_list: list[torch.Tensor]) -> list[torch.Tensor]:
        """Preserve the STAIR window axis across the upstream collective."""
        if not self._preserve_expert_load_time_series:
            return super()._allreduce_list(tensor_list)
        temporal_load_windows = (
            tensor_list
            if all(tensor.dim() == 3 for tensor in tensor_list)
            else self._build_logical_expert_load_time_series()
        )
        shapes = [tensor.shape for tensor in temporal_load_windows]
        flattened = [tensor.reshape(-1, tensor.shape[-1]) for tensor in temporal_load_windows]
        reduced = super()._allreduce_list(flattened)
        return [tensor.reshape(shape) for tensor, shape in zip(reduced, shapes)]

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

        self._preserve_expert_load_time_series = True
        try:
            result = super().rearrange(is_profile=is_profile, rank_mapping=rank_mapping)
        finally:
            self._preserve_expert_load_time_series = False
        if not is_profile:
            self._has_fresh_recorded_load = False
        return result

    def commit_policy_layer(self, model_state: Any, layer_idx: int) -> None:
        """Commit STAIR hysteresis after upstream commits weights and maps."""
        load_window = getattr(model_state, "_ascend_eplb_policy_load", None)
        if load_window is None or model_state.eplb_stats is None:
            return
        self.stair_policy.commit_layer(
            load_window,
            layer_idx,
            model_state.physical_to_logical_map[layer_idx].cpu(),
            model_state.eplb_stats.num_gpus,
        )

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
            refresh_model_routing_tables(model_state)
        return state
