# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

"""Ascend-owned extensions for the upstream EPLB state."""

import inspect
from contextvars import ContextVar
from dataclasses import fields
from typing import Any

import torch
from torch.distributed import all_reduce
from vllm.distributed import get_ep_group, get_eplb_group
from vllm.distributed.eplb import eplb_state as _eplb_state

from vllm_ascend.distributed.eplb.policy import PreparedLoadStats
from vllm_ascend.ops.fused_moe import eplb as _eplb_ops

ASYNC_EPLB_CYCLE_COMMITTED_LOG = "Ascend async EPLB cycle committed"
EXPERT_MAPPING_EP_SIZE: ContextVar[int] = ContextVar("vllm_ascend_expert_mapping_ep_size", default=1)


def _upstream_from_mapping_accepts_valid_expert_count() -> bool:
    """Return whether the selected vLLM uses the release mapping contract."""
    return "num_valid_physical_experts" in inspect.signature(_eplb_state.EplbState.from_mapping).parameters


class AscendEplbLayerState(_eplb_state.EplbLayerState):
    """EPLB layer state with a graph-stable replica routing table."""

    def __init__(self) -> None:
        super().__init__()
        self.expert_replica_routing_table: torch.Tensor | None = None
        self.local_expert_start = 0
        self.local_expert_count = 0

    @classmethod
    def from_upstream(
        cls,
        state: _eplb_state.EplbLayerState,
    ) -> "AscendEplbLayerState":
        ascend_state = cls()
        for field in fields(_eplb_state.EplbLayerState):
            setattr(ascend_state, field.name, getattr(state, field.name))
        if ascend_state.expert_load_view is not None:
            ascend_state._set_local_expert_range(ascend_state.expert_load_view)
        return ascend_state

    def _set_local_expert_range(self, expert_load_view: torch.Tensor) -> None:
        ep_group = get_ep_group()
        num_physical_experts = expert_load_view.shape[-1]
        if num_physical_experts % ep_group.world_size:
            raise ValueError("The number of physical experts must be divisible by EP size")
        self.local_expert_count = num_physical_experts // ep_group.world_size
        self.local_expert_start = ep_group.rank_in_group * self.local_expert_count

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
        self._set_local_expert_range(expert_load_view)
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

    def __init__(self, parallel_config, device: torch.device) -> None:
        super().__init__(parallel_config, device)
        self._has_fresh_recorded_load = False
        self._is_load_sampling_step = False
        self._should_collect_local_load = False
        if self.cuda_device_index is None:
            self.cuda_device_index = torch.accelerator.current_device_index()

    @property
    def uses_custom_load_stats(self) -> bool:
        """Whether the selected policy transforms temporal load samples."""
        return callable(getattr(self.policy, "prepare_local_load_stats", None))

    def add_model(self, model, model_config) -> None:
        """Build the EP-aware layout and initialize custom load statistics."""
        token = EXPERT_MAPPING_EP_SIZE.set(get_ep_group().world_size)
        try:
            super().add_model(model, model_config)
        finally:
            EXPERT_MAPPING_EP_SIZE.reset(token)
        if self.uses_custom_load_stats:
            self._initialize_load_stats_state(
                self.model_states[model_config.compute_hash()]
            )

    def _initialize_load_stats_state(self, model_state: Any) -> None:
        model_state._load_mapping_generation = 0
        model_state._observed_load_mapping_generation = 0
        if hasattr(self, "_local_load_collection_mask"):
            return
        self._local_load_collection_mask = torch.zeros(
            self.expert_load_window_size,
            dtype=torch.int32,
            device="cpu",
        )
        self._physical_load_sample_slots = torch.full(
            (self.expert_load_window_size,),
            -1,
            dtype=torch.long,
            device="cpu",
        )
        self._num_recorded_load_steps = 0
        self._load_stats_window_start_index = 0
        self._load_stats_window_write_index = 0

    def _discard_samples_from_old_mapping(self) -> None:
        mapping_changed = any(
            state._load_mapping_generation
            != state._observed_load_mapping_generation
            for state in self.model_states.values()
        )
        if not mapping_changed:
            return
        self._local_load_collection_mask.zero_()
        self._physical_load_sample_slots.fill_(-1)
        self._num_recorded_load_steps = 0
        self._load_stats_window_start_index = 0
        self._load_stats_window_write_index = 0
        for state in self.model_states.values():
            state._observed_load_mapping_generation = (
                state._load_mapping_generation
            )

    def _ordered_load_step_indices(self) -> torch.Tensor:
        indices = torch.arange(self.expert_load_window_size, dtype=torch.long)
        return (
            indices + self._load_stats_window_start_index
        ) % self.expert_load_window_size

    @staticmethod
    def _map_physical_stats_to_logical(
        model_state: Any,
        physical_stats: PreparedLoadStats,
    ) -> PreparedLoadStats:
        values = physical_stats.values
        num_logical_experts = model_state.model.num_logical_experts
        invalid_expert = torch.full_like(
            model_state.physical_to_logical_map, num_logical_experts
        )
        logical_indices = torch.where(
            model_state.physical_to_logical_map >= 0,
            model_state.physical_to_logical_map,
            invalid_expert,
        ).long()
        logical_values = values.new_zeros(
            (*values.shape[:-1], num_logical_experts + 1)
        )
        logical_values.scatter_add_(
            -1,
            logical_indices.unsqueeze(0).expand(values.shape[0], -1, -1),
            values,
        )
        return PreparedLoadStats(
            logical_values[..., :-1], physical_stats.sample_counts
        )

    def step(
        self,
        is_dummy: bool = False,
        is_profile: bool = False,
        log_stats: bool = False,
    ) -> None:
        """Advance the custom time axis alongside the upstream load window."""
        is_sampling = (
            self._is_load_sampling_step and not is_dummy and not is_profile
        )
        should_collect = self._should_collect_local_load
        self._is_load_sampling_step = False
        self._should_collect_local_load = False
        if self.uses_custom_load_stats:
            self._discard_samples_from_old_mapping()
            if not is_profile:
                index = self._load_stats_window_write_index
                has_sample = is_sampling and should_collect
                self._local_load_collection_mask[index] = has_sample
                self._physical_load_sample_slots[index] = (
                    self.expert_load_window_step if has_sample else -1
                )
                if self._num_recorded_load_steps < self.expert_load_window_size:
                    self._num_recorded_load_steps += 1
                else:
                    self._load_stats_window_start_index = (
                        self._load_stats_window_start_index + 1
                    ) % self.expert_load_window_size
                self._load_stats_window_write_index = (
                    index + 1
                ) % self.expert_load_window_size
        super().step(
            is_dummy=is_dummy,
            is_profile=is_profile,
            log_stats=log_stats,
        )

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
