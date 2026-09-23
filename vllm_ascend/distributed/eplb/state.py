# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

"""Ascend-owned extensions for the upstream EPLB state."""

import inspect
import time
from contextvars import ContextVar
from dataclasses import fields
from typing import Any

import numpy as np
import torch
from torch.distributed import all_reduce
from vllm.distributed import get_ep_group, get_eplb_group
from vllm.distributed.eplb import eplb_state as _eplb_state
from vllm.distributed.eplb.policy import AbstractEplbPolicy
from vllm.distributed.parallel_state import in_the_same_node_as

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

    def __init__(
        self,
        parallel_config,
        device: torch.device,
        policy: AbstractEplbPolicy | None = None,
    ) -> None:
        super().__init__(parallel_config, device)
        self._configured_policy = policy
        if policy is not None:
            self.policy = policy
        self._has_fresh_recorded_load = False
        self._is_load_sampling_step = False
        self._should_collect_local_load = False
        if getattr(self, "cuda_device_index", None) is None:
            self.cuda_device_index = torch.accelerator.current_device_index()

    def add_model(self, model, model_config) -> None:
        """Register a model and initialize policy-specific load history."""
        token = EXPERT_MAPPING_EP_SIZE.set(get_ep_group().world_size)
        try:
            super().add_model(model, model_config)
        finally:
            EXPERT_MAPPING_EP_SIZE.reset(token)
        if self._configured_policy is not None:
            self.policy = self._configured_policy
        model_state = self.model_states[model_config.compute_hash()]
        if not self.uses_custom_load_stats:
            return
        self._initialize_load_stats_state(model_state)

    def _initialize_load_stats_state(self, model_state: Any) -> None:
        model = model_state.model
        # NaN means this layer has no committed mean-ratio anchor yet.
        model_state._last_committed_mean_ratios = np.full(model.num_moe_layers, np.nan)
        model_state._load_mapping_generation = 0
        model_state._observed_load_mapping_generation = 0
        if not hasattr(self, "_local_load_collection_mask"):
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

    def get_rank_node_ids(self) -> np.ndarray:
        """Cache node ordinals in EPLB-rank order as ``[num_ranks]``.

        Ranks detected in the same shared-memory domain share one ordinal.
        """
        rank_node_ids = getattr(self, "_rank_node_ids", None)
        if rank_node_ids is not None:
            return rank_node_ids

        cpu_group = get_eplb_group().cpu_group
        num_ranks = cpu_group.size()
        rank_node_ids = np.full(num_ranks, -1, dtype=np.int64)
        next_node_id = 0
        for source_rank in range(num_ranks):
            if rank_node_ids[source_rank] >= 0:
                continue
            same_node = np.asarray(in_the_same_node_as(cpu_group, source_rank), dtype=bool)
            if same_node.shape != (num_ranks,) or not same_node[source_rank]:
                raise RuntimeError("EPLB node discovery returned an invalid rank mask")
            rank_node_ids[same_node] = next_node_id
            next_node_id += 1
        self._rank_node_ids = rank_node_ids
        return rank_node_ids

    @property
    def uses_custom_load_stats(self) -> bool:
        """Whether the selected policy transforms temporal load samples."""
        return callable(getattr(self.policy, "prepare_local_load_stats", None))

    def step(self, is_dummy: bool = False, is_profile: bool = False, log_stats: bool = False) -> None:
        """Advance the shared collection mask alongside the upstream window."""
        is_load_sampling_step = getattr(self, "_is_load_sampling_step", False) and not is_dummy and not is_profile
        should_collect_local_load = getattr(self, "_should_collect_local_load", False)
        self._is_load_sampling_step = False
        self._should_collect_local_load = False
        if self.uses_custom_load_stats:
            self._discard_samples_from_old_mapping()
            if not is_profile:
                write_index = self._load_stats_window_write_index
                has_sample = is_load_sampling_step and should_collect_local_load
                self._local_load_collection_mask[write_index] = has_sample
                self._physical_load_sample_slots[write_index] = self.expert_load_window_step if has_sample else -1
                if self._num_recorded_load_steps < self.expert_load_window_size:
                    self._num_recorded_load_steps += 1
                else:
                    self._load_stats_window_start_index = (
                        self._load_stats_window_start_index + 1
                    ) % self.expert_load_window_size
                self._load_stats_window_write_index = (write_index + 1) % self.expert_load_window_size
        super().step(is_dummy=is_dummy, is_profile=is_profile, log_stats=log_stats)

    def _discard_samples_from_old_mapping(self) -> None:
        mapping_changed = any(
            model_state._load_mapping_generation != model_state._observed_load_mapping_generation
            for model_state in self.model_states.values()
        )
        if not mapping_changed:
            return
        self._local_load_collection_mask.zero_()
        self._physical_load_sample_slots.fill_(-1)
        self._num_recorded_load_steps = 0
        self._load_stats_window_start_index = 0
        self._load_stats_window_write_index = 0
        for model_state in self.model_states.values():
            model_state._observed_load_mapping_generation = model_state._load_mapping_generation

    def _ordered_load_step_indices(self) -> torch.Tensor:
        indices = torch.arange(self.expert_load_window_size, dtype=torch.long)
        return (indices + self._load_stats_window_start_index) % self.expert_load_window_size

    @staticmethod
    def _map_physical_stats_to_logical(
        model_state: Any,
        physical_stats: PreparedLoadStats,
    ) -> PreparedLoadStats:
        values = physical_stats.values
        num_logical_experts = model_state.model.num_logical_experts
        invalid_expert = torch.full_like(model_state.physical_to_logical_map, num_logical_experts)
        logical_indices = torch.where(
            model_state.physical_to_logical_map >= 0,
            model_state.physical_to_logical_map,
            invalid_expert,
        ).long()
        logical_values = values.new_zeros((*values.shape[:-1], num_logical_experts + 1))
        logical_values.scatter_add_(
            -1,
            logical_indices.unsqueeze(0).expand(values.shape[0], -1, -1),
            values,
        )
        return PreparedLoadStats(logical_values[..., :-1], physical_stats.sample_counts)

    def collect_global_load_stats(self) -> dict[str, PreparedLoadStats] | None:
        """Prepare and reduce policy statistics, or skip an empty global window."""
        prepare_load_stats = getattr(self.policy, "prepare_local_load_stats", None)
        if prepare_load_stats is None:
            raise TypeError("The selected EPLB policy does not prepare custom load statistics")
        self._discard_samples_from_old_mapping()
        if self._num_recorded_load_steps == 0:
            return None
        eplb_group = get_eplb_group()
        step_indices = self._ordered_load_step_indices()
        collecting_rank_counts = self._local_load_collection_mask[step_indices].clone()
        # Rank-local phases may differ; every rank filters the same time axis.
        all_reduce(collecting_rank_counts, group=eplb_group.cpu_group)
        included_sample_mask = collecting_rank_counts > 0
        if not included_sample_mask.any():
            return None
        local_stats = {}
        for model_key, model_state in self.model_states.items():
            included_steps = step_indices[included_sample_mask]
            physical_slots = self._physical_load_sample_slots[included_steps]
            physical_samples = model_state.expert_load_window.new_zeros(
                (included_steps.numel(), *model_state.expert_load_window.shape[1:])
            )
            local_sample_mask = physical_slots >= 0
            if local_sample_mask.any():
                device_mask = local_sample_mask.to(physical_samples.device)
                device_slots = physical_slots[local_sample_mask].to(physical_samples.device)
                physical_samples[device_mask] = model_state.expert_load_window.index_select(0, device_slots)
            physical_stats = prepare_load_stats(physical_samples)
            local_stats[model_key] = self._map_physical_stats_to_logical(model_state, physical_stats)
        flat_values = [stats.values.reshape(-1, stats.values.shape[-1]) for stats in local_stats.values()]
        shapes = [values.shape for values in flat_values]
        concatenated = torch.cat(flat_values, dim=0)
        all_reduce(concatenated, group=eplb_group.device_group)
        global_values = list(concatenated.split([shape[0] for shape in shapes]))
        return {
            model_key: PreparedLoadStats(global_values[index].reshape(stats.values.shape), stats.sample_counts)
            for index, (model_key, stats) in enumerate(local_stats.items())
        }

    def publish_async_load_stats(self, global_load_stats: dict[str, PreparedLoadStats]) -> None:
        """Publish one complete statistics snapshot to the async planner."""
        if global_load_stats.keys() != self.model_states.keys():
            raise ValueError("Load statistics must contain exactly one entry per EPLB model")
        num_ranks = get_ep_group().device_group.size()
        rank_node_ids = self.get_rank_node_ids()
        num_nodes = len(np.unique(rank_node_ids))
        for model_key, model_state in self.model_states.items():
            load_stats = global_load_stats[model_key]
            model_state._policy_load_stats = load_stats
            model = model_state.model
            model_state.eplb_stats = _eplb_state.EplbStats(
                global_expert_load_window=load_stats.values,
                num_replicas=model.num_physical_experts,
                num_groups=model.num_expert_groups,
                num_nodes=num_nodes,
                num_gpus=num_ranks,
            )
        # Publish only after every model's statistics are ready.
        for model_state in self.model_states.values():
            model_state.rebalanced = True
        self.rearrange_event.record()

    def _has_global_fresh_recorded_load(self) -> bool:
        """Synchronize whether any EP rank recorded load since rearranging."""
        eplb_group = get_eplb_group()
        cpu_group = getattr(eplb_group, "cpu_group", None)
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

        device_group = eplb_group.device_group
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
        use_custom_async_stats = (
            self.is_async and not is_profile and rank_mapping is None and self.uses_custom_load_stats
        )
        should_gate = (
            use_custom_async_stats
            and hasattr(self, "_has_fresh_recorded_load")
            and not is_profile
            and rank_mapping is None
            and not self.parallel_config.enable_elastic_ep
        )
        if should_gate and not self._has_global_fresh_recorded_load():
            return None

        if use_custom_async_stats:
            global_load_stats = self.collect_global_load_stats()
            if global_load_stats is not None:
                self.publish_async_load_stats(global_load_stats)
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

    def drain_async(self) -> None:
        """Acknowledge all in-flight layer results, including no-op cycles."""
        if not self.is_async:
            return
        for model_state in self.model_states.values():
            while model_state.rebalanced:
                if self._all_ranks_result_ready(model_state):
                    result = model_state.pending_result
                    assert result is not None
                    if getattr(result, "is_last_result", result.layer_idx == model_state.model.num_moe_layers - 1):
                        model_state.rebalanced = False
                    model_state.pending_result = None
                    result.consumed_event.record()
                else:
                    time.sleep(0.001)

    @classmethod
    def from_mapping(
        cls,
        model,
        model_config,
        device: torch.device,
        parallel_config,
        expanded_physical_to_logical: torch.Tensor,
        num_valid_physical_experts: int | None = None,
        policy: AbstractEplbPolicy | None = None,
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
        state._configured_policy = policy
        if policy is not None:
            state.policy = policy
        if state.uses_custom_load_stats:
            for model_state in state.model_states.values():
                state._initialize_load_stats_state(model_state)
        for model_state in state.model_states.values():
            refresh_model_routing_tables(model_state)
        return state
