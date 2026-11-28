# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

"""Ascend-owned extensions for the upstream EPLB state."""

import inspect
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager, suppress
from contextvars import ContextVar
from dataclasses import fields
from datetime import timedelta
from typing import Any

import numpy as np
import torch
from torch.distributed import ReduceOp, all_gather_object, all_reduce
from vllm.config.utils import hash_config_cached
from vllm.distributed import get_ep_group, get_eplb_group
from vllm.distributed import parallel_state as _parallel_state
from vllm.distributed.eplb import eplb_state as _eplb_state
from vllm.distributed.eplb.policy import AbstractEplbPolicy
from vllm.distributed.parallel_state import (
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    in_the_same_node_as,
)
from vllm.forward_context import get_forward_context
from vllm.logger import logger
from vllm.utils.gpu_sync_debug import gpu_sync_allowed
from vllm.utils.network_utils import get_ip
from vllm.v1.worker.ubatching import dbo_current_ubatch_id

from vllm_ascend.device.hardware_profile import MoECommPolicy, get_current_hardware_profile
from vllm_ascend.distributed.eplb.eplb_communicator import AscendHixlEplbCommunicator, _HixlSession
from vllm_ascend.distributed.eplb.placement import PlacementExporter
from vllm_ascend.distributed.eplb.policy import PreparedLoadStats
from vllm_ascend.distributed.eplb.tensor_ops import (
    build_expert_replica_routing_table,
    count_allgather_tokens,
    make_allgather_mask,
)

ASYNC_EPLB_CYCLE_COMMITTED_LOG = "Ascend async EPLB cycle committed"
_EPLB_CLOSE_TIMEOUT_SECONDS = 300
_EPLB_EVENT_POLL_SECONDS = 0.1
EXPERT_MAPPING_EP_SIZE: ContextVar[int] = ContextVar("vllm_ascend_expert_mapping_ep_size", default=1)
# ForwardContext.additional_kwargs key holding one gathered AllGather source
# mask per micro-batch. Read directly rather than through _EXTRA_CTX: the
# proxy cannot create the list atomically, which DBO needs.
_ALLGATHER_SOURCE_MASK_SLOTS = "eplb_allgather_source_mask_slots"


@contextmanager
def _configured_upstream_policy(name: str, policy: AbstractEplbPolicy | None) -> Iterator[None]:
    """Expose an Ascend policy only while upstream initializes its state."""
    policies: dict[str, Any] = _eplb_state.EPLB_POLICIES
    if policy is None or name in policies:
        yield
        return
    policies[name] = policy
    try:
        yield
    finally:
        policies.pop(name)


def _upstream_from_mapping_accepts_valid_expert_count() -> bool:
    """Return whether the selected vLLM uses the release mapping contract."""
    return "num_valid_physical_experts" in inspect.signature(_eplb_state.EplbState.from_mapping).parameters


def _get_pp_topology() -> tuple[int, int, Any | None]:
    if getattr(_parallel_state, "_PP", None) is None and get_pp_group is _parallel_state.get_pp_group:
        return 0, 1, None
    try:
        pp_group = get_pp_group()
    except AssertionError:
        return 0, 1, None
    return pp_group.rank_in_group, pp_group.world_size, getattr(pp_group, "cpu_group", None)


class AscendEplbLayerState(_eplb_state.EplbLayerState):
    """EPLB layer state with a graph-stable replica routing table."""

    def __init__(self) -> None:
        super().__init__()
        self.expert_replica_routing_table: torch.Tensor | None = None
        self.allgather_expert_replica_routing_table: torch.Tensor | None = None
        self.allgather_record_tensor: torch.Tensor | None = None
        self.allgather_source_is_padding: torch.Tensor | None = None
        self.local_expert_start = 0
        self.local_expert_count = 0
        # Static mode keeps this disabled except during a scheduled observation.
        self.load_recording_enabled = True

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

    def record_load(
        self,
        expert_tokens: torch.Tensor | None,
        group_list_type: int,
        *,
        topk_ids: torch.Tensor,
        num_source_tokens: int,
        is_sequence_parallel: bool = False,
        all_gather_mask: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> None:
        """Record MRV2 load, constructing and gathering source eligibility when needed."""
        if not self.load_recording_enabled:
            return
        local_load = None
        if all_gather_mask is not None:
            assert self.num_unpadded_tokens_tensors is not None and self.should_record_tensor is not None
            num_unpadded_tokens = self.num_unpadded_tokens_tensors[dbo_current_ubatch_id()]
            # Every MoE layer of this model derives the same mask from the same
            # per-forward inputs, so gathering it in each one repeats a
            # collective as many times as the model has MoE layers. Hold it on
            # the forward context, which lives exactly as long as the mask is
            # valid: warmup, graph capture and draft forwards each enter their
            # own, and none of them passes through the EPLB controller.
            # ``setdefault`` because DBO runs its micro-batches on two threads
            # that share one context, and a lost list would leave the ranks
            # gathering a different number of times.
            ubatch = dbo_current_ubatch_id()
            forward_context = get_forward_context()
            slots = forward_context.additional_kwargs.setdefault(
                _ALLGATHER_SOURCE_MASK_SLOTS,
                [None] * len(self.num_unpadded_tokens_tensors),
            )
            token_mask = slots[ubatch]
            if token_mask is None:
                is_padding = self.allgather_source_is_padding
                if is_padding is None:
                    is_padding = getattr(forward_context, "is_padding", None)
                token_mask = make_allgather_mask(
                    num_source_tokens,
                    num_unpadded_tokens,
                    self.should_record_tensor,
                    is_padding=is_padding,
                    tp_size=get_tensor_model_parallel_world_size() if is_sequence_parallel else 1,
                    tp_rank=get_tensor_model_parallel_rank() if is_sequence_parallel else 0,
                )
                token_mask = all_gather_mask(token_mask)
                slots[ubatch] = token_mask
            local_load = count_allgather_tokens(topk_ids, token_mask, self.local_expert_start, self.local_expert_count)
            assert self.allgather_record_tensor is not None
            torch.maximum(
                self.allgather_record_tensor,
                token_mask.any().to(torch.int32),
                out=self.allgather_record_tensor,
            )
        if expert_tokens is None:
            raise RuntimeError("MRV2 EPLB requires operator-provided expert counts")
        expert_load_view = self.expert_load_view

        record_enabled = self.should_record_tensor
        if local_load is not None:
            expert_tokens = local_load
            assert self.allgather_record_tensor is not None
            record_enabled = self.allgather_record_tensor > 0
            group_list_type = 1
        if expert_load_view is None or record_enabled is None:
            raise RuntimeError("MRV2 EPLB load-recording state is not initialized")
        if expert_tokens.numel() != self.local_expert_count:
            raise RuntimeError(
                "MoE expert count does not match local EPLB experts: "
                f"{expert_tokens.numel()} != {self.local_expert_count}"
            )
        torch.ops.vllm.ascend_eplb_record_expert_tokens(
            expert_tokens,
            expert_load_view,
            record_enabled,
            group_list_type,
            self.local_expert_start,
        )

    def refresh_expert_replica_routing_table(self) -> None:
        logical_to_physical_map = self.logical_to_physical_map
        logical_replica_count = self.logical_replica_count
        if logical_to_physical_map is None or logical_replica_count is None:
            raise RuntimeError("Cannot build the replica routing table before EPLB layer state is initialized.")

        new_routing_table = build_expert_replica_routing_table(
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
        if get_current_hardware_profile().moe_comm_policy is MoECommPolicy.CAPACITY_AND_WORLD_SIZE:
            common_table = build_expert_replica_routing_table(logical_to_physical_map, logical_replica_count, ep_rank=0)
            if self.allgather_expert_replica_routing_table is None:
                self.allgather_expert_replica_routing_table = common_table
            else:
                self.allgather_expert_replica_routing_table.copy_(common_table, non_blocking=True)


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
    if hasattr(model_state, "_allgather_physical_map"):
        current_map = model_state.physical_to_logical_map
        # Async layer refreshes follow changed-layer commits. Comparing the
        # device maps here would synchronize the foreground inference stream.
        if layer_idx is not None or not torch.equal(current_map, model_state._allgather_physical_map):
            if model_state._allgather_physical_map.shape == current_map.shape:
                model_state._allgather_physical_map.copy_(current_map)
            else:
                model_state._allgather_physical_map = current_map.clone()
            model_state.expert_load_pass.zero_()
            model_state.expert_load_window.zero_()
            if hasattr(model_state, "_load_mapping_generation"):
                model_state._load_mapping_generation += 1


def _raise_if_async_worker_stopped(state: Any) -> None:
    worker = getattr(state, "async_worker", None)
    if worker is not None and not worker.is_alive():
        error = RuntimeError("EPLB background worker terminated unexpectedly")
        session = getattr(state, "_hixl_session", None)
        if session is not None:
            # Device/stream setup can fail before the migration loop is entered.
            session._error = session._error or error
            session._usable = False
            state.raise_if_close_failed()
        raise error


class AscendEplbState(_eplb_state.EplbState):
    """Keep Ascend routing and load-recording state around upstream EPLB."""

    cuda_device_index: int | None
    device_index: int | None
    async_worker: threading.Thread | None

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
        self.allgather_record_tensor: torch.Tensor | None = None
        self._uses_phase_filter = False
        self._stop_async = threading.Event()
        self._suspended = False
        self._close_error: Exception | None = None
        self._hixl_session: _HixlSession | None = None
        self._exporter: PlacementExporter | None = None
        self._placement_model_roles: dict[str, str] = {}
        self._last_included_load_steps: torch.Tensor | None = None
        # vLLM main renamed ``EplbState.cuda_device_index`` to ``device_index``
        # (and passes it to EplbModelState on add_model), so keep both lanes
        # pointing at the attribute upstream actually reads.
        if self.device_index is None:
            self.device_index = torch.accelerator.current_device_index()
        self.static_mode = False

    @property
    def uses_custom_load_stats(self) -> bool:
        """Whether the selected policy transforms temporal load samples."""
        return callable(getattr(getattr(self, "policy", None), "prepare_local_load_stats", None))

    @property
    def _tracks_load_sample_axis(self) -> bool:
        return self.uses_custom_load_stats or getattr(self, "_exporter", None) is not None

    def _add_static_model(self, model, model_config, model_name: str | None = None) -> None:
        self.validate_ep_configuration(model)
        self.is_async = False
        role_placement = getattr(model, "_static_role_placement", None)
        if role_placement is not None:
            physical_to_logical_map = torch.tensor(
                role_placement.physical_to_logical_map,
                dtype=torch.long,
                device=self.device,
            )
        else:
            token = EXPERT_MAPPING_EP_SIZE.set(get_ep_group().world_size)
            try:
                initial = _eplb_state.EplbState.build_initial_global_physical_to_logical_map(
                    model.num_routed_experts,
                    model.num_redundant_experts,
                )
            finally:
                EXPERT_MAPPING_EP_SIZE.reset(token)
            physical_to_logical_map = (
                torch.tensor(initial, dtype=torch.long, device=self.device)
                .unsqueeze(0)
                .expand(model.num_moe_layers, -1)
                .contiguous()
            )
        logical_cpu, replicas_cpu = _eplb_state.compute_logical_maps(
            physical_to_logical_map.cpu(),
            model.num_logical_experts,
        )
        padding = (model.num_redundant_experts + 1) - logical_cpu.shape[-1]
        if padding < 0:
            raise ValueError("Static EPLB placement exceeds the allocated replica capacity")
        if padding:
            logical_cpu = torch.nn.functional.pad(logical_cpu, (0, padding), value=-1)
        logical_to_physical_map = logical_cpu.to(device=self.device, dtype=torch.long)
        logical_replica_count = replicas_cpu.to(device=self.device, dtype=torch.long)
        expert_load_pass = torch.zeros(
            (model.num_moe_layers, model.num_physical_experts),
            dtype=torch.int32,
            device=self.device,
        )
        model.expert_weights = []
        for layer_idx, layer in enumerate(model.moe_layers):
            layer.set_eplb_state(
                moe_layer_idx=layer_idx,
                expert_load_view=expert_load_pass,
                logical_to_physical_map=logical_to_physical_map,
                logical_replica_count=logical_replica_count,
            )
            if isinstance(layer.eplb_state, _eplb_state.EplbLayerState) and not isinstance(
                layer.eplb_state, AscendEplbLayerState
            ):
                layer.eplb_state = AscendEplbLayerState.from_upstream(layer.eplb_state)
                layer.eplb_state.refresh_expert_replica_routing_table()
            if layer.eplb_state is not None:
                layer.eplb_state.load_recording_enabled = False
        empty_communicator: Any = None
        self.model_states[hash_config_cached(model_config)] = _eplb_state.EplbModelState(
            physical_to_logical_map=physical_to_logical_map,
            physical_to_logical_map_buffer=physical_to_logical_map,
            logical_to_physical_map=logical_to_physical_map,
            logical_replica_count=logical_replica_count,
            expert_load_pass=expert_load_pass,
            expert_load_pass_buffer=expert_load_pass,
            expert_load_window=torch.empty(
                (0, model.num_moe_layers, model.num_physical_experts),
                dtype=torch.int32,
                device=self.device,
            ),
            last_expert_load=torch.empty((0, model.num_logical_experts), dtype=torch.int64, device=self.device),
            model_name=model_name or model_config.model,
            model=model,
            expert_buffer=[],
            rebalanced=False,
            eplb_stats=None,
            device_index=self.device_index,
            communicator=empty_communicator,
            num_unpadded_tokens_tensors=None,
        )

    def add_model(self, model, model_config, model_name: str | None = None) -> None:
        """Build the EP-aware layout and initialize custom load statistics."""
        if getattr(self, "static_mode", False):
            self._add_static_model(model, model_config, model_name=model_name)
            return
        if self._hixl_session is not None:
            raise RuntimeError("Cannot add EPLB models after HIXL initialization")
        token = EXPERT_MAPPING_EP_SIZE.set(get_ep_group().world_size)
        try:
            with _configured_upstream_policy(
                self.parallel_config.eplb_config.policy,
                self._configured_policy,
            ):
                super().add_model(model, model_config, model_name=model_name)
        finally:
            EXPERT_MAPPING_EP_SIZE.reset(token)
        # Snapshot the placement for every profile, not just AllGather. The load
        # history is keyed by physical slot, so a committed rearrangement that
        # moves a different logical expert into a slot invalidates it, and
        # refresh_model_routing_tables needs the previous placement to notice.
        # Seeding here rather than on first use also covers the first commit,
        # which already carries the new mapping by the time the tables refresh.
        key = hash_config_cached(model_config)
        model_state = self.model_states[key]
        model_state._allgather_physical_map = model_state.physical_to_logical_map.clone()
        if self._tracks_load_sample_axis:
            self._initialize_load_stats_state(model_state)
        role = "draft" if model_name is not None else "target"
        roles = getattr(self, "_placement_model_roles", None)
        if roles is None:
            roles = {}
            self._placement_model_roles = roles
        roles[key] = role
        exporter = getattr(self, "_exporter", None)
        if exporter is not None and exporter.is_stage_leader:
            with gpu_sync_allowed():
                model_state._placement_export_map = model_state.physical_to_logical_map.to("cpu", copy=True)
            model_state._on_placement_layer_committed = lambda r=role: exporter.mark_committed(r)

    def enable_placement_export(self, path: str) -> None:
        if getattr(self, "model_states", None):
            raise RuntimeError("EPLB placement export must be configured before registering models.")
        if not torch.distributed.is_initialized():
            raise RuntimeError("EPLB placement export requires initialized distributed worker groups.")
        ep_group = get_ep_group()
        pp_rank, pp_size, _ = _get_pp_topology()
        if ep_group.world_size * pp_size != torch.distributed.get_world_size():
            raise ValueError("EPLB placement export requires one EP group per PP stage covering all worker ranks.")
        self._placement_model_roles = {}
        self._exporter = PlacementExporter(
            ep_size=ep_group.world_size,
            pp_rank=pp_rank,
            pp_size=pp_size,
            is_stage_leader=ep_group.rank_in_group == 0,
            path=path,
        )

    def _get_role_model_state(self, role: str) -> Any | None:
        roles = getattr(self, "_placement_model_roles", {})
        return next((ms for key, ms in self.model_states.items() if roles.get(key) == role), None)

    def _build_local_role_entry(self, role: str) -> dict[str, Any] | None:
        ms = self._get_role_model_state(role)
        export_map = None if ms is None else getattr(ms, "_placement_export_map", None)
        if ms is None or export_map is None:
            return None
        num_logical = int(ms.model.num_logical_experts)
        return {
            "num_logical_experts": num_logical,
            "num_redundant_experts": int(ms.model.num_physical_experts) - num_logical,
            "physical_to_logical_map": export_map.tolist(),
        }

    def setup_pp_placement_export(self) -> None:
        exporter = getattr(self, "_exporter", None)
        if exporter is None or not exporter.is_stage_leader:
            return
        pp_rank, pp_size, pp_cpu_group = _get_pp_topology()
        if pp_size <= 1 or exporter._listener is not None or exporter._endpoint is not None:
            return
        if pp_cpu_group is None:
            raise RuntimeError("PP CPU group is required for multi-stage EPLB placement export.")
        endpoint = exporter.start_pp_listener(get_ip()) if pp_rank == 0 else None
        local_stage = {
            "pp_rank": pp_rank,
            "target": self._build_local_role_entry("target"),
            "draft": self._build_local_role_entry("draft"),
            "endpoint": endpoint,
        }
        gathered: list[Any] = [None] * pp_size
        all_gather_object(gathered, local_stage, group=pp_cpu_group)
        same_node = False
        with suppress(Exception):
            same_node_flags = in_the_same_node_as(pp_cpu_group, 0)
            same_node = pp_rank != 0 and bool(same_node_flags[pp_rank])
        exporter.configure_pp_stages(gathered, same_node)

    def _compute_model_window_ratio_stats(
        self,
        model_state: Any | None,
        included_steps: torch.Tensor | None = None,
    ) -> tuple[float, int] | None:
        window = None if model_state is None else getattr(model_state, "expert_load_window", None)
        if window is None:
            return None
        ep_group = get_ep_group()
        ep_size = ep_group.world_size
        if ep_size < 1 or window.shape[-1] % ep_size != 0:
            return None
        slots = window.shape[-1] // ep_size
        local_slice = slice(ep_group.rank_in_group * slots, (ep_group.rank_in_group + 1) * slots)
        if included_steps is not None:
            if included_steps.numel() == 0:
                return None
            phys_slots = self._physical_load_sample_slots[included_steps]
            local_load = torch.zeros(
                (included_steps.numel(), window.shape[1]), dtype=torch.float32, device=window.device
            )
            mask = phys_slots >= 0
            if mask.any():
                selected = window.index_select(0, phys_slots[mask].to(window.device))[:, :, local_slice]
                local_load[mask.to(window.device)] = selected.sum(dim=-1, dtype=torch.float32)
        else:
            local_load = window[:, :, local_slice].sum(dim=-1, dtype=torch.float32)
        if ep_size == 1:
            with gpu_sync_allowed():
                count = int((local_load > 0).sum().item())
            return (float(count), count) if count > 0 else None
        eplb_group = get_eplb_group()
        group = (
            eplb_group.cpu_group
            if local_load.is_cpu and getattr(eplb_group, "cpu_group", None) is not None
            else eplb_group.device_group
        )
        max_load, sum_load = local_load.clone(), local_load.clone()
        all_reduce(max_load, op=ReduceOp.MAX, group=group)
        all_reduce(sum_load, op=ReduceOp.SUM, group=group)
        mean_load = sum_load / float(ep_size)
        valid = mean_load > 0
        ratios = torch.where(valid, max_load.double() / mean_load.double().clamp_min(1e-12), 0.0)
        with gpu_sync_allowed():
            ratio_sum, valid_count = torch.stack((ratios.sum(), valid.sum().double())).tolist()
        count = int(valid_count)
        return (float(ratio_sum), count) if count > 0 else None

    def _maybe_evaluate_and_export_placement(self, included_steps: torch.Tensor | None = None) -> None:
        exporter = getattr(self, "_exporter", None)
        if exporter is None:
            return
        target_stats = self._compute_model_window_ratio_stats(self._get_role_model_state("target"), included_steps)
        draft_stats = self._compute_model_window_ratio_stats(self._get_role_model_state("draft"), included_steps)
        if target_stats is None and draft_stats is None:
            return
        if not exporter.is_stage_leader:
            if exporter.pp_size == 1 and target_stats is not None:
                ratio = target_stats[0] / target_stats[1]
                if exporter.min_peak_to_mean_ratio is None or ratio < exporter.min_peak_to_mean_ratio:
                    exporter.min_peak_to_mean_ratio = ratio
            return
        try:
            exporter.publish_evaluation(
                target_stats=target_stats,
                target_entry=self._build_local_role_entry("target"),
                draft_stats=draft_stats,
                draft_entry=self._build_local_role_entry("draft"),
            )
        except (OSError, EOFError, ValueError):
            logger.exception("Could not export committed EPLB placement")

    def _propagate_shared_tensors(self, model, num_unpadded_tokens_tensors) -> None:
        super()._propagate_shared_tensors(model, num_unpadded_tokens_tensors)
        if get_current_hardware_profile().moe_comm_policy is not MoECommPolicy.CAPACITY_AND_WORLD_SIZE:
            return
        if self.should_record_tensor is None:
            return
        if self.allgather_record_tensor is None:
            self.allgather_record_tensor = torch.full((), -1, dtype=torch.int32, device=self.device)
        for layer in model.moe_layers:
            if isinstance(layer.eplb_state, AscendEplbLayerState):
                layer.eplb_state.allgather_record_tensor = self.allgather_record_tensor

    def wait_for_rearrangement(self, stream) -> bool:
        """Allow an idle worker to stop without recording a fake device event."""
        while not self.rearrange_event._recorded.wait(_EPLB_EVENT_POLL_SECONDS):
            if self._stop_async.is_set():
                return False
        self.rearrange_event.wait(stream=stream)
        return True

    def close(self) -> None:
        """Close EPLB while its CPU groups and NPU mappings are still alive."""
        self.raise_if_close_failed()
        try:
            self._suspended = True
            worker = self.async_worker
            if worker is not None:
                _raise_if_async_worker_stopped(self)
                # Acknowledge real completed results before stopping the idle loop.
                # This also handles the worker waiting for consumed_event.
                self.drain_async()
                self._stop_async.set()
                worker.join(timeout=_EPLB_CLOSE_TIMEOUT_SECONDS)
                if worker.is_alive():
                    raise TimeoutError("EPLB background worker could not be stopped; worker must terminate")
                self.async_worker = None
            exporter = getattr(self, "_exporter", None)
            if exporter is not None:
                exporter.close()
            if self._hixl_session is not None:
                self._hixl_session.close()
        except Exception as error:
            self._close_error = error
            self.raise_if_close_failed()

    def raise_if_close_failed(self) -> None:
        if self._hixl_session is not None and self._hixl_session._error is not None:
            self._close_error = self._hixl_session._error
        if self._close_error is not None:
            logger.error("EPLB cleanup failed; terminating the worker: %s", self._close_error)
            # Ordinary RPC exceptions are caught by the upstream worker loop.
            # SystemExit reaches its process-death monitor and engine shutdown.
            raise SystemExit(
                "EPLB cleanup failed; worker must terminate before releasing registered memory"
            ) from self._close_error

    def start_async_loop(self, *args, **kwargs) -> None:
        if not self._suspended and not getattr(self, "static_mode", False):
            communicators = {
                key: model_state.communicator
                for key, model_state in self.model_states.items()
                if isinstance(model_state.communicator, AscendHixlEplbCommunicator)
            }
            if communicators and self._hixl_session is None:
                self._hixl_session = _HixlSession(next(iter(communicators.values()))._cpu_group)
                self._hixl_session.initialize(communicators)
            if self._hixl_session is not None:
                self._hixl_session.check_usable()
            super().start_async_loop(*args, **kwargs)

    def prepare_forward(
        self,
        model_config,
        num_unpadded_tokens: int,
        ubatch_slices: list | None = None,
    ) -> None:
        if getattr(self, "static_mode", False):
            return
        super().prepare_forward(model_config, num_unpadded_tokens, ubatch_slices)

    def _initialize_load_stats_state(self, model_state: Any) -> None:
        model_state._last_committed_mean_ratios = np.full(
            model_state.model.num_moe_layers,
            np.nan,
        )
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

    def get_rank_node_ids(self) -> np.ndarray:
        """Return cached node ordinals in stage-local EPLB rank order."""
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
            same_node = np.asarray(
                in_the_same_node_as(cpu_group, source_rank),
                dtype=bool,
            )
            if same_node.shape != (num_ranks,) or not same_node[source_rank]:
                raise RuntimeError("EPLB node discovery returned an invalid rank mask")
            rank_node_ids[same_node] = next_node_id
            next_node_id += 1
        self._rank_node_ids = rank_node_ids
        return rank_node_ids

    def _discard_samples_from_old_mapping(self) -> None:
        mapping_changed = any(
            state._load_mapping_generation != state._observed_load_mapping_generation
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
            state._observed_load_mapping_generation = state._load_mapping_generation

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
        ).to(device=values.device, dtype=torch.long)
        logical_values = values.new_zeros((*values.shape[:-1], num_logical_experts + 1))
        if values.ndim > logical_indices.ndim:
            logical_indices = logical_indices.unsqueeze(0).expand(values.shape[0], -1, -1)
        logical_values.scatter_add_(
            -1,
            logical_indices,
            values,
        )
        return PreparedLoadStats(logical_values[..., :-1], physical_stats.sample_counts)

    def step(
        self,
        is_dummy: bool = False,
        is_profile: bool = False,
        log_stats: bool = False,
    ) -> None:
        """Advance the custom time axis alongside the upstream load window."""
        if getattr(self, "static_mode", False):
            return
        self.raise_if_close_failed()
        if self._hixl_session is not None:
            _raise_if_async_worker_stopped(self)
        is_sampling = getattr(self, "_is_load_sampling_step", False) and not is_profile
        should_collect = getattr(self, "_should_collect_local_load", False)
        allgather_record = getattr(self, "allgather_record_tensor", None)
        if is_sampling and allgather_record is not None:
            if is_dummy or getattr(self, "_uses_phase_filter", False):
                # A source-idle owner can still execute remote tokens. The
                # device decision also preserves phase filtering under replay.
                gathered = int(allgather_record.item())
                if gathered >= 0:
                    should_collect = gathered != 0
                    if should_collect:
                        is_dummy = False
            if not is_dummy and should_collect:
                self._has_fresh_recorded_load = True
        is_sampling = is_sampling and not is_dummy
        self._is_load_sampling_step = False
        self._should_collect_local_load = False
        if self._tracks_load_sample_axis:
            self._discard_samples_from_old_mapping()
            if not is_profile:
                index = self._load_stats_window_write_index
                has_sample = is_sampling and should_collect
                self._local_load_collection_mask[index] = has_sample
                self._physical_load_sample_slots[index] = self.expert_load_window_step if has_sample else -1
                if self._num_recorded_load_steps < self.expert_load_window_size:
                    self._num_recorded_load_steps += 1
                else:
                    self._load_stats_window_start_index = (
                        self._load_stats_window_start_index + 1
                    ) % self.expert_load_window_size
                self._load_stats_window_write_index = (index + 1) % self.expert_load_window_size
        super().step(
            is_dummy=is_dummy,
            is_profile=is_profile,
            log_stats=log_stats,
        )

    def _sync_load_pass(self, *args: Any, **kwargs: Any) -> list[torch.Tensor]:
        """Observability hook point; upstream owns the behaviour.

        Upstream ``EplbState.step`` calls this only on balancedness logging
        steps and uses the returned, already reduced load pass. The metric
        provider can only hook ``vllm_ascend`` symbols, so this pass-through
        override exists solely to expose that call point to it.
        """
        return super()._sync_load_pass(*args, **kwargs)

    def _resolve_included_load_steps(self) -> torch.Tensor | None:
        if not hasattr(self, "_local_load_collection_mask"):
            return None
        self._discard_samples_from_old_mapping()
        if self._num_recorded_load_steps == 0:
            return torch.empty(0, dtype=torch.long)
        group = get_eplb_group()
        step_indices = self._ordered_load_step_indices()
        rank_counts = self._local_load_collection_mask[step_indices].clone()
        cpu_group = getattr(group, "cpu_group", None)
        if cpu_group is not None:
            all_reduce(rank_counts, group=cpu_group)
        else:
            dev_counts = rank_counts.to(self.device)
            all_reduce(dev_counts, group=group.device_group)
            with gpu_sync_allowed():
                rank_counts = dev_counts.cpu()
        return step_indices[rank_counts > 0]

    def collect_global_load_stats(
        self,
    ) -> dict[str, PreparedLoadStats] | None:
        """Prepare and reduce policy statistics on a shared time axis."""
        prepare_load_stats = getattr(self.policy, "prepare_local_load_stats", None)
        if prepare_load_stats is None:
            raise TypeError("The selected EPLB policy does not prepare load statistics")
        self._last_included_load_steps = None
        included_steps = self._resolve_included_load_steps()
        if included_steps is None or included_steps.numel() == 0:
            return None
        self._last_included_load_steps = included_steps
        group = get_eplb_group()

        local_stats: dict[str, PreparedLoadStats] = {}
        for model_key, model_state in self.model_states.items():
            physical_slots = self._physical_load_sample_slots[included_steps]
            physical_samples = model_state.expert_load_window.new_zeros(
                (
                    included_steps.numel(),
                    *model_state.expert_load_window.shape[1:],
                )
            )
            local_mask = physical_slots >= 0
            if local_mask.any():
                device_mask = local_mask.to(physical_samples.device)
                device_slots = physical_slots[local_mask].to(physical_samples.device)
                physical_samples[device_mask] = model_state.expert_load_window.index_select(0, device_slots)
            physical_stats = prepare_load_stats(physical_samples)
            local_stats[model_key] = self._map_physical_stats_to_logical(model_state, physical_stats)

        flat_values = [stats.values.reshape(-1, stats.values.shape[-1]) for stats in local_stats.values()]
        row_counts = [values.shape[0] for values in flat_values]
        reduced = torch.cat(flat_values)
        all_reduce(reduced, group=group.device_group)
        split_values = reduced.split(row_counts)
        return {
            model_key: PreparedLoadStats(
                split_values[index].reshape(stats.values.shape),
                stats.sample_counts,
            )
            for index, (model_key, stats) in enumerate(local_stats.items())
        }

    def publish_async_load_stats(self, global_load_stats: dict[str, PreparedLoadStats]) -> None:
        """Publish one complete statistics snapshot to the async worker."""
        if global_load_stats.keys() != self.model_states.keys():
            raise ValueError("Load statistics must contain exactly one entry per EPLB model")
        num_gpus = get_eplb_group().device_group.size()
        num_nodes = len(np.unique(self.get_rank_node_ids()))
        for model_key, model_state in self.model_states.items():
            load_stats = global_load_stats[model_key]
            model_state._policy_load_stats = load_stats
            model = model_state.model
            model_state.eplb_stats = _eplb_state.EplbStats(
                global_expert_load_window=load_stats.values,
                num_replicas=model.num_physical_experts,
                num_groups=model.num_expert_groups,
                num_nodes=num_nodes,
                num_gpus=num_gpus,
            )
        for model_state in self.model_states.values():
            model_state.rebalanced = True
        self.rearrange_event.record()

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
        if self._suspended or getattr(self, "static_mode", False):
            return None
        use_custom_async_stats = (
            self.is_async and not is_profile and rank_mapping is None and self.uses_custom_load_stats
        )
        should_gate = (
            hasattr(self, "_has_fresh_recorded_load")
            and not is_profile
            and rank_mapping is None
            and not self.parallel_config.enable_elastic_ep
        )
        if should_gate and not self._has_global_fresh_recorded_load():
            return None

        if use_custom_async_stats:
            self._last_included_load_steps = None
            global_load_stats = self.collect_global_load_stats()
            if global_load_stats is not None:
                if getattr(self, "_exporter", None) is not None:
                    self._maybe_evaluate_and_export_placement(included_steps=self._last_included_load_steps)
                self.publish_async_load_stats(global_load_stats)
            result = None
        else:
            if getattr(self, "_exporter", None) is not None and not is_profile and rank_mapping is None:
                self._maybe_evaluate_and_export_placement(included_steps=self._resolve_included_load_steps())
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
        """Acknowledge changed-layer and no-op results through one lifecycle."""
        if not self.is_async:
            return
        deadline = time.monotonic() + _EPLB_CLOSE_TIMEOUT_SECONDS
        for model_state in self.model_states.values():
            work = getattr(model_state, "_eplb_ready_work", None)
            if work is not None:
                work.wait(timeout=timedelta(seconds=_EPLB_CLOSE_TIMEOUT_SECONDS))
                del model_state._eplb_ready_work
                del model_state._eplb_ready_flag
            while model_state.rebalanced:
                _raise_if_async_worker_stopped(self)
                if time.monotonic() >= deadline:
                    raise TimeoutError("EPLB migration could not be drained; worker must terminate")
                result = model_state.pending_result
                if result is not None:
                    if getattr(
                        result,
                        "is_last_result",
                        result.layer_idx == model_state.model.num_moe_layers - 1,
                    ):
                        model_state.rebalanced = False
                    model_state.pending_result = None
                    result.consumed_event.record()
                else:
                    time.sleep(0.001)

    def _all_ranks_result_ready(self, model_state: Any) -> bool:
        """Pipeline readiness reduction behind one foreground iteration."""
        self.raise_if_close_failed()
        _raise_if_async_worker_stopped(self)
        model_state._eplb_migration_span_steps = getattr(model_state, "_eplb_migration_span_steps", 0) + 1
        group = get_ep_group().cpu_group
        if group.size() <= 1:
            ready = model_state.pending_result is not None
        else:
            ready = self._poll_pipelined_readiness(model_state, group)
        if not ready:
            model_state._eplb_migration_deferred_steps = getattr(model_state, "_eplb_migration_deferred_steps", 0) + 1
        return ready

    @staticmethod
    def _poll_pipelined_readiness(model_state: Any, group: Any) -> bool:
        previous_work = getattr(model_state, "_eplb_ready_work", None)
        if previous_work is not None:
            wait_started_at = time.perf_counter()
            previous_work.wait()
            ready = int(model_state._eplb_ready_flag.item()) == group.size()
            model_state._eplb_foreground_wait_ms = (
                getattr(model_state, "_eplb_foreground_wait_ms", 0.0) + (time.perf_counter() - wait_started_at) * 1000
            )
            del model_state._eplb_ready_work
            del model_state._eplb_ready_flag
            if ready:
                return True

        flag = torch.tensor(
            (int(model_state.pending_result is not None),),
            dtype=torch.int32,
            device="cpu",
        )
        work = torch.distributed.all_reduce(flag, group=group, async_op=True)
        model_state._eplb_ready_flag = flag
        model_state._eplb_ready_work = work
        return False

    def update_mapping(self, model_config, expanded_physical_to_logical: torch.Tensor) -> None:
        super().update_mapping(model_config, expanded_physical_to_logical)
        # Verified main updates a live state during elastic EP. Refresh the
        # Ascend replica table in place so captured graphs use the new mapping.
        key = hash_config_cached(model_config)
        refresh_model_routing_tables(self.model_states[key])

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
        with _configured_upstream_policy(parallel_config.eplb_config.policy, policy):
            state = super().from_mapping(**from_mapping_kwargs)
        state._configured_policy = policy
        if policy is not None:
            state.policy = policy
        if state.uses_custom_load_stats:
            for model_state in state.model_states.values():
                state._initialize_load_stats_state(model_state)
        for model_state in state.model_states.values():
            refresh_model_routing_tables(model_state)
            # Elastic EP builds the state here instead of through add_model, so
            # seed the placement snapshot on the same terms.
            model_state._allgather_physical_map = model_state.physical_to_logical_map.clone()
        return state
