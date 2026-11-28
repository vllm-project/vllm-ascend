# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
from vllm.distributed import get_ep_group
from vllm.distributed.eplb import eplb_state as _eplb_state
from vllm.logger import logger
from vllm.model_executor.models.interfaces import (
    SupportsMultiModal,
    get_mixture_of_experts_model,
    is_mixture_of_experts,
)
from vllm.v1.worker.gpu.eplb_utils import EPLBController, draft_model_supports_eplb

from vllm_ascend.ascend_config import EplbConfig
from vllm_ascend.device.hardware_profile import MoECommPolicy, get_current_hardware_profile
from vllm_ascend.distributed.eplb.eplb_state import AscendEplbState, _get_pp_topology
from vllm_ascend.distributed.eplb.placement import (
    ACTIVE_STATIC_CONTROLLER,
    ACTIVE_STATIC_LOAD,
    _bcast_rank0,
    apply_static_rows_to_model,
    read_static_placement_snapshot,
    slice_static_role_rows,
)
from vllm_ascend.distributed.eplb.policy.factory import create_eplb_policy
from vllm_ascend.patch.platform.patch_eplb import resolve_ascend_eplb_communicator


def is_eplb_load_collection_phase_matched(
    load_collection_phase: str,
    batch_has_prefill: bool,
) -> bool:
    """Return whether the batch belongs to the configured collection phase."""
    if load_collection_phase == "all":
        return True
    batch_phase = "prefill" if batch_has_prefill else "decode"
    return load_collection_phase == batch_phase


def _unwrap_moe(model: nn.Module) -> nn.Module:
    if not is_mixture_of_experts(model) and isinstance(model, SupportsMultiModal):
        return model.get_language_model()
    return model


class AscendEPLBController(EPLBController):
    """Construct Ascend state and apply phase-filtered or static EPLB placement."""

    def __init__(
        self,
        parallel_config: Any,
        device: torch.device,
        ascend_eplb_config: EplbConfig | None = None,
    ) -> None:
        super().__init__(parallel_config, device)
        ascend_eplb_config = ascend_eplb_config or EplbConfig()
        self.load_collection_phase = ascend_eplb_config.load_collection_phase
        self.expert_placement_path = ascend_eplb_config.expert_placement_path
        self.mode = ascend_eplb_config.mode
        # The communicator choice needs one binding class on every EPLB rank,
        # so reach a group consensus before the policy or state is built.
        stair_config = (
            resolve_ascend_eplb_communicator(parallel_config, ascend_eplb_config)
            if parallel_config.enable_eplb and self.mode != "static"
            else ascend_eplb_config.stair_config
        )
        self.eplb_policy = create_eplb_policy(
            parallel_config.eplb_config.policy,
            stair_config,
        )
        self._load_collection_phase_matched = True
        self._snapshot: dict[str, Any] | None = None
        self._draft_bound = False

    def prepare_load(self) -> None:
        self.state = None
        self._has_registered_models = False
        self._snapshot = None
        self._draft_bound = False
        ACTIVE_STATIC_LOAD.set(None)
        ACTIVE_STATIC_CONTROLLER.set(self if self.parallel_config.enable_eplb and self.mode == "static" else None)
        if self.parallel_config.enable_eplb and self.mode != "static":
            self.state = AscendEplbState(self.parallel_config, self.device, self.eplb_policy)
            if self.expert_placement_path is not None:
                self.state.enable_placement_export(self.expert_placement_path)

    def bind_model_for_static_load(self, model: nn.Module, vllm_config: Any, model_config: Any) -> None:
        ACTIVE_STATIC_LOAD.set(None)
        if self.mode != "static" or not vllm_config.parallel_config.enable_eplb:
            return
        moe_model = get_mixture_of_experts_model(model)
        ep_group = get_ep_group()
        if getattr(vllm_config.load_config, "load_format", None) == "dummy":
            if moe_model is not None and list(getattr(moe_model, "moe_layers", ())):
                initial = list(
                    _eplb_state.EplbState.build_initial_global_physical_to_logical_map(
                        moe_model.num_routed_experts,
                        moe_model.num_redundant_experts,
                    )
                )
                apply_static_rows_to_model(
                    moe_model,
                    [initial] * len(moe_model.moe_layers),
                    ep_group.world_size,
                    ep_group.rank_in_group,
                    self.device,
                )
            return

        spec_config = getattr(vllm_config, "speculative_config", None)
        is_draft = spec_config is not None and model_config is getattr(spec_config, "draft_model_config", None)
        if is_draft:
            if getattr(spec_config, "prefill_context_kv_only", False) or not draft_model_supports_eplb(
                spec_config, moe_model
            ):
                return
            if self._snapshot is None:
                raise RuntimeError("Static EPLB draft model cannot load before the target snapshot")
            rows = slice_static_role_rows(self._snapshot, "draft", moe_model, ep_group.cpu_group)
            local_ids = apply_static_rows_to_model(
                moe_model, rows, ep_group.world_size, ep_group.rank_in_group, self.device
            )
            self._draft_bound = True
            ACTIVE_STATIC_LOAD.set((True, local_ids))
            return

        assert self.expert_placement_path is not None
        pp_rank, pp_size, pp_cpu_group = _get_pp_topology()
        if pp_size > 1 and pp_cpu_group is None:
            raise RuntimeError("PP CPU group is required for multi-stage static EPLB.")
        pp_group = pp_cpu_group if pp_size > 1 else None
        self._snapshot = read_static_placement_snapshot(
            self.expert_placement_path, ep_group.cpu_group, pp_group, pp_rank
        )
        rows = slice_static_role_rows(self._snapshot, "target", moe_model, ep_group.cpu_group, pp_group, pp_rank)
        if rows:
            local_ids = apply_static_rows_to_model(
                moe_model, rows, ep_group.world_size, ep_group.rank_in_group, self.device
            )
            ACTIVE_STATIC_LOAD.set((True, local_ids))

    def maybe_register_model(self, model: nn.Module, model_config: Any, *args, **kwargs) -> bool:
        if self.mode == "static":
            return False
        return super().maybe_register_model(model, model_config, *args, **kwargs)

    def maybe_register_speculator(
        self,
        speculator: Any | None,
        speculative_config: Any | None,
        load_dummy_weights: bool,
    ) -> bool:
        if self.mode == "static":
            return False
        # The upstream controller checks target EPLB, which can differ from
        # the replicated draft's setting.
        if speculator is not None:
            draft_vllm_config = getattr(speculator, "draft_vllm_config", None)
            draft_parallel_config = (
                draft_vllm_config.parallel_config
                if draft_vllm_config is not None
                else speculator.vllm_config.parallel_config
            )
            if not draft_parallel_config.enable_eplb:
                return False
        return super().maybe_register_speculator(speculator, speculative_config, load_dummy_weights)

    def maybe_start_async_loop(
        self,
        eplb_models_added: bool,
        load_dummy_weights: bool = False,
    ) -> None:
        ACTIVE_STATIC_CONTROLLER.set(None)
        ACTIVE_STATIC_LOAD.set(None)
        if self.mode == "static":
            if load_dummy_weights or self._snapshot is None:
                return
            if "draft" in self._snapshot:
                _, pp_size, pp_cpu_group = _get_pp_topology()
                any_draft = self._draft_bound
                if pp_size > 1 and pp_cpu_group is not None:
                    ep_cpu_group = get_ep_group().cpu_group
                    gathered: list[bool] | None = None
                    if ep_cpu_group.rank() == 0:
                        gathered = [False] * pp_cpu_group.size()
                        dist.all_gather_object(gathered, self._draft_bound, group=pp_cpu_group)
                    any_draft = any(_bcast_rank0(gathered, ep_cpu_group) or ())
                if not any_draft:
                    raise ValueError("Static EPLB placement contains 'draft', but no EPLB draft model is registered")
            logger.info(
                "Ascend static EPLB placement loaded: target_layers=%d draft_layers=%d",
                len(self._snapshot["target"]["physical_to_logical_map"]),
                len(self._snapshot["draft"]["physical_to_logical_map"]) if "draft" in self._snapshot else 0,
            )
            return
        if not load_dummy_weights and self.state is not None and self.expert_placement_path is not None:
            self.state.setup_pp_placement_export()
        super().maybe_start_async_loop(eplb_models_added, load_dummy_weights)

    def set_batch_phase(self, batch_has_prefill: bool) -> None:
        self._load_collection_phase_matched = is_eplb_load_collection_phase_matched(
            self.load_collection_phase,
            batch_has_prefill,
        )

    def prepare_forward(
        self,
        model_config: Any,
        num_unpadded_tokens: int,
        ubatch_slices: list | None = None,
    ) -> None:
        state = self.state
        if state is None or not self.parallel_config.enable_eplb:
            return
        supports_allgather = get_current_hardware_profile().moe_comm_policy is MoECommPolicy.CAPACITY_AND_WORLD_SIZE
        if not state.uses_custom_load_stats or supports_allgather:
            state.prepare_forward(model_config, num_unpadded_tokens, ubatch_slices)
            if not supports_allgather and self.expert_placement_path is None:
                return
        if supports_allgather:
            state._uses_phase_filter = self.load_collection_phase != "all"
            if state.allgather_record_tensor is not None:
                state.allgather_record_tensor.fill_(-1)
        if state.should_record_tensor is not None:
            is_sampling = state._should_record_current_step(log_stats=self.parallel_config.eplb_config.log_balancedness)
            should_record = is_sampling and self._load_collection_phase_matched
            state.should_record_tensor.fill_(should_record)
            state._is_load_sampling_step = is_sampling
            state._should_collect_local_load = should_record
            if should_record and not supports_allgather:
                state._has_fresh_recorded_load = True

    def setup_from_mapping(
        self,
        model_config: Any,
        expanded_physical_to_logical: torch.Tensor,
    ) -> None:
        super().setup_from_mapping(model_config, expanded_physical_to_logical)
