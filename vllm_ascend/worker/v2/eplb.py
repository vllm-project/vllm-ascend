# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
from vllm.distributed import get_ep_group
from vllm.distributed import parallel_state as _parallel_state
from vllm.distributed.parallel_state import get_pp_group
from vllm.logger import logger
from vllm.model_executor.models.interfaces import (
    SupportsMultiModal,
    get_mixture_of_experts_model,
    is_mixture_of_experts,
)
from vllm.v1.worker.gpu.eplb_utils import EPLBController, draft_model_supports_eplb

from vllm_ascend.ascend_config import EplbConfig
from vllm_ascend.device.hardware_profile import MoECommPolicy, get_current_hardware_profile
from vllm_ascend.distributed.eplb.eplb_state import AscendEplbState
from vllm_ascend.distributed.eplb.placement_runtime import (
    ACTIVE_STATIC_CONTROLLER,
    ACTIVE_STATIC_ROLE_PLACEMENT,
    PlacementSnapshot,
    bind_role_placement_to_model,
    build_model_role_spec,
    collective_call,
    read_and_slice_target_placement,
    validate_role_against_spec,
)
from vllm_ascend.distributed.eplb.policy.factory import create_eplb_policy
from vllm_ascend.patch.platform.patch_eplb import resolve_ascend_eplb_communicator


def _get_pp_topology() -> tuple[int, int, Any | None]:
    if getattr(_parallel_state, "_PP", None) is None and get_pp_group is _parallel_state.get_pp_group:
        return 0, 1, None
    try:
        pp_group = get_pp_group()
    except AssertionError:
        return 0, 1, None
    return pp_group.rank_in_group, pp_group.world_size, getattr(pp_group, "cpu_group", None)


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
        self._snapshot: PlacementSnapshot | None = None
        self._draft_registered = False

    def prepare_load(self) -> None:
        self.state = None
        self._has_registered_models = False
        self._snapshot = None
        self._draft_registered = False
        ACTIVE_STATIC_ROLE_PLACEMENT.set(None)
        if not self.parallel_config.enable_eplb:
            ACTIVE_STATIC_CONTROLLER.set(None)
            return
        self.state = AscendEplbState(self.parallel_config, self.device, self.eplb_policy)
        self.state.static_mode = self.mode == "static"
        ACTIVE_STATIC_CONTROLLER.set(self if self.mode == "static" else None)
        if self.mode == "dynamic" and self.expert_placement_path is not None:
            self.state.enable_placement_export(self.expert_placement_path)

    def bind_model_for_static_load(self, model: nn.Module, vllm_config: Any, model_config: Any) -> None:
        ACTIVE_STATIC_ROLE_PLACEMENT.set(None)
        if (
            self.mode != "static"
            or getattr(vllm_config.load_config, "load_format", None) == "dummy"
            or not vllm_config.parallel_config.enable_eplb
        ):
            return
        spec_config = getattr(vllm_config, "speculative_config", None)
        is_draft = spec_config is not None and model_config is getattr(spec_config, "draft_model_config", None)
        moe_model = get_mixture_of_experts_model(model)
        ep_group = get_ep_group()
        if is_draft:
            if getattr(spec_config, "prefill_context_kv_only", False) or not draft_model_supports_eplb(
                spec_config, moe_model
            ):
                return
            if self._snapshot is None:
                raise RuntimeError("Static EPLB draft model cannot load before the target snapshot")
            snapshot = self._snapshot

            def validate_draft() -> None:
                draft_spec = build_model_role_spec(moe_model)
                if draft_spec is None:
                    raise ValueError("Static EPLB draft model has no MoE layers")
                if snapshot.draft is None:
                    raise ValueError("EPLB placement snapshot is missing the expected draft placement")
                validate_role_against_spec(snapshot.draft, "draft", draft_spec)

            collective_call(validate_draft, ep_group.cpu_group, "draft placement validation")
            assert snapshot.draft is not None and moe_model is not None
            bind_role_placement_to_model(moe_model, snapshot.draft, ep_group.world_size, ep_group.rank_in_group)
            ACTIVE_STATIC_ROLE_PLACEMENT.set(snapshot.draft)
            return

        assert self.expert_placement_path is not None
        pp_rank, pp_size, pp_cpu_group = _get_pp_topology()
        if pp_size > 1 and pp_cpu_group is None:
            raise RuntimeError("PP CPU group is required for multi-stage static EPLB.")
        snapshot, target_role = read_and_slice_target_placement(
            self.expert_placement_path,
            lambda: build_model_role_spec(moe_model),
            ep_group.cpu_group,
            pp_cpu_group if pp_size > 1 else None,
            pp_rank,
        )
        self._snapshot = snapshot
        if target_role is not None:
            assert moe_model is not None
            bind_role_placement_to_model(moe_model, target_role, ep_group.world_size, ep_group.rank_in_group)
            ACTIVE_STATIC_ROLE_PLACEMENT.set(target_role)

    def maybe_register_model(self, model: nn.Module, model_config: Any, *args, **kwargs) -> bool:
        if self.mode == "static":
            moe_model = get_mixture_of_experts_model(model)
            if moe_model is None or not list(getattr(moe_model, "moe_layers", ())):
                return False
        return super().maybe_register_model(model, model_config, *args, **kwargs)

    def maybe_register_speculator(
        self,
        speculator: Any | None,
        speculative_config: Any | None,
        load_dummy_weights: bool,
    ) -> bool:
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
        added = super().maybe_register_speculator(speculator, speculative_config, load_dummy_weights)
        if added:
            self._draft_registered = True
        return added

    def maybe_start_async_loop(
        self,
        eplb_models_added: bool,
        load_dummy_weights: bool = False,
    ) -> None:
        if self.mode != "static":
            if not load_dummy_weights and self.state is not None and self.expert_placement_path is not None:
                self.state.setup_pp_placement_export()
            super().maybe_start_async_loop(eplb_models_added, load_dummy_weights=load_dummy_weights)
            return
        ACTIVE_STATIC_CONTROLLER.set(None)
        ACTIVE_STATIC_ROLE_PLACEMENT.set(None)
        if load_dummy_weights or self.state is None or self._snapshot is None:
            return
        _, pp_size, pp_cpu_group = _get_pp_topology()
        any_draft = self._draft_registered
        if pp_size > 1:
            assert pp_cpu_group is not None
            ep_cpu_group = get_ep_group().cpu_group
            flags: list[Any] = [False]
            if ep_cpu_group.rank() == 0:
                gathered: list[Any] = [False] * pp_cpu_group.size()
                dist.all_gather_object(gathered, self._draft_registered, group=pp_cpu_group)
                flags[0] = any(gathered)
            if ep_cpu_group.size() > 1:
                dist.broadcast_object_list(flags, src=dist.get_global_rank(ep_cpu_group, 0), group=ep_cpu_group)
            any_draft = bool(flags[0])
        if self._snapshot.draft is not None and not any_draft:
            raise ValueError("Static EPLB placement contains 'draft', but no EPLB draft model is registered")
        logger.info(
            "Ascend static EPLB placement loaded: sha256=%s target_layers=%d draft_layers=%d",
            self._snapshot.digest(),
            self._snapshot.target.num_layers,
            0 if self._snapshot.draft is None else self._snapshot.draft.num_layers,
        )

    def step(self, is_dummy: bool = False, is_profile: bool = False) -> None:
        if self.mode == "static":
            return
        super().step(is_dummy=is_dummy, is_profile=is_profile)

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
        if state is None or not self.parallel_config.enable_eplb or self.mode == "static":
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
