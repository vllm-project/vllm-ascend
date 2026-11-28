# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from typing import Any

import torch
import torch.nn as nn
from vllm.config import get_current_vllm_config
from vllm.config.utils import hash_config_cached
from vllm.distributed import get_ep_group
from vllm.logger import logger
from vllm.model_executor.models.interfaces import (
    SupportsMultiModal,
    is_mixture_of_experts,
)
from vllm.v1.worker.gpu.eplb_utils import EPLBController

from vllm_ascend.ascend_config import EplbConfig
from vllm_ascend.device.hardware_profile import MoECommPolicy, get_current_hardware_profile
from vllm_ascend.distributed.eplb.config import resolve_placement_scope
from vllm_ascend.distributed.eplb.eplb_state import AscendEplbState
from vllm_ascend.distributed.eplb.placement import build_placement_metadata
from vllm_ascend.distributed.eplb.placement_runtime import (
    collective_call,
    install_placement,
    read_placement_collectively,
)
from vllm_ascend.distributed.eplb.policy.factory import create_eplb_policy
from vllm_ascend.patch.platform.patch_eplb import resolve_ascend_eplb_communicator
from vllm_ascend.utils import vllm_version_is


def _config_hash(config: Any) -> str:
    if vllm_version_is("0.33.0"):
        return config.compute_hash()
    return hash_config_cached(config)


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
    """Construct Ascend state and apply phase-filtered load collection."""

    def __init__(
        self,
        parallel_config: Any,
        device: torch.device,
        ascend_eplb_config: EplbConfig | None = None,
    ) -> None:
        super().__init__(parallel_config, device)
        ascend_eplb_config = ascend_eplb_config or EplbConfig()
        self.mode = ascend_eplb_config.mode
        eplb_config = getattr(parallel_config, "eplb_config", None)
        self._static_observability_enabled = self.mode == "static" and bool(
            getattr(eplb_config, "log_balancedness", False)
        )
        # A static engine reads its deployment document from this path.
        self.expert_placement_path = ascend_eplb_config.expert_placement_path
        self._placement_model_config: Any | None = None
        self._draft_placement_model_config: Any | None = None
        self._draft_parallel_config: Any | None = None
        self.load_collection_phase = ascend_eplb_config.load_collection_phase
        try:
            current_vllm_config = get_current_vllm_config()
            kv_transfer_config = getattr(current_vllm_config, "kv_transfer_config", None)
            self._kv_role = getattr(kv_transfer_config, "kv_role", None)
        except Exception:
            self._kv_role = None
        self.placement_scope = resolve_placement_scope(self._kv_role, self.load_collection_phase)
        # The communicator choice needs one binding class on every EPLB rank,
        # so reach a group consensus before the policy or state is built.
        stair_config = (
            resolve_ascend_eplb_communicator(parallel_config, ascend_eplb_config)
            if parallel_config.enable_eplb
            else ascend_eplb_config.stair_config
        )
        self.eplb_policy = create_eplb_policy(
            parallel_config.eplb_config.policy,
            stair_config,
        )
        self._load_collection_phase_matched = True

    def prepare_load(self) -> None:
        self.state = None
        self._has_registered_models = False
        self._placement_model_config = None
        self._draft_placement_model_config = None
        self._draft_parallel_config = None
        if self.parallel_config.enable_eplb:
            self.state = AscendEplbState(self.parallel_config, self.device, self.eplb_policy)
            self.state.static_mode = self.mode == "static"
            self.state.load_collection_phase = self.load_collection_phase

    def maybe_register_model(self, model: nn.Module, model_config: Any, *args, **kwargs) -> bool:
        added = super().maybe_register_model(model, model_config, *args, **kwargs)
        if added:
            self._placement_model_config = model_config
            if self.state is not None and self.mode == "static":
                self.state.disable_static_recording(force=True)
        return added

    def _placement_metadata(self) -> dict[str, Any]:
        model_state = self._target_placement_model_state()
        if self.state is None or self._placement_model_config is None:
            raise ValueError("EPLB deployments require exactly one registered target MoE model")
        return build_placement_metadata(
            model_state.model,
            self._placement_model_config,
            self.parallel_config,
            get_ep_group().world_size,
            self.placement_scope,
        )

    def _target_placement_model_state(self) -> Any:
        if self.state is None or self._placement_model_config is None:
            raise ValueError("EPLB deployments require exactly one registered target MoE model")
        model_hash = _config_hash(self._placement_model_config)
        model_state = self.state.model_states.get(model_hash)
        if model_state is None:
            raise ValueError("EPLB deployment could not locate the registered target model state")
        return model_state

    def _draft_placement_model_state(self) -> Any:
        if self.state is None or self._draft_placement_model_config is None:
            raise ValueError("EPLB deployment could not locate the registered draft model state")
        model_hash = _config_hash(self._draft_placement_model_config)
        model_state = self.state.model_states.get(model_hash)
        if model_state is None:
            raise ValueError("EPLB deployment could not locate the registered draft model state")
        return model_state

    def _optional_draft_placement_metadata(self) -> dict[str, Any] | None:
        if self._draft_placement_model_config is None:
            return None
        model_state = self._draft_placement_model_state()
        parallel_config = self._draft_parallel_config or self.parallel_config
        return build_placement_metadata(
            model_state.model,
            self._draft_placement_model_config,
            parallel_config,
            get_ep_group().world_size,
            self.placement_scope,
        )

    def _placement_model_states(self) -> dict[str, Any]:
        model_states = {"target": self._target_placement_model_state()}
        if self._draft_placement_model_config is not None:
            model_states["draft"] = self._draft_placement_model_state()
        return model_states

    def maybe_start_async_loop(self, eplb_models_added: bool, load_dummy_weights: bool = False) -> None:
        if self.mode != "static":
            return super().maybe_start_async_loop(eplb_models_added, load_dummy_weights=load_dummy_weights)
        if not eplb_models_added or load_dummy_weights:
            return
        group = get_ep_group().cpu_group
        metadata = collective_call(self._placement_metadata, group, "static metadata")
        draft_metadata = collective_call(
            self._optional_draft_placement_metadata,
            group,
            "static draft metadata",
        )
        assert self.expert_placement_path is not None and self.state is not None
        placement = read_placement_collectively(
            self.expert_placement_path,
            metadata,
            group,
            draft_metadata=draft_metadata,
        )
        install_placement(
            self.state,
            placement,
            group,
            torch.accelerator.current_stream(self.device),
            model_states=self._placement_model_states(),
        )
        logger.info(
            "Ascend static EPLB deployment installed: scope=%s sha256=%s layers=%d",
            self.placement_scope,
            placement.digest(),
            len(placement.physical_to_logical),
        )

    def step(self, is_dummy: bool = False, is_profile: bool = False) -> None:
        if self.mode == "static":
            if self.state is not None and self._static_observability_enabled:
                self.state.step_static_observability(
                    is_dummy=is_dummy,
                    is_profile=is_profile,
                )
            return
        super().step(is_dummy=is_dummy, is_profile=is_profile)

    def set_batch_phase(self, batch_has_prefill: bool) -> None:
        self._load_collection_phase_matched = is_eplb_load_collection_phase_matched(
            self.load_collection_phase,
            batch_has_prefill,
        )

    def maybe_register_speculator(
        self,
        speculator: Any | None,
        speculative_config: Any | None,
        load_dummy_weights: bool,
    ) -> bool:
        # The upstream controller checks target EPLB, which can differ from
        # the replicated draft's setting.
        draft_vllm_config = getattr(speculator, "draft_vllm_config", None)
        draft_parallel_config = getattr(draft_vllm_config, "parallel_config", None)
        draft_enable_eplb = getattr(draft_parallel_config, "enable_eplb", None) if speculator is not None else None
        if draft_enable_eplb is None and speculator is not None:
            draft_enable_eplb = speculator.vllm_config.parallel_config.enable_eplb
        if speculator is not None and not draft_enable_eplb:
            logger.info(
                "EPLB speculator registration skipped: draft enable_eplb=%s kv_role=%s",
                bool(draft_enable_eplb),
                self._kv_role,
            )
            return False
        spec_method = getattr(speculative_config, "method", None)
        logger.info(
            "EPLB speculator candidate: mode=%s kv_role=%s spec_method=%s draft_enable_eplb=%s",
            self.mode,
            self._kv_role,
            spec_method,
            bool(draft_enable_eplb),
        )
        # The draft MoE is registered exactly like the target: a dynamic
        # deployment rebalances both models, and a static one installs a single
        # placement for both.
        added = super().maybe_register_speculator(speculator, speculative_config, load_dummy_weights)
        if added:
            self._draft_placement_model_config = getattr(
                speculative_config,
                "draft_model_config",
                None,
            )
            self._draft_parallel_config = draft_parallel_config
        if added and self.state is not None and self.mode == "static":
            self.state.disable_static_recording(force=True)
        return added

    def prepare_forward(
        self,
        model_config: Any,
        num_unpadded_tokens: int,
        ubatch_slices: list | None = None,
    ) -> None:
        state = self.state
        if state is None:
            return
        if not self.parallel_config.enable_eplb:
            return
        if self.mode == "static":
            if not self._static_observability_enabled:
                state.disable_static_recording()
                state._static_observation_due = False
                state._is_load_sampling_step = False
                state._should_collect_local_load = False
                return
            if state._static_observation_due and state._static_recording_enabled:
                state.step_static_observability()
            should_record = state.update_static_observation_schedule() and state.should_record_tensor is not None
            state.set_static_recording_enabled(should_record)
            if state.should_record_tensor is not None:
                state.should_record_tensor.fill_(should_record)
            if should_record and state.allgather_record_tensor is not None:
                state.allgather_record_tensor.fill_(-1)
            state._is_load_sampling_step = should_record
            state._should_collect_local_load = should_record
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
        # AllGather needs source validity; other paths use operator counts.
        if state.should_record_tensor is not None:
            is_sampling = state._should_record_current_step(log_stats=self.parallel_config.eplb_config.log_balancedness)
            should_record = is_sampling and self._load_collection_phase_matched
            state.should_record_tensor.fill_(should_record)
            state._is_load_sampling_step = is_sampling
            state._should_collect_local_load = should_record
            if should_record and not supports_allgather:
                state._has_fresh_recorded_load = True

    # The two supported upstream revisions intentionally expose different signatures.
    def setup_from_mapping(
        self,
        model_config: Any,
        expanded_physical_to_logical: torch.Tensor,
    ) -> None:
        # Elastic EP reuses the live state and communicator on verified main.
        super().setup_from_mapping(model_config, expanded_physical_to_logical)
