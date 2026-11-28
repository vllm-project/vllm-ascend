# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import os
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
    export_committed_placement,
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
        self.expert_placement_path = ascend_eplb_config.expert_placement_path
        self.export_after_rearranges = ascend_eplb_config.export_after_rearranges
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
            self.state.load_collection_phase = self.load_collection_phase
            self.state.export_after_rearranges = self.export_after_rearranges

    def maybe_register_model(self, model: nn.Module, model_config: Any, *args, **kwargs) -> bool:
        added = super().maybe_register_model(model, model_config, *args, **kwargs)
        if added:
            self._placement_model_config = model_config
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
            raise ValueError("EPLB deployment export could not locate the registered target model state")
        return model_state

    def _draft_placement_model_state(self) -> Any:
        if self.state is None or self._draft_placement_model_config is None:
            raise ValueError("EPLB deployment export could not locate the registered draft model state")
        model_hash = _config_hash(self._draft_placement_model_config)
        model_state = self.state.model_states.get(model_hash)
        if model_state is None:
            raise ValueError("EPLB deployment export could not locate the registered draft model state")
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

    def _auto_export_path(self) -> str:
        """Resolve the one-shot path, tagged with PD role and DP rank."""
        assert self.expert_placement_path is not None
        root, suffix = os.path.splitext(self.expert_placement_path)
        role = self._kv_role or "standalone"
        dp_rank = getattr(self.parallel_config, "data_parallel_rank", 0)
        return f"{root}.{role}.dp{dp_rank}{suffix}"

    def _run_auto_export(self) -> None:
        """Export the committed deployment once, between two forwards."""
        try:
            path = self._auto_export_path()
            logger.info("Ascend EPLB deployment export running in a forward gap: %s", path)
            result = self.export_placement(path)
        except Exception:
            logger.exception("Ascend EPLB deployment export failed; continuing without it")
            return
        logger.info("Ascend EPLB deployment exported: %s", result)

    def _export_workload(self) -> str:
        """Describe how the deployment was produced, for the file provenance."""
        return (
            f"Ascend EPLB deployment; "
            f"placement_scope={self.placement_scope}; "
            f"load_collection_phase={self.load_collection_phase}"
        )

    def export_placement(self, path: str | None = None) -> dict[str, Any]:
        """Offline collective step: finish the active cycle and export its result."""
        group = get_ep_group().cpu_group
        metadata = collective_call(self._placement_metadata, group, "export metadata")
        assert self.state is not None
        path = path or self.expert_placement_path
        if path is None:
            raise ValueError("EPLB export requires eplb_config.expert_placement_path")
        workload = self._export_workload()
        model_state = self._target_placement_model_state()
        draft_metadata = collective_call(
            self._optional_draft_placement_metadata,
            group,
            "draft export metadata",
        )
        draft_model_state = self._draft_placement_model_state() if draft_metadata is not None else None
        return export_committed_placement(
            self.state,
            metadata,
            path,
            workload,
            group,
            model_state,
            draft_metadata=draft_metadata,
            draft_model_state=draft_model_state,
        )

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
            "EPLB speculator candidate: kv_role=%s spec_method=%s draft_enable_eplb=%s",
            self._kv_role,
            spec_method,
            bool(draft_enable_eplb),
        )
        # The draft MoE is registered exactly like the target, so a dynamic
        # deployment rebalances both models.
        added = super().maybe_register_speculator(speculator, speculative_config, load_dummy_weights)
        if added:
            self._draft_placement_model_config = getattr(
                speculative_config,
                "draft_model_config",
                None,
            )
            self._draft_parallel_config = draft_parallel_config
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
        if getattr(self, "export_after_rearranges", None) is not None and state.take_export_request():
            self._run_auto_export()
        supports_allgather = get_current_hardware_profile().moe_comm_policy is MoECommPolicy.CAPACITY_AND_WORLD_SIZE
        if not state.uses_custom_load_stats or supports_allgather:
            state.prepare_forward(model_config, num_unpadded_tokens, ubatch_slices)
            if not supports_allgather:
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
