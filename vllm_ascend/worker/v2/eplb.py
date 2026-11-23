# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from typing import Any

import torch
import torch.nn as nn
from vllm.distributed import get_ep_group
from vllm.logger import logger
from vllm.model_executor.models.interfaces import (
    SupportsMultiModal,
    is_mixture_of_experts,
)
from vllm.v1.worker.gpu.eplb_utils import EPLBController

from vllm_ascend.ascend_config import EplbConfig
from vllm_ascend.device.hardware_profile import MoECommPolicy, get_current_hardware_profile
from vllm_ascend.distributed.eplb.eplb_state import AscendEplbState
from vllm_ascend.distributed.eplb.placement import build_placement_metadata
from vllm_ascend.distributed.eplb.placement_runtime import (
    collective_call,
    export_committed_placement,
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
    """Construct Ascend state and apply phase-filtered load collection."""

    def __init__(
        self,
        parallel_config: Any,
        device: torch.device,
        ascend_eplb_config: EplbConfig | None = None,
    ) -> None:
        super().__init__(parallel_config, device)
        ascend_eplb_config = ascend_eplb_config or EplbConfig()
        self.export_path = ascend_eplb_config.export_path
        self.export_workload = ascend_eplb_config.export_workload
        self._placement_model_config: Any | None = None
        self.load_collection_phase = ascend_eplb_config.load_collection_phase
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
        if self.parallel_config.enable_eplb:
            self.state = AscendEplbState(self.parallel_config, self.device, self.eplb_policy)
            self.state.load_collection_phase = self.load_collection_phase

    def maybe_register_model(self, model: nn.Module, model_config: Any, *args, **kwargs) -> bool:
        added = super().maybe_register_model(model, model_config, *args, **kwargs)
        if added:
            self._placement_model_config = model_config
            if self.state is not None:
                model_state = self.state.model_states.get(model_config.compute_hash())
                if model_state is not None:
                    first_layer = model_state.physical_to_logical_map[0][:8].tolist()
                    logger.info(
                        "EPLB target model registered: model=%s moe_layers=%d physical_to_logical(layer0, first8)=%s",
                        model_config.model,
                        len(model_state.physical_to_logical_map),
                        first_layer,
                    )
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
        )

    def _target_placement_model_state(self) -> Any:
        if self.state is None or self._placement_model_config is None:
            raise ValueError("EPLB deployments require exactly one registered target MoE model")
        model_hash = self._placement_model_config.compute_hash()
        model_state = self.state.model_states.get(model_hash)
        if model_state is None:
            raise ValueError("EPLB deployment export could not locate the registered target model state")
        return model_state

    def export_placement(
        self,
        path: str | None = None,
        workload: str | None = None,
    ) -> dict[str, Any]:
        """Offline collective RPC: finish the active cycle and export its result."""
        group = get_ep_group().cpu_group
        metadata = collective_call(self._placement_metadata, group, "export metadata")
        assert self.state is not None
        path = path or self.export_path
        workload = workload or self.export_workload
        if path is None or not workload:
            raise ValueError("EPLB export requires a path and non-empty workload description")
        model_state = self._target_placement_model_state()
        return export_committed_placement(self.state, metadata, path, workload, group, model_state)

    def maybe_start_async_loop(self, eplb_models_added: bool, load_dummy_weights: bool = False) -> None:
        return super().maybe_start_async_loop(eplb_models_added, load_dummy_weights=load_dummy_weights)

    def step(self, is_dummy: bool = False, is_profile: bool = False) -> None:
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
        if speculator is not None and not speculator.vllm_config.parallel_config.enable_eplb:
            return False
        return super().maybe_register_speculator(speculator, speculative_config, load_dummy_weights)

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
