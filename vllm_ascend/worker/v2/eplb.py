# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from typing import Any

import torch
import torch.nn as nn
from vllm.model_executor.models.interfaces import (
    SupportsMultiModal,
    is_mixture_of_experts,
)
from vllm.v1.worker.gpu.eplb_utils import EPLBController

from vllm_ascend.distributed.eplb_state import AscendEplbState


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
        load_collection_phase: str = "all",
    ) -> None:
        super().__init__(parallel_config, device)
        self.load_collection_phase = load_collection_phase
        self._load_collection_phase_matched = True

    def prepare_load(self) -> None:
        self.state = None
        self._has_registered_models = False
        if self.parallel_config.enable_eplb:
            self.state = AscendEplbState(self.parallel_config, self.device)

    def set_batch_phase(self, batch_has_prefill: bool) -> None:
        self._load_collection_phase_matched = is_eplb_load_collection_phase_matched(
            self.load_collection_phase,
            batch_has_prefill,
        )

    def step(
        self,
        is_dummy: bool = False,
        is_profile: bool = False,
    ) -> None:
        state = self.state
        if not self.parallel_config.enable_eplb or self.suppressed or state is None or not self._has_registered_models:
            return

        discard_current_load = not is_profile and not self._load_collection_phase_matched
        if (
            not is_dummy
            and not is_profile
            and not discard_current_load
            and not state._should_record_current_step(log_stats=self.parallel_config.eplb_config.log_balancedness)
        ):
            # Ascend records local GMM counts after every MoE call. Clear
            # them once per pass while the upstream window is closed.
            for model_state in state.model_states.values():
                model_state.expert_load_pass.zero_()

        # Phase selection may change the load submitted by each rank, but all
        # ranks must advance the EPLB state machine and enter collectives in
        # the same order. Treat a non-matching batch as an EPLB dummy step and
        # let the upstream controller preserve the global logging schedule.
        super().step(is_dummy=is_dummy or discard_current_load, is_profile=is_profile)

    def setup_from_mapping(
        self,
        model: nn.Module,
        model_config: Any,
        expanded_physical_to_logical: torch.Tensor,
        old_num_physical_experts: int,
    ) -> None:
        model = _unwrap_moe(model)
        assert is_mixture_of_experts(model)
        self.state = AscendEplbState.from_mapping(
            model=model,
            model_config=model_config,
            device=self.device,
            parallel_config=self.parallel_config,
            expanded_physical_to_logical=expanded_physical_to_logical,
            num_valid_physical_experts=old_num_physical_experts,
        )
        self._has_registered_models = True
