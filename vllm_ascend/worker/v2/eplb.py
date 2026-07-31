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


def is_eplb_load_scope_matched(load_scope: str, batch_has_prefill: bool) -> bool:
    """Return whether the whole batch belongs to the configured load scope."""
    if load_scope == "all":
        return True
    batch_scope = "prefill" if batch_has_prefill else "decode"
    return load_scope == batch_scope


def _unwrap_moe(model: nn.Module) -> nn.Module:
    if not is_mixture_of_experts(model) and isinstance(model, SupportsMultiModal):
        return model.get_language_model()
    return model


class AscendEPLBController(EPLBController):
    """Construct Ascend state and apply batch-scoped load collection."""

    def __init__(
        self,
        parallel_config: Any,
        device: torch.device,
        load_scope: str = "all",
    ) -> None:
        super().__init__(parallel_config, device)
        self.load_scope = load_scope
        self._scope_matched = True

    def prepare_load(self) -> None:
        self.state = None
        self._has_registered_models = False
        if self.parallel_config.enable_eplb:
            self.state = AscendEplbState(self.parallel_config, self.device)

    def set_batch_scope(self, batch_has_prefill: bool) -> None:
        self._scope_matched = is_eplb_load_scope_matched(
            self.load_scope,
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

        if not is_dummy and not is_profile:
            if not self._scope_matched:
                state.step(True, False, log_stats=False)
                return
            elif not state._should_record_current_step(log_stats=self.parallel_config.eplb_config.log_balancedness):
                # Ascend records local GMM counts after every MoE call. Clear
                # them once per pass while the upstream window is closed.
                for model_state in state.model_states.values():
                    model_state.expert_load_pass.zero_()
        super().step(is_dummy=is_dummy, is_profile=is_profile)

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
