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

from vllm_ascend.distributed.eplb.state import AscendEplbState


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

    def prepare_forward(
        self,
        model_config: Any,
        num_unpadded_tokens: int,
        ubatch_slices: list | None = None,
    ) -> None:
        state = self.state
        if state is None or not self.parallel_config.enable_eplb:
            return
        state.prepare_forward(model_config, num_unpadded_tokens, ubatch_slices)
        if state.should_record_tensor is not None:
            should_record = (
                state._should_record_current_step(log_stats=self.parallel_config.eplb_config.log_balancedness)
                and self._load_collection_phase_matched
            )
            state.should_record_tensor.fill_(should_record)
            if should_record:
                state._has_fresh_recorded_load = True

    def setup_from_mapping(
        self,
        model: nn.Module,
        model_config: Any,
        expanded_physical_to_logical: torch.Tensor,
        old_num_physical_experts: int | None = None,
    ) -> None:
        model = _unwrap_moe(model)
        assert is_mixture_of_experts(model)
        from_mapping_kwargs: dict[str, Any] = dict(
            model=model,
            model_config=model_config,
            device=self.device,
            parallel_config=self.parallel_config,
            expanded_physical_to_logical=expanded_physical_to_logical,
        )
        if old_num_physical_experts is not None:
            from_mapping_kwargs["num_valid_physical_experts"] = old_num_physical_experts
        self.state = AscendEplbState.from_mapping(**from_mapping_kwargs)
        self._has_registered_models = True
