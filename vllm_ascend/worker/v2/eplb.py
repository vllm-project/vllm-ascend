# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from contextlib import contextmanager
from multiprocessing import Manager
from typing import Any

import torch
import torch.nn as nn
from vllm.model_executor.models.interfaces import (
    SupportsMultiModal,
    is_mixture_of_experts,
)
from vllm.v1.worker.gpu.eplb_utils import EPLBController

from vllm_ascend.distributed.eplb.state import AscendEplbState
from vllm_ascend.eplb.adaptor.vllm_adaptor import VllmEplbAdaptor
from vllm_ascend.eplb.core.eplb_device_transfer_loader import D2DExpertWeightLoader
from vllm_ascend.eplb.core.eplb_worker import EplbProcess
from vllm_ascend.eplb.eplb_updator import EplbUpdator


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
        legacy_config: Any = None,
    ) -> None:
        super().__init__(parallel_config, device)
        self.load_collection_phase = load_collection_phase
        self._load_collection_phase_matched = True
        self.legacy_config = legacy_config
        self.legacy_updator = None
        self.legacy_manager = None
        self.legacy_load_enabled = None
        self.legacy_counter_enabled = None
        self._legacy_forward_pending = False
        self._legacy_is_dummy = False

    def maybe_register_model(self, model, model_config, load_dummy_weights) -> bool:
        if self.legacy_config is None:
            return super().maybe_register_model(model, model_config, load_dummy_weights)
        if load_dummy_weights:
            return False
        adaptor = VllmEplbAdaptor(model=model)
        loader = D2DExpertWeightLoader()
        self.legacy_manager = Manager()
        shared_dict = self.legacy_manager.dict({"expert_map": None, "moe_load": None, "expert_maps": None})
        process = EplbProcess(
            shared_dict=shared_dict,
            policy_type=self.legacy_config.eplb_policy_type,
            enable_d2d=True,
            tp_size=self.parallel_config.tensor_parallel_size,
        )
        self.legacy_updator = EplbUpdator(self.legacy_config, loader, process, process._launch_process())
        loader.set_adator(adaptor)
        self.legacy_updator.set_adaptor(adaptor)
        # Stable device gates are captured with the load accumulation. Capture
        # and profiling contribute neither load nor a time-window iteration.
        self.legacy_load_enabled = torch.zeros((), dtype=torch.int32, device=self.device)
        self.legacy_counter_enabled = torch.zeros((), dtype=torch.int32, device=self.device)
        for layer in adaptor.moe_layers:
            layer.register_buffer("eplb_load_enabled", self.legacy_load_enabled)
            layer.register_buffer("eplb_counter_enabled", self.legacy_counter_enabled)
        self.legacy_updator.warm_up_eplb()
        return False  # The legacy planner owns its process, not AscendEplbState.

    @contextmanager
    def suppress_legacy(self, enabled: bool = True):
        """Exclude initialization runs while keeping graph load operations."""
        if self.legacy_config is None or not enabled:
            yield
            return
        suppressed = self.suppressed
        self.suppressed = True
        if self.legacy_load_enabled is not None:
            self.legacy_load_enabled.zero_()
            self.legacy_counter_enabled.zero_()
        try:
            yield
        finally:
            self.suppressed = suppressed

    def step(self, is_dummy: bool = False, is_profile: bool = False) -> None:
        if self.legacy_config is None:
            return super().step(is_dummy=is_dummy, is_profile=is_profile)
        if self.suppressed or is_profile or not self._legacy_forward_pending:
            return
        assert self.legacy_updator is not None
        # Async sampling can return before the target graph finishes. Finish
        # its reads before replacing the weights and logical/physical maps.
        if self.legacy_updator.update_expert_weight_flag():
            torch.npu.current_stream().synchronize()
        self.legacy_updator.forward_end()
        self._legacy_forward_pending = False
        self.legacy_load_enabled.zero_()
        self.legacy_counter_enabled.zero_()

    def shutdown_legacy(self) -> None:
        if self.legacy_updator is not None:
            self.legacy_updator.shutdown()
        if self.legacy_manager is not None:
            self.legacy_manager.shutdown()

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
        if self.legacy_config is not None:
            if self.suppressed or self.legacy_updator is None:
                return
            self.legacy_updator.forward_before()
            # Idle DP ranks still advance the same FlashLB window and join
            # collectives, but their synthetic tokens must not count as load.
            self.legacy_load_enabled.fill_(not self._legacy_is_dummy)
            self.legacy_counter_enabled.fill_(1)
            self._legacy_forward_pending = True
            return
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
