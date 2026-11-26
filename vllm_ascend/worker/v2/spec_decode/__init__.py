# Adapt from https://github.com/vllm-project/vllm/blob/main/vllm/v1/worker/gpu/sample/spec_decode/__init__.py
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#
import copy
from contextlib import suppress
from contextvars import ContextVar
from functools import wraps
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from vllm.config import CompilationMode, SpeculativeConfig, VllmConfig

from vllm_ascend.ascend_config import draft_config_context
from vllm_ascend.ascend_forward_context import bind_draft_moe_comm_context

if TYPE_CHECKING:
    from vllm.v1.worker.gpu.spec_decode.speculator import BaseSpeculator
    from vllm.v1.worker.gpu.states import RequestState


_ACTIVE_SPECULATOR_CONTEXT: ContextVar["BaseSpeculator | None"] = ContextVar("active_speculator", default=None)


def resolve_speculator_draft_vllm_config(speculator, vllm_config: VllmConfig | None = None) -> VllmConfig | None:
    """Resolve the draft's model metadata and runtime parallel settings."""
    if speculator is None:
        speculator = _ACTIVE_SPECULATOR_CONTEXT.get()
    for attr in ("draft_vllm_config", "_draft_vllm_config", "_ascend_resolved_draft_vllm_config"):
        config = getattr(speculator, attr, None)
        if config is not None:
            return config

    base_config = getattr(speculator, "vllm_config", None)
    if base_config is None:
        base_config = vllm_config
    if base_config is None:
        return None

    speculative_config = getattr(base_config, "speculative_config", None)
    draft_model_config = getattr(speculator, "draft_model_config", None)
    if draft_model_config is None:
        draft_model_config = getattr(speculative_config, "draft_model_config", None)
    if draft_model_config is None:
        return base_config

    draft_parallel = None
    # Standalone AR loads from the declared topology. Other loaders derive
    # their runtime parallel settings from the target.
    if getattr(speculative_config, "method", None) == "draft_model":
        draft_parallel = getattr(speculative_config, "draft_parallel_config", None)
        draft_tp = getattr(speculative_config, "draft_tensor_parallel_size", None)
        if draft_parallel is None and draft_tp is not None:
            draft_parallel = SpeculativeConfig.create_draft_parallel_config(
                base_config.parallel_config, draft_tp, draft_model_config
            )
    if draft_parallel is None and getattr(base_config, "model_config", None) is draft_model_config:
        return base_config

    resolved = copy.copy(base_config)
    resolved.model_config = draft_model_config
    parallel_config = draft_parallel if draft_parallel is not None else getattr(base_config, "parallel_config", None)
    draft_is_moe = getattr(draft_model_config, "is_moe", None)
    if (
        parallel_config is not None
        and isinstance(draft_is_moe, bool)
        and getattr(parallel_config, "is_moe_model", None) is not draft_is_moe
    ):
        parallel_config = copy.copy(parallel_config)
        parallel_config.is_moe_model = draft_is_moe
    if parallel_config is not None:
        resolved.parallel_config = parallel_config
    if (
        getattr(draft_model_config, "quantization", None) is None
        and getattr(base_config, "quant_config", None) is not None
    ):
        resolved.quant_config = None
    if speculator is not None:
        with suppress(Exception):
            speculator._ascend_resolved_draft_vllm_config = resolved
    return resolved


def _needs_draft_moe_comm_isolation(speculator, vllm_config: VllmConfig) -> bool:
    """Return False only when both target and draft are provably non-MoE."""
    target_model_cfg = getattr(vllm_config, "model_config", None)
    target_parallel_cfg = getattr(vllm_config, "parallel_config", None)
    target_is_moe = getattr(target_model_cfg, "is_moe", None)
    if not isinstance(target_is_moe, bool):
        target_is_moe = getattr(target_parallel_cfg, "is_moe_model", None)

    spec_cfg = getattr(vllm_config, "speculative_config", None)
    draft_model_cfg = getattr(speculator, "draft_model_config", None) or getattr(spec_cfg, "draft_model_config", None)
    draft_is_moe = getattr(draft_model_cfg, "is_moe", None)
    return not (target_is_moe is False and draft_is_moe is False)


def _wrap_speculator_draft_model_context(speculator, vllm_config: VllmConfig):
    """Ensure ``speculator.model`` isolates draft MoE comm across eager, PIECEWISE, and capture."""
    if not _needs_draft_moe_comm_isolation(speculator, vllm_config):
        return speculator
    load_model = getattr(speculator, "load_model", None)
    if not callable(load_model):
        return speculator

    @wraps(load_model)
    def wrapped_load_model(*args, **kwargs):
        draft_config = getattr(speculator, "draft_vllm_config", None)
        parallel_config = getattr(draft_config, "parallel_config", None)
        if draft_config is not None and parallel_config is not None:
            # The upstream loader can revalidate a target-derived config.
            draft_config.parallel_config = copy.copy(parallel_config)
        try:
            with draft_config_context():
                result = load_model(*args, **kwargs)
        finally:
            if draft_config is not None and parallel_config is not None:
                draft_config.parallel_config = parallel_config
        draft_model = getattr(speculator, "model", None)
        if isinstance(draft_model, nn.Module):
            bind_draft_moe_comm_context(draft_model, resolve_speculator_draft_vllm_config(speculator, vllm_config))
        return result

    speculator.load_model = wrapped_load_model
    init_cudagraph_manager = getattr(speculator, "init_cudagraph_manager", None)
    if callable(init_cudagraph_manager):

        @wraps(init_cudagraph_manager)
        def wrapped_init_cudagraph_manager(*args, **kwargs):
            token = _ACTIVE_SPECULATOR_CONTEXT.set(speculator)
            try:
                return init_cudagraph_manager(*args, **kwargs)
            finally:
                _ACTIVE_SPECULATOR_CONTEXT.reset(token)

        speculator.init_cudagraph_manager = wrapped_init_cudagraph_manager
    return speculator


def init_speculator(
    vllm_config: VllmConfig,
    device: torch.device,
    req_states: "RequestState | None" = None,
):
    """Override GPU init_speculator for Ascend NPUs.
    Use AscendEagleSpeculator when eagle is used.
    """
    # TODO: Align init_speculator dispatch with upstream vLLM by mapping the
    # resolved speculator class to its Ascend implementation.
    speculative_config = vllm_config.speculative_config
    assert speculative_config is not None
    if speculative_config.enforce_eager:
        vllm_config = copy.copy(vllm_config)
        vllm_config.compilation_config = copy.copy(vllm_config.compilation_config)
        vllm_config.compilation_config.mode = CompilationMode.NONE
    if speculative_config.method in ("ngram", "ngram_gpu"):
        if req_states is None:
            raise ValueError("MRV2 n-gram speculative decoding requires request states")
        # Older vLLM pins lack this module; MRV2 n-gram requires upstream #40704.
        from vllm.v1.worker.gpu.spec_decode.ngram.speculator import (  # type: ignore[import-not-found]
            NgramGPUSpeculator,
        )

        # MRV2 postprocess has already updated history. The MRV1 NPU kernel
        # appends sampled tokens itself and cannot be used here unchanged.
        return NgramGPUSpeculator(vllm_config, device, req_states)
    if speculative_config.method == "extract_hidden_states":
        # No Ascend-specific behavior beyond update_stream assignment in
        # NPUModelRunner; reuse upstream ExtractHiddenStatesSpeculator as-is.
        from vllm.v1.worker.gpu.spec_decode.extract_hidden_states import (
            ExtractHiddenStatesSpeculator,
        )

        return ExtractHiddenStatesSpeculator(vllm_config, device)
    if speculative_config.use_dspark():
        from vllm_ascend.worker.v2.spec_decode.dspark.speculator import (
            AscendDSparkSpeculator,
        )

        return _wrap_speculator_draft_model_context(AscendDSparkSpeculator(vllm_config, device), vllm_config)
    if speculative_config.use_dflash():
        if "LiLiCorrDraftModel" in speculative_config.draft_model_config.architectures:
            from vllm_ascend.worker.v2.spec_decode.lilicorr.speculator import (
                AscendLiLiCorrSpeculator,
            )

            return _wrap_speculator_draft_model_context(AscendLiLiCorrSpeculator(vllm_config, device), vllm_config)
        if "DFlash2DraftModel" in speculative_config.draft_model_config.architectures:
            from vllm_ascend.worker.v2.spec_decode.dflash2.speculator import (
                AscendDFlash2Speculator,
            )

            return _wrap_speculator_draft_model_context(AscendDFlash2Speculator(vllm_config, device), vllm_config)
        from vllm_ascend.worker.v2.spec_decode.dflash.speculator import (
            AscendDFlashSpeculator,
        )

        return _wrap_speculator_draft_model_context(AscendDFlashSpeculator(vllm_config, device), vllm_config)
    if speculative_config.use_gemma4_mtp():
        from vllm_ascend.worker.v2.spec_decode.gemma4.speculator import (
            AscendGemma4Speculator,
        )

        return _wrap_speculator_draft_model_context(AscendGemma4Speculator(vllm_config, device), vllm_config)
    if speculative_config.use_multi_module_mtp():
        from vllm_ascend.worker.v2.spec_decode.multi_module_mtp.speculator import AscendMultiModuleMTPSpeculator

        return _wrap_speculator_draft_model_context(AscendMultiModuleMTPSpeculator(vllm_config, device), vllm_config)
    if speculative_config.method == "mtp":
        from vllm_ascend.worker.v2.spec_decode.mtp.speculator import (
            AscendMTPSpeculator,
        )

        return _wrap_speculator_draft_model_context(AscendMTPSpeculator(vllm_config, device), vllm_config)
    if speculative_config.use_eagle():
        from vllm_ascend.worker.v2.spec_decode.eagle.speculator import AscendEagleSpeculator

        return _wrap_speculator_draft_model_context(AscendEagleSpeculator(vllm_config, device), vllm_config)
    if speculative_config.uses_draft_model():
        from vllm_ascend.worker.v2.spec_decode.standalone_ar.speculator import (
            AscendStandaloneARSpeculator,
        )

        return _wrap_speculator_draft_model_context(AscendStandaloneARSpeculator(vllm_config, device), vllm_config)
    raise NotImplementedError(f"{speculative_config.method} is not supported yet.")
