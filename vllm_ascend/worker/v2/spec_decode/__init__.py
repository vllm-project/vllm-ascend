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
from typing import TYPE_CHECKING

import torch
from vllm.config import CompilationMode, VllmConfig
from vllm.logger import logger

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.worker.v2.spec_decode.draft_window_policy import (
    DraftWindowPolicyDecision,
    DraftWindowPolicyState,
    resolve_draft_window_policy,
)

if TYPE_CHECKING:
    from vllm.v1.worker.gpu.states import RequestState


def _builtin_v4_dspark_identity(speculative_config, target_model_config) -> bool | None:
    """Identify the dedicated V4 DSpark category from normalized config.

    Upstream registers these architectures for V4 DSpark stages under mtp.*.
    Paths, revisions and packaging do not change that model category. This is
    route identification, not a claim to reconstruct checkpoint training history.
    """
    from vllm_ascend.utils import is_deepseek_v41

    draft = speculative_config.draft_model_config
    target_hf = target_model_config.hf_config
    model_type = "deepseek_v41" if is_deepseek_v41(target_hf) else target_hf.model_type
    expected_architectures = {
        "deepseek_v4": ("DSparkDraftModel",),
        # Ascend normalizes the upstream name after multimodal conversion.
        "deepseek_v41": ("DSparkV41DraftModel", "DeepseekV41DSparkModel"),
    }.get(model_type, ())
    if (
        bool(expected_architectures)
        and draft.hf_config.model_type == model_type
        and any(architecture in draft.architectures for architecture in expected_architectures)
    ):
        return True
    # Recognized external categories follow the switch even with a V4 target.
    # Compatibility with that target remains the model loader's contract.
    external_architectures = ("Qwen3DSparkModel", "Qwen3OmniDSparkModel", "Glm5DSparkForCausalLM")
    if any(architecture in draft.architectures for architecture in external_architectures):
        return False
    return None


def _resolve_window_policy(vllm_config: VllmConfig) -> DraftWindowPolicyDecision | None:
    requested = get_ascend_config().draft_window_size
    if requested is None:
        return None
    speculative_config = vllm_config.speculative_config
    method = speculative_config.method
    # Forced OFF must precede numerical checks and accesses to optional model
    # fields, dynamic-K state, or backend capabilities.
    if method == "mtp":
        return resolve_draft_window_policy(requested_window_size=requested, method=method)
    target = vllm_config.model_config
    target_model_type = target.hf_config.model_type
    builtin_v4 = _builtin_v4_dspark_identity(speculative_config, target) if method == "dspark" else None
    if builtin_v4 is True:
        return resolve_draft_window_policy(
            requested_window_size=requested,
            method=method,
            target_model_type=target_model_type,
            builtin_v4_dspark=True,
        )
    architectures = speculative_config.draft_model_config.architectures
    if method == "dflash" and (
        not architectures
        or len(set(architectures)) != 1
        or architectures[0] not in {"DFlashDraftModel", "DFlash2DraftModel"}
    ):
        raise NotImplementedError(
            "draft_window_size with method 'dflash' requires a supported "
            "DFlashDraftModel or DFlash2DraftModel architecture"
        )
    return resolve_draft_window_policy(
        requested_window_size=requested,
        method=method,
        target_model_type=target_model_type,
        builtin_v4_dspark=builtin_v4,
        dynamic_k=speculative_config.uses_dynamic_speculative_decoding(),
        is_dflash2="DFlash2DraftModel" in architectures,
    )


def init_speculator(vllm_config: VllmConfig, device: torch.device, req_states: "RequestState | None" = None):
    """Resolve Window policy before the original device-dependent dispatch."""
    speculative_config = vllm_config.speculative_config
    assert speculative_config is not None
    policy = _resolve_window_policy(vllm_config)
    if policy is not None and policy.enabled and speculative_config.enforce_eager is not True:
        raise NotImplementedError(
            "MRV2 draft_window_size requires speculative_config.enforce_eager=True; "
            "Draft Window graph execution is not supported. Target graph mode is independent."
        )
    speculator = _init_speculator(vllm_config, device, req_states)
    if policy is not None and policy.state is DraftWindowPolicyState.FORCED_OFF:
        logger.warning(
            "MRV2 draft_window_size is ignored for %s; Draft Sliding Window is disabled. "
            "The original speculative method is retained.",
            "MTP" if policy.reason == "mtp" else "the built-in DeepSeek V4 DSpark route",
        )
    if policy is not None and policy.enabled:
        speculator.draft_window_policy = policy
    return speculator


def _init_speculator(
    vllm_config: VllmConfig,
    device: torch.device,
    req_states: "RequestState | None" = None,
):
    """Override GPU init_speculator for Ascend NPUs.
    Use AscendEagleSpeculator when eagle is used.
    """
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

        return AscendDSparkSpeculator(vllm_config, device)
    if speculative_config.use_dflash():
        if "DFlash2DraftModel" in speculative_config.draft_model_config.architectures:
            from vllm_ascend.worker.v2.spec_decode.dflash2.speculator import (
                AscendDFlash2Speculator,
            )

            return AscendDFlash2Speculator(vllm_config, device)
        from vllm_ascend.worker.v2.spec_decode.dflash.speculator import (
            AscendDFlashSpeculator,
        )

        return AscendDFlashSpeculator(vllm_config, device)
    if speculative_config.use_gemma4_mtp():
        from vllm_ascend.worker.v2.spec_decode.gemma4.speculator import (
            AscendGemma4Speculator,
        )

        return AscendGemma4Speculator(vllm_config, device)
    if speculative_config.method == "mtp" and not speculative_config.use_step3p5_mtp():
        from vllm_ascend.worker.v2.spec_decode.mtp.speculator import (
            AscendMTPSpeculator,
        )

        return AscendMTPSpeculator(vllm_config, device)
    if speculative_config.use_multi_module_mtp():
        from vllm_ascend.worker.v2.spec_decode.multi_module_mtp.speculator import AscendMultiModuleMTPSpeculator

        return AscendMultiModuleMTPSpeculator(vllm_config, device)
    if speculative_config.use_eagle():
        from vllm_ascend.worker.v2.spec_decode.eagle.speculator import AscendEagleSpeculator

        return AscendEagleSpeculator(vllm_config, device)
    if speculative_config.uses_draft_model():
        from vllm_ascend.worker.v2.spec_decode.standalone_ar.speculator import (
            AscendStandaloneARSpeculator,
        )

        return AscendStandaloneARSpeculator(vllm_config, device)
    raise NotImplementedError(f"{speculative_config.method} is not supported yet.")
