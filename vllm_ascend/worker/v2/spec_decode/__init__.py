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
from typing import TYPE_CHECKING, Any

import torch
from vllm.config import CompilationMode, VllmConfig
from vllm.logger import logger

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.worker.v2.spec_decode.draft_window_policy import (
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


def _resolve_window_policy(vllm_config: VllmConfig) -> int | None:
    requested = get_ascend_config().draft_window_size
    if requested is None:
        return None
    speculative_config = vllm_config.speculative_config
    method = speculative_config.method
    # Forced OFF must precede numerical checks and accesses to optional model
    # fields, dynamic-K state, or backend capabilities.
    if method == "mtp":
        forced_off: str | None = "MTP"
    else:
        target = vllm_config.model_config
        target_model_type = target.hf_config.model_type
        builtin_v4 = _builtin_v4_dspark_identity(speculative_config, target) if method == "dspark" else None
        forced_off = "the built-in DeepSeek V4 DSpark route" if builtin_v4 is True else None
    if forced_off is not None:
        logger.warning(
            "MRV2 draft_window_size is ignored for %s; Draft Sliding Window is disabled. "
            "The original speculative method is retained.",
            forced_off,
        )
        return None
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


def _use_graph_window_owner(vllm_config: VllmConfig, policy: int | None, support: Any) -> bool:
    """Keep the ordinary eager runtime and persistent Graph owner exclusive."""
    if policy is None or vllm_config.speculative_config.enforce_eager:
        return False
    from vllm_ascend import envs

    if not envs.VLLM_ASCEND_EXPERIMENTAL_DRAFT_WINDOW_GRAPH:
        raise NotImplementedError("Draft Window Graph requires its explicit experimental Graph entry")
    from vllm_ascend.ascend_config import validate_draft_window_graph_config

    validate_draft_window_graph_config(vllm_config, support=support)
    return True


def init_speculator(vllm_config: VllmConfig, device: torch.device, req_states: "RequestState | None" = None):
    """Resolve PR method policy before choosing typed eager or Graph ownership."""
    speculative_config = vllm_config.speculative_config
    assert speculative_config is not None
    review_limit = None
    if getattr(speculative_config.draft_model_config, "model", None) is not None:
        from vllm_ascend import envs
        from vllm_ascend.worker.v2.spec_decode.draft_graph_review_limits import resolve_draft_graph_review_limit

        review_limit = resolve_draft_graph_review_limit(
            vllm_config, experimental_graph_review=envs.VLLM_ASCEND_EXPERIMENTAL_DRAFT_WINDOW_GRAPH
        )
    policy = _resolve_window_policy(vllm_config)
    use_graph_owner = _use_graph_window_owner(vllm_config, policy, review_limit)
    if review_limit is not None and req_states is None:
        raise ValueError("Qualified Draft Graph requires the runner's actual RequestState")
    speculator = _init_speculator(vllm_config, device, req_states)
    if review_limit is not None:
        speculator._draft_graph_review_limit = review_limit
        speculator._draft_graph_request_state = req_states
        logger.info_once(
            "Draft Graph support: %s, TP4/DP%d, local request capacity %d, static K%d, "
            "model capacity %d; actual request/padded shapes are checked before execution.",
            speculative_config.method,
            vllm_config.parallel_config.data_parallel_size,
            vllm_config.scheduler_config.max_num_seqs,
            speculative_config.num_speculative_tokens,
            vllm_config.model_config.max_model_len,
        )
    if policy is not None:
        if use_graph_owner:
            if any(
                hasattr(speculator, name)
                for name in ("_draft_window_policy", "draft_window_policy", "_draft_window_runtime")
            ):
                raise RuntimeError("Graph owner and typed eager runtime must be mutually exclusive")
            speculator._draft_window_policy = policy
        else:
            speculator.draft_window_policy = policy
    return speculator


def initialize_draft_window_owner(owner) -> None:
    """Install the resolved read owner once, after actual Draft attention exists."""
    from vllm_ascend.worker.v2.spec_decode.draft_window_native_attention import report_owner_draft_native_attention

    report_owner_draft_native_attention(owner)
    graph = getattr(owner, "_draft_window_policy", None)
    eager = getattr(owner, "draft_window_policy", None)
    if graph is not None and eager is not None:
        raise RuntimeError("Draft Window eager and Graph owners must be mutually exclusive")
    if graph is not None or eager is not None:
        if hasattr(owner, "_draft_window_runtime") or hasattr(owner, "_draft_window_graph_binding_provider"):
            raise RuntimeError("Draft Window ownership is already initialized")
        if graph is not None:
            from vllm_ascend.worker.v2.spec_decode.draft_window import create_draft_window_graph_provider

            owner._draft_window_graph_binding_provider = create_draft_window_graph_provider(owner)
            if owner.method == "eagle3":
                owner._initialize_eagle3_draft_window_state()
            else:
                owner._draft_window_dp_dummy_active = False
        else:
            from vllm_ascend.worker.v2.spec_decode.draft_window_runtime import DraftWindowRuntime

            owner._draft_window_runtime = DraftWindowRuntime(owner, eager)


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
        if "LiLiCorrDraftModel" in speculative_config.draft_model_config.architectures:
            from vllm_ascend.worker.v2.spec_decode.lilicorr.speculator import (
                AscendLiLiCorrSpeculator,
            )

            return AscendLiLiCorrSpeculator(vllm_config, device)
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
