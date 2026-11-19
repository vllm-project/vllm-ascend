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
import inspect
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, cast

import torch
from vllm.config import CompilationMode, VllmConfig
from vllm.utils.import_utils import resolve_obj_by_qualname

from vllm_ascend.utils import vllm_version_is

if TYPE_CHECKING:
    from vllm.v1.worker.gpu.states import RequestState

_NGRAM_SPECULATOR_QUALNAME = "vllm.v1.worker.gpu.spec_decode.ngram.speculator.NgramGPUSpeculator"

_UPSTREAM_TO_ASCEND_SPECULATOR: dict[str, str] = {
    _NGRAM_SPECULATOR_QUALNAME: _NGRAM_SPECULATOR_QUALNAME,
    "vllm.v1.worker.gpu.spec_decode.extract_hidden_states.ExtractHiddenStatesSpeculator": (
        "vllm.v1.worker.gpu.spec_decode.extract_hidden_states.ExtractHiddenStatesSpeculator"
    ),
    "vllm.v1.worker.gpu.spec_decode.dspark.speculator.DSparkSpeculator": (
        "vllm_ascend.worker.v2.spec_decode.dspark.speculator.AscendDSparkSpeculator"
    ),
    "vllm.v1.worker.gpu.spec_decode.lilicorr.speculator.LiLiCorrSpeculator": (
        "vllm_ascend.worker.v2.spec_decode.lilicorr.speculator.AscendLiLiCorrSpeculator"
    ),
    "vllm.v1.worker.gpu.spec_decode.dflash2.speculator.DFlash2Speculator": (
        "vllm_ascend.worker.v2.spec_decode.dflash2.speculator.AscendDFlash2Speculator"
    ),
    "vllm.v1.worker.gpu.spec_decode.dflash.speculator.DFlashSpeculator": (
        "vllm_ascend.worker.v2.spec_decode.dflash.speculator.AscendDFlashSpeculator"
    ),
    "vllm.v1.worker.gpu.spec_decode.gemma4.speculator.Gemma4Speculator": (
        "vllm_ascend.worker.v2.spec_decode.gemma4.speculator.AscendGemma4Speculator"
    ),
    "vllm.v1.worker.gpu.spec_decode.multi_module_mtp.speculator.MultiModuleMTPSpeculator": (
        "vllm_ascend.worker.v2.spec_decode.multi_module_mtp.speculator.AscendMultiModuleMTPSpeculator"
    ),
    "vllm.v1.worker.gpu.spec_decode.mtp.speculator.MTPSpeculator": (
        "vllm_ascend.worker.v2.spec_decode.mtp.speculator.AscendMTPSpeculator"
    ),
    "vllm.v1.worker.gpu.spec_decode.eagle.speculator.EagleSpeculator": (
        "vllm_ascend.worker.v2.spec_decode.eagle.speculator.AscendEagleSpeculator"
    ),
    "vllm.v1.worker.gpu.spec_decode.standalone_ar.speculator.StandaloneARSpeculator": (
        "vllm_ascend.worker.v2.spec_decode.standalone_ar.speculator.AscendStandaloneARSpeculator"
    ),
}


def _init_speculator_v033(
    vllm_config: VllmConfig,
    device: torch.device,
    req_states: "RequestState | None" = None,
) -> Any:
    """Legacy predicate-based speculator dispatch for the v0.33.0 compatibility pin."""
    speculative_config = vllm_config.speculative_config
    assert speculative_config is not None
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
    if speculative_config.use_multi_module_mtp():
        from vllm_ascend.worker.v2.spec_decode.multi_module_mtp.speculator import AscendMultiModuleMTPSpeculator

        return AscendMultiModuleMTPSpeculator(vllm_config, device)
    if speculative_config.method == "mtp":
        from vllm_ascend.worker.v2.spec_decode.mtp.speculator import (
            AscendMTPSpeculator,
        )

        return AscendMTPSpeculator(vllm_config, device)
    if speculative_config.use_eagle():
        from vllm_ascend.worker.v2.spec_decode.eagle.speculator import AscendEagleSpeculator

        return AscendEagleSpeculator(vllm_config, device)
    if speculative_config.uses_draft_model():
        from vllm_ascend.worker.v2.spec_decode.standalone_ar.speculator import (
            AscendStandaloneARSpeculator,
        )

        return AscendStandaloneARSpeculator(vllm_config, device)
    raise NotImplementedError(f"{speculative_config.method} is not supported yet.")


def init_speculator(
    vllm_config: VllmConfig,
    device: torch.device,
    req_states: "RequestState | None" = None,
) -> Any:
    """Build the Ascend or upstream MRV2 speculator selected by vllm_config."""
    speculative_config = vllm_config.speculative_config
    assert speculative_config is not None
    if speculative_config.enforce_eager:
        vllm_config = copy.copy(vllm_config)
        vllm_config.compilation_config = copy.copy(vllm_config.compilation_config)
        vllm_config.compilation_config.mode = CompilationMode.NONE
    if vllm_version_is("0.33.0"):
        return _init_speculator_v033(vllm_config, device, req_states)

    speculator_config = speculative_config.speculator_config
    qualname = speculator_config.speculator_class
    if qualname is None:
        raise NotImplementedError(f"{speculative_config.method} is not supported yet.")

    target_qualname = _UPSTREAM_TO_ASCEND_SPECULATOR.get(qualname)
    if target_qualname is not None:
        factory = cast(Callable[..., Any], resolve_obj_by_qualname(target_qualname))
        if target_qualname == _NGRAM_SPECULATOR_QUALNAME:
            if req_states is None:
                raise ValueError("MRV2 n-gram speculative decoding requires request states")
            # MRV2 postprocess has already updated history. The MRV1 NPU kernel
            # appends sampled tokens itself and cannot be used here unchanged.
            return factory(vllm_config, device, req_states)
        return factory(vllm_config, device)

    custom_method = getattr(speculative_config, "custom_method", None)
    if custom_method is not None:
        from vllm.v1.worker.gpu.spec_decode.speculator import BaseSpeculator

        speculator_cls = resolve_obj_by_qualname(qualname)
        if not (isinstance(speculator_cls, type) and issubclass(speculator_cls, BaseSpeculator)):
            raise TypeError(
                f"speculator_class '{qualname}' of {type(speculator_config).__name__} "
                f"must be a subclass of {BaseSpeculator.__module__}."
                f"{BaseSpeculator.__qualname__}."
            )
        factory = cast(Callable[..., Any], speculator_cls)
        if "req_states" in inspect.signature(factory).parameters:
            if req_states is None:
                raise ValueError(f"MRV2 {custom_method} speculative decoding requires request states")
            return factory(vllm_config, device, req_states=req_states)
        return factory(vllm_config, device)

    raise NotImplementedError(f"{speculative_config.method} is not supported yet.")
