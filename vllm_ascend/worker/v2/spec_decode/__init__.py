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

    raise NotImplementedError(
        f"Speculator class '{qualname}' (method='{speculative_config.method}') is not supported on Ascend MRV2 yet."
    )
