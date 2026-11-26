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
from vllm.utils.import_utils import resolve_obj_by_qualname
from vllm.v1.worker.gpu.spec_decode.dflash.config import DFlashSpeculatorConfig
from vllm.v1.worker.gpu.spec_decode.dflash2.config import DFlash2SpeculatorConfig
from vllm.v1.worker.gpu.spec_decode.dspark.config import DSparkSpeculatorConfig
from vllm.v1.worker.gpu.spec_decode.eagle.config import EagleSpeculatorConfig
from vllm.v1.worker.gpu.spec_decode.extract_hidden_states_config import ExtractHiddenStatesSpeculatorConfig
from vllm.v1.worker.gpu.spec_decode.gemma4.config import Gemma4MTPSpeculatorConfig
from vllm.v1.worker.gpu.spec_decode.lilicorr.config import LiLiCorrSpeculatorConfig
from vllm.v1.worker.gpu.spec_decode.mtp.config import MTPSpeculatorConfig
from vllm.v1.worker.gpu.spec_decode.multi_module_mtp.config import MultiModuleMTPSpeculatorConfig
from vllm.v1.worker.gpu.spec_decode.ngram.config import NgramSpeculatorConfig
from vllm.v1.worker.gpu.spec_decode.standalone_ar.config import StandaloneARSpeculatorConfig

if TYPE_CHECKING:
    from vllm.v1.worker.gpu.states import RequestState


# Keep backend imports lazy. The draft checkpoint and method are resolved by vLLM.
_ASCEND_SPECULATORS = {
    DFlashSpeculatorConfig.speculator_class: f"{__name__}.dflash.speculator.AscendDFlashSpeculator",
    DFlash2SpeculatorConfig.speculator_class: f"{__name__}.dflash2.speculator.AscendDFlash2Speculator",
    DSparkSpeculatorConfig.speculator_class: f"{__name__}.dspark.speculator.AscendDSparkSpeculator",
    EagleSpeculatorConfig.speculator_class: f"{__name__}.eagle.speculator.AscendEagleSpeculator",
    Gemma4MTPSpeculatorConfig.speculator_class: f"{__name__}.gemma4.speculator.AscendGemma4Speculator",
    LiLiCorrSpeculatorConfig.speculator_class: f"{__name__}.lilicorr.speculator.AscendLiLiCorrSpeculator",
    MTPSpeculatorConfig.speculator_class: f"{__name__}.mtp.speculator.AscendMTPSpeculator",
    MultiModuleMTPSpeculatorConfig.speculator_class: (
        f"{__name__}.multi_module_mtp.speculator.AscendMultiModuleMTPSpeculator"
    ),
    StandaloneARSpeculatorConfig.speculator_class: f"{__name__}.standalone_ar.speculator.AscendStandaloneARSpeculator",
}


def init_speculator(
    vllm_config: VllmConfig,
    device: torch.device,
    req_states: "RequestState | None" = None,
):
    """Build the NPU implementation of the speculator selected by vLLM."""
    speculative_config = vllm_config.speculative_config
    assert speculative_config is not None
    if speculative_config.enforce_eager:
        vllm_config = copy.copy(vllm_config)
        vllm_config.compilation_config = copy.copy(vllm_config.compilation_config)
        vllm_config.compilation_config.mode = CompilationMode.NONE
    qualname = speculative_config.speculator_config.speculator_class
    if qualname == NgramSpeculatorConfig.speculator_class:
        if req_states is None:
            raise ValueError("MRV2 n-gram speculative decoding requires request states")
        return resolve_obj_by_qualname(qualname)(vllm_config, device, req_states)
    if qualname == ExtractHiddenStatesSpeculatorConfig.speculator_class:
        return resolve_obj_by_qualname(qualname)(vllm_config, device)

    implementation = _ASCEND_SPECULATORS.get(qualname)
    if implementation is None:
        raise NotImplementedError(f"No Ascend MRV2 implementation for {speculative_config.method!r} ({qualname!r})")
    return resolve_obj_by_qualname(implementation)(vllm_config, device)
