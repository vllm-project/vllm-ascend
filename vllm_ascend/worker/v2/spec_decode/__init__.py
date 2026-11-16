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

from vllm_ascend.worker.spec_decode_family import (
    SpecDecodeFamily,
    resolve_spec_decode_family,
)

if TYPE_CHECKING:
    from vllm.v1.worker.gpu.states import RequestState


def init_speculator(
    vllm_config: VllmConfig,
    device: torch.device,
    req_states: "RequestState | None" = None,
):
    """Override GPU init_speculator for Ascend NPUs.
    Use AscendEagleSpeculator when eagle is used.
    """
    speculative_config = vllm_config.speculative_config
    assert speculative_config is not None
    family = resolve_spec_decode_family(vllm_config)
    if speculative_config.enforce_eager:
        vllm_config = copy.copy(vllm_config)
        vllm_config.compilation_config = copy.copy(vllm_config.compilation_config)
        vllm_config.compilation_config.mode = CompilationMode.NONE
    if family in (SpecDecodeFamily.NGRAM, SpecDecodeFamily.NGRAM_GPU):
        if req_states is None:
            raise ValueError("MRV2 n-gram speculative decoding requires request states")
        # Older vLLM pins lack this module; MRV2 n-gram requires upstream #40704.
        from vllm.v1.worker.gpu.spec_decode.ngram.speculator import (  # type: ignore[import-not-found]
            NgramGPUSpeculator,
        )

        # MRV2 postprocess has already updated history. The MRV1 NPU kernel
        # appends sampled tokens itself and cannot be used here unchanged.
        return NgramGPUSpeculator(vllm_config, device, req_states)
    if family is SpecDecodeFamily.EXTRACT_HIDDEN_STATES:
        # No Ascend-specific behavior beyond update_stream assignment in
        # NPUModelRunner; reuse upstream ExtractHiddenStatesSpeculator as-is.
        from vllm.v1.worker.gpu.spec_decode.extract_hidden_states import (
            ExtractHiddenStatesSpeculator,
        )

        return ExtractHiddenStatesSpeculator(vllm_config, device)
    if family is SpecDecodeFamily.DSPARK:
        from vllm_ascend.worker.v2.spec_decode.dspark.speculator import (
            AscendDSparkSpeculator,
        )

        return AscendDSparkSpeculator(vllm_config, device)
    if family is SpecDecodeFamily.LILICORR:
        from vllm_ascend.worker.v2.spec_decode.lilicorr.speculator import AscendLiLiCorrSpeculator

        return AscendLiLiCorrSpeculator(vllm_config, device)
    if family is SpecDecodeFamily.DFLASH2:
        from vllm_ascend.worker.v2.spec_decode.dflash2.speculator import (
            AscendDFlash2Speculator,
        )

        return AscendDFlash2Speculator(vllm_config, device)
    if family is SpecDecodeFamily.DFLASH:
        from vllm_ascend.worker.v2.spec_decode.dflash.speculator import (
            AscendDFlashSpeculator,
        )

        return AscendDFlashSpeculator(vllm_config, device)
    if family is SpecDecodeFamily.GEMMA4_MTP:
        from vllm_ascend.worker.v2.spec_decode.gemma4.speculator import (
            AscendGemma4Speculator,
        )

        return AscendGemma4Speculator(vllm_config, device)
    if family is SpecDecodeFamily.MTP:
        from vllm_ascend.worker.v2.spec_decode.mtp.speculator import (
            AscendMTPSpeculator,
        )

        return AscendMTPSpeculator(vllm_config, device)
    if family is SpecDecodeFamily.MULTI_MODULE_MTP:
        from vllm_ascend.worker.v2.spec_decode.multi_module_mtp.speculator import AscendMultiModuleMTPSpeculator

        return AscendMultiModuleMTPSpeculator(vllm_config, device)
    if family is SpecDecodeFamily.EAGLE:
        from vllm_ascend.worker.v2.spec_decode.eagle.speculator import AscendEagleSpeculator

        return AscendEagleSpeculator(vllm_config, device)
    if family is SpecDecodeFamily.DRAFT_MODEL:
        from vllm_ascend.worker.v2.spec_decode.standalone_ar.speculator import (
            AscendStandaloneARSpeculator,
        )

        return AscendStandaloneARSpeculator(vllm_config, device)
    raise NotImplementedError(f"{speculative_config.method} is not supported yet.")
