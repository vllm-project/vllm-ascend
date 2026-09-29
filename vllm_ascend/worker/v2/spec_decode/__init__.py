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
import torch
from vllm.config import VllmConfig


def init_speculator(
    vllm_config: VllmConfig,
    device: torch.device,
):
    """Override GPU init_speculator for Ascend NPUs.
    Use AscendEagleSpeculator when eagle is used.
    """
    speculative_config = vllm_config.speculative_config
    assert speculative_config is not None
    if speculative_config.method == "extract_hidden_states":
        # No Ascend-specific behavior beyond update_stream assignment in
        # NPUModelRunner; reuse upstream ExtractHiddenStatesSpeculator as-is.
        from vllm.v1.worker.gpu.spec_decode.extract_hidden_states import (
            ExtractHiddenStatesSpeculator,
        )

        return ExtractHiddenStatesSpeculator(vllm_config, device)
    # H-Spec (mamba_attn_hybrid) exists only in vLLM builds carrying the
    # H-Spec patch; guard with getattr so vLLM versions without the method
    # never hit an AttributeError here, regardless of the requested method.
    use_mamba_hybrid = getattr(speculative_config, "use_mamba_attn_hybrid", None)
    if callable(use_mamba_hybrid) and use_mamba_hybrid():
        try:
            from vllm_ascend.worker.v2.spec_decode.mamba_attn_hybrid.speculator import (
                AscendMambaAttnHybridSpeculator,
            )
        except ImportError as e:
            raise NotImplementedError(
                "mamba_attn_hybrid requires a vLLM build with H-Spec support "
                "(missing vllm.v1.worker.gpu.spec_decode.mamba_attn_hybrid); "
                f"underlying import error: {e}"
            ) from e
        return AscendMambaAttnHybridSpeculator(vllm_config, device)
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
    if (
        speculative_config.method == "mtp"
        and not speculative_config.use_gemma4_mtp()
        and not speculative_config.use_step3p5_mtp()
    ):
        from vllm_ascend.worker.v2.spec_decode.mtp.speculator import (
            AscendMTPSpeculator,
        )

        return AscendMTPSpeculator(vllm_config, device)
    if speculative_config.use_eagle():
        from vllm_ascend.worker.v2.spec_decode.eagle.speculator import AscendEagleSpeculator

        return AscendEagleSpeculator(vllm_config, device)
    raise NotImplementedError(f"{speculative_config.method} is not supported yet.")
