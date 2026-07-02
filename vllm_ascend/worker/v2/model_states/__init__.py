# Adapt from https://github.com/vllm-project/vllm/blob/main/vllm/v1/worker/gpu/model_states/__init__.py
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
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.v1.worker.gpu.mm.encoder_cache import EncoderCache


def init_asecnd_model_state(
    vllm_config: VllmConfig,
    model: nn.Module,
    encoder_cache: EncoderCache | None,
    device: torch.device,
):
    # DiffusionGemma exposes a custom upstream ModelState for block-diffusion
    # canvas denoising. Route only that known state to the Ascend-adapted
    # implementation; other models keep the existing AscendModelState fallback.
    get_state_cls = getattr(model, "get_model_state_cls", None)
    if callable(get_state_cls):
        state_cls = get_state_cls()
        if state_cls is not None:
            name = getattr(state_cls, "__name__", "")
            if name == "DiffusionGemmaModelState":
                from vllm_ascend.worker.v2.model_states.diffusion_gemma import (
                    AscendDiffusionGemmaModelState,
                )

                return AscendDiffusionGemmaModelState(vllm_config, model, encoder_cache, device)

    from vllm_ascend.worker.v2.model_states.default import AscendModelState

    return AscendModelState(vllm_config, model, encoder_cache, device)
