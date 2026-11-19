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
from vllm.config import VllmConfig, get_layers_from_vllm_config
from vllm.model_executor.layers.attention import Attention, CrossAttention
from vllm.v1.attention.backend import AttentionType
from vllm.v1.worker.gpu.mm.encoder_cache import EncoderCache

from vllm_ascend.device.hardware_profile import HardwareCapability, get_current_hardware_profile


def _has_encoder_only_layers(vllm_config: VllmConfig) -> bool:
    """Whether the model has non-causal, KV-cache-less attention layers."""
    return any(
        layer.attn_type == AttentionType.ENCODER_ONLY
        for layer in get_layers_from_vllm_config(vllm_config, Attention).values()
    )


def init_asecnd_model_state(
    vllm_config: VllmConfig,
    model: nn.Module,
    encoder_cache: EncoderCache | None,
    device: torch.device,
):
    cls = _resolve_ascend_model_state_cls(vllm_config, model)

    # Upstream ``init_model_state`` rejects this combination, but this function
    # replaces it wholesale on Ascend, so the guard has to be repeated here.
    if getattr(vllm_config.model_config, "enable_prompt_embeds", False) and not cls.supports_prompt_embeds:
        raise ValueError(f"--enable-prompt-embeds not supported with {cls.__name__}.")

    return cls(vllm_config, model, encoder_cache, device)


def _resolve_ascend_model_state_cls(
    vllm_config: VllmConfig,
    model: nn.Module,
):
    """Pick the Ascend model state class; mirrors upstream ``resolve_model_state_cls``."""
    # Keep model-provided state overrides ahead of the platform defaults.
    if hasattr(model, "get_model_state_cls"):
        return model.get_model_state_cls()

    # 310P uses Triton-free states under ``vllm_ascend._310p.worker.v2.model_state``.
    if vllm_config.model_config.is_hybrid:
        if not get_current_hardware_profile().supports(HardwareCapability.STANDARD_WORKER_PATCHES):
            # Hybrid models that also declare ENCODER_ONLY layers are not
            # adapted on 310P; that combination is owned by the 310P team.
            from vllm_ascend._310p.worker.v2.model_state import Ascend310PMambaHybridModelState

            return Ascend310PMambaHybridModelState

        # Hybrid models that also declare bidirectional, KV-cache-less layers
        # (ColQwen3.5: GDN layers + ENCODER_ONLY full attention) need both
        # mamba and encoder-only metadata, so the mamba state alone is not
        # enough.
        if _has_encoder_only_layers(vllm_config):
            from vllm_ascend.worker.v2.model_states.hybrid_encoder import (
                AscendHybridEncoderModelState,
            )

            return AscendHybridEncoderModelState

        from vllm_ascend.worker.v2.model_states.mamba_hybrid import (
            AscendMambaHybridModelState,
        )

        return AscendMambaHybridModelState

    if not get_current_hardware_profile().supports(HardwareCapability.STANDARD_WORKER_PATCHES):
        from vllm_ascend._310p.worker.v2.model_state import (
            Ascend310PEncoderOnlyModelState,
            Ascend310PModelState,
        )

        if _has_encoder_only_layers(vllm_config):
            return Ascend310PEncoderOnlyModelState

        return Ascend310PModelState

    if any(isinstance(module, CrossAttention) for module in model.modules()):
        from vllm_ascend.worker.v2.model_states.encoder_decoder import (
            AscendEncoderDecoderModelState,
        )

        return AscendEncoderDecoderModelState

    # Encoder-only attention (BERT/BGE) is non-causal and keeps no KV cache, so
    # it cannot use the KV-backed AscendModelState.
    if _has_encoder_only_layers(vllm_config):
        from vllm_ascend.worker.v2.model_states.encoder_only import (
            AscendEncoderOnlyModelState,
        )

        return AscendEncoderOnlyModelState

    from vllm_ascend.worker.v2.model_states.default import AscendModelState

    return AscendModelState
