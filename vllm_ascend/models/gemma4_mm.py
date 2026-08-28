# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
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

"""Ascend-specific Gemma4 multimodal model adaptations."""

import torch
from transformers import AutoModel
from vllm.config import VllmConfig
from vllm.model_executor.models.gemma4 import Gemma4ForCausalLM
from vllm.model_executor.models.gemma4_mm import (
    Gemma4ForConditionalGeneration,
    Gemma4MultimodalEmbedder,
)
from vllm.model_executor.models.transformers.utils import recursive_replace_linear
from vllm.model_executor.models.utils import init_vllm_registered_model, maybe_prefix

from vllm_ascend.quantization.methods import (
    AscendW4A4MXFP4DynamicLinearMethod,
    AscendW8A8MXFP8DynamicLinearMethod,
)

# NOTE: The upstream Gemma4 vision patch embedder may cast pixel_values to the
# projection weight dtype. For Ascend MXFP4/MXFP8 dynamic quantization, the
# packed/quantized weight dtype is not the logical activation dtype expected
# by input_proj. Keep the patch-embedder activation in model_dtype and let the
# Ascend quantized linear method perform activation quantization internally.


def _ascend_gemma4_vision_patch_embedder_forward(
    module,
    pixel_values: torch.Tensor,
    pixel_position_ids: torch.Tensor,
    padding_positions: torch.Tensor,
) -> torch.Tensor:
    pixel_values = 2 * (pixel_values - 0.5)
    pixel_values = pixel_values.to(module._ascend_activation_dtype)
    hidden_states = module.input_proj(pixel_values)
    position_embeddings = module._position_embeddings(pixel_position_ids, padding_positions)
    return hidden_states + position_embeddings.to(hidden_states.dtype)


def _patch_gemma4_vision_patch_embedder(
    patch_embedder: torch.nn.Module,
    activation_dtype: torch.dtype,
) -> None:
    patch_embedder._ascend_activation_dtype = activation_dtype
    patch_embedder.forward = _ascend_gemma4_vision_patch_embedder_forward.__get__(
        patch_embedder,
        patch_embedder.__class__,
    )


class AscendGemma4ForConditionalGeneration(Gemma4ForConditionalGeneration):
    """Gemma4 multimodal model with ModelSlim quantization for its towers."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        # Keep this constructor aligned with upstream Gemma4 multimodal
        # initialization. The Ascend-specific changes are intentionally limited
        # to tower quant_config injection and the MXFP4 vision patch-embedder
        # activation dtype workaround below.
        torch.nn.Module.__init__(self)
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        multimodal_config = vllm_config.model_config.multimodal_config
        self.config = config
        self.quant_config = quant_config
        self.multimodal_config = multimodal_config
        self.model_dtype = vllm_config.model_config.dtype
        self.vllm_config = vllm_config

        with self._mark_tower_model(vllm_config, {"image", "video"}):
            self.vision_tower = AutoModel.from_config(config=config.vision_config)
            self.embed_vision = Gemma4MultimodalEmbedder(
                config.vision_config,
                config.text_config,
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "embed_vision"),
            )
            recursive_replace_linear(
                self.vision_tower,
                quant_config,
                prefix=maybe_prefix(prefix, "vision_tower"),
            )
            linear_method = self.vision_tower.patch_embedder.input_proj.quant_method
            quant_method = getattr(linear_method, "quant_method", linear_method)

            if isinstance(
                quant_method,
                (
                    AscendW4A4MXFP4DynamicLinearMethod,
                    AscendW8A8MXFP8DynamicLinearMethod,
                ),
            ):
                _patch_gemma4_vision_patch_embedder(
                    self.vision_tower.patch_embedder,
                    self.model_dtype,
                )

        if config.audio_config is not None:
            with self._mark_tower_model(vllm_config, "audio"):
                self.audio_tower = AutoModel.from_config(config=config.audio_config)
                self.audio_tower.post_init()
                self.embed_audio = Gemma4MultimodalEmbedder(
                    config.audio_config,
                    config.text_config,
                    quant_config=quant_config,
                    prefix=maybe_prefix(prefix, "embed_audio"),
                )
                recursive_replace_linear(
                    self.audio_tower,
                    quant_config,
                    prefix=maybe_prefix(prefix, "audio_tower"),
                )
        else:
            self.audio_tower = None
            self.embed_audio = None

        with self._mark_language_model(vllm_config):
            self.language_model: Gemma4ForCausalLM = init_vllm_registered_model(
                vllm_config=vllm_config,
                hf_config=config.text_config,
                prefix=maybe_prefix(prefix, "language_model"),
                architectures=["Gemma4ForCausalLM"],
            )

            ple_dim = config.text_config.hidden_size_per_layer_input
            if ple_dim is not None and ple_dim > 0:
                embed = self.language_model.model.embed_tokens
                self.per_layer_embeddings = torch.zeros(
                    vllm_config.scheduler_config.max_num_batched_tokens,
                    config.text_config.num_hidden_layers,
                    ple_dim,
                    device=next(embed.parameters()).device,
                    dtype=vllm_config.model_config.dtype,
                )
            else:
                self.per_layer_embeddings = None

        self.make_empty_intermediate_tensors = self.language_model.make_empty_intermediate_tensors

        self._full_attn_layer_idxs: frozenset[int] = frozenset()
        text_config = config.text_config
        if getattr(text_config, "use_bidirectional_attention", None) == "vision":
            layer_types = getattr(text_config, "layer_types", None)
            if layer_types:
                self._full_attn_layer_idxs = frozenset(
                    i for i, layer_type in enumerate(layer_types) if layer_type != "sliding_attention"
                )

        self.moe_layers = self.language_model.moe_layers
        self.num_moe_layers = self.language_model.num_moe_layers
        self.num_logical_experts = self.language_model.num_logical_experts
        self.num_physical_experts = self.language_model.num_physical_experts
        self.num_local_physical_experts = self.language_model.num_local_physical_experts
        self.num_routed_experts = self.language_model.num_routed_experts
        self.num_expert_groups = self.language_model.num_expert_groups
        self.num_shared_experts = self.language_model.num_shared_experts
        self.num_redundant_experts = self.language_model.num_redundant_experts

        generation_config = vllm_config.model_config.try_get_generation_config()
        self._suppress_token_ids = generation_config.get("suppress_tokens") if generation_config else None
