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
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.models.gemma4 import Gemma4ForCausalLM
from vllm.model_executor.models.gemma4_mm import (
    Gemma4ForConditionalGeneration,
    Gemma4MultimodalEmbedder,
)
from vllm.model_executor.models.transformers.utils import recursive_replace_linear
from vllm.model_executor.models.utils import init_vllm_registered_model, maybe_prefix

from vllm_ascend.utils import ASCEND_QUANTIZATION_METHOD


def _get_tower_quant_config(
    vllm_config: VllmConfig,
    *,
    enable_ascend: bool = False,
) -> QuantizationConfig | None:
    """Select quantization for Gemma4 multimodal towers.

    When explicitly enabled for the vision tower, ModelSlim MXFP4 supports
    the Gemma4 ViT dimensions, including its 4304 intermediate size. Other
    towers and backends retain vLLM's divisibility guard.
    """
    quant_config = vllm_config.quant_config
    unrestricted_quant_methods = {
        "bitsandbytes",
        "torchao",
        "compressed-tensors",
    }
    if enable_ascend:
        unrestricted_quant_methods.add(ASCEND_QUANTIZATION_METHOD)
    if quant_config and quant_config.get_name() in unrestricted_quant_methods:
        return quant_config

    vision_config = vllm_config.model_config.hf_config.vision_config
    quantizable = vision_config.hidden_size % 64 == 0 and vision_config.intermediate_size % 64 == 0
    return quant_config if quantizable else None


class AscendGemma4ForConditionalGeneration(Gemma4ForConditionalGeneration):
    """Gemma4 multimodal model with ModelSlim quantization for its towers."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        # This mirrors the upstream constructor. The only intentional behavior
        # difference is that _get_tower_quant_config permits Ascend ModelSlim
        # quantization for the ViT's non-64-aligned intermediate dimension.
        torch.nn.Module.__init__(self)
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        multimodal_config = vllm_config.model_config.multimodal_config
        self.config = config
        self.quant_config = quant_config
        self.multimodal_config = multimodal_config
        self.model_dtype = vllm_config.model_config.dtype
        self.vllm_config = vllm_config
        vision_tower_quant = _get_tower_quant_config(vllm_config, enable_ascend=True)
        audio_tower_quant = _get_tower_quant_config(vllm_config)

        with self._mark_tower_model(vllm_config, {"image", "video"}):
            self.vision_tower = AutoModel.from_config(config=config.vision_config)
            self.embed_vision = Gemma4MultimodalEmbedder(
                config.vision_config,
                config.text_config,
                quant_config=vision_tower_quant,
                prefix=maybe_prefix(prefix, "embed_vision"),
            )
            recursive_replace_linear(
                self.vision_tower,
                vision_tower_quant,
                prefix=maybe_prefix(prefix, "vision_tower"),
            )

        if config.audio_config is not None:
            with self._mark_tower_model(vllm_config, "audio"):
                self.audio_tower = AutoModel.from_config(config=config.audio_config)
                self.audio_tower.post_init()
                self.embed_audio = Gemma4MultimodalEmbedder(
                    config.audio_config,
                    config.text_config,
                    quant_config=audio_tower_quant,
                    prefix=maybe_prefix(prefix, "embed_audio"),
                )
                recursive_replace_linear(
                    self.audio_tower,
                    audio_tower_quant,
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
