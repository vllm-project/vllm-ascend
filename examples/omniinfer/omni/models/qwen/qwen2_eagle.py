# SPDX-License-Identifier: Apache-2.0

# Adapted from
# https://github.com/huggingface/transformers/blob/v4.28.0/src/transformers/models/qwen2/modeling_qwen2.py
# Copyright 2024 The Qwen team.
# Copyright 2023 The vLLM team.
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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
"""Inference-only Qwen2 model compatible with HuggingFace weights."""
from collections.abc import Iterable
from typing import Any, Optional, Union, List

import torch
from torch import nn
from transformers import Qwen2Config
from transformers import PretrainedConfig

from vllm.attention import Attention, AttentionType, AttentionMetadata
from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import get_pp_group, get_tensor_model_parallel_world_size
from vllm.forward_context import get_forward_context, set_forward_context
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader, maybe_remap_kv_scale_name)
from vllm.model_executor.models.utils import (AutoWeightsLoader, PPMissingLayer, is_pp_missing_parameter,
                    make_empty_intermediate_tensors_factory, make_layers,
                    maybe_prefix)
from vllm.model_executor.sampling_metadata import SamplingMetadata

from omni.adaptors.vllm.worker.npu_model_runner import GraphCompileConfiguration
from omni.layers.attention.backend.attention import AscendAttentionState
from omni.models.qwen.qwen2 import Qwen2DecoderLayer, Qwen2Model, Qwen2ForCausalLM

class EagleQwen2DecoderLayer(Qwen2DecoderLayer):
    def __init__(
        self,
        config: Qwen2Config,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        use_input_layernorm: bool = False,
    ) -> None:
        super().__init__(
            config, cache_config, quant_config, prefix
        )

        if not use_input_layernorm:
            del self.input_layernorm
            self.input_layernorm = nn.Identity()

class EagleQwen2Model(Qwen2Model):
    def __init__(self,
                 config: Qwen2Config,
                 cache_config: Optional[CacheConfig] = None,
                 quant_config: Optional[QuantizationConfig] = None,
                 prefix: str = "",
                 start_layer_id: int = 0):
        super(Qwen2Model, self).__init__()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.config = config
        self.cache_config = cache_config
        self.quant_config = quant_config

        self.vocab_size = self.config.vocab_size

        self.share_embed = True
        self.embed_tokens = None # get from main model
        self.layers = nn.ModuleList([
            EagleQwen2DecoderLayer(
                config=self.config,
                cache_config=self.cache_config,
                quant_config=self.quant_config,
                prefix=f"{prefix}.layers.{start_layer_id + i}",
            )
            for i in range(self.config.num_hidden_layers)
        ])
        self.start_layer = 0
        self.end_layer = self.config.num_hidden_layers

        self.fc = torch.nn.Linear(self.config.hidden_size * 2,
                                  self.config.hidden_size,
                                  bias=False)

        self.full_cos = None # get from main model
        self.full_sin = None # get from main model

        self.aux_hidden_state_layers = tuple()

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input_embeds = self.embed_tokens(input_ids)
        hidden_states = self.fc(
            torch.cat((input_embeds, hidden_states), dim=-1))
        residual = None

        cos = torch.index_select(self.full_cos, dim=0, index=positions)  # cos.shape [num_tokens, head_size]
        sin = torch.index_select(self.full_sin, dim=0, index=positions)

        attn_metadata = get_forward_context().attn_metadata
        if attn_metadata is not None and attn_metadata[next(iter(attn_metadata))].attn_state == AscendAttentionState.PrefillNoCache and self.tp_size > 1:
            hidden_states, residual, _ = self.forward_layers_prefill_microbatch_tp8_allreduce(positions, hidden_states, residual, kv_caches, cos, sin)
        else:
            hidden_states, residual, _ = self.forward_layers(positions, hidden_states, residual, kv_caches, cos, sin)

        hidden_states = hidden_states + residual
        return None, hidden_states

@support_torch_compile
class EagleQwen2ForCausalLM(Qwen2ForCausalLM, GraphCompileConfiguration):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        nn.Module.__init__(self)
        self.config = vllm_config. \
            speculative_config.draft_model_config.hf_config
        target_layer_num = vllm_config.model_config.get_num_layers(
            vllm_config.parallel_config)
        self.target_layer_num = target_layer_num
        self.model = EagleQwen2Model(
            config=self.config,
            cache_config=vllm_config.cache_config,
            quant_config=vllm_config.quant_config,
                                prefix="model",
                                start_layer_id=target_layer_num)

        logit_scale = getattr(self.config, "logit_scale", 1.0)
        self.logits_processor = LogitsProcessor(self.config.vocab_size,
                                                scale=logit_scale)

        self.share_lm_head = True
        self.lm_head = None # get from main model

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata,
        previous_hidden_states,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        kv_caches = None if kv_caches is None else kv_caches[self.target_layer_num:]
        return self.model(input_ids, positions, previous_hidden_states, kv_caches, attn_metadata)

    def set_share_weight(self, target_model):
        self.model.full_cos = target_model.model.full_cos
        self.model.full_sin = target_model.model.full_sin
        self.model.embed_tokens = target_model.model.embed_tokens
        self.lm_head = target_model.lm_head

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[torch.Tensor]:
        logits = self.logits_processor(self.lm_head, hidden_states,
                                       sampling_metadata)
        return logits

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=None,
        )
        model_weights = {}
        for name, loaded_weight in weights:
            if self.model.share_embed and "embed_tokens" in name:
                continue
            if self.share_lm_head and "lm_head" in name:
                continue
            if "lm_head" not in name:
                name = "model." + name
            model_weights[name] = loaded_weight
        return loader.load_weights(model_weights.items())

    def should_use_eager_mode(self, *args, **kwargs):
        attn_metadata = kwargs.get("attn_metadata", None)
        if not attn_metadata:
            return True
        if isinstance(attn_metadata, dict):
            attn_metadata = attn_metadata[self.model.layers[self.model.start_layer].layer_name]
        return attn_metadata.attn_state != AscendAttentionState.DecodeOnly

    def mark_static_for_graph(self, *args, **kwargs):
        torch._dynamo.mark_static(self.model.full_cos)
        torch._dynamo.mark_static(self.model.full_sin)
