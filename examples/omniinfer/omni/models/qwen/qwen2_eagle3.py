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

from collections.abc import Iterable
from typing import Any, Optional, Union, List

import torch
from torch import nn
from transformers import Qwen2Config
from transformers import PretrainedConfig

from vllm.attention import Attention, AttentionType, AttentionMetadata
from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import get_tensor_model_parallel_world_size, get_tensor_model_parallel_rank
from vllm.forward_context import get_forward_context, set_forward_context
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader, maybe_remap_kv_scale_name)
from vllm.model_executor.models.utils import (AutoWeightsLoader, PPMissingLayer, is_pp_missing_parameter,
                    make_empty_intermediate_tensors_factory, make_layers,
                    maybe_prefix)
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead

from omni.adaptors.vllm.worker.npu_model_runner import GraphCompileConfiguration
from omni.layers.attention.backend.attention import AscendAttentionState
from omni.layers.layernorm import RMSNormFlashComm
from omni.layers.linear import QKVParallelFlashCommLinear
from omni.models.qwen.qwen2 import Qwen2DecoderLayer, Qwen2Model, Qwen2ForCausalLM

class Eagle3Qwen2DecoderLayer(Qwen2DecoderLayer):
    def __init__(
        self,
        config: Qwen2Config,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__(
            config, cache_config, quant_config, prefix
        )
        tp_size = get_tensor_model_parallel_world_size()
        tp_rank=get_tensor_model_parallel_rank()
        # override qkv
        self.self_attn.qkv_proj = QKVParallelFlashCommLinear(
            2 * self.hidden_size,
            self.self_attn.head_dim,
            self.self_attn.total_num_heads,
            self.self_attn.total_num_kv_heads,
            tp_size,
            tp_rank,
            bias=getattr(config, "qkv_bias", False),
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn.qkv_proj",
        )

        self.hidden_norm = RMSNormFlashComm(config.hidden_size, eps=config.rms_norm_eps)

        if getattr(config, "norm_before_residual", False):
            self._residual_norm = self._norm_before_residual
        else:
            self._residual_norm = self._norm_after_residual

    def _norm_before_residual(
            self,
            hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden_states = self.hidden_norm(hidden_states)
        residual = hidden_states
        return hidden_states, residual

    def _norm_after_residual(
            self,
            hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        residual = hidden_states
        hidden_states = self.hidden_norm(hidden_states)
        return hidden_states, residual

    def forward(
        self,
        positions: torch.Tensor,
        embeds: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
        kv_cache: Optional[torch.Tensor],
        cos: Optional[torch.Tensor],
        sin: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        embeds = self.input_layernorm(embeds)
        hidden_states, residual = self._residual_norm(hidden_states)
        hidden_states = torch.cat([embeds, hidden_states], dim=-1)
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            kv_cache=kv_cache,
            cos=cos,
            sin=sin
        )

        # Fully Connected
        attn_metadata = get_forward_context().attn_metadata
        if attn_metadata is not None and attn_metadata[next(iter(attn_metadata))].attn_state == AscendAttentionState.PrefillNoCache:
            is_prefill = True
        else:
            is_prefill = False
        hidden_states, residual = self.post_attention_layernorm(
            hidden_states, residual)
        hidden_states = self.mlp(hidden_states, x_transform=None, reduce_type="AR", is_prefill=is_prefill)
        return hidden_states, residual

class Eagle3Qwen2Model(Qwen2Model):
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

        # TODO control whether or not to init a new one by config
        self.share_embed = True
        self.embed_tokens = None # get from main model
        self.layers = nn.ModuleList([
            Eagle3Qwen2DecoderLayer(
                config=self.config,
                cache_config=self.cache_config,
                quant_config=self.quant_config,
                prefix=f"{prefix}.layers.{start_layer_id}",
            )
        ])
        
        if hasattr(self.config, "target_hidden_size"):
            self.fc = torch.nn.Linear(self.config.target_hidden_size * 3,
                                      self.config.hidden_size,
                                      bias=False)
        else:
            self.fc = torch.nn.Linear(self.config.hidden_size * 3,
                                      self.config.hidden_size,
                                      bias=False)

        self.norm = RMSNormFlashComm(
            self.config.hidden_size,
            eps=self.config.rms_norm_eps,
        )
        
        self.full_cos = None # get from main model
        self.full_sin = None # get from main model

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: List[torch.Tensor],
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input_embeds = self.embed_tokens(input_ids)
        hidden_states = self.fc(torch.cat(hidden_states, dim=-1))
        assert hidden_states.shape[-1] == input_embeds.shape[-1]
        residual = None

        cos = torch.index_select(self.full_cos, dim=0, index=positions)  # cos.shape [num_tokens, head_size]
        sin = torch.index_select(self.full_sin, dim=0, index=positions)

        hidden_states, residual = self.layers[0](
            positions,
            input_embeds,
            hidden_states,
            residual,
            kv_caches[0] if kv_caches is not None else None,
            cos, sin
        )

        hidden_states = hidden_states + residual
        return None, hidden_states

@support_torch_compile
class Eagle3Qwen2ForCausalLM(Qwen2ForCausalLM, GraphCompileConfiguration):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        nn.Module.__init__(self)
        self.config = vllm_config. \
            speculative_config.draft_model_config.hf_config
        target_layer_num = vllm_config.model_config.get_num_layers(
            vllm_config.parallel_config)
        self.target_layer_num = target_layer_num
        self.model = Eagle3Qwen2Model(
            config=self.config,
            cache_config=vllm_config.cache_config,
            quant_config=vllm_config.quant_config,
                                prefix="model",
                                start_layer_id=target_layer_num)

        logit_scale = getattr(self.config, "logit_scale", 1.0)
        self.logits_processor = LogitsProcessor(self.config.vocab_size,
                                                scale=logit_scale)
        self.lm_head = ParallelLMHead(
            self.config.draft_vocab_size,
            self.config.hidden_size,
            quant_config=vllm_config.quant_config,
            org_num_embeddings=self.config.draft_vocab_size,
            prefix=maybe_prefix(prefix, "lm_head"),
            parallel_lmhead=False,
        )

        self.draft_id_to_target_id = nn.Parameter(
            torch.zeros(self.config.draft_vocab_size, dtype=torch.long),
            requires_grad=False,
        ) if self.config.draft_vocab_size != self.config.vocab_size else None

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
        if not self.draft_id_to_target_id is None:
            base = torch.arange(self.config.draft_vocab_size, device=self.draft_id_to_target_id.device)
            self.draft_id_to_target_id += base

        target_model.set_aux_hidden_state_layers(target_model.get_eagle3_aux_hidden_state_layers())

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[torch.Tensor]:
        logits = self.logits_processor(self.lm_head, hidden_states,
                                       sampling_metadata)
        if self.draft_id_to_target_id is None:
            assert logits.shape[1] == self.config.vocab_size, \
                "Expected logits to have shape " \
                f"(*, {self.config.vocab_size}), but got {logits.shape}"
            return logits

        
        logits_new = logits.new_full((
            logits.shape[0],
            self.config.vocab_size,
        ), float('-inf'))
        logits_new[:, self.draft_id_to_target_id] = logits
        return logits_new

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        model_weights = {}
        includes_draft_id_mapping = False
        includes_embed_tokens = False
        for name, loaded_weight in weights:
            if "t2d" in name:
                continue
            if self.model.share_embed and "embed_tokens" in name:
                continue
            if "d2t" in name:
                if self.draft_id_to_target_id is None:
                    raise ValueError("It is found that draft_vocab_size == vocab_size in config, while get d2t(draft_id_to_target_id) in weight.")
                name = name.replace("d2t", "draft_id_to_target_id")
            elif "lm_head" not in name:
                name = "model." + name
            if 'midlayer.' in name:
                name = name.replace('midlayer.', 'layers.0.')
            model_weights[name] = loaded_weight

        loader = AutoWeightsLoader(
            self,
            skip_prefixes=None,
        )
        return loader.load_weights(model_weights.items())

    def should_use_eager_mode(self, *args, **kwargs):
        attn_metadata = kwargs.get("attn_metadata", None)
        if not attn_metadata:
            return True
        if isinstance(attn_metadata, dict):
            attn_metadata = attn_metadata[self.model.layers[0].layer_name]
        return attn_metadata.attn_state != AscendAttentionState.DecodeOnly

    def mark_static_for_graph(self, *args, **kwargs):
        torch._dynamo.mark_static(self.model.full_cos)
        torch._dynamo.mark_static(self.model.full_sin)
