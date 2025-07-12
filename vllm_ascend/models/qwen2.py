# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
#
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
# Adapted from vllm/model_executor/models/qwen3_moe.py
# This file is a part of the vllm-ascend project.

from typing import Optional, Union

import torch
from torch import nn
from transformers import Qwen2Config

from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import get_pp_group
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.sequence import IntermediateTensors
from vllm.model_executor.models.utils import PPMissingLayer, maybe_prefix
from vllm.model_executor.models.qwen2 import (Qwen2ForCausalLM,
                                              Qwen2Model,
                                              Qwen2DecoderLayer)

from vllm_ascend.ops.sequence_parallel import init_metadata_for_sp, MetadataForPadding
import vllm_ascend.envs as envs_ascend


class AscendQwen2DecoderLayer(Qwen2DecoderLayer):

    def __init__(
        self,
        config: Qwen2Config,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__(config, cache_config, quant_config, prefix)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
        _metadata_for_padding:  Optional[MetadataForPadding] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        # To prevent precision issues during the decoder phase when only prefilling enables SP
        if not envs_ascend.VLLM_ENABLE_SP:
            self.self_attn.o_proj.reduce_results = True
            self.mlp.down_proj.reduce_results = True
        else:
            self.self_attn.o_proj.reduce_results =  not _metadata_for_padding.not_dummy_and_is_prefill
            self.mlp.down_proj.reduce_results =  not _metadata_for_padding.not_dummy_and_is_prefill

        if residual is None:
            residual = hidden_states
            if _metadata_for_padding and _metadata_for_padding.not_dummy_and_is_prefill:
                residual = _metadata_for_padding.padding_slice(residual)

            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(
                hidden_states, residual)                
        
            if _metadata_for_padding and _metadata_for_padding.not_dummy_and_is_prefill:
                hidden_states = _metadata_for_padding.allgather_unpadding_aligned(hidden_states)
    
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
        )
        if _metadata_for_padding and _metadata_for_padding.not_dummy_and_is_prefill:
            hidden_states = _metadata_for_padding.padding_aligned_reduce_scatter(hidden_states)

        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(
            hidden_states, residual)
            
        if _metadata_for_padding and _metadata_for_padding.not_dummy_and_is_prefill:
            hidden_states = _metadata_for_padding.allgather_unpadding_aligned(hidden_states)
    
        hidden_states = self.mlp(hidden_states)
    
        if _metadata_for_padding and _metadata_for_padding.not_dummy_and_is_prefill:
            hidden_states = _metadata_for_padding.padding_aligned_reduce_scatter(hidden_states)
                
        return hidden_states, residual


@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        # positions is of shape (3, seq_len) if mrope is enabled for qwen2-vl,
        # otherwise (seq_len, ).
        "positions": -1,
        "intermediate_tensors": 0,
        "inputs_embeds": 0,
    })
class AscendQwen2Model(Qwen2Model):

    def __init__(self,
                 *,
                 vllm_config: VllmConfig,
                 prefix: str = "",
                 decoder_layer_type: type[nn.Module] = AscendQwen2DecoderLayer):
        super().__init__(vllm_config=vllm_config,
                         prefix=prefix,
                         decoder_layer_type=decoder_layer_type)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        _metadata_for_padding:  Optional[MetadataForPadding] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.get_input_embeddings(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]
        for layer in self.layers[self.start_layer:self.end_layer]:
            hidden_states, residual = layer(
                positions,
                hidden_states,
                residual,
                _metadata_for_padding
            )
        if not get_pp_group().is_last_rank:
            return IntermediateTensors({
                "hidden_states": hidden_states,
                "residual": residual
            })
        hidden_states, _ = self.norm(hidden_states, residual)
        
        if _metadata_for_padding and _metadata_for_padding.not_dummy_and_is_prefill:
            hidden_states = _metadata_for_padding.allgather_unpadding_aligned(hidden_states)
        
        return hidden_states


class AscendQwen2ForCausalLM(Qwen2ForCausalLM):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        nn.Module.__init__(self)
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        lora_config = vllm_config.lora_config

        self.config = config
        self.lora_config = lora_config

        self.quant_config = quant_config
        self.model = AscendQwen2Model(vllm_config=vllm_config,
                                      prefix=maybe_prefix(prefix, "model"))

        if get_pp_group().is_last_rank:
            if config.tie_word_embeddings:
                self.lm_head = self.model.embed_tokens
            else:
                self.lm_head = ParallelLMHead(config.vocab_size,
                                              config.hidden_size,
                                              quant_config=quant_config,
                                              prefix=maybe_prefix(
                                                  prefix, "lm_head"))
        else:
            self.lm_head = PPMissingLayer()

        self.logits_processor = LogitsProcessor(config.vocab_size)

        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        _metadata_for_padding = init_metadata_for_sp(input_ids)
        
        hidden_states = self.model(input_ids, positions, intermediate_tensors,
                                   inputs_embeds, _metadata_for_padding)
        return hidden_states