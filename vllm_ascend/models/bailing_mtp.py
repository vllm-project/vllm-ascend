#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Adapted from vllm/model_executor/models/deepseek_mtp.py
# Copyright 2023 The vLLM team.
#
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
from typing import Iterable, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig
from vllm.attention.backends.abstract import AttentionMetadata
from vllm.config import CacheConfig, ModelConfig, VllmConfig
from vllm.distributed import (get_pp_group, 
                              get_tensor_model_parallel_world_size, 
                              get_tensor_model_parallel_rank)
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.sampler import get_sampler
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead, VocabParallelEmbedding)
from vllm.model_executor.layers.fused_moe.layer import FusedMoE
from vllm.model_executor.models.utils import (
    maybe_prefix, AutoWeightsLoader,
    is_pp_missing_parameter, PPMissingLayer)
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.sequence import IntermediateTensors
from vllm_ascend.models.bailing_moe_v2 import (BailingMoe, 
                                               BailingAttention, 
                                               BailingMoeV2RMSNorm,
                                               BailingMoeV2RotaryEmbedding,
                                               get_spec_layer_idx_from_weight_name)


class CustomBailingMultiTokenPredictorLayer(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        prefix: str,
        model_config: ModelConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
    ) -> None:
        nn.Module.__init__(self)
        self.input_layernorm = BailingMoeV2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.enorm = BailingMoeV2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hnorm = BailingMoeV2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.eh_proj = nn.Linear(config.hidden_size * 2,
                                 config.hidden_size,
                                 bias=False)
        self.post_attention_layernorm = BailingMoeV2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.attention = BailingAttention(
            config=config,
            cache_config=cache_config,
            quant_config=quant_config,
            reduce_results=True,
            prefix=f"{prefix}.attention",
        )
        self.rotary_emb = BailingMoeV2RotaryEmbedding(config=config)
        self.mlp = BailingMoe(
            intermediate_size=config.hidden_size,
            config=config,
            quant_config=quant_config,
            reduce_results=True,
            prefix=f"{prefix}.mlp",
        )

        self.final_layernorm = BailingMoeV2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
        previous_hidden_states: torch.Tensor,
        inputs_embeds: Optional[torch.Tensor] = None,
        spec_step_index: int = 0,
    ) -> torch.Tensor:
        assert inputs_embeds is not None
        # masking inputs at position 0, as not needed by MTP
        inputs_embeds = self.enorm(inputs_embeds)
        hidden_states = self.hnorm(previous_hidden_states)
        hidden_states = self.eh_proj(torch.cat([inputs_embeds, hidden_states], dim=-1))
        residual = hidden_states
        position_embeddings = self.rotary_emb(hidden_states, positions)
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.attention(
            hidden_states=hidden_states,
            position_ids=positions,
            position_embeddings=position_embeddings,
        )
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states.to(residual.device)
        hidden_states = self.final_layernorm(hidden_states)

        return hidden_states


class CustomBailingMultiTokenPredictor(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        nn.Module.__init__(self)
        config = vllm_config.model_config.hf_config
        self.config = config
        self.mtp_start_layer_idx = config.num_hidden_layers
        self.num_mtp_layers = config.num_mtp_layers
        # to map the exact layer index from weights
        self.layers = torch.nn.ModuleDict({
            str(idx):
            CustomBailingMultiTokenPredictorLayer(
                config,
                f"{prefix}.layers.{idx}",
                model_config=vllm_config.model_config,
                cache_config=vllm_config.cache_config,
                quant_config=vllm_config.quant_config,
            )
            for idx in range(self.mtp_start_layer_idx,
                             self.mtp_start_layer_idx + self.num_mtp_layers)
        })
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
        )

        # Note: torch._dynamo.exc.Unsupported: builtin: str
        self.layers_list = [
            self.layers[str(idx)]
            for idx in range(self.mtp_start_layer_idx,
                             self.mtp_start_layer_idx + self.num_mtp_layers)
        ]
    
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: torch.Tensor,
        attn_metadata: AttentionMetadata,
        previous_hidden_states: torch.Tensor,
        inputs_embeds: Optional[torch.Tensor] = None,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        current_step_idx = (spec_step_idx % self.num_mtp_layers)
        step_kv_cache = kv_caches[
            current_step_idx] if kv_caches is not None else None
        return self.layers_list[current_step_idx](
            input_ids,
            positions,
            step_kv_cache,
            attn_metadata,
            previous_hidden_states,
            inputs_embeds,
            current_step_idx,
        )
    
    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        return FusedMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.num_experts,
        )        

    def load_weights(self, weights: Iterable[tuple[str,
                                                   torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        params_dict = dict(self.named_parameters(remove_duplicate=False))
        loaded_params: set[str] = set()
        expert_params_mapping = self.get_expert_mapping()
        for name, loaded_weight in weights:
            spec_layer = get_spec_layer_idx_from_weight_name(self.config, name)
            if spec_layer is None:
                continue
            if (
                hasattr(self.config, "norm_head")
                and self.config.norm_head
                and "lm_head.weight" in name
            ):
                loaded_weight = F.normalize(loaded_weight,
                                            dim=0,
                                            p=2,
                                            eps=1e-7)

            for (param_name, weight_name, shard_id) in stacked_params_mapping:
                if weight_name not in name:
                    continue
                if "mlp.experts" in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if name not in params_dict:
                    continue

                if is_pp_missing_parameter(name, self):
                    continue

                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                for mapping in expert_params_mapping:
                    param_name, weight_name, expert_id, shard_id = mapping
                    if weight_name not in name:
                        continue
                    name = name.replace(weight_name, param_name)

                    if is_pp_missing_parameter(name, self):
                        continue
                    param = params_dict[name]
                    weight_loader = param.weight_loader
                    weight_loader(param,
                                  loaded_weight,
                                  name,
                                  shard_id=shard_id,
                                  expert_id=expert_id)
                    break
                else:
                    if name.endswith(".bias") and name not in params_dict:
                        continue
                    if name not in params_dict:
                        continue

                    if is_pp_missing_parameter(name, self):
                        continue

                    param = params_dict[name]
                    weight_loader = getattr(param, "weight_loader",
                                            default_weight_loader)
                    weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params


class CustomBailingMTP(nn.Module):
    packed_modules_mapping = {
        "query_key_value": ["query_key_value"],
        "gate_up_proj": [
            "gate_proj",
            "up_proj",
        ],
    }
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        self.config = vllm_config.model_config.hf_config
        self.tie_word_embeddings = getattr(self.config, "tie_word_embeddings", False)
        self.model = CustomBailingMultiTokenPredictor(vllm_config=vllm_config,
                                                       prefix=maybe_prefix(
                                                           prefix, "model"))
        if get_pp_group().is_last_rank:
            if self.tie_word_embeddings:
                self.lm_head = self.model.word_embeddings
            else:
                self.lm_head = ParallelLMHead(
                    self.config.vocab_size,
                    self.config.hidden_size,
                    quant_config=vllm_config.quant_config,
                    prefix=f"{prefix}.lm_head",
                )
            self.logits_processor = LogitsProcessor(self.config.vocab_size)
        else:
            self.lm_head = PPMissingLayer()
        self.sampler = get_sampler()
    
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: Optional[List[torch.Tensor]] = None,
        attn_metadata: Optional[AttentionMetadata] = None,
        previous_hidden_states: Optional[torch.Tensor] = None,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        hidden_states = self.model(input_ids, positions, kv_caches,
                                   attn_metadata, previous_hidden_states,
                                   inputs_embeds, spec_step_idx)
        return hidden_states
    
    def load_weights(self, weights: Iterable[tuple[str,
                                                torch.Tensor]]) -> set[str]:
        skip_prefixes = []
        if self.config.tie_word_embeddings:
            skip_prefixes.append("lm_head.")
        loader = AutoWeightsLoader(self, skip_prefixes=skip_prefixes,)
        return loader.load_weights(weights)
    
    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        return self.model.get_expert_mapping()
    
    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
        spec_step_idx: int = 0,
    ) -> Optional[torch.Tensor]:
        logits = self.logits_processor(self.lm_head,
                                       hidden_states,
                                       sampling_metadata)
        return logits