# SPDX-License-Identifier: Apache-2.0

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
"""Inference-only Qwen3MoE model compatible with HuggingFace weights."""
from collections.abc import Iterable
from typing import Any, Optional, Union, List, Tuple

import torch
from torch import nn
from transformers import PretrainedConfig
from vllm.attention import Attention, AttentionMetadata
from vllm.config import CacheConfig, VllmConfig
from vllm.compilation.decorators import support_torch_compile
from vllm.distributed import get_pp_group, get_tp_group
from vllm.forward_context import ForwardContext, get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.sampler import Sampler, SamplerOutput
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead, VocabParallelEmbedding)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import IntermediateTensors
from vllm.model_executor.models.interfaces import SupportsPP
from vllm.model_executor.models.utils import (AutoWeightsLoader, PPMissingLayer, is_pp_missing_parameter,
                    make_empty_intermediate_tensors_factory, make_layers,
                    maybe_prefix)

from omni.models.qwen.fused_moe import FusedMoE
from omni.adaptors.vllm.worker.npu_model_runner import GraphCompileConfiguration
from omni.layers.layernorm import RMSNormFlashComm, RMSNorm
from omni.layers.linear import (RowParallelFlashCommLinear, 
                                              QKVParallelFlashCommLinear)
from omni.layers.rotary_embedding import get_rope
from omni.layers.attention.backend.attention import AscendAttentionState


logger = init_logger(__name__)
SEQ_SPLIT_LENGTH = 4096

class Qwen3MoeSparseMoeBlock(nn.Module):

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()

        self.experts = FusedMoE(num_experts=config.num_experts,
                                top_k=config.num_experts_per_tok,
                                hidden_size=config.hidden_size,
                                intermediate_size=config.moe_intermediate_size,
                                renormalize=config.norm_topk_prob,
                                quant_config=quant_config,
                                prefix=f"{prefix}.experts")

        self.gate = ReplicatedLinear(config.hidden_size,
                                     config.num_experts,
                                     bias=False,
                                     quant_config=None,
                                     prefix=f"{prefix}.gate")

    def forward(self, hidden_states: torch.Tensor, is_prefill: bool = False) -> torch.Tensor:
        # NOTE: hidden_states can have either 1D or 2D shape.
        orig_shape = hidden_states.shape
        hidden_dim = hidden_states.shape[-1]
        hidden_states = hidden_states.view(-1, hidden_dim)
        # router_logits: (num_tokens, n_experts)
        router_logits, _ = self.gate(hidden_states)
        final_hidden_states = self.experts(hidden_states=hidden_states,
                                           router_logits=router_logits,
                                           is_prefill=is_prefill)
        final_hidden_states = final_hidden_states

        return final_hidden_states.view(orig_shape)


class Qwen3MoeAttention(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        rope_theta: float = 10000,
        rope_scaling: Optional[dict[str, Any]] = None,
        max_position_embeddings: int = 8192,
        head_dim: Optional[int] = None,
        rms_norm_eps: float = 1e-06,
        qkv_bias: bool = False,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        tp_size = get_tp_group().world_size
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = head_dim or (hidden_size // self.total_num_heads)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings
        tp_rank = get_tp_group().rank_in_group

        self.qkv_proj = QKVParallelFlashCommLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            tp_size=tp_size,
            tp_rank=tp_rank,
            bias=qkv_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj"
        )
        self.o_proj = RowParallelFlashCommLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            tp_size=tp_size,
            tp_rank=tp_rank,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj"
        )

        if rope_scaling is None:
            rope_scaling = {'factor': '0'}
        rope_scaling["rope_type"] = 'qwen'
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position_embeddings,
            base=rope_theta,
            rope_scaling=rope_scaling,
        )
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn")
        
        self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: Tuple[torch.Tensor, torch.Tensor],
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states, x_transform='AG')
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        q_by_head = q.view(*q.shape[:-1], q.shape[-1] // self.head_dim,
                           self.head_dim)
        q_by_head = self.q_norm(q_by_head)
        q = q_by_head.view(q.shape)

        k_by_head = k.view(*k.shape[:-1], k.shape[-1] // self.head_dim,
                            self.head_dim)
        k_by_head = self.k_norm(k_by_head)
        k = k_by_head.view(k.shape)

        if attn_metadata is None:
            cos, sin = self.rotary_emb.get_cos_sin(positions)
        else:
            cos = attn_metadata.cos
            sin = attn_metadata.sin

        q, k = self.rotary_emb(positions, q, k, cos, sin)
        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output, reduce_type="RS")
        return output


class Qwen3MoeDecoderLayer(nn.Module):

    def __init__(
        self,
        config: PretrainedConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.layer_name = f"{prefix}.self_attn.attn"
        self.hidden_size = config.hidden_size
        rope_theta = getattr(config, "rope_theta", 10000)
        rope_scaling = getattr(config, "rope_scaling", None)
        max_position_embeddings = getattr(config, "max_position_embeddings",
                                          8192)
        self.self_attn = Qwen3MoeAttention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            max_position_embeddings=max_position_embeddings,
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=getattr(config, 'attention_bias', False),
            head_dim=getattr(config, 'head_dim', None),
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
        )

        # 'mlp_only_layers' in the config.
        layer_idx = int(prefix.split(sep='.')[-1])
        mlp_only_layers = ([] if not hasattr(config, "mlp_only_layers") else
                           config.mlp_only_layers)
        if (layer_idx not in mlp_only_layers) and (
                config.num_experts > 0 and
            (layer_idx + 1) % config.decoder_sparse_step == 0):
            self.mlp = Qwen3MoeSparseMoeBlock(config=config,
                                              quant_config=quant_config,
                                              prefix=f"{prefix}.mlp")
        else:
            raise NotImplementedError("Qwen3MoeMLP not implemented")
        self.input_layernorm = RMSNormFlashComm(config.hidden_size,
                                       eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNormFlashComm(config.hidden_size,
                                                eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
        kv_cache: Optional[torch.Tensor],
        attn_metadata: Optional[AttentionMetadata]
    ) -> torch.Tensor:
        if isinstance(attn_metadata, dict):
            attn_metadata = attn_metadata[self.layer_name]
        # Self Attention
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(
                hidden_states, residual)
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            kv_cache=kv_cache,
            attn_metadata=attn_metadata,
        )

        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(
            hidden_states, residual)
        forward_context: ForwardContext = get_forward_context()
        attn_metadata = forward_context.attn_metadata
        if isinstance(attn_metadata, dict):
            attn_metadata = attn_metadata[next(iter(attn_metadata))]
        is_prefill = attn_metadata is None or not attn_metadata.is_pd_seperate_d
        if is_prefill:
            local_length = hidden_states.shape[0]
            hidden_states_list = hidden_states.split(SEQ_SPLIT_LENGTH)
            hidden_states_out = []
            for i in range(len(hidden_states_list)):
                hidden_states = self.mlp(hidden_states_list[i], is_prefill=is_prefill)
                hidden_states_out.append(hidden_states)
            hidden_states = torch.cat(hidden_states_out)[:local_length]
        else:
            hidden_states = self.mlp(hidden_states, is_prefill=is_prefill)
        return hidden_states, residual


class Qwen3MoeModel(nn.Module):

    def __init__(self,
                 config: PretrainedConfig,
                 cache_config: Optional[CacheConfig] = None,
                 quant_config: Optional[QuantizationConfig] = None,
                 prefix: str = ""):
        super().__init__()

        config = config
        cache_config = cache_config
        quant_config = quant_config

        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.config = config
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            prefix=f"{prefix}.embed_tokens")

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: Qwen3MoeDecoderLayer(config=config,
                                                cache_config=cache_config,
                                                quant_config=quant_config,
                                                prefix=prefix),
            prefix=f"{prefix}.layers",
        )
        self.norm = RMSNormFlashComm(config.hidden_size, eps=config.rms_norm_eps)
        self.make_empty_intermediate_tensors = (
            make_empty_intermediate_tensors_factory(
                ["hidden_states", "residual"], config.hidden_size))

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)
    
    def forward_opening(attn_metadata: AttentionMetadata):
        if attn_metadata.prefill_metadata is not None:
            pass

    def get_tp_slice(self, x: torch.Tensor):
        tp_size = get_tp_group().world_size
        tp_rank = get_tp_group().rank_in_group

        assert x.shape[0] % tp_size == 0, f"x can't be divided along tp_size {tp_size}!"
        slice_size = x.shape[0] // tp_size
        x_slice = x[tp_rank * slice_size: (tp_rank + 1) * slice_size]

        return x_slice

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
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

        # 采用FlashComm1.0, 通过slice使用hidden_states转为DP
        hidden_states = self.get_tp_slice(hidden_states)

        for i in range(self.start_layer, self.end_layer):
            layer = self.layers[i]
            hidden_states, residual = layer(positions,
                                            hidden_states,
                                            residual,
                                            kv_caches[i] if kv_caches is not None else None,
                                            attn_metadata)

        if not get_pp_group().is_last_rank:
            return IntermediateTensors({
                "hidden_states": hidden_states,
                "residual": residual
            })
        hidden_states, _ = self.norm(hidden_states, residual, y_transform='AG')
        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
    
        moe_params_mapping = FusedMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.num_experts)

        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        for name, loaded_weight in weights:
            ### The weight_scale often has shape (n,1)
            if 'weight_scale' in name:
                loaded_weight = loaded_weight.view(-1)
            for (param_name, weight_name, shard_id) in stacked_params_mapping:

                if weight_name not in name:
                    continue
                if "mlp.experts" in name:
                    continue
                name = name.replace(weight_name, param_name)

                if is_pp_missing_parameter(name, self):
                    continue
                if name not in params_dict:
                    continue

                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                for mapping in moe_params_mapping:
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

                    if is_pp_missing_parameter(name, self):
                        continue
                    param = params_dict[name]
                    weight_loader = getattr(param, "weight_loader",
                                            default_weight_loader)
                    weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params

@support_torch_compile
class Qwen3MoeForCausalLM(nn.Module, SupportsPP, GraphCompileConfiguration):
    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        "gate_up_proj": [
            "gate_proj",
            "up_proj",
        ],
    }

    fall_back_to_pt_during_load = False

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.quant_config = quant_config
        self.model = Qwen3MoeModel(config, vllm_config.cache_config, quant_config,
                                   prefix=f"model")
        self.lm_head = ParallelLMHead(config.vocab_size,
                                      config.hidden_size,
                                      quant_config=quant_config,
                                      parallel_lmhead=False)
        
        self.sampler = Sampler()
        if self.config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors)

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.get_input_embeddings(input_ids)

    def forward(
            self,
            input_ids: torch.Tensor,
            positions: torch.Tensor,
            kv_caches: List[torch.Tensor] = None,
            attn_metadata: AttentionMetadata = None,
            selected_indices: Optional[torch.Tensor] = None,
            intermediate_tensors: Optional[IntermediateTensors] = None,
            inputs_embeds = None,
            **kwargs
    ) -> Optional[torch.Tensor]:
        hidden_states = self.model(input_ids, positions, kv_caches, attn_metadata, intermediate_tensors, None)
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[torch.Tensor]:
        logits = self.logits_processor(self.lm_head, hidden_states,
                                       sampling_metadata)
        return logits

    def load_weights(self, weights: Iterable[tuple[str,
                                                   torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights)

    def sample(
            self,
            logits: Optional[torch.Tensor],
            sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        next_tokens = self.sampler(logits, sampling_metadata)
        return next_tokens
    
    def mark_static_for_graph(self, input_ids, positions, attn_metadata, kv_caches):
        # if not self.input_marked:
        torch._dynamo.mark_static(input_ids)
        torch._dynamo.mark_static(positions)
        for i in range(len(kv_caches)):
            if kv_caches[i][0] is not None:
                torch._dynamo.mark_static(kv_caches[i][0])
            if kv_caches[i][1] is not None:
                torch._dynamo.mark_static(kv_caches[i][1])

    def should_use_eager_mode(self, *args, **kwargs):
        attn_metadata = kwargs.get("attn_metadata", None)
        if not attn_metadata:
            return True

        if isinstance(attn_metadata, dict):
            attn_metadata = attn_metadata[self.model.layers[self.model.start_layer].layer_name]

        return attn_metadata.attn_state != AscendAttentionState.DecodeOnly