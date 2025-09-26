# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 The vLLM team.
# Copyright 2025 Google Inc. HuggingFace Inc. team. All rights reserved.
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
from collections.abc import Iterable
from typing import Optional, Union, List

import torch
import torch.nn.functional as F
from torch import nn
from transformers import Gemma3TextConfig

from vllm.forward_context import get_forward_context
from vllm.attention import Attention, AttentionMetadata
from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import (
    get_pp_group, 
    get_tensor_model_parallel_world_size, 
    get_tensor_model_parallel_rank)
from vllm.logger import init_logger

from vllm.model_executor.layers.activation import GeluAndMul
from vllm.model_executor.layers.layernorm import GemmaRMSNorm
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader, maybe_remap_kv_scale_name)
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import IntermediateTensors

from vllm.model_executor.models.interfaces import SupportsLoRA, SupportsPP
from vllm.model_executor.models.utils import (
    AutoWeightsLoader, 
    extract_layer_index,
    is_pp_missing_parameter,
    make_empty_intermediate_tensors_factory, 
    make_layers,
    maybe_prefix)

from omni.layers.linear import RowParallelFlashCommLinear, QKVParallelFlashCommLinear, MergedColumnParallelFlashCommLinear
from omni.layers.rotary_embedding import get_rope
from omni.layers.attention.backend.attention import AscendAttentionState

logger = init_logger(__name__)


class Gemma3MLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_activation: str,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        tp_size = get_tensor_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank()
        self.gate_up_proj = MergedColumnParallelFlashCommLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            tp_size=tp_size,
            tp_rank=tp_rank,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelFlashCommLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            tp_size=tp_size,
            tp_rank=tp_rank,
            quant_config=quant_config,
            prefix=f"{prefix}.down_proj",
        )
        if hidden_activation != "gelu_pytorch_tanh":
            raise ValueError(
                "Gemma3 uses `gelu_pytorch_tanh` as the hidden activation "
                "function. Please set `hidden_act` and `hidden_activation` to "
                "`gelu_pytorch_tanh`.")
        self.act_fn = GeluAndMul(approximate="tanh")

    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class Gemma3Attention(nn.Module):

    def __init__(self,
                 config: Gemma3TextConfig,
                 hidden_size: int,
                 num_heads: int,
                 num_kv_heads: int,
                 head_dim: int,
                 max_position_embeddings: int,
                 cache_config: Optional[CacheConfig] = None,
                 quant_config: Optional[QuantizationConfig] = None,
                 attn_logits_soft_cap: Optional[float] = None,
                 prefix: str = "") -> None:
        super().__init__()
        self.config = config
        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            assert self.total_num_kv_heads % tp_size == 0
        else:
            assert tp_size % self.total_num_kv_heads == 0

        self.layer_name = f'{prefix}.attn'
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = head_dim
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = config.query_pre_attn_scalar**-0.5

        self.qkv_proj = QKVParallelFlashCommLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=config.attention_bias,
            tp_size=tp_size,
            tp_rank=tp_rank,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelFlashCommLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=config.attention_bias,
            tp_size=tp_size,
            tp_rank=tp_rank,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        self.q_norm = GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)

        layer_idx = extract_layer_index(prefix)
        self.is_sliding = (getattr(
            config, "interleaved_sliding_window", None) is not None and bool((layer_idx + 1) % config.sliding_window_pattern))
        # Initialize the rotary embedding.
        if self.is_sliding:
            # Local attention. Override the values in config.json.
            self.rope_theta = config.rope_local_base_freq
            self.rope_scaling = {"rope_type": "gemma_default"}
            self.sliding_window = config.interleaved_sliding_window
        else:
            # Global attention. Use the values in config.json.
            self.rope_theta = config.rope_theta
            self.rope_scaling = config.rope_scaling
            self.sliding_window = None

        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position_embeddings,
            base=self.rope_theta,
            is_neox_style=True,
            rope_scaling=self.rope_scaling,
        )

        # Initialize the attention.
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            logits_soft_cap=attn_logits_soft_cap,
            per_layer_sliding_window=self.sliding_window,
            prefix=f"{prefix}.attn"
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        attn_metadata = get_forward_context().attn_metadata
        if attn_metadata is not None and attn_metadata[next(iter(attn_metadata))].attn_state == AscendAttentionState.PrefillNoCache:
            is_prefill = True
        else:
            is_prefill = False
        qkv, _ = self.qkv_proj(hidden_states, is_prefill=is_prefill)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        q = q.unflatten(-1, (self.num_heads, self.head_dim))
        q = self.q_norm(q)
        q = q.flatten(-2, -1)
        k = k.unflatten(-1, (self.num_kv_heads, self.head_dim))
        k = self.k_norm(k)
        k = k.flatten(-2, -1)

        q, k = self.rotary_emb(positions, q, k, layer_name=self.layer_name)
        attn_output = self.attn(q, k, v)

        if not kwargs.get("has_images", False):
            # Fast path for text-only inputs. The performance for the text-only
            # inputs are not affected by the naive attention below.
            output, _ = self.o_proj(attn_output)
            return output
        
        attn_output = self.naive_attn_with_masks(q,
                                                 k,
                                                 v,
                                                 out=attn_output,
                                                 **kwargs)
        output, _ = self.o_proj(attn_output)
        return output

    def naive_attn_with_masks(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        out: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:

        q = q.view(-1, self.num_heads, self.head_dim)
        # Expand the key and value to handle GQA.
        num_queries_per_kv = self.num_heads // self.num_kv_heads
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        k = k.repeat_interleave(num_queries_per_kv, dim=-2)
        v = v.view(-1, self.num_kv_heads, self.head_dim)
        v = v.repeat_interleave(num_queries_per_kv, dim=-2)

        if self.is_sliding:
            attn_masks = kwargs["local_attn_masks"]
        else:
            attn_masks = kwargs["global_attn_masks"]

        seq_lens = kwargs["seq_lens"]
        start_idx = 0
        for seq_len, attn_mask in zip(seq_lens, attn_masks):
            end_idx = start_idx + seq_len
            query = q[start_idx:end_idx].unsqueeze(0)
            key = k[start_idx:end_idx].unsqueeze(0)
            value = v[start_idx:end_idx].unsqueeze(0)

            # Transpose.
            query = query.transpose(1, 2)
            key = key.transpose(1, 2)
            value = value.transpose(1, 2)

            output = F.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask,
                self.scaling,
            )
            output = output.transpose(1, 2).flatten(-2, -1)
            out[start_idx:end_idx] = output
            start_idx = end_idx
        return out


class Gemma3DecoderLayer(nn.Module):

    def __init__(
        self,
        config: Gemma3TextConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Gemma3Attention(
            config=config,
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            cache_config=cache_config,
            quant_config=quant_config,
            attn_logits_soft_cap=None,
            prefix=f"{prefix}.self_attn",
        )
        self.hidden_size = config.hidden_size
        self.mlp = Gemma3MLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_activation=config.hidden_activation,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
        )
        self.input_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.pre_feedforward_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_feedforward_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
        kv_cache: Optional[torch.Tensor],
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            kv_cache=kv_cache,
            **kwargs,
        )

        hidden_states = self.post_attention_layernorm(hidden_states)

        hidden_states, residual = self.pre_feedforward_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        return hidden_states, residual


@support_torch_compile
class Gemma3Model(nn.Module):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.quant_config = quant_config

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            prefix=f"{prefix}.embed_tokens",
        )
        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: Gemma3DecoderLayer(
                                config, 
                                cache_config, 
                                quant_config, 
                                prefix=prefix),
            prefix=f"{prefix}.layers")
        self.norm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        normalizer = self.config.hidden_size**0.5
        self.register_buffer("normalizer", torch.tensor(normalizer))
        self.make_empty_intermediate_tensors = (
            make_empty_intermediate_tensors_factory(
                ["hidden_states", "residual"], config.hidden_size))

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids) * self.normalizer

    def forward(
        self,
        input_ids: Optional[torch.Tensor],
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor] = None,
        attn_metadata: AttentionMetadata = None,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs,
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
        for layer_idx in range(self.start_layer, self.end_layer):
            layer = self.layers[layer_idx]
            hidden_states, residual = layer(
                positions,
                hidden_states,
                residual,
                kv_caches[layer_idx] if kv_caches is not None else None,
                **kwargs,
            )
        if not get_pp_group().is_last_rank:
            return IntermediateTensors({
                "hidden_states": hidden_states,
                "residual": residual
            })
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str,
                                                   torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        for name, loaded_weight in weights:
            if (self.quant_config is not None and
                (scale_name := self.quant_config.get_cache_scale(name))):
                # Loading kv cache scales for compressed-tensors quantization
                param = params_dict[scale_name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                loaded_weight = loaded_weight[0]
                weight_loader(param, loaded_weight)
                loaded_params.add(scale_name)
                continue
            for (param_name, shard_name, shard_id) in stacked_params_mapping:
                if shard_name not in name:
                    continue
                name = name.replace(shard_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if is_pp_missing_parameter(name, self):
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                # Remapping the name of FP8 kv-scale.
                name = maybe_remap_kv_scale_name(name, params_dict)
                if name is None:
                    continue
                if is_pp_missing_parameter(name, self):
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(name)

        return loaded_params


class Gemma3ForCausalLM(nn.Module, SupportsLoRA, SupportsPP):
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

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        lora_config = vllm_config.lora_config
        del lora_config  # Unused.
        super().__init__()
        self.config = config
        # currently all existing Gemma models have `tie_word_embeddings` enabled
        assert config.tie_word_embeddings
        self.quant_config = quant_config
        self.model = Gemma3Model(vllm_config=vllm_config,
                                 prefix=maybe_prefix(prefix, "model"))
        self.logits_processor = LogitsProcessor(
            config.vocab_size, soft_cap=config.final_logit_softcapping)
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
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        hidden_states = self.model(input_ids, 
                                   positions, 
                                   kv_caches=kv_caches,
                                   attn_metadata=attn_metadata,
                                   intermediate_tensors=intermediate_tensors,
                                   inputs_embeds=inputs_embeds, 
                                   **kwargs)
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[torch.Tensor]:
        logits = self.logits_processor(self.model.embed_tokens, hidden_states,
                                       sampling_metadata)
        return logits

    def load_weights(self, weights: Iterable[tuple[str,
                                                   torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=(["lm_head."]
                           if self.config.tie_word_embeddings else None),
        )
        return loader.load_weights(weights)
