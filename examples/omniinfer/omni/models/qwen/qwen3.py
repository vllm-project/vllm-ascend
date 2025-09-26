# SPDX-License-Identifier: Apache-2.0

# Adapted from
# https://github.com/huggingface/transformers/blob/v4.28.0/src/transformers/models/qwen3/modeling_qwen3.py
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
"""Inference-only Qwen3 model compatible with HuggingFace weights."""
from collections.abc import Iterable
from typing import Any, Optional, Union, List

import torch
from torch import nn
from transformers import Qwen3Config

from vllm.forward_context import get_forward_context
from vllm.attention import Attention, AttentionType, AttentionMetadata
from vllm.config import CacheConfig, VllmConfig
from vllm.compilation.decorators import support_torch_compile
from vllm.distributed import (
    get_pp_group,
    get_tp_group,
    get_tensor_model_parallel_world_size,
    get_tensor_model_parallel_rank
)
from vllm.model_executor.layers.sampler import Sampler, SamplerOutput
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name
)
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import IntermediateTensors
from vllm.model_executor.models.interfaces import SupportsLoRA, SupportsPP
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    is_pp_missing_parameter,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)
from omni.layers.layernorm import RMSNormFlashComm
from omni.layers.linear import (
    RowParallelFlashCommLinear,
    QKVParallelFlashCommLinear,
    ColumnParallelFlashCommLinear,
)
from omni.layers.rotary_embedding import get_rope, QwenRotaryEmbedding
from omni.layers.fused_mlp import FusedMLP
from omni.layers.attention.backend.attention import AscendAttentionState

# if use weight nz, this config must be True
torch.npu.config.allow_internal_format = True

MICROBATCH_TOKEN_THRESHOLD = 4096
DEFAULT_ROPE_THETA = 1000000


class Qwen3MLP(FusedMLP):
    pass


class Qwen3Attention(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position: int = 4096 * 32,
        head_dim: Optional[int] = None,
        rms_norm_eps: float = 1e-06,
        qkv_bias: bool = False,
        rope_theta: float = 10000,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        rope_scaling: Optional[tuple] = None,
        prefix: str = "",
        attn_type: str = AttentionType.DECODER,
        dual_chunk_attention_config: Optional[dict[str, Any]] = None,
        kv_stream = None,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        tp_rank=get_tensor_model_parallel_rank()
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
        self.head_dim = head_dim or hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.rope_theta = rope_theta
        self.dual_chunk_attention_config = dual_chunk_attention_config

        self.qkv_proj = QKVParallelFlashCommLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            tp_size=tp_size,
            tp_rank=tp_rank,
            bias=qkv_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelFlashCommLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            tp_size=tp_size,
            tp_rank=tp_rank,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )
        self.a2a_o_proj = ColumnParallelFlashCommLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            tp_size=1,
            tp_rank=0,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.a2a_o_proj",
        )
        self.group_idx = 0
        self.is_quant = quant_config is not None

        if rope_scaling is None:
            rope_scaling = {'factor': '0'}
        rope_scaling["rope_type"] = 'qwen'
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position,
            base=self.rope_theta,
            rope_scaling=rope_scaling,
        )
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
            kv_stream=kv_stream)
        self.q_norm = RMSNormFlashComm(self.head_dim, eps=rms_norm_eps)
        self.k_norm = RMSNormFlashComm(self.head_dim, eps=rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor
    ) -> torch.Tensor:
        attn_metadata = get_forward_context().attn_metadata
        if attn_metadata is not None and attn_metadata[next(iter(attn_metadata))].attn_state == AscendAttentionState.PrefillNoCache:
            is_prefill = True
        else:
            is_prefill = False
        qkv, _ = self.qkv_proj(hidden_states, is_prefill=is_prefill)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q_by_head = q.view(*q.shape[:-1], q.shape[-1] // self.head_dim,
                           self.head_dim)
        q_by_head = self.q_norm(q_by_head)
        q = q_by_head.view(q.shape)
        k_by_head = k.view(*k.shape[:-1], k.shape[-1] // self.head_dim,
                            self.head_dim)
        k_by_head = self.k_norm(k_by_head)
        k = k_by_head.view(k.shape)
        q, k = self.rotary_emb(positions, q, k, cos, sin)
        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output, reduce_type="AR")
        return output


class Qwen3DecoderLayer(nn.Module):

    def __init__(
        self,
        config: Qwen3Config,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        kv_stream = None,
        micro_stream = None,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_name = f"{prefix}.self_attn.attn"
        self.layer_idx = int(prefix.split('.')[-1])
        self.micro_stream = micro_stream
        # Requires transformers > 4.32.0
        rope_theta = getattr(config, "rope_theta", DEFAULT_ROPE_THETA)
        rope_scaling = getattr(config, "rope_scaling", None)
        dual_chunk_attention_config = getattr(config,
                                              "dual_chunk_attention_config",
                                              None)

        # By default, Qwen3 uses causal attention as it is a decoder-only model.
        # You can override the HF config with `is_causal=False` to enable
        # bidirectional attention, which is used in some embedding models
        # (e.g. Alibaba-NLP/gte-Qwen3-7B-instruct)
        if getattr(config, "is_causal", True):
            attn_type = AttentionType.DECODER
        else:
            attn_type = AttentionType.ENCODER_ONLY

        self.self_attn = Qwen3Attention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            max_position=config.max_position_embeddings,
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=getattr(config, 'attention_bias', False),
            head_dim=getattr(config, 'head_dim', None),            
            num_kv_heads=config.num_key_value_heads,
            rope_theta=rope_theta,
            cache_config=cache_config,
            quant_config=quant_config,
            rope_scaling=rope_scaling,
            prefix=f"{prefix}.self_attn",
            attn_type=attn_type,
            dual_chunk_attention_config=dual_chunk_attention_config,
            kv_stream=kv_stream,
        )
        self.mlp = Qwen3MLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
        )
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
        cos: Optional[torch.Tensor],
        sin: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
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


class Qwen3Model(nn.Module):

    def __init__(self,
                 config: Qwen3Config,
                 cache_config: Optional[CacheConfig] = None,
                 quant_config: Optional[QuantizationConfig] = None,
                 prefix: str = "",
                 decoder_layer_type: type[nn.Module] = Qwen3DecoderLayer):
        super().__init__()

        self.tp_size = get_tensor_model_parallel_world_size()
        # TODO (@robertgshaw2): see if this can be moved out
        if (cache_config.sliding_window is not None
                and hasattr(config, "max_window_layers")):
            assert config.max_window_layers == config.num_hidden_layers, (
                "Sliding window for some but all layers is not supported. "
                "This model uses sliding window but `max_window_layers` = {} "
                "is less than `num_hidden_layers` = {}. Please open an issue "
                "to discuss this feature.".format(
                    config.max_window_layers,
                    config.num_hidden_layers,
                ))

        self.config = config
        self.quant_config = quant_config
        self.vocab_size = config.vocab_size

        if get_pp_group().is_first_rank or (config.tie_word_embeddings
                                            and get_pp_group().is_last_rank):
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=f"{prefix}.embed_tokens",
            )
        else:
            self.embed_tokens = PPMissingLayer()

        base = getattr(config, "rope_theta", DEFAULT_ROPE_THETA)
        rotary_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        max_len = config.max_position_embeddings
        full_cos, full_sin = QwenRotaryEmbedding.compute_full_cos_sin(base, rotary_dim, max_len)
        self.register_buffer("full_cos", full_cos, persistent=False)
        self.register_buffer("full_sin", full_sin, persistent=False)
        self.kv_stream = torch.npu.Stream()
        self.micro_stream = torch.npu.Stream()

        # Use the provided decoder layer type or default to Qwen3DecoderLayer
        decoder_layer_type = decoder_layer_type or Qwen3DecoderLayer
        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: decoder_layer_type(config,
                                              cache_config,
                                              quant_config,
                                              prefix,
                                              kv_stream=self.kv_stream,
                                              micro_stream=self.micro_stream),
            prefix=f"{prefix}.layers",
        )   

        self.make_empty_intermediate_tensors = (
            make_empty_intermediate_tensors_factory(
                ["hidden_states", "residual"], config.hidden_size))
        if get_pp_group().is_last_rank:
            self.norm = RMSNormFlashComm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

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

        cos = torch.index_select(self.full_cos, dim=0, index=positions)  # cos.shape [num_tokens, head_size]
        sin = torch.index_select(self.full_sin, dim=0, index=positions)
        attn_metadata = get_forward_context().attn_metadata
        if attn_metadata is not None and attn_metadata[next(iter(attn_metadata))].attn_state == AscendAttentionState.PrefillNoCache and self.tp_size > 1:
            n_tokens = hidden_states.shape[0]
            if n_tokens <= MICROBATCH_TOKEN_THRESHOLD:
                hidden_states, residual = self.forward_layers_prefill_microbatch_tp8_all_reduce(
                    positions, hidden_states, residual, kv_caches, cos, sin)
            else:
                hidden_states, residual = self.forward_layers_prefill_microbatch_tp8_all_to_all(
                    positions, hidden_states, residual, kv_caches, cos, sin)
        else:
            hidden_states, residual = self.forward_layers(
                positions, hidden_states, residual, kv_caches, cos, sin)

        if not get_pp_group().is_last_rank:
            return IntermediateTensors({
                "hidden_states": hidden_states,
                "residual": residual
            })
        hidden_states, _ = self.norm(hidden_states, residual, y_transform=None)

        return hidden_states

    def forward_layers(self, positions, hidden_states, residual, kv_caches, cos, sin):
        for layer_idx in range(self.start_layer, self.end_layer):
            layer = self.layers[layer_idx]
            hidden_states, residual = layer(
                positions,
                hidden_states,
                residual,
                kv_caches[layer_idx] if kv_caches is not None else None,
                cos, sin
            )
        return hidden_states, residual

    def forward_layers_prefill_microbatch_tp8_all_reduce(self, positions, hidden_states, residual, kv_caches, cos, sin):
        n_tokens = hidden_states.shape[0]
        if n_tokens % 2 == 0:
            split_sizes = [n_tokens // 2, n_tokens // 2]
        else:
            split_sizes = [n_tokens // 2, n_tokens // 2 + 1]
        hidden_states_mb0, hidden_states_mb1 = torch.split(hidden_states, split_sizes)
        assert residual is None
        residual_mb0, residual_mb1 = None, None
        cos_mb0, cos_mb1 = torch.split(cos, split_sizes)
        sin_mb0, sin_mb1 = torch.split(sin, split_sizes)
        hidden_states_mb0_handle, hidden_states_mb1_handle = None, None

        for layer_idx in range(self.start_layer, self.end_layer):
            layer = self.layers[layer_idx]

            if hidden_states_mb0_handle is not None:
                hidden_states_mb0_handle.wait()
            if isinstance(layer.input_layernorm, nn.Identity):
                hidden_states_mb0 = hidden_states_mb0
                residual_mb0 = hidden_states_mb0
            else:
                hidden_states_mb0, residual_mb0 = layer.input_layernorm.forward_with_residual(hidden_states_mb0, residual_mb0)

            qkv_mb0, _ = layer.self_attn.qkv_proj(hidden_states_mb0, is_prefill=True)
            q_mb0, k_mb0, v_mb0 = qkv_mb0.split([layer.self_attn.q_size, layer.self_attn.kv_size, layer.self_attn.kv_size], dim=-1)
            # Apply Q/K normalization
            q_mb0_by_head = q_mb0.view(*q_mb0.shape[:-1], q_mb0.shape[-1] // layer.self_attn.head_dim, layer.self_attn.head_dim)
            q_mb0_by_head = layer.self_attn.q_norm(q_mb0_by_head)
            q_mb0 = q_mb0_by_head.view(q_mb0.shape)
            k_mb0_by_head = k_mb0.view(*k_mb0.shape[:-1], k_mb0.shape[-1] // layer.self_attn.head_dim, layer.self_attn.head_dim)
            k_mb0_by_head = layer.self_attn.k_norm(k_mb0_by_head)
            k_mb0 = k_mb0_by_head.view(k_mb0.shape)
            q_mb0, k_mb0 = layer.self_attn.rotary_emb.forward(None, q_mb0, k_mb0, cos_mb0, sin_mb0)

            if hidden_states_mb1_handle is not None:
                hidden_states_mb1_handle.wait()
            if isinstance(layer.input_layernorm, nn.Identity):
                hidden_states_mb1 = hidden_states_mb1
                residual_mb1 = hidden_states_mb1
            else:
                hidden_states_mb1, residual_mb1 = layer.input_layernorm.forward_with_residual(hidden_states_mb1, residual_mb1)

            qkv_mb1, _ = layer.self_attn.qkv_proj(hidden_states_mb1, is_prefill=True)
            q_mb1, k_mb1, v_mb1 = qkv_mb1.split([layer.self_attn.q_size, layer.self_attn.kv_size, layer.self_attn.kv_size], dim=-1)
            
            # Apply Q/K normalization
            q_mb1_by_head = q_mb1.view(*q_mb1.shape[:-1], q_mb1.shape[-1] // layer.self_attn.head_dim, layer.self_attn.head_dim)
            q_mb1_by_head = layer.self_attn.q_norm(q_mb1_by_head)
            q_mb1 = q_mb1_by_head.view(q_mb1.shape)
            k_mb1_by_head = k_mb1.view(*k_mb1.shape[:-1], k_mb1.shape[-1] // layer.self_attn.head_dim, layer.self_attn.head_dim)
            k_mb1_by_head = layer.self_attn.k_norm(k_mb1_by_head)
            k_mb1 = k_mb1_by_head.view(k_mb1.shape)
            q_mb1, k_mb1 = layer.self_attn.rotary_emb.forward(None, q_mb1, k_mb1, cos_mb1, sin_mb1)

            q = torch.cat([q_mb0, q_mb1])
            k = torch.cat([k_mb0, k_mb1])
            v = torch.cat([v_mb0, v_mb1])
            attn_output = layer.self_attn.attn(q.contiguous(), k.contiguous(), v.contiguous())

            attn_output_mb0, attn_output_mb1 = torch.split(attn_output, split_sizes)

            output_mb0, _ = layer.self_attn.o_proj(attn_output_mb0, reduce_type=None)
            hidden_states_mb0, hidden_states_mb0_handle = get_tp_group().all_reduce_async(output_mb0)

            output_mb1, _ = layer.self_attn.o_proj(attn_output_mb1, reduce_type=None)
            hidden_states_mb1, hidden_states_mb1_handle = get_tp_group().all_reduce_async(output_mb1)

            hidden_states_mb0_handle.wait()
            hidden_states_mb0, residual_mb0 = layer.post_attention_layernorm(hidden_states_mb0, residual_mb0)
            hidden_states_mb0 = layer.mlp(hidden_states_mb0, x_transform=None, reduce_type=None, is_prefill=True)
            hidden_states_mb0, hidden_states_mb0_handle = get_tp_group().all_reduce_async(hidden_states_mb0)

            hidden_states_mb1_handle.wait()
            hidden_states_mb1, residual_mb1 = layer.post_attention_layernorm(hidden_states_mb1, residual_mb1)
            hidden_states_mb1 = layer.mlp(hidden_states_mb1, x_transform=None, reduce_type=None, is_prefill=True)
            hidden_states_mb1, hidden_states_mb1_handle = get_tp_group().all_reduce_async(hidden_states_mb1)

        hidden_states_mb0_handle.wait()
        hidden_states_mb1_handle.wait()
        hidden_states = torch.cat([hidden_states_mb0, hidden_states_mb1])
        residual = torch.cat([residual_mb0, residual_mb1])
        return hidden_states, residual

    def forward_layers_prefill_microbatch_tp8_all_to_all(self, positions, hidden_states, residual, kv_caches, cos, sin):
        n_tokens =  hidden_states.shape[0]
        hidden_size = hidden_states.shape[1]
        tp_rank = get_tensor_model_parallel_rank()
        tp_world_size = get_tensor_model_parallel_world_size()
        split_sizes_scatter = [n_tokens // tp_world_size // 2, n_tokens // tp_world_size - n_tokens // tp_world_size // 2]
        split_sizes = split_sizes_scatter.copy()
        n_splits = len(split_sizes)
        for i in range(n_splits):
            split_sizes[i] *= tp_world_size
        
        mb0_scatter_bg = tp_rank * split_sizes_scatter[0]
        mb0_scatter_ed = (tp_rank + 1) * split_sizes_scatter[0]
        mb1_scatter_bg = tp_rank * split_sizes_scatter[1]
        mb1_scatter_ed = (tp_rank + 1) * split_sizes_scatter[1]
        hidden_states_mb0, hidden_states_mb1 = torch.split(hidden_states, split_sizes)
        assert residual is None
        residual_mb0, residual_mb1 = None, None
        cos_mb0, cos_mb1 = torch.split(cos, split_sizes)
        sin_mb0, sin_mb1 = torch.split(sin, split_sizes)
        hidden_states_mb0_handle, hidden_states_mb1_handle = None, None
        main_stream = torch.npu.current_stream()

        hidden_states_mb0_buffer = torch.empty_like(hidden_states_mb0)
        hidden_states_mb1_buffer = torch.empty_like(hidden_states_mb1)
        hidden_states_mb0_size = tuple(hidden_states_mb0.size())
        hidden_states_mb1_size = tuple(hidden_states_mb1.size())
        dtype = hidden_states_mb0.dtype
        device = hidden_states_mb0.device
        hidden_states_mb0_scatter_buffer = torch.empty((hidden_states_mb0_size[0] // tp_world_size, ) + hidden_states_mb0_size[1:], dtype=dtype, device=device)
        hidden_states_mb1_scatter_buffer = torch.empty((hidden_states_mb1_size[0] // tp_world_size, ) + hidden_states_mb1_size[1:], dtype=dtype, device=device)
        
        layer = self.layers[self.start_layer]
        intermediate_size = layer.mlp.gate_up_proj.output_size_per_partition
        for layer_idx in range(self.start_layer, self.end_layer):
            assert layer.mlp.intermediate_size == self.layers[layer_idx].mlp.intermediate_size
            assert layer.self_attn.attn.num_heads == self.layers[layer_idx].self_attn.attn.num_heads
            assert layer.self_attn.attn.head_size == self.layers[layer_idx].self_attn.attn.head_size
        intermediate_states_up_mb0_buffer = torch.empty((split_sizes[0], intermediate_size), dtype=dtype, device=device)
        intermediate_states_up_mb1_buffer = torch.empty((split_sizes[1], intermediate_size), dtype=dtype, device=device)
        intermediate_states_down_mb0_buffer = torch.empty((split_sizes[0], hidden_size), dtype=dtype, device=device)
        intermediate_states_down_mb1_buffer = torch.empty((split_sizes[1], hidden_size), dtype=dtype, device=device)
        attn_output_buffer = torch.empty((n_tokens, layer.self_attn.attn.num_heads,
                                          layer.self_attn.attn.head_size),
                                          dtype=dtype,
                                          device=device)
        self_attn = layer.self_attn
        qkv_size = self_attn.qkv_proj.output_size_per_partition
        qkv_mb0_buffer = torch.empty((hidden_states_mb0_size[0], qkv_size), dtype=dtype, device=device)
        qkv_mb1_buffer = torch.empty((hidden_states_mb1_size[0], qkv_size), dtype=dtype, device=device)
        q_buffer = torch.empty((n_tokens, self_attn.q_size), dtype=dtype, device=device)
        k_buffer = torch.empty((n_tokens, self_attn.kv_size), dtype=dtype, device=device)
        v_buffer = torch.empty((n_tokens, self_attn.kv_size), dtype=dtype, device=device)

        hidden_states_mb0 = hidden_states_mb0[mb0_scatter_bg:mb0_scatter_ed]
        hidden_states_mb1 = hidden_states_mb1[mb1_scatter_bg:mb1_scatter_ed]

        for layer_idx in range(self.start_layer, self.end_layer):
            layer = self.layers[layer_idx]
            if layer_idx == self.start_layer:
                if isinstance(layer.input_layernorm, nn.Identity):
                    hidden_states_mb0 = hidden_states_mb0
                    residual_mb0 = hidden_states_mb0
                else:
                    hidden_states_mb0, residual_mb0 = layer.input_layernorm.forward_with_residual(
                        hidden_states_mb0, residual_mb0)
                hidden_states_mb0, hidden_states_mb0_handle = get_tp_group().all_gather_async(
                    hidden_states_mb0, dim=0, output_tensor=hidden_states_mb0_buffer)
            hidden_states_mb0_handle.wait()
            with torch.npu.stream(self.micro_stream):
                torch.npu.current_stream().wait_stream(main_stream)
                if layer.self_attn.qkv_proj.bias is not None:
                    qkv_mb0 = torch.addmm(layer.self_attn.qkv_proj.bias, hidden_states_mb0, layer.self_attn.qkv_proj.weight, out=qkv_mb0_buffer)
                else:
                    qkv_mb0 = torch.matmul(hidden_states_mb0, layer.self_attn.qkv_proj.weight, out=qkv_mb0_buffer)
            if hidden_states_mb1_handle is not None:
                hidden_states_mb1_handle.wait()
            if isinstance(layer.input_layernorm, nn.Identity):
                hidden_states_mb1 = hidden_states_mb1
                residual_mb1 = hidden_states_mb1
            else:
                hidden_states_mb1, residual_mb1 = layer.input_layernorm.forward_with_residual(hidden_states_mb1, residual_mb1)
            hidden_states_mb1, hidden_states_mb1_handle = get_tp_group().all_gather_async(
                hidden_states_mb1, dim=0, output_tensor=hidden_states_mb1_buffer)

            torch.npu.current_stream().wait_stream(self.micro_stream)
            q_mb0, k_mb0, v_mb0 = qkv_mb0.split([layer.self_attn.q_size, layer.self_attn.kv_size, layer.self_attn.kv_size], dim=-1)
            # Apply Q/K normalization
            q_mb0_by_head = q_mb0.view(*q_mb0.shape[:-1], q_mb0.shape[-1] // layer.self_attn.head_dim, layer.self_attn.head_dim)
            q_mb0_by_head = layer.self_attn.q_norm(q_mb0_by_head)
            q_mb0 = q_mb0_by_head.view(q_mb0.shape)
            k_mb0_by_head = k_mb0.view(*k_mb0.shape[:-1], k_mb0.shape[-1] // layer.self_attn.head_dim, layer.self_attn.head_dim)
            k_mb0_by_head = layer.self_attn.k_norm(k_mb0_by_head)
            k_mb0 = k_mb0_by_head.view(k_mb0.shape)
            q_mb0, k_mb0 = layer.self_attn.rotary_emb.forward(None, q_mb0, k_mb0, cos_mb0, sin_mb0)
            hidden_states_mb1_handle.wait()
            if layer.self_attn.qkv_proj.bias is not None:
                qkv_mb1 = torch.addmm(layer.self_attn.qkv_proj.bias, hidden_states_mb1, layer.self_attn.qkv_proj.weight, out=qkv_mb1_buffer)
            else:
                qkv_mb1 = torch.matmul(hidden_states_mb1, layer.self_attn.qkv_proj.weight, out=qkv_mb1_buffer)
            q_mb1, k_mb1, v_mb1 = qkv_mb1.split([layer.self_attn.q_size, layer.self_attn.kv_size, layer.self_attn.kv_size], dim=-1)
            # Apply Q/K normalization
            q_mb1_by_head = q_mb1.view(*q_mb1.shape[:-1], q_mb1.shape[-1] // layer.self_attn.head_dim, layer.self_attn.head_dim)
            q_mb1_by_head = layer.self_attn.q_norm(q_mb1_by_head)
            q_mb1 = q_mb1_by_head.view(q_mb1.shape)
            k_mb1_by_head = k_mb1.view(*k_mb1.shape[:-1], k_mb1.shape[-1] // layer.self_attn.head_dim, layer.self_attn.head_dim)
            k_mb1_by_head = layer.self_attn.k_norm(k_mb1_by_head)
            k_mb1 = k_mb1_by_head.view(k_mb1.shape)
            q_mb1, k_mb1 = layer.self_attn.rotary_emb.forward(None, q_mb1, k_mb1, cos_mb1, sin_mb1)

            q = torch.cat([q_mb0, q_mb1], out=q_buffer)
            k = torch.cat([k_mb0, k_mb1], out=k_buffer)
            v = torch.cat([v_mb0, v_mb1], out=v_buffer)

            attn_output = layer.self_attn.attn.impl.forward(layer.self_attn.attn, q.contiguous(), k.contiguous(), v.contiguous(),
                                                       kv_cache=kv_caches[layer_idx],
                                                       attn_metadata=get_forward_context().attn_metadata[layer.self_attn.attn.layer_name],
                                                       output=attn_output_buffer)
            attn_output_mb0, attn_output_mb1 = torch.split(attn_output, split_sizes)

            attn_output_mb0 = get_tp_group().all_to_all(attn_output[:split_sizes[0]])
            with torch.npu.stream(self.micro_stream):
                torch.npu.current_stream().wait_stream(main_stream)
                attn_output_mb1 = get_tp_group().all_to_all(attn_output[split_sizes[0]:])
            hidden_states_mb0, _ = layer.self_attn.a2a_o_proj(attn_output_mb0)
            hidden_states_mb0, residual_mb0 = layer.post_attention_layernorm(hidden_states_mb0, residual_mb0)
            hidden_states_mb0, hidden_states_mb0_handle = get_tp_group().all_gather_async(
                hidden_states_mb0, dim=0, output_tensor=hidden_states_mb0_buffer)
            torch.npu.current_stream().wait_stream(self.micro_stream)
            hidden_states_mb1, _ = layer.self_attn.a2a_o_proj(attn_output_mb1)
            hidden_states_mb1, residual_mb1 = layer.post_attention_layernorm(hidden_states_mb1, residual_mb1)
            hidden_states_mb1, hidden_states_mb1_handle = get_tp_group().all_gather_async(
                hidden_states_mb1, dim=0, output_tensor=hidden_states_mb1_buffer)

            hidden_states_mb0_handle.wait()
            hidden_states_mb0 = torch.matmul(hidden_states_mb0, layer.mlp.gate_up_proj.weight, out=intermediate_states_up_mb0_buffer)
            hidden_states_mb0 = layer.mlp.act_fn(hidden_states_mb0)
            hidden_states_mb0 = torch.matmul(hidden_states_mb0, layer.mlp.down_proj.weight, out=intermediate_states_down_mb0_buffer)
            hidden_states_mb0, hidden_states_mb0_handle = get_tp_group().reduce_scatter_async(
                hidden_states_mb0, output_tensor=hidden_states_mb0_scatter_buffer)
            hidden_states_mb1_handle.wait()
            hidden_states_mb1 = torch.matmul(hidden_states_mb1, layer.mlp.gate_up_proj.weight, out=intermediate_states_up_mb1_buffer)
            hidden_states_mb1 = layer.mlp.act_fn(hidden_states_mb1)
            if layer_idx != self.end_layer - 1:
                with torch.npu.stream(self.micro_stream):
                    torch.npu.current_stream().wait_stream(main_stream)
                    next_layer = self.layers[layer_idx + 1]
                    hidden_states_mb0_handle.wait()
                    if isinstance(next_layer.input_layernorm, nn.Identity):
                        hidden_states_mb0 = hidden_states_mb0
                        residual_mb0 = hidden_states_mb0
                    else:
                        hidden_states_mb0, residual_mb0 = next_layer.input_layernorm.forward_with_residual(hidden_states_mb0, residual_mb0)
                    hidden_states_mb0, hidden_states_mb0_handle = get_tp_group().all_gather_async(
                        hidden_states_mb0, dim=0, output_tensor=hidden_states_mb0_buffer)
                hidden_states_mb1 = torch.matmul(hidden_states_mb1, layer.mlp.down_proj.weight, out=intermediate_states_down_mb1_buffer)
                hidden_states_mb1, hidden_states_mb1_handle = get_tp_group().reduce_scatter_async(
                    hidden_states_mb1, output_tensor=hidden_states_mb1_scatter_buffer)
                torch.npu.current_stream().wait_stream(self.micro_stream)
            else:
                hidden_states_mb1 = torch.matmul(hidden_states_mb1, layer.mlp.down_proj.weight, out=intermediate_states_down_mb1_buffer)
                hidden_states_mb1, hidden_states_mb1_handle = get_tp_group().reduce_scatter_async(hidden_states_mb1, output_tensor=hidden_states_mb1_scatter_buffer)
        hidden_states_mb0, hidden_states_mb0_handle = get_tp_group().all_gather_async(hidden_states_mb0, dim=0, output_tensor=hidden_states_mb0_buffer)
        hidden_states_mb1, hidden_states_mb1_handle = get_tp_group().all_gather_async(hidden_states_mb1, dim=0, output_tensor=hidden_states_mb1_buffer)
        residual_mb0, residual_mb0_handle = get_tp_group().all_gather_async(residual_mb0, dim=0)
        residual_mb1, residual_mb1_handle = get_tp_group().all_gather_async(residual_mb1, dim=0)
        hidden_states_mb0_handle.wait()
        hidden_states_mb1_handle.wait()
        residual_mb0_handle.wait()
        residual_mb1_handle.wait()
        hidden_states = torch.cat([hidden_states_mb0, hidden_states_mb1])
        residual = torch.cat([residual_mb0, residual_mb1])
        return hidden_states, residual
        

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
        duplicate_params_mapping = [
            ("a2a_o_proj", "o_proj"),
        ]
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        loaded_params: set[str] = set()
        for name, loaded_weight in weights:
            if name.endswith(".dequant_scale") and name not in params_dict:
                name = name.replace("dequant_scale", "weight_scale")
            if "rotary_emb.inv_freq" in name:
                continue
            if (self.quant_config is not None and
                (scale_name := self.quant_config.get_cache_scale(name))):
                # Loading kv cache quantization scales
                param = params_dict[scale_name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                loaded_weight = (loaded_weight if loaded_weight.dim() == 0 else
                                 loaded_weight[0])
                weight_loader(param, loaded_weight)
                loaded_params.add(scale_name)
                continue
            for (param_name, weight_name, shard_id) in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
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
                for param_name, weight_name in duplicate_params_mapping:
                    if weight_name not in name:
                        continue
                    duplicate_name = name.replace(weight_name, param_name)
                    if is_pp_missing_parameter(duplicate_name, self):
                        continue
                    if duplicate_name not in params_dict:
                        continue
                    param = params_dict[duplicate_name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)
                    loaded_params.add(duplicate_name)
                    break
            loaded_params.add(name)
        return loaded_params

@support_torch_compile
class Qwen3ForCausalLM(nn.Module, SupportsLoRA, SupportsPP):
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
        super().__init__()
        lora_config = None

        self.config = vllm_config.model_config.hf_config
        self.lora_config = lora_config

        self.quant_config = vllm_config.quant_config
        self.model = Qwen3Model(self.config, vllm_config.cache_config, vllm_config.quant_config,
                                prefix=maybe_prefix(prefix, "model"))
        self.sampler = Sampler()

        if get_pp_group().is_last_rank:
            if self.config.tie_word_embeddings:
                self.lm_head = self.model.embed_tokens
            else:
                self.lm_head = ParallelLMHead(self.config.vocab_size,
                                              self.config.hidden_size,
                                              quant_config=vllm_config.quant_config,
                                              prefix=maybe_prefix(
                                                  prefix, "lm_head"),
                                              parallel_lmhead=False)
        else:
            self.lm_head = PPMissingLayer()

        self.logits_processor = LogitsProcessor(self.config.vocab_size)

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
    ) -> Union[torch.Tensor, IntermediateTensors]:
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
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=(["lm_head."]
                           if self.config.tie_word_embeddings else None),
        )
        return loader.load_weights(weights)

    def sample(
            self,
            logits: Optional[torch.Tensor],
            sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        next_tokens = self.sampler(logits, sampling_metadata)
        return next_tokens

    def should_use_eager_mode(self, *args, **kwargs):
        attn_metadata = kwargs.get("attn_metadata", None)
        if not attn_metadata:
            return True
        if isinstance(attn_metadata, dict):
            attn_metadata = attn_metadata[self.model.layers[self.model.start_layer].layer_name]
        return attn_metadata.attn_state != AscendAttentionState.DecodeOnly
