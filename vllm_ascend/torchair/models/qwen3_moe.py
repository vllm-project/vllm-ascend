# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2024 The Qwen team.
# Copyright 2023 The vLLM team.
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved. Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
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
from typing import Any, List, Optional, Union

import torch
import vllm.envs as envs
from torch import nn
from transformers import PretrainedConfig
from vllm.attention import Attention, AttentionMetadata
from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, CompilationLevel, VllmConfig
from vllm.distributed import get_pp_group, get_tensor_model_parallel_world_size
from vllm.distributed.parallel_state import (get_dp_group, get_ep_group,
                                             get_tp_group)
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.fused_moe.layer import FusedMoE
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (QKVParallelLinear,
                                               ReplicatedLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead, VocabParallelEmbedding)
from vllm.model_executor.models.interfaces import (MixtureOfExperts,
                                                   SupportsLoRA, SupportsPP)
from vllm.model_executor.models.qwen3_moe import (Qwen3MoeAttention,
                                                  Qwen3MoeDecoderLayer,
                                                  Qwen3MoeForCausalLM,
                                                  Qwen3MoeMLP, Qwen3MoeModel,
                                                  Qwen3MoeSparseMoeBlock)
from vllm.model_executor.models.utils import (
    PPMissingLayer, extract_layer_index,
    make_empty_intermediate_tensors_factory, make_layers, maybe_prefix)
from vllm.sequence import IntermediateTensors

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.ops.fused_moe import AscendFusedMoE
from vllm_ascend.ops.sequence_parallel import (MetadataForPadding,
                                               init_metadata_for_sp)


class CustomSparseMoeBlock(Qwen3MoeSparseMoeBlock):

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        nn.Module.__init__(self)
        self.tp_size = get_tensor_model_parallel_world_size()
        if self.tp_size > config.num_experts:
            raise ValueError(
                f"Tensor parallel size {self.tp_size} is greater than "
                f"the number of experts {config.num_experts}.")

        self.gate = ReplicatedLinear(
            config.hidden_size,
            config.num_experts,
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.gate",
        )

        self.experts = AscendFusedMoE(
            num_experts=config.num_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            reduce_results=False,
            renormalize=config.norm_topk_prob,
            quant_config=quant_config,
            prefix=f"{prefix}.experts",
        )

        self.top_k = config.num_experts_per_tok

        self.dp_size = get_dp_group().world_size

        self.tp_group = get_tp_group().device_group
        self.tp_rank = get_tp_group().rank_in_group
        self.ep_group = get_ep_group()

        self.params_dtype = torch.get_default_dtype()

    def forward(
        self,
        hidden_states,
        attn_metadata=None,
        _metadata_for_padding: Optional[MetadataForPadding] = None,
    ):
        if attn_metadata is None:
            attn_metadata = get_forward_context().attn_metadata
        # when profile runs, force experts to load balanced tokens
        # to avoid high memory consumption on a single rank.
        enable_force_load_balance = get_forward_context().in_profile_run
        is_prefill = get_forward_context().with_prefill

        # router_logits: (num_tokens, n_experts)
        router_logits, _ = self.gate(hidden_states)

        hidden_states = self.experts(
            hidden_states=hidden_states,
            router_logits=router_logits,
            is_prefill=is_prefill,
            top_k=self.top_k,
            enable_force_load_balance=enable_force_load_balance,
            shared_experts=None,
            _metadata_for_padding=_metadata_for_padding,
        )

        return hidden_states


class CustomQwen3MoeAttention(Qwen3MoeAttention):

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
        nn.Module.__init__(self)
        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()
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

        self.qkv_proj = QKVParallelLinear(hidden_size,
                                          self.head_dim,
                                          self.total_num_heads,
                                          self.total_num_kv_heads,
                                          bias=qkv_bias,
                                          quant_config=quant_config,
                                          prefix=f"{prefix}.qkv_proj")

        self.o_proj = RowParallelLinear(self.total_num_heads * self.head_dim,
                                        hidden_size,
                                        bias=False,
                                        quant_config=quant_config,
                                        prefix=f"{prefix}.o_proj")

        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position_embeddings,
            base=rope_theta,
            rope_scaling=rope_scaling,
        )
        self.attn = Attention(self.num_heads,
                              self.head_dim,
                              self.scaling,
                              num_kv_heads=self.num_kv_heads,
                              cache_config=cache_config,
                              quant_config=quant_config,
                              prefix=f"{prefix}.attn")

        self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
        ascend_config = get_ascend_config()
        self.torchair_graph_enabled = ascend_config.torchair_graph_config.enabled

    @staticmethod
    def normalize_qkv(qkv: torch.Tensor, q_size: int, kv_size: int,
                      head_dim: int, q_norm, k_norm):
        q, k, v = qkv.split([q_size, kv_size, kv_size], dim=-1)

        q_by_head = q.view(*q.shape[:-1], q.shape[-1] // head_dim, head_dim)
        q_by_head = q_norm(q_by_head)
        q = q_by_head.view(q.shape)

        k_by_head = k.view(*k.shape[:-1], k.shape[-1] // head_dim, head_dim)
        k_by_head = k_norm(k_by_head)
        k = k_by_head.view(k.shape)

        return q, k, v

    def forward(
            self,
            positions: torch.Tensor,
            hidden_states: torch.Tensor,
            kv_cache: Optional[torch.Tensor] = None,
            attn_metadata: Optional[AttentionMetadata] = None) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = self.normalize_qkv(qkv, self.q_size, self.kv_size,
                                     self.head_dim, self.q_norm, self.k_norm)

        if (self.torchair_graph_enabled and attn_metadata is not None and
                attn_metadata.attn_state == AscendAttentionState.DecodeOnly):
            q, k = self.rotary_emb(positions,
                                   q,
                                   k,
                                   is_prefill=False,
                                   is_qwen_torchair=True)
            forward_kwargs = {}
            if envs.VLLM_USE_V1:
                output_shape = q.shape
                output = torch.empty(output_shape,
                                     dtype=q.dtype,
                                     device=q.device)
                forward_kwargs['output'] = output

            attn_output = self.attn.impl.forward(self.attn,
                                                 q,
                                                 k,
                                                 v,
                                                 kv_cache=kv_cache,
                                                 attn_metadata=attn_metadata,
                                                 trace_flag=False,
                                                 **forward_kwargs)
            output, _ = self.o_proj(attn_output)
            return output
        else:
            q, k = self.rotary_emb(positions, q, k, is_qwen_torchair=True)
            attn_output = self.attn(q, k, v)
            output, _ = self.o_proj(attn_output)
            return output


class CustomQwen3MoeDecoderLayer(Qwen3MoeDecoderLayer):

    def __init__(
        self,
        config: PretrainedConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        vllm_config: Optional[VllmConfig] = None,
        prefix: str = "",
    ) -> None:

        nn.Module.__init__(self)
        self.hidden_size = config.hidden_size
        rope_theta = getattr(config, "rope_theta", 10000)
        rope_scaling = getattr(config, "rope_scaling", None)
        max_position_embeddings = getattr(config, "max_position_embeddings",
                                          8192)
        self.self_attn = CustomQwen3MoeAttention(
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

        # `mlp_only_layers` in the config.
        layer_idx = extract_layer_index(prefix)
        mlp_only_layers = ([] if not hasattr(config, "mlp_only_layers") else
                           config.mlp_only_layers)
        self.use_aclgraph = (vllm_config is not None
                             and vllm_config.compilation_config.level
                             == CompilationLevel.PIECEWISE
                             and not vllm_config.model_config.enforce_eager)
        if (layer_idx not in mlp_only_layers) and (
                config.num_experts > 0 and
            (layer_idx + 1) % config.decoder_sparse_step == 0):
            if not self.use_aclgraph:
                # FIXME: custom sparse moe block doesn't work with aclgraph.
                self.mlp = CustomSparseMoeBlock(config=config,
                                                quant_config=quant_config,
                                                prefix=f"{prefix}.mlp")
            else:
                self.mlp = Qwen3MoeSparseMoeBlock(config=config,
                                                  quant_config=quant_config,
                                                  prefix=f"{prefix}.mlp")
        else:
            self.mlp = Qwen3MoeMLP(hidden_size=config.hidden_size,
                                   intermediate_size=config.intermediate_size,
                                   hidden_act=config.hidden_act,
                                   quant_config=quant_config,
                                   prefix=f"{prefix}.mlp")
        self.input_layernorm = RMSNorm(config.hidden_size,
                                       eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size,
                                                eps=config.rms_norm_eps)

        self.enable_sequence_parallelism = (
            vllm_config.compilation_config.pass_config.
            enable_sequence_parallelism if vllm_config is not None else False)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
        kv_cache: Optional[torch.Tensor] = None,
        attn_metadata: Optional[AttentionMetadata] = None,
        _metadata_for_padding: Optional[MetadataForPadding] = None,
    ) -> torch.Tensor:

        # To prevent precision issues during the decoder phase when only prefilling enables SP
        if not self.enable_sequence_parallelism:
            self.self_attn.o_proj.reduce_results = True
        else:
            self.self_attn.o_proj.reduce_results = not _metadata_for_padding.not_dummy_and_is_prefill if _metadata_for_padding is not None else True

        # Self Attention
        if residual is None:
            residual = hidden_states
            if _metadata_for_padding and _metadata_for_padding.not_dummy_and_is_prefill:
                residual = _metadata_for_padding.padding_slice(residual)

            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(
                hidden_states, residual)

            if _metadata_for_padding and _metadata_for_padding.not_dummy_and_is_prefill:
                hidden_states = _metadata_for_padding.allgather_unpadding_aligned(
                    hidden_states)

        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            kv_cache=kv_cache,
            attn_metadata=attn_metadata,
        )

        if _metadata_for_padding and _metadata_for_padding.not_dummy_and_is_prefill:
            hidden_states = _metadata_for_padding.padding_aligned_reduce_scatter(
                hidden_states)

        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(
            hidden_states, residual)

        if not self.use_aclgraph:
            hidden_states = self.mlp(
                hidden_states, _metadata_for_padding=_metadata_for_padding)
        else:
            hidden_states = self.mlp(hidden_states)

        return hidden_states, residual


@support_torch_compile
class CustomQwen3MoeModel(Qwen3MoeModel):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        nn.Module.__init__(self)
        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config

        parallel_config = vllm_config.parallel_config
        eplb_config = parallel_config.eplb_config
        self.num_redundant_experts = eplb_config.num_redundant_experts
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.config = config
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            prefix=f"{prefix}.embed_tokens")
        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: CustomQwen3MoeDecoderLayer(
                config=config,
                cache_config=cache_config,
                quant_config=quant_config,
                vllm_config=vllm_config,
                prefix=prefix),
            prefix=f"{prefix}.layers",
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.make_empty_intermediate_tensors = (
            make_empty_intermediate_tensors_factory(
                ["hidden_states", "residual"], config.hidden_size))

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: Optional[List[torch.Tensor]] = None,
        attn_metadata: Optional[AttentionMetadata] = None,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        _metadata_for_padding: Optional[MetadataForPadding] = None,
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
        for i in range(self.start_layer, self.end_layer):
            layer = self.layers[i]
            hidden_states, residual = layer(
                positions,
                hidden_states,
                residual,
                kv_caches[i -
                          self.start_layer] if kv_caches is not None else None,
                attn_metadata,
                _metadata_for_padding=_metadata_for_padding)
        if not get_pp_group().is_last_rank:
            return IntermediateTensors({
                "hidden_states": hidden_states,
                "residual": residual
            })

        hidden_states, _ = self.norm(hidden_states, residual)

        if _metadata_for_padding and _metadata_for_padding.not_dummy_and_is_prefill:
            hidden_states = _metadata_for_padding.allgather_unpadding_aligned(
                hidden_states)

        return hidden_states


class CustomQwen3MoeForCausalLM(Qwen3MoeForCausalLM):
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
        "experts":
        ["experts.0.gate_proj", "experts.0.up_proj", "experts.0.down_proj"],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        nn.Module.__init__(self)
        SupportsPP.__init__(self)
        SupportsLoRA.__init__(self)
        MixtureOfExperts.__init__(self)
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.quant_config = quant_config
        self.model = CustomQwen3MoeModel(vllm_config=vllm_config,
                                         prefix=maybe_prefix(prefix, "model"))
        self.lm_head = ParallelLMHead(config.vocab_size,
                                      config.hidden_size,
                                      quant_config=quant_config,
                                      prefix=maybe_prefix(prefix, "lm_head"))
        if self.config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors)

        self.enable_sequence_parallelism = vllm_config.compilation_config.pass_config.enable_sequence_parallelism
        # Set MoE hyperparameters
        self.expert_weights: list[torch.Tensor] = []

        self.moe_layers: list[FusedMoE] = []
        example_layer = None
        for layer in self.model.layers:
            if isinstance(layer, PPMissingLayer):
                continue

            assert isinstance(layer, Qwen3MoeDecoderLayer)
            if isinstance(layer.mlp, Qwen3MoeSparseMoeBlock):
                example_layer = layer.mlp
                self.moe_layers.append(layer.mlp.experts)

        if example_layer is None:
            raise RuntimeError("No Qwen3MoE layer found in the model.layers.")

        self.num_moe_layers = len(self.moe_layers)
        self.num_expert_groups = 1
        self.num_shared_experts = 0

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: Optional[List[torch.Tensor]] = None,
        attn_metadata: Optional[AttentionMetadata] = None,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        _metadata_for_padding = init_metadata_for_sp(
            input_ids, self.enable_sequence_parallelism)
        hidden_states = self.model(input_ids, positions, kv_caches,
                                   attn_metadata, intermediate_tensors,
                                   inputs_embeds, _metadata_for_padding)
        return hidden_states
