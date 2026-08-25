# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import copy
from collections.abc import Iterable

import torch
from torch import nn
from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig, get_current_vllm_config
from vllm.distributed import get_pp_group
from vllm.model_executor.layers.fused_moe.utils import is_model_fused_shared_expert_compatible
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import ColumnParallelLinear, ReplicatedLinear
from vllm.model_executor.layers.mla import MLAModules, MultiHeadLatentAttentionWrapper
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
from vllm.model_executor.models.deepseek_v2 import (
    DeepseekV2DecoderLayer,
    DeepseekV2ForCausalLM,
    DeepseekV2MLAAttention,
    DeepseekV2MLP,
    DeepseekV2Model,
    DeepseekV2MoE,
)
from vllm.model_executor.models.utils import (
    PPMissingLayer,
    make_empty_intermediate_tensors_factory,
    make_layers,
    skip_spec_layers,
)
from vllm.platforms import current_platform

from vllm_ascend.ops.mla import AscendMultiHeadLatentAttention


class Dots3NoteAttention(DeepseekV2MLAAttention):
    def __init__(
        self,
        vllm_config: VllmConfig,
        config,
        prefix: str = "",
        topk_indices_buffer: torch.Tensor | None = None,
        reduce_results: bool = True,
        sliding_window: int | None = None,
    ) -> None:
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        local_config = copy.copy(config)
        local_config.rope_parameters = dict(config.rope_parameters)
        super().__init__(
            vllm_config=vllm_config,
            config=local_config,
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            qk_nope_head_dim=config.qk_nope_head_dim,
            qk_rope_head_dim=config.qk_rope_head_dim,
            v_head_dim=config.v_head_dim,
            q_lora_rank=config.q_lora_rank,
            kv_lora_rank=config.kv_lora_rank,
            max_position_embeddings=getattr(config, "max_position_embeddings", 8192),
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=prefix,
            topk_indices_buffer=topk_indices_buffer,
            reduce_results=reduce_results,
        )

        old_mla_attn = self._modules.pop("mla_attn")
        assert isinstance(old_mla_attn, MultiHeadLatentAttentionWrapper)
        skip_topk = old_mla_attn.skip_topk
        self.k_rope_only_layernorm = RMSNorm(self.qk_rope_head_dim, eps=config.rms_norm_eps)
        self.sdpa_gate_type = config.attention_gate_type
        gate_output_size = self.num_heads * self.v_head_dim if self.sdpa_gate_type == "elementwise" else self.num_heads
        gate_cls = ReplicatedLinear if self.sdpa_gate_type == "headwise" else ColumnParallelLinear
        self.g_proj = gate_cls(
            self.hidden_size,
            gate_output_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.g_proj",
        )

        modules = MLAModules(
            kv_a_layernorm=self.kv_a_layernorm,
            kv_b_proj=self.kv_b_proj,
            rotary_emb=self.rotary_emb,
            o_proj=self.o_proj,
            fused_qkv_a_proj=(self.fused_qkv_a_proj if self.q_lora_rank is not None else None),
            kv_a_proj_with_mqa=(self.kv_a_proj_with_mqa if self.q_lora_rank is None else None),
            q_a_layernorm=(self.q_a_layernorm if self.q_lora_rank is not None else None),
            q_b_proj=self.q_b_proj if self.q_lora_rank is not None else None,
            q_proj=self.q_proj if self.q_lora_rank is None else None,
            indexer=self.indexer,
            indexer_rotary_emb=self.indexer_rope_emb,
            is_sparse=self.is_v32,
            topk_indices_buffer=topk_indices_buffer,
            g_proj=self.g_proj,
        )
        q_lora_scale = kv_lora_scale = 1.0
        if getattr(config, "apply_mla_qkv_lora_rescale", False):
            assert self.q_lora_rank is not None
            q_lora_scale = (self.hidden_size / self.q_lora_rank) ** 0.5
            kv_lora_scale = (self.hidden_size / self.kv_lora_rank) ** 0.5
        static_context = get_current_vllm_config().compilation_config.static_forward_context
        static_context.pop(prefix, None)
        static_context.pop(f"{prefix}.attn", None)
        self.mla_attn = AscendMultiHeadLatentAttention(
            self.hidden_size,
            self.num_local_heads,
            self.scaling,
            self.qk_nope_head_dim,
            self.qk_rope_head_dim,
            self.v_head_dim,
            self.q_lora_rank,
            self.kv_lora_rank,
            modules,
            cache_config,
            quant_config,
            prefix,
            skip_topk=skip_topk,
            sliding_window=sliding_window,
            k_rope_only_layernorm=self.k_rope_only_layernorm,
            sdpa_gate_type=self.sdpa_gate_type,
            q_lora_scale=q_lora_scale,
            kv_lora_scale=kv_lora_scale,
        )


def get_dots3_note_layer_config(config, layer_idx: int):
    local_config = copy.copy(config)
    if config.layer_types[layer_idx] == "sliding_attention":
        del local_config.index_topk
        local_config.num_attention_heads = config.swa_num_attention_heads
        local_config.q_lora_rank = config.swa_q_lora_rank
        local_config.kv_lora_rank = config.swa_kv_lora_rank
        local_config.qk_nope_head_dim = config.swa_qk_nope_head_dim
        local_config.qk_rope_head_dim = config.swa_qk_rope_head_dim
        local_config.v_head_dim = config.swa_v_head_dim
        local_config.attention_gate_type = config.swa_attention_gate_type
        local_config.rope_parameters = {
            "rope_type": "default",
            "rope_theta": config.swa_rope_theta,
        }
    return local_config


class Dots3NoteDecoderLayer(DeepseekV2DecoderLayer):
    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str,
        config=None,
        topk_indices_buffer: torch.Tensor | None = None,
    ) -> None:
        nn.Module.__init__(self)
        if config is None:
            config = vllm_config.model_config.hf_config
        layer_idx = int(prefix.rsplit(".", 1)[-1])
        sliding_window = (
            config.sliding_window_size - 1 if config.layer_types[layer_idx] == "sliding_attention" else None
        )
        config = get_dots3_note_layer_config(config, layer_idx)
        quant_config = vllm_config.quant_config
        parallel_config = vllm_config.parallel_config

        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        self.use_mha = False
        moe_layer_freq = config.moe_layer_freq
        is_moe_layer = (
            config.n_routed_experts is not None
            and layer_idx >= config.first_k_dense_replace
            and layer_idx % moe_layer_freq == 0
        )
        self.use_sequence_parallel_moe = (
            parallel_config.use_sequence_parallel_moe and parallel_config.pipeline_parallel_size == 1 and is_moe_layer
        )
        self.self_attn = Dots3NoteAttention(
            vllm_config=vllm_config,
            config=config,
            prefix=f"{prefix}.self_attn",
            topk_indices_buffer=topk_indices_buffer,
            reduce_results=not self.use_sequence_parallel_moe,
            sliding_window=sliding_window,
        )
        if is_moe_layer:
            self.mlp = DeepseekV2MoE(
                config=config,
                parallel_config=parallel_config,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
                apply_routed_scale_to_output=False,
            )
        else:
            self.mlp = DeepseekV2MLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
            )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.routed_scaling_factor = getattr(config, "routed_scaling_factor", 1.0)


@support_torch_compile
class Dots3NoteModel(DeepseekV2Model):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        nn.Module.__init__(self)
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.device = current_platform.device_type
        self.hidden_size = config.hidden_size
        self.vocab_size = config.vocab_size
        self.is_v32 = True
        topk_indices_buffer = torch.empty(
            vllm_config.scheduler_config.max_num_batched_tokens,
            config.index_topk,
            dtype=torch.int32,
            device=self.device,
        )
        if get_pp_group().is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=f"{prefix}.embed_tokens",
            )
        else:
            self.embed_tokens = PPMissingLayer()
        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: Dots3NoteDecoderLayer(
                vllm_config=vllm_config,
                prefix=prefix,
                topk_indices_buffer=topk_indices_buffer,
            ),
            prefix=f"{prefix}.layers",
        )
        self.is_fused_shared_expert_enabled = is_model_fused_shared_expert_compatible(
            self.layers,
            DeepseekV2MoE,
            "mlp",
        )
        if get_pp_group().is_last_rank:
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], config.hidden_size
        )
        self.aux_hidden_state_layers = tuple[int, ...]()
        self.use_mha = False
        self.num_redundant_experts = vllm_config.parallel_config.eplb_config.num_redundant_experts

    def load_weights(
        self,
        weights: Iterable[tuple[str, torch.Tensor]],
    ) -> set[str]:
        def main_model_weights():
            for name, weight in weights:
                if name.startswith(("model.mtp.", "mtp.")):
                    continue
                yield name, weight

        return super().load_weights(skip_spec_layers(main_model_weights(), self.config))


class Dots3NoteLanguageModelForCausalLM(DeepseekV2ForCausalLM):
    model_cls = Dots3NoteModel
