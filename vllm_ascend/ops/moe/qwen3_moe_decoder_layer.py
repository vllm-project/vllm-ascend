#!/user/bin/env python3
# -*- coding: utf-8 -*-
from typing import Optional, Union

import torch
from torch import nn
from transformers import PretrainedConfig
from vllm.config import CacheConfig, CompilationLevel, VllmConfig
from vllm.distributed import get_pp_group, get_tensor_model_parallel_world_size
from vllm.distributed.parallel_state import (get_dp_group, get_ep_group,
                                             get_tp_group)
from vllm.forward_context import get_forward_context

from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import ReplicatedLinear

from vllm.model_executor.layers.quantization import QuantizationConfig

from vllm.model_executor.models.qwen3_moe import (Qwen3MoeAttention,
                                                  Qwen3MoeDecoderLayer,
                                                  Qwen3MoeForCausalLM,
                                                  Qwen3MoeMLP, Qwen3MoeModel,
                                                  Qwen3MoeSparseMoeBlock)
from vllm.model_executor.models.utils import (
    PPMissingLayer, extract_layer_index,
    make_empty_intermediate_tensors_factory, make_layers, maybe_prefix)

from vllm_ascend.ops.comm_utils import get_sp_metadata_context
from vllm_ascend.ops.fused_moe import AscendFusedMoE
from vllm_ascend.ops.layers.qwen3_decoder_layer import AscendQwen3DecoderLayer
from vllm_ascend.ops.sequence_parallel import (MetadataForPadding)


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


class AscendQwen3MoeDecoderLayer(AscendQwen3DecoderLayer, Qwen3MoeDecoderLayer):
    def __init__(
            self,
            config: PretrainedConfig,
            cache_config: Optional[CacheConfig] = None,
            quant_config: Optional[QuantizationConfig] = None,
            vllm_config: Optional[VllmConfig] = None,
            enable_eplb: bool = False,
            prefix: str = "",
    ) -> None:
        nn.Module.__init__(self)
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
            _metadata_for_padding: Optional[MetadataForPadding] = None,
    ) -> torch.Tensor:
        sp_metadata, enable_sp = get_sp_metadata_context()
        if not enable_sp:
            self.self_attn.o_proj.reduce_results = True
        else:
            self.self_attn.o_proj.reduce_results = \
                not sp_metadata.metadata_for_padding.not_dummy_and_is_prefill \
                    if sp_metadata is not None else True

        hidden_states, residual = self._forward(hidden_states, positions, residual, sp_metadata)

        if not self.use_aclgraph:
            hidden_states = self.mlp(
                hidden_states,
                _metadata_for_padding=sp_metadata.metadata_for_padding if sp_metadata is not None else None)
        else:
            hidden_states = self.mlp(hidden_states)

        return hidden_states, residual
