# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
# Copyright 2023 DeepSeek-AI and the HuggingFace Inc. team. All rights reserved.
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
# # Adapted from
# # vllm-project/vllm/blob/main/vllm/model_executor/models/deepseek_v2.py
# # https://github.com/huggingface/transformers/blob/v4.28.0/src/transformers/models/llama/modeling_llama.py
# # vllm-project/vllm/vllm/model_executor/models/deepseek_v2.py
# """Inference-only DeepseekV2/DeepseekV3 model."""

from typing import Any, Dict, Iterable, List, Optional, Union

import torch
import torch.distributed as dist
import torch_npu  # noqa: F401
from torch import nn
from transformers import PretrainedConfig
from vllm.attention import Attention, AttentionMetadata
from vllm.config import CacheConfig, ModelConfig, VllmConfig
from vllm.distributed import (get_pp_group,
                              get_tensor_model_parallel_world_size,
                              get_tp_group, tensor_model_parallel_all_reduce)
from vllm.distributed.parallel_state import get_dp_group
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                                               ReplicatedLinear)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.sampler import get_sampler
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead, VocabParallelEmbedding)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader, maybe_remap_kv_scale_name)
from vllm.model_executor.models.deepseek_v2 import \
    DeepseekV2ForCausalLM  # noqa: E501
from vllm.model_executor.models.deepseek_v2 import \
    yarn_get_mscale  # noqa: E501
from vllm.model_executor.models.deepseek_v2 import (
    DeepseekV2Attention, DeepseekV2DecoderLayer, DeepseekV2MLAAttention,
    get_spec_layer_idx_from_weight_name)
from vllm.model_executor.models.utils import (
    PPMissingLayer, is_pp_missing_parameter,
    make_empty_intermediate_tensors_factory, make_layers, maybe_prefix)
from vllm.sequence import IntermediateTensors

import vllm_ascend.envs as envs_ascend
from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.models.deepseek_v2 import (CustomDeepseekV2MLP,
                                            CustomDeepseekV2RowParallelLinear)
from vllm_ascend.multistream.base import MSEventKey
from vllm_ascend.multistream.context import (
    advance_step_multistream_layer_context, get_multistream_comm_context,
    get_multistream_layer_context, set_multistream_context)
from vllm_ascend.multistream.layers import (MultiStreamPostTransformerLayer,
                                            MultiStreamPreTransformerLayer)
from vllm_ascend.multistream.metadata import (MultiStreamConfig,
                                              MultiStreamStepMetadata,
                                              make_multistream_metadata_ds)
from vllm_ascend.ops.fused_moe import AscendFusedMoE
from vllm_ascend.utils import dispose_tensor

VLLM_ASCEND_ENABLE_DBO: bool = envs_ascend.VLLM_ASCEND_ENABLE_DBO


class CustomDeepseekDBOMLP(CustomDeepseekV2MLP):

    def _forward_ms_mlp(self, x):
        current_ms_metadata = get_multistream_comm_context()
        assert current_ms_metadata is not None
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        current_ms_metadata.before_comm_event.record()
        with torch.npu.stream(current_ms_metadata.comm_stream):
            current_ms_metadata.before_comm_event.wait()
            x, _ = self.down_proj(x)
            current_ms_metadata.after_comm_event.record()
        return x


class CustomDeepseekDBOMoE(nn.Module):

    top_k: int

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.routed_scaling_factor = config.routed_scaling_factor
        self.n_shared_experts = config.n_shared_experts
        self.routed_scaling_factor = config.routed_scaling_factor
        if self.tp_size > config.n_routed_experts:
            raise ValueError(
                f"Tensor parallel size {self.tp_size} is greater than "
                f"the number of experts {config.n_routed_experts}.")

        if config.hidden_act != "silu":
            raise ValueError(f"Unsupported activation: {config.hidden_act}. "
                             "Only silu is supported for now.")

        self.gate = ReplicatedLinear(config.hidden_size,
                                     config.n_routed_experts,
                                     bias=False,
                                     quant_config=None,
                                     prefix=f"{prefix}.gate")
        if config.topk_method == "noaux_tc":
            self.gate.e_score_correction_bias = nn.Parameter(
                torch.empty(config.n_routed_experts))
        else:
            self.gate.e_score_correction_bias = None

        self.experts = AscendFusedMoE(
            num_experts=config.n_routed_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            reduce_results=False,
            renormalize=config.norm_topk_prob,
            quant_config=quant_config,
            use_grouped_topk=True,
            num_expert_group=config.n_group,
            topk_group=config.topk_group,
            prefix=f"{prefix}.experts",
            scoring_func=config.scoring_func,
            e_score_correction_bias=self.gate.e_score_correction_bias)

        if config.n_shared_experts is not None:
            intermediate_size = (config.moe_intermediate_size *
                                 config.n_shared_experts)
            self.shared_experts = CustomDeepseekDBOMLP(
                hidden_size=config.hidden_size,
                intermediate_size=intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                reduce_results=True,
                prefix=f"{prefix}.shared_experts",
            )
        CustomDeepseekDBOMoE.top_k = config.num_experts_per_tok

        self.dp_size = get_dp_group().world_size

        self.tp_group = get_tp_group().device_group
        self.tp_rank = get_tp_group().rank_in_group

        self.params_dtype = torch.get_default_dtype()

        ascend_config = get_ascend_config()
        self.torchair_graph_enabled = ascend_config.torchair_graph_config.enabled

    def forward(
            self,
            hidden_states: torch.Tensor,
            attn_metadata: Optional[AttentionMetadata] = None) -> torch.Tensor:
        forward_context = get_forward_context()
        # when profile runs, force experts to load balanced tokens
        # to avoid high memory consumption on a single rank.
        enable_force_load_balance = forward_context.in_profile_run

        is_prefill = forward_context.with_prefill

        old_hidden_states = hidden_states.clone()

        # router_logits: (num_tokens, n_experts)
        router_logits, _ = self.gate(hidden_states)

        hidden_states = self.experts(
            hidden_states=hidden_states,
            router_logits=router_logits,
            is_prefill=is_prefill,
            top_k=CustomDeepseekDBOMoE.top_k,
            enable_force_load_balance=enable_force_load_balance,
        ) * self.routed_scaling_factor

        if self.n_shared_experts is not None:
            shared_output = self.shared_experts(old_hidden_states)

        if shared_output is not None:
            hidden_states = hidden_states + shared_output

        return hidden_states

    # ----------------------------------------- TBO-related --------------------------------------------
    def _forward_ms_op_shared_expert(
        self,
        hidden_states: torch.Tensor,
    ):
        shared_output = self.shared_experts._forward_ms_mlp(hidden_states)
        return shared_output

    def _forward_ms_op_gate(
        self,
        hidden_states: torch.Tensor,
    ):
        # router_logits: (num_tokens, n_experts)
        router_logits, _ = self.gate(hidden_states)
        return router_logits

    def _forward_ms_op_tp_allgather(
        self,
        hidden_states: torch.Tensor,
        chunk_hidden_states: torch.Tensor,
        num_tokens: int = 0,
    ):
        current_ms_metadata = get_multistream_comm_context()
        if current_ms_metadata is None:
            dist.all_gather(list(chunk_hidden_states), hidden_states,
                            self.tp_group)
            final_hidden_states = torch.cat(chunk_hidden_states, dim=0)
            if num_tokens > 0:
                final_hidden_states = final_hidden_states[:-num_tokens]
        else:
            current_ms_metadata.before_comm_event.record()
            with torch.npu.stream(current_ms_metadata.comm_stream):
                current_ms_metadata.before_comm_event.wait()
                dist.all_gather(list(chunk_hidden_states), hidden_states,
                                self.tp_group)
                final_hidden_states = torch.cat(chunk_hidden_states, dim=0)
                if num_tokens > 0:
                    final_hidden_states = final_hidden_states[:-num_tokens]
                current_ms_metadata.after_comm_event.record()
        return final_hidden_states


class CustomDeepseekDBOMLAAttention(DeepseekV2MLAAttention):

    def __init__(
        self,
        config: PretrainedConfig,
        hidden_size: int,
        num_heads: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        q_lora_rank: Optional[int],
        kv_lora_rank: int,
        rope_theta: float = 10000,
        rope_scaling: Optional[Dict[str, Any]] = None,
        max_position_embeddings: int = 8192,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        nn.Module.__init__(self)
        self.hidden_size = hidden_size
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.v_head_dim = v_head_dim

        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank

        self.num_heads = num_heads
        tp_size = get_tensor_model_parallel_world_size()
        assert num_heads % tp_size == 0
        self.num_local_heads = num_heads // tp_size

        self.scaling = self.qk_head_dim**-0.5
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings

        if self.q_lora_rank is not None:
            self.q_a_proj = ReplicatedLinear(self.hidden_size,
                                             self.q_lora_rank,
                                             bias=False,
                                             quant_config=quant_config,
                                             prefix=f"{prefix}.q_a_proj")
            self.q_a_layernorm = RMSNorm(self.q_lora_rank,
                                         eps=config.rms_norm_eps)
            self.q_b_proj = ColumnParallelLinear(q_lora_rank,
                                                 self.num_heads *
                                                 self.qk_head_dim,
                                                 bias=False,
                                                 quant_config=quant_config,
                                                 prefix=f"{prefix}.q_b_proj")
        else:
            self.q_proj = ColumnParallelLinear(self.hidden_size,
                                               self.num_heads *
                                               self.qk_head_dim,
                                               bias=False,
                                               quant_config=quant_config,
                                               prefix=f"{prefix}.q_proj")

        self.kv_a_proj_with_mqa = ReplicatedLinear(
            self.hidden_size,
            self.kv_lora_rank + self.qk_rope_head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.kv_a_proj_with_mqa")
        self.kv_a_layernorm = RMSNorm(self.kv_lora_rank,
                                      eps=config.rms_norm_eps)
        self.kv_b_proj = ColumnParallelLinear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.kv_b_proj")
        self.o_proj = CustomDeepseekV2RowParallelLinear(
            self.num_heads * self.v_head_dim,
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj")

        if rope_scaling:
            rope_scaling["rope_type"] = 'deepseek_yarn'
        self.rotary_emb = get_rope(qk_rope_head_dim,
                                   rotary_dim=qk_rope_head_dim,
                                   max_position=max_position_embeddings,
                                   base=rope_theta,
                                   rope_scaling=rope_scaling,
                                   is_neox_style=False)
        if rope_scaling:
            mscale_all_dim = rope_scaling.get("mscale_all_dim", False)
            scaling_factor = rope_scaling["factor"]
            mscale = yarn_get_mscale(scaling_factor, float(mscale_all_dim))
            self.scaling = self.scaling * mscale * mscale

        # In the MLA backend, kv_cache includes both k_c and
        # pe (i.e. decoupled position embeddings). In particular,
        # the concat_and_cache_mla op requires
        #     k_c.size(1) + k_pe.size(1) == kv_cache.size(2)
        # i.e.
        #     kv_lora_rank + qk_rope_head_dim == head_size
        self.mla_attn = Attention(
            num_heads=self.num_local_heads,
            head_size=self.kv_lora_rank + self.qk_rope_head_dim,
            scale=self.scaling,
            num_kv_heads=1,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
            use_mla=True,
            # MLA Args
            q_lora_rank=self.q_lora_rank,
            kv_lora_rank=self.kv_lora_rank,
            qk_nope_head_dim=self.qk_nope_head_dim,
            qk_rope_head_dim=self.qk_rope_head_dim,
            qk_head_dim=self.qk_head_dim,
            v_head_dim=self.v_head_dim,
            rotary_emb=self.rotary_emb,
            q_proj=self.q_proj if self.q_lora_rank is None else self.q_b_proj,
            kv_a_proj_with_mqa=self.kv_a_proj_with_mqa,
            kv_a_layernorm=self.kv_a_layernorm,
            kv_b_proj=self.kv_b_proj,
            o_proj=self.o_proj,
        )

        self.prefix = prefix
        self.debug_layer_idx = int(self.prefix.split(".")[-2])

        ascend_config = get_ascend_config()
        self.torchair_graph_enabled = ascend_config.torchair_graph_config.enabled

    def forward(
            self,
            positions: torch.Tensor,
            hidden_states: torch.Tensor,
            kv_cache: Optional[torch.Tensor] = None,
            attn_metadata: Optional[AttentionMetadata] = None) -> torch.Tensor:
        if self.q_lora_rank is not None:
            ckq = self.q_a_proj(hidden_states)[0]
            hidden_states_or_q_c = self.q_a_layernorm(ckq)
        else:
            hidden_states_or_q_c = hidden_states
        if self.torchair_graph_enabled:
            forward_kwargs = {}
            output_shape = hidden_states.shape
            output = torch.empty(output_shape,
                                 dtype=hidden_states_or_q_c.dtype,
                                 device=hidden_states_or_q_c.device)
            forward_kwargs['output'] = output
            output = self.mla_attn.impl.forward(self.mla_attn,
                                                hidden_states_or_q_c,
                                                hidden_states, None, kv_cache,
                                                attn_metadata,
                                                **forward_kwargs)
            output = output.view(-1, output_shape[-1])
            return output
        else:
            kv_c, k_pe = self.kv_a_proj_with_mqa(hidden_states)[0].split(
                [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
            kv_c_normed = self.kv_a_layernorm(kv_c.contiguous())
            return self.mla_attn(hidden_states_or_q_c,
                                 kv_c_normed,
                                 k_pe,
                                 output_shape=hidden_states.shape)


class CustomDeepseekDBODecoderLayer(DeepseekV2DecoderLayer):

    def __init__(
        self,
        config: PretrainedConfig,
        prefix: str,
        model_config: ModelConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
    ) -> None:
        nn.Module.__init__(self)
        self.hidden_size = config.hidden_size
        rope_theta = getattr(config, "rope_theta", 10000)
        rope_scaling = getattr(config, "rope_scaling", None)
        max_position_embeddings = getattr(config, "max_position_embeddings",
                                          8192)
        # DecoderLayers are created with `make_layers` which passes the prefix
        # with the layer's index.
        layer_idx = int(prefix.split(sep='.')[-1])
        self.layer_idx = layer_idx
        # TODO: enable mla in vllm-ascend
        if model_config.use_mla:
            attn_cls = CustomDeepseekDBOMLAAttention
        else:
            attn_cls = DeepseekV2Attention
        self.self_attn = attn_cls(
            config=config,
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            qk_nope_head_dim=config.qk_nope_head_dim,
            qk_rope_head_dim=config.qk_rope_head_dim,
            v_head_dim=config.v_head_dim,
            q_lora_rank=config.q_lora_rank
            if hasattr(config, "q_lora_rank") else None,
            kv_lora_rank=config.kv_lora_rank,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            max_position_embeddings=max_position_embeddings,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
        )

        if (config.n_routed_experts is not None
                and layer_idx >= config.first_k_dense_replace
                and layer_idx % config.moe_layer_freq == 0):
            self.mlp = CustomDeepseekDBOMoE(
                config=config,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
            )
        else:
            self.mlp = CustomDeepseekDBOMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
            )
        self.input_layernorm = RMSNorm(config.hidden_size,
                                       eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size,
                                                eps=config.rms_norm_eps)
        self.routed_scaling_factor = config.routed_scaling_factor

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
        kv_cache: Optional[torch.Tensor] = None,
        attn_metadata: Optional[AttentionMetadata] = None,
    ) -> torch.Tensor:
        # Self Attention
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            previous_hidden_states, previous_residual = hidden_states, residual
            hidden_states, residual = self.input_layernorm(
                hidden_states, residual)
            # Dispose hidden_states and residual from the previous layer
            # to save npu memory because they're no longer used.
            dispose_tensor(previous_hidden_states)
            dispose_tensor(previous_residual)

        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            kv_cache=kv_cache,
            attn_metadata=attn_metadata,
        )

        if hidden_states.dtype == torch.float16:
            # Fix FP16 overflow
            # We scale both hidden_states and residual before
            # rmsnorm, and rmsnorm result would not affect by scale.
            hidden_states *= 1. / self.routed_scaling_factor
            if self.layer_idx == 0:
                # The residual is shared by all layers, we only scale it on
                # first layer.
                residual *= 1. / self.routed_scaling_factor

        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(
            hidden_states, residual)

        if isinstance(self.mlp, CustomDeepseekDBOMoE):
            hidden_states = self.mlp(hidden_states, attn_metadata)
        else:
            hidden_states = self.mlp(hidden_states)

        if isinstance(
                self.mlp,
                CustomDeepseekDBOMLP) and hidden_states.dtype == torch.float16:
            # Fix FP16 overflow
            # Scaling the DeepseekV2MLP output, it is the input of
            # input_layernorm of next decoder layer.
            # The scaling of DeepseekV2MOE output would be done in the forward
            # of DeepseekV2MOE
            hidden_states *= 1. / self.routed_scaling_factor

        return hidden_states, residual

    # ----------------------------------------- TBO-related --------------------------------------------
    def _forward_ms_layer(
        self,
        positions: List[torch.Tensor],
        hidden_states: List[torch.Tensor],
        residual: List[torch.Tensor],
        attn_metadata: List[AttentionMetadata],
        kv_cache: Optional[torch.Tensor] = None,
        is_prefill: bool = False,
    ) -> tuple[List[torch.Tensor], List[torch.Tensor]]:
        layer_index, ms_metadata, _ = get_multistream_layer_context()
        assert layer_index >= 0 and ms_metadata is not None
        num_micro_batchs = ms_metadata.ms_config.num_micro_batches
        assert isinstance(self.mlp, CustomDeepseekDBOMoE)
        assert len(positions) == num_micro_batchs
        assert len(hidden_states) == num_micro_batchs
        assert residual is not None
        assert attn_metadata is not None
        num_tokens = []
        hidden_dims = []
        shared_outputs = []
        router_logits = []
        chunk_hidden_states = []

        # block 1 : attention
        # block 2 : attn tp communication
        # the attn computation of microbatch 1 can be overlapped with the moe
        # communication in the previous layer, and the attn computation of microbatch 2
        # can be overlapped with the attn communication of microbatch 1
        for i in range(num_micro_batchs):
            # wait last layer moe finishing communication
            ms_metadata.try_wait_event(layer_index - 1, i,
                                       MSEventKey.FFN_AR_FINISH)
            context = MultiStreamStepMetadata(
                comm_stream=ms_metadata.communicate_stream,
                before_comm_event=ms_metadata.ms_events[layer_index][i][
                    MSEventKey.ATTN_COM_FINISH],
                after_comm_event=ms_metadata.ms_events[layer_index][i][
                    MSEventKey.ATTN_AR_FINISH],
            )

            with set_multistream_context(context, i):
                forward_context = get_forward_context()
                forward_context.attn_metadata = attn_metadata[i]

                # input layernorm
                hidden_states[i], residual[
                    i] = self._forward_ms_op_input_layernorm(
                        hidden_states[i], residual[i])
                # attention and tp allreduce
                hidden_states[i], residual[i] = self._forward_ms_op_attn(
                    positions[i], hidden_states[i], residual[i], kv_cache,
                    attn_metadata[i])

        # block 3 : shared experts
        # if there is an allreduce ops in shared expert, we can overlap it with the computation of the
        # shared expert for next microbatch or moe gating
        for i in range(num_micro_batchs):
            ms_metadata.try_wait_event(layer_index, i,
                                       MSEventKey.ATTN_AR_FINISH)
            context = MultiStreamStepMetadata(
                comm_stream=ms_metadata.communicate_stream,
                before_comm_event=ms_metadata.ms_events[layer_index][i][
                    MSEventKey.MOE_SE_COMP_FINISH],
                after_comm_event=ms_metadata.ms_events[layer_index][i][
                    MSEventKey.MOE_SE_COMM_FINISH],
            )
            with set_multistream_context(context, i):
                # compute shared expert after finishing ATTN AR
                hidden_states[i], residual[
                    i] = self._forward_ms_op_post_attn_layernorm(
                        hidden_states[i], residual[i])

                num_token, hidden_dim = hidden_states[i].shape
                hidden_states[i] = hidden_states[i].view(-1, hidden_dim)
                num_tokens.append(num_token)
                hidden_dims.append(hidden_dim)
                if self.mlp.n_shared_experts is not None:
                    # TODO: we can move shared expert computation into next block if reduce results is false
                    shared_output = self.mlp._forward_ms_op_shared_expert(
                        hidden_states[i])
                    shared_outputs.append(shared_output)

        # block 4 : moe
        for i in range(num_micro_batchs):
            # when profile runs, force experts to load balanced tokens
            # to avoid high memory consumption on a single rank.
            # TODO: need a better flag to indicate whether in profile run or not.
            if attn_metadata[i] is None:
                # for profile run
                is_prefill = True
                enable_force_load_balance = True
            else:
                is_prefill = attn_metadata[i].num_prefills > 0
                enable_force_load_balance = False

            if self.mlp.tp_size > 1:
                num_token, _ = hidden_states[i].shape
                padded_num_tokens = (self.mlp.tp_size - num_tokens[i] %
                                     self.mlp.tp_size) % self.mlp.tp_size
                if padded_num_tokens > 0:
                    hidden_states[i] = nn.functional.pad(
                        hidden_states[i], (0, 0, 0, padded_num_tokens))
                chunk_hidden_state = torch.tensor_split(hidden_states[i],
                                                        self.mlp.tp_size,
                                                        dim=0)
                chunk_hidden_states.append(chunk_hidden_state)
                local_hidden_states = chunk_hidden_state[self.mlp.tp_rank]
            else:
                local_hidden_states = hidden_states[i]

            router_logit = self.mlp._forward_ms_op_gate(local_hidden_states)
            router_logits.append(router_logit)

            if CustomDeepseekDBOMoE.top_k:
                real_top_k = CustomDeepseekDBOMoE.top_k
            else:
                real_top_k = self.mlp.experts.top_k

            hidden_states[i] = self.mlp.experts._forward_ms_fused_moe_comp(
                local_hidden_states, router_logits[i], is_prefill, real_top_k,
                enable_force_load_balance)

            # the following kernels will be submitted to the comm stream to overlap the computation of the
            # moe computation of next microbatch and the attn computation of next layer
            context = MultiStreamStepMetadata(
                comm_stream=ms_metadata.communicate_stream,
                before_comm_event=ms_metadata.ms_events[layer_index][i][
                    MSEventKey.FFN_COM_FINISH],
                after_comm_event=ms_metadata.ms_events[layer_index][i][
                    MSEventKey.MOE_AFTER_COMM],
            )
            context.before_comm_event.record()
            with torch.npu.stream(ms_metadata.communicate_stream):
                context.before_comm_event.wait()
                if self.mlp.experts.reduce_results and (
                        self.mlp.experts.tp_size > 1
                        or self.mlp.experts.ep_size > 1):
                    hidden_states[i] = tensor_model_parallel_all_reduce(
                        hidden_states[i])
                hidden_states[
                    i] = hidden_states[i] * self.mlp.routed_scaling_factor
                context.after_comm_event.record()

            context = MultiStreamStepMetadata(
                comm_stream=ms_metadata.communicate_stream,
                before_comm_event=ms_metadata.ms_events[layer_index][i][
                    MSEventKey.MOE_AFTER_COMM],
                after_comm_event=ms_metadata.ms_events[layer_index][i][
                    MSEventKey.FFN_AR_FINISH],
            )
            with set_multistream_context(context, i):
                if self.mlp.tp_size > 1:
                    hidden_states[i] = self.mlp._forward_ms_op_tp_allgather(
                        hidden_states[i], chunk_hidden_states[i],
                        padded_num_tokens)
            with torch.npu.stream(ms_metadata.communicate_stream):
                # last
                if shared_outputs[i] is not None:
                    hidden_states[i] = hidden_states[i] + shared_outputs[i]
                hidden_states[i] = hidden_states[i].view(
                    num_tokens[i], hidden_dims[i])
                if isinstance(self.mlp, CustomDeepseekDBOMLP
                              ) and hidden_states[i].dtype == torch.float16:
                    # Fix FP16 overflow
                    # Scaling the DeepseekV2MLP output, it is the input of
                    # input_layernorm of next decoder layer.
                    # The scaling of DeepseekV2MOE output would be done in the forward
                    # of DeepseekV2MOE
                    hidden_states[i] *= 1. / self.routed_scaling_factor
                context.after_comm_event.record()
        return hidden_states, residual

    # should split ops in Decoder Layer
    def _forward_ms_op_input_layernorm(
        self,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(
                hidden_states, residual)
        return hidden_states, residual

    def _forward_ms_op_attn(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        kv_cache: Optional[torch.Tensor] = None,
        attn_metadata: Optional[AttentionMetadata] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            kv_cache=kv_cache,
            attn_metadata=attn_metadata,
        )
        if hidden_states.dtype == torch.float16:
            # Fix FP16 overflow
            # We scale both hidden_states and residual before
            # rmsnorm, and rmsnorm result would not affect by scale.
            hidden_states *= 1. / self.routed_scaling_factor
            if self.layer_idx == 0:
                # The residual is shared by all layers, we only scale it on
                # first layer.
                residual *= 1. / self.routed_scaling_factor
        return hidden_states, residual

    def _forward_ms_op_post_attn_layernorm(
        self,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
    ):
        hidden_states, residual = self.post_attention_layernorm(
            hidden_states, residual)
        return hidden_states, residual


class CustomDeepseekDBOModel(nn.Module):

    fall_back_to_pt_during_load = False

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config = vllm_config.model_config.hf_config
        model_config = vllm_config.model_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config

        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.first_k_dense_replace = config.first_k_dense_replace

        if get_pp_group().is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=f"{prefix}.embed_tokens")
        else:
            self.embed_tokens = PPMissingLayer()

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: CustomDeepseekDBODecoderLayer(
                config,
                prefix,
                model_config=model_config,
                cache_config=cache_config,
                quant_config=quant_config,
            ),
            prefix=f"{prefix}.layers")

        if get_pp_group().is_last_rank:
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()
        self.make_empty_intermediate_tensors = (
            make_empty_intermediate_tensors_factory(
                ["hidden_states", "residual"], config.hidden_size))

        # tbo related members
        if VLLM_ASCEND_ENABLE_DBO:
            self.use_mla = model_config.use_mla
            self.multistream_config = MultiStreamConfig()
            multistream_metadata = make_multistream_metadata_ds(
                start_layer=self.start_layer + self.first_k_dense_replace,
                end_layer=self.end_layer,
                causal_lm=getattr(config, "causal_lm", True),
                multistream_config=self.multistream_config,
            )
            self.ms_pre_layer = MultiStreamPreTransformerLayer(
                multistream_metadata)
            self.ms_post_layer = MultiStreamPostTransformerLayer(
                multistream_metadata)

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: Optional[List[torch.Tensor]] = None,
        attn_metadata: Optional[AttentionMetadata] = None,
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

        num_normal_layers = (self.first_k_dense_replace
                             if VLLM_ASCEND_ENABLE_DBO and self.can_run_ms()
                             else self.end_layer - self.start_layer)

        moe_start_layer = self.start_layer + num_normal_layers
        for i in range(self.start_layer, min(moe_start_layer, self.end_layer)):
            layer = self.layers[i]
            hidden_states, residual = layer(
                positions, hidden_states, residual,
                kv_caches[i -
                          self.start_layer] if kv_caches is not None else None,
                attn_metadata)

        if moe_start_layer < self.end_layer:
            # if we enable multistream/dbo, process sparse layers here
            hidden_states, residual = self._forward_ms_layers(
                positions=positions,
                hidden_states=hidden_states,
                residual=residual,
                moe_start_layer=moe_start_layer,
                kv_caches=kv_caches,
            )

        if not get_pp_group().is_last_rank:
            return IntermediateTensors({
                "hidden_states": hidden_states,
                "residual": residual
            })

        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states

    def can_run_ms(self):
        attn_metadata = get_forward_context().attn_metadata
        # enable prefill overlap
        return not (attn_metadata is None or attn_metadata.num_prefills == 0
                    or not attn_metadata.enable_dbo_across_dp)

    def _forward_ms_layers(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        moe_start_layer: int,
        kv_caches: Optional[List[torch.Tensor]] = None,
        is_prefill: bool = False,
    ):

        if moe_start_layer == self.end_layer:
            return hidden_states, residual

        attn_metadata, [positions, hidden_states,
                        residual] = self.ms_pre_layer(
                            [positions, hidden_states, residual], )
        # the rest layers
        for i in range(moe_start_layer, self.end_layer):
            layer = self.layers[i]
            hidden_states, residual = layer._forward_ms_layer(
                positions=positions,
                hidden_states=hidden_states,
                residual=residual,
                attn_metadata=attn_metadata,
                kv_cache=kv_caches[i - self.start_layer]
                if kv_caches is not None else None,
                is_prefill=is_prefill)
            advance_step_multistream_layer_context()

        [hidden_states,
         residual] = self.ms_post_layer([hidden_states, residual], )
        return hidden_states, residual


class CustomDeepseekDBOForCausalLM(DeepseekV2ForCausalLM):
    # add `packed_modules_mapping` in `DeepseekV2ForCausalLM` to support weight merging
    packed_modules_mapping = {
        "gate_up_proj": ["gate_proj", "up_proj"],
        "experts":
        ["experts.0.gate_proj", "experts.0.up_proj", "experts.0.down_proj"]
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        nn.Module.__init__(self)
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.quant_config = quant_config
        self.model = CustomDeepseekDBOModel(vllm_config=vllm_config,
                                            prefix=maybe_prefix(
                                                prefix, "model"))
        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(config.vocab_size,
                                          config.hidden_size,
                                          quant_config=quant_config)
        else:
            self.lm_head = PPMissingLayer()
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.sampler = get_sampler()
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors)

    # NOTE: This `load_weights` is mainly copied from
    # https://github.com/vllm-project/vllm/commit/07b8fae219b1fff51ef115c38c44b51395be5bb5
    # to fix CI, and it is different from the implementation in main
    # TODO: support eplb style load_weights
    def load_weights(self, weights: Iterable[tuple[str,
                                                   torch.Tensor]]) -> set[str]:
        """"""
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        # Params for weights, fp8 weight scales, fp8 activation scales
        # (param_name, weight_name, expert_id, shard_id)
        expert_params_mapping = AscendFusedMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.n_routed_experts)

        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue

            spec_layer = get_spec_layer_idx_from_weight_name(self.config, name)
            if spec_layer is not None:
                continue  # skip spec decode layers for main model

            for (param_name, weight_name, shard_id) in stacked_params_mapping:
                # Skip non-stacked layers and experts (experts handled below).
                if weight_name not in name:
                    continue
                # We have mlp.experts[0].gate_proj in the checkpoint.
                # Since we handle the experts below in expert_params_mapping,
                # we need to skip here BEFORE we update the name, otherwise
                # name will be updated to mlp.experts[0].gate_up_proj, which
                # will then be updated below in expert_params_mapping
                # for mlp.experts[0].gate_gate_up_proj, which breaks load.
                if (("mlp.experts." in name) and name not in params_dict):
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
                                  expert_id=expert_id,
                                  return_success=False)
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

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: Optional[List[torch.Tensor]] = None,
        attn_metadata: Optional[AttentionMetadata] = None,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        hidden_states = self.model(input_ids, positions, kv_caches,
                                   attn_metadata, intermediate_tensors,
                                   inputs_embeds)
        return hidden_states
