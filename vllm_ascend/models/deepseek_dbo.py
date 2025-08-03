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

from typing import List, Optional, Union

import torch
import torch.distributed as dist
import torch_npu  # noqa: F401
from torch import nn
from transformers import PretrainedConfig
from vllm.attention import AttentionMetadata
from vllm.config import CacheConfig, ModelConfig, VllmConfig
from vllm.distributed import (get_pp_group, get_tensor_model_parallel_rank,
                              get_tensor_model_parallel_world_size,
                              get_tp_group, tensor_model_parallel_all_reduce,
                              tensor_model_parallel_all_gather,)
from vllm.distributed.parallel_state import get_dp_group, get_ep_group
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import UnquantizedLinearMethod
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.sampler import get_sampler
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead, VocabParallelEmbedding)
from vllm.model_executor.models.deepseek_v2 import \
    DeepseekV2ForCausalLM  # noqa: E501
from vllm.model_executor.models.utils import (
    PPMissingLayer, make_empty_intermediate_tensors_factory, make_layers,
    maybe_prefix)
from vllm.sequence import IntermediateTensors

import vllm_ascend.envs as envs_ascend
from vllm_ascend.ascend_forward_context import FusedMoEState
from vllm_ascend.distributed.tensor_parallel import \
    gather_from_sequence_parallel_region
from vllm_ascend.models.deepseek_v2 import (CustomDeepseekV2DecoderLayer,
                                            CustomDeepseekV2MLP,
                                            CustomDeepseekV2MoE,
                                            CustomDeepseekV2ForCausalLM)
from vllm_ascend.multistream.base import MSEventKey
from vllm_ascend.multistream.context import (
    advance_step_multistream_layer_context, get_multistream_comm_context,
    get_multistream_layer_context, set_multistream_context)
from vllm_ascend.multistream.layers import (MultiStreamPostTransformerLayer,
                                            MultiStreamPreTransformerLayer)
from vllm_ascend.multistream.metadata import (MultiStreamConfig,
                                              MultiStreamStepMetadata,
                                              make_multistream_metadata_ds)
from vllm_ascend.ops.fused_moe import select_experts
from vllm_ascend.quantization.w8a8_dynamic import (
    AscendW8A8DynamicLinearMethod, apply_mlp)
from vllm_ascend.utils import dispose_tensor

VLLM_ASCEND_ENABLE_DBO: bool = envs_ascend.VLLM_ASCEND_ENABLE_DBO


class CustomDeepseekDBOMLP(CustomDeepseekV2MLP):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: Optional[QuantizationConfig] = None,
        reduce_results: bool = True,
        force_replicate: bool = False,
        prefix: str = "",
    ) -> None:
        super().__init__(hidden_size=hidden_size,
                         intermediate_size=intermediate_size,
                         hidden_act=hidden_act,
                         quant_config=quant_config,
                         prefix=prefix,
                         force_replicate=force_replicate,
                         reduce_results=reduce_results)
        self.is_dynamic_quant = not isinstance(
            self.gate_up_proj.quant_method,
            UnquantizedLinearMethod) and isinstance(
                self.gate_up_proj.quant_method.quant_method,
                AscendW8A8DynamicLinearMethod)

    def _forward_ms_mlp(self, x):
        current_ms_metadata = get_multistream_comm_context()
        assert current_ms_metadata is not None
        gate_up, _ = self.gate_up_proj(x)
        if self.is_dynamic_quant:
            x, dynamic_scale = self.act_fn(gate_up)
            x = torch_npu.npu_quant_matmul(
                x,
                self.down_proj.weight,
                self.down_proj.weight_scale,
                pertoken_scale=dynamic_scale,
                output_dtype=torch.bfloat16,
            )
            if self.down_proj.reduce_results and self.down_proj.tp_size > 1:
                current_ms_metadata.before_comm_event.record()
                with torch.npu.stream(current_ms_metadata.comm_stream):
                    current_ms_metadata.before_comm_event.wait()
                    x = tensor_model_parallel_all_reduce(x)
                    current_ms_metadata.after_comm_event.record()
        else:
            x = self.act_fn(gate_up)
            x, _ = self.down_proj(x)
        return x


class CustomDeepseekDBOMoE(CustomDeepseekV2MoE):

    top_k: int

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__(config=config,
                         quant_config=quant_config,
                         prefix=prefix)

        if config.n_shared_experts is not None:
            self.all_reduce_merge = self.experts.all_reduce_merge
            reduce_results = not self.all_reduce_merge
            intermediate_size = (config.moe_intermediate_size *
                                 config.n_shared_experts)
            self.shared_experts = CustomDeepseekDBOMLP(
                hidden_size=config.hidden_size,
                intermediate_size=intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                reduce_results=reduce_results,
                force_replicate=self.enable_multistream_moe,
                prefix=f"{prefix}.shared_experts",
            )
        else:
            self.shared_experts = None  # type: ignore
        CustomDeepseekDBOMoE.top_k = config.num_experts_per_tok
        self.config = config

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

    # def _forward_op_gating(
    #         self,
    #         hidden_states: torch.Tensor,
    #         attn_metadata: Optional[AttentionMetadata] = None) -> torch.Tensor:
    #     if attn_metadata is None:
    #         attn_metadata = get_forward_context().attn_metadata
    #     # when profile runs, force experts to load balanced tokens
    #     # to avoid high memory consumption on a single rank.
    #     # TODO: need a better flag to indicate whether in profile run or not.
    #     enable_force_load_balance = get_forward_context().in_profile_run

    #     num_tokens, hidden_dim = hidden_states.shape

    #     if self.tp_size > 1:
    #         # pass
    #         num_tokens, hidden_size = hidden_states.shape
    #         if num_tokens < self.tp_size:
    #             hidden_states = nn.functional.pad(
    #                 hidden_states, (0, 0, 0, self.tp_size - num_tokens))
    #         chunk_hidden_states = torch.tensor_split(hidden_states,
    #                                                  self.tp_size,
    #                                                  dim=0)
    #         chunked_hidden_states_sizes = [
    #             x.shape[0] for x in chunk_hidden_states
    #         ]
    #         local_hidden_states = chunk_hidden_states[self.tp_rank]
    #     else:
    #         local_hidden_states = hidden_states
    #         chunked_hidden_states_sizes = None

    #     # router_logits: (num_tokens, n_experts)
    #     router_logits, _ = self.gate(local_hidden_states)

    #     # NOTE: now npu_moe_gating_top_k can only support `group_count=256` pattern
    #     if self.config.n_routed_experts == 256:
    #         topk_weights, topk_ids, _ = torch_npu.npu_moe_gating_top_k(
    #             router_logits,
    #             k=self.config.num_experts_per_tok,
    #             bias=self.gate.e_score_correction_bias,
    #             k_group=self.config.topk_group,  # fix: 4
    #             group_count=self.config.n_group,  # fix 8
    #             group_select_mode=1,  # 0: max in group; 1: topk2.sum(fix)
    #             renorm=0,  # 0: softmax->topk(fix); 1: topk->softmax
    #             norm_type=1,  # 0: softmax; 1: sigmoid(fix)
    #             routed_scaling_factor=1,
    #             eps=float(1e-20))
    #     else:
    #         topk_weights, topk_ids = select_experts(
    #             hidden_states=local_hidden_states,
    #             router_logits=router_logits,
    #             top_k=self.config.num_experts_per_tok,
    #             use_grouped_topk=True,
    #             renormalize=self.config.norm_topk_prob,
    #             topk_group=self.config.topk_group,
    #             num_expert_group=self.config.n_group,
    #             custom_routing_function=None,
    #             scoring_func=self.config.scoring_func,
    #             e_score_correction_bias=self.gate.e_score_correction_bias,
    #         )

    #     topk_weights = topk_weights.to(hidden_states.dtype)
    #     # this is a naive implementation for experts load balance so as
    #     # to avoid accumulating too much tokens on a single rank.
    #     # currently it is only activated when doing profile runs.
    #     if enable_force_load_balance:
    #         topk_ids = torch.randint_like(topk_ids, 0,
    #                                       self.config.n_routed_experts)

    #     return topk_weights, topk_ids, local_hidden_states, chunked_hidden_states_sizes

    # def _forward_op_shared_experts(self, hidden_states):
    #     if self.n_shared_experts is not None:
    #         shared_output = self.shared_experts(hidden_states)

    #     return shared_output

    # def _forward_op_grouped_mlp(self, dispatched_input, tokens_per_expert):
    #     from vllm_ascend.ops.fused_moe import apply_mlp
    #     return apply_mlp(dispatched_input, self.experts.w13_weight,
    #                      self.experts.w2_weight, tokens_per_expert)

    # def _forward_combine_comm(self, hidden_states, microbatch_id, num_tokens,
    #                           chunked_hidden_states_sizes):
    #     token_dispatcher = self.experts.token_dispatchers[microbatch_id]
    #     final_hidden_states, _ = token_dispatcher.token_unpermutation(
    #         hidden_states)
    #     if hasattr(self, 'routed_scaling_factor'):
    #         final_hidden_states = final_hidden_states * self.routed_scaling_factor

    #     if self.tp_size > 1:
    #         final_hidden_states = gather_from_sequence_parallel_region(
    #             final_hidden_states, self.tp_group,
    #             chunked_hidden_states_sizes)
    #         if num_tokens < self.tp_size:
    #             final_hidden_states = final_hidden_states[:num_tokens]

    #     if self.shared_experts is not None:
    #         final_hidden_states = final_hidden_states + token_dispatcher.cached_shared_expert_output
    #         token_dispatcher.cached_shared_expert_output.untyped_storage(
    #         ).resize_(0)
    #         token_dispatcher.cached_shared_expert_output = None

    #     final_hidden_states = final_hidden_states.view(num_tokens, -1)

    #     return final_hidden_states


class CustomDeepseekDBODecoderLayer(CustomDeepseekV2DecoderLayer):

    def __init__(
        self,
        config: PretrainedConfig,
        prefix: str,
        model_config: ModelConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
    ) -> None:
        super().__init__(config=config,
                         prefix=prefix,
                         model_config=model_config,
                         cache_config=cache_config,
                         quant_config=quant_config)
        self.tp_size = get_tensor_model_parallel_world_size()
        self.dp_size = get_dp_group().world_size
        self.tp_group = get_tp_group().device_group
        self.global_num_experts = config.n_routed_experts

        if (config.n_routed_experts is not None
                and self.layer_idx >= config.first_k_dense_replace
                and self.layer_idx % config.moe_layer_freq == 0):
            self.mlp = CustomDeepseekDBOMoE(
                config=config,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
            )
            self.mla_moe_communication = ascend_config.torchair_graph_config.enable_multistream_moe \
                and model_config.use_mla and self.tp_size > 1
        else:
            self.mlp = CustomDeepseekDBOMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,

                prefix=f"{prefix}.mlp",
            )
            self.mla_moe_communication = False

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
        kv_cache: Optional[torch.Tensor] = None,
        attn_metadata: Optional[AttentionMetadata] = None,
        replace_allreduce: bool = False,
    ) -> torch.Tensor:
        if attn_metadata is not None and attn_metadata.num_decodes > 0:
            mla_moe_communication = self.mla_moe_communication and replace_allreduce
        else:
            mla_moe_communication = False
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
        if mla_moe_communication and residual.shape[0] != hidden_states.shape[
                0]:
            chunk_hidden_states = torch.tensor_split(residual,
                                                     self.tp_size,
                                                     dim=0)
            residual = chunk_hidden_states[self.tp_rank]

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
            hidden_states = self.mlp(hidden_states,
                                     attn_metadata,
                                     replace_allreduce=mla_moe_communication)
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
        if mla_moe_communication and self.layer_idx == self.layers - 1:
            hidden_states = tensor_model_parallel_all_gather(hidden_states,
                                                             dim=0)
            residual = tensor_model_parallel_all_gather(residual, dim=0)

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
        chunk_router_logits = []
        topk_weights = []
        topk_ids = []
        num_moe_tokens = []
        original_shapes = []
        expanded_row_idx = []
        scatter_size_list = []
        gather_size_list = []
        local_expert_idx = []
        scatter_sizes = []
        expanded_expert_idx = []
        sorted_local_expert_idx = []
        sorted_idx = []
        global_num_experts = len(
            self.mlp.experts.expert_map
        ) if self.mlp.experts.expert_map is not None else self.global_num_experts
        ep_group = get_ep_group()
        local_num_experts = global_num_experts // ep_group.world_size
        fused_moe_state = get_forward_context().fused_moe_state
        ep_rank = torch.distributed.get_rank(group=ep_group)
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
            router_logit = self.mlp._forward_ms_op_gate(hidden_states[i])
            router_logits.append(router_logit)
            batch_size, hidden_size = hidden_states[i].shape
            if CustomDeepseekDBOMoE.top_k:
                real_top_k = CustomDeepseekDBOMoE.top_k
            else:
                real_top_k = self.mlp.experts.top_k

            # NOTE: now npu_moe_gating_top_k can only support `group_count=256` pattern
            topk_weight, topk_id, _ = torch_npu.npu_moe_gating_top_k(
                router_logits[i],
                k=real_top_k,  # topk当前写8
                bias=self.mlp.experts.e_score_correction_bias,
                k_group=self.mlp.experts.topk_group,  # fix: 4
                group_count=self.mlp.experts.num_expert_group,  # fix 8
                group_select_mode=1,  # 0: group中的最大; 1: topk2.sum(fix)
                renorm=0,  # 0: softmax->topk(fix); 1: topk->softmax
                norm_type=1,  # 0: softmax; 1: sigmoid(fix)
                # out_flag=False, # todo new api; 第三个输出是否输出
                # y2_flag=False, # old api; 第三个输出是否输出
                routed_scaling_factor=1,
                eps=float(1e-20))

            topk_weight = topk_weight.to(hidden_states[i].dtype)
            topk_weights.append(topk_weight)
            topk_ids.append(topk_id)
            original_shape = hidden_states[i].shape
            original_shapes.append(original_shape)
            if len(original_shapes[i]) == 3:
                hidden_states[i] = hidden_states[i].view(
                    -1, hidden_states[i].shape[-1])
            num_token, _ = hidden_states[i].shape
            num_moe_tokens.append(num_token)
            device = hidden_states[i].device

            row_idx_len = num_moe_tokens[i] * real_top_k
            row_idx = (torch.arange(0,
                                    row_idx_len,
                                    dtype=torch.int32,
                                    device=device).view(real_top_k,
                                                        -1).permute(
                                                            1, 0).contiguous())

            # ----------------------------------------------------------------------------
            hidden_states[i], pertoken_scale = torch_npu.npu_dynamic_quant(hidden_states[i])

            hidden_states[i], expanded_x_idx, expert_tokens, pertoken_scale = torch_npu.npu_moe_init_routing_v2(
                hidden_states[i],
                topk_ids[i],
                scale=pertoken_scale,
                offset=None,
                active_num=num_tokens[i] * real_top_k,
                expert_num=global_num_experts,
                expert_tokens_num_type=1,
                expert_tokens_num_flag=True,
                active_expert_range=[
                    ep_rank * local_num_experts, (ep_rank + 1) * local_num_experts
                ],
                quant_mode=-1,
                row_idx_type=1)

            sorted_topk_weight = torch.index_select(topk_weights[i].view(-1), 0,
                                                    expanded_x_idx)
            row_index = expanded_x_idx // topk_ids[i].shape[-1]
            row_index = row_index.to(torch.int64)
            share_input = torch.zeros((batch_size, hidden_size),
                                        dtype=torch.bfloat16,
                                        device="npu")

            hidden_states[i] = torch_npu.npu_grouped_matmul(
                x=[hidden_states[i]],
                weight=[self.mlp.experts.w13_weight,],
                split_item=3,
                group_list_type=1,
                group_type=0,
                group_list=expert_tokens,
                output_dtype=torch.int32)[0]

            hidden_states[i], pertoken_scale = torch_npu.npu_dequant_swiglu_quant(
                x=hidden_states[i],
                weight_scale=self.mlp.experts.w13_weight_scale.to(torch.float32),
                activation_scale=pertoken_scale,
                bias=None,
                quant_scale=None,
                quant_offset=None,
                group_index=expert_tokens,
                activate_left=True,
                quant_mode=1,
            )            

            hidden_states[i] = torch_npu.npu_grouped_matmul_finalize_routing(
                hidden_states[i],
                self.mlp.experts.w2_weight,
                scale=self.mlp.experts.w2_weight_scale.to(torch.float32),
                bias=None,
                pertoken_scale=pertoken_scale.view(-1),
                group_list=expert_tokens,
                shared_input=share_input,
                logit=sorted_topk_weight.to(torch.float32),
                row_index=row_index,
                output_bs=batch_size).to(torch.bfloat16)

            if len(original_shape) == 3:
                hidden_states[i] = hidden_states[i].view(original_shape)


            context = MultiStreamStepMetadata(
                comm_stream=ms_metadata.communicate_stream,
                before_comm_event=ms_metadata.ms_events[layer_index][i][
                    MSEventKey.MOE_GATE_FINISH],
                after_comm_event=ms_metadata.ms_events[layer_index][i][
                    MSEventKey.FFN_AR_FINISH],
            )
            context.before_comm_event.record()
            with torch.npu.stream(ms_metadata.communicate_stream):
                context.before_comm_event.wait()
                # When all_reduce_merge is in progress, shared_experts does not do all_reduce in mlp, but waits until shared_experts+router_experts are completed before doing all_reduce
                hidden_states[i] = tensor_model_parallel_all_reduce(hidden_states[i])

                if shared_outputs[i] is not None:
                    hidden_states[i] = hidden_states[
                        i] * self.routed_scaling_factor + shared_outputs[i]
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

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()

        config = vllm_config.model_config.hf_config
        model_config = vllm_config.model_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config

        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.tp_size = get_tensor_model_parallel_world_size()
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
        graph_enable: Optional[bool] = True
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

        replace_allreduce = hidden_states.shape[0] % self.tp_size == 0

        num_normal_layers = (self.first_k_dense_replace
                             if VLLM_ASCEND_ENABLE_DBO and not graph_enable
                             and self.can_run_ms() else self.end_layer -
                             self.start_layer)
        moe_start_layer = self.start_layer + num_normal_layers
        for i in range(self.start_layer, min(moe_start_layer, self.end_layer)):
            layer = self.layers[i]
            hidden_states, residual = layer(
                positions, hidden_states, residual,
                kv_caches[i -
                          self.start_layer] if kv_caches is not None else None,
                attn_metadata,
                replace_allreduce=replace_allreduce)

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
        if attn_metadata is None or attn_metadata.num_prefills == 0 or not attn_metadata.enable_dbo_across_dp:
            return False
        return True

    def _forward_ms_layers(self,
                           positions: torch.Tensor,
                           hidden_states: torch.Tensor,
                           residual: torch.Tensor,
                           moe_start_layer: int,
                           kv_caches: Optional[List[torch.Tensor]] = None,
                           is_prefill: bool = False):

        if moe_start_layer == self.end_layer:
            return hidden_states, residual

        fused_moe_state = get_forward_context().fused_moe_state
        attn_metadata, [positions, hidden_states,
                        residual] = self.ms_pre_layer(
                            [positions, hidden_states, residual], )
        # the rest layers
        for i in range(moe_start_layer, self.end_layer):
            layer = self.layers[i]
            ms_layer_forward_func = layer._forward_ms_layer
            # if fused_moe_state == FusedMoEState.All2AllSeq:
            #     ms_layer_forward_func = layer._forward_ms_layer_alltoallv_finegrained
            hidden_states, residual = ms_layer_forward_func(
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


class CustomDeepseekDBOForCausalLM(CustomDeepseekV2ForCausalLM):
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
                                          quant_config=quant_config,
                                          prefix=maybe_prefix(
                                              prefix, "lm_head"))
        else:
            self.lm_head = PPMissingLayer()
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.sampler = get_sampler()
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: Optional[List[torch.Tensor]] = None,
        attn_metadata: Optional[AttentionMetadata] = None,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        graph_enable: Optional[bool] = True
    ) -> Union[torch.Tensor, IntermediateTensors]:
        hidden_states = self.model(input_ids, positions, kv_caches,
                                   attn_metadata, intermediate_tensors,
                                   inputs_embeds, graph_enable)
        return hidden_states