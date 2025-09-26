# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

# Adapted from
# https://github.com/huggingface/transformers/blob/v4.28.0/src/transformers/models/llama/modeling_llama.py
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
"""Inference-only DeepseekV3 model."""
import os
from typing import Dict, Optional, Tuple
import torch, torch_npu
from torch import nn
from transformers import PretrainedConfig
import torchair as tng
torch._logging.set_logs(recompiles=True)
# vllm adaptor
from vllm.platforms import current_platform
from vllm.config import QuantizationConfig
from vllm.attention import AttentionMetadata
from vllm.distributed import (
    get_ep_group,
    get_pp_group,
    get_dp_group,
    get_world_group,
)
from vllm.model_executor.layers.linear import (
    ReplicatedLinear,
)
from omni.layers.linear import (
    MergedReplicatedLinear,
)
from omni.layers.activation import SiluAndMul
from omni.layers.moe.fused_moe.layer import FusedMoE, UNQUANT_MODE, DYNAMIC_QUANT_MODE
from omni.adaptors.vllm.distributed.communication_op import (
    all_gather_two_stage,
    reduce_scatter_two_stage,
    prefill_reduce_scatter_pipeline,
    all_gather_local, reduce_scatter_local,
    all_gather_cross
)
from omni.adaptors.vllm.distributed.parallel_state import (
    get_round_cross_group_from_list
)
from omni.layers.moe.fused_moe.layer import FusedMoE
from omni.models.common.config.model_config import model_extra_config
from omni.layers.moe.fused_moe.fused_moe import fused_experts_moe_dispatch_combine
from omni.adaptors.vllm.utils import get_attr_by_names

if model_extra_config.operator_opt_config.use_omni_placement:
    from omni.accelerators.placement.omni_placement.omni_planner import OmniPlanner

"""NPU Stream Switch Names"""
STREAM_SHARED_EXPERT = 'stream_shared_expert'
SEQ_SPLIT_LENGTH = 4096


class ReplicatedDeepseekMLP(nn.Module):
    """Replicates the inputs and weights across multiple GPUs. No memory saving."""
    def __init__(
            self,
            hidden_size: int,
            intermediate_size: int,
            hidden_act: str,
            quant_config: Optional[QuantizationConfig] = None,
            reduce_results: bool = True,
            prefix: str = "",
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedReplicatedLinear(
            hidden_size, [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj")
        self.gate_up_proj.throw_dequant = True
        self.down_proj = ReplicatedLinear(intermediate_size,
                                          hidden_size,
                                          bias=False,
                                          quant_config=quant_config,
                                          prefix=f"{prefix}.down_proj")
        if hidden_act != "silu":
            raise ValueError(f"Unsupported activation: {hidden_act}. "
                             "Only silu is supported for now.")
        self.act_fn_obj = SiluAndMul()
        self.quant_symbol = True if quant_config else False
        self.tp_size = 1
        self.quant_mode = DYNAMIC_QUANT_MODE if quant_config else UNQUANT_MODE
        if model_extra_config.parall_config.redundancy_shared_expert_num > 0 and model_extra_config.operator_opt_config.decode_moe_dispatch_combine:
            # Adapt the dispatch combine operator
            self.ep_size = get_ep_group().world_size
            self.global_rank = get_world_group().rank_in_group
            self.world_size = get_world_group().world_size
            # self.n_shared_experts = n_shared_experts

            self.moe_all_to_all_group = get_world_group().device_group
            self.moe_all_to_all_group_name = self.moe_all_to_all_group._get_backend(torch.device(current_platform.device_type)).get_hccl_comm_name(
                self.global_rank)
            self.moe_rs_group = get_pp_group().device_group
            self.moe_rs_group_rank = get_pp_group().rank_in_group
            self.moe_rs_group_name = self.moe_rs_group._get_backend(torch.device(current_platform.device_type)).get_hccl_comm_name(
                                                 self.moe_rs_group_rank)


    def act_fn(self, x, quant_symbol):
        if quant_symbol and isinstance(x, tuple):
            x = dict(zip(['x_int8', 'pertoken_scale'], x))
            x['out_scale'] = self.gate_up_proj.weight_scale
        return self.act_fn_obj(x, quant_symbol)

    def forward(self, x):
        if isinstance(x, Dict):
            token_num = x.get('x_int8').shape[0]
        else:
            token_num = x.shape[0]
        if token_num > SEQ_SPLIT_LENGTH:  # Split seq to reduce memory usage
            x_list = x.split(SEQ_SPLIT_LENGTH)
            out = []
            for i in range(len(x_list)):
                x = x_list[i]
                gate_up, _ = self.gate_up_proj.forward(x)
                x = self.act_fn(gate_up, self.quant_symbol)
                x, _ = self.down_proj.forward(x)
                out.append(x)
            return torch.concat(out)
        gate_up, _ = self.gate_up_proj.forward(x)
        x = self.act_fn(gate_up, self.quant_symbol)
        x, _ = self.down_proj.forward(x)
        return x

def DynamicPruningUnsorted(topk_weights: torch.Tensor, 
                           topk_ids: torch.Tensor, 
                           thresholds: torch.Tensor):
    invalid_id = 512
    topk_weight_sorted, sorted_indices = torch.sort(topk_weights, dim=1, descending=True)
    topk_id_sorted = torch.gather(topk_ids, dim=1, index=sorted_indices)

    topk_weight_sum = torch.sum(topk_weight_sorted, dim=1, keepdim=True)
    mask = (topk_weight_sorted < topk_weight_sum * thresholds).to(topk_ids.dtype)
    new_topk_weights = topk_weight_sorted * (1-mask)
    new_topk_ids = topk_id_sorted * (1-mask) + invalid_id*  mask
    return new_topk_weights, new_topk_ids

class DeepseekMoE(nn.Module):

    def __init__(
            self,
            config: PretrainedConfig,
            quant_config: Optional[QuantizationConfig] = None,
            prefix: str = "",
    ):
        super().__init__()
        self.prefix = prefix
        self.ep_size = get_ep_group().world_size
        self.routed_scaling_factor = config.routed_scaling_factor
        self.device_count = torch.npu.device_count()
        self.node_rank = get_world_group().rank_in_group // self.device_count
        self.which_half = get_world_group().rank_in_group // (get_world_group().world_size // 2)

        n_routed_experts_names = ['num_routed_experts', 'n_routed_experts']
        self.n_routed_experts = get_attr_by_names(config, n_routed_experts_names, 256)
        self.redundancy_shared_expert_num = model_extra_config.parall_config.redundancy_shared_expert_num
        self.quant_symbol = quant_config is not None
        self.is_init_gate = False
        if os.getenv("ASCEND_PLATFORM", "A3")=="A2":
            self.is_A2 = True
            params_dtype = torch.bfloat16
        else:
            self.is_A2 = False
            params_dtype = torch.float32
        if self.ep_size > (self.n_routed_experts + self.redundancy_shared_expert_num):
            raise ValueError(
                f"Tensor parallel size {self.ep_size} is greater than "
                f"the number of experts {self.n_routed_experts} + {self.redundancy_shared_expert_num}.")

        if config.hidden_act != "silu":
            raise ValueError(f"Unsupported activation: {config.hidden_act}. "
                             "Only silu is supported for now.")

        self.gate = ReplicatedLinear(config.hidden_size,
                                     self.n_routed_experts,
                                     bias=False,
                                     quant_config=None,
                                     params_dtype=params_dtype,
                                     prefix=f"{prefix}.gate")
        if getattr(config, "topk_method", "topk") == "noaux_tc":
            self.gate.e_score_correction_bias = nn.Parameter(
                torch.empty(self.n_routed_experts, dtype=torch.float), requires_grad=False)
        else:
            self.gate.e_score_correction_bias = None

        self.top_k = config.num_experts_per_tok
        self.use_grouped_topk = True
        self.renormalize = getattr(config, "norm_topk_prob", True)
        self.topk_group = getattr(config, "topk_group", 1)
        self.num_expert_group = getattr(config, "n_group", 1)
        self.custom_routing_function = None
        self.scoring_func = getattr(config, "scoring_func", "sigmoid")
        n_shared_experts_names = ['num_shared_experts', 'n_shared_experts']
        first_k_dense_replace_names = ['num_dense_layers', 'first_k_dense_replace']
        self.n_shared_experts = get_attr_by_names(config, n_shared_experts_names, 1)
        self.first_k_dense_replace = get_attr_by_names(config, first_k_dense_replace_names, 3)
        
        
        self.shared_experts = None
        self.experts = None
        self.global_rank = get_world_group().rank_in_group
        self.planner = None
        self.moe_layer_idx = None
        self.expert_mapping = None
        self.attn_prefetch = None

        if self.global_rank >= self.redundancy_shared_expert_num:
            moe_prefix = f"{prefix}.experts"
            # omni placement for redundancy route experts
            if model_extra_config.operator_opt_config.use_omni_placement:
                self.planner = OmniPlanner(config_file= model_extra_config.operator_opt_config.omni_placement_config_path, device="npu",
                                           rank=get_world_group().rank_in_group,
                                           world_size=get_world_group().world_size,
                                           num_experts=self.n_routed_experts,
                                           num_redundancy_shared_expert_rank=self.redundancy_shared_expert_num)
                self.moe_layer_idx = OmniPlanner.get_deepseek_v3_moe_layer_idx(moe_prefix, first_k_dense_replace=self.first_k_dense_replace)
                self.expert_mapping = self.planner.expert_mapping_on_current_layer(self.moe_layer_idx)
            self.experts = FusedMoE(
                num_experts=self.n_routed_experts,
                top_k=self.top_k,
                hidden_size=config.hidden_size,
                intermediate_size=config.moe_intermediate_size,
                reduce_results=False,
                renormalize=self.renormalize,
                quant_config=quant_config,
                use_grouped_topk=self.use_grouped_topk,
                num_expert_group=self.num_expert_group,
                topk_group=self.topk_group,
                prefix=moe_prefix,
                scoring_func=self.scoring_func,
                e_score_correction_bias=self.gate.e_score_correction_bias,
                planner=self.planner,
                moe_layer_idx=self.moe_layer_idx,
                expert_mapping=self.expert_mapping,
				first_k_dense_replace=self.first_k_dense_replace
            )
        if self.n_shared_experts is not None and \
            (self.redundancy_shared_expert_num == 0 or self.global_rank < self.redundancy_shared_expert_num):
            intermediate_size = config.moe_intermediate_size * self.n_shared_experts
            # omni placement for redundancy shared experts
            if self.redundancy_shared_expert_num > 0 and model_extra_config.operator_opt_config.use_omni_placement:
                # The order that first initializing OmniPlanner, then ReplicatedDeepseekMLP, should correspond to the router expert rank initialization order in the layer.py file.
                self.planner = OmniPlanner(config_file=model_extra_config.operator_opt_config.omni_placement_config_path, device="npu",
                                           rank=self.global_rank, world_size=self.ep_size,
                                           num_experts=self.n_routed_experts,
                                           num_redundancy_shared_expert_rank=self.redundancy_shared_expert_num)
                self.moe_layer_idx = OmniPlanner.get_deepseek_v3_moe_layer_idx(f"{prefix}.share_experts", first_k_dense_replace=self.first_k_dense_replace)
                self.expert_mapping = self.planner.expert_mapping_on_current_layer(self.moe_layer_idx, is_prefill=False)

            self.shared_experts = ReplicatedDeepseekMLP(
                hidden_size=config.hidden_size,
                intermediate_size=intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                reduce_results=False,
                prefix=f"{prefix}.shared_experts",
            )

        if self.experts is not None:
            self.w13_prefetch_size = model_extra_config.operator_opt_config.expert_gate_up_prefetch * 1024 * 1024
            self.w2_prefetch_size = 0
            self.local_expert_num = self.experts.w13_weight.shape[0]
            if self.quant_symbol:
                self.in_scale_2 = torch.ones((self.local_expert_num, config.moe_intermediate_size), dtype=torch.float32, device=current_platform.device_type)
                # call the mark_static to reduce memory usage
                torch._dynamo.mark_static(self.in_scale_2)
                if self.ep_size > 64:
                    self.w2_prefetch_size = model_extra_config.operator_opt_config.expert_down_prefetch * 1024 * 1024

        self.tuning_config = None
        if not model_extra_config.operator_opt_config.gmm_nz:
            self.tuning_config = model_extra_config.operator_opt_config.decode_gear_list[:1]
        elif model_extra_config.operator_opt_config.decode_gear_list[0] >= 32:
            self.tuning_config = [256]
        
        self.experts_pruning = (model_extra_config.operator_opt_config.experts_pruning and 
                                model_extra_config.operator_opt_config.prefill_moe_all_to_all)
        if self.experts_pruning:
            self.experts_pruning_threshold = torch.tensor(
                    [0, 0.01, 0.01, 0.01, 0.0665, 0.086, 0.125, 0.135])

    def forward(self, hidden_states: torch.Tensor, residual: torch.Tensor, attn_metadata: AttentionMetadata, layer_id: int, next_attention_weights: Optional[dict]=None) -> torch.Tensor:
        if self.redundancy_shared_expert_num > 0:
            if attn_metadata is None or attn_metadata.prefill is not None:
                return self.forward_separate_expert_prefill(hidden_states, residual, attn_metadata)
            else:
                return self.forward_separate_expert_decode(hidden_states, residual, attn_metadata)
        else:
            if not self.is_init_gate:
                self.gate.weight.data = torch_npu.npu_format_cast(self.gate.weight.data, 2)
                self.is_init_gate = True
            if attn_metadata is None or attn_metadata.prefill is not None:
                if self.is_A2 and not model_extra_config.operator_opt_config.prefill_moe_all_to_all:
                    return self.forward_prefill_a2(hidden_states, residual, attn_metadata)
                else:
                    return self._forward_prefill_norm(hidden_states, residual, attn_metadata)
            elif self.is_A2:
                return self.forward_decode_a2(hidden_states, residual, attn_metadata, layer_id, next_attention_weights)
            else:
                return self._forward_decode_norm(hidden_states, residual, attn_metadata, layer_id, next_attention_weights)

    def _forward_prefill_norm(self, hidden_states: torch.Tensor, residual: torch.Tensor, attn_metadata: AttentionMetadata) -> torch.Tensor:
        shared_output = self.shared_experts(hidden_states)

        if not model_extra_config.operator_opt_config.prefill_moe_all_to_all:
            hidden_states_int8, pertoken_scale = torch_npu.npu_dynamic_quant(hidden_states)
            global_hidden_states = all_gather_two_stage(hidden_states_int8, idx=0, dim=0)
        else:
            global_hidden_states = hidden_states
            global_pertoken_scale = None

        router_logits, _ = self.gate.forward(hidden_states.float())
        topk_weights, topk_ids, _ = FusedMoE.select_experts(hidden_states, router_logits,
                                                                    self.experts.top_k, self.experts.use_grouped_topk, self.experts.renormalize,
                                                                    self.experts.topk_group, self.experts.num_expert_group, self.experts.custom_routing_function,
                                                                    self.experts.scoring_func, self.experts.e_score_correction_bias, self.routed_scaling_factor,
                                                                    layer=self.experts  # ENABLE_OMNI_PLANNER
                                                                    )
        topk_ids = self.experts.apply_expert_load_balance(topk_ids=topk_ids)

        if self.experts_pruning:
            topk_weights, topk_ids = DynamicPruningUnsorted(topk_weights, 
                                                            topk_ids, 
                                                            self.experts_pruning_threshold)

        if not model_extra_config.operator_opt_config.prefill_moe_all_to_all:
            topk_cat = torch.cat((topk_weights, topk_ids.to(torch.float), pertoken_scale.unsqueeze(-1)), dim=-1)
            topk_all = all_gather_two_stage(topk_cat, idx=0, dim=0)
            topk_weights, topk_ids, global_pertoken_scale = torch.split(
                topk_all, [topk_weights.shape[-1], topk_ids.shape[-1], 1], dim=-1)
            topk_ids = torch.round(topk_ids).to(torch.int32)
            global_pertoken_scale = global_pertoken_scale.squeeze(-1)

        final_hidden_states_list = self.experts(
            hidden_states=global_hidden_states,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            pertoken_scale=global_pertoken_scale,
            attn_metadata=attn_metadata
        )

        if model_extra_config.operator_opt_config.prefill_moe_all_to_all:
            if len(final_hidden_states_list) != 4:
                raise RuntimeError("len(final_hidden_states_list) != 4")
            final_hidden_states = final_hidden_states_list[0]
            gathered_tokens = final_hidden_states_list[1]
            expanded_row_idx = final_hidden_states_list[3]
        else:
            final_hidden_states = final_hidden_states_list

        if not model_extra_config.operator_opt_config.prefill_moe_all_to_all:
            final_hidden_states = reduce_scatter_two_stage(final_hidden_states, idx=0)

        if model_extra_config.operator_opt_config.prefill_moe_all_to_all:
            final_hidden_states = torch_npu.npu_moe_finalize_routing(
                gathered_tokens,
                skip1=shared_output,
                skip2=None,
                bias=None,
                scales=topk_weights.to(gathered_tokens.dtype),
                expanded_src_to_dst_row=expanded_row_idx,
                export_for_source_row=None,
                drop_pad_mode=2
            )
        else:
            final_hidden_states = final_hidden_states + shared_output

        return final_hidden_states, residual

    def _forward_decode_norm(self, hidden_states: torch.Tensor, residual: torch.Tensor, attn_metadata: AttentionMetadata, layer_id: int, next_attention_weights: Optional[dict]=None) -> torch.Tensor:
        if model_extra_config.operator_opt_config.moe_multi_stream_tune and \
            model_extra_config.operator_opt_config.decode_moe_dispatch_combine:
            if model_extra_config.operator_opt_config.use_super_kernel:
                with tng.scope.super_kernel(self.prefix, 'stream-fusion=1'):
                    return self._forward_decode_dispatch_combine(hidden_states, residual, attn_metadata, layer_id, next_attention_weights)
            else:
                return self._forward_decode_dispatch_combine(hidden_states, residual, attn_metadata, layer_id, next_attention_weights)
        if model_extra_config.operator_opt_config.moe_multi_stream_tune:
            with tng.scope.npu_stream_switch('21'):
                hidden_states = tng.scope.npu_wait_tensor(hidden_states, hidden_states)
                shared_output = self.shared_experts(hidden_states)
        else:
            shared_output = self.shared_experts(hidden_states)

        if not model_extra_config.operator_opt_config.decode_moe_dispatch_combine:
            hidden_states_int8, pertoken_scale = torch_npu.npu_dynamic_quant(hidden_states)
            global_hidden_states = get_world_group().all_gather(hidden_states_int8, dim=0)
        else:
            global_hidden_states = hidden_states
            global_pertoken_scale = None

        router_logits, _ = self.gate.forward(hidden_states.float())
        # Here, we do a 2d-3d conversion and then convert back to 2d to trigger the fusion rule, fusing add rms and cast into AddRmsNormCast.
        hidden_states_3d = hidden_states.unsqueeze(1)
        hidden_states = hidden_states_3d.squeeze(1)
        topk_weights, topk_ids, _ = FusedMoE.select_experts(hidden_states, router_logits,
                                                            self.experts.top_k, self.experts.use_grouped_topk,
                                                            self.experts.renormalize,
                                                            self.experts.topk_group, self.experts.num_expert_group,
                                                            self.experts.custom_routing_function,
                                                            self.experts.scoring_func,
                                                            self.experts.e_score_correction_bias,
                                                            self.routed_scaling_factor,
                                                            layer=self.experts  # ENABLE_OMNI_PLANNER
                                                            )
        topk_ids = self.experts.apply_expert_load_balance(topk_ids=topk_ids, best_topk_ids=attn_metadata.decode.best_topk)
        if not model_extra_config.operator_opt_config.decode_moe_dispatch_combine:
            topk_cat = torch.cat((topk_weights, topk_ids.to(torch.float), pertoken_scale.unsqueeze(-1)), dim=-1)
            topk_all = get_world_group().all_gather(topk_cat, dim=0)

            topk_all = topk_all.view(-1, self.device_count, topk_weights.shape[0], topk_all.shape[-1]) \
                                .transpose(0, 1) \
                                .reshape(-1, topk_all.shape[-1])
            topk_weights, topk_ids, global_pertoken_scale = torch.split(topk_all, [topk_weights.shape[-1], topk_ids.shape[-1], 1], dim=-1)
            topk_ids = torch.round(topk_ids).to(torch.int32)
            global_pertoken_scale = global_pertoken_scale.squeeze(-1)

        final_hidden_states = self.experts(
            hidden_states=global_hidden_states,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            pertoken_scale=global_pertoken_scale,
            attn_metadata=attn_metadata
        )

        if not model_extra_config.operator_opt_config.decode_moe_dispatch_combine:
            final_hidden_states = get_world_group().reduce_scatter(final_hidden_states)

        final_hidden_states = final_hidden_states + shared_output

        return final_hidden_states, residual

    def _forward_decode_dispatch_combine(self, hidden_states: torch.Tensor, residual: torch.Tensor, attn_metadata: AttentionMetadata, layer_id: int, next_attention_weights: Optional[dict]=None) -> torch.Tensor:
        is_prefill = (attn_metadata is None or attn_metadata.prefill is not None)
        router_logits, _ = self.gate.forward(hidden_states.float())
        # Here, we do a 2D to 3D conversion, and then convert back to 2D to trigger the fusion rule, fusing add rms and cast into AddRmsNormCast.
        hidden_states_3d = hidden_states.unsqueeze(1)
        hidden_states = hidden_states_3d.squeeze(1)

        with tng.scope.npu_stream_switch(STREAM_SHARED_EXPERT):
            hidden_states = tng.scope.npu_wait_tensor(hidden_states, router_logits)
            # shared_experts w13
            gate_up_share, _ = self.shared_experts.gate_up_proj.forward(hidden_states)
        wait_gate = gate_up_share if isinstance(gate_up_share, torch.Tensor) else gate_up_share[0]
        
        # expert weight prefetch
        if self.w13_prefetch_size > 0:
            torch_npu.npu_prefetch(self.experts.w13_weight, wait_gate, self.w13_prefetch_size)
        if self.w2_prefetch_size > 0:
            torch_npu.npu_prefetch(self.experts.w2_weight, wait_gate, self.w2_prefetch_size)

        topk_weights, topk_ids, _ = FusedMoE.select_experts(hidden_states, router_logits,
                                                                self.experts.top_k, self.experts.use_grouped_topk,
                                                                self.experts.renormalize,
                                                                self.experts.topk_group, self.experts.num_expert_group,
                                                                self.experts.custom_routing_function,
                                                                self.experts.scoring_func,
                                                                self.experts.e_score_correction_bias,
                                                                self.routed_scaling_factor,
                                                                layer=self.experts  # ENABLE_OMNI_PLANNER
                                                                )
        topk_ids = self.experts.apply_expert_load_balance(topk_ids=topk_ids, best_topk_ids=attn_metadata.decode.best_topk)

        mc2_mask = attn_metadata.decode.mc2_mask if attn_metadata is not None and attn_metadata.decode is not None else None
        layer = self.experts
        
        max_num_deployed_expert = self.local_expert_num * get_dp_group().world_size
        act_dtype = hidden_states.dtype
        shared_expert_rank_num = 0
        kwargs = {
            "x": hidden_states,
            "expert_ids": topk_ids,  # [n*topk]
            "expert_shard_type": 0,  # Set it to 0 for now
            "shared_expert_rank_num": shared_expert_rank_num,  # 32
            "moe_expert_num": max_num_deployed_expert, #ENABLE_OMNI_PLANNER, 0 redundancy 256, 1 redundancy expert 320
            "global_bs": 0,  # 0 Default (all); all tokens can be set
        }

        experts_tp_size = layer.tp_size
        world_size = get_world_group().world_size
        # In fact, what we get is the die number, and the ep group is not adapted by default.
        # The default ep group is experts_num/die_num.
        global_rank = get_world_group().rank_in_group
        all_to_all_group_size = world_size // experts_tp_size

        kwargs.update({
            "scales": None,  # Quantization coefficient
            "quant_mode": layer.quant_mode,  # 0: Non-quantization; 1: Static quantization; 2: Dynamic quantization
            "group_ep": layer.moe_all_to_all_group_name,  # Unlike torch, it is obtained by name.
            "ep_world_size": all_to_all_group_size,
            "ep_rank_id": global_rank // experts_tp_size,
            "group_tp": layer.moe_rs_group_name,
            "tp_world_size": experts_tp_size,
            "tp_rank_id": global_rank % experts_tp_size,
            "x_active_mask": mc2_mask,
        })

        output = torch_npu.npu_moe_distribute_dispatch_v2(**kwargs)
        expand_x, dynamic_scale, expand_idx, expert_token_nums, ep_recv_counts = output[0:5]

        group_list = expert_token_nums.to(torch.int64)
        if model_extra_config.operator_opt_config.use_omni_placement:
            layer.planner.record_activation(layer.moe_layer_idx, group_list, support_multi_stream=model_extra_config.operator_opt_config.moe_multi_stream_tune and (not is_prefill))

        # cal experts
        weight1_3 = self.experts.w13_weight
        weight2 = self.experts.w2_weight
        if self.quant_symbol:
            if self.experts.weight_num_bits == 8:
                weight_scale1_3 = self.experts.w13_weight_scale
                weight_scale2 = self.experts.w2_weight_scale
            elif self.experts.weight_num_bits == 4:
                weight_scale1_3 = self.experts.w13_weight_int4_scale
                weight_scale2 = self.experts.w2_weight_int4_scale
                weight_bias1_3 = self.experts.w13_weight_bias
                weight_bias2 = self.experts.w2_weight_bias
            else:
                raise NotImplementedError(f"Unsupported compress tensor type. num bits: {self.experts.weight_num_bits}")

            if self.experts.quant_mode:  # 0: no quant 1: static quant 2: dynamic quant
                pertoken_scale = dynamic_scale
            else:
                expand_x, pertoken_scale = torch_npu.npu_dynamic_quant(expand_x)

        with tng.scope.npu_stream_switch(STREAM_SHARED_EXPERT):
            wait_gate = gate_up_share if isinstance(gate_up_share, torch.Tensor) else gate_up_share[0]
            wait_gate = tng.scope.npu_wait_tensor(wait_gate, expand_x)
            if not isinstance(gate_up_share, torch.Tensor):
                gate_up_share = (wait_gate, gate_up_share[1])
            intermediate_hiddenstates_share = self.shared_experts.act_fn(gate_up_share, self.shared_experts.quant_symbol)
        if self.quant_symbol:
            # w8a8
            if self.experts.weight_num_bits == 8:
                gate_up_proj = torch_npu.npu_grouped_matmul([expand_x], [weight1_3], bias=None, group_list=group_list,
                                                            split_item=3, output_dtype=torch.int32, group_type=0,
                                                            group_list_type=1)[0]

                gate_up_proj, pertoken_scale = torch_npu.npu_dequant_swiglu_quant(
                    gate_up_proj, weight_scale=weight_scale1_3, activation_scale=pertoken_scale, bias=None, quant_scale=self.in_scale_2, quant_offset=None,
                    group_index=group_list, activate_left=True, quant_mode=1)

                hidden_states_experts = torch_npu.npu_grouped_matmul([gate_up_proj], [weight2], scale=[weight_scale2],
                                                per_token_scale=[pertoken_scale],bias=None,
                                                group_list=group_list, split_item=3, output_dtype=act_dtype,
                                                group_type=0,
                                                group_list_type=1)[0]
            elif self.experts.weight_num_bits == 4:
                gate_up_proj = \
                    torch_npu.npu_grouped_matmul([expand_x], [weight1_3], bias=[weight_bias1_3], scale=[weight_scale1_3],
                                                 offset=None, antiquant_scale=None, antiquant_offset=None,
                                                 per_token_scale=[pertoken_scale],
                                                 group_list=group_list,
                                                 activation_input=None, activation_quant_scale=None,
                                                 activation_quant_offset=None, split_item=3, group_type=0,
                                                 group_list_type=1, act_type=0,
                                                 tuning_config=self.tuning_config, output_dtype=torch.bfloat16)[0]

                fake_scale = torch.ones(weight_bias1_3.shape, dtype=torch.float32, device="npu").view(-1,weight_bias1_3.shape[1])
                pertoken_scale = torch.ones(pertoken_scale.shape, dtype=torch.float32, device="npu")
                gate_up_proj, pertoken_scale = torch_npu.npu_dequant_swiglu_quant(gate_up_proj,
                                                                                  weight_scale=fake_scale,
                                                                                  activation_scale=pertoken_scale,
                                                                                  bias=None, quant_scale=None,
                                                                                  quant_offset=None,
                                                                                  group_index=group_list,
                                                                                  activate_left=True,
                                                                                  quant_mode=1)

                hidden_states_experts = torch_npu.npu_grouped_matmul([gate_up_proj], [weight2], scale=[weight_scale2],
                                                                     per_token_scale=[pertoken_scale],
                                                                     bias=[weight_bias2],
                                                                     group_list=group_list, split_item=3,
                                                                     output_dtype=act_dtype,
                                                                     group_type=0,
                                                                     group_list_type=1,
                                                                     tuning_config=self.tuning_config)[0]
            else:
                raise NotImplementedError(f"Unsupported compress tensor type. num bits: {self.experts.weight_num_bits}")
        else:
            # bf16
            gate_up_proj = torch_npu.npu_grouped_matmul([expand_x], [weight1_3], bias=None, group_list=group_list,
                                                    split_item=3, group_type=0, group_list_type=1)[0]
        
            gate_up_proj = torch_npu.npu_swiglu(gate_up_proj)

            hidden_states_experts = torch_npu.npu_grouped_matmul([gate_up_proj], [weight2],bias=None,
                                            group_list=group_list, split_item=3, output_dtype=act_dtype,
                                            group_type=0, group_list_type=1)[0]

        # moeCombine
        kwargs = {
            "expand_x": hidden_states_experts,
            "expert_ids": topk_ids,  # [n*topk]
            "assist_info_for_combine": expand_idx,
            "expert_scales": topk_weights.to(torch.float32),  # weight [n*topk]
            "expert_shard_type": 0,
            "shared_expert_rank_num": shared_expert_rank_num,
            "moe_expert_num":  max_num_deployed_expert, #ENABLE_OMNI_PLANNER, 0 redundancy 256, 1 redundancy expert 320
            "global_bs": 0,  # 0 Default (all); all tokens can be set
        }
        tp_recv_counts = output[5]
        stage3_kwargs = {
            "ep_send_counts": ep_recv_counts,  # dispatch's send_counts
            "group_ep": layer.moe_all_to_all_group_name,  # Unlike torch, it is obtained by name.
            "ep_world_size": all_to_all_group_size,
            "ep_rank_id": global_rank // experts_tp_size,
            "tp_send_counts": tp_recv_counts,
            "group_tp": layer.moe_rs_group_name,
            "tp_world_size": experts_tp_size,
            "tp_rank_id": global_rank % experts_tp_size,
            "x_active_mask": mc2_mask,
        }
        kwargs.update(stage3_kwargs)

        with tng.scope.npu_stream_switch(STREAM_SHARED_EXPERT):
            if isinstance(intermediate_hiddenstates_share, dict):
                intermediate_hiddenstates_share['x_int8'] = tng.scope.npu_wait_tensor(intermediate_hiddenstates_share.get('x_int8'), hidden_states_experts)
            else:
                intermediate_hiddenstates_share = tng.scope.npu_wait_tensor(intermediate_hiddenstates_share, hidden_states_experts)
            shared_output, _ = self.shared_experts.down_proj.forward(intermediate_hiddenstates_share)

        # prefetch weights for attention next layer
        if model_extra_config.operator_opt_config.attn_prefetch > 0 and next_attention_weights is not None and next_attention_weights['q_a_proj_weight'] is not None:
                attn_prefetch_size = model_extra_config.operator_opt_config.attn_prefetch * 1024 * 1024
                attn_prefetch_flag = shared_output
                torch_npu.npu_prefetch(next_attention_weights['q_a_proj_weight'], attn_prefetch_flag, attn_prefetch_size)
                if self.quant_symbol:
                    torch_npu.npu_prefetch(next_attention_weights['kv_a_proj_with_mqa_weight'], attn_prefetch_flag, attn_prefetch_size)
                torch_npu.npu_prefetch(next_attention_weights['q_b_proj_weight'], attn_prefetch_flag, attn_prefetch_size)
                torch_npu.npu_prefetch(next_attention_weights['W_UK'], attn_prefetch_flag, attn_prefetch_size)

        hidden_states_route = torch_npu.npu_moe_distribute_combine_v2(**kwargs)

        if shared_output is not None:
            final_hidden_states = (hidden_states_route, shared_output)

        return final_hidden_states, residual

    def forward_separate_expert_decode(self,
                                       hidden_states: torch.Tensor,
                                       residual: torch.Tensor,
                                       attn_metadata: AttentionMetadata) -> torch.Tensor:
        router_logits, _ = self.gate.forward(hidden_states.float())
        
        # Here, we do a 2D to 3D conversion, and then convert back to 2D to trigger the fusion rule, fusing add rms and cast into AddRmsNormCast.
        hidden_states_3d = hidden_states.unsqueeze(1)
        hidden_states = hidden_states_3d.squeeze(1)

        topk_weights, topk_ids, _ = FusedMoE.select_experts(hidden_states, router_logits,
                                                            self.top_k, self.use_grouped_topk,
                                                            self.renormalize,
                                                            self.topk_group, self.num_expert_group,
                                                            self.custom_routing_function,
                                                            self.scoring_func,
                                                            self.gate.e_score_correction_bias,
                                                            self.routed_scaling_factor,
                                                            layer=self.experts)
        max_num_deployed_expert=self.n_routed_experts
        if model_extra_config.operator_opt_config.use_omni_placement:
            if self.planner.is_moe_layer(self.moe_layer_idx):
                hidden_states, topk_ids, topk_weights = self.planner.plan(layer_idx_moe=self.moe_layer_idx,
                                                                          tokens=hidden_states,
                                                                          token_expert_ids=topk_ids,
                                                                          token_expert_scores=topk_weights,
                                                                          top_k=self.top_k,
                                                                          expert_mapping=self.expert_mapping,
                                                                          is_prefill=False)
                max_num_deployed_expert_per_rank = self.planner.get_max_num_deployed_expert_per_rank()
                max_num_deployed_expert = max_num_deployed_expert_per_rank * (self.ep_size - self.redundancy_shared_expert_num)
        if model_extra_config.operator_opt_config.best_ep and attn_metadata.decode.best_topk is not None:
            fake_topk_ids = attn_metadata.decode.best_topk
            topk_ids = tng.scope.npu_wait_tensor(fake_topk_ids, topk_ids)
        hidden_states = fused_experts_moe_dispatch_combine(self.shared_experts or self.experts,
                                                                hidden_states,
                                                                topk_weights,
                                                                topk_ids,
                                                                max_num_deployed_expert=max_num_deployed_expert,
                                                                is_prefill=False,
                                                                is_route_expert=self.experts is not None)
        return hidden_states, residual

    def forward_separate_expert_prefill(self, hidden_states: torch.Tensor, residual: torch.Tensor, attn_metadata: AttentionMetadata) -> torch.Tensor:
        global_hidden_states = get_ep_group().all_gather(hidden_states, dim=0)
        if self.shared_experts:
            avg_tokens_per_shared_experts = global_hidden_states.shape[0] // self.redundancy_shared_expert_num
            shared_experts_mask = torch.zeros(global_hidden_states.shape[0], 1, dtype=torch.int32, device="npu")
            shared_experts_mask[self.global_rank * avg_tokens_per_shared_experts : (self.global_rank + 1) * avg_tokens_per_shared_experts] = 1
            shared_experts_hidden_states = global_hidden_states * shared_experts_mask
            shared_output = self.shared_experts(shared_experts_hidden_states)
        else:
            shared_output = torch.zeros_like(global_hidden_states)
        shared_output = get_ep_group().reduce_scatter(shared_output)

        if self.experts:
            router_logits, _ = self.gate.forward(global_hidden_states)
            topk_weights, topk_ids, _ = FusedMoE.select_experts(global_hidden_states, router_logits,
                                                                self.experts.top_k, self.experts.use_grouped_topk,
                                                                self.experts.renormalize,
                                                                self.experts.topk_group, self.experts.num_expert_group,
                                                                self.experts.custom_routing_function,
                                                                self.experts.scoring_func,
                                                                self.experts.e_score_correction_bias,
                                                                self.routed_scaling_factor,
                                                                layer=self.experts)
            global_hidden_states, global_pertoken_scale = torch_npu.npu_dynamic_quant(global_hidden_states)
            output = self.experts(
                hidden_states=global_hidden_states,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                pertoken_scale=global_pertoken_scale,
                attn_metadata=attn_metadata
            )
        else:
            output = torch.zeros_like(global_hidden_states)
        final_hidden_states = get_ep_group().reduce_scatter(output)
        if shared_output is not None:
            final_hidden_states = final_hidden_states + shared_output
        return final_hidden_states, residual

    def chunked_gmm(self, hidden_states: torch.Tensor, topk_weights: torch.Tensor, topk_ids: torch.Tensor,
                    pertoken_scale: torch.Tensor, attn_metadata: AttentionMetadata, chunk_size: int):

        if hidden_states.shape[0] > chunk_size:
            out = []
            hidden_states_list = torch.split(hidden_states, chunk_size)
            topk_weights_list = torch.split(topk_weights, chunk_size)
            topk_ids_list = torch.split(topk_ids, chunk_size)
            pertoken_scale_list = torch.split(pertoken_scale, chunk_size)
            for hid_states, topk_w, topk_id, scale in zip(hidden_states_list, topk_weights_list, topk_ids_list, pertoken_scale_list):
                out.append(self.experts(hidden_states=hid_states,
                                        topk_weights=topk_w,
                                        topk_ids=topk_id,
                                        pertoken_scale=scale,
                                        attn_metadata=attn_metadata))
            return torch.cat(out)

        return self.experts(hidden_states=hidden_states,
                            topk_weights=topk_weights,
                            topk_ids=topk_ids,
                            pertoken_scale=pertoken_scale,
                            attn_metadata=attn_metadata)
    
    def forward_decode_a2(self, hidden_states: torch.Tensor, residual: torch.Tensor,
                       attn_metadata: AttentionMetadata, layer_id: int, kv_prefetch: torch.Tensor = None) -> torch.Tensor:
        """stream name"""
        STREAM_TOPK_COMPUTE = 'topk_compute'
        STREAM_SHARED_EXPERT = 'shared_expert'
        STREAM_TOPK_COMM = 'topk_comm'
        STREAM_PREFETCH = 'prefetch'
        STREAM_INTERNODE_COMM_0 = 'internode_comm_0'
        STREAM_INTERNODE_COMM_1 = 'internode_comm_1'
        STREAM_INTERNODE_COMM_2 = 'internode_comm_2'
        MAX_PREFETCH_SIZE = 90000000
        LARGE_BATCH, MEDIUM_BATCH, SMALL_BATCH = False, False, False
        if hidden_states.shape[0] >= 120:
            LARGE_BATCH = True
        elif hidden_states.shape[0] >= 60:
            MEDIUM_BATCH = True
        else:
            SMALL_BATCH = True

        hidden_states_int8, pertoken_scale = torch_npu.npu_dynamic_quant(hidden_states)
        if model_extra_config.operator_opt_config.moe_multi_stream_tune:
            with tng.scope.npu_stream_switch(STREAM_TOPK_COMPUTE):
                router_logits, _ = self.gate.forward(hidden_states)
                topk_weights, topk_ids, _ = FusedMoE.select_experts(hidden_states, router_logits,
                                                                    self.experts.top_k, self.experts.use_grouped_topk,
                                                                    self.experts.renormalize,
                                                                    self.experts.topk_group, self.experts.num_expert_group,
                                                                    self.experts.custom_routing_function,
                                                                    self.experts.scoring_func,
                                                                    self.experts.e_score_correction_bias,
                                                                    self.routed_scaling_factor,
                                                                    layer=self.experts)
                topk_ids = self.experts.apply_expert_load_balance(
                    topk_ids=topk_ids, 
                    best_topk_ids=attn_metadata.decode.best_topk if attn_metadata is not None and attn_metadata.decode is not None else None
                )
                if attn_metadata is not None and attn_metadata.decode is not None:
                    actual_batch_mask = attn_metadata.decode.mc2_mask \
                                                            .to(torch.int32).view(-1, 1) \
                                                            .repeat(1, self.experts.top_k)
                    topk_ids = actual_batch_mask * topk_ids + (1 - actual_batch_mask) * self.n_routed_experts

                topk_cat = torch.cat((topk_weights, topk_ids.to(torch.float), pertoken_scale.unsqueeze(-1)), dim=-1)

        ###### PIPELINE ALL_GATHER STARTS
            if LARGE_BATCH:
                with tng.scope.npu_stream_switch(STREAM_SHARED_EXPERT):
                    hidden_states_dict = {"x_int8": hidden_states_int8, "pertoken_scale":pertoken_scale}
                    shared_output = self.shared_experts(hidden_states_dict)

                with tng.scope.npu_stream_switch(STREAM_PREFETCH):
                    torch_npu.npu_prefetch(self.experts.w13_weight, shared_output, MAX_PREFETCH_SIZE)
            
                with tng.scope.npu_stream_switch(STREAM_TOPK_COMM):
                    topk_local_all = all_gather_local(topk_cat, idx=1, dim=0)

                input_ag = all_gather_local(hidden_states_int8, idx=0, dim=0)
                with tng.scope.npu_stream_switch(STREAM_INTERNODE_COMM_0):
                    round0_swp = tng.scope.npu_wait_tensor(hidden_states_int8, hidden_states_int8)
                    round0_swp = get_round_cross_group_from_list(round=0).swap(round0_swp, method="all2allv")
                with tng.scope.npu_stream_switch(STREAM_INTERNODE_COMM_1):
                    round1_swp = tng.scope.npu_wait_tensor(hidden_states_int8, input_ag)
                    round1_swp = get_round_cross_group_from_list(round=1).swap(round1_swp, method="all2allv")
                with tng.scope.npu_stream_switch(STREAM_INTERNODE_COMM_2):
                    round2_swp = tng.scope.npu_wait_tensor(hidden_states_int8, round1_swp)
                    round2_swp = get_round_cross_group_from_list(round=2).swap(round2_swp, method="all2allv")

                with tng.scope.npu_stream_switch(STREAM_TOPK_COMM):
                    topk_local_all_wait = tng.scope.npu_wait_tensor(topk_local_all, round0_swp)
                    topk_all = all_gather_cross(topk_local_all_wait, idx=1, dim=0)

                round0_swp = tng.scope.npu_wait_tensor(round0_swp, input_ag)
                round0_ag = all_gather_local(round0_swp, idx=0, dim=0)
                round1_swp = tng.scope.npu_wait_tensor(round1_swp, round0_ag)
                round1_ag = all_gather_local(round1_swp, idx=0, dim=0)
                round2_swp = tng.scope.npu_wait_tensor(round2_swp, round1_ag)
                round2_ag = all_gather_local(round2_swp, idx=0, dim=0)

            elif MEDIUM_BATCH:
                with tng.scope.npu_stream_switch(STREAM_TOPK_COMM):
                    topk_local_all = all_gather_local(topk_cat, idx=1, dim=0)

                input_ag = all_gather_local(hidden_states_int8, idx=0, dim=0)
                with tng.scope.npu_stream_switch(STREAM_INTERNODE_COMM_0):
                    round0_swp = tng.scope.npu_wait_tensor(hidden_states_int8, hidden_states_int8)
                    round0_swp = get_round_cross_group_from_list(round=0).swap(round0_swp, method="all2allv")
                    round0_ag = all_gather_local(round0_swp, idx=-1, dim=0)
                with tng.scope.npu_stream_switch(STREAM_INTERNODE_COMM_1):
                    round1_swp = tng.scope.npu_wait_tensor(hidden_states_int8, topk_weights)
                    round1_swp = get_round_cross_group_from_list(round=1).swap(round1_swp, method="all2allv")
                    round1_ag = all_gather_local(round1_swp, idx=-2, dim=0)
                with tng.scope.npu_stream_switch(STREAM_INTERNODE_COMM_2):
                    round2_swp = tng.scope.npu_wait_tensor(hidden_states_int8, input_ag)
                    round2_swp = get_round_cross_group_from_list(round=2).swap(round2_swp, method="all2allv")
                    round2_ag = all_gather_local(round2_swp, idx=-3, dim=0)
                
                with tng.scope.npu_stream_switch(STREAM_TOPK_COMM):
                    topk_local_all_wait = tng.scope.npu_wait_tensor(topk_local_all, topk_local_all)
                    topk_all = all_gather_cross(topk_local_all_wait, idx=1, dim=0)

                with tng.scope.npu_stream_switch(STREAM_PREFETCH):
                    torch_npu.npu_prefetch(self.experts.w13_weight, input_ag, MAX_PREFETCH_SIZE)

                with tng.scope.npu_stream_switch(STREAM_SHARED_EXPERT):
                    hidden_states_int8 = tng.scope.npu_wait_tensor(hidden_states_int8, input_ag)
                    hidden_states_dict = {"x_int8": hidden_states_int8, "pertoken_scale":pertoken_scale}
                    shared_output = self.shared_experts(hidden_states_dict)

            elif SMALL_BATCH:
                with tng.scope.npu_stream_switch(STREAM_TOPK_COMM):
                    topk_local_all = all_gather_local(topk_cat, idx=1, dim=0)

                input_ag = all_gather_local(hidden_states_int8, idx=0, dim=0)
                with tng.scope.npu_stream_switch(STREAM_INTERNODE_COMM_0):
                    round0_swp = tng.scope.npu_wait_tensor(hidden_states_int8, hidden_states_int8)
                    round0_swp = get_round_cross_group_from_list(round=0).swap(round0_swp, method="all2allv")
                    round0_ag = all_gather_local(round0_swp, idx=-1, dim=0)
                with tng.scope.npu_stream_switch(STREAM_INTERNODE_COMM_1):
                    round1_swp = tng.scope.npu_wait_tensor(hidden_states_int8, hidden_states_int8)
                    round1_swp = get_round_cross_group_from_list(round=1).swap(round1_swp, method="all2allv")
                    round1_ag = all_gather_local(round1_swp, idx=-2, dim=0)
                with tng.scope.npu_stream_switch(STREAM_INTERNODE_COMM_2):
                    round2_swp = tng.scope.npu_wait_tensor(hidden_states_int8, input_ag)
                    round2_swp = get_round_cross_group_from_list(round=2).swap(round2_swp, method="all2allv")
                    round2_ag = all_gather_local(round2_swp, idx=-3, dim=0)
                
                with tng.scope.npu_stream_switch(STREAM_TOPK_COMM):
                    topk_local_all_wait = tng.scope.npu_wait_tensor(topk_local_all, topk_local_all)
                    topk_all = all_gather_cross(topk_local_all_wait, idx=1, dim=0)

                with tng.scope.npu_stream_switch(STREAM_PREFETCH):
                    torch_npu.npu_prefetch(self.experts.w13_weight, input_ag, MAX_PREFETCH_SIZE)

                with tng.scope.npu_stream_switch(STREAM_SHARED_EXPERT):
                    hidden_states_int8 = tng.scope.npu_wait_tensor(hidden_states_int8, input_ag)
                    hidden_states_dict = {"x_int8": hidden_states_int8, "pertoken_scale":pertoken_scale}
                    shared_output = self.shared_experts(hidden_states_dict)
            ###### PIPELINE ALL_GATHER ENDS

            with tng.scope.npu_stream_switch(STREAM_TOPK_COMPUTE):
                topk_weights, topk_ids, global_pertoken_scale = torch.split(topk_all,
                                                                            [topk_weights.shape[-1], topk_ids.shape[-1], 1],
                                                                            dim=-1)
                topk_ids = torch.round(topk_ids).to(torch.int32)
                global_pertoken_scale = global_pertoken_scale.squeeze(-1)
        
        else:
            router_logits, _ = self.gate.forward(hidden_states)
            topk_weights, topk_ids, _ = FusedMoE.select_experts(hidden_states, router_logits,
                                                                self.experts.top_k, self.experts.use_grouped_topk,
                                                                self.experts.renormalize,
                                                                self.experts.topk_group, self.experts.num_expert_group,
                                                                self.experts.custom_routing_function,
                                                                self.experts.scoring_func,
                                                                self.experts.e_score_correction_bias,
                                                                self.routed_scaling_factor,
                                                                layer=self.experts)
            topk_ids = self.experts.apply_expert_load_balance(
                topk_ids=topk_ids, 
                best_topk_ids=attn_metadata.decode.best_topk if attn_metadata is not None and attn_metadata.decode is not None else None
            )
            if attn_metadata is not None and attn_metadata.decode is not None:
                actual_batch_mask = attn_metadata.decode.mc2_mask \
                                                        .to(torch.int32).view(-1, 1) \
                                                        .repeat(1, self.experts.top_k)
                topk_ids = actual_batch_mask * topk_ids + (1 - actual_batch_mask) * self.n_routed_experts

            topk_cat = torch.cat((topk_weights, topk_ids.to(torch.float), pertoken_scale.unsqueeze(-1)), dim=-1)

            shared_output = self.shared_experts(hidden_states)
            if model_extra_config.operator_opt_config.use_prefetch:
                torch_npu.npu_prefetch(self.experts.w13_weight, shared_output, MAX_PREFETCH_SIZE)
        
            topk_local_all = all_gather_local(topk_cat, idx=1, dim=0)

            input_ag = all_gather_local(hidden_states_int8, idx=0, dim=0)
            round0_swp = get_round_cross_group_from_list(round=0).swap(hidden_states_int8, method="all2allv")
            round1_swp = get_round_cross_group_from_list(round=1).swap(hidden_states_int8, method="all2allv")
            round2_swp = get_round_cross_group_from_list(round=2).swap(hidden_states_int8, method="all2allv")
            topk_all = all_gather_cross(topk_local_all, idx=1, dim=0)
            round0_ag = all_gather_local(round0_swp, idx=0, dim=0)
            round1_ag = all_gather_local(round1_swp, idx=0, dim=0)
            round2_ag = all_gather_local(round2_swp, idx=0, dim=0)
            topk_weights, topk_ids, global_pertoken_scale = torch.split(topk_all,
                                                                            [topk_weights.shape[-1], topk_ids.shape[-1], 1],
                                                                            dim=-1)
            topk_ids = torch.round(topk_ids).to(torch.int32)
            global_pertoken_scale = global_pertoken_scale.squeeze(-1)

        if self.node_rank == 0:
            global_hidden_states = torch.cat([input_ag, round0_ag, round1_ag, round2_ag], dim=0)
        elif self.node_rank == 1:
            global_hidden_states = torch.cat([round0_ag, input_ag, round2_ag, round1_ag], dim=0)
        elif self.node_rank == 2:
            global_hidden_states = torch.cat([round1_ag, round2_ag, input_ag, round0_ag], dim=0)
        elif self.node_rank == 3:
            global_hidden_states = torch.cat([round2_ag, round1_ag, round0_ag, input_ag], dim=0)

        final_hidden_states = self.experts(
            hidden_states=global_hidden_states,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            pertoken_scale=global_pertoken_scale,
            attn_metadata=attn_metadata
        )

        if self.node_rank == 0:
            input_self, round0, round1, round2 = torch.split(final_hidden_states, final_hidden_states.shape[0] // 4, dim=0)
        elif self.node_rank == 1:
            round0, input_self, round2, round1 = torch.split(final_hidden_states, final_hidden_states.shape[0] // 4, dim=0)
        elif self.node_rank == 2:
            round1, round2, input_self, round0 = torch.split(final_hidden_states, final_hidden_states.shape[0] // 4, dim=0)
        elif self.node_rank == 3:
            round2, round1, round0, input_self = torch.split(final_hidden_states, final_hidden_states.shape[0] // 4, dim=0)

        if model_extra_config.operator_opt_config.moe_multi_stream_tune:
            ##### PIPELINE REDUCE_SCATTER STARTS
            round2 = round2.to(torch.bfloat16)
            with tng.scope.npu_stream_switch(STREAM_TOPK_COMPUTE):
                round1 = round1.to(torch.bfloat16)
                round0 = round0.to(torch.bfloat16)
                input_self = input_self.to(torch.bfloat16)

            with tng.scope.npu_stream_switch(STREAM_PREFETCH):
                if self.attn_prefetch is not None:
                    torch_npu.npu_prefetch(self.attn_prefetch.q_a_proj.weight, input_self, MAX_PREFETCH_SIZE)
                    torch_npu.npu_prefetch(self.attn_prefetch.kv_a_proj_with_mqa.weight, input_self, MAX_PREFETCH_SIZE)
                    torch_npu.npu_prefetch(self.attn_prefetch.q_b_proj.weight, input_self, MAX_PREFETCH_SIZE)
                    torch_npu.npu_prefetch(self.attn_prefetch.W_UK, input_self, MAX_PREFETCH_SIZE)
                if kv_prefetch is not None and isinstance(kv_prefetch, Tuple) and kv_prefetch[0].numel():
                    torch_npu.npu_prefetch(kv_prefetch[0], input_self, MAX_PREFETCH_SIZE)

            round2_rs = reduce_scatter_local(round2, idx=0)
            round1 = tng.scope.npu_wait_tensor(round1, round2_rs)
            round1_rs = reduce_scatter_local(round1, idx=0)
            round0 = tng.scope.npu_wait_tensor(round0, round1_rs)
            round0_rs = reduce_scatter_local(round0, idx=0)
            input_self = tng.scope.npu_wait_tensor(input_self, round0_rs)
            input_rs = reduce_scatter_local(input_self, idx=0)
            with tng.scope.npu_stream_switch(STREAM_INTERNODE_COMM_2):
                round2_swp = get_round_cross_group_from_list(round=2).swap(round2_rs, method="all2allv")
            with tng.scope.npu_stream_switch(STREAM_INTERNODE_COMM_1):
                round1_swp = get_round_cross_group_from_list(round=1).swap(round1_rs, method="all2allv")
            with tng.scope.npu_stream_switch(STREAM_INTERNODE_COMM_0):
                round0_swp = get_round_cross_group_from_list(round=0).swap(round0_rs, method="all2allv")
            ##### PIPELINE REDUCE_SCATTER ENDS
        else:
            round2 = round2.to(torch.bfloat16)
            round1 = round1.to(torch.bfloat16)
            round0 = round0.to(torch.bfloat16)
            input_self = input_self.to(torch.bfloat16)

            if self.attn_prefetch is not None:
                torch_npu.npu_prefetch(self.attn_prefetch.q_a_proj.weight, input_self, MAX_PREFETCH_SIZE)
                torch_npu.npu_prefetch(self.attn_prefetch.kv_a_proj_with_mqa.weight, input_self, MAX_PREFETCH_SIZE)
                torch_npu.npu_prefetch(self.attn_prefetch.q_b_proj.weight, input_self, MAX_PREFETCH_SIZE)
                torch_npu.npu_prefetch(self.attn_prefetch.W_UK, input_self, MAX_PREFETCH_SIZE)
            if kv_prefetch is not None and isinstance(kv_prefetch, Tuple) and kv_prefetch[0].numel():
                torch_npu.npu_prefetch(kv_prefetch[0], input_self, MAX_PREFETCH_SIZE)
        
            round2_rs = reduce_scatter_local(round2, idx=0)
            round1_rs = reduce_scatter_local(round1, idx=0)   
            round0_rs = reduce_scatter_local(round0, idx=0)
            input_rs = reduce_scatter_local(input_self, idx=0)
            round2_swp = get_round_cross_group_from_list(round=2).swap(round2_rs, method="all2allv")
            round1_swp = get_round_cross_group_from_list(round=1).swap(round1_rs, method="all2allv")
            round0_swp = get_round_cross_group_from_list(round=0).swap(round0_rs, method="all2allv")

        final_hidden_states = input_rs + round0_swp + round1_swp + round2_swp + shared_output

        return final_hidden_states, residual
    
    def forward_prefill_a2(self, hidden_states: torch.Tensor, residual: torch.Tensor,
                                    attn_metadata: AttentionMetadata) -> torch.Tensor:
        MULTISTREAM_THRESHOLD = 1200
        GMM_CHUNK_SIZE = MULTISTREAM_THRESHOLD * get_ep_group().world_size
        enable_prefill_moe_multi_stream = True if hidden_states.shape[0] <= MULTISTREAM_THRESHOLD else False
        enable_prefill_pipeline_comm = model_extra_config.operator_opt_config.enable_pipeline_comm and hidden_states.shape[0] <= MULTISTREAM_THRESHOLD
        hidden_states_int8, pertoken_scale = torch_npu.npu_dynamic_quant(hidden_states)

        if enable_prefill_moe_multi_stream:
            shared_stream = torch.npu.Stream()
            curr_stream = torch.npu.current_stream()
            shared_stream.wait_stream(curr_stream)
            with torch.npu.stream(shared_stream):
                global_hidden_states = all_gather_two_stage(hidden_states_int8, idx=0, dim=0)
        else:
            global_hidden_states = all_gather_two_stage(hidden_states_int8, idx=0, dim=0)

        if self.n_routed_experts is not None:
            hidden_states_dict = {"x_int8": hidden_states_int8, "pertoken_scale":pertoken_scale}
            shared_output = self.shared_experts(hidden_states_dict)

        router_logits, _ = self.gate.forward(hidden_states.to(torch.bfloat16))
        topk_weights, topk_ids, _ = FusedMoE.select_experts(hidden_states, router_logits,
                                                            self.experts.top_k, self.experts.use_grouped_topk,
                                                            self.experts.renormalize,
                                                            self.experts.topk_group, self.experts.num_expert_group,
                                                            self.experts.custom_routing_function,
                                                            self.experts.scoring_func,
                                                            self.experts.e_score_correction_bias,
                                                            self.routed_scaling_factor,
                                                            layer=self.experts
                                                            )
        topk_ids = self.experts.apply_expert_load_balance(
            topk_ids=topk_ids, 
            best_topk_ids=attn_metadata.decode.best_topk if attn_metadata is not None and attn_metadata.decode is not None else None
        )
        
        topk_cat = torch.cat((topk_weights, topk_ids.to(torch.float), pertoken_scale.unsqueeze(-1)), dim=-1)
        topk_all = all_gather_two_stage(topk_cat, idx=1, dim=0)
        topk_weights, topk_ids, global_pertoken_scale = torch.split(topk_all, [topk_weights.shape[-1], topk_ids.shape[-1],1], dim=-1)
        topk_ids = torch.round(topk_ids).to(torch.int32)

        global_pertoken_scale = global_pertoken_scale.squeeze(-1)

        if enable_prefill_moe_multi_stream:
            torch.npu.current_stream().wait_stream(shared_stream)
            shared_stream.wait_stream(torch.npu.current_stream())

        
        final_hidden_states = self.chunked_gmm(
            hidden_states=global_hidden_states,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            pertoken_scale=global_pertoken_scale,
            attn_metadata=attn_metadata,
            chunk_size=GMM_CHUNK_SIZE
        )

        if enable_prefill_pipeline_comm:
            final_hidden_states = prefill_reduce_scatter_pipeline(final_hidden_states, idx=1, which_half=self.which_half)
        else:
            final_hidden_states = reduce_scatter_two_stage(final_hidden_states, idx=0)

        if shared_output is not None:
            final_hidden_states = final_hidden_states + shared_output

        return final_hidden_states, residual