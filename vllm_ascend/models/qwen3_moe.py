# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
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
# Adapted from vllm/model_executor/models/qwen3_moe.py
# This file is a part of the vllm-ascend project.

import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from functools import wraps

import torch
import torch.distributed as dist
import torch_npu
import vllm
import vllm.envs as envs
from torch import nn
from torch.nn import functional as F
from transformers import PretrainedConfig
from vllm.attention import AttentionMetadata
from vllm.distributed import (get_dp_group,
                              get_pp_group, 
                              get_tensor_model_parallel_world_size,
                              get_tp_group, 
                              tensor_model_parallel_all_gather, 
                              tensor_model_parallel_reduce_scatter)
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.linear import ReplicatedLinear

                                               
from vllm.model_executor.layers.quantization import QuantizationConfig

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.distributed.parallel_state import get_ep_group
from vllm_ascend.ops.fused_moe import AscendFusedMoE
import vllm_ascend.envs as envs_ascend

from vllm.model_executor.models.qwen3_moe import Qwen3MoeForCausalLM
from transformers import PretrainedConfig
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.sequence import IntermediateTensors
from vllm.forward_context import get_forward_context


VLLM_ENABLE_SP: bool = envs_ascend.VLLM_ENABLE_SP


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


class AscendQwen3MoeSparseMoeBlock(nn.Module):
    
    top_k: int

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.tp_size = get_tensor_model_parallel_world_size()
        if self.tp_size > config.num_experts:
            raise ValueError(
                f"Tensor parallel size {self.tp_size} is greater than "
                f"the number of experts {config.num_experts}.")

        ascend_config = get_ascend_config()
        self.torchair_graph_enabled = ascend_config.torchair_graph_config.enabled
        self.enable_multistream_moe = \
            ascend_config.torchair_graph_config.enable_multistream_moe

        self.gate = ReplicatedLinear(config.hidden_size,
                                     config.num_experts,
                                     bias=False,
                                     quant_config=None,
                                     prefix=f"{prefix}.gate")

        self.experts = AscendFusedMoE(
            num_experts=config.num_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            reduce_results=False,
            renormalize=config.norm_topk_prob,
            quant_config=quant_config,
            prefix=f"{prefix}.experts")

        
        self.top_k = config.num_experts_per_tok

        self.dp_size = get_dp_group().world_size

        self.tp_group = get_tp_group().device_group
        self.tp_rank = get_tp_group().rank_in_group
        self.ep_group = get_ep_group()

        self.params_dtype = torch.get_default_dtype()

    def forward(
            self,
            hidden_states: torch.Tensor,
            attn_metadata: Optional[AttentionMetadata] = None, not_Dummy = False) -> torch.Tensor:
        if attn_metadata is None:
            attn_metadata = get_forward_context().attn_metadata
        # when profile runs, force experts to load balanced tokens
        # to avoid high memory consumption on a single rank.
        # TODO: need a better flag to indicate whether in profile run or not.
        if attn_metadata is None:
            # for profile run
            is_prefill = True
            enable_force_load_balance = True
        else:
            # is_prefill = attn_metadata.num_prefills > 0 is_prefill or
            enable_force_load_balance = False
            if hasattr(attn_metadata, 'with_prefill_across_dp'):
                is_prefill = attn_metadata.with_prefill_across_dp

        # router_logits: (num_tokens, n_experts)
        router_logits, _ = self.gate(hidden_states)

        hidden_states = self.experts(
            hidden_states=hidden_states,
            router_logits=router_logits,
            is_prefill=is_prefill,
            top_k=self.top_k,
            enable_force_load_balance=enable_force_load_balance,
            shared_experts=None,
            not_Dummy=not_Dummy
        )

        return hidden_states


def Qwen3MoeForCausalLM_forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        attn_metadata = get_forward_context().attn_metadata
        global _dp_metadata_for_padding
        is_perifll = 0 
        if attn_metadata is not None:
            if hasattr(attn_metadata, 'is_only_prefill') and attn_metadata.is_only_prefill:
                is_perifll = 1
            if hasattr(attn_metadata, 'num_prefills') and attn_metadata.num_prefills > 0:
                is_perifll = 1
        if attn_metadata is not None and is_perifll:
            lengths_sum_unpadding = input_ids.shape[0]
            self.tp_size = get_tensor_model_parallel_world_size()
            lengths_sum_padding = ((lengths_sum_unpadding + self.tp_size - 1) // self.tp_size) * self.tp_size
            if lengths_sum_unpadding == lengths_sum_padding:
                padding_flag = False
            else:
                padding_flag = True
            pad_size = lengths_sum_padding - lengths_sum_unpadding
            
            _dp_metadata_for_padding = MetadataForPadding(padding_flag, lengths_sum_padding, lengths_sum_unpadding, pad_size, True)

        else:
            _dp_metadata_for_padding = MetadataForPadding(False, None, None, None, False)
        
        hidden_states = self.model(input_ids, positions, intermediate_tensors,
                                   inputs_embeds, is_perifll)
        return hidden_states


def Qwen3MoeModel_forward(
    self,
    input_ids: torch.Tensor,
    positions: torch.Tensor,
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
    for i in range(self.start_layer, self.end_layer):
        layer = self.layers[i]
        hidden_states, residual = layer(positions, hidden_states, residual)
    if not get_pp_group().is_last_rank:
        return IntermediateTensors({
            "hidden_states": hidden_states,
            "residual": residual
        })
    hidden_states, _ = self.norm(hidden_states, residual)
    
    if VLLM_ENABLE_SP and _dp_metadata_for_padding.not_Dummy:              
        hidden_states = tensor_model_parallel_all_gather(hidden_states, 0)
        if _dp_metadata_for_padding.padding_flag:
            hidden_states = unpadding_aligned_wp(hidden_states)

    return hidden_states


def Qwen3MoeDecoderLayer_init_wrapper(fn):
    @wraps(fn)
    def wrapper(self, *arg, **kwargs):
        fn(self, *arg, **kwargs)
        self.tp_size = get_tp_group().world_size
        self.tp_rank_in_group = get_tp_group().rank_in_group
    
    return wrapper


def Qwen3MoeDecoderLayer_forward(
    self,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    residual: Optional[torch.Tensor],
) -> torch.Tensor:
    
    # To prevent precision issues during the decoder phase when only prefilling enables SP
    self.self_attn.o_proj.reduce_results = not _dp_metadata_for_padding.not_Dummy

    # Self Attention
    if residual is None:
        is_start = True
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
    else:
        is_start = False
    
        hidden_states, residual = self.input_layernorm(
            hidden_states, residual)

        if VLLM_ENABLE_SP and _dp_metadata_for_padding.not_Dummy:
            hidden_states = tensor_model_parallel_all_gather(hidden_states, 0)
            if _dp_metadata_for_padding.padding_flag:
                hidden_states = unpadding_aligned_wp(hidden_states)
    
    hidden_states = self.self_attn(
        positions=positions,
        hidden_states=hidden_states,
    )
    
    if VLLM_ENABLE_SP and _dp_metadata_for_padding.not_Dummy:
        if _dp_metadata_for_padding.padding_flag:
            hidden_states = padding_aligned_tp(hidden_states)
        hidden_states = tensor_model_parallel_reduce_scatter(hidden_states, 0)

    if is_start and VLLM_ENABLE_SP and _dp_metadata_for_padding.not_Dummy:
        reduce_scatter_tokens = hidden_states.size(0)
        residual = F.pad(residual, (0, 0, 0, reduce_scatter_tokens * self.tp_size - residual.size(0)))
        start = self.tp_rank_in_group * reduce_scatter_tokens
        residual = residual[start:start + reduce_scatter_tokens]

    # Fully Connected
    hidden_states, residual = self.post_attention_layernorm(
        hidden_states, residual)
    
    hidden_states = self.mlp(hidden_states, sp_prefill=self.is_perifll, not_Dummy=_dp_metadata_for_padding.not_Dummy)
    
    return hidden_states, residual


@dataclass
class MetadataForPadding:
    padding_flag: bool
    lengths_sum_padding: torch.int
    lengths_sum_unpadding: torch.int
    pad_size: torch.int
    # not_Dummy is True when perfill is true and not dummy
    not_Dummy: bool


def padding_aligned_tp(data: torch.Tensor) -> torch.Tensor:
    pad_size = _dp_metadata_for_padding.pad_size
    return F.pad(data, (0, 0, 0, pad_size))


def unpadding_aligned_wp(padded_data: torch.Tensor) -> torch.Tensor:
    lengths_sum_unpadding= _dp_metadata_for_padding.lengths_sum_unpadding
    return padded_data[:lengths_sum_unpadding]

_dp_metadata_for_padding: Optional[MetadataForPadding] = None

vllm.model_executor.models.qwen3_moe.Qwen3MoeSparseMoeBlock = AscendQwen3MoeSparseMoeBlock
vllm.model_executor.models.qwen3_moe.Qwen3MoeForCausalLM.forward = Qwen3MoeForCausalLM_forward
vllm.model_executor.models.qwen3_moe.Qwen3MoeModel.forward = Qwen3MoeModel_forward
vllm.model_executor.models.qwen3_moe.Qwen3MoeDecoderLayer.__init__ = Qwen3MoeDecoderLayer_init_wrapper
vllm.model_executor.models.qwen3_moe.Qwen3MoeDecoderLayer.forward = Qwen3MoeDecoderLayer_forward
