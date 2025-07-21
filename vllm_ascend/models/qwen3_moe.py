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

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
import torch_npu
import vllm
import vllm.envs as envs
from torch import nn
from transformers import PretrainedConfig
from vllm.attention import AttentionMetadata
from vllm.distributed import (get_tensor_model_parallel_world_size,
                              get_tp_group)
from vllm.distributed.parallel_state import get_dp_group
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.linear import ReplicatedLinear
                                               
from vllm.model_executor.layers.quantization import QuantizationConfig

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.distributed.parallel_state import get_ep_group
from vllm_ascend.ops.fused_moe import AscendFusedMoE

from vllm.model_executor.models.qwen3_moe import Qwen3MoeForCausalLM
from transformers import PretrainedConfig
from vllm.model_executor.layers.quantization import QuantizationConfig

from vllm_ascend.ops.oproj import oproj_RowParallelLinear
from vllm.config import CacheConfig, ModelConfig, VllmConfig, get_current_vllm_config
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (MergedColumnParallelLinear,
                                               QKVParallelLinear,
                                               ReplicatedLinear,
                                               RowParallelLinear)
from vllm.attention import Attention
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.forward_context import ForwardContext, get_forward_context


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


class AscendQwen3MoeAttention(nn.Module):

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

        self.additional_config = get_current_vllm_config().additional_config

        self.qkv_proj = QKVParallelLinear(hidden_size,
                                          self.head_dim,
                                          self.total_num_heads,
                                          self.total_num_kv_heads,
                                          bias=qkv_bias,
                                          quant_config=quant_config,
                                          prefix=f"{prefix}.qkv_proj")
        
        if self.additional_config is not None and self.additional_config.get("oproj_tensor_parallel_size", False):
            self.o_proj = oproj_RowParallelLinear(self.total_num_heads * self.head_dim,
                                            hidden_size,
                                            bias=False,
                                            quant_config=quant_config,
                                            prefix=f"{prefix}.o_proj")
        else:
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

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        attn_metadata = get_forward_context().attn_metadata

        if self.additional_config is not None and \
                self.additional_config.get("oproj_tensor_parallel_size", False) and \
                attn_metadata and \
                getattr(attn_metadata, 'is_dummy', None) is not None and \
                getattr(attn_metadata, 'is_dummy') == True:
            dummy_hidden_size = self.head_dim * self.num_heads 
            dummy_hidden_state = torch.empty(attn_metadata.num_actual_tokens, dummy_hidden_size, dtype=hidden_states.dtype, device=hidden_states.device)
            dummy_out_put = self.o_proj(dummy_hidden_state)
            return dummy_out_put[0]

        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        # Add qk-norm
        q_by_head = q.view(*q.shape[:-1], q.shape[-1] // self.head_dim,
                           self.head_dim)
        q_by_head = self.q_norm(q_by_head)
        q = q_by_head.view(q.shape)

        k_by_head = k.view(*k.shape[:-1], k.shape[-1] // self.head_dim,
                           self.head_dim)
        k_by_head = self.k_norm(k_by_head)
        k = k_by_head.view(k.shape)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output


vllm.model_executor.models.qwen3_moe.Qwen3MoeAttention = AscendQwen3MoeAttention