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
from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
from vllm.attention.backends.abstract import AttentionMetadata
from vllm.config import CacheConfig, get_current_vllm_config
from vllm.forward_context import ForwardContext, get_forward_context
from vllm.model_executor.layers.mla import MultiHeadLatentAttentionWrapper
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.utils.torch_utils import direct_register_custom_op

from vllm_ascend.models.layer.attention.layer import DSAAttention


@dataclass
class DSAModules:
    """Modules used in SFA V2."""

    wq_a: torch.nn.Module
    q_norm: torch.nn.Module
    wq_b: torch.nn.Module
    wkv: torch.nn.Module
    kv_norm: torch.nn.Module
    wo_a: torch.nn.Module
    wo_b: torch.nn.Module
    attn_sink: torch.nn.Module
    indexer: torch.nn.Module | None
    compressor: torch.nn.Module | None
    topk_indices_buffer: torch.Tensor | None
    indexer_rotary_emb: torch.nn.Module | None = None


class AscendDeepseekSparseAttention(MultiHeadLatentAttentionWrapper):

    def __init__(
        self,
        dim: int,
        n_heads: int,
        scale: float,
        n_local_heads: int,
        q_lora_rank: int,
        o_lora_rank: int,
        head_dim: int,
        rope_head_dim: int | None,
        nope_head_dim: int,
        eps: float,
        n_groups: int,
        n_local_groups: int,
        window_size: int,
        compress_ratio: int,
        dsa_modules: DSAModules,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        nn.Module.__init__(self)
        self.dim = dim
        self.n_heads = n_heads
        self.scale = scale
        self.n_local_heads = n_local_heads
        self.q_lora_rank = q_lora_rank
        self.o_lora_rank = o_lora_rank
        self.head_dim = head_dim
        self.rope_head_dim = rope_head_dim
        self.nope_head_dim = nope_head_dim
        self.eps = eps
        self.n_groups = n_groups
        self.n_local_groups = n_local_groups
        self.window_size = window_size
        self.compress_ratio = compress_ratio

        self.wq_a = dsa_modules.wq_a
        self.q_norm = dsa_modules.q_norm
        self.wq_b = dsa_modules.wq_b
        self.wkv = dsa_modules.wkv
        self.kv_norm = dsa_modules.kv_norm
        self.wo_a = dsa_modules.wo_a
        self.wo_b = dsa_modules.wo_b
        self.attn_sink = dsa_modules.attn_sink
        self.indexer = dsa_modules.indexer
        self.compressor = dsa_modules.compressor
        self.topk_indices_buffer = dsa_modules.topk_indices_buffer
        self.indexer_rotary_emb = dsa_modules.indexer_rotary_emb
        self.prefix = prefix

        self.dsa_attn = DSAAttention(
            dim=self.dim,
            n_heads=self.n_heads,
            scale=self.scale,
            n_local_heads=self.n_local_heads,
            q_lora_rank=self.q_lora_rank,
            o_lora_rank=self.o_lora_rank,
            head_dim=self.head_dim,
            rope_head_dim=self.rope_head_dim,
            nope_head_dim=self.nope_head_dim,
            n_groups=self.n_groups,
            n_local_groups=self.n_local_groups,
            window_size=self.window_size,
            compress_ratio=self.compress_ratio,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",

            #extra
            wq_a=self.wq_a,
            wq_b=self.wq_b,
            wkv=self.wkv,
            q_norm=self.q_norm,
            kv_norm=self.kv_norm,
            indexer=self.indexer,
            compressor=self.compressor,
            wo_a=self.wo_a,
            wo_b=self.wo_b,
            attn_sink=self.attn_sink,
            eps=self.eps,
        )

        compilation_config = get_current_vllm_config().compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self

    def forward(
            self,
            positions: torch.Tensor,
            hidden_states: torch.Tensor,
            kv_cache: Optional[torch.Tensor] = None,
            attn_metadata: Optional[AttentionMetadata] = None) -> torch.Tensor:
        need_gather_q_kv = get_forward_context().sp_enabled
        output_shape = hidden_states.shape
        # FIXME: This does not seem right, should make sure the buffer is fixed
        output = torch.empty(output_shape,
                             dtype=hidden_states.dtype,
                             device=hidden_states.device)
        torch.ops.vllm.dsa_forward(hidden_states, need_gather_q_kv, output,
                                   self.prefix)
        output = output.view(-1, output_shape[-1])
        return output


def dsa_forward(
    hidden_states: torch.Tensor,
    need_gather_q_kv: bool,
    output: torch.Tensor,
    layer_name: str,
) -> None:
    forward_context: ForwardContext = get_forward_context()
    self = forward_context.no_compile_layers[layer_name]
    if forward_context.attn_metadata:
        attn_metadata = forward_context.attn_metadata[self.dsa_attn.layer_name]
    else:
        attn_metadata = forward_context.attn_metadata
    kv_cache = self.dsa_attn.kv_cache[forward_context.virtual_engine]
    kv_state = self.dsa_attn.kv_state[0]  # TODO support PP
    self.dsa_attn.impl.forward(self.dsa_attn.layer_name, hidden_states,
                               kv_cache, attn_metadata, need_gather_q_kv,
                               output, kv_state)
    return


def dsa_forward_fake(
    hidden_states: torch.Tensor,
    need_gather_q_kv: bool,
    output: torch.Tensor,
    layer_name: str,
) -> None:
    return


direct_register_custom_op(
    op_name="dsa_forward",
    op_func=dsa_forward,
    mutates_args=["output"],
    fake_impl=dsa_forward_fake,
    dispatch_key="PrivateUse1",
)
