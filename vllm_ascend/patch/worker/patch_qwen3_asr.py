#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
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
# from collections.abc import Iterable
# mypy: ignore-errors


from typing import Any

import torch
from torch import nn

from vllm.config import CacheConfig
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.logger import init_logger
from vllm.model_executor.layers.attention.encoder_only_attention import (
    Attention,
    EncoderOnlyAttention,
)
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import QKVParallelLinear, RowParallelLinear
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.models.qwen3 import Qwen3Attention
from vllm.model_executor.models.utils import extract_layer_index
from vllm.v1.attention.backend import AttentionType

logger = init_logger(__name__)

class AscendQwen3Attention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        rope_parameters: dict,
        max_position: int = 4096 * 32,
        head_dim: int | None = None,
        rms_norm_eps: float = 1e-06,
        qkv_bias: bool = False,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        attn_type: str = AttentionType.DECODER,
        dual_chunk_attention_config: dict[str, Any] | None = None,
    ) -> None:
        # super().__init__()
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
        self.head_dim = head_dim or hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.dual_chunk_attention_config = dual_chunk_attention_config

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=qkv_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        self.rotary_emb = get_rope(
            self.head_dim,
            max_position=max_position,
            rope_parameters=rope_parameters,
            dual_chunk_attention_config=dual_chunk_attention_config,
        )
        attn_cls = EncoderOnlyAttention if attn_type == AttentionType.ENCODER_ONLY else Attention
        self.attn = attn_cls(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
            attn_type=attn_type,
            **{
                "layer_idx": extract_layer_index(prefix),
                "dual_chunk_attention_config": dual_chunk_attention_config,
            }
            if dual_chunk_attention_config
            else {},
        )
        self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)

        self.rms_norm_eps = rms_norm_eps
        self.rope_dim = self.rotary_emb.rotary_dim
        self.is_interleaved = self.rotary_emb.mrope_interleaved
        self.mrope_section = self.rotary_emb.mrope_section
        half_rope_dim = self.rope_dim // 2
        cos_sin_nblk_idx = torch.arange(0, half_rope_dim)
        cos_sin_mask = torch.ones((1, half_rope_dim), dtype=torch.bfloat16)
        if self.is_interleaved:
            h_nmask_end_interleaved = 3 * self.mrope_section[1]
            w_nmask_end_interleaved = 3 * self.mrope_section[2]
            h_nmask = ((cos_sin_nblk_idx % 3) == 1) & (cos_sin_nblk_idx <= h_nmask_end_interleaved)
            w_nmask = ((cos_sin_nblk_idx % 3) == 2) & (cos_sin_nblk_idx <= w_nmask_end_interleaved)
            t_nmask = ~(h_nmask | w_nmask)
        else:
            t_nmask = cos_sin_nblk_idx < self.mrope_section[0]
            h_nmask = (self.mrope_section[0] - 1 < cos_sin_nblk_idx) & (
                cos_sin_nblk_idx < mrope_section[0]+mrope_section[1]
            )
            w_nmask = (self.mrope_section[0]+self.mrope_section[1] - 1 < cos_sin_nblk_idx) & (
                cos_sin_nblk_idx < half_rope_dim
            )
        h_nmask = torch.where(h_nmask, cos_sin_mask, 0.0)
        w_nmask = torch.where(w_nmask, cos_sin_mask, 0.0)
        t_nmask = torch.where(t_nmask, cos_sin_mask, 0.0)
        self.thw_mask = torch.stack([t_nmask, h_nmask, w_nmask], dim=0)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        cos_sin = self.rotary_emb.cos_sin_cache[positions]
        if cos_sin.device != qkv.device:
            cos_sin = cos_sin.to(qkv.device)
        if cos_sin.dtype != qkv.dtype:
            cos_sin = cos_sin.to(qkv.dtype)
        q, k, v = torch.ops.vllm.triton_split_qkv_rmsnorm_mrope(
            qkv=qkv,
            q_weight=self.q_norm.weight,
            k_weight=self.k_norm.weight,
            cos_sin=cos_sin.contiguous(),
            num_q_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            head_size=self.head_dim,
            eps=self.rms_norm_eps,
            mrope_section=self.mrope_section,
            rope_dim=self.rope_dim,
            thw_mask = self.thw_mask.to(qkv.device)
        )

        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output

Qwen3Attention.__init__ = AscendQwen3Attention.__init__
Qwen3Attention.forward = AscendQwen3Attention.forward
