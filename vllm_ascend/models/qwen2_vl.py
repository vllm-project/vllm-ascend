#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Adapted from vllm/model_executor/models/qwen2_vl.py
# Copyright 2023 The vLLM team.
# 
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
#

import torch

import torch_npu
from einops import rearrange

from vllm.model_executor.models.qwen2_vl import apply_rotary_pos_emb_vision
from vllm.model_executor.models.qwen2_vl import Qwen2VisionAttention, Qwen2VisionTransformer


def qwen2_vl_vision_attention_forward(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: torch.Tensor,
    ) -> torch.Tensor:

    # [s, b, c] --> [s, b, 3 * head * head_dim]
    x, _ = self.qkv(x)

    # [s, b, 3 * head * head_dim] -> 3 * [s, b, head, head_dim]
    q, k, v = self.split_qkv(x)
    batch_size = q.shape[1]

    q, k, v = [
        rearrange(x, "s b ... -> b s ...").contiguous()
        for x in (q, k, v)
    ]

    if rotary_pos_emb is not None:
        q = apply_rotary_pos_emb_vision(q, rotary_pos_emb)
        k = apply_rotary_pos_emb_vision(k, rotary_pos_emb)

    q, k, v = [
        rearrange(x, "b s h d -> (b s) h d").contiguous()
        for x in (q, k, v)
    ]

    context_layer = torch.torch.empty_like(q)

    # this requires pta version >= B033
    torch_npu._npu_flash_attention_unpad(query=q, key=k, value=v,
                                        seq_len=cu_seqlens,
                                        scale_value=self.hidden_size_per_attention_head ** -0.5,
                                        num_heads=self.num_attention_heads_per_partition,
                                        num_kv_heads=self.num_attention_heads_per_partition,
                                        out=context_layer)
    context_layer = rearrange(context_layer,
                               "(b s) h d -> s b (h d)", b=batch_size).contiguous()

    output, _ = self.proj(context_layer)
    return output


def qwen2_vl_vision_transformer_forward(
        self,
        x: torch.Tensor,
        grid_thw: torch.Tensor,
    ) -> torch.Tensor:

    x = x.to(device=self.device, dtype=self.dtype)
    x = self.patch_embed(x)
    rotary_pos_emb = self.rot_pos_emb(grid_thw)

    #do not cumsum cu_seqlens to meet the requirements of unpadFA.
    cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2],
                                         grid_thw[:, 0]).cpu().to(torch.int32)

    x = x.unsqueeze(1)
    for blk in self.blocks:
        x = blk(x, cu_seqlens=cu_seqlens, rotary_pos_emb=rotary_pos_emb)

    x = self.merger(x)
    return x

Qwen2VisionAttention.forward = qwen2_vl_vision_attention_forward
Qwen2VisionTransformer.forward = qwen2_vl_vision_transformer_forward