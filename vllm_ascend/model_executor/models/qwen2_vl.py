# Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.

"""
MindIE is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details.
"""

import torch
import torch_npu
from einops import rearrange

from vllm.model_executor.models.qwen2_vl import apply_rotary_pos_emb_vision
from vllm.model_executor.models.qwen2_vl import Qwen2VisionAttention,Qwen2VisionTransformer


def qwen2_vl_vision_attention_forward(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: torch.Tensor = None,
    ) -> torch.Tensor:

    x, _ = self.qkv(x)
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

    cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2],
                                         grid_thw[:, 0]).cpu().to(torch.int32)

    x = x.unsqueeze(1)
    for blk in self.blocks:
        x = blk(x, cu_seqlens=cu_seqlens, rotary_pos_emb=rotary_pos_emb)

    x = self.merger(x)
    return x

Qwen2VisionAttention.forward = qwen2_vl_vision_attention_forward
Qwen2VisionTransformer.forward = qwen2_vl_vision_transformer_forward