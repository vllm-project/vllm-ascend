#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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

"""Qwen2.5-VL vision-attention patch for fused Ascend preprocessing."""

import einops
import torch
from vllm.model_executor.models.qwen2_5_vl import Qwen2_5_VisionAttention


def qwen2_5_vision_attention_forward(
    self: Qwen2_5_VisionAttention,
    x: torch.Tensor,
    cu_seqlens: torch.Tensor,
    rotary_pos_emb_cos: torch.Tensor,
    rotary_pos_emb_sin: torch.Tensor,
    max_seqlen: torch.Tensor,
    sequence_lengths: torch.Tensor | None,
) -> torch.Tensor:
    """Use fused QKV/RoPE/padding when the Ascend adapter accepts the input."""
    x, _ = self.qkv(x)
    seq_len, batch_size, _ = x.shape

    fused_qkv_fia = getattr(self.attn, "forward_qkv_rope_pad_fia", None)
    if fused_qkv_fia is not None:
        context_layer = fused_qkv_fia(
            x,
            rotary_pos_emb_cos,
            rotary_pos_emb_sin,
            cu_seqlens=cu_seqlens,
            sequence_lengths=sequence_lengths,
        )
        if context_layer is not None:
            context_layer = einops.rearrange(
                context_layer,
                "b s h d -> s b (h d)",
                b=batch_size,
            ).contiguous()
            output, _ = self.proj(context_layer)
            return output

    qkv = einops.rearrange(
        x,
        "s b (three head head_dim) -> b s three head head_dim",
        three=3,
        head=self.num_attention_heads_per_partition,
    )

    if rotary_pos_emb_cos is not None and rotary_pos_emb_sin is not None:
        qk, v = qkv[:, :, :2], qkv[:, :, 2]
        qk_reshaped = einops.rearrange(
            qk,
            "b s two head head_dim -> (two b) s head head_dim",
            two=2,
        ).contiguous()
        qk_rotated = self.apply_rotary_emb(
            qk_reshaped,
            rotary_pos_emb_cos,
            rotary_pos_emb_sin,
        )
        qk_rotated = qk_rotated.view(
            2,
            batch_size,
            seq_len,
            self.num_attention_heads_per_partition,
            self.hidden_size_per_attention_head,
        )
        q, k = qk_rotated.unbind(dim=0)
    else:
        q, k, v = qkv.unbind(dim=2)

    context_layer = self.attn(
        query=q,
        key=k,
        value=v,
        cu_seqlens=cu_seqlens,
        max_seqlen=max_seqlen,
        sequence_lengths=sequence_lengths,
    )
    context_layer = einops.rearrange(
        context_layer,
        "b s h d -> s b (h d)",
        b=batch_size,
    ).contiguous()
    output, _ = self.proj(context_layer)
    return output


Qwen2_5_VisionAttention.forward = qwen2_5_vision_attention_forward  # type: ignore[method-assign]
