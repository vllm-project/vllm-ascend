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
#

from contextlib import contextmanager

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_npu
from einops import rearrange
from vllm.model_executor.models.qwen2_5_vl import Qwen2_5_VisionAttention

import vllm_ascend.envs as envs_ascend

MIN_PAD_SIZE = 64  # min_size to pad weight
MAX_PAD_SIZE = 128  # max_size to pad weight


@contextmanager
def _padding_manager(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    origin_shape: int,
    hidden_size_per_attention_head: int,
):
    enable_pad = (envs_ascend.USE_OPTIMIZED_MODEL
                  and hidden_size_per_attention_head > MIN_PAD_SIZE
                  and hidden_size_per_attention_head < MAX_PAD_SIZE)

    if enable_pad:
        half_pad_hidden_size_per_attention_head = (
            MAX_PAD_SIZE - hidden_size_per_attention_head) // 2
        hidden_size_per_attention_head = MAX_PAD_SIZE

        pad_len = MAX_PAD_SIZE - origin_shape
        # q/k/v: [b, s, head, head_dim] -> [b, s, head, MAX_PAD_SIZE]
        q = F.pad(q, (0, pad_len), mode="constant", value=0)
        k = F.pad(k, (0, pad_len), mode="constant", value=0)
        v = F.pad(v, (0, pad_len), mode="constant", value=0)
        # cos/sin: [seqlen, rotary_dim / 2] -> [b, s, head, MAX_PAD_SIZE / 2]
        cos = torch.nn.functional.pad(
            cos, (0, half_pad_hidden_size_per_attention_head))
        sin = torch.nn.functional.pad(
            sin, (0, half_pad_hidden_size_per_attention_head))

    cos = rearrange(
        torch.stack((cos, cos), dim=-1),
        "... d two -> ...(d two)",
        two=2,
    )
    sin = rearrange(
        torch.stack((sin, sin), dim=-1),
        "... d two -> ...(d two)",
        two=2,
    )
    cos = cos.reshape(1, -1, 1, hidden_size_per_attention_head)
    sin = sin.reshape(1, -1, 1, hidden_size_per_attention_head)

    try:
        yield (q, k, v, cos, sin, enable_pad)
    finally:
        pass


class AscendQwen2_5_VisionAttention(nn.Module):

    def forward(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb_cos: torch.Tensor,
        rotary_pos_emb_sin: torch.Tensor,
        max_seqlen: torch.Tensor,
        seqlens: torch.Tensor,
    ) -> torch.Tensor:
        # [s, b, c] --> [s, b, head * 3 * head_dim]
        x, _ = self.qkv(x)

        # [s, b, 3 * head * head_dim] -> 3 * [s, b, head, head_dim]
        q, k, v = self.split_qkv(x)
        batch_size = q.shape[1]
        origin_shape = q.shape[-1]

        q, k, v = (rearrange(x, "s b ... -> b s ...").contiguous()
                   for x in (q, k, v))

        # Convert cumulative tensor to intervals and move it to cpu.
        cu_seqlens = torch.diff(cu_seqlens).to("cpu")

        with _padding_manager(
                q=q,
                k=k,
                v=v,
                cos=rotary_pos_emb_cos,
                sin=rotary_pos_emb_sin,
                origin_shape=origin_shape,
                hidden_size_per_attention_head=self.
                hidden_size_per_attention_head,
        ) as (q, k, v, cos, sin, enable_pad):
            q = torch_npu.npu_rotary_mul(q, cos, sin)
            k = torch_npu.npu_rotary_mul(k, cos, sin)

            q, k, v = [
                rearrange(x, "b s h d -> (b s) h d").contiguous()
                for x in (q, k, v)
            ]

            context_layer = torch.empty_like(q)

            # operator requires pta version >= 2.5.1
            torch_npu._npu_flash_attention_unpad(
                query=q,
                key=k,
                value=v,
                seq_len=cu_seqlens,
                scale_value=self.hidden_size_per_attention_head**-0.5,
                num_heads=self.num_attention_heads_per_partition,
                num_kv_heads=self.num_attention_heads_per_partition,
                out=context_layer,
            )

            if enable_pad:
                context_layer = context_layer[..., :origin_shape]

            context_layer = rearrange(context_layer,
                                      "(b s) h d -> s b (h d)",
                                      b=batch_size).contiguous()

        output, _ = self.proj(context_layer)
        return output


Qwen2_5_VisionAttention.forward = AscendQwen2_5_VisionAttention.forward
