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

import einops
import torch
import torch.nn.functional as F
import torch_npu
from vllm.attention.layers.mm_encoder_attention import MMEncoderAttention
from vllm.config import MultiModalConfig

import vllm_ascend.envs as envs_ascend
from vllm_ascend.utils import is_310p

MIN_PAD_SIZE = 64  # min_size to pad weight
MAX_PAD_SIZE = 128  # max_size to pad weight


class AscendMMEncoderAttention(MMEncoderAttention):

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float | None = None,
        num_kv_heads: int | None = None,
        prefix: str = "",
        multimodal_config: MultiModalConfig | None = None,
    ) -> None:
        """
        Args:
            num_heads: number of attention heads per partition.
            head_size: hidden_size per attention head.
            scale: scale factor.
            num_kv_heads: number of kv heads.
            prefix: This has no effect, it is only here to make it easier to
                    swap between Attention and MMEncoderAttention.
            multimodal_config: configs for multi-modal.
        """
        super().__init__(
            num_heads=num_heads,
            head_size=head_size,
            scale=scale,
            num_kv_heads=num_kv_heads,
            prefix=prefix,
            multimodal_config=multimodal_config,
        )

    def reshape_qkv_to_3d(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        bsz: int,
        q_len: int,
        kv_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Reshape query, key, value to 3D tensors:
        (batch_size * seq_len, num_heads, head_size)
        """
        query = query.view(bsz * q_len, self.num_heads, self.head_size)
        key = key.view(bsz * kv_len, self.num_kv_heads, self.head_size)
        value = value.view(bsz * kv_len, self.num_kv_heads, self.head_size)
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads
        if (num_repeat := self.num_queries_per_kv) > 1:
            # Handle MQA and GQA
            key = torch.repeat_interleave(key, num_repeat, dim=1)
            value = torch.repeat_interleave(value, num_repeat, dim=1)

        return query, key, value

def forward_oot(
            self,
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            cu_seqlens: torch.Tensor | None = None,
            max_seqlen: torch.Tensor
        | None = None,  # Only used for Flash Attention
    ):
        bsz, q_len = query.size()[:2]
        kv_len = key.size(1)
        is_reshaped = query.dim() == 4

        # q, k, v: [b, s, head, head_dim] -> [b * s, head, head_dim]
        q, k, v = self.reshape_qkv_to_3d(query, key, value, bsz, q_len, kv_len)

        enable_pad = (envs_ascend.USE_OPTIMIZED_MODEL
                      and self.head_size > MIN_PAD_SIZE
                      and self.head_size < MAX_PAD_SIZE)

        origin_shape = q.shape[-1]
        if enable_pad:
            pad_len = MAX_PAD_SIZE - origin_shape
            # q, k, v: [b * s, head, head_dim] -> [b * s, head, MAX_PAD_SIZE]
            q = F.pad(q, (0, pad_len), mode="constant", value=0)
            k = F.pad(k, (0, pad_len), mode="constant", value=0)
            v = F.pad(v, (0, pad_len), mode="constant", value=0)

        if is_310p():
            original_dim = origin_shape

            cur_dim = q.shape[-1]
            pad16 = (16 - cur_dim % 16) % 16
            if pad16:
                q = F.pad(q, (0, pad16), mode="constant", value=0)
                k = F.pad(k, (0, pad16), mode="constant", value=0)
                v = F.pad(v, (0, pad16), mode="constant", value=0)

            if cu_seqlens is None:
                cu_seqlens = torch.arange(
                    0,
                    (bsz + 1) * q_len,
                    step=q_len,
                    dtype=torch.int32,
                    device=query.device,
                )

            total_q_tokens = bsz * q_len
            context_flat = q.new_empty((total_q_tokens, self.num_heads, q.shape[-1]))

            st = 0
            cu_lens = torch.diff(cu_seqlens).to("cpu", dtype=torch.int64).tolist()
            for seg_len in cu_lens:
                seg_len = int(seg_len)
                ed = st + seg_len

                q_i = q[st:ed].unsqueeze(0)
                k_i = k[st:ed].unsqueeze(0)
                v_i = v[st:ed].unsqueeze(0)

                qs = int(q_i.shape[1])
                kvs = int(k_i.shape[1])

                out_i = torch_npu.npu_prompt_flash_attention(
                    q_i,
                    k_i,
                    v_i,
                    input_layout="BSND",
                    num_heads=self.num_heads,
                    num_key_value_heads=self.num_kv_heads,
                    scale_value=self.head_size**-0.5,
                    pre_tokens=qs,
                    next_tokens=kvs,
                )
                context_flat[st:ed] = out_i[0]
                st = ed

            context_flat = context_flat[..., :original_dim]

            context_layer = einops.rearrange(
                context_flat, "(b s) h d -> b s h d", b=bsz
            ).contiguous()
            return context_layer

        context_layer = torch.empty_like(q)

        if cu_seqlens is None:
            cu_seqlens = torch.arange(0, (bsz + 1) * q_len,
                                      step=q_len,
                                      dtype=torch.int32,
                                      device=query.device)

        cu_seqlens = torch.diff(cu_seqlens).to("cpu")

        # operator requires pta version >= 2.5.1
        torch_npu._npu_flash_attention_unpad(
            query=q,
            key=k,
            value=v,
            seq_len=cu_seqlens,
            scale_value=self.head_size**-0.5,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            out=context_layer,
        )

        if enable_pad:
            context_layer = context_layer[..., :origin_shape]

        if is_reshaped:
            context_layer = einops.rearrange(context_layer,
                                             "(b s) h d -> b s h d",
                                             b=bsz).contiguous()
        else:
            context_layer = einops.rearrange(context_layer,
                                             "(b s) h d -> b s (h d)",
                                             b=bsz).contiguous()
        return context_layer
