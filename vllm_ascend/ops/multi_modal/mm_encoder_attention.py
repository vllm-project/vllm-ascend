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

import torch.nn.functional as F

from vllm.attention.layers.mm_encoder_attention import MMEncoderAttention


MIN_PAD_SIZE = 64  # min_size to pad weight
MAX_PAD_SIZE = 128  # max_size to pad weight


class AscendMMEncoderAttention(MMEncoderAttention):
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        projection_size: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__(
            embed_dim,
            num_heads,
            projection_size,
            quant_config,
            prefix,
        )

        self.embed_dim = embed_dim
        self.hidden_size_per_attention_head = dist_utils.divide(
            projection_size, num_heads)
        self.enable_pad = False
        self.finish_pad = False

        # TODO(shen-shanshan): Add verification for env vars (enable unpad).
        if self.hidden_size_per_attention_head > MIN_PAD_SIZE \
            and self.hidden_size_per_attention_head < MAX_PAD_SIZE:
            self.enable_pad = True
            self.origin_hidden_size_per_attention_head = self.hidden_size_per_attention_head
            self.half_origin_hidden_size_per_attention_head = self.hidden_size_per_attention_head // 2
            self.half_pad_hidden_size_per_attention_head = (
                MAX_PAD_SIZE - self.hidden_size_per_attention_head) // 2
            self.hidden_size_per_attention_head = MAX_PAD_SIZE

    def split_qkv(self, qkv: torch.Tensor) -> tuple[torch.Tensor, ...]:
        # [s, b, 3 * head * head_dim]
        seq_len, bs, _ = qkv.shape

        # [s, b, 3 * head * head_dim] -> 3 * [s, b, head * head_dim]
        q, k, v = qkv.chunk(3, dim=2)

        # 3 * [s, b, head * head_dim] -> 3 * [s, b, head, head_dim]
        new_shape = (seq_len, bs, self.num_attention_heads_per_partition,
                     self.hidden_size_per_attention_head)
        q, k, v = (x.view(*new_shape) for x in (q, k, v))
        return q, k, v

    def forward(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        # [s, b, c] --> [s, b, head * 3 * head_dim]
        x, _ = self.qkv(x)

        # [s, b, 3 * head * head_dim] -> 3 * [s, b, head, head_dim]
        q, k, v = self.split_qkv(x)
        batch_size = q.shape[1]

        q, k, v = (rearrange(x, "s b ... -> b s ...").contiguous()
                   for x in (q, k, v))
        q = torch_npu.npu_rotary_mul(q, cos, sin)
        k = torch_npu.npu_rotary_mul(k, cos, sin)

        q, k, v = [
            rearrange(x, "b s h d -> (b s) h d").contiguous()
            for x in (q, k, v)
        ]

        # TODO(shen-shanshan): use context manager.
        if self.enable_pad:
            origin_shape = q.shape[-1]
            pad_len = MAX_PAD_SIZE - origin_shape
            q = F.pad(q, (0, pad_len), mode="constant", value=0)
            k = F.pad(k, (0, pad_len), mode="constant", value=0)
            v = F.pad(v, (0, pad_len), mode="constant", value=0)

        context_layer = torch.empty_like(q)

        # operator requires pta version >= 2.5.1
        torch_npu._npu_flash_attention_unpad(
            query=q,
            key=k,
            value=v,
            seq_len=cu_seqlens,
            scale_value=self.origin_hidden_size_per_attention_head**-0.5,
            num_heads=self.num_attention_heads_per_partition,
            num_kv_heads=self.num_attention_heads_per_partition,
            out=context_layer)

        # TODO(shen-shanshan): use context manager.
        if self.enable_pad:
            context_layer = context_layer[..., :origin_shape]

        context_layer = rearrange(context_layer,
                                  "(b s) h d -> s b (h d)",
                                  b=batch_size).contiguous()

        output, _ = self.proj(context_layer)
        return output
