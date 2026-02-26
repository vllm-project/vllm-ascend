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
from vllm.model_executor.layers.attention.mm_encoder_attention import MMEncoderAttention  # type: ignore

from vllm_ascend.utils import vllm_version_is

MIN_PAD_SIZE: int = 64  # min_size to pad weight
MAX_PAD_SIZE: int = 128  # max_size to pad weight

# Use seq_lens CPU cache to avoid frequent d2h copy.
# AscendMMEncoderAttention will copy the cu_seqlens from NPU to CPU in every
# forward, since the op _npu_flash_attention_unpad() requires CPU cu_seqlens
# (otherwise it will break down).
# Thus, we use seq_lens_cpu_cache to cache this tensor, since it's shared
# between all layers, but may change in different forward step. When the
# current layer_index is 0, we update the cache, otherwise we directly use the
# cache to avoid frequent diff and copy operations, which are costful.
seq_lens_cpu_cache: torch.Tensor = None


class AscendMMEncoderAttention(MMEncoderAttention):
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float | None = None,
        num_kv_heads: int | None = None,
        prefix: str = "",
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
        )

        if not vllm_version_is("0.15.0"):
            self.layer_index = int("".join(filter(str.isdigit, prefix)))

        self.enable_pad = self.head_size > MIN_PAD_SIZE and self.head_size < MAX_PAD_SIZE
        self.scale_value = self.head_size**-0.5

    def _reshape_qkv_to_3d(
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
        max_seqlen: torch.Tensor | None = None,  # Only used for Flash Attention
    ):
        bsz, q_len = query.size()[:2]
        kv_len = key.size(1)
        is_reshaped = query.dim() == 4

        if vllm_version_is("0.15.0"):
            if cu_seqlens is None:
                cu_seqlens = torch.arange(0, (bsz + 1) * q_len, step=q_len, dtype=torch.int32, device="cpu")
            seq_lens_cpu = torch.diff(cu_seqlens).to("cpu")
        else:
            global seq_lens_cpu_cache
            if self.layer_index == 0:
                if cu_seqlens is None:
                    cu_seqlens = torch.arange(0, (bsz + 1) * q_len, step=q_len, dtype=torch.int32, device="cpu")
                # Update seq_lens cpu cache.
                seq_lens_cpu_cache = torch.diff(cu_seqlens).to("cpu")
            # Directly use seq_lens cpu cache to avoid d2h copy.
            seq_lens_cpu = seq_lens_cpu_cache

        # q, k, v: [b, s, head, head_dim] -> [b * s, head, head_dim]
        q, k, v = self._reshape_qkv_to_3d(query, key, value, bsz, q_len, kv_len)

        if self.enable_pad:
            origin_shape = q.shape[-1]
            pad_len = MAX_PAD_SIZE - origin_shape
            # [b * s, head, head_dim] -> [b * s, head, MAX_PAD_SIZE]
            q = F.pad(q, (0, pad_len), mode="constant", value=0)
            k = F.pad(k, (0, pad_len), mode="constant", value=0)
            v = F.pad(v, (0, pad_len), mode="constant", value=0)

        context_layer = torch.empty_like(q)

        # operator requires pta version >= 2.5.1
        torch_npu._npu_flash_attention_unpad(
            query=q,
            key=k,
            value=v,
            seq_len=seq_lens_cpu,
            scale_value=self.scale_value,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            out=context_layer,
        )

        if self.enable_pad:
            context_layer = context_layer[..., :origin_shape]

        if is_reshaped:
            context_layer = einops.rearrange(context_layer, "(b s) h d -> b s h d", b=bsz).contiguous()
        else:
            context_layer = einops.rearrange(context_layer, "(b s) h d -> b s (h d)", b=bsz).contiguous()
        return context_layer
