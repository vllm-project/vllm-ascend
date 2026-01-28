#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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
# This file is a part of the vllm-ascend project.
#

import einops
import torch
import torch_npu

from vllm_ascend.ops.mm_encoder_attention import AscendMMEncoderAttention as _Base


class AscendMMEncoderAttention310(_Base):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward_oot(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: int | None = None,
        **kwargs,
    ):
        bsz, q_len = query.size()[:2]
        kv_len = key.size(1)

        q, k, v = self.reshape_qkv_to_3d(query, key, value, bsz, q_len, kv_len)

        if cu_seqlens is None:
            cu_seqlens = torch.arange(
                0,
                (bsz + 1) * q_len,
                step=q_len,
                dtype=torch.int32,
                device=query.device,
            )

        seq_len = torch.diff(cu_seqlens).to("cpu", dtype=torch.int32)

        context_layer = torch.empty_like(q)
        torch_npu._npu_flash_attention_unpad(
            query=q,
            key=k,
            value=v,
            seq_len=seq_len,
            scale_value=self.head_size**-0.5,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            out=context_layer,
        )

        context_layer = einops.rearrange(context_layer, "(b s) h d -> b s h d", b=bsz).contiguous()
        return context_layer
