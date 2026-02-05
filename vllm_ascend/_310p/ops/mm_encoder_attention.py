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

import torch
import torch_npu

from vllm_ascend.ops.mm_encoder_attention import AscendMMEncoderAttention


class AscendMMEncoderAttention310(AscendMMEncoderAttention):
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
        query = query.view(bsz * q_len, self.num_heads, self.head_size)
        key = key.view(bsz * kv_len, self.num_kv_heads, self.head_size)
        value = value.view(bsz * kv_len, self.num_kv_heads, self.head_size)

        if cu_seqlens is None:
            seq_len = torch.tensor([q_len for _ in range(bsz)])

        seq_len = torch.diff(cu_seqlens.to("cpu", dtype=torch.int32))

        output = torch.empty_like(query)
        torch_npu._npu_flash_attention_unpad(
            query=query,
            key=key,
            value=value,
            seq_len=seq_len,
            scale_value=self.head_size**-0.5,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            out=output,
        )

        output = output.view(bsz, -1, self.num_heads, self.head_size)
        return output
