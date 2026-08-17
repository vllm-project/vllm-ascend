#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import torch

from vllm_ascend.device.device_op import DeviceOperator
from vllm_ascend.patch.dots3_note_audio import Dots3NoteAudioAttentionBackend


class AscendDots3NoteAudioAttentionBackend(Dots3NoteAudioAttentionBackend):
    def forward_oot(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        cu_seqlens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, query_length = query.shape[:2]
        if key.shape[1] != query_length or value.shape[1] != query_length:
            raise ValueError("Dots3 Note audio attention requires self-attention")
        if cu_seqlens is None:
            seq_lens_cpu = torch.full((batch_size,), query_length, dtype=torch.int32)
        else:
            seq_lens_cpu = torch.diff(cu_seqlens).to("cpu", dtype=torch.int32)

        output = DeviceOperator.npu_flash_attention(
            query=query.reshape(-1, self.num_heads, self.head_size),
            key=key.reshape(-1, self.num_kv_heads, self.head_size),
            value=value.reshape(-1, self.num_kv_heads, self.head_size),
            seq_lens_cpu=seq_lens_cpu,
            head_num=self.num_heads,
            scale_value=self.scale,
            num_kv_heads=self.num_kv_heads,
        )
        return output.reshape(
            batch_size,
            query_length,
            self.num_heads,
            self.head_size,
        )
