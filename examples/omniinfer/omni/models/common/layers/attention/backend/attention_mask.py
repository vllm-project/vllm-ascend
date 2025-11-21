# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

from typing import Optional

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


class AttentionMaskBuilder:

    def __init__(self, attn_mask: torch.Tensor):
        self.attn_mask_cache = attn_mask
        self._seq_len_cached = attn_mask.shape[0]
        self.splitfuse_mask_value = -10000

    @classmethod
    def initialize_from_len(cls,
                            max_seq_len: int,
                            dtype: torch.dtype = torch.float16,
                            mask_value: Optional[int] = None):
        return cls(generate_attn_mask(max_seq_len, dtype, mask_value))

    def update_attn_cache(self, seqlen: int, dtype: torch.dtype,
                          device: torch.device):
        if seqlen > self._seq_len_cached or self.attn_mask_cache.dtype != dtype:
            self._seq_len_cached = seqlen
            self.attn_mask_cache = generate_attn_mask(seqlen, dtype)
        if self.attn_mask_cache.device != device:
            self.attn_mask_cache = self.attn_mask_cache.to(device)

    def get_decode_attn_mask(
        self,
        input_lengths: torch.tensor,
        max_s: int,
        dtype: torch.dtype,
        device: torch.device,
    ):
        self.update_attn_cache(max_s, dtype, device)
        return (self.attn_mask_cache.index_select(
            0, input_lengths)[:, :max_s].view(-1, 1, max_s).contiguous())

    def get_attn_mask(self, max_seq_len: int, dtype: torch.dtype,
                      device: torch.device):
        self.update_attn_cache(max_seq_len, dtype, device)
        return self.attn_mask_cache[:max_seq_len, :max_seq_len].contiguous()

    def get_splitfuse_attn_mask(
        self,
        seq_lens,
        query_lens,
        position,
        dtype,
        device,
    ) -> torch.Tensor:
        max_seq_len = max(seq_lens, default=0)
        if max_seq_len <= self._seq_len_cached:
            self.update_attn_cache(max_seq_len, dtype, device)
            if self.attn_mask_cache.numel(
            ) > 1 and self.attn_mask_cache[0][1] > 0:
                attn_mask = self.get_attn_mask(  # type: ignore
                    max_seq_len, dtype, device)
                attn_mask *= -10000
            else:
                attn_mask = self.attn_mask_cache
            return torch.index_select(attn_mask, dim=0,
                                      index=position)[:, :max_seq_len]
        total_q_len = sum(query_lens)
        attn_mask = torch.zeros((total_q_len, max_seq_len),
                                dtype=dtype,
                                device="cpu")

        current_row = 0
        for i in range(len(query_lens)):
            seq_len = seq_lens[i]
            q_len = query_lens[i]
            context_len = seq_len - q_len

            if context_len < 0:
                raise ValueError("context_len must be non-negative")
            attn_mask[current_row:current_row + q_len,
                      context_len:] = self.splitfuse_mask_value
            right_tensor = attn_mask[current_row:current_row + q_len,
                                     context_len:seq_len]
            right_tensor.masked_fill_(
                right_tensor.tril() == self.splitfuse_mask_value, 0)
            current_row += q_len

        return attn_mask.to(device, non_blocking=True)


def generate_attn_mask(max_seq_len: int, dtype=torch.float16, mask_value=None):
    # Construct lower triangle matrix.
    mask_flag = torch.tril(
        torch.ones((max_seq_len, max_seq_len),
                   dtype=torch.bool)).view(max_seq_len, max_seq_len)
    # Create upper triangle matrix used to mark mask positions.
    mask_flag = ~mask_flag
    # Currently for fp16 dtype, the mask value should be set to -inf.
    if mask_value is None:
        if dtype == torch.float16:
            mask_value = torch.finfo(torch.float32).min
        else:
            mask_value = 1
    attn_mask = torch.masked_fill(torch.zeros(size=(max_seq_len, max_seq_len)),
                                  mask_flag, mask_value).to(dtype)
    return attn_mask