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
import torch


def _generate_attn_mask(max_seq_len, dtype):
    # Construct lower triangle matrix.
    mask_flag = torch.tril(
        torch.ones((max_seq_len, max_seq_len),
                   dtype=torch.bool)).view(max_seq_len, max_seq_len)
    # Create upper triangle matrix used to mark mask positions.
    mask_flag = ~mask_flag
    # Currently for fp16 dtype, the mask value should be set to -inf.
    # TODO: Eliminate this part in the future.
    if dtype == torch.float16:
        mask_value = torch.finfo(torch.float32).min
    else:
        mask_value = 1
    attn_mask = torch.masked_fill(torch.zeros(size=(max_seq_len, max_seq_len)),
                                  mask_flag, mask_value).to(dtype)
    return attn_mask


class AttentionMaskBuilder:

    def __init__(
        self,
        max_seq_len: int,
        dtype: torch.dtype,
    ):
        attn_mask = _generate_attn_mask(max_seq_len, dtype)

        self._seq_len_cached = attn_mask.shape[0]
        self.attn_mask_cache = attn_mask

    @staticmethod
    def get_mask_scale_factor(dtype: torch.dtype = torch.float16):
        if dtype == torch.float16:
            mask_scale_factor = 1
        elif dtype == torch.bfloat16:
            mask_scale_factor = -10000
        else:
            raise ValueError(
                "The current operation now only supports data types: torch.float16 and "
                "torch.bfloat16. Please ensure the input is of one of these types."
            )
        return mask_scale_factor

    def get_attn_mask(self, max_seq_len: int, dtype: torch.dtype,
                      device: torch.device):
        self._update_attn_cache(max_seq_len, dtype)
        return self.attn_mask_cache[:max_seq_len, :max_seq_len].contiguous(
        ).to(device)

    def get_splitfuse_attn_mask(
        self,
        seq_lens: torch.Tensor,
        position: torch.Tensor,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        if dtype not in [torch.float16, torch.bfloat16]:
            raise ValueError(
                "splitfuse_attn_mask now only supports bf16 and fp16")
        max_seq_len = max(seq_lens, default=0)
        self._update_attn_cache(max_seq_len, dtype)
        # FIXME: Currently the mask value of chunked-prefill situation and Prefill-Only situation
        # is not the same. Fix this in the future when kernel is ready.
        mask_scale_factor = AttentionMaskBuilder.get_mask_scale_factor(dtype)
        attn_mask = torch.index_select(self.attn_mask_cache,
                                       dim=0,
                                       index=position)[:, :max_seq_len]
        attn_mask *= mask_scale_factor
        return attn_mask.contiguous().to(device, non_blocking=True)

    def _update_attn_cache(self, seqlen: int, dtype: torch.dtype):
        if seqlen > self._seq_len_cached:
            self._seq_len_cached = seqlen
            self.attn_mask_cache = _generate_attn_mask(seqlen, dtype)
        if self.attn_mask_cache.dtype != dtype:
            self.attn_mask_cache = self.attn_mask_cache.to(dtype)
