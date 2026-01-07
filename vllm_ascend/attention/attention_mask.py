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
    mask_flag = torch.ones((max_seq_len, max_seq_len),
                           dtype=torch.bool).tril_()
    # Create upper triangle matrix used to mark mask positions.
    mask_flag = ~mask_flag
    # Currently for fp16 dtype, the mask value should be set to -inf.
    # TODO: Eliminate this part in the future.
    mask_value = float('-inf') if dtype == torch.float16 else 1
    attn_mask = torch.zeros(size=(max_seq_len, max_seq_len), dtype=dtype) \
        .masked_fill_(mask_flag, mask_value)
    return attn_mask


class AttentionMaskBuilder:

    def __init__(self, device: torch.device):
        self.attn_mask_cache = None
        self._seq_len_cached = 0
        self.device = device
        self.mla_mask = None
        self.chunked_prefill_attn_mask = None
        self.pcp_mla_mask = None
        self.swa_mask = None

    def get_attn_mask(self, max_seq_len: int, dtype: torch.dtype):
        if self.attn_mask_cache is None or max_seq_len > self._seq_len_cached:
            self.attn_mask_cache = _generate_attn_mask(max_seq_len, dtype)
            self._seq_len_cached = max_seq_len
        assert self.attn_mask_cache is not None, "Something is wrong in generate_attn_mask."
        if self.attn_mask_cache.dtype != dtype:
            self.attn_mask_cache = self.attn_mask_cache.to(dtype)
        return self.attn_mask_cache[:max_seq_len, :max_seq_len].contiguous(
        ).to(self.device, non_blocking=True)

    def get_splitfuse_attn_mask(
        self,
        seq_lens: torch.Tensor = None,
        position: torch.Tensor = None,
        dtype: torch.dtype = None,
        device: torch.device = None,
    ) -> torch.Tensor:

        from vllm_ascend.utils import get_ascend_device_type, AscendDeviceType

        target_device = device or getattr(self, "device", None) or torch.device("npu")

        if get_ascend_device_type() == AscendDeviceType._310P:
            target_dtype = dtype or torch.float16

            cache = getattr(self, "_splitfuse_attn_mask_cache", None)
            if cache is None:
                self._splitfuse_attn_mask_cache = {}
                cache = self._splitfuse_attn_mask_cache

            key = str(target_dtype)
            if key not in cache:
                max_seq_len = int(seq_lens.max().item()) if seq_lens is not None else 2048
                max_seq_len = min(max_seq_len, 2048)
                cache[key] = self.get_attn_mask(max_seq_len, target_dtype)
            return cache[key]

        if (self.chunked_prefill_attn_mask is None
                or self.chunked_prefill_attn_mask.device != target_device):
            self.chunked_prefill_attn_mask = torch.triu(
                torch.ones(2048,
                           2048), diagonal=1).to(torch.int8).to(self.device)
        return self.chunked_prefill_attn_mask

    def get_mla_mask(self, dtype: torch.dtype) -> torch.Tensor:
        if self.mla_mask is None or self.mla_mask.dtype != dtype:
            if dtype == torch.float16:
                mask_value = torch.finfo(torch.float32).min
            else:
                mask_value = 1
            prefill_mask = torch.triu(
                torch.ones(512, 512, device=self.device, dtype=dtype), 1)
            self.mla_mask = torch.where(prefill_mask == 1, mask_value,
                                        0).to(dtype)
        return self.mla_mask

    def get_pcp_mla_mask(self, dtype: torch.dtype):
        if self.pcp_mla_mask is None or self.pcp_mla_mask.dtype != dtype:
            self.pcp_mla_mask = torch.triu(
                torch.ones(512, 512, device=self.device, dtype=dtype), 1)
        return self.pcp_mla_mask

    def get_swa_mask(self, dtype: torch.dtype, sliding_window):
        if self.swa_mask is None or self.swa_mask.dtype != dtype:
            if sliding_window is not None:
                mask = torch.ones(2048, 2048, dtype=torch.bool)
                triu_mask = torch.triu(mask, diagonal=1).to(self.device)
                tril_mask = torch.tril(mask, -sliding_window).to(self.device)
                self.swa_mask = triu_mask + tril_mask
        return self.swa_mask