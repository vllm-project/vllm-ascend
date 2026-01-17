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

from collections.abc import Callable
from typing import Any

import torch
import torch_npu

import vllm_ascend.attention.attention_mask as _base_mask
from vllm_ascend.utils import ACL_FORMAT_FRACTAL_NZ, nd_to_nz_spec

_BASE_BUILDER: Callable[[torch.device], Any] = _base_mask.AttentionMaskBuilder


def _gen_causal_additive_mask_fp16(max_seq_len: int, device: torch.device) -> torch.Tensor:
    tril = torch.ones((max_seq_len, max_seq_len), dtype=torch.bool, device=device).tril_()
    upper = ~tril
    m = torch.zeros((max_seq_len, max_seq_len), dtype=torch.float16, device=device)
    m.masked_fill_(upper, float("-inf"))
    return m


def build_splitfuse_attn_mask_310p(attn_metadata, device, *, full_mask_cache=None, full_mask_cache_len=0):
    qsl = attn_metadata.query_start_loc.detach().to("cpu", dtype=torch.int32)
    qlens = qsl[1:] - qsl[:-1]

    context_lens = attn_metadata.seq_lens.to(dtype=torch.int32)
    L = int(context_lens.max().item())

    q_list = qlens.tolist()
    c_list = context_lens.detach().to("cpu", dtype=torch.int64).tolist()
    pos_list = [p for ql, cl in zip(q_list, c_list) for p in range(cl - ql, cl)]
    position = torch.tensor(pos_list, dtype=torch.long, device=device)

    if full_mask_cache is None or full_mask_cache.device != device or full_mask_cache_len < L:
        tril = torch.ones((L, L), dtype=torch.bool, device=device).tril_()
        full = torch.zeros((L, L), dtype=torch.float16, device=device)
        full.masked_fill_(~tril, float("-inf"))
        full_mask_cache, full_mask_cache_len = full, L
    else:
        full = full_mask_cache[:L, :L].contiguous()

    rows = full.index_select(0, position).contiguous()
    mask = torch_npu.npu_format_cast(nd_to_nz_spec(rows).contiguous(), ACL_FORMAT_FRACTAL_NZ)
    return mask, full_mask_cache, full_mask_cache_len


class _AttentionMaskBuilder310P:
    """
    310P adapter:
      - overrides fp16 causal additive mask generation (use -inf fp16)
      - delegates all other behaviors to base AttentionMaskBuilder
      - pooling runner_type is NOT supported on 310P (explicit)
    """

    def __init__(self, device: torch.device):
        self._base = _BASE_BUILDER(device)

        self._fp16_mask_cache: torch.Tensor | None = None
        self._fp16_mask_cached_len: int = 0

    def __getattr__(self, name: str) -> Any:
        return getattr(self._base, name)

    @property
    def device(self) -> torch.device:
        return self._base.device

    def _get_fp16_mask(self, max_seq_len: int) -> torch.Tensor:
        if self._fp16_mask_cache is None or max_seq_len > self._fp16_mask_cached_len:
            self._fp16_mask_cache = _gen_causal_additive_mask_fp16(max_seq_len, self.device)
            self._fp16_mask_cached_len = max_seq_len
        assert self._fp16_mask_cache is not None
        return self._fp16_mask_cache[:max_seq_len, :max_seq_len].contiguous()

    def get_attention_mask(self, model_config) -> torch.Tensor:
        if getattr(model_config, "runner_type", None) == "pooling":
            raise NotImplementedError("310P does not support runner_type='pooling'")
        return self._get_fp16_mask(2048)


def AttentionMaskBuilder(device: torch.device) -> _AttentionMaskBuilder310P:
    return _AttentionMaskBuilder310P(device)
