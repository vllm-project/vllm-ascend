#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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

from collections.abc import Iterable
from typing import Any

import torch
from vllm.v1.kv_cache_interface import FullAttentionSpec
from vllm.v1.worker.utils import AttentionGroup, KVBlockZeroer


class AscendKVBlockZeroer310(KVBlockZeroer):
    """310P KV block zeroer without Triton.

    Atlas 300I DUO does not support Triton. For MTP >= 2 hybrid models, newly
    allocated attention KV blocks must be zeroed via direct tensor writes.
    """

    def __init__(self, device: torch.device, pin_memory: bool) -> None:
        self.device = device
        self.pin_memory = pin_memory
        self._kv_tensors: list[torch.Tensor] = []
        self._logical_page_ratio: int = 1

    def init_meta(
        self,
        attn_groups_iter: Iterable["AttentionGroup"],
        kernel_block_sizes: list[list[int]],
        cache_dtype: str,
        runner_only_attn_layers: set[str],
        static_forward_context: dict[str, Any],
    ) -> None:
        seen_ptrs: set[int] = set()
        self._kv_tensors = []
        self._logical_page_ratio = 1

        for group in attn_groups_iter:
            spec = group.kv_cache_spec
            if not isinstance(spec, FullAttentionSpec):
                continue
            if group.kv_cache_group_id >= len(kernel_block_sizes):
                continue
            kernel_bs = kernel_block_sizes[group.kv_cache_group_id][0]
            ratio = spec.block_size // kernel_bs
            if not self._kv_tensors:
                self._logical_page_ratio = ratio

            for layer_name in group.layer_names:
                if layer_name in runner_only_attn_layers:
                    continue
                kv_tuple = static_forward_context[layer_name].kv_cache
                assert len(kv_tuple) == 2, "K and V are not stored separately"
                for kv in kv_tuple:
                    dp = kv.data_ptr()
                    if dp in seen_ptrs:
                        continue
                    seen_ptrs.add(dp)
                    self._kv_tensors.append(kv)

    def zero_block_ids(self, block_ids: list[int]) -> None:
        if not block_ids or not self._kv_tensors:
            return

        ratio = self._logical_page_ratio
        for block_id in block_ids:
            start = block_id * ratio
            end = start + ratio
            for kv in self._kv_tensors:
                kv[start:end].zero_()
