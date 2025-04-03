#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
# Adapted from vllm-project/vllm/vllm/worker/model_runner.py
# Copyright 2023 The vLLM team.
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

from typing import List, Tuple

import torch
from vllm.utils import is_pin_memory_available
from vllm.worker.cache_engine import CacheEngine

from vllm_ascend.utils import VLLM_ENABLE_GRAPH_MODE


def allocate_kv_cache(
    self,
    num_blocks: int,
    device: str,
) -> List[Tuple]:
    """Allocates KV cache on the specified device."""
    kv_cache_shape = self.attn_backend.get_kv_cache_shape(
        num_blocks, self.block_size, self.num_kv_heads, self.head_size)
    pin_memory = is_pin_memory_available() if device == "cpu" else False
    kv_cache: List[Tuple] = []

    # Align entries so they are 256 byte aligned for better performance
    # Primarily targets MLA as this typically only ends up having entries
    # be 128 byte aligned.
    alloc_shape = kv_cache_shape

    for _ in range(self.num_attention_layers):
        # null block in CpuGpuBlockAllocator requires at least that
        # block to be zeroed-out.
        # We zero-out everything for simplicity.
        layer_kv_cache_nope = torch.zeros(
            alloc_shape[:-1] +
            (self.model_config.hf_text_config.kv_lora_rank, ),
            dtype=self.dtype,
            pin_memory=pin_memory,
            device=device)
        layer_kv_cache_pe = torch.zeros(
            alloc_shape[:-1] +
            (self.model_config.hf_text_config.qk_rope_head_dim, ),
            dtype=self.dtype,
            pin_memory=pin_memory,
            device=device)

        # view back to (TOTAL_PAGES, PAGE_SIZE, entry_shape...) for cases
        # when entry_shape is higher than 1D
        kv_cache.append((layer_kv_cache_nope, layer_kv_cache_pe))
    return kv_cache


if VLLM_ENABLE_GRAPH_MODE == '1':
    CacheEngine._allocate_kv_cache = allocate_kv_cache