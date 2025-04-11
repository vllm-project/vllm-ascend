# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
# Adapted from vllm/tests/kernels/test_moe.py
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

import torch
import torch_npu

def concat_and_cache_mla(
    kv_c_normed: torch.Tensor,  # [num_tokens, num_kv_head, nope]
    k_pe: torch.Tensor,         # [num_tokens, num_kv_head, rope]
    kv_cache: torch.Tensor,     # [num_blocks, block_size, num_kv_head, nope + rope]
    slot_mapping,               # [num_tokens]
):
  num_blocks = kv_cache.size()[0]
  block_size = kv_cache.size()[1]
  num_kv_head = k_pe.size()[1]

  idx_for_copy = slot_mapping // block_size * block_size + slot_mapping % block_size
  kv_cache = kv_cache.view(num_blocks * block_size, num_kv_head, -1)
  kv_cache[idx_for_copy] = torch.cat([kv_c_normed.unsqueeze(1), k_pe], dim=-1)