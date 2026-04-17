# Adapt from https://github.com/vllm-project/vllm/blob/main/vllm/v1/worker/gpu/attn_utils.py
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
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
#
import torch
import vllm
from vllm.config import get_current_vllm_config
from vllm.v1.kv_cache_interface import KVCacheConfig

def _align_memory(tensor: torch.Tensor, alignment: int) -> torch.Tensor:
        data_ptr = tensor.data_ptr()
        aligned_addr = (data_ptr + alignment - 1) // alignment * alignment
        offset = (aligned_addr - data_ptr) // tensor.element_size()
        return tensor[int(offset) :]

def ascend_allocate_kv_cache(kv_cache_config: KVCacheConfig, device: torch.device):
    kv_cache_raw_tensors: dict[str, torch.Tensor] = {}
    for kv_cache_tensor in kv_cache_config.kv_cache_tensors:
        if get_current_vllm_config().kv_transfer_config is None:
            tensor = torch.zeros(kv_cache_tensor.size, dtype=torch.int8, device=device)
        else:
            alignment = 2 *1024 * 1024
            tensor_size = kv_cache_tensor.size
            tensor = torch.zeros(tensor_size + alignment, dtype=torch.int8, device=device)
            tensor = _align_memory(tensor, alignment)[:tensor_size]
            assert (tensor.data_ptr() % alignment == 0), "When using the HCCS link for transmission on the NPU, "\
            "the starting address of the KV cache tensor is required to be 2M-aligned"

        for layer_name in kv_cache_tensor.shared_by:
            kv_cache_raw_tensors[layer_name] = tensor

    layer_names = set()
    for group in kv_cache_config.kv_cache_groups:
        for layer_name in group.layer_names:
            layer_names.add(layer_name)
    assert layer_names == set(kv_cache_raw_tensors.keys()), (
        "Some layers are not correctly initialized"
    )
    return kv_cache_raw_tensors

vllm.v1.worker.gpu.attn_utils._allocate_kv_cache = ascend_allocate_kv_cache