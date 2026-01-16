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

from __future__ import annotations

from typing import Any

import torch
import torch_npu
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig
from vllm.v1.worker.utils import bind_kv_cache

from vllm_ascend.utils import ACL_FORMAT_FRACTAL_NZ
from vllm_ascend.worker.model_runner_v1 import NPUModelRunner


class NPUModelRunner310(NPUModelRunner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._acl_format = ACL_FORMAT_FRACTAL_NZ

    def _initialize_kv_cache_tensors_310p(self, kv_cache_config: KVCacheConfig) -> dict[str, Any]:
        if self.vllm_config.kv_transfer_config is not None:
            raise ValueError("KV cache transfer is not supported for 310P.")

        kv_cache_sizes: dict[str, int] = {}
        for kv_cache_tensor in kv_cache_config.kv_cache_tensors:
            assert len(kv_cache_tensor.shared_by) == 1, (
                "KV cache tensor shared by multiple layers is not supported in 310P."
            )
            kv_cache_sizes[kv_cache_tensor.shared_by[0]] = kv_cache_tensor.size

        kv_caches: dict[str, Any] = {}

        for group in self._kv_cache_spec_attn_group_iterator():
            kv_cache_spec = group.kv_cache_spec
            attn_backend = group.backend

            if not isinstance(kv_cache_spec, FullAttentionSpec):
                raise ValueError("Unknown KV cache spec type.")

            for layer_name in group.layer_names:
                if layer_name in self.runner_only_attn_layers:
                    continue

                tensor_size = kv_cache_sizes[layer_name]
                assert tensor_size % kv_cache_spec.page_size_bytes == 0
                num_blocks = tensor_size // kv_cache_spec.page_size_bytes
                assert num_blocks >= kv_cache_config.num_blocks

                if hasattr(attn_backend, "get_supported_block_size") and self.use_hybrid_blocks:
                    block_size = attn_backend.get_supported_block_size()[0]
                    block_size_chunk = kv_cache_spec.block_size // block_size
                    kv_cache_shape = attn_backend.get_kv_cache_shape(
                        num_blocks * block_size_chunk,
                        block_size,
                        kv_cache_spec.num_kv_heads,
                        kv_cache_spec.head_size,
                    )
                else:
                    kv_cache_shape = attn_backend.get_kv_cache_shape(
                        num_blocks,
                        kv_cache_spec.block_size,
                        kv_cache_spec.num_kv_heads,
                        kv_cache_spec.head_size,
                    )

                dtype = kv_cache_spec.dtype

                if "attn" in layer_name:
                    k_tensor = torch.zeros(kv_cache_shape[1:], dtype=dtype, device=self.device)
                    v_tensor = torch.zeros(kv_cache_shape[1:], dtype=dtype, device=self.device)
                    k_cache = torch_npu.npu_format_cast(k_tensor, self._acl_format)
                    v_cache = torch_npu.npu_format_cast(v_tensor, self._acl_format)
                    kv_caches[layer_name] = (k_cache, v_cache)

        bind_kv_cache(
            kv_caches,
            self.compilation_config.static_forward_context,
            self.kv_caches,
            1,  # 310p devices donnot support: hf_config.model_type == "longcat_flash"
        )
        return kv_caches

    def initialize_kv_cache_tensors(self, kv_cache_config: KVCacheConfig) -> dict[str, Any]:
        return self._initialize_kv_cache_tensors_310p(kv_cache_config)
