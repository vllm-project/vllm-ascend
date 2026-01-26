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

    def initialize_kv_cache_tensors(
        self,
        kv_cache_config: KVCacheConfig,
    ) -> dict[str, Any]:
        """
        Initialize KV cache tensors for 310P.

        1) allocate buffers
        2) reshape / transform to the final layout
        3) optional cross-layer sharing
        4) bind buffers to the static forward context
        """
        # 310P limitation: KV transfer is not supported.
        if self.vllm_config.kv_transfer_config is not None:
            raise ValueError("KV cache transfer is not supported for 310P.")

        kv_cache_raw_tensors = self._allocate_kv_cache_tensors_310p(kv_cache_config)
        kv_caches = self._reshape_kv_cache_tensors_310p(kv_cache_config, kv_cache_raw_tensors)

        # Keep the same cross-layer KV cache sharing logic as the main branch.
        # For 310P, this is expected to be empty in most cases, but keeping it
        # makes the code path consistent and easier to reason about.
        for layer_name, target_layer_name in self.shared_kv_cache_layers.items():
            kv_caches[layer_name] = kv_caches[target_layer_name]

        # 310P devices do not support the "longcat_flash" special case here, so always be "1".
        bind_kv_cache(
            kv_caches,
            self.compilation_config.static_forward_context,
            self.kv_caches,
            1,
        )
        return kv_caches

    def _allocate_kv_cache_tensors_310p(
        self,
        kv_cache_config: KVCacheConfig,
    ) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        """
        Allocate KV cache buffers for each attention layer.

        Unlike the non-310p path, 310P uses torch.zeros directly with the final dtype,
        and defers layout casting (ACL format) to the reshape step.
        """
        # Build a mapping: layer_name -> tensor_size(bytes).
        kv_cache_sizes: dict[str, int] = {}
        for kv_cache_tensor in kv_cache_config.kv_cache_tensors:
            # 310P limitation: a KV cache tensor must not be shared by multiple layers.
            assert len(kv_cache_tensor.shared_by) == 1, (
                "KV cache tensor shared by multiple layers is not supported in 310P."
            )
            kv_cache_sizes[kv_cache_tensor.shared_by[0]] = kv_cache_tensor.size

        kv_cache_raw_tensors: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

        for group in self._kv_cache_spec_attn_group_iterator():
            kv_cache_spec = group.kv_cache_spec
            attn_backend = group.backend

            if not isinstance(kv_cache_spec, FullAttentionSpec):
                raise ValueError("Unknown KV cache spec type.")

            for layer_name in group.layer_names:
                if layer_name in self.runner_only_attn_layers:
                    continue

                if "attn" not in layer_name:
                    continue

                # Compute how many blocks this layer can hold.
                tensor_size = kv_cache_sizes[layer_name]
                assert tensor_size % kv_cache_spec.page_size_bytes == 0
                num_blocks = tensor_size // kv_cache_spec.page_size_bytes

                # `num_blocks` must be >= the number KVCacheManager may allocate.
                assert num_blocks >= kv_cache_config.num_blocks

                # Determine the KV cache shape from backend.
                kv_cache_shape = self._get_kv_cache_shape_310p(
                    attn_backend=attn_backend,
                    kv_cache_spec=kv_cache_spec,
                    num_blocks=num_blocks,
                )

                shape = kv_cache_shape[1:]
                dtype = kv_cache_spec.dtype

                k_tensor = torch.zeros(shape, dtype=dtype, device=self.device)
                v_tensor = torch.zeros(shape, dtype=dtype, device=self.device)
                kv_cache_raw_tensors[layer_name] = (k_tensor, v_tensor)

        return kv_cache_raw_tensors

    def _reshape_kv_cache_tensors_310p(
        self,
        kv_cache_config: KVCacheConfig,
        kv_cache_raw_tensors: dict[str, tuple[torch.Tensor, torch.Tensor]],
    ) -> dict[str, Any]:
        """
        Transform allocated KV cache buffers into the final layout required by 310P.

        For 310P, this mainly means casting tensors into the expected ACL format.
        """
        kv_caches: dict[str, Any] = {}

        for group in self._kv_cache_spec_attn_group_iterator():
            kv_cache_spec = group.kv_cache_spec
            if not isinstance(kv_cache_spec, FullAttentionSpec):
                raise ValueError("Unknown KV cache spec type.")

            for layer_name in group.layer_names:
                if layer_name in self.runner_only_attn_layers:
                    continue
                if "attn" not in layer_name:
                    continue

                k_tensor, v_tensor = kv_cache_raw_tensors[layer_name]

                # In-place ACL layout cast to avoid the extra allocation of npu_format_cast,
                # which can spike peak memory (~2x KV cache) during initialization and trigger OOM.
                torch_npu.npu_format_cast_(k_tensor, self._acl_format)
                torch_npu.npu_format_cast_(v_tensor, self._acl_format)
                kv_caches[layer_name] = (k_tensor, v_tensor)

        return kv_caches

    def _get_kv_cache_shape_310p(
        self,
        attn_backend: Any,
        kv_cache_spec: FullAttentionSpec,
        num_blocks: int,
    ) -> tuple[int, ...]:
        """
        Compute KV cache shape with (optional) hybrid block support.
        """
        if hasattr(attn_backend, "get_supported_block_size") and self.use_hybrid_blocks:
            block_size = attn_backend.get_supported_block_size()[0]
            block_size_chunk = kv_cache_spec.block_size // block_size
            return attn_backend.get_kv_cache_shape(
                num_blocks * block_size_chunk,
                block_size,
                kv_cache_spec.num_kv_heads,
                kv_cache_spec.head_size,
            )

        return attn_backend.get_kv_cache_shape(
            num_blocks,
            kv_cache_spec.block_size,
            kv_cache_spec.num_kv_heads,
            kv_cache_spec.head_size,
        )
