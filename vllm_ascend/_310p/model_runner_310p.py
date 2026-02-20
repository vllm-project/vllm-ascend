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

import torch
import torch_npu
from vllm.logger import logger
from vllm.utils.torch_utils import get_dtype_size
from vllm.v1.kv_cache_interface import AttentionSpec, KVCacheConfig, MambaSpec

from vllm_ascend.utils import ACL_FORMAT_FRACTAL_NZ
from vllm_ascend.worker.model_runner_v1 import NPUModelRunner


class NPUModelRunner310(NPUModelRunner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._acl_format = ACL_FORMAT_FRACTAL_NZ

    def initialize_kv_cache_tensors(
        self,
        kv_cache_config: KVCacheConfig,
    ) -> dict[str, torch.Tensor]:
        """
        Initialize the memory buffer for KV cache.

        Args:
            kv_cache_config: The KV cache config
        Returns:
            Dict[str, torch.Tensor]: A map between layer names to their
            corresponding memory buffer for KV cache.
        """
        # 310P limitation: KV transfer is not supported.
        if self.vllm_config.kv_transfer_config is not None:
            raise ValueError("KV cache transfer is not supported for 310P.")
        if self.use_sparse:
            raise ValueError("Deepseek Sparse Attention is not supported for 310P.")
        if self.model_config.use_mla:
            raise ValueError("MLAAttention is not supported for 310P.")

        kv_cache_raw_tensors = self._allocate_kv_cache_tensors(kv_cache_config)
        kv_caches = self._reshape_kv_cache_tensors(kv_cache_config, kv_cache_raw_tensors)

        # Set up cross-layer KV cache sharing.
        for layer_name, target_layer_name in self.shared_kv_cache_layers.items():
            logger.debug("%s reuses KV cache of %s", layer_name, target_layer_name)
            kv_caches[layer_name] = kv_caches[target_layer_name]

        from vllm.v1.worker.utils import bind_kv_cache

        bind_kv_cache(kv_caches, self.compilation_config.static_forward_context, self.kv_caches)
        return kv_caches

    def _allocate_kv_cache_tensors(self, kv_cache_config: KVCacheConfig) -> dict[str, torch.Tensor]:
        """
        Allocate raw KV buffers for Mamba layers only.

        Attention cache on 310P must use FRACTAL_NZ memory format and will be
        allocated in `_reshape_kv_cache_tensors`.
        """
        mamba_layers = set()
        for group in kv_cache_config.kv_cache_groups:
            if isinstance(group.kv_cache_spec, MambaSpec):
                for layer_name in group.layer_names:
                    if layer_name not in self.runner_only_attn_layers:
                        mamba_layers.add(layer_name)

        kv_cache_raw_tensors: dict[str, torch.Tensor] = {}
        for kv_cache_tensor in kv_cache_config.kv_cache_tensors:
            shared_mamba_layers = [layer_name for layer_name in kv_cache_tensor.shared_by if layer_name in mamba_layers]
            if not shared_mamba_layers:
                continue
            tensor = torch.zeros(kv_cache_tensor.size, dtype=torch.int8, device=self.device)
            for layer_name in shared_mamba_layers:
                kv_cache_raw_tensors[layer_name] = tensor

        assert mamba_layers == set(kv_cache_raw_tensors.keys()), "Some mamba layers are not correctly initialized"

        return kv_cache_raw_tensors

    def _reshape_kv_cache_tensors(
        self,
        kv_cache_config: KVCacheConfig,
        kv_cache_raw_tensors: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """
        Reshape raw KV cache tensors to layer-specific views.
        """
        layer_to_tensor_size: dict[str, int] = {}
        layer_to_shared_key: dict[str, tuple[str, ...]] = {}
        for kv_cache_tensor in kv_cache_config.kv_cache_tensors:
            shared_key = tuple(sorted(kv_cache_tensor.shared_by))
            for layer_name in kv_cache_tensor.shared_by:
                layer_to_tensor_size[layer_name] = kv_cache_tensor.size
                layer_to_shared_key[layer_name] = shared_key

        attention_cache_by_shared_key: dict[tuple[str, ...], tuple[torch.Tensor, torch.Tensor]] = {}
        kv_caches: dict[str, torch.Tensor] = {}
        for group in self._kv_cache_spec_attn_group_iterator():
            kv_cache_spec = group.kv_cache_spec
            attn_backend = group.backend
            for layer_name in group.layer_names:
                if layer_name in self.runner_only_attn_layers:
                    continue

                if isinstance(kv_cache_spec, AttentionSpec):
                    tensor_size = layer_to_tensor_size[layer_name]
                    assert tensor_size % kv_cache_spec.page_size_bytes == 0
                    num_blocks = tensor_size // kv_cache_spec.page_size_bytes
                    assert num_blocks >= kv_cache_config.num_blocks

                    if hasattr(attn_backend, "get_supported_kernel_block_sizes") and self.use_hybrid_blocks:
                        block_size = attn_backend.get_supported_kernel_block_sizes()[0]
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

                    shared_key = layer_to_shared_key[layer_name]
                    if shared_key in attention_cache_by_shared_key:
                        kv_caches[layer_name] = attention_cache_by_shared_key[shared_key]
                        continue

                    dtype = kv_cache_spec.dtype
                    cache_shape = kv_cache_shape[1:]
                    k_cache = torch_npu.empty_with_format(
                        size=cache_shape,
                        dtype=dtype,
                        device=self.device,
                        acl_format=self._acl_format,
                    )
                    v_cache = torch_npu.empty_with_format(
                        size=cache_shape,
                        dtype=dtype,
                        device=self.device,
                        acl_format=self._acl_format,
                    )
                    attention_cache_by_shared_key[shared_key] = (k_cache, v_cache)
                    kv_caches[layer_name] = (k_cache, v_cache)
                elif isinstance(kv_cache_spec, MambaSpec):
                    raw_tensor = kv_cache_raw_tensors[layer_name]
                    assert raw_tensor.numel() % kv_cache_spec.page_size_bytes == 0
                    num_blocks = raw_tensor.numel() // kv_cache_spec.page_size_bytes
                    assert num_blocks >= kv_cache_config.num_blocks

                    state_tensors = []
                    storage_offset_bytes = 0
                    for shape, dtype in zip(kv_cache_spec.shapes, kv_cache_spec.dtypes):
                        dtype_size = get_dtype_size(dtype)
                        num_element_per_page = kv_cache_spec.page_size_bytes // dtype_size
                        target_shape = (num_blocks, *shape)
                        stride = torch.empty(target_shape).stride()
                        target_stride = (num_element_per_page, *stride[1:])
                        assert storage_offset_bytes % dtype_size == 0
                        tensor = torch.as_strided(
                            raw_tensor.view(dtype),
                            size=target_shape,
                            stride=target_stride,
                            storage_offset=storage_offset_bytes // dtype_size,
                        )
                        state_tensors.append(tensor)
                        storage_offset_bytes += stride[0] * dtype_size
                    kv_caches[layer_name] = state_tensors
                else:
                    raise ValueError("Unknown KV cache spec type.")

        expected_layers = set()
        for group in kv_cache_config.kv_cache_groups:
            for layer_name in group.layer_names:
                if layer_name not in self.runner_only_attn_layers:
                    expected_layers.add(layer_name)
        assert expected_layers == set(kv_caches.keys()), "Some layers are not correctly initialized"

        return kv_caches

