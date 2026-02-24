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

import os

import torch
import torch_npu
from vllm.logger import logger
from vllm.v1.kv_cache_interface import AttentionSpec, KVCacheConfig, MambaSpec

from vllm_ascend.utils import ACL_FORMAT_FRACTAL_NZ
from vllm_ascend.worker.model_runner_v1 import NPUModelRunner


class NPUModelRunner310(NPUModelRunner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._acl_format = ACL_FORMAT_FRACTAL_NZ
        # For MLA on 310P, ND format is safer for memory footprint; NZ may
        # inflate allocations due to padding/alignment.
        self._mla_kv_cache_format = os.getenv("VLLM_ASCEND_310P_MLA_KV_CACHE_FORMAT", "ND").upper()

    def initialize_kv_cache_tensors(self, kv_cache_config: KVCacheConfig) -> dict[str, torch.Tensor]:
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
        # MLA is now supported on 310P via software implementation.

        # Initialize the memory size for KV cache
        kv_cache_size = self._calculate_kv_cache_tensors_size(kv_cache_config)
        # Allocate and reshape KV cache Tensors
        kv_caches = self._allocate_kv_cache_and_reshape_tensors(kv_cache_config, kv_cache_size)
        # Set up cross-layer KV cache sharing
        for layer_name, target_layer_name in self.shared_kv_cache_layers.items():
            logger.debug("%s reuses KV cache of %s", layer_name, target_layer_name)
            kv_caches[layer_name] = kv_caches[target_layer_name]

        from vllm.v1.worker.utils import bind_kv_cache

        bind_kv_cache(kv_caches, self.compilation_config.static_forward_context, self.kv_caches)
        return kv_caches

    def _calculate_kv_cache_tensors_size(self, kv_cache_config: KVCacheConfig) -> dict[str, int | tuple[int, int]]:
        """
        Initializes the KV cache size. The buffer needs to be reshaped to the desired shape before being used by
        the models.

        Args:
            kv_cache_config: The KV cache config
        Returns:
            dict[str, int]: A map between layer names to their
            corresponding memory buffer size.
        """
        kv_cache_sizes: dict[str, int | tuple[int, int]] = {}
        for kv_cache_tensor in kv_cache_config.kv_cache_tensors:
            # TODO: REFACTOR ME to sharing hybrid cache
            for idx in range(len(kv_cache_tensor.shared_by)):
                layer_name = kv_cache_tensor.shared_by[idx]
                if "linear_attn" in layer_name and layer_name not in kv_cache_sizes:
                    kv_cache_size = kv_cache_tensor.size
                    for layer_name_inner in kv_cache_tensor.shared_by:
                        if "linear_attn" in layer_name_inner:
                            kv_cache_sizes[layer_name_inner] = kv_cache_size
                elif "attn" in layer_name and layer_name not in kv_cache_sizes:
                    if self.model_config.use_mla:
                        head_size = (
                            self.model_config.hf_text_config.qk_rope_head_dim
                            + self.model_config.hf_text_config.kv_lora_rank
                        )
                        k_tensor_split_factor = head_size / self.model_config.hf_text_config.kv_lora_rank
                        v_tensor_split_factor = head_size / self.model_config.hf_text_config.qk_rope_head_dim
                        kv_tensor_size = (
                            int(kv_cache_tensor.size // k_tensor_split_factor),
                            int(kv_cache_tensor.size // v_tensor_split_factor),
                        )
                    else:
                        kv_tensor_split_factor = 2
                        kv_tensor_size = int(kv_cache_tensor.size // kv_tensor_split_factor)
                    for layer_name_inner in kv_cache_tensor.shared_by:
                        if "attn" in layer_name_inner and "linear_attn" not in layer_name_inner:
                            kv_cache_sizes[layer_name_inner] = kv_tensor_size

        layer_names = set()
        for group in kv_cache_config.kv_cache_groups:
            for layer_name in group.layer_names:
                if layer_name in self.runner_only_attn_layers:
                    continue
                layer_names.add(layer_name)
        assert layer_names == set(kv_cache_sizes.keys()), "Some layers are not correctly initialized"

        return kv_cache_sizes

    def _allocate_kv_cache_and_reshape_tensors(
        self,
        kv_cache_config: KVCacheConfig,
        kv_cache_sizes: dict[str, int | tuple[int, int]],
    ) -> dict[str, torch.Tensor]:
        """
        Allocate the KV cache tensors to the desired shape and dtype.

        Args:
            kv_cache_config: The KV cache config
            kv_cache_sizes: The KV cache size of each layer
        Returns:
            dict[str, torch.Tensor]: A map between layer names to their
            corresponding memory buffer for KV cache.
        """
        kv_caches: dict[str, torch.Tensor] = {}
        for group in self._kv_cache_spec_attn_group_iterator():
            kv_cache_spec = group.kv_cache_spec
            attn_backend = group.backend
            for layer_name in group.layer_names:
                if layer_name in self.runner_only_attn_layers:
                    continue
                if isinstance(kv_cache_spec, AttentionSpec):
                    kv_tensor_size = kv_cache_sizes[layer_name]
                    assert kv_tensor_size is not None
                    if isinstance(kv_tensor_size, tuple):
                        k_tensor_size, v_tensor_size = kv_tensor_size
                    else:
                        k_tensor_size = kv_tensor_size
                        v_tensor_size = kv_tensor_size
                    sum_page_size_bytes = k_tensor_size + v_tensor_size
                    assert sum_page_size_bytes % kv_cache_spec.page_size_bytes == 0
                    num_blocks = sum_page_size_bytes // kv_cache_spec.page_size_bytes
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
                    dtype = kv_cache_spec.dtype
                    if self.model_config.use_mla:
                        mla_num_blocks, mla_block_size, num_kv_heads, _ = kv_cache_shape
                        k_shape = [
                            mla_num_blocks,
                            mla_block_size,
                            num_kv_heads,
                            self.model_config.hf_text_config.kv_lora_rank,
                        ]
                        v_shape = [
                            mla_num_blocks,
                            mla_block_size,
                            num_kv_heads,
                            self.model_config.hf_text_config.qk_rope_head_dim,
                        ]
                    else:
                        k_shape = kv_cache_shape[1:]
                        v_shape = k_shape

                    if self.model_config.use_mla and self._mla_kv_cache_format == "ND":
                        k_cache = torch.empty(size=k_shape, dtype=dtype, device=self.device)
                        v_cache = torch.empty(size=v_shape, dtype=dtype, device=self.device)
                    elif self.model_config.use_mla and self._mla_kv_cache_format == "ND_KPE_ONLY":
                        k_cache = torch_npu.empty_with_format(
                            size=k_shape,
                            dtype=dtype,
                            device=self.device,
                            acl_format=self._acl_format,
                        )
                        v_cache = torch.empty(size=v_shape, dtype=dtype, device=self.device)
                    else:
                        k_cache = torch_npu.empty_with_format(
                            size=k_shape,
                            dtype=dtype,
                            device=self.device,
                            acl_format=self._acl_format,
                        )
                        v_cache = torch_npu.empty_with_format(
                            size=v_shape,
                            dtype=dtype,
                            device=self.device,
                            acl_format=self._acl_format,
                        )
                    kv_caches[layer_name] = (k_cache, v_cache)
                elif isinstance(kv_cache_spec, MambaSpec):
                    tensor_size = kv_cache_sizes[layer_name]
                    dtype = kv_cache_spec.dtype
                    tensor_element_size = torch.tensor([], dtype=dtype).element_size()
                    raw_tensor = torch.zeros(tensor_size // tensor_element_size, dtype=dtype, device=self.device)
                    assert tensor_size is not None
                    assert tensor_size % kv_cache_spec.page_size_bytes == 0
                    num_blocks = tensor_size // kv_cache_spec.page_size_bytes
                    assert num_blocks >= kv_cache_config.num_blocks

                    state_tensors = []
                    target_idx = 0
                    start_idx = 0
                    for shape, dtype in zip(kv_cache_spec.shapes, kv_cache_spec.dtypes):
                        target_shape = (num_blocks, *shape)
                        target_idx += torch.prod(torch.tensor(target_shape)).item()
                        tensor = raw_tensor[start_idx:target_idx].view(target_shape)
                        start_idx = target_idx
                        state_tensors.append(tensor)
                    kv_caches[layer_name] = state_tensors
                else:
                    raise ValueError("Unknown KV cache spec type.")

        return kv_caches
