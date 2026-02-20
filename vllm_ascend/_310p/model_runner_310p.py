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
from vllm.v1.kv_cache_interface import AttentionSpec, KVCacheConfig, MambaSpec

from vllm_ascend.utils import ACL_FORMAT_FRACTAL_NZ
from vllm_ascend.worker.model_runner_v1 import NPUModelRunner


class NPUModelRunner310(NPUModelRunner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._acl_format = ACL_FORMAT_FRACTAL_NZ

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
        if self.model_config.use_mla:
            raise ValueError("MLAAttention is not supported for 310P.")
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

    def _calculate_kv_cache_tensors_size(self, kv_cache_config: KVCacheConfig) -> dict[str, int]:
        """
        Initializes the KV cache size. The buffer needs to be reshaped to the desired shape before being used by
        the models.

        Args:
            kv_cache_config: The KV cache config
        Returns:
            dict[str, int]: A map between layer names to their
            corresponding memory buffer size.
        """
        # init kv cache tensors
        kv_cache_sizes: dict[str, int] = {}

        # First pass: collect all Mamba and Attention layers from kv_cache_groups
        # This ensures Mamba layers are correctly identified and not mistaken for Attention layers
        mamba_layers = set()
        for group in kv_cache_config.kv_cache_groups:
            kv_cache_spec = group.kv_cache_spec
            if isinstance(kv_cache_spec, MambaSpec):
                for layer_name in group.layer_names:
                    if layer_name not in self.runner_only_attn_layers:
                        mamba_layers.add(layer_name)

        # Second pass: process kv_cache_tensors, but skip Mamba layers
        # They will be handled separately
        for kv_cache_tensor in kv_cache_config.kv_cache_tensors:
            for idx in range(len(kv_cache_tensor.shared_by)):
                layer_name = kv_cache_tensor.shared_by[idx]

                # Skip Mamba layers - they will be handled in the third pass
                if layer_name in mamba_layers:
                    continue

                if "linear_attn" in layer_name and layer_name not in kv_cache_sizes:
                    # for mamba linear attention
                    kv_cache_size = kv_cache_tensor.size
                    for layer_name_inner in kv_cache_tensor.shared_by:
                        # shared the kvcache between the self_attn specs in the same group
                        if "linear_attn" in layer_name_inner:
                            kv_cache_sizes[layer_name_inner] = kv_cache_size
                elif "attn" in layer_name and layer_name not in kv_cache_sizes:
                    kv_tensor_split_factor = 2
                    kv_tensor_size = int(kv_cache_tensor.size // kv_tensor_split_factor)
                    for layer_name_inner in kv_cache_tensor.shared_by:
                        # shared the kvcache between the self_attn specs in the same group
                        if "attn" in layer_name_inner and "linear_attn" not in layer_name_inner:
                            kv_cache_sizes[layer_name_inner] = kv_tensor_size

        # Third pass: handle Mamba layers
        # FIX: Use the same size calculation as the main branch (model_runner_v1.py)
        # The main branch uses kv_cache_tensor.size from kv_cache_config.kv_cache_tensors
        # instead of calculating page_size_bytes * num_blocks
        logger.info(f"[MambaCalc DEBUG] Starting Third pass for Mamba layers (FIXED version)")
        
        # Build a mapping from layer_name to kv_cache_tensor.size
        # This mimics the main branch's approach in _allocate_kv_cache_tensors
        mamba_layer_sizes = {}
        for kv_cache_tensor in kv_cache_config.kv_cache_tensors:
            for layer_name in kv_cache_tensor.shared_by:
                # Only process Mamba layers (those with "linear_attn" in the name)
                if "linear_attn" in layer_name:
                    mamba_layer_sizes[layer_name] = kv_cache_tensor.size
                    logger.info(f"[MambaCalc DEBUG] Found Mamba layer {layer_name} with size={kv_cache_tensor.size / 1024**3:.3f} GiB")
        
        # Process Mamba layers using the sizes from kv_cache_tensors
        for group in kv_cache_config.kv_cache_groups:
            kv_cache_spec = group.kv_cache_spec
            if isinstance(kv_cache_spec, MambaSpec):
                for layer_name in group.layer_names:
                    if layer_name in self.runner_only_attn_layers:
                        logger.info(f"[MambaCalc DEBUG] Skipping layer {layer_name} (in runner_only_attn_layers)")
                        continue
                    
                    # Use the size from kv_cache_tensors (same as main branch)
                    if layer_name in mamba_layer_sizes:
                        size = mamba_layer_sizes[layer_name]
                        kv_cache_sizes[layer_name] = size
                        logger.info(f"[MambaCalc DEBUG] Layer {layer_name}: size={size / 1024**3:.3f} GiB (from kv_cache_tensors)")
                    else:
                        # Fallback to old calculation if not found (should not happen)
                        size = kv_cache_spec.page_size_bytes * kv_cache_config.num_blocks
                        kv_cache_sizes[layer_name] = size
                        logger.warning(f"[MambaCalc DEBUG] Layer {layer_name}: size={size / 1024**3:.3f} GiB (FALLBACK - not found in kv_cache_tensors)")
        
        # DEBUG: Print total Mamba size
        total_mamba_size = sum(size for name, size in kv_cache_sizes.items() 
                               if any(layer_name in name for layer_name in 
                                     [g.layer_names for g in kv_cache_config.kv_cache_groups 
                                      if isinstance(g.kv_cache_spec, MambaSpec)]))
        logger.info(f"[MambaCalc DEBUG] Total Mamba KV cache size: {total_mamba_size / 1024**3:.3f} GiB")

        # Verification: ensure all layers are accounted for
        layer_names = set()
        for group in kv_cache_config.kv_cache_groups:
            for layer_name in group.layer_names:
                if layer_name in self.runner_only_attn_layers:
                    continue
                layer_names.add(layer_name)

        missing_layers = layer_names - set(kv_cache_sizes.keys())
        assert not missing_layers, f"Some layers are not correctly initialized: {missing_layers}"

        return kv_cache_sizes

    def _allocate_kv_cache_and_reshape_tensors(
        self,
        kv_cache_config: KVCacheConfig,
        kv_cache_sizes: dict[str, int],
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
                    sum_page_size_bytes = kv_tensor_size * 2
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
                        kv_cache_shape = self.attn_backend.get_kv_cache_shape(
                            num_blocks, kv_cache_spec.block_size, kv_cache_spec.num_kv_heads, kv_cache_spec.head_size
                        )
                    dtype = kv_cache_spec.dtype
                    k_shape = kv_cache_shape[1:]
                    v_shape = k_shape
                    k_cache = torch_npu.empty_with_format(
                        size=k_shape, dtype=dtype, device=self.device, acl_format=self._acl_format
                    )
                    v_cache = torch_npu.empty_with_format(
                        size=v_shape, dtype=dtype, device=self.device, acl_format=self._acl_format
                    )
                    kv_caches[layer_name] = (k_cache, v_cache)
                elif isinstance(kv_cache_spec, MambaSpec):
                    tensor_size = kv_cache_sizes[layer_name]
                    logger.info(f"[MambaSpec DEBUG] layer_name={layer_name}, tensor_size={tensor_size / 1024**3:.3f} GiB")
                    logger.info(f"[MambaSpec DEBUG] kv_cache_spec.shapes={kv_cache_spec.shapes}")
                    logger.info(f"[MambaSpec DEBUG] kv_cache_spec.dtypes={kv_cache_spec.dtypes}")
                    logger.info(f"[MambaSpec DEBUG] kv_cache_spec.page_size_bytes={kv_cache_spec.page_size_bytes}")
                    logger.info(f"[MambaSpec DEBUG] kv_cache_config.num_blocks={kv_cache_config.num_blocks}")
                    # 获取NPU显存使用情况
                    try:
                        total_memory = torch.npu.get_device_properties(self.device).total_memory
                        allocated_memory = torch.npu.memory_allocated(self.device)
                        reserved_memory = torch.npu.memory_reserved(self.device)
                        free_memory = total_memory - allocated_memory
                        logger.info(f"[MambaSpec DEBUG] NPU Memory: total={total_memory / 1024**3:.3f} GiB, "
                                    f"allocated={allocated_memory / 1024**3:.3f} GiB, "
                                    f"reserved={reserved_memory / 1024**3:.3f} GiB, "
                                    f"free={free_memory / 1024**3:.3f} GiB")
                    except Exception as e:
                        logger.warning(f"[MambaSpec DEBUG] Failed to get NPU memory info: {e}")
                    raw_tensor = torch.zeros(tensor_size, dtype=torch.int8, device=self.device)
                    assert tensor_size is not None
                    assert tensor_size % kv_cache_spec.page_size_bytes == 0
                    num_blocks = tensor_size // kv_cache_spec.page_size_bytes
                    logger.info(f"[MambaSpec DEBUG] num_blocks={num_blocks}")
                    assert num_blocks >= kv_cache_config.num_blocks

                    state_tensors = []
                    target_idx = 0
                    start_idx = 0
                    for shape, dtype in zip(kv_cache_spec.shapes, kv_cache_spec.dtypes):
                        # normally, there is conv state and ssm state in this loop. And there is only
                        # a conv state in some special models.
                        target_shape = (num_blocks, *shape)

                        target_idx += torch.prod(torch.tensor(target_shape)).item()
                        tensor = raw_tensor.view(dtype)[start_idx:target_idx].view(target_shape)
                        logger.info(f"[MambaSpec DEBUG] Created state tensor: shape={target_shape}, dtype={dtype}")
                        start_idx = target_idx
                        state_tensors.append(tensor)
                    kv_caches[layer_name] = state_tensors
                else:
                    raise ValueError("Unknown KV cache spec type.")

        return kv_caches
