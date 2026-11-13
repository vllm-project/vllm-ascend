# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from functools import wraps

import torch
from vllm.config import get_current_vllm_config_or_none
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheLayout, UniformTypeKVCacheSpecs, create_kv_cache_views

from vllm_ascend.hisparse.binding import _allocate_aligned_slab


def _allocate_hisparse_slab(
    kv_cache_config: KVCacheConfig,
    device: torch.device,
    layout: KVCacheLayout,
    kernel_block_sizes: list[int] | None,
) -> dict[str, torch.Tensor]:
    (size,) = {tensor.size for tensor in kv_cache_config.kv_cache_tensors}
    slab = _allocate_aligned_slab(size, device)
    group_ids = {
        name: index for index, group in enumerate(kv_cache_config.kv_cache_groups) for name in group.layer_names
    }
    caches = {}
    for tensor in kv_cache_config.kv_cache_tensors:
        name = tensor.layers[0]
        group_id = group_ids[name]
        spec = kv_cache_config.kv_cache_groups[group_id].kv_cache_spec
        if isinstance(spec, UniformTypeKVCacheSpecs):
            spec = spec.kv_cache_specs[name]
        if not spec.has_layer_views:
            caches.update((layer_name, slab) for layer_name in tensor.layers)
            continue
        kernel_block_size = None
        if kernel_block_sizes is not None and group_id < len(kernel_block_sizes):
            kernel_block_size = kernel_block_sizes[group_id]
        views = create_kv_cache_views(
            slab, spec, kv_cache_config.num_blocks_of(tensor), layout, tensor, kernel_block_size=kernel_block_size
        )
        caches.update(zip(tensor.layers, views))
    return caches


def install_hisparse_slab_allocator() -> None:
    from vllm.v1.hisparse import binding

    original = binding.allocate_kv_cache
    if getattr(original, "_ascend_aligned_slab", False):
        return

    @wraps(original)
    def allocate(kv_cache_config, device, layout, kernel_block_sizes=None):
        config = get_current_vllm_config_or_none()
        transfer = config.kv_transfer_config if config is not None else None
        if (
            device.type == "npu"
            and transfer is not None
            and transfer.has_connector("AscendStoreConnector")
            and kv_cache_config.kv_cache_tensors
        ):
            return _allocate_hisparse_slab(kv_cache_config, device, layout, kernel_block_sizes)
        return original(kv_cache_config, device, layout, kernel_block_sizes)

    allocate.__dict__["_ascend_aligned_slab"] = True
    binding.allocate_kv_cache = allocate
