# SPDX-License-Identifier: Apache-2.0
import torch
from vllm.config import VllmConfig
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheSpec, UniformTypeKVCacheSpecs

from vllm_ascend.ascend_config import get_kvpp_offload_config
from vllm_ascend.core.kv_cache_placement import (
    KVPP_SCRATCH_BUFFER_COUNT,
    build_kvpp_layer_layout,
    create_kvpp_cache_allocation_plan,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise_cache_layout import build_layerwise_reuse_layout
from vllm_ascend.distributed.parallel_state import get_kvpp_group

KVPP_BUFFER_ALIGNMENT = 2 * 1024 * 1024


def _allocate_kvpp_buffer(size: int, device: torch.device) -> torch.Tensor:
    raw = torch.zeros(size + KVPP_BUFFER_ALIGNMENT, dtype=torch.int8, device=device)
    return raw.narrow(0, (-raw.data_ptr()) % KVPP_BUFFER_ALIGNMENT, size)


def get_kvpp_cache_specs(kv_cache_config: KVCacheConfig) -> dict[str, KVCacheSpec]:
    specs: dict[str, KVCacheSpec] = {}
    for group in kv_cache_config.kv_cache_groups:
        spec = group.kv_cache_spec
        for name in group.layer_names:
            specs[name] = spec.kv_cache_specs[name] if isinstance(spec, UniformTypeKVCacheSpecs) else spec
    return specs


def allocate_kvpp_cache(
    vllm_config: VllmConfig, kv_cache_config: KVCacheConfig, device: torch.device
) -> dict[str, tuple[torch.Tensor, ...]]:
    """Allocate contiguous layer bundles and two shared Target scratch buffers."""
    plan = create_kvpp_cache_allocation_plan(
        vllm_config, get_kvpp_cache_specs(kv_cache_config), get_kvpp_group().rank_in_group
    )
    layouts = {
        name: build_kvpp_layer_layout(bundle, plan.tensor_sizes, kv_cache_config.num_blocks)
        for name, bundle in plan.layer_bundles.items()
    }
    scratch_size = max((size for name, (_, size) in layouts.items() if name in plan.layer_owner_ranks), default=0)
    extra = get_kvpp_offload_config(vllm_config)
    shared_slots: dict[str, int] = {}
    shared_buffers: dict[int, torch.Tensor] = {}
    if extra is not None:
        # Use offload's existing slot plan, including independent layers.
        reuse = build_layerwise_reuse_layout(
            plan.logical_cache_spec, vllm_config.model_config.get_num_layers(vllm_config.parallel_config), extra
        )
        for slot_index, layers in enumerate(reuse.buffer_slots):
            names = [reuse.layer_cache_specs[layer].main.layer_name for layer in layers]
            shared_buffers[slot_index] = _allocate_kvpp_buffer(max(layouts[name][1] for name in names), device)
            shared_slots.update((name, slot_index) for name in names)
    scratch = (
        [_allocate_kvpp_buffer(scratch_size, device) for _ in range(KVPP_SCRATCH_BUFFER_COUNT)]
        if scratch_size and extra is None
        else []
    )
    caches: dict[str, tuple[torch.Tensor, ...]] = {}
    target_index = 0
    for name, (layout, size) in layouts.items():
        owner = plan.layer_owner_ranks.get(name)
        if extra is not None:
            buffer = shared_buffers[shared_slots[name]]
        elif owner is None or owner == plan.kvpp_rank:
            buffer = _allocate_kvpp_buffer(size, device)
        else:
            buffer = scratch[target_index % KVPP_SCRATCH_BUFFER_COUNT]
        for cache_name, parts in layout.items():
            caches[cache_name] = tuple(buffer.narrow(0, offset, length) for offset, length in parts)
        if owner is not None:
            target_index += 1
    return caches
