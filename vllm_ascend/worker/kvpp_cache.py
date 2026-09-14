# SPDX-License-Identifier: Apache-2.0
import torch
from vllm.config import VllmConfig
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheSpec, UniformTypeKVCacheSpecs

from vllm_ascend.core.kv_cache_placement import (
    KVPP_SCRATCH_BUFFER_COUNT,
    build_kvpp_layer_layout,
    create_kvpp_cache_allocation_plan,
)
from vllm_ascend.distributed.parallel_state import get_kvpp_group


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
    scratch = (
        [torch.zeros(scratch_size, dtype=torch.int8, device=device) for _ in range(KVPP_SCRATCH_BUFFER_COUNT)]
        if scratch_size
        else []
    )
    caches: dict[str, tuple[torch.Tensor, ...]] = {}
    target_index = 0
    for name, (layout, size) in layouts.items():
        owner = plan.layer_owner_ranks.get(name)
        if owner is None or owner == plan.kvpp_rank:
            buffer = torch.zeros(size, dtype=torch.int8, device=device)
        else:
            buffer = scratch[target_index % KVPP_SCRATCH_BUFFER_COUNT]
        for cache_name, parts in layout.items():
            caches[cache_name] = tuple(buffer.narrow(0, offset, length) for offset, length in parts)
        if owner is not None:
            target_index += 1
    return caches
