# SPDX-License-Identifier: Apache-2.0
import torch
from vllm.v1.kv_cache_interface import KVCacheConfig

from vllm_ascend.core.kv_cache_placement import (
    KVPP_SCRATCH_BUFFER_COUNT,
    build_kvpp_layer_layout,
    get_kvpp_cache_plan,
)

KVPP_BUFFER_ALIGNMENT = 2 * 1024 * 1024


def _allocate_kvpp_buffer(size: int, device: torch.device) -> torch.Tensor:
    raw = torch.zeros(size + KVPP_BUFFER_ALIGNMENT, dtype=torch.int8, device=device)
    return raw.narrow(0, (-raw.data_ptr()) % KVPP_BUFFER_ALIGNMENT, size)


def allocate_kvpp_cache(kv_cache_config: KVCacheConfig, device: torch.device) -> dict[str, tuple[torch.Tensor, ...]]:
    """Allocate the budgeted layer bundles and two target scratch buffers."""
    plan = get_kvpp_cache_plan(kv_cache_config)
    layouts = {
        name: build_kvpp_layer_layout(bundle, plan.tensor_sizes, kv_cache_config.num_blocks)
        for name, bundle in plan.layer_bundles.items()
    }
    scratch_size = max((size for name, (_, size) in layouts.items() if name in plan.layer_owner_ranks), default=0)
    scratch = (
        [_allocate_kvpp_buffer(scratch_size, device) for _ in range(KVPP_SCRATCH_BUFFER_COUNT)] if scratch_size else []
    )
    caches: dict[str, tuple[torch.Tensor, ...]] = {}
    target_index = 0
    for name, (layout, size) in layouts.items():
        owner = plan.layer_owner_ranks.get(name)
        if plan.is_persistent(name):
            buffer = _allocate_kvpp_buffer(size, device)
        else:
            buffer = scratch[target_index % KVPP_SCRATCH_BUFFER_COUNT]
        for cache_name, parts in layout.items():
            caches[cache_name] = tuple(buffer.narrow(0, offset, length) for offset, length in parts)
        if owner is not None:
            target_index += 1
    return caches
