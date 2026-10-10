# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable

import torch
from vllm.utils.torch_utils import get_dtype_size

from vllm_ascend.core.kv_cache_interface import AscendSFAIndexerCacheSpec


def allocate_shared_indexer_cache(
    specs: dict[str, AscendSFAIndexerCacheSpec],
    num_blocks: int,
    allocate: Callable[[int], torch.Tensor],
) -> dict[str, tuple[torch.Tensor, ...]]:
    """Share raw bytes while retaining each layer's data and scale layout.

    The caller must fence reuse between layers. Returned byte views contain
    exactly the layer's logical capacity, so reshape and transfer code never
    mistake spare backing capacity for extra blocks.
    """
    regions: dict[str, tuple[int, int, int]] = {}
    capacity = 0
    for name, spec in specs.items():
        rows = num_blocks * spec.sfa_dcp_replicated_indexer_size * spec.block_size * spec.num_kv_heads
        data_bytes = rows * spec.head_size * get_dtype_size(spec.dtype)
        scale_bytes = rows * spec.scale_dim * get_dtype_size(spec.scale_dtype)
        scale_alignment = get_dtype_size(spec.scale_dtype)
        scale_offset = ((data_bytes + scale_alignment - 1) // scale_alignment) * scale_alignment
        regions[name] = (data_bytes, scale_offset, scale_bytes)
        capacity = max(capacity, scale_offset + scale_bytes if scale_bytes else data_bytes)
    backing = allocate(capacity)
    views: dict[str, tuple[torch.Tensor, ...]] = {}
    region_views: dict[tuple[int, int, int], tuple[torch.Tensor, ...]] = {}
    for name, region in regions.items():
        if region not in region_views:
            data_bytes, scale_offset, scale_bytes = region
            data = backing[:data_bytes]
            region_views[region] = (
                (data, backing[scale_offset : scale_offset + scale_bytes]) if scale_bytes else (data,)
            )
        views[name] = region_views[region]
    return views
