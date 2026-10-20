# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

from collections.abc import Callable, Iterable

import torch
from vllm.config import VllmConfig
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheSpec,
    KVQuantMode,
    SlidingWindowSpec,
    UniformTypeKVCacheSpecs,
)


def can_share_full_and_sliding_kv_cache(vllm_config: VllmConfig, specs: Iterable[KVCacheSpec]) -> bool:
    """Keep the budget and both runners on the same dense attention layout.

    Equal physical pages let full and sliding-window groups overlay the same
    pool even when their head dimensions and tokens per block differ. Keep
    specialized/quantized caches and distributed transfer on their existing
    allocation paths.
    """
    specs = tuple(specs)
    if {type(spec) for spec in specs} != {FullAttentionSpec, SlidingWindowSpec}:
        return False
    if vllm_config.kv_transfer_config is not None:
        return False
    parallel_config = vllm_config.parallel_config
    if parallel_config.decode_context_parallel_size != 1 or parallel_config.prefill_context_parallel_size != 1:
        return False
    layout = vllm_config.cache_config.get_resolved_kv_cache_layout()
    if not (layout.is_layer_compact and layout.is_block_compact):
        return False
    return len({spec.page_size_bytes for spec in specs}) == 1 and all(
        spec.dtype in (torch.float16, torch.bfloat16)
        and spec.kv_quant_mode == KVQuantMode.NONE
        and spec.head_size == spec.head_size_v
        and spec.page_size_bytes == spec.real_page_size_bytes
        and spec.tokens_per_state == 1
        and spec.num_head_slots is None
        and spec.state_content_bytes is None
        for spec in specs
    )


def allocate_shared_attention_kv_cache(
    kv_cache_config: KVCacheConfig,
    allocate: Callable[[int], torch.Tensor],
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """Materialize overlay descriptors as contiguous K/V views of one pool.

    Groups share block IDs from one scheduler pool, so the same ID cannot be
    live in two groups at once. Split each layer slot into K and V halves;
    equal, unpadded pages and equal K/V widths make this mapping identical
    across groups while preserving the contiguous layout required by PA.
    Validate all descriptors before allocating: falling back to private
    buffers after retaining the full budget would overallocate device memory.
    """
    descriptors = kv_cache_config.kv_cache_tensors
    specs = {
        name: (
            group.kv_cache_spec.kv_cache_specs[name]
            if isinstance(group.kv_cache_spec, UniformTypeKVCacheSpecs)
            else group.kv_cache_spec
        )
        for group in kv_cache_config.kv_cache_groups
        for name in group.layer_names
    }
    sizes = {descriptor.size for descriptor in descriptors}
    if len(sizes) != 1 or kv_cache_config.num_blocks <= 0:
        raise ValueError("Shared attention KV cache requires one nonempty backing allocation.")
    backing_size = sizes.pop()
    regions: dict[str, tuple[int, int]] = {}
    for descriptor in descriptors:
        for index, name in enumerate(descriptor.layers):
            spec = specs[name]
            layer_size = kv_cache_config.num_blocks * spec.page_size_bytes
            start = descriptor.offset + index * descriptor.layer_stride
            if (
                layer_size <= 0
                or layer_size % 2
                or descriptor.layer_stride != layer_size
                or descriptor.block_stride != spec.page_size_bytes
                or start < 0
                or start % layer_size
                or start + layer_size > backing_size
                or name in regions
            ):
                raise ValueError(f"Invalid shared attention KV cache descriptor for {name}.")
            regions[name] = (start, layer_size)
    if regions.keys() != specs.keys():
        raise ValueError("Shared attention KV cache descriptors do not cover all layers.")
    backing = allocate(backing_size)
    return {
        name: (
            backing[start : start + layer_size // 2],
            backing[start + layer_size // 2 : start + layer_size],
        )
        for name, (start, layer_size) in regions.items()
    }
