# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch

from vllm_ascend.attention.cache_layout import (
    LayerCacheLayoutDescriptor,
    MemoryKind,
    bind_layer_cache_views,
)


def validate_decode_host_kv_caches(
    num_blocks_by_group: Mapping[int, int],
    descriptors: Sequence[LayerCacheLayoutDescriptor],
    layer_caches: Mapping[str, Sequence[torch.Tensor] | torch.Tensor],
) -> None:
    """Validate the fixed Host-backed Main cache allocated by Model Runner.

    The scheduler remains the only physical block allocator. Model Runner owns
    the backing tensors and passes the same ``kv_caches`` mapping to the
    Connector and AttentionBackend, so this helper deliberately keeps no
    second pool, address registry, request lifecycle, or allocation API.
    """

    host_main_components = 0
    for descriptor in descriptors:
        if descriptor.is_mtp_layer:
            continue
        num_blocks = num_blocks_by_group.get(descriptor.group_id)
        if num_blocks is None or num_blocks <= 0:
            raise ValueError(
                f"Missing Host KV capacity for group {descriptor.group_id}"
            )
        if descriptor.layer_name not in layer_caches:
            raise KeyError(descriptor.layer_name)
        raw = layer_caches[descriptor.layer_name]
        raw_tuple = (raw,) if isinstance(raw, torch.Tensor) else tuple(raw)
        views = bind_layer_cache_views(descriptor, raw_tuple)
        host_main = tuple(
            component
            for component in descriptor.main_components
            if component.decode_memory_kind is MemoryKind.HOST_DEVICE_VISIBLE
        )
        device_indexer = tuple(
            component
            for component in descriptor.indexer_components
            if component.decode_memory_kind is MemoryKind.DEVICE
        )
        if (
            not host_main
            or len(host_main) != len(descriptor.main_components)
            or not device_indexer
            or len(device_indexer) != len(descriptor.indexer_components)
        ):
            raise ValueError(
                "Decode Host KV cache requires Host Main and Device Indexer "
                f"components for {descriptor.layer_name}"
            )
        for component in descriptor.components:
            tensor = views[component.component_role]
            if tensor.shape[0] != num_blocks:
                raise ValueError(
                    f"Cache {descriptor.layer_name}/"
                    f"{component.component_role.value} has {tensor.shape[0]} "
                    f"blocks, expected {num_blocks}"
                )
        host_main_components += len(host_main)
    if host_main_components == 0:
        raise ValueError("Decode Host KV cache has no Host Main components")
