# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.attention.cache_layout import (
    MemoryKind,
    build_mla_layer_cache_descriptor,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.decode_host_kv_pool import (
    validate_decode_host_kv_caches,
)


def _descriptor(layer_id: int, group_id: int = 0, mtp: bool = False):
    spec = SimpleNamespace(
        block_size=4,
        num_kv_heads=1,
        sparse_head_dim=(8, 2, 4),
        dtype=torch.bfloat16,
        cache_sparse_c8=False,
        c8_k_cache_dtype=torch.int8,
        c8_k_scale_cache_dtype=torch.float16,
    )
    return build_mla_layer_cache_descriptor(
        layer_name=f"layer.{layer_id}",
        layer_id=layer_id,
        group_id=group_id,
        cache_family_id=f"family-{group_id}",
        kv_cache_spec=spec,
        is_mtp_layer=mtp,
    )


def _layer_cache(descriptor, num_blocks: int):
    return tuple(
        torch.empty((num_blocks, *component.shape_per_block), dtype=component.dtype)
        for component in descriptor.components
    )


def test_validate_decode_host_kv_caches_accepts_group_capacities_and_skips_mtp():
    group0 = _descriptor(0, 0)
    group1 = _descriptor(1, 1)
    mtp = _descriptor(2, 0, mtp=True)

    validate_decode_host_kv_caches(
        {0: 2, 1: 5},
        [group0, group1, mtp],
        {
            group0.layer_name: _layer_cache(group0, 2),
            group1.layer_name: _layer_cache(group1, 5),
        },
    )


def test_validate_decode_host_kv_caches_rejects_wrong_capacity():
    descriptor = _descriptor(0)
    with pytest.raises(ValueError, match="has 2 blocks, expected 3"):
        validate_decode_host_kv_caches(
            {0: 3},
            [descriptor],
            {descriptor.layer_name: _layer_cache(descriptor, 2)},
        )


def test_validate_decode_host_kv_caches_rejects_wrong_indexer_capacity():
    descriptor = _descriptor(0)
    caches = list(_layer_cache(descriptor, 3))
    indexer = descriptor.indexer_components[0]
    caches[indexer.tensor_index] = torch.empty(
        (2, *indexer.shape_per_block),
        dtype=indexer.dtype,
    )

    with pytest.raises(ValueError, match="indexer_k has 2 blocks, expected 3"):
        validate_decode_host_kv_caches(
            {0: 3},
            [descriptor],
            {descriptor.layer_name: tuple(caches)},
        )


def test_validate_decode_host_kv_caches_rejects_missing_layer():
    descriptor = _descriptor(0)
    with pytest.raises(KeyError, match="layer.0"):
        validate_decode_host_kv_caches({0: 1}, [descriptor], {})


def test_validate_decode_host_kv_caches_rejects_wrong_memory_kind():
    descriptor = _descriptor(0)
    main = descriptor.main_components[0]
    components = tuple(
        replace(component, decode_memory_kind=MemoryKind.DEVICE)
        if component is main
        else component
        for component in descriptor.components
    )
    invalid_descriptor = replace(descriptor, components=components)

    with pytest.raises(
        ValueError, match="requires Host Main and Device Indexer"
    ):
        validate_decode_host_kv_caches(
            {0: 1},
            [invalid_descriptor],
            {descriptor.layer_name: _layer_cache(invalid_descriptor, 1)},
        )
