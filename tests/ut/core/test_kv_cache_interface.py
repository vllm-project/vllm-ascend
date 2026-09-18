# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

from types import SimpleNamespace

import pytest
import torch
from vllm.v1.kv_cache_interface import UniformTypeKVCacheSpecs

from vllm_ascend.core.kv_cache_interface import (
    AscendMLAAttentionSpec,
    AscendSlidingWindowMLASpec,
    get_kv_cache_compression_ratio,
    get_storage_block_size,
    is_circular_kv_cache_spec,
    is_prefix_cacheable,
)


def _mla_spec():
    return AscendMLAAttentionSpec(
        block_size=16,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.bfloat16,
    )


def test_get_storage_block_size_and_dcp_memory():
    spec = _mla_spec()
    # On main, storage_block_size is an optional dataclass field and may be
    # None. Ascend derives physical rows from block_size / compression ratio.
    expected = spec.block_size // get_kv_cache_compression_ratio(spec)
    assert get_storage_block_size(spec) == expected

    uniform = UniformTypeKVCacheSpecs(block_size=16, kv_cache_specs={"layer": spec})
    assert get_storage_block_size(uniform) == expected

    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(max_model_len=128),
        parallel_config=SimpleNamespace(decode_context_parallel_size=2),
    )
    assert spec.max_memory_usage_bytes(vllm_config) > 0


def test_sliding_window_mla_storage_and_page_size():
    spec = AscendSlidingWindowMLASpec(
        block_size=16,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.bfloat16,
        sliding_window=64,
    )
    assert spec.storage_block_size == 16
    assert spec.real_page_size_bytes == 16 * 128 * 2


@pytest.mark.parametrize("circular", [False, True])
def test_legacy_circular_capability_is_preserved(circular):
    spec = SimpleNamespace(is_circular=circular)
    assert is_circular_kv_cache_spec(spec) is circular
    uniform = UniformTypeKVCacheSpecs(block_size=32, kv_cache_specs={"state": spec})
    assert is_circular_kv_cache_spec(uniform) is circular
    assert not is_circular_kv_cache_spec(SimpleNamespace())


def test_mixed_and_empty_groups_are_not_circular():
    for specs in ({}, {"state": SimpleNamespace(is_circular=True), "kv": SimpleNamespace()}):
        uniform = UniformTypeKVCacheSpecs(block_size=32, kv_cache_specs=specs)
        assert not is_circular_kv_cache_spec(uniform)


@pytest.mark.parametrize(
    "flags, expected",
    [
        ({}, True),
        ({"prefix_cacheable": False}, False),
        ({"participates_in_prefix_caching": False}, False),
        ({"prefix_cacheable": True, "participates_in_prefix_caching": False}, False),
    ],
)
def test_prefix_cache_capabilities_across_versions(flags, expected):
    spec = SimpleNamespace(**flags)
    assert is_prefix_cacheable(spec) is expected
    uniform = UniformTypeKVCacheSpecs(block_size=32, kv_cache_specs={"state": spec})
    assert is_prefix_cacheable(uniform) is expected
