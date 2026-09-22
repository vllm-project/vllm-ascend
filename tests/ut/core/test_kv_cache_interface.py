# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

from types import SimpleNamespace

import pytest
import torch
from vllm.v1.kv_cache_interface import UniformTypeKVCacheSpecs

from vllm_ascend.core.kv_cache_interface import (
    AscendMLAAttentionSpec,
    AscendSlidingWindowMLASpec,
    KVCacheBlockGeometry,
    get_block_geometry,
    get_kernel_block_size,
    get_kv_cache_compression_ratio,
    get_storage_block_size,
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


def test_explicit_block_geometry_is_owned_by_spec():
    geometry = KVCacheBlockGeometry(
        manager_block_size=4096,
        kernel_block_size=4096,
        storage_block_size=32,
    )
    spec = AscendMLAAttentionSpec(
        block_size=4096,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.bfloat16,
        tokens_per_state=128,
        block_geometry=geometry,
    )

    assert get_block_geometry(spec) is geometry
    assert get_kernel_block_size(spec) == 4096
    assert get_storage_block_size(spec) == 32


def test_block_geometry_rejects_mismatched_manager_size():
    geometry = KVCacheBlockGeometry(
        manager_block_size=128,
        kernel_block_size=128,
        storage_block_size=32,
    )
    with pytest.raises(ValueError, match="does not match"):
        AscendMLAAttentionSpec(
            block_size=256,
            num_kv_heads=1,
            head_size=128,
            dtype=torch.bfloat16,
            tokens_per_state=4,
            block_geometry=geometry,
        )
