# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

from types import SimpleNamespace

import pytest
import torch
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.single_type_kv_cache_manager import CircularBufferManager, register_all_kvcache_specs
from vllm.v1.kv_cache_interface import CircularBufferSpec, UniformTypeKVCacheSpecs
from vllm.v1.kv_cache_spec_registry import KVCacheSpecRegistry

from vllm_ascend.core.kv_cache_interface import (
    AscendCircularBufferSpec,
    AscendMLAAttentionSpec,
    AscendSlidingWindowMLASpec,
    get_kv_cache_compression_ratio,
    get_storage_block_size,
    register_ascend_kv_cache_specs,
    supports_component_major_mla_pd,
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


@pytest.mark.parametrize("ascend", [False, True])
@pytest.mark.parametrize("zeroing", [False, True])
def test_circular_registry_preserves_spec_and_reports_recycled_pages(ascend, zeroing):
    from vllm_ascend.core.single_type_kv_cache_manager import AscendCircularBufferManager

    register_all_kvcache_specs(None)
    register_ascend_kv_cache_specs()
    spec_type = AscendCircularBufferSpec if ascend else CircularBufferSpec
    spec = spec_type(block_size=32, num_kv_heads=1, head_size=16, head_size_v=0, dtype=torch.float32)
    merged = spec_type.merge([spec, spec])
    assert type(merged) is spec_type
    assert KVCacheSpecRegistry.get_uniform_type_base_spec(merged) is spec_type
    manager_type = KVCacheSpecRegistry.get_manager_class(merged)
    assert manager_type is (AscendCircularBufferManager if ascend else CircularBufferManager)
    pool = BlockPool(4, False, 32)
    # Reuse physical IDs released by another cache group.
    old_blocks = pool.get_new_blocks(pool.get_num_free_blocks())
    old_ids = {block.block_id for block in old_blocks}
    pool.free_blocks(old_blocks)
    manager = manager_type(merged, pool, False, 0, scheduler_block_size=32, needs_kv_cache_zeroing=zeroing)
    blocks = manager.allocate_new_blocks("request", 1, 1)
    assert len(blocks) == 1 and blocks[0].block_id in old_ids
    assert manager.new_block_ids == ([blocks[0].block_id] if ascend and zeroing else [])
    assert manager.allocate_new_blocks("request", 2, 2) == []
    assert not spec.prefix_cacheable and not spec.uses_slot_mapping
    manager.free("request")
    assert pool.get_num_free_blocks() == len(old_ids)


def test_component_major_mla_pd_gate():
    assert supports_component_major_mla_pd(SimpleNamespace(kv_transfer_config=None))

    for connector in ("MooncakeConnectorV2", "MooncakePullConnector"):
        config = SimpleNamespace(
            kv_transfer_config=SimpleNamespace(
                kv_connector=connector,
                kv_connector_module_path=None,
            )
        )
        assert supports_component_major_mla_pd(config)

    for connector in ("MooncakeConnectorV1", "AscendStoreConnector", "MultiConnector"):
        config = SimpleNamespace(
            kv_transfer_config=SimpleNamespace(
                kv_connector=connector,
                kv_connector_module_path=None,
            )
        )
        assert not supports_component_major_mla_pd(config)

    config = SimpleNamespace(
        kv_transfer_config=SimpleNamespace(
            kv_connector="MooncakeConnectorV2",
            kv_connector_module_path="custom.connector",
        )
    )
    assert not supports_component_major_mla_pd(config)
