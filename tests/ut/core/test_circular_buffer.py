# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import generate_scheduler_kv_cache_config
from vllm.v1.core.single_type_kv_cache_manager import CircularBufferManager
from vllm.v1.kv_cache_interface import (
    CircularBufferSpec,
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.kv_cache_spec_registry import KVCacheSpecRegistry

from vllm_ascend.core.compressor_tail import AscendCompressorTailSpec, CompressorTailManager
from vllm_ascend.core.kv_cache_interface import is_circular_kv_cache_spec, is_prefix_cacheable
from vllm_ascend.patch.platform.patch_kv_cache_coordinator import AscendHybridKVCacheCoordinator
from vllm_ascend.utils import vllm_version_is
from vllm_ascend.worker.block_table import BlockTable


def ring_spec():
    return CircularBufferSpec(
        block_size=32,
        num_kv_heads=1,
        head_size=1024,
        head_size_v=0,
        dtype=torch.float32,
    )


def test_ring_lifetime_reuse_and_external_tokens():
    pool = BlockPool(num_gpu_blocks=5, enable_caching=True, hash_block_size=32)
    manager = CircularBufferManager(
        ring_spec(),
        pool,
        enable_caching=True,
        kv_cache_group_id=0,
        scheduler_block_size=128,
        needs_kv_cache_zeroing=True,
    )
    free = pool.get_num_free_blocks()
    assert manager.get_num_blocks_to_allocate("a", 129, [], 0, 0, 129) == 1
    a = manager.allocate_new_blocks("a", 129, 129)
    b = manager.allocate_new_blocks("b", 1, 1)
    assert len(a) == len(b) == 1 and a[0].block_id != b[0].block_id
    assert not a[0].is_null and pool.get_num_free_blocks() == free - 2
    for tokens in (130, 1024, 65536):
        assert manager.get_num_blocks_to_allocate("a", tokens, [], tokens - 1, 0, tokens) == 0
        assert manager.allocate_new_blocks("a", tokens, tokens) == []
        manager.allocate_external_computed_blocks("a", 0, tokens)
        manager.remove_skipped_blocks("a", tokens)
        manager.cache_blocks(
            SimpleNamespace(request_id="a"),
            tokens,
            **({} if vllm_version_is("0.29.0") else {"replay_boundaries": [tokens - 1]}),
        )
        assert manager.req_to_blocks["a"] == a
    assert manager.take_new_block_ids() == []
    assert manager.get_num_common_prefix_blocks("a") == manager.get_num_skipped_tokens(65536) == 0
    manager.free("a")  # Finish or preempt, then resume under a new allocation.
    manager.free("b")
    assert pool.get_num_free_blocks() == free
    manager.allocate_external_computed_blocks("a", 0, 65536)
    assert len(manager.req_to_blocks["a"]) == 1
    manager.free("a")
    assert pool.get_num_free_blocks() == free


def test_uniform_properties_and_single_plane_size():
    spec = ring_spec()
    uniform = UniformTypeKVCacheSpecs(block_size=32, kv_cache_specs={"s0": spec, "s1": spec})
    assert spec.page_size_bytes == 131072
    assert spec.max_memory_usage_bytes(None) == 131072
    assert uniform.max_num_blocks_per_req(None, 65536) == 1
    assert is_circular_kv_cache_spec(uniform) and not is_prefix_cacheable(uniform)
    assert not uniform.prefix_cacheable
    assert not is_prefix_cacheable(SimpleNamespace(participates_in_prefix_caching=False))


def test_scratch_groups_do_not_reduce_prefix_hits_or_truncation():
    full = FullAttentionSpec(block_size=128, num_kv_heads=1, head_size=8, dtype=torch.float32)
    scratch = ring_spec()
    groups = [KVCacheGroupSpec(["kv"], full), KVCacheGroupSpec(["state"], scratch)]
    coordinator = SimpleNamespace(
        kv_cache_config=SimpleNamespace(kv_cache_groups=groups),
        single_type_managers=[SimpleNamespace(), SimpleNamespace()],
        eagle_group_ids=set(),
        scheduler_block_size=128,
        _get_effective_block_size=lambda spec: spec.block_size,
    )
    AscendHybridKVCacheCoordinator.verify_and_split_kv_cache_groups(coordinator)
    assert len(coordinator.attention_groups) == 1
    assert coordinator.attention_groups[0].group_ids == [0]
    host = SimpleNamespace(
        kv_cache_config=SimpleNamespace(kv_cache_groups=groups),
        coordinator=SimpleNamespace(
            single_type_managers=[SimpleNamespace(block_size=128), SimpleNamespace(block_size=32)]
        ),
        create_kv_cache_blocks=lambda blocks: blocks,
    )
    assert KVCacheManager.truncate_computed_blocks(host, SimpleNamespace(blocks=([1, 2], [])), 128) == ([1], [])

    mamba = MambaSpec(block_size=128, shapes=((1,),), dtypes=(torch.float32,))
    host.kv_cache_config.kv_cache_groups = [groups[0], KVCacheGroupSpec(["mamba"], mamba), groups[1]]
    host.coordinator.single_type_managers = [
        SimpleNamespace(block_size=128),
        SimpleNamespace(block_size=128),
        SimpleNamespace(block_size=32),
    ]
    assert KVCacheManager.truncate_computed_blocks(host, SimpleNamespace(blocks=([1, 2], [], [])), 128) == ([1], [], [])


@pytest.mark.parametrize("draft", [False, True])
def test_ring_bypasses_position_to_page_mapping(draft):
    mapping = torch.zeros(8, dtype=torch.int64)
    table = SimpleNamespace(is_circular_group=True, slot_mapping=SimpleNamespace(gpu=mapping))
    if draft:
        BlockTable.compute_slot_mapping_draft(table, Mock(), Mock())
    else:
        BlockTable.compute_slot_mapping(table, 1, Mock(), Mock())
    assert mapping.tolist() == [-1] * 8


@pytest.mark.parametrize("ratio,block_rows,tail_rows,pages", [(4, 8, 8, 1), (128, 32, 128, 4)])
@pytest.mark.parametrize("zeroing", [False, True])
def test_compressor_private_ring_lifetime_and_zeroing(ratio, block_rows, tail_rows, pages, zeroing):
    spec = AscendCompressorTailSpec(
        block_size=block_rows,
        num_kv_heads=1,
        head_size=2048 if ratio == 4 else 1024,
        dtype=torch.float32,
        sliding_window=tail_rows,
        page_size_padded=131072,
        tail_tokens=tail_rows,
        ring_blocks_per_request=pages,
    )
    pool = BlockPool(2 * pages + 1, False, 128)
    manager = CompressorTailManager(
        spec,
        pool,
        enable_caching=False,
        kv_cache_group_id=0,
        scheduler_block_size=128,
        needs_kv_cache_zeroing=zeroing,
    )
    first = manager.allocate_new_blocks("first", 32768, 32768)
    second = manager.allocate_new_blocks("second", 891, 891)
    assert {b.block_id for b in first}.isdisjoint(b.block_id for b in second)
    expected = [b.block_id for b in first + second] if zeroing else []
    assert manager.take_new_block_ids() == expected
    assert manager.allocate_new_blocks("first", 63778, 63778) == []
    manager.remove_skipped_blocks("first", 63778)
    assert manager.req_to_blocks["first"] == first
    held = manager.pop_blocks_for_free("first")
    assert pool.get_num_free_blocks() == 0 and all(b.ref_cnt == 1 for b in held)
    pool.free_blocks(reversed(held))
    recycled = manager.allocate_new_blocks("third", 1, 1)
    assert {b.block_id for b in recycled} == {b.block_id for b in first}
    manager.free("second")
    manager.free("third")
    assert pool.get_num_free_blocks() == 2 * pages


def test_compressor_uniform_tuple_keeps_distinct_page_widths():
    main = AscendCompressorTailSpec(
        block_size=8,
        num_kv_heads=1,
        head_size=2048,
        dtype=torch.float32,
        sliding_window=8,
        page_size_padded=131072,
        tail_tokens=8,
        ring_blocks_per_request=1,
    )
    indexer = replace(main, head_size=512, page_size_padded=16640)
    uniform = UniformTypeKVCacheSpecs.from_specs({"main": main, "indexer": indexer})
    assert uniform is not None and uniform.page_size_bytes == 147712
    assert uniform.max_num_blocks_per_req(None, 65536) == 1
    assert not main.is_uniform_with_collection({"other": replace(main, tail_tokens=16, ring_blocks_per_request=2)})
    KVCacheSpecRegistry.register(
        kvcache_spec_cls=AscendCompressorTailSpec,
        manager_class=CompressorTailManager,
        uniform_type_base_spec=AscendCompressorTailSpec,
    )
    config = KVCacheConfig(5, [], [KVCacheGroupSpec(["main", "indexer"], uniform)])
    scheduler_config = generate_scheduler_kv_cache_config([config])
    assert scheduler_config.kv_cache_groups[0].kv_cache_spec == main
    assert isinstance(config.kv_cache_groups[0].kv_cache_spec, UniformTypeKVCacheSpecs)
    manager = KVCacheManager(scheduler_config, 65536, 128, 128, enable_caching=False)
    assert isinstance(manager.coordinator.single_type_managers[0], CompressorTailManager)
