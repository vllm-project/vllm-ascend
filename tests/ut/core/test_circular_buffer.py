# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheGroupSpec, UniformTypeKVCacheSpecs

from vllm_ascend.core.circular_buffer import (
    AscendCircularBufferManager,
    AscendCircularBufferSpec,
    is_circular_spec,
    prefix_cacheable,
)
from vllm_ascend.patch.platform.patch_circular_buffer import _truncate_computed_blocks
from vllm_ascend.patch.platform.patch_kv_cache_coordinator import AscendHybridKVCacheCoordinator
from vllm_ascend.worker.block_table import BlockTable


def ring_spec():
    return AscendCircularBufferSpec(block_size=32, num_kv_heads=1, head_size=1024, dtype=torch.float32)


def test_ring_lifetime_reuse_and_external_tokens():
    pool = BlockPool(num_gpu_blocks=5, enable_caching=True, hash_block_size=32)
    manager = AscendCircularBufferManager(
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
            replay_boundary=tokens - 1,
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
    assert is_circular_spec(uniform) and not prefix_cacheable(uniform)
    assert not uniform.prefix_cacheable
    assert not prefix_cacheable(SimpleNamespace(participates_in_prefix_caching=False))


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
    assert _truncate_computed_blocks(host, SimpleNamespace(blocks=([1, 2], [])), 128) == ([1], [])
    coordinator.kv_cache_config.kv_cache_groups = [groups[1]]
    AscendHybridKVCacheCoordinator.verify_and_split_kv_cache_groups(coordinator)
    assert AscendHybridKVCacheCoordinator.find_longest_cache_hit(coordinator, [], 128) == (([],), 0)


@pytest.mark.parametrize("draft", [False, True])
def test_ring_bypasses_position_to_page_mapping(draft):
    mapping = torch.zeros(8, dtype=torch.int64)
    table = SimpleNamespace(is_circular_group=True, slot_mapping=SimpleNamespace(gpu=mapping))
    if draft:
        BlockTable.compute_slot_mapping_draft(table, Mock(), Mock())
    else:
        BlockTable.compute_slot_mapping(table, 1, Mock(), Mock())
    assert mapping.tolist() == [-1] * 8
