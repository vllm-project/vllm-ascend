# SPDX-License-Identifier: Apache-2.0
"""Keep PMU CoW on the upstream lifecycle instead of growing missing sources."""

from types import SimpleNamespace

import pytest
from vllm.v1.core.single_type_kv_cache_manager import FullAttentionManager

from vllm_ascend.patch.platform.patch_mamba_manager import AscendMambaManager


class _BlockPool:
    def __init__(self):
        self.allocated = []

    def touch(self, blocks):
        for block in blocks:
            block.ref_cnt += 1

    def get_new_blocks(self, count):
        blocks = [
            SimpleNamespace(block_id=100 + len(self.allocated) + i, ref_cnt=1, is_null=False) for i in range(count)
        ]
        self.allocated.extend(blocks)
        return blocks


def _manager(record_new_blocks=True):
    manager = FullAttentionManager.__new__(FullAttentionManager)
    manager.block_size = 16
    manager.enable_caching = True
    manager.block_pool = _BlockPool()
    manager.req_to_blocks = {"req": []}
    manager.num_cached_block = {}
    manager._partial_hit_reqs = {}
    manager._pending_cow_copies = []
    manager._record_new_block_ids = record_new_blocks
    manager.new_block_ids = []
    manager._null_block = SimpleNamespace(block_id=0, ref_cnt=0, is_null=True)
    manager._max_admission_blocks_per_request = None
    # No sliding-window skips in this FA contract test.
    manager.get_num_skipped_tokens = lambda tokens: 0
    return manager


@pytest.mark.parametrize("local_tokens", [1, 15, 17, 31, 33])
@pytest.mark.parametrize("external_tokens", [0, 16, 32])
@pytest.mark.parametrize("record_new_blocks", [False, True])
def test_partial_hit_source_is_mounted_before_external_growth_and_cow(local_tokens, external_tokens, record_new_blocks):
    """Public registration makes the source slot valid before any allocation."""
    manager = _manager(record_new_blocks)
    source_blocks = [
        SimpleNamespace(block_id=i + 1, ref_cnt=1, is_null=False)
        for i in range((local_tokens + manager.block_size - 1) // manager.block_size)
    ]
    total_tokens = local_tokens + external_tokens
    scheduled_tokens = total_tokens + 17
    reserved = manager.get_num_blocks_to_allocate(
        "req", scheduled_tokens, source_blocks, total_tokens, local_tokens, scheduled_tokens
    )
    manager.add_local_computed_blocks("req", source_blocks, local_tokens, external_tokens)
    idx, source = manager._partial_hit_reqs["req"]
    assert 0 <= idx < len(manager.req_to_blocks["req"])
    assert manager.req_to_blocks["req"][idx] is source is source_blocks[-1]
    assert source.ref_cnt == 2

    manager.allocate_external_computed_blocks("req", local_tokens, external_tokens)
    assert manager.req_to_blocks["req"][idx] is source
    allocated = manager.allocate_new_blocks("req", scheduled_tokens, scheduled_tokens)

    cow = manager.req_to_blocks["req"][idx]
    assert cow is not source
    assert allocated[0] is cow
    assert manager._pending_cow_copies == [(source, cow)]
    assert source.ref_cnt == cow.ref_cnt == 2
    assert "req" not in manager._partial_hit_reqs
    assert len(manager.block_pool.allocated) == reserved
    assert len(manager.req_to_blocks["req"]) == (scheduled_tokens + 15) // 16
    assert cow.block_id in manager.new_block_ids
    assert all(block is not source for block in manager.block_pool.allocated)
    # Model copy completion: release copy holds, leaving the cache owner and
    # request owner respectively. Real worker execution is an integration gate.
    source.ref_cnt -= 1
    cow.ref_cnt -= 1
    assert source.ref_cnt == cow.ref_cnt == 1


@pytest.mark.parametrize("local_tokens", [0, 16, 32])
def test_full_boundary_does_not_reserve_or_copy_a_partial_tail(local_tokens):
    manager = _manager()
    blocks = [
        SimpleNamespace(block_id=i + 1, ref_cnt=1, is_null=False) for i in range(local_tokens // manager.block_size)
    ]
    reserved = manager.get_num_blocks_to_allocate(
        "req", local_tokens + 1, blocks, local_tokens, local_tokens, local_tokens + 1
    )
    manager.add_local_computed_blocks("req", blocks, local_tokens, 0)
    manager.allocate_new_blocks("req", local_tokens + 1, local_tokens + 1)
    assert manager._partial_hit_reqs == {}
    assert manager._pending_cow_copies == []
    assert len(manager.block_pool.allocated) == reserved == 1


@pytest.mark.parametrize("registration", ["out_of_range", "stale", "null"])
def test_inconsistent_registration_fails_closed_instead_of_silently_skipping_cow(registration):
    manager = _manager()
    source = SimpleNamespace(block_id=1, ref_cnt=2, is_null=registration == "null")
    manager.req_to_blocks["req"] = [source]
    idx = 1 if registration == "out_of_range" else 0
    registered_source = SimpleNamespace(block_id=2, ref_cnt=2, is_null=False) if registration == "stale" else source
    manager._partial_hit_reqs["req"] = (idx, registered_source)
    with pytest.raises(AssertionError):
        manager.allocate_new_blocks("req", 32, 32)
    assert manager._pending_cow_copies == []
    assert manager.req_to_blocks["req"] == [source]


def test_mamba_align_keeps_its_specialized_upstream_allocator():
    upstream_mamba = AscendMambaManager.__mro__[1]
    assert AscendMambaManager.allocate_new_blocks is upstream_mamba.allocate_new_blocks
    assert upstream_mamba.allocate_new_blocks is not FullAttentionManager.allocate_new_blocks
