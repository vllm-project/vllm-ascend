# SPDX-License-Identifier: Apache-2.0
"""A5 packed-page allocation must preserve reusable prefix blocks."""

import pytest
from vllm.v1.core.kv_cache_utils import BlockHash, make_block_hash_with_group_id

from vllm_ascend.patch.platform.patch_kv_cache_coordinator import _GroupStableBlockPool


def _pool(num_gpu_blocks: int, *, allow_reassignment: bool = False) -> _GroupStableBlockPool:
    return _GroupStableBlockPool(
        num_gpu_blocks=num_gpu_blocks,
        enable_caching=True,
        hash_block_size=128,
        allow_group_reassignment=allow_reassignment,
    )


def test_unowned_pages_are_used_before_recycling_group_pages():
    pool = _pool(4)
    blocks = pool.get_new_blocks_for_group(2, group_id=0)
    pool.free_blocks_for_group(blocks, group_id=0)

    fresh = pool.get_new_blocks_for_group(1, group_id=0)

    assert fresh[0].block_id == 3


def test_touched_page_rejoins_at_tail_without_stale_queue_entry():
    pool = _pool(3)
    oldest, newer = pool.get_new_blocks_for_group(2, group_id=0)
    pool.free_blocks_for_group([oldest, newer], group_id=0)

    pool.touch([oldest])
    pool.free_blocks_for_group([oldest], group_id=0)

    recycled = pool.get_new_blocks_for_group(1, group_id=0)

    assert recycled[0] is newer


def test_reclaim_idle_group_pages_preserves_cached_prefix():
    pool = _pool(7, allow_reassignment=True)
    donor = pool.get_new_blocks_for_group(4, 0)
    cached = pool.get_new_blocks_for_group(2, 1)
    hashes = [BlockHash(b"prefix" + bytes([part]) + bytes(25)) for part in range(2)]
    for block, hash_key in zip(cached, hashes):
        pool._insert_block_hash(make_block_hash_with_group_id(hash_key, 1), block, num_tokens=128)
    pool.free_blocks_for_group(donor, 0)
    pool.free_blocks_for_group(cached, 1)

    reused = pool.get_new_blocks_for_group(1, 1)[0]

    assert reused.block_id in {b.block_id for b in donor}
    assert pool._owners[reused.block_id] == 1
    for block, hash_key in zip(cached, hashes):
        assert pool.get_cached_block(hash_key, [1])[0] is block


def test_failed_group_allocation_keeps_free_queues_intact():
    pool = _pool(7)
    donor = pool.get_new_blocks_for_group(4, 0)
    own = pool.get_new_blocks_for_group(2, 1)
    pool.free_blocks_for_group(donor, 0)
    pool.free_blocks_for_group(own, 1)

    with pytest.raises(ValueError, match="only 2"):
        pool.get_new_blocks_for_group(3, 1)

    assert pool.get_num_free_blocks() == 6
    assert {b.block_id for b in pool.get_new_blocks_for_group(2, 1)} == {b.block_id for b in own}


@pytest.mark.parametrize("lanes", [1, 16, 48])
def test_reclaim_idle_pages_keeps_repeated_prefix_at_concurrency(lanes):
    # A transient group briefly uses nine pages per lane; two full-attention
    # pages per lane hold its repeated prefix. The 110 usable pages can hold
    # every 48-lane prefix plus one transient request at the same time.
    pool = _pool(111, allow_reassignment=True)
    hashes = [
        [BlockHash(lane.to_bytes(2, "big") + part.to_bytes(2, "big") + bytes(28)) for part in range(2)]
        for lane in range(lanes)
    ]

    def transient_step():
        blocks = pool.get_new_blocks_for_group(9, 1)
        pool.free_blocks_for_group(blocks, 1)

    for lane in range(lanes):
        transient_step()
        blocks = pool.get_new_blocks_for_group(2, 0)
        for block, block_hash in zip(blocks, hashes[lane]):
            pool._insert_block_hash(make_block_hash_with_group_id(block_hash, 0), block, num_tokens=128)
        pool.free_blocks_for_group(blocks, 0)

    for lane in range(lanes):
        transient_step()
        cached = [pool.get_cached_block(block_hash, [0]) for block_hash in hashes[lane]]
        assert all(cached)
        blocks = [hit[0] for hit in cached]
        pool.touch(blocks)
        pool.free_blocks_for_group(blocks, 0)


def test_reassigned_cached_donor_loses_old_hash():
    pool = _pool(3, allow_reassignment=True)
    donor = pool.get_new_blocks_for_group(1, 0)[0]
    active = pool.get_new_blocks_for_group(1, 1)[0]
    hash_key = BlockHash(b"donor".ljust(32, b"\0"))
    pool._insert_block_hash(make_block_hash_with_group_id(hash_key, 0), donor, num_tokens=128)
    pool.free_blocks_for_group([donor], 0)

    reused = pool.get_new_blocks_for_group(1, 1)[0]

    assert reused is donor
    assert pool.get_cached_block(hash_key, [0]) is None
    assert pool._owners[active.block_id] == 1
    assert active.ref_cnt == 1


@pytest.mark.parametrize("allow_reassignment", [False, True])
def test_cached_page_eviction_respects_global_age_when_reassignment_is_safe(
    allow_reassignment,
):
    pool = _pool(5, allow_reassignment=allow_reassignment)
    older = pool.get_new_blocks_for_group(2, 0)
    newer = pool.get_new_blocks_for_group(2, 1)
    hashes = [BlockHash(bytes([part]) + bytes(31)) for part in range(4)]
    for block, hash_key in zip(older + newer, hashes):
        group = pool._owners[block.block_id]
        pool._insert_block_hash(make_block_hash_with_group_id(hash_key, group), block, num_tokens=128)
    pool.free_blocks_for_group(older, 0)
    pool.free_blocks_for_group(newer, 1)

    reused = pool.get_new_blocks_for_group(1, 1)[0]

    expected = older[0] if allow_reassignment else newer[0]
    assert reused is expected
    if allow_reassignment:
        for block, hash_key in zip(newer, hashes[2:]):
            assert pool.get_cached_block(hash_key, [1])[0] is block
        assert pool.get_cached_block(hashes[0], [0]) is None


@pytest.mark.parametrize("lanes", [1, 16, 48])
def test_cached_previous_cohort_does_not_evict_current_lane_prefixes(lanes):
    pool = _pool(111, allow_reassignment=True)
    previous = pool.get_new_blocks_for_group(110, 1)
    for block in previous:
        key = BlockHash(b"old" + block.block_id.to_bytes(2, "big") + bytes(27))
        pool._insert_block_hash(make_block_hash_with_group_id(key, 1), block, num_tokens=128)
    pool.free_blocks_for_group(previous, 1)

    lane_hashes = []
    for lane in range(lanes):
        hashes = [BlockHash(b"new" + lane.to_bytes(2, "big") + bytes([part]) + bytes(26)) for part in range(2)]
        blocks = pool.get_new_blocks_for_group(2, 0)
        for block, key in zip(blocks, hashes):
            pool._insert_block_hash(make_block_hash_with_group_id(key, 0), block, num_tokens=128)
        pool.free_blocks_for_group(blocks, 0)
        lane_hashes.append(hashes)

    for hashes in lane_hashes:
        # 96 prefix pages and this nine-page transient fit in the 110
        # usable pages. Older previous-cohort pages must go first.
        transient = pool.get_new_blocks_for_group(9, 1)
        pool.free_blocks_for_group(transient, 1)
        cached = [pool.get_cached_block(key, [0]) for key in hashes]
        assert all(cached)
        blocks = [hit[0] for hit in cached]
        pool.touch(blocks)
        pool.free_blocks_for_group(blocks, 0)
