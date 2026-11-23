# SPDX-License-Identifier: Apache-2.0
from vllm.v1.core.kv_cache_utils import make_block_hash_with_group_id

from vllm_ascend.core.mamba_replay_cohort import MambaReplayCohortPool

PREFIX = b"prefix"


def make_pool(missing_group=None, position=6144):
    pool = MambaReplayCohortPool(16, True, 64)
    pool.enable_mamba_retention({1: 2048, 2: 2048, 3: 2048}, [0])
    attention, *rest = pool.get_new_blocks(15)
    recurrent, victim = rest[:3], rest[3]
    for group, block in enumerate((attention, *recurrent)):
        if group != missing_group:
            pool._insert_block_hash(make_block_hash_with_group_id(PREFIX, group), block, position)
    pool._insert_block_hash(make_block_hash_with_group_id(b"another", 0), victim, position)
    return pool, attention, recurrent, victim


def test_defers_the_whole_reusable_cohort():
    pool, attention, recurrent, victim = make_pool()
    pool.free_blocks([*recurrent, victim])
    assert pool.get_new_blocks(1) == [victim]
    assert pool.get_num_free_blocks() == 3
    assert pool.get_cached_block(PREFIX, [0, 1, 2, 3]) == [attention, *recurrent]
    assert pool.get_new_blocks(3) == recurrent


def test_incomplete_or_partial_cohort_does_not_defeat_fifo():
    for missing, position in ((3, 6144), (None, 6208)):
        pool, _, recurrent, victim = make_pool(missing, position)
        freed = recurrent[:2] if missing else recurrent
        pool.free_blocks([*freed, victim])
        assert pool.get_new_blocks(1) == recurrent[:1]


def test_capacity_wins_when_only_part_of_cohort_can_be_deferred():
    pool, _, recurrent, victim = make_pool()
    pool.free_blocks([*recurrent, victim])
    assert pool.get_new_blocks(2) == recurrent[:2]


def test_does_not_move_a_cohort_with_an_earlier_victim():
    pool, attention, recurrent, victim = make_pool()
    second_victim = pool.blocks[8]
    pool._insert_block_hash(make_block_hash_with_group_id(b"other", 0), second_victim, 6144)
    pool.free_blocks([attention, recurrent[0], victim, second_victim])
    assert pool.get_new_blocks(2) == [attention, recurrent[0]]


def test_one_deferral_per_registration():
    pool, _, recurrent, victim = make_pool()
    pool.free_blocks([*recurrent, victim])
    assert pool.get_new_blocks(1) == [victim]
    pool._insert_block_hash(make_block_hash_with_group_id(b"another", 0), victim, 6144)
    pool.free_blocks([victim])
    assert pool.get_new_blocks(1) == recurrent[:1]
