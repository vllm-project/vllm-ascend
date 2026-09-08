# SPDX-License-Identifier: Apache-2.0

import importlib.util
import sys
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).parents[3] / "vllm_ascend" / "core" / "typed_kv_cache.py"
SPEC = importlib.util.spec_from_file_location("typed_kv_cache_under_test", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
typed_kv_cache = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = typed_kv_cache
SPEC.loader.exec_module(typed_kv_cache)

TypedBlockId = typed_kv_cache.TypedBlockId
TypedKVCachePlan = typed_kv_cache.TypedKVCachePlan
TypedPageSpec = typed_kv_cache.TypedPageSpec
TypedSuperpagePool = typed_kv_cache.TypedSuperpagePool
TypedRegionPool = typed_kv_cache.TypedRegionPool
TypedAddressPool = typed_kv_cache.TypedAddressPool
make_block_byte_view = typed_kv_cache.make_block_byte_view
make_group_byte_view = typed_kv_cache.make_group_byte_view
typed_block_ids_to_zero = typed_kv_cache.typed_block_ids_to_zero


def _plan(num_superpages: int = 4):
    specs = (
        TypedPageSpec(group_id=0, page_size_bytes=12, block_size_tokens=4),
        TypedPageSpec(group_id=1, page_size_bytes=8, block_size_tokens=2),
    )
    return TypedKVCachePlan.exact_lcm(specs, total_memory_bytes=24 * num_superpages)


def test_exact_lcm_plan_translates_typed_and_dense_ids() -> None:
    plan = _plan()
    attention = TypedBlockId(group_id=0, superpage_id=2, slot=1)
    mamba = TypedBlockId(group_id=1, superpage_id=2, slot=2)

    assert plan.superpage_size_bytes == 24
    assert plan.capacity(0) == 2
    assert plan.capacity(1) == 3
    assert plan.to_group_local_id(attention) == 5
    assert plan.to_group_local_id(mamba) == 8
    assert plan.from_group_local_id(0, 5) == attention
    assert plan.from_group_local_id(1, 8) == mamba
    assert plan.byte_offset(attention) == 60
    assert plan.byte_offset(mamba) == 64


def test_slot_mapping_preserves_existing_kernel_formula() -> None:
    plan = _plan()

    assert plan.slot_mapping(group_id=0, block_id=5, offset=3) == 23
    assert plan.slot_mapping(group_id=1, block_id=8, offset=1) == 17


def test_prefix_hit_pages_are_excluded_from_worker_zeroing() -> None:
    assert typed_block_ids_to_zero(
        [2, 3, 4, 0, 5],
        num_computed_tokens=8,
        block_size_tokens=4,
        prefix_caching_enabled=True,
    ) == (4, 5)
    assert typed_block_ids_to_zero(
        [2, 3, 4],
        num_computed_tokens=8,
        block_size_tokens=4,
        prefix_caching_enabled=False,
    ) == (2, 3, 4)


def test_prefix_zeroing_rejects_unaligned_or_oversized_hits() -> None:
    with pytest.raises(ValueError, match="aligned"):
        typed_block_ids_to_zero(
            [2, 3],
            num_computed_tokens=6,
            block_size_tokens=4,
            prefix_caching_enabled=True,
        )
    with pytest.raises(ValueError, match="exceeds"):
        typed_block_ids_to_zero(
            [2],
            num_computed_tokens=8,
            block_size_tokens=4,
            prefix_caching_enabled=True,
        )


def test_group_facades_allocate_dense_ids_from_typed_superpages() -> None:
    pool = TypedSuperpagePool(_plan())
    attention = pool.for_group(0).get_new_blocks(3)
    mamba = pool.for_group(1).get_new_blocks(2)

    # Superpage zero is reserved for the shared NULL address. Attention fills
    # superpage one and spills into two; Mamba therefore starts at three.
    assert [block.block_id for block in attention] == [2, 3, 4]
    assert [block.block_id for block in mamba] == [9, 10]
    assert pool.get_num_free_superpages() == 0


def test_empty_superpage_can_be_retyped_after_free() -> None:
    pool = TypedSuperpagePool(_plan(num_superpages=3))
    attention_pool = pool.for_group(0)
    mamba_pool = pool.for_group(1)
    blocks = attention_pool.get_new_blocks(2)
    attention_pool.free_blocks(reversed(blocks))

    mamba_blocks = mamba_pool.get_new_blocks(3)

    assert [block.block_id for block in mamba_blocks] == [3, 4, 5]
    assert pool.get_num_free_superpages() == 1


def test_multi_group_admission_is_measured_in_superpages() -> None:
    pool = TypedSuperpagePool(_plan(num_superpages=5))
    pool.for_group(0).get_new_blocks(1)

    # One free Attention slot remains in its current superpage. The additional
    # Attention block costs no superpage, while four Mamba blocks cost two.
    assert pool.additional_superpages_required({0: 1, 1: 4}) == 2
    assert pool.additional_superpages_required({0: 2, 1: 1}) == 2


def test_wrong_group_free_and_oom_do_not_corrupt_pool() -> None:
    pool = TypedSuperpagePool(_plan(num_superpages=2))
    attention_block = pool.for_group(0).get_new_blocks(1)[0]

    with pytest.raises(ValueError, match="different typed group"):
        pool.for_group(1).free_blocks([attention_block])
    with pytest.raises(ValueError, match="cannot allocate"):
        pool.for_group(1).get_new_blocks(1)

    assert attention_block.ref_cnt == 1
    assert pool.get_num_free_superpages() == 0


def test_partitioned_plan_uses_dense_group_local_ids() -> None:
    specs = (
        TypedPageSpec(group_id=0, page_size_bytes=12, block_size_tokens=4),
        TypedPageSpec(group_id=1, page_size_bytes=8, block_size_tokens=2),
    )
    plan = TypedKVCachePlan.partitioned(specs, 80, (4, 4))

    assert plan.is_partitioned
    assert plan.total_managed_bytes == 80
    assert plan.region_offsets_bytes == (0, 48)
    assert plan.byte_offset(TypedBlockId(1, 0, 2)) == 64
    assert plan.slot_mapping(1, 2, 1) == 5

    pool = TypedRegionPool(plan)
    blocks = pool.for_group(0).get_new_blocks(2)
    assert [block.block_id for block in blocks] == [1, 2]
    assert pool.can_allocate({0: 1, 1: 3})
    assert not pool.can_allocate({0: 2, 1: 3})
    pool.for_group(0).free_blocks(blocks)
    assert pool.for_group(0).get_num_free_blocks() == 3


def _addressed_plan():
    specs = (
        TypedPageSpec(group_id=0, page_size_bytes=12, block_size_tokens=4),
        TypedPageSpec(group_id=1, page_size_bytes=8, block_size_tokens=2),
    )
    return TypedKVCachePlan.addressed(
        specs,
        total_memory_bytes=96,
        page_address_tables_bytes=(
            (0, 24, 48, 72),
            (0, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88),
        ),
    )


def test_addressed_plan_translates_logical_ids_to_physical_pages() -> None:
    plan = _addressed_plan()

    assert plan.is_addressed
    assert plan.byte_offset(TypedBlockId(0, 0, 2)) == 48
    assert plan.physical_block_id(0, 2) == 4
    assert plan.physical_block_id(1, 7) == 8
    assert plan.slot_mapping(0, 2, 3) == 19
    assert plan.kernel_page_address_table(0) == (0, 2, 4, 6)
    assert plan.kernel_page_address_table(1, 2)[14:16] == (16, 17)


def test_address_pool_retypes_released_physical_interval() -> None:
    plan = _addressed_plan()
    pool = TypedAddressPool(plan)
    attention = pool.for_group(0).get_new_blocks(1)

    assert attention[0].block_id == 1
    assert plan.byte_offset(TypedBlockId(0, 0, attention[0].block_id)) == 24

    pool.for_group(0).free_blocks(attention)
    mamba = pool.for_group(1).get_new_blocks(9)
    mamba_offsets = {plan.byte_offset(TypedBlockId(1, 0, block.block_id)) for block in mamba}

    assert 24 in mamba_offsets
    assert pool.get_usage() > 0


def test_address_pool_zero_demand_does_not_consume_candidate_intervals() -> None:
    plan = _addressed_plan()
    pool = TypedAddressPool(plan)

    assert pool.for_group(0).get_new_blocks(0) == []
    assert pool.can_allocate({0: 0, 1: 9})
    assert len(pool.for_group(1).get_new_blocks(9)) == 9


def test_addressed_kernel_and_block_views_share_the_raw_arena() -> None:
    import torch

    plan = _addressed_plan()
    raw = torch.arange(96, dtype=torch.uint8)

    assert make_group_byte_view(raw, plan, 0).shape == (7, 12)
    assert make_group_byte_view(raw, plan, 1).shape == (12, 8)
    assert torch.equal(make_block_byte_view(raw, plan, 0, 2), raw[48:60])
