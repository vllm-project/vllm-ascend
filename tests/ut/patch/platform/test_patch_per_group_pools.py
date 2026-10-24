# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Independent pools must not lend sliding capacity to full attention."""

import pytest
from vllm.v1.core.block_pool import BlockPool

from vllm_ascend.patch.platform.patch_per_group_pools import (
    GroupBlockReservation,
    PerGroupBlockPools,
)


def _pool(capacity: int) -> BlockPool:
    return BlockPool(
        num_gpu_blocks=capacity,
        enable_caching=False,
        hash_block_size=16,
    )


def test_capacity_follows_full_pool_not_group_order_or_total_free():
    sliding, full = _pool(16), _pool(8)
    full.get_new_blocks(5)
    pools = PerGroupBlockPools([sliding, full], full_group_ids=(1,))

    # The null block and five live blocks leave two full blocks, not 17.
    assert pools.get_num_free_blocks() == 2
    assert pools.get_usage() == full.get_usage()


@pytest.mark.parametrize(
    ("free_sliding", "free_full", "need", "accepted"),
    [
        (12, 1, (1, 2), False),  # Total free fits; full does not.
        (0, 12, (1, 1), False),  # Full fits; sliding does not.
        (1, 2, (1, 2), True),  # Exact fit in each independent pool.
        (0, 2, (0, 2), True),  # A group with no new demand need not be free.
        (1, 1, (1, 1), True),  # Sum(need) > min(free) must not reject.
    ],
)
def test_admission_preserves_group_local_constraints(free_sliding, free_full, need, accepted):
    sliding, full = _pool(16), _pool(16)
    sliding.get_new_blocks(15 - free_sliding)
    full.get_new_blocks(15 - free_full)
    pools = PerGroupBlockPools([sliding, full], full_group_ids=(1,))

    assert pools.can_allocate(need) is accepted
    # Admission predicts only: it must not allocate or touch blocks.
    assert sliding.get_num_free_blocks() == free_sliding
    assert full.get_num_free_blocks() == free_full


def test_full_pressure_is_not_diluted_below_75_percent():
    sliding, full = _pool(100), _pool(20)
    full.get_new_blocks(18)
    pools = PerGroupBlockPools([sliding, full], full_group_ids=(1,))

    assert pools.get_usage() == pytest.approx(18 / 19)
    assert not pools.can_allocate((1, 2))


def test_additional_full_pool_is_checked_before_admission():
    sliding, full_a, full_b = _pool(16), _pool(16), _pool(8)
    full_b.get_new_blocks(7)
    pools = PerGroupBlockPools([sliding, full_a, full_b], full_group_ids=(1, 2))

    assert not pools.can_allocate((1, 1, 1))
    assert pools.get_usage() == full_b.get_usage()


@pytest.mark.parametrize("full_group_ids", [(), (-1,), (2,), (1, 1)])
def test_invalid_full_pool_selection_fails_before_scheduling(full_group_ids):
    with pytest.raises(ValueError):
        PerGroupBlockPools([_pool(8), _pool(16)], full_group_ids=full_group_ids)


def test_watermark_only_reserves_full_capacity():
    pools = PerGroupBlockPools([_pool(2), _pool(4)], full_group_ids=(1,))

    assert pools.can_allocate((1, 2), watermark_blocks=1)
    assert not pools.can_allocate((1, 3), watermark_blocks=1)


def test_inflight_sliding_reservation_cannot_be_spent_by_new_request():
    pools = PerGroupBlockPools([_pool(4), _pool(16)], full_group_ids=(1,))

    assert pools.can_allocate((2, 1))
    assert not pools.can_allocate((2, 1), GroupBlockReservation((2, 1)))
    assert pools.can_allocate((1, 1), GroupBlockReservation((2, 1)))


def test_ambiguous_scalar_reservation_is_not_silently_discarded():
    pools = PerGroupBlockPools([_pool(4), _pool(16)], full_group_ids=(1,))

    with pytest.raises(ValueError, match="reservation"):
        pools.can_allocate((1, 1), 1)


def test_admission_rejects_missing_group_demand():
    pools = PerGroupBlockPools([_pool(8), _pool(16)], full_group_ids=(1,))

    with pytest.raises(ValueError, match="group"):
        pools.can_allocate((1,))
