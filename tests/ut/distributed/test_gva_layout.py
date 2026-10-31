# SPDX-License-Identifier: Apache-2.0

from enum import IntEnum

import pytest

from vllm_ascend.distributed.gva import GvaRowPartition, GvaRowRange


class _IntegralValue(IntEnum):
    ZERO = 0
    ONE = 1
    THREE = 3


def test_row_range_normalizes_integral_fields():
    row_range = GvaRowRange(_IntegralValue.ONE, _IntegralValue.ZERO, _IntegralValue.THREE)

    assert type(row_range.owner_rank) is int
    assert type(row_range.start) is int
    assert type(row_range.end) is int


def test_balanced_partition_assigns_remainder_to_earlier_owners():
    partition = GvaRowPartition.balanced(10, (4, 7, 9))

    assert partition.ranges == (
        GvaRowRange(owner_rank=4, start=0, end=4),
        GvaRowRange(owner_rank=7, start=4, end=7),
        GvaRowRange(owner_rank=9, start=7, end=10),
    )
    assert partition.owner_and_local_row(0) == (4, 0)
    assert partition.owner_and_local_row(3) == (4, 3)
    assert partition.owner_and_local_row(4) == (7, 0)
    assert partition.owner_and_local_row(9) == (9, 2)


def test_partition_sorts_ranges_but_preserves_owner_order():
    partition = GvaRowPartition(
        global_rows=5,
        owner_ranks=(8, 2),
        ranges=(GvaRowRange(2, 2, 5), GvaRowRange(8, 0, 2)),
    )

    assert partition.owner_ranks == (8, 2)
    assert partition.ranges == (GvaRowRange(8, 0, 2), GvaRowRange(2, 2, 5))


@pytest.mark.parametrize(
    "ranges, message",
    [
        ((GvaRowRange(0, 0, 2), GvaRowRange(1, 3, 5)), "gap"),
        ((GvaRowRange(0, 0, 3), GvaRowRange(1, 2, 5)), "overlap"),
        ((GvaRowRange(0, 0, 2), GvaRowRange(1, 2, 4)), "expected global_rows=5"),
        ((GvaRowRange(0, 0, 5),), "exactly one row range"),
        ((GvaRowRange(0, 0, 2), GvaRowRange(0, 2, 5)), "exactly one range for each owner"),
    ],
)
def test_partition_rejects_invalid_ranges(ranges, message):
    owners = (0, 1)
    with pytest.raises(ValueError, match=message):
        GvaRowPartition(global_rows=5, owner_ranks=owners, ranges=ranges)


def test_partition_rejects_invalid_ids_and_out_of_range_rows():
    partition = GvaRowPartition.balanced(4, (0, 1))

    with pytest.raises(TypeError, match="integer"):
        GvaRowPartition.balanced(4, (0, True))
    with pytest.raises(IndexError, match="outside"):
        partition.owner_and_local_row(4)
    with pytest.raises(ValueError, match="unique"):
        GvaRowPartition.balanced(4, (0, 0))
    with pytest.raises(ValueError, match="smaller than owner count"):
        GvaRowPartition.balanced(1, (0, 1))
