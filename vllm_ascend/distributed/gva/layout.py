# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pure helpers for validating global-row ownership used by GVA tables."""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from numbers import Integral


def _require_int(value: object, name: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {result}")
    return result


@dataclass(frozen=True, slots=True)
class GvaRowRange:
    """A contiguous half-open global-row range owned by one rank."""

    owner_rank: int
    start: int
    end: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "owner_rank", _require_int(self.owner_rank, "owner_rank", minimum=0))
        object.__setattr__(self, "start", _require_int(self.start, "start", minimum=0))
        object.__setattr__(self, "end", _require_int(self.end, "end", minimum=1))
        if self.end <= self.start:
            raise ValueError(f"row range must be non-empty, got [{self.start}, {self.end})")


@dataclass(frozen=True, slots=True)
class GvaRowPartition:
    """A complete contiguous partition of a table's global row space.

    Each owner has exactly one non-empty contiguous range. Model adapters
    derive these ranges from their own ownership rules; this type only checks
    that the resulting layout covers every row exactly once.
    """

    global_rows: int
    owner_ranks: tuple[int, ...]
    ranges: tuple[GvaRowRange, ...]

    def __post_init__(self) -> None:
        global_rows = _require_int(self.global_rows, "global_rows", minimum=1)
        if not self.owner_ranks:
            raise ValueError("owner_ranks must not be empty")
        owner_ranks = tuple(_require_int(rank, "owner_rank", minimum=0) for rank in self.owner_ranks)
        if len(set(owner_ranks)) != len(owner_ranks):
            raise ValueError("owner_ranks must be unique")
        if len(self.ranges) != len(owner_ranks):
            raise ValueError("each owner rank must have exactly one row range")

        ranges = tuple(sorted(self.ranges, key=lambda row_range: row_range.start))
        actual_owners = tuple(row_range.owner_rank for row_range in ranges)
        if set(actual_owners) != set(owner_ranks) or len(set(actual_owners)) != len(actual_owners):
            raise ValueError("row ranges must contain exactly one range for each owner rank")

        cursor = 0
        for row_range in ranges:
            if row_range.start != cursor:
                relation = "overlap" if row_range.start < cursor else "gap"
                raise ValueError(f"row partition has a {relation} at global row {cursor}")
            cursor = row_range.end
        if cursor != global_rows:
            raise ValueError(f"row partition ends at {cursor}; expected global_rows={global_rows}")

        object.__setattr__(self, "global_rows", global_rows)
        object.__setattr__(self, "owner_ranks", owner_ranks)
        object.__setattr__(self, "ranges", ranges)

    @classmethod
    def balanced(cls, global_rows: int, owner_ranks: tuple[int, ...]) -> GvaRowPartition:
        """Build a balanced contiguous partition, assigning remainder rows first."""
        global_rows = _require_int(global_rows, "global_rows", minimum=1)
        if not owner_ranks:
            raise ValueError("owner_ranks must not be empty")
        owners = tuple(_require_int(rank, "owner_rank", minimum=0) for rank in owner_ranks)
        if len(set(owners)) != len(owners):
            raise ValueError("owner_ranks must be unique")
        if global_rows < len(owners):
            raise ValueError(f"global_rows={global_rows} is smaller than owner count={len(owners)}")

        base, remainder = divmod(global_rows, len(owners))
        ranges = []
        start = 0
        for index, owner_rank in enumerate(owners):
            end = start + base + (index < remainder)
            ranges.append(GvaRowRange(owner_rank, start, end))
            start = end
        return cls(global_rows, owners, tuple(ranges))

    def owner_and_local_row(self, global_row: int) -> tuple[int, int]:
        """Return the row owner and its owner-local row index."""
        global_row = _require_int(global_row, "global_row", minimum=0)
        if global_row >= self.global_rows:
            raise IndexError(f"global_row={global_row} is outside [0, {self.global_rows})")
        starts = [row_range.start for row_range in self.ranges]
        index = bisect_right(starts, global_row) - 1
        row_range = self.ranges[index]
        return row_range.owner_rank, global_row - row_range.start

    def owner_range(self, owner_rank: int) -> GvaRowRange:
        """Return one owner's range or raise when it is not in this partition."""
        owner_rank = _require_int(owner_rank, "owner_rank", minimum=0)
        for row_range in self.ranges:
            if row_range.owner_rank == owner_rank:
                return row_range
        raise KeyError(f"owner_rank={owner_rank} is not part of this row partition")
