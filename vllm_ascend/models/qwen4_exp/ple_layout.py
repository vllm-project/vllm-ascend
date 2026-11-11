# SPDX-License-Identifier: Apache-2.0
"""Address PLE rows without allowing one row to cross a mapping boundary."""

from dataclasses import dataclass


@dataclass(frozen=True)
class PLELayout:
    rows: int
    row_bytes: int
    mapping_bytes: int = 1 << 30

    def __post_init__(self):
        if self.rows <= 0 or self.row_bytes <= 0 or self.mapping_bytes < self.row_bytes:
            raise ValueError("Invalid PLE mapping dimensions")

    @property
    def rows_per_mapping(self) -> int:
        return self.mapping_bytes // self.row_bytes

    @property
    def mappings(self) -> int:
        return (self.rows + self.rows_per_mapping - 1) // self.rows_per_mapping

    @property
    def allocation_bytes(self) -> int:
        return self.mappings * self.mapping_bytes

    def offset(self, row: int) -> int:
        if not 0 <= row < self.rows:
            raise ValueError("PLE row is outside the table")
        mapping, within = divmod(row, self.rows_per_mapping)
        return mapping * self.mapping_bytes + within * self.row_bytes

    def spans(self, start: int, end: int):
        """Yield logical row ranges and byte offsets, split at mapping boundaries."""
        if not 0 <= start <= end <= self.rows:
            raise ValueError("PLE row range is outside the table")
        while start < end:
            next_boundary = (start // self.rows_per_mapping + 1) * self.rows_per_mapping
            stop = min(end, next_boundary)
            yield start, stop, self.offset(start)
            start = stop
