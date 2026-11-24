# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Physical registration boundaries for direct Mooncake Host transfers.

Rank selection and block mapping remain in the V2 planner.
"""

from bisect import bisect_right
from collections.abc import Sequence

MAX_REGISTER_MEMORY_BYTES = 64 * 1024**3


class TransferRegionIndex:
    """Validate and locate disjoint registered or directly mapped ranges."""

    def __init__(self, regions: Sequence[tuple[int, int]]) -> None:
        ordered = sorted(set(regions))
        self.starts: list[int] = []
        self.ends: list[int] = []
        for ptr, size in ordered:
            if ptr < 0 or size <= 0:
                raise ValueError("Invalid transfer memory range")
            if self.ends and ptr < self.ends[-1]:
                raise ValueError("Overlapping transfer memory ranges")
            self.starts.append(ptr)
            self.ends.append(ptr + size)

    def available(self, address: int) -> int:
        index = bisect_right(self.starts, address) - 1
        if index < 0 or address >= self.ends[index]:
            raise ValueError(f"Transfer address {address:#x} is outside registered or mapped memory")
        return self.ends[index] - address


def bound_register_regions(
    ptrs: list[int], lengths: list[int], *, max_region_bytes: int = MAX_REGISTER_MEMORY_BYTES
) -> tuple[list[int], list[int]]:
    """Split allocations at the same boundaries advertised to receivers."""
    if max_region_bytes <= 0:
        raise ValueError("Registration limit must be positive")
    regions = TransferRegionIndex(list(zip(ptrs, lengths, strict=True)))
    bounded_ptrs, bounded_lengths = [], []
    for ptr, end in zip(regions.starts, regions.ends, strict=True):
        while ptr < end:
            size = min(end - ptr, max_region_bytes)
            bounded_ptrs.append(ptr)
            bounded_lengths.append(size)
            ptr += size
    return bounded_ptrs, bounded_lengths


def split_transfer_ranges(
    local_addrs: list[int],
    remote_addrs: list[int],
    lengths: list[int],
    *,
    local_regions: TransferRegionIndex,
    remote_regions: TransferRegionIndex,
) -> tuple[list[int], list[int], list[int]]:
    """Validate the whole plan, then split READs at both endpoints' MR edges."""
    local_result, remote_result, length_result = [], [], []
    for local, remote, size in zip(local_addrs, remote_addrs, lengths, strict=True):
        if size <= 0:
            raise ValueError("Transfer length must be positive")
        remaining = size
        while remaining:
            count = min(remaining, local_regions.available(local), remote_regions.available(remote))
            local_result.append(local)
            remote_result.append(remote)
            length_result.append(count)
            local += count
            remote += count
            remaining -= count
    return local_result, remote_result, length_result
