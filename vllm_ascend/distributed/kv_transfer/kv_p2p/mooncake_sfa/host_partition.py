# SPDX-License-Identifier: Apache-2.0
"""Assign Manager-owned Host destinations to disjoint TP receivers."""


def partition_shared_block_groups(
    local_groups: list[list[int]],
    remote_groups: list[list[int]],
    rank: int,
    size: int,
    *,
    coalesce: bool = True,
) -> tuple[list[list[int]], list[list[int]]]:
    """Assign balanced consecutive ranges of sorted destination blocks.

    Keep every source/destination pair together when sorting. Repeated
    destinations stay on one rank and preserve their original source order.
    The caller disables coalescing when adjacent block IDs do not describe
    adjacent byte ranges, for example with strided or partial-block copies.
    """
    if size < 1 or not 0 <= rank < size:
        raise ValueError("Invalid TP ownership group")
    if len(local_groups) != len(remote_groups):
        raise ValueError("Shared Host block group counts differ")
    sources_by_destination: dict[int, list[int]] = {}
    for local, remote in zip(local_groups, remote_groups):
        if len(local) != len(remote):
            raise ValueError("Shared Host block group lengths differ")
        for dst, src in zip(local, remote):
            if dst < 0 or src < 0:
                raise ValueError("Shared Host block IDs must be nonnegative")
            sources_by_destination.setdefault(dst, []).append(src)
    destinations = sorted(sources_by_destination)
    blocks_per_rank, remainder = divmod(len(destinations), size)
    start = rank * blocks_per_rank + min(rank, remainder)
    count = blocks_per_rank + int(rank < remainder)
    local_result: list[list[int]] = []
    remote_result: list[list[int]] = []
    for dst in destinations[start : start + count]:
        for src in sources_by_destination[dst]:
            if coalesce and local_result and dst == local_result[-1][-1] + 1 and src == remote_result[-1][-1] + 1:
                local_result[-1].append(dst)
                remote_result[-1].append(src)
            else:
                local_result.append([dst])
                remote_result.append([src])
    return local_result, remote_result
