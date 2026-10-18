"""Shared Store source intervals for Worker reads and Scheduler source leases.

Both sides must select the same retained pages before the Worker materializes
addresses or the Scheduler pins sources. This module owns no execution state.
"""


def store_block_range(start_token: int, end_token: int, block_size: int, hash_count: int, block_count: int) -> range:
    """Intersect the Store interval, confirmed hashes, and retained block-table suffix."""

    logical_count = min(hash_count, -(-end_token // block_size))
    block_offset = max(logical_count - block_count, 0)
    return range(max(start_token // block_size, block_offset), logical_count)


def checkpoint_companion_indices(
    boundary_token: int, published_store_end_token: int, block_size: int, block_count: int
) -> range:
    """Locate the unpublished attention pages and the checkpoint's boundary page."""

    logical_count = -(-boundary_token // block_size)
    block_offset = max(logical_count - block_count, 0)
    first_block = max(min(published_store_end_token // block_size, logical_count - 1), block_offset)
    return range(first_block - block_offset, logical_count - block_offset)
