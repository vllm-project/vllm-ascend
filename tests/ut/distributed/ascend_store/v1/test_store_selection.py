"""Worker reads and Scheduler leases retain one shared source-interval rule."""

import pytest

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.scheduler import source_leases
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.store_selection import (
    checkpoint_companion_indices,
    store_block_range,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.worker.bulk import worker as bulk_worker
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.worker.transfer import rows


def test_worker_and_scheduler_share_source_selection_functions() -> None:
    assert source_leases.store_block_range is rows.store_block_range is store_block_range
    assert source_leases.checkpoint_companion_indices is checkpoint_companion_indices
    assert bulk_worker.checkpoint_companion_indices is checkpoint_companion_indices


@pytest.mark.parametrize(
    "start,end,hash_count,block_count,expected",
    ((0, 24, 6, 6, (0, 1, 2, 3, 4, 5)), (4, 24, 6, 2, (4, 5)), (8, 24, 4, 6, (2, 3)), (24, 24, 6, 6, ())),
)
def test_store_interval_intersects_hashes_and_retained_suffix(start, end, hash_count, block_count, expected) -> None:
    assert tuple(store_block_range(start, end, 4, hash_count, block_count)) == expected


@pytest.mark.parametrize(
    "boundary,published_end,block_count,expected",
    ((18, 0, 5, (0, 1, 2, 3, 4)), (18, 8, 2, (0, 1)), (18, 20, 5, (4,)), (16, 16, 4, (3,)), (18, 0, 0, ())),
)
def test_checkpoint_companions_keep_unpublished_pages_and_boundary(boundary, published_end, block_count, expected):
    assert tuple(checkpoint_companion_indices(boundary, published_end, 4, block_count)) == expected
