"""Tests for worker.transfer.fixtures."""

import numpy as np

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.worker.transfer.batch import (
    KVGroupBatch,
)


def make_group(group_id, request_splits, axis_count=2):
    row_count = request_splits[-1]
    return KVGroupBatch(
        group_id,
        np.arange(row_count, dtype=np.uint64) + 10,
        np.full(row_count, 128, dtype=np.uint64),
        tuple(tuple(f"g{group_id}-axis{axis}-row{row}" for row in range(row_count)) for axis in range(axis_count)),
        np.array(request_splits, dtype=np.intp),
        (2, 5, 9),
        4096,
    )
