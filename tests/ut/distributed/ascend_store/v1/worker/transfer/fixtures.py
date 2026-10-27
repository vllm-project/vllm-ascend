"""Tests for worker.transfer.fixtures."""

import numpy as np

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.worker.transfer.batch import (
    KVGroupBatch,
)


def make_group(group_id, request_transfer_record_boundaries, partition_count=2):
    transfer_record_count = request_transfer_record_boundaries[-1]
    return KVGroupBatch(
        group_id,
        np.arange(transfer_record_count, dtype=np.uint64) + 10,
        tuple(
            tuple(f"g{group_id}-axis{axis}-row{row}" for row in range(transfer_record_count))
            for axis in range(partition_count)
        ),
        np.array(request_transfer_record_boundaries, dtype=np.intp),
        (2, 5, 9),
        4096,
    )
