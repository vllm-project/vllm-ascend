"""Build a Worker batch for complete-payload byte-oracle tests."""

import numpy as np

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.worker.bulk.transfer import materialize_bulk_arguments
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.worker.transfer.batch import (
    KVGroupBatch,
    KVTransferBatch,
)


def bulk_arguments(mapping, keys, local_block_ids, *, selected_keys=None):
    local = mapping.local_layout
    ids = np.asarray(local_block_ids, dtype=np.uint64)
    group = KVGroupBatch(
        local.group_id,
        ids,
        np.full(len(ids), local.group_block_size, dtype=np.uint64),
        keys,
        np.array([0, len(ids)], dtype=np.intp),
        local.physical_layer_ids,
        local.local_group_block_bytes,
        None if selected_keys is None else np.asarray(selected_keys, dtype=np.bool_),
        physical_layer_ids_by_partition=local.physical_layer_ids_by_partition,
    )
    result = materialize_bulk_arguments({local.group_id: mapping}, KVTransferBatch(("request",), (group,)))
    return result.keys, result.addresses, result.sizes
