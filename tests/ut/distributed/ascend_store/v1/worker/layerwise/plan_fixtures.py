"""Adapt historical Mapping snapshots only for paired CPU benchmarks."""

import numpy as np

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.worker.layerwise.transfer import (
    make_layer_transfer_group,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.worker.transfer.batch import KVGroupBatch


def prepare_layer_group(mapping, keys, local_block_ids, *, selected_keys=None):
    local = mapping.local_layout
    ids = np.asarray(local_block_ids, dtype=np.uint64)
    batch = KVGroupBatch(
        local.group_id,
        ids,
        tuple(tuple(partition_keys) for partition_keys in keys),
        np.array([0, len(ids)], dtype=np.intp),
        local.physical_layer_ids,
        mapping.remote_object_layouts[0].size_bytes,
        None if selected_keys is None else np.asarray(selected_keys, dtype=np.bool_),
        physical_layer_ids_by_partition=local.physical_layer_ids_by_partition,
    )
    return make_layer_transfer_group(batch, selected_keys=batch.selected_keys())


def layer_arguments(mapping, group, *, layer_id, layouts=None):
    keys = [
        key
        for layout in mapping.layer_layouts.get(layer_id, ())
        for key in group.selected_keys[group.selected_transfer_slices_by_partition[layout.partition_index]]
    ]
    addresses, sizes, offsets = [], [], []
    for layout in mapping.layer_layouts.get(layer_id, ()):
        ids = group.selected_local_block_ids_by_partition[layout.partition_index]
        if not len(ids):
            continue
        rows = layout.local_segments.addresses(ids)
        addresses.extend(rows.tolist())
        sizes.extend(np.broadcast_to(layout.local_segments.segment_lengths_bytes, rows.shape).tolist())
        offsets.extend(np.broadcast_to(layout.remote_segment_offsets_bytes, rows.shape).tolist())
    return keys, addresses, sizes, offsets
