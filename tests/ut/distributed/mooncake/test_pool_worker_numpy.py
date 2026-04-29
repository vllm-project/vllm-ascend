from types import SimpleNamespace

import numpy as np
import torch

if not hasattr(torch, "npu"):
    torch.npu = SimpleNamespace(Event=object)  # type: ignore[attr-defined]

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker import (
    KVPoolWorker,
)


def test_process_chunks_incremental_uses_vectorized_layout():
    worker = KVPoolWorker.__new__(KVPoolWorker)
    worker.block_size = 16
    worker.page_size_bytes = 300
    worker.num_ranks_per_layer = 4
    worker.my_key_index = 2
    worker._block_len_np = np.asarray([100, 200], dtype=np.int64)
    worker._kv_caches_base_addr_np = np.asarray(
        [1000, 1500, 2000, 3000],
        dtype=np.int64,
    )
    worker._full_block_inner_offsets_np = np.asarray([0, 100], dtype=np.int64)

    tracker = {
        "processed_count": 0,
        "chunk_addr_list": [],
        "chunk_size_list": [],
        "chunk_gvas_list": [],
    }
    block_ids = [3, 5]
    chunk_gvas = [10000, 20000]

    worker._process_chunks_incremental(
        tracker,
        block_ids,
        layer_id=1,
        chunk_gvas=chunk_gvas,
        num_blocks=2,
        block_ids_np=np.asarray(block_ids, dtype=np.int64),
        chunk_gvas_np=np.asarray(chunk_gvas, dtype=np.int64),
    )

    assert tracker["processed_count"] == 2
    assert tracker["chunk_addr_list"] == [2300, 3600, 2500, 4000]
    assert tracker["chunk_size_list"] == [100, 200, 100, 200]
    assert tracker["chunk_gvas_list"] == [11800, 11900, 21800, 21900]


def test_process_chunks_incremental_appends_only_new_blocks():
    worker = KVPoolWorker.__new__(KVPoolWorker)
    worker.block_size = 16
    worker.page_size_bytes = 300
    worker.num_ranks_per_layer = 4
    worker.my_key_index = 2
    worker._block_len_np = np.asarray([100, 200], dtype=np.int64)
    worker._kv_caches_base_addr_np = np.asarray(
        [1000, 1500, 2000, 3000],
        dtype=np.int64,
    )
    worker._full_block_inner_offsets_np = np.asarray([0, 100], dtype=np.int64)

    tracker = {
        "processed_count": 1,
        "chunk_addr_list": [2100, 3200],
        "chunk_size_list": [100, 200],
        "chunk_gvas_list": [10800, 10900],
    }
    block_ids = [1, 3]
    chunk_gvas = [9000, 10000]

    worker._process_chunks_incremental(
        tracker,
        block_ids,
        layer_id=1,
        chunk_gvas=chunk_gvas,
        num_blocks=2,
        block_ids_np=np.asarray(block_ids, dtype=np.int64),
        chunk_gvas_np=np.asarray(chunk_gvas, dtype=np.int64),
    )

    assert tracker["processed_count"] == 2
    assert tracker["chunk_addr_list"] == [2100, 3200, 2300, 3600]
    assert tracker["chunk_size_list"] == [100, 200, 100, 200]
    assert tracker["chunk_gvas_list"] == [10800, 10900, 11800, 11900]


def test_process_chunks_incremental_batch_merges_requests():
    worker = KVPoolWorker.__new__(KVPoolWorker)
    worker.block_size = 16
    worker.page_size_bytes = 300
    worker.num_ranks_per_layer = 4
    worker.my_key_index = 2
    worker._block_len_np = np.asarray([100, 200], dtype=np.int64)
    worker._kv_caches_base_addr_np = np.asarray(
        [1000, 1500, 2000, 3000],
        dtype=np.int64,
    )
    worker._full_block_inner_offsets_np = np.asarray([0, 100], dtype=np.int64)

    first_tracker = {
        "processed_count": 0,
        "chunk_addr_list": [],
        "chunk_size_list": [],
        "chunk_gvas_list": [],
    }
    second_tracker = {
        "processed_count": 1,
        "chunk_addr_list": [2400, 3800],
        "chunk_size_list": [100, 200],
        "chunk_gvas_list": [12800, 12900],
    }

    worker._process_chunks_incremental_batch(
        [
            (
                first_tracker,
                [3],
                [10000],
                1,
                np.asarray([3], dtype=np.int64),
                np.asarray([10000], dtype=np.int64),
            ),
            (
                second_tracker,
                [4, 5],
                [11000, 12000],
                2,
                np.asarray([4, 5], dtype=np.int64),
                np.asarray([11000, 12000], dtype=np.int64),
            ),
        ],
        layer_id=1,
    )

    assert first_tracker["processed_count"] == 1
    assert first_tracker["chunk_addr_list"] == [2300, 3600]
    assert first_tracker["chunk_size_list"] == [100, 200]
    assert first_tracker["chunk_gvas_list"] == [11800, 11900]
    assert second_tracker["processed_count"] == 2
    assert second_tracker["chunk_addr_list"] == [2400, 3800, 2500, 4000]
    assert second_tracker["chunk_size_list"] == [100, 200, 100, 200]
    assert second_tracker["chunk_gvas_list"] == [12800, 12900, 13800, 13900]
