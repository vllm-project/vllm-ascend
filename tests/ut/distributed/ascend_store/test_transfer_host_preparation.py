# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import threading
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from tests.ut.distributed.ascend_store.test_dsv4_layerwise_runtime import make_worker
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise import kv_transfer
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.metadata import (
    LayerBatchReqMeta,
    LayerBlockRange,
    LayerTransferTask,
    SharedBlockData,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import ReqMeta


def make_builder():
    # Unequal cache components, padded device strides and one empty layer.
    database = SimpleNamespace(
        group_block_len={0: [8, 2, 16]},
        group_kv_caches_base_addr={0: [1000, 2000, 3000]},
        group_block_stride={0: [32, 32, 64]},
        group_layer_cache_entry_offsets={0: [0, 2, 2, 3]},
    )
    return kv_transfer.LayerBatchBuilder(database, 26, 3, layer_byte_offset=100)


def test_gva_layout_keeps_payload_stride_and_rank_offset_distinct(monkeypatch):
    builder = make_builder()

    class NoDebugList(np.ndarray):
        def tolist(self):
            raise AssertionError("Disabled DEBUG must not materialize the address list")

    monkeypatch.setattr(kv_transfer.logger, "isEnabledFor", lambda level: False)
    shared = SharedBlockData(
        block_ids_arr=np.array([2, 4]),
        block_gvas_arr=np.array([10000, 20000]).view(NoDebugList),
        req_ids=["r"],
        is_last_chunks=[False],
    )
    first = builder.build_addrs(shared, 0)
    np.testing.assert_array_equal(first.addr_array, [1064, 2064, 1128, 2128])
    np.testing.assert_array_equal(first.size_array, [8, 2, 8, 2])
    np.testing.assert_array_equal(first.gvas_array, [10100, 10108, 20100, 20108])
    empty = builder.build_addrs(shared, 1)
    assert empty.gvas_array.size == empty.addr_array.size == empty.size_array.size == 0
    last = builder.build_addrs(shared, 2)
    np.testing.assert_array_equal(last.addr_array, [3128, 3256])
    np.testing.assert_array_equal(last.gvas_array, [10110, 20110])
    # A new lease may relocate its object. Layout caching must not cache GVAs.
    shared.block_gvas_arr = np.array([30000, 40000])
    np.testing.assert_array_equal(builder.build_addrs(shared, 2).gvas_array, [30110, 40110])


def test_key_major_layout_does_not_apply_memcache_rank_offset():
    builder = make_builder()
    shared = SharedBlockData(
        block_ids_arr=np.array([2]), block_gvas_arr=None, block_keys=["key"], req_ids=["r"], is_last_chunks=[False]
    )
    first = builder.build_addrs(shared, 0)
    assert first.all_offsets == [[0, 8]]
    assert first.all_sizes == [[8, 2]]
    assert first.all_buffers == [[1064, 2064]]
    assert builder.build_addrs(shared, 2).all_offsets == [[10]]


def test_disjoint_ranges_collect_publication_keys_once_without_dropping_payload():
    builder = make_builder()
    request = ReqMeta("r", target_token_len=384, block_ids_by_group_np=[np.array([2, 3, 4])])
    request.block_gvas_by_group_np = [np.array([100, 200, 300])]
    request.save_keys = ["a", "b", "tail"]
    task = LayerTransferTask(
        layer_id=0,
        group_id=0,
        layer_idx_in_group=0,
        block_ranges=[LayerBlockRange(request, 0, 1), LayerBlockRange(request, 2, 3)],
    )
    shared = builder.build_shared(task)
    assert shared.save_keys == ["a", "b", "tail"]
    np.testing.assert_array_equal(shared.block_ids_arr, [2, 4])
    np.testing.assert_array_equal(shared.block_gvas_arr, [100, 300])


@pytest.mark.parametrize(
    "mask,expected",
    [(None, [0, 1, 2, 3]), ([False, True], [1, 2, 3]), ([False] * 4, []), ([True, False, False, True], [0, 3])],
)
def test_save_selection_keeps_unmasked_suffix_and_public_key_identity(mask, expected):
    worker = make_worker()
    worker.layerwise_offload = False
    request = ReqMeta(
        "r",
        target_token_len=512,
        save_end_token=512,
        block_ids_by_group_np=[np.array([2, 3, 4, 5])],
        block_hashes=["a", "b", "c", "d"],
        can_save=True,
    )
    request.store_masks = None if mask is None else [mask]
    plan = worker._plan_save_group(request, 0)
    assert list(plan.keys) == expected
    assert list(plan.keys.values()) == ["full:" + request.block_hashes[i] for i in expected]


def test_sender_assembles_host_arrays_before_wait_but_never_copies_early(monkeypatch):
    trace = []
    sender = kv_transfer.KVCacheStoreLayerSendingThread.__new__(kv_transfer.KVCacheStoreLayerSendingThread)
    sender.save_disabled = threading.Event()
    sender.max_transfer_blocks = sender.max_transfer_bytes = 0
    sender.sync_save_events = [SimpleNamespace(synchronize=lambda: trace.append("writes_finished"))]
    metas = [
        LayerBatchReqMeta(
            req_ids=["r"],
            layer_id=0,
            gvas_array=np.array([100 + i]),
            addr_array=np.array([200 + i]),
            size_array=np.array([32]),
        )
        for i in range(2)
    ]
    sender.group_builders = [SimpleNamespace(build_addrs=Mock(return_value=m)) for m in metas]
    shared = SimpleNamespace(save_keys=[])
    tasks = [
        SimpleNamespace(layer_id=0, group_id=i, layer_idx_in_group=0, shared_block_data=shared, write_finish_keys=[])
        for i in range(2)
    ]
    concatenate = np.concatenate

    def assemble(arrays):
        trace.append("host_assembly")
        return concatenate(arrays)

    def copy(gvas, addresses, sizes, direction, *limits):
        trace.append("device_copy")
        assert direction == 0
        np.testing.assert_array_equal(gvas, [100, 101])
        return 0

    monkeypatch.setattr(kv_transfer.np, "concatenate", assemble)
    sender._batch_copy_with_limits = copy
    assert sender._copy_gva_layers(tasks)
    assert trace == ["host_assembly"] * 3 + ["writes_finished", "device_copy"]
