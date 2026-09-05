#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#

import threading
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheGroupSpec, MambaSpec, UniformTypeKVCacheSpecs

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store import pool_worker as worker_module
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.kv_transfer import (
    KVCacheStoreKeyLayerSendingThread,
    KVCacheStoreLayerRecvingThread,
    KVCacheStoreLayerSendingThread,
    KVCacheStoreSendingThread,
    LayerBatchBuilder,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import (
    AscendConnectorMetadata,
    ChunkedTokenDatabase,
    KeyMetadata,
    LayerBlockRange,
    LayerTransferTask,
    LoadSpec,
    ReqMeta,
    SharedBlockData,
    get_partial_block_index,
)


@pytest.fixture
def lookup_worker():
    worker = worker_module.KVPoolWorker.__new__(worker_module.KVPoolWorker)
    worker.token_database = ChunkedTokenDatabase([KeyMetadata("model", 0, 0, 0, 0)], [4], None)
    worker.m_store = MagicMock()
    worker.cache_coordinator = None
    worker.cache_transfer_granularity = 8
    worker.group_uses_align_state = [False]
    worker.num_kv_cache_groups = 1
    worker.hash_block_size = 4
    worker.num_layers = 2
    worker.tp_mismatch = False
    worker.tp_size = worker.pp_size = worker.dcp_size = worker.num_kv_head = 1
    worker.tp_rank = 0
    worker.use_mla = worker.use_sparse = False
    worker.max_model_len = 64
    return worker


@pytest.fixture
def layer_worker(lookup_worker):
    worker = lookup_worker
    worker.token_database.set_group_buffers({0: [100, 200]}, {0: [8, 8]}, {0: [12, 12]}, group_num_layers={0: 2})
    worker.layer_save_finished_events = [threading.Event(), threading.Event()]
    worker.layer_load_finished_events = [threading.Event(), threading.Event()]
    worker.sync_save_events = [MagicMock(), MagicMock()]
    worker.kv_send_thread = KVCacheStoreLayerSendingThread(
        worker.m_store,
        worker.token_database,
        4,
        0,
        1,
        1,
        16,
        threading.Event(),
        2,
        worker.layer_save_finished_events,
        worker.sync_save_events,
    )
    worker.kv_recv_thread = KVCacheStoreLayerRecvingThread(
        worker.m_store,
        worker.token_database,
        4,
        0,
        1,
        1,
        16,
        threading.Event(),
        threading.Event(),
        worker.layer_load_finished_events,
        worker.layer_save_finished_events,
        worker.sync_save_events,
        2,
    )
    worker.layer_save_tasks = [[], []]
    worker.layer_load_tasks = [[], []]
    worker.current_layer = worker.next_layer_to_submit = 0
    worker.num_prefetch_layers = 1
    worker.prefetch_layer_map = {}
    worker.external_slot_release_waiter = None
    return worker


@pytest.mark.parametrize("hybrid", [False, True])
@pytest.mark.parametrize("result", [None, [1, 0], [0, 0], [0, 0, 1]])
def test_bulk_load_records_single_group_errors_without_invalidating_hybrid_ids(layer_worker, hybrid, result):
    worker = layer_worker
    worker.use_layerwise = worker.load_async = False
    worker.grouped_block_size = [4]
    worker._invalid_block_ids = set()
    worker.m_store.get.return_value = result
    request = ReqMeta(
        "r",
        token_len_chunk=8,
        block_hashes=[b"a", b"b"],
        block_ids_by_group=[[2, 3], [4, 5]] if hybrid else [[2, 3]],
        load_spec=LoadSpec(0, 8, True),
        kv_cache_group_ids=[0],
    )
    metadata = AscendConnectorMetadata(set())
    metadata.add_request(request)
    worker.start_load_kv(metadata)
    expected = set() if hybrid or result in ([0, 0], [0, 0, 1]) else ({2, 3} if result is None else {2})
    assert worker._invalid_block_ids == expected
    keys, addresses, sizes = worker.m_store.get.call_args.args
    assert len(keys) == 2
    assert addresses == [[124, 224], [136, 236]]
    assert sizes == [[8, 8], [8, 8]]
    assert request.load_spec.token_len == 8
    assert request.skip_null_blocks_by_group is worker.group_uses_align_state


@pytest.mark.parametrize("case", ["no_spec", "cannot_load", "no_hash", "invalid_group", "full_hit_tail"])
def test_bulk_load_skips_inapplicable_requests_and_recovers_full_hit_tail(layer_worker, case):
    worker = layer_worker
    worker.use_layerwise = worker.load_async = False
    worker.grouped_block_size = [4]
    worker._invalid_block_ids = set()
    worker.m_store.get.return_value = [0, 0]
    request = ReqMeta(
        "r", token_len_chunk=8, block_ids=[2, 3], block_hashes=[b"a", b"b"], load_spec=LoadSpec(0, 7, True)
    )
    if case == "no_spec":
        request.load_spec = None
    elif case == "cannot_load":
        request.load_spec.can_load = False
    elif case == "no_hash":
        request.block_hashes = []
    elif case == "invalid_group":
        request.kv_cache_group_ids = [1]
    metadata = AscendConnectorMetadata(set())
    metadata.add_request(request)
    worker.start_load_kv(metadata)
    if case == "full_hit_tail":
        assert request.load_spec.token_len == 8
        assert len(worker.m_store.get.call_args.args[0]) == 2
    else:
        worker.m_store.get.assert_not_called()
    assert worker._invalid_block_ids == set()


@pytest.mark.parametrize("operation", ["save", "load"])
@pytest.mark.parametrize("empty", [False, True])
def test_shared_layer_data_built_once_and_attached_to_matching_group(layer_worker, operation, empty):
    worker = layer_worker
    request = ReqMeta(
        "r",
        block_ids_np=np.array([2]),
        block_gvas_np=np.array([300]),
        load_block_gvas_np=np.array([400]),
    )
    request.save_keys, request.load_keys = ["saved"], ["loaded"]
    tasks = [
        [LayerTransferTask(layer, [] if empty else [LayerBlockRange(request, 0, 1)], group_id=0)] for layer in range(2)
    ]
    # A second group has no scheduled transfer in this step.
    worker.num_kv_cache_groups = 2
    setattr(worker, f"layer_{operation}_tasks", tasks)
    getattr(worker, f"_build_shared_{operation}_data")()
    shared = tasks[0][0].shared_block_data
    assert tasks[1][0].shared_block_data is shared
    if empty:
        assert shared is None
    else:
        np.testing.assert_array_equal(shared.block_ids_arr, [2])
        np.testing.assert_array_equal(shared.block_gvas_arr, [300 if operation == "save" else 400])
    if operation == "save":
        assert tasks[0][0].write_finish_keys == []
        assert tasks[1][0].write_finish_keys == ([] if empty else ["saved"])


@pytest.mark.parametrize("empty", ["all", "ranges", "none"])
def test_key_layer_shared_token_cache_is_reused_without_changing_worker_tasks(layer_worker, empty):
    worker = layer_worker
    worker.kv_send_thread = KVCacheStoreKeyLayerSendingThread(
        worker.m_store,
        worker.token_database,
        4,
        0,
        1,
        1,
        1,
        threading.Event(),
        2,
        worker.layer_save_finished_events,
        worker.sync_save_events,
    )
    request = ReqMeta("r", block_ids=[2], block_hashes=[b"a"], token_len_chunk=4)
    if empty != "all":
        worker.layer_save_tasks[1] = [
            LayerTransferTask(1, [] if empty == "ranges" else [LayerBlockRange(request, 0, 1)])
        ]
    worker._build_shared_save_data()
    if empty == "none":
        cached = worker.layer_save_tasks[1][0].cached_process_tokens
        assert cached is not None
        assert len(cached) == 1
    elif empty == "ranges":
        assert worker.layer_save_tasks[1][0].cached_process_tokens is None
    assert worker.kv_send_thread.request_queue.empty()


@pytest.mark.parametrize("operation", ["save", "load"])
def test_shared_arrays_remain_separate_for_cache_groups_on_different_layers(layer_worker, operation):
    worker = layer_worker
    worker.num_kv_cache_groups = 2
    db = ChunkedTokenDatabase([KeyMetadata("model", 0, 0, 0, 0), KeyMetadata("model", 0, 0, 0, 0, 1)], [4, 4], None)
    db.set_group_buffers({0: [100], 1: [200]}, {0: [8], 1: [16]}, {0: [12], 1: [20]}, group_num_layers={0: 1, 1: 1})
    thread = worker.kv_send_thread if operation == "save" else worker.kv_recv_thread
    thread.group_builders = [LayerBatchBuilder(db, 8, 1, group_id=0), LayerBatchBuilder(db, 16, 1, group_id=1)]
    request = ReqMeta(
        "r",
        block_ids_by_group_np=[np.array([2]), np.array([3])],
        block_gvas_by_group_np=[np.array([300]), np.array([400])],
        load_block_gvas_by_group_np=[np.array([500]), np.array([600])],
    )
    tasks = [
        [LayerTransferTask(0, [LayerBlockRange(request, 0, 1)], group_id=0)],
        [LayerTransferTask(1, [LayerBlockRange(request, 0, 1)], group_id=1)],
    ]
    setattr(worker, f"layer_{operation}_tasks", tasks)
    getattr(worker, f"_build_shared_{operation}_data")()
    first, second = tasks[0][0].shared_block_data, tasks[1][0].shared_block_data
    assert first is not second
    np.testing.assert_array_equal(first.block_ids_arr, [2])
    np.testing.assert_array_equal(second.block_ids_arr, [3])
    np.testing.assert_array_equal(first.block_gvas_arr, [300 if operation == "save" else 500])
    np.testing.assert_array_equal(second.block_gvas_arr, [400 if operation == "save" else 600])


def test_empty_layer_load_without_slot_callback_clears_stale_event(layer_worker):
    worker = layer_worker
    worker.layer_load_finished_events[0].set()

    worker.wait_for_layer_load()

    assert not worker.layer_load_finished_events[0].is_set()
    assert worker.kv_recv_thread.request_queue.empty()
    assert worker.next_layer_to_submit == worker.num_layers


def test_save_batch_reuses_one_recorded_device_event(layer_worker):
    worker = layer_worker
    worker.group_uses_align_state = [False]
    sender = MagicMock()
    worker.kv_send_thread = sender
    requests = [ReqMeta("a", can_save=True), ReqMeta("skip", can_save=False), ReqMeta("b", can_save=True)]
    metadata = AscendConnectorMetadata(set())
    for request in requests:
        metadata.add_request(request)

    worker.wait_for_save(metadata)

    torch.npu.Event.assert_called_once_with()
    torch.npu.Event.return_value.record.assert_called_once_with()
    assert requests[0].current_event is requests[2].current_event
    assert [call.args[0] for call in sender.add_request.call_args_list] == [requests[0], requests[2]]
    assert [call.args for call in sender.add_stored_request.call_args_list] == [("a",), ("b",)]
    assert requests[0].skip_null_blocks_by_group == requests[2].skip_null_blocks_by_group == [False]
    sender.request_queue.join.assert_called_once_with()


def test_synchronous_receive_cancellation_preserves_other_completion_records(layer_worker):
    worker = layer_worker
    worker.use_layerwise, worker.load_async = True, False
    # Key-based transfer owns its own completion bookkeeping.
    worker.kv_send_thread = KVCacheStoreKeyLayerSendingThread(
        worker.m_store,
        worker.token_database,
        4,
        0,
        1,
        1,
        16,
        threading.Event(),
        2,
        worker.layer_save_finished_events,
        worker.sync_save_events,
    )
    sender, receiver = worker.kv_send_thread, worker.kv_recv_thread
    for thread in (sender, receiver):
        thread.set_finished_request("cancelled")
        thread.set_finished_request("other")
    metadata = AscendConnectorMetadata({"cancelled"}, loading_req_ids={"other"})

    assert worker.get_finished(set(), metadata) == (set(), set())

    assert sender.finished_requests == set()
    assert receiver.finished_requests == {"other"}


def test_layer_wait_surfaces_failure_arriving_during_wait(layer_worker):
    worker = layer_worker
    worker.layer_load_tasks[0] = [LayerTransferTask(0, [])]
    event = MagicMock()

    def fail_during_wait(**kwargs):
        worker.kv_recv_thread._fatal_error = RuntimeError("late SDK failure")
        return False

    event.wait.side_effect = fail_during_wait
    worker.layer_load_finished_events[0] = event
    with pytest.raises(RuntimeError, match="asynchronous transfer"):
        worker.wait_for_layer_load()
    event.wait.assert_called_once_with(timeout=10)
    event.clear.assert_not_called()


@pytest.mark.parametrize("layer", [0, 1])
def test_empty_save_layers_advance_without_stale_completion_events(layer_worker, layer):
    worker = layer_worker
    worker.current_layer = layer
    worker.save_kv_layer(AscendConnectorMetadata(set()))
    assert worker.current_layer == layer + 1
    worker.sync_save_events[layer].record.assert_called_once_with()
    assert worker.layer_save_finished_events[layer].is_set() == (layer == 0)
    assert worker.kv_send_thread.request_queue.empty()


@pytest.mark.parametrize("scenario", ["finished", "empty", "ready", "late", "failed"])
def test_layer_load_wait_observes_completion_failure_and_slot_release(layer_worker, scenario):
    worker = layer_worker
    worker.external_slot_release_waiter = MagicMock()
    if scenario == "finished":
        worker.current_layer = worker.num_layers
    elif scenario != "empty":
        worker.layer_load_tasks[0] = [LayerTransferTask(0, [])]
        worker.layer_load_finished_events[0] = MagicMock()
        worker.layer_load_finished_events[0].wait.side_effect = [False, True] if scenario == "late" else [True]
    if scenario == "failed":
        worker.kv_recv_thread._fatal_error = RuntimeError("SDK load failed")
        with pytest.raises(RuntimeError, match="asynchronous transfer"):
            worker.wait_for_layer_load()
        assert worker.kv_recv_thread.request_queue.empty()
        return
    worker.wait_for_layer_load()
    if scenario in {"ready", "late"}:
        worker.layer_load_finished_events[0].clear.assert_called_once_with()
        assert worker.layer_load_finished_events[0].wait.call_count == (2 if scenario == "late" else 1)
        assert worker.kv_recv_thread.request_queue.get_nowait().layer_id == 0
    if scenario == "empty":
        worker.external_slot_release_waiter.assert_called_once_with(0)
    else:
        worker.external_slot_release_waiter.assert_not_called()


@pytest.mark.parametrize("scenario", ["finished", "empty", "queued", "late", "failed"])
def test_layer_save_records_event_and_preserves_reuse_barriers(layer_worker, scenario):
    worker = layer_worker
    worker.current_layer = 1
    worker.prefetch_layer_map = {1: 0}
    worker.layer_save_finished_events[0].set()
    if scenario == "finished":
        worker.current_layer = 2
    elif scenario in {"queued", "late", "failed"}:
        request = ReqMeta("r", block_ids=[2])
        worker.layer_save_tasks[1] = [LayerTransferTask(1, [LayerBlockRange(request, 0, 1)])]
        worker.layer_save_finished_events[1] = MagicMock()
        worker.layer_save_finished_events[1].wait.side_effect = [False, True] if scenario == "late" else [True]
    if scenario == "failed":
        worker.kv_send_thread._fatal_error = RuntimeError("SDK store failed")
        with pytest.raises(RuntimeError, match="asynchronous transfer"):
            worker.save_kv_layer(AscendConnectorMetadata(set()))
        worker.sync_save_events[1].record.assert_not_called()
        return
    worker.save_kv_layer(AscendConnectorMetadata(set()))
    assert worker.current_layer == 2
    assert worker.layer_save_finished_events[0].is_set()
    if scenario != "finished":
        worker.sync_save_events[1].record.assert_called_once_with()
    if scenario in {"queued", "late"}:
        assert worker.kv_send_thread.stored_requests == {"r": 1}
        assert worker.kv_send_thread.request_queue.get_nowait() is worker.layer_save_tasks[1]
        worker.layer_save_finished_events[1].clear.assert_called_once_with()


@pytest.mark.parametrize(
    ("states", "align", "expected"),
    [
        ([1, 1, 1, 1], False, 16),
        ([1, 1, 1, 0], False, 8),
        ([1, 0, 1, 1], False, 0),
        ([0, 0, 0, 0], False, 0),
        ([0, 1, 0, 1], True, 16),
        ([0, 1, 1, 0], True, 8),
        ([1, 0, 1, 0], True, 0),
    ],
)
def test_lookup_respects_alignment_and_sparse_state_hits(lookup_worker, states, align, expected):
    worker = lookup_worker
    worker.m_store.exists.return_value = states
    worker.group_uses_align_state = [align]
    assert worker.lookup(16, [b"a", b"b", b"c", b"d"]) == expected
    keys = worker.m_store.exists.call_args.args[0]
    assert len(keys) == 4
    assert keys[0].endswith("@61")


@pytest.mark.parametrize("method", ["lookup", "lookup_scheduler"])
def test_lookup_empty_and_backend_exception_return_cache_miss(lookup_worker, method):
    worker = lookup_worker
    assert getattr(worker, method)(0, []) == 0
    worker.m_store.exists.assert_not_called()
    worker.m_store.exists.side_effect = RuntimeError("SDK unavailable")
    assert getattr(worker, method)(16, [b"a"] * 4) == 0
    worker.m_store.exists.assert_called_once()


def test_layerwise_lookup_requires_all_layers_and_all_rank_shards(lookup_worker):
    worker = lookup_worker
    worker.m_store.exists.return_value = [1, 1, 1, 1, 1, 0, 1, 1]
    assert worker.lookup(16, [b"a", b"b", b"c", b"d"], use_layerwise=True) == 8
    worker.pp_size = 2
    worker.m_store.exists.return_value = [1] * 8 + [1, 1, 1, 1, 0, 1, 1, 1]
    assert worker.lookup_scheduler(16, [b"a", b"b", b"c", b"d"], use_layerwise=True) == 8
    assert len(worker.m_store.exists.call_args.args[0]) == 16


def test_coordinator_lookup_merges_hbm_hits_with_external_hits(lookup_worker):
    import torch
    from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheGroupSpec

    from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.coordinator import AscendStoreCoordinator

    worker = lookup_worker
    spec = FullAttentionSpec(block_size=4, num_kv_heads=1, head_size=2, dtype=torch.int8)
    worker.cache_coordinator = AscendStoreCoordinator([KVCacheGroupSpec(["a"], spec)], 8, 4, [4], ["c1"])
    worker.m_store.exists.return_value = [1, 0]
    hashes = [b"a", b"b", b"c", b"d"]
    assert worker.lookup_scheduler(16, hashes, hbm_hit_tokens=8) == 12
    assert len(worker.m_store.exists.call_args.args[0]) == 2
    worker.m_store.reset_mock()
    assert worker.lookup_scheduler(16, hashes, hbm_hit_tokens=16) == 16
    worker.m_store.exists.assert_not_called()
    worker.m_store.exists.return_value = [1, 1, 0, 0]
    assert worker.lookup(16, hashes) == 8
    assert worker._lookup_with_coordinator(16, hashes, [1], False, False) is None


@pytest.mark.parametrize(
    ("mla", "sparse", "align", "mismatch", "expected"),
    [
        (True, False, False, False, 1),
        (False, True, False, False, 1),
        (False, False, True, False, 4),
        (False, False, False, True, 8),
    ],
)
def test_group_rank_count_respects_cache_family(lookup_worker, mla, sparse, align, mismatch, expected):
    worker = lookup_worker
    worker.use_mla, worker.use_sparse, worker.tp_mismatch = mla, sparse, mismatch
    worker.group_uses_align_state = [align]
    worker.tp_size, worker.effective_tp_size = 4, 8
    assert worker.get_group_tp_size(0) == expected
    if align:
        assert worker._get_group_num_kv_heads(0) == 1


def test_lookup_helpers_handle_terminal_fields_and_nonaligned_positions(lookup_worker):
    worker = lookup_worker
    assert worker._replace_key_field("model@pp_rank:0", "pp_rank", 3) == "model@pp_rank:3"
    assert worker._max_intersection_hit_position([[4], [8]]) == 0
    assert worker.find_all_continuous_hit_positions([[1, 1, 1]], [4, 8, 12], 3, 8, 8) == [8]
    assert worker.find_all_discontinuous_hit_positions([[1, 1, 1]], [4, 8, 12], 3, 16, 8) == [8]
    worker.m_store = SimpleNamespace()
    worker.ensure_store_initialized()
    worker.m_store.ensure_initialized = MagicMock()
    worker.ensure_store_initialized()
    worker.m_store.ensure_initialized.assert_called_once_with()


@pytest.fixture
def mismatch_worker(lookup_worker):
    worker = lookup_worker
    worker.tp_mismatch = True
    worker.block_size = 4
    worker.num_sub_keys = 2
    worker.sub_size_bytes = 2
    worker.group_kv_caches_base_addr = {0: [100]}
    worker.group_block_len = {0: [16]}
    worker.group_block_stride = {0: [20]}
    worker._invalid_block_ids = set()
    worker._invalid_block_ids_lock = threading.Lock()
    worker.kv_send_thread = MagicMock()
    worker.enable_kv_events = False
    return worker


@pytest.mark.parametrize(("result", "invalid"), [([0, 0], set()), ([0, 1], {2}), (None, {2})])
def test_tp_mismatch_load_tracks_failed_blocks_and_exact_head_addresses(mismatch_worker, result, invalid):
    worker = mismatch_worker
    worker.m_store.get.return_value = result
    worker._load_kv_tp_mismatch([b"a"], [2], 4, 0)
    keys, addresses, sizes = worker.m_store.get.call_args.args
    assert len(keys) == 2
    assert "@head_or_tp_rank:0@" in keys[0]
    assert "@head_or_tp_rank:1@" in keys[1]
    assert addresses == [[140, 144, 148, 152], [142, 146, 150, 154]]
    assert sizes == [[2] * 4, [2] * 4]
    assert worker._invalid_block_ids == invalid
    worker.m_store.get.reset_mock()
    worker._load_kv_tp_mismatch([], [], 0, 0)
    worker.m_store.get.assert_not_called()


@pytest.mark.parametrize("failure", [False, True])
def test_tp_mismatch_thread_runs_real_store_and_completes_event(mismatch_worker, failure):
    worker = mismatch_worker
    worker.token_database.set_group_buffers({0: [100]}, {0: [16]}, {0: [20]}, group_num_layers={0: 1})
    store = worker.m_store
    store.exists.return_value = [0, 0]
    if failure:
        store.put.side_effect = RuntimeError("SDK store failed")
    thread = KVCacheStoreSendingThread(store, worker.token_database, 4, 0, worker=worker)
    worker.kv_send_thread = thread
    request = ReqMeta("r", token_len_chunk=4, block_ids=[2], block_hashes=[b"a"], event_id=7)
    thread.add_stored_request("r")
    thread.add_request(request)
    thread._handle_request(thread.request_queue.get_nowait())
    assert thread.finished_requests == {"r"}
    assert thread.stored_requests == {}
    assert thread.get_completed_events() == {7: 1}
    assert thread.request_queue.unfinished_tasks == 0
    assert store.put.call_args.args[1] == [[140, 144, 148, 152], [142, 146, 150, 154]]


@pytest.mark.parametrize("scenario", ["missing_thread", "empty", "cached", "partial", "failure", "events"])
def test_tp_mismatch_store_releases_request_for_every_outcome(mismatch_worker, scenario):
    worker = mismatch_worker
    send = worker.kv_send_thread
    metadata = ReqMeta(
        "r", token_len_chunk=4, block_ids=[2], block_hashes=[b"a"], original_block_size=[4], token_ids=[1, 2, 3, 4]
    )
    metadata.current_event = MagicMock()
    send.lookup.return_value = [1, 0]
    if scenario == "missing_thread":
        worker.kv_send_thread = None
    elif scenario == "empty":
        metadata.block_hashes = []
    elif scenario == "cached":
        send.lookup.return_value = [1, 1]
    elif scenario == "failure":
        worker.m_store.put.side_effect = RuntimeError("put failed")
    elif scenario == "events":
        worker.enable_kv_events = True
    if scenario == "failure":
        with pytest.raises(RuntimeError, match="put failed"):
            worker._store_kv_tp_mismatch(metadata)
    else:
        worker._store_kv_tp_mismatch(metadata)
    if scenario == "missing_thread":
        send.dec_stored_request.assert_not_called()
    else:
        send.dec_stored_request.assert_called_once_with("r")
    if scenario in {"partial", "failure", "events"}:
        metadata.current_event.synchronize.assert_called_once_with()
        keys, addresses, sizes = worker.m_store.put.call_args.args
        assert len(keys) == 1 and "@head_or_tp_rank:1@" in keys[0]
        assert addresses == [[142, 146, 150, 154]]
        assert sizes == [[2] * 4]
    else:
        worker.m_store.put.assert_not_called()
    if scenario == "events":
        event = send.update_kv_event.call_args.args[0][0]
        assert event.token_ids == [1, 2, 3, 4]
        assert event.block_size == 4
        assert event.parent_block_hash is None


@pytest.mark.parametrize(
    ("layerwise", "protocol", "role", "consumer_put", "async_load", "send_name", "recv_name"),
    [
        (False, False, "kv_producer", False, False, "KVCacheStoreSendingThread", None),
        (False, False, "kv_consumer", False, True, None, "KVCacheStoreRecvingThread"),
        (False, False, "kv_consumer", True, True, "KVCacheStoreSendingThread", "KVCacheStoreRecvingThread"),
        (
            True,
            False,
            "kv_producer",
            False,
            False,
            "KVCacheStoreKeyLayerSendingThread",
            "KVCacheStoreKeyLayerRecvingThread",
        ),
        (True, True, "kv_both", False, False, "KVCacheStoreLayerSendingThread", "KVCacheStoreLayerRecvingThread"),
        (True, True, "kv_consumer", False, False, None, "KVCacheStoreLayerRecvingThread"),
    ],
)
def test_transfer_thread_construction_routes_roles_without_starting_os_threads(
    lookup_worker, monkeypatch, layerwise, protocol, role, consumer_put, async_load, send_name, recv_name
):
    worker = lookup_worker
    worker._init_state_vars()
    worker.use_layerwise, worker.use_layerwise_transfer = layerwise, protocol
    worker.kv_role, worker.consumer_is_to_put, worker.load_async = role, consumer_put, async_load
    worker.block_size = 4
    worker.grouped_block_size = [4]
    worker.put_step = 1
    worker.enable_kv_events = False
    worker._invalid_block_ids = set()
    worker._invalid_block_ids_lock = threading.Lock()
    worker.page_size_bytes = 16
    worker.layerwise_max_transfer_blocks = worker.layerwise_max_transfer_bytes = worker.h2d_stagger_us = 0
    worker.group_num_layers = {0: 2}
    worker.group_block_len = {0: [8, 8]}
    worker.token_database.set_group_buffers(
        {0: [100, 200]}, worker.group_block_len, group_num_layers=worker.group_num_layers
    )
    launched = []

    def start_thread(thread):
        launched.append(thread)
        thread.ready_event.set()

    monkeypatch.setattr(threading.Thread, "start", start_thread)
    worker._start_kv_transfer_threads()
    assert (type(worker.kv_send_thread).__name__ if worker.kv_send_thread else None) == send_name
    assert (type(worker.kv_recv_thread).__name__ if worker.kv_recv_thread else None) == recv_name
    assert all(thread.ready_event.is_set() and not thread.is_alive() for thread in launched)
    assert all(
        thread.m_store is worker.m_store and thread.token_database is worker.token_database for thread in launched
    )
    assert worker._transfer_threads_started is True
    original = list(launched)
    worker._start_kv_transfer_threads()
    assert launched == original
    if protocol:
        waiter = MagicMock()
        assert worker.set_external_slot_release_waiter(waiter)
        assert worker.kv_recv_thread.external_slot_release_waiter is waiter


@pytest.mark.parametrize(
    ("partition", "pp_size", "expected", "error"),
    [
        ("a,2", 2, None, "Invalid partition"),
        ("3", 2, None, "does not match"),
        ("1,1", 2, None, "does not match"),
        (None, 2, [3, 2], None),
        (None, 1, [5], None),
        ("2,3", 2, [2, 3], None),
    ],
)
def test_consumer_pp_partition_validation_and_remainder_distribution(partition, pp_size, expected, error):
    context = unittest.TestCase()
    try:
        extra = {"consumer_is_to_put": True, "prefill_pp_size": pp_size}
        if partition is not None:
            extra["prefill_pp_layer_partition"] = partition
        if error:
            with pytest.raises(ValueError, match=error):
                make_worker(context, kv_role="kv_consumer", num_hidden_layers=5, extra_config=extra)
        else:
            worker = make_worker(context, kv_role="kv_consumer", num_hidden_layers=5, extra_config=extra)
            assert worker.token_database.partitions == expected
    finally:
        context.doCleanups()


@pytest.mark.parametrize("layerwise", [False, True])
def test_hybrid_worker_constructor_maps_physical_layers_without_starting_threads(layerwise):
    context = unittest.TestCase()
    spec = FullAttentionSpec(block_size=16, num_kv_heads=1, head_size=2, dtype=torch.int8)
    mamba = MambaSpec(block_size=16, shapes=((2,),), dtypes=(torch.int8,), mamba_cache_mode="align")
    cache = SimpleNamespace(
        kv_cache_groups=[
            KVCacheGroupSpec(["model.layers.0.self_attn"], spec),
            KVCacheGroupSpec(["model.layers.1.mamba"], mamba),
        ]
    )
    try:
        worker = make_worker(
            context,
            kv_cache_config=cache,
            prefix_match_unit=16,
            num_hidden_layers=2,
            use_layerwise=layerwise,
            extra_config={"backend": "mooncake"},
        )
        assert worker.num_layers == 2
        assert worker.hash_block_size == 16
        assert not worker._transfer_threads_started
        assert worker.kv_send_thread is worker.kv_recv_thread is None
        assert worker.cache_coordinator is not None
    finally:
        context.doCleanups()


def test_mtp_group_keeps_base_layer_index_when_hf_config_omits_num_hidden_layers():
    context = unittest.TestCase()
    spec = FullAttentionSpec(block_size=16, num_kv_heads=1, head_size=2, dtype=torch.int8)
    cache = SimpleNamespace(
        kv_cache_groups=[
            KVCacheGroupSpec(["model.layers.0", "model.layers.1"], spec),
            KVCacheGroupSpec(["mtp.layers.0"], spec),
        ]
    )
    try:
        worker = make_worker(context, kv_cache_config=cache, num_layers=2, use_layerwise=True)
        assert worker.num_layers == 3
        assert worker._extract_physical_layer_index("mtp.layers.0") == 2
        assert worker.physical_layer_to_group_layers == {0: [(0, 0)], 1: [(0, 1)], 2: [(1, 0)]}
        worker._init_layerwise_config()
        assert worker.num_layers == 3
        assert worker.physical_layer_to_group_layers[2] == [(1, 0)]
    finally:
        context.doCleanups()


def test_tp_mismatch_rejects_real_hybrid_cache_config():
    context = unittest.TestCase()
    spec = FullAttentionSpec(block_size=16, num_kv_heads=2, head_size=2, dtype=torch.int8)
    mamba = MambaSpec(block_size=16, shapes=((2,),), dtypes=(torch.int8,), mamba_cache_mode="align")
    cache = SimpleNamespace(
        kv_cache_groups=[
            KVCacheGroupSpec(["model.layers.0.self_attn"], spec),
            KVCacheGroupSpec(["model.layers.1.mamba"], mamba),
        ]
    )
    try:
        with pytest.raises(NotImplementedError, match="hybrid KV cache layouts"):
            make_worker(
                context,
                kv_cache_config=cache,
                kv_role="kv_consumer",
                tp_size=2,
                num_kv_heads=8,
                extra_config={"prefill_tp_size": 4},
            )
    finally:
        context.doCleanups()


def test_worker_accepts_empty_cache_group_description_without_creating_transfer_threads():
    context = unittest.TestCase()
    try:
        worker = make_worker(context, kv_cache_config=SimpleNamespace(kv_cache_groups=[]), num_hidden_layers=2)
        assert worker.physical_layer_to_group_layers == {}
        assert worker.num_layers == 2
        assert worker.kv_send_thread is worker.kv_recv_thread is None
    finally:
        context.doCleanups()


def test_register_real_cpu_hybrid_views_registers_one_aligned_storage_region(monkeypatch):
    context = unittest.TestCase()
    full_spec = FullAttentionSpec(block_size=16, num_kv_heads=1, head_size=2, dtype=torch.int8)
    mamba_spec = MambaSpec(block_size=16, shapes=((2,),), dtypes=(torch.int8,), mamba_cache_mode="align")
    full_name, mamba_name = "model.layers.0.self_attn", "model.layers.1.mamba"
    cache = SimpleNamespace(
        num_blocks=4,
        kv_cache_groups=[
            KVCacheGroupSpec([full_name], full_spec),
            KVCacheGroupSpec([mamba_name], mamba_spec),
        ],
    )
    alignment = 2 * 1024 * 1024
    raw = torch.empty(alignment + 512, dtype=torch.int8)
    offset = (-raw.data_ptr()) % alignment
    full = raw[offset : offset + 256].view(4, 64)
    mamba = raw[offset + 256 : offset + 264].view(4, 2)
    launched = []

    def ready(thread):
        launched.append(thread)
        thread.ready_event.set()

    monkeypatch.setattr(threading.Thread, "start", ready)
    try:
        worker = make_worker(context, kv_cache_config=cache, prefix_match_unit=16, num_hidden_layers=2)
        worker.register_kv_caches({full_name: full, mamba_name: [mamba]})
        worker.m_store.register_buffer.assert_called_once_with([full.data_ptr()], [264])
        assert worker.group_block_len == {0: [64], 1: [2]}
        assert worker.group_block_stride == {0: [64], 1: [2]}
        assert worker.group_num_layers == {0: 1, 1: 1}
        assert worker.physical_layer_to_group_layers == {0: [(0, 0)], 1: [(1, 0)]}
        assert len(launched) == 1 and not launched[0].is_alive()
    finally:
        context.doCleanups()


def test_legacy_tensor_storage_api_preserves_real_cpu_pointer():
    tensor = torch.empty(4, dtype=torch.int8)

    class LegacyTensor:
        def storage(self):
            return tensor.untyped_storage()

    assert worker_module.KVPoolWorker._get_storage_key(LegacyTensor()) == tensor.data_ptr()


def test_preempted_requests_clear_real_transfer_bookkeeping(layer_worker):
    worker = layer_worker
    worker.use_layerwise, worker.load_async = False, True
    sender, receiver = worker.kv_send_thread, worker.kv_recv_thread
    for request in ("cancelled", "done", "other"):
        sender.add_stored_request(request)
        sender.set_finished_request(request)
        receiver.set_finished_request(request)
    metadata = AscendConnectorMetadata({"cancelled"}, loading_req_ids={"done"}, delayed_free_req_ids={"done"})
    assert worker.get_finished({"done"}, metadata) == ({"done"}, {"done"})
    assert "cancelled" not in sender.stored_requests
    assert sender.finished_requests == receiver.finished_requests == {"other"}


@pytest.mark.parametrize("hit", [False, True])
def test_scheduler_lookup_uses_discontinuous_mamba_hits(lookup_worker, hit):
    worker = lookup_worker
    worker.group_uses_align_state = [True]
    worker.m_store.exists.return_value = [0, 1, 0, 1] if hit else [0, 0, 0, 0]
    assert worker.lookup_scheduler(16, [b"a", b"b", b"c", b"d"]) == (16 if hit else 0)


def test_group_metadata_ignores_out_of_range_physical_layer(gva_worker):
    worker = gva_worker
    worker.num_layers = 2
    worker.num_blocks = 2
    worker.num_kv_cache_groups = 2
    worker.kv_caches = {"model.layers.0": torch.empty((2, 4), dtype=torch.int8)}
    worker.group_kv_caches_base_addr = {}
    worker.group_block_stride = {}
    worker.group_layer_cache_entry_offsets = {}
    worker._infer_cache_group_metadata(0, ["model.layers.0", "model.layers.9"])

    assert worker.group_kv_caches_base_addr[0] == [worker.kv_caches["model.layers.0"].data_ptr()]
    assert worker.group_block_len[0] == [4]
    assert worker.group_block_stride[0] == [4]
    assert worker.group_layer_cache_entry_offsets[0] == [0, 1]
    assert worker.group_num_layers[0] == 1


def test_empty_layerwise_step_does_not_reuse_prior_task_lists(layer_worker):
    worker = layer_worker
    worker.use_layerwise = True
    worker.use_layerwise_transfer = False
    worker.kv_send_thread = None
    worker.kv_recv_thread = None
    worker.put_step = 1
    worker.grouped_block_size = [4]
    worker.layerwise_offload = False
    worker.independent_layers = []
    worker.physical_layer_to_group_layers = {}
    old_save, old_load = worker.layer_save_tasks, worker.layer_load_tasks
    metadata = AscendConnectorMetadata(set())
    metadata.add_request(ReqMeta("r", can_save=False))
    worker.start_load_kv(metadata)
    assert worker.layer_save_tasks == worker.layer_load_tasks == [[], []]
    assert worker.layer_save_tasks is not old_save and worker.layer_load_tasks is not old_load
    worker.tp_rank, worker.put_step = 1, 2
    worker._process_save_for_layer_batch([ReqMeta("r", can_save=True)], 0)
    assert worker.layer_save_tasks == [[], []]


def test_hybrid_worker_detects_direct_and_uniform_mamba_specs(lookup_worker):
    worker = lookup_worker
    full = FullAttentionSpec(block_size=4, num_kv_heads=1, head_size=2, dtype=torch.int8)
    mamba = MambaSpec(block_size=4, shapes=((2,),), dtypes=(torch.int8,), mamba_cache_mode="align")
    groups = [
        KVCacheGroupSpec(["a"], full),
        KVCacheGroupSpec(["b"], mamba),
        KVCacheGroupSpec(["c"], UniformTypeKVCacheSpecs(block_size=4, kv_cache_specs={"c": mamba})),
    ]
    worker.kv_cache_config = SimpleNamespace(kv_cache_groups=groups)
    assert worker._infer_group_uses_align_state() == [False, True, True]
    assert worker._uses_mamba_kv_cache(True, worker.kv_cache_config)
    assert not worker._uses_mamba_kv_cache(True, SimpleNamespace(kv_cache_groups=[groups[0]]))
    assert not worker._uses_mamba_kv_cache(
        True,
        SimpleNamespace(
            kv_cache_groups=[KVCacheGroupSpec(["a"], UniformTypeKVCacheSpecs(block_size=4, kv_cache_specs={"a": full}))]
        ),
    )


@pytest.mark.parametrize("retention", [None, 8])
@pytest.mark.parametrize("eagle", [False, True])
def test_worker_builds_real_hybrid_cache_coordinator(lookup_worker, monkeypatch, retention, eagle):
    worker = lookup_worker
    spec = FullAttentionSpec(block_size=4, num_kv_heads=1, head_size=2, dtype=torch.int8)
    worker.kv_cache_config = SimpleNamespace(kv_cache_groups=[KVCacheGroupSpec(["a"], spec)])
    worker.use_hybrid = True
    worker.grouped_block_size = [4]
    worker.kv_cache_group_families = ["c1"]
    monkeypatch.setattr(worker_module.envs, "VLLM_PREFIX_CACHE_RETENTION_INTERVAL", retention, raising=False)
    config = SimpleNamespace(speculative_config=SimpleNamespace(use_eagle=lambda: eagle))
    coordinator = worker._build_cache_coordinator(config)
    assert coordinator.use_eagle is eagle
    assert coordinator.retention_interval == retention
    assert coordinator.group_effective_specs == [spec]
    worker.kv_cache_config = SimpleNamespace(
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["a"],
                UniformTypeKVCacheSpecs(
                    block_size=4,
                    kv_cache_specs={
                        "a": MambaSpec(block_size=4, shapes=((2,),), dtypes=(torch.int8,), mamba_cache_mode="align")
                    },
                ),
            )
        ]
    )
    assert worker._uses_mamba_kv_cache(True, worker.kv_cache_config)


def test_hybrid_pointer_alignment_preserves_region_end_and_rejects_invalid_base(lookup_worker):
    worker = lookup_worker
    worker.use_hybrid = True
    alignment = 2 * 1024 * 1024
    regions = {alignment: (alignment + 128, alignment + 256)}
    worker._align_kv_ptrs(regions)
    assert regions == {alignment: (alignment, alignment + 256)}
    with pytest.raises(AssertionError, match="align to 2MB"):
        worker._align_kv_ptrs({alignment + 1: (alignment + 128, alignment + 256)})


@pytest.mark.parametrize("has_hashes", [False, True])
@pytest.mark.parametrize("get_completed", [False, True])
def test_legacy_layer_generators_keep_one_yield_per_layer(lookup_worker, has_hashes, get_completed):
    worker = lookup_worker
    worker.block_size = 4
    worker.grouped_block_size = [4]
    worker.get_event = MagicMock(wait=MagicMock(return_value=get_completed))
    worker.kv_recv_thread = MagicMock()
    worker.kv_send_thread = MagicMock()
    request = ReqMeta(
        "r",
        token_len_chunk=8,
        block_hashes=[b"a", b"b"] if has_hashes else [],
        block_ids=[1, 2],
        load_spec=LoadSpec(4, 8, True),
        token_ids=list(range(8)),
        original_block_size=[4],
    )
    received = list(worker.retrieve_layer(request))
    assert received[:2] == [None, None]
    assert received[2].tolist() == ([False] * 4 + [True] * 4 if has_hashes else [False] * 8)
    assert worker.kv_recv_thread.add_request.call_count == (2 if has_hashes else 0)
    assert list(worker.store_layer(request, None)) == [None, None]
    assert worker.kv_send_thread.add_request.call_count == (2 if has_hashes else 0)
    if has_hashes:
        request_meta = worker.kv_recv_thread.add_request.call_args.args[0]
        assert request_meta.starts == [4]
        assert request_meta.ends == [8]
        assert request_meta.layer_id == 1


def test_layer_load_submission_skips_empty_layers_and_gates_prefetch(lookup_worker):
    worker = lookup_worker
    worker.num_layers = 4
    worker.num_prefetch_layers = 2
    worker.current_layer = worker.next_layer_to_submit = 0
    worker.prefetch_layer_map = {2: 0}
    worker.layer_load_tasks = [[LayerTransferTask(0, [])], [], [], [LayerTransferTask(3, [])]]
    worker.kv_recv_thread = MagicMock()
    worker._submit_ready_layer_loads()
    first, second = [call.args[0] for call in worker.kv_recv_thread.add_request.call_args_list]
    assert (first.layer_id, first.wait_for_save_layer, first.attention_start_gate) == (0, None, None)
    assert (second.layer_id, second.wait_for_save_layer) == (2, 0)
    assert second.attention_start_gate is None
    assert worker.next_layer_to_submit == 3
    worker.current_layer = 1
    worker._submit_ready_layer_loads()
    third = worker.kv_recv_thread.add_request.call_args.args[0]
    assert third.layer_id == 3
    assert third.attention_start_gate is not None
    worker._submit_ready_layer_loads()
    assert worker.kv_recv_thread.add_request.call_count == 3


@pytest.fixture
def gva_worker():
    context = TestKVPoolWorkerProcessLayerData()
    try:
        yield context._make_gva_worker()
    finally:
        context.doCleanups()


@pytest.mark.parametrize("states", [[], [2]])
def test_gva_refresh_rejects_incomplete_or_failed_existence_result(gva_worker, states):
    worker = gva_worker
    worker._allocated_gvas = {"a": 100}
    worker.m_store.batch_is_exist.return_value = states
    with pytest.raises(RuntimeError, match="MemCache exists check"):
        worker._refresh_allocated_gvas(["a", "a"])
    assert worker._allocated_gvas == {"a": 100}
    worker.m_store.batch_is_exist.assert_called_once_with(["a"])


@pytest.mark.parametrize("state", [0, 1])
def test_gva_refresh_evicts_only_missing_keys(gva_worker, state):
    worker = gva_worker
    worker._allocated_gvas = {"a": 100, "unrelated": 200}
    worker.m_store.batch_is_exist.return_value = [state]
    worker._refresh_allocated_gvas(["a"])
    assert worker._allocated_gvas == ({"a": 100, "unrelated": 200} if state else {"unrelated": 200})


@pytest.mark.parametrize("reason", ["no_protocol", "consumer", "replicated_rank", "cannot_save", "missing_ids"])
def test_gva_allocation_rejects_or_skips_inapplicable_requests(gva_worker, reason):
    worker = gva_worker
    request = TestKVPoolWorkerProcessLayerData._make_gva_request(can_save=True)
    if reason == "no_protocol":
        worker.use_layerwise_transfer = False
    elif reason == "consumer":
        worker.kv_role = "kv_consumer"
    elif reason == "replicated_rank":
        worker.tp_rank, worker.put_step = 1, 2
    elif reason == "cannot_save":
        request.can_save = False
    else:
        request.block_ids_np = request.block_ids_by_group_np = None
    if reason == "missing_ids":
        with pytest.raises(RuntimeError, match="Block IDs are not initialized"):
            worker._alloc_gvas_for_save([request])
    else:
        worker._alloc_gvas_for_save([request])
    worker.m_store.batch_alloc.assert_not_called()


@pytest.mark.parametrize("allocation", [[], [0], [-1]])
def test_failed_partial_allocation_is_not_retained(gva_worker, allocation):
    worker = gva_worker
    request = TestKVPoolWorkerProcessLayerData._make_gva_request(can_save=True)
    request.target_token_len = request.save_end_token = 15
    request.block_hashes = []
    worker.m_store.batch_alloc.return_value = allocation
    worker._alloc_gvas_for_save([request])
    assert worker._allocated_gvas == {}
    assert request.save_keys == []
    assert request.partial_save_gva_per_group == [allocation[0] if allocation else 0]


@pytest.mark.parametrize("lease_results", [[], [1, 2]])
def test_load_rejects_lease_result_count_mismatch(gva_worker, lease_results):
    request = TestKVPoolWorkerProcessLayerData._make_gva_request(load_spec=LoadSpec(0, 16, True))
    info = MagicMock(size=MagicMock(return_value=64), gva_list=MagicMock(return_value=[100]))
    gva_worker.m_store.batch_get_key_info.return_value = [info]
    gva_worker.m_store.batch_add_lease.return_value = lease_results
    with pytest.raises(RuntimeError, match="lease returned unexpected number"):
        gva_worker._prepare_load_gvas([request])
    assert request.load_keys is None


def test_partial_lease_retry_exhaustion_marks_block_for_recompute(gva_worker, monkeypatch):
    request = TestKVPoolWorkerProcessLayerData._make_gva_request(load_spec=LoadSpec(0, 15, True))
    request.block_hashes = []
    info = MagicMock(size=MagicMock(return_value=64), gva_list=MagicMock(return_value=[100]))
    gva_worker.m_store.batch_get_key_info.return_value = [info]
    gva_worker.m_store.batch_add_lease.return_value = [worker_module.MEMCACHE_UNMATCHED_STATE]
    sleep = MagicMock()
    monkeypatch.setattr(worker_module.time, "sleep", sleep)

    gva_worker._prepare_load_gvas([request])

    assert sleep.call_count == worker_module.PARTIAL_LEASE_RETRY_COUNT
    assert all(call.args == (worker_module.PARTIAL_LEASE_RETRY_INTERVAL_S,) for call in sleep.call_args_list)
    assert gva_worker.m_store.batch_add_lease.call_count == worker_module.PARTIAL_LEASE_RETRY_COUNT + 1
    assert request.load_keys == []
    np.testing.assert_array_equal(request.load_block_gvas_np, [0])
    assert gva_worker.get_block_ids_with_load_errors() == {7}
    assert gva_worker.get_block_ids_with_load_errors() == set()


def test_partial_lease_retry_rejects_incomplete_sdk_reply(gva_worker, monkeypatch):
    request = TestKVPoolWorkerProcessLayerData._make_gva_request(load_spec=LoadSpec(0, 15, True))
    request.block_hashes = []
    info = MagicMock(size=MagicMock(return_value=64), gva_list=MagicMock(return_value=[100]))
    gva_worker.m_store.batch_get_key_info.return_value = [info]
    gva_worker.m_store.batch_add_lease.side_effect = [[worker_module.MEMCACHE_UNMATCHED_STATE], []]
    sleep = MagicMock()
    monkeypatch.setattr(worker_module.time, "sleep", sleep)
    with pytest.raises(RuntimeError, match="partial lease retry"):
        gva_worker._prepare_load_gvas([request])
    sleep.assert_called_once_with(worker_module.PARTIAL_LEASE_RETRY_INTERVAL_S)


@pytest.mark.parametrize("scenario", ["no_protocol", "no_load", "missing_ids", "empty_range", "bad_gva"])
def test_load_preparation_handles_empty_and_invalid_metadata(gva_worker, scenario):
    request = TestKVPoolWorkerProcessLayerData._make_gva_request(load_spec=LoadSpec(0, 16, True))
    if scenario == "no_protocol":
        gva_worker.use_layerwise_transfer = False
    elif scenario == "no_load":
        request.load_spec.can_load = False
    elif scenario == "missing_ids":
        request.block_ids_np = request.block_ids_by_group_np = None
    elif scenario == "empty_range":
        request.load_spec.kvpool_cached_tokens = 0
    else:
        gva_worker.m_store.batch_get_key_info.return_value = [MagicMock(size=MagicMock(return_value=0))]
    gva_worker._prepare_load_gvas([request])
    gva_worker.m_store.batch_add_lease.assert_not_called()
    if scenario == "bad_gva":
        assert gva_worker.get_block_ids_with_load_errors() == {7}
        assert request.load_keys == []
        np.testing.assert_array_equal(request.load_block_gvas_np, [0])
    else:
        gva_worker.m_store.batch_get_key_info.assert_not_called()


def test_gva_allocation_reuses_readable_prefix_and_noncontiguous_cached_blocks(gva_worker):
    worker = gva_worker
    request = ReqMeta(
        "r",
        token_len_chunk=48,
        save_start_token=0,
        save_end_token=48,
        target_token_len=48,
        block_hashes=[b"a", b"b", b"c"],
        block_ids=[7, 8, 9],
        block_ids_np=np.array([7, 8, 9]),
        can_save=True,
    )
    keys = [worker._make_layerwise_full_key(0, value) for value in ("61", "62", "63")]
    worker._allocated_gvas = {keys[0]: 100, keys[2]: 300}
    worker.m_store.batch_is_exist.return_value = [1, 1]
    worker.m_store.batch_alloc.return_value = [200]
    worker._alloc_gvas_for_save([request])
    worker.m_store.batch_alloc.assert_called_once_with([keys[1]], [64])
    assert request.save_keys == [keys[1]]
    np.testing.assert_array_equal(request.block_gvas_np, [0, 200, 300])
    assert worker._allocated_gvas == dict(zip(keys, [100, 200, 300]))


def test_failed_full_block_allocation_stays_unpublished(gva_worker):
    request = TestKVPoolWorkerProcessLayerData._make_gva_request(can_save=True)
    gva_worker.m_store.batch_alloc.return_value = [0]
    gva_worker._alloc_gvas_for_save([request])
    assert gva_worker._allocated_gvas == {}
    assert request.save_keys == []
    np.testing.assert_array_equal(request.block_gvas_np, [0])


def test_existing_partial_gva_is_reused_once_then_dropped(gva_worker):
    request = TestKVPoolWorkerProcessLayerData._make_gva_request(can_save=True)
    request.target_token_len = request.save_end_token = 15
    request.block_hashes = []
    key = gva_worker._make_layerwise_partial_key(request, 0, 0, 15)
    gva_worker._allocated_gvas = {key: 500}
    gva_worker._alloc_gvas_for_save([request])
    gva_worker.m_store.batch_alloc.assert_not_called()
    assert request.partial_save_gva_per_group == [500]
    assert gva_worker._allocated_gvas == {}


@pytest.mark.parametrize("invalid", [False, True])
def test_short_block_id_metadata_does_not_write_past_request_buffers(gva_worker, invalid):
    request = TestKVPoolWorkerProcessLayerData._make_gva_request(can_save=True)
    request.target_token_len = request.save_end_token = 32
    request.block_hashes = [b"a", b"b"]
    gva_worker.m_store.batch_alloc.return_value = [100, 200]
    gva_worker._alloc_gvas_for_save([request])
    np.testing.assert_array_equal(request.block_gvas_np, [100])
    request.load_spec = LoadSpec(0, 32, True)
    info = MagicMock(size=MagicMock(return_value=64), gva_list=MagicMock(return_value=[100]))
    gva_worker.m_store.batch_get_key_info.return_value = [info, info]
    gva_worker.m_store.batch_add_lease.return_value = [0, 1 if invalid else 0]
    gva_worker._prepare_load_gvas([request])
    assert gva_worker.get_block_ids_with_load_errors() == set()
    np.testing.assert_array_equal(request.load_block_gvas_np, [100])
    assert len(request.load_keys) == (1 if invalid else 2)


def test_missing_gva_beyond_short_block_ids_does_not_invent_recompute_ids(gva_worker):
    request = TestKVPoolWorkerProcessLayerData._make_gva_request(load_spec=LoadSpec(0, 32, True))
    request.block_hashes = [b"a", b"b"]
    valid = MagicMock(size=MagicMock(return_value=64), gva_list=MagicMock(return_value=[100]))
    missing = MagicMock(size=MagicMock(return_value=0))
    gva_worker.m_store.batch_get_key_info.return_value = [valid, missing]
    gva_worker.m_store.batch_add_lease.return_value = [0]

    gva_worker._prepare_load_gvas([request])

    assert gva_worker.get_block_ids_with_load_errors() == set()
    assert request.load_keys == [gva_worker._make_layerwise_full_key(0, "61")]
    np.testing.assert_array_equal(request.load_block_gvas_np, [100])
    missing.gva_list.assert_not_called()


def test_hybrid_missing_first_group_gva_fails_without_releasing_unowned_leases(gva_worker):
    worker = gva_worker
    worker.num_kv_cache_groups = 2
    worker.grouped_block_size = [16, 16]
    worker.kv_cache_group_families = ["default", "default"]
    worker.group_block_len = {0: [64], 1: [64]}
    worker.group_num_layers = {0: 1, 1: 1}
    request = TestKVPoolWorkerProcessLayerData._make_gva_request(num_groups=2, load_spec=LoadSpec(0, 16, True))
    worker.m_store.batch_get_key_info.return_value = [MagicMock(size=MagicMock(return_value=0))]

    with pytest.raises(RuntimeError, match="multi-group KV load failed"):
        worker._prepare_load_gvas([request])

    worker.m_store.batch_add_lease.assert_not_called()
    worker.m_store.batch_remove_lease.assert_not_called()
    assert worker.get_block_ids_with_load_errors() == set()
    assert request.load_keys is None


def test_mtp_full_hit_load_keeps_tail_recompute_when_no_blocks_are_available(gva_worker):
    worker = gva_worker
    worker.use_eagle = True
    request = TestKVPoolWorkerProcessLayerData._make_gva_request(load_spec=LoadSpec(0, 15, True))
    request.block_hashes = []
    request.block_ids_np = np.array([], dtype=np.int64)
    request.block_ids_by_group_np = [request.block_ids_np]
    worker._prepare_load_gvas([request])
    worker._process_load_for_layer_batch([request], 0)
    assert worker.layer_load_tasks[0] == []
    worker.m_store.batch_get_key_info.assert_not_called()


def start_patch(test: unittest.TestCase, *args, **kwargs):
    patcher = patch(*args, **kwargs)
    mocked = patcher.start()
    test.addCleanup(patcher.stop)
    return mocked


def make_worker(
    test: unittest.TestCase,
    *,
    kv_role="kv_producer",
    tp_rank=0,
    tp_size=1,
    num_kv_heads=1,
    num_layers=2,
    extra_config=None,
    use_layerwise=False,
    use_mla=False,
    enable_kv_events=False,
    num_hidden_layers=None,
    kv_cache_config=None,
    prefix_match_unit=None,
):
    module = "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker"
    start_patch(test, f"{module}.get_tensor_model_parallel_rank", return_value=tp_rank)
    start_patch(test, f"{module}.get_tensor_model_parallel_world_size", return_value=tp_size)
    pcp_group = start_patch(test, f"{module}.get_pcp_group")
    pcp_group.return_value.world_size = 1
    start_patch(test, f"{module}.get_decode_context_model_parallel_world_size", return_value=1)
    start_patch(test, f"{module}.get_decode_context_model_parallel_rank", return_value=0)
    importlib = start_patch(test, f"{module}.importlib")
    importlib.import_module.return_value = MagicMock()

    config = MagicMock()
    config.model_config.model = "org/llama-7b"
    config.model_config.use_mla = use_mla
    config.model_config.hf_text_config = MagicMock(spec=[])
    if num_hidden_layers is not None:
        config.model_config.hf_text_config.num_hidden_layers = num_hidden_layers
    config.model_config.get_num_layers.return_value = num_layers
    config.model_config.get_total_num_kv_heads.return_value = num_kv_heads
    config.parallel_config.data_parallel_rank = 0
    config.parallel_config.rank = 0
    config.parallel_config.pipeline_parallel_size = 1
    config.kv_transfer_config.kv_role = kv_role
    config.kv_transfer_config.kv_connector_extra_config = {
        "backend": "mooncake",
        **(extra_config or {}),
    }
    config.cache_config.block_size = 16
    if prefix_match_unit is not None:
        config.cache_config.prefix_match_unit = prefix_match_unit
    if kv_cache_config is not None:
        config.scheduler_config.disable_hybrid_kv_cache_manager = False
    config.kv_events_config = None
    if enable_kv_events:
        config.kv_events_config = MagicMock(enable_kv_cache_events=True)

    from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker import KVPoolWorker

    return KVPoolWorker(config, use_layerwise=use_layerwise, kv_cache_config=kv_cache_config)


class TestKVPoolWorkerHelpers(unittest.TestCase):
    """Test the pure helper methods on KVPoolWorker without full init."""

    def _make_worker_class(self):
        from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker import KVPoolWorker

        return KVPoolWorker

    def test_check_all_layers_exists(self):
        cls = self._make_worker_class()
        cases = [
            ([1, 1, 1, 1, 1, 1], 3, [1, 1]),
            ([1, 1, 0, 1, 1, 1], 3, [0, 1]),
            ([0, 0, 0], 3, [0]),
        ]
        for exists, num_layers, expected in cases:
            with self.subTest(exists=exists):
                self.assertEqual(cls.check_all_layers_exists(None, exists, num_layers), expected)

    def test_uses_mamba_kv_cache_inside_uniform_group(self):
        from vllm.v1.kv_cache_interface import MambaSpec, UniformTypeKVCacheSpecs

        cls = self._make_worker_class()
        mamba_spec = MambaSpec(block_size=384, shapes=((1,),), dtypes=(np.dtype("float32"),))
        uniform_spec = UniformTypeKVCacheSpecs.from_specs({"mamba.layer": mamba_spec})
        self.assertIsNotNone(uniform_spec)
        kv_cache_config = SimpleNamespace(kv_cache_groups=[SimpleNamespace(kv_cache_spec=uniform_spec)])

        self.assertTrue(cls._uses_mamba_kv_cache(True, kv_cache_config))

    def test_find_all_continuous_hit_positions(self):
        cls = self._make_worker_class()
        cases = [
            ([[1, 1, 0], [1, 0, 1]], [16, 32, 48], 3, [16]),
            ([[1, 1, 1], [1, 1, 1]], [16, 32, 48], 3, [16, 32, 48]),
            ([[0, 1], [1, 0]], [16, 32], 2, []),
            ([], [], 0, []),
        ]
        for exists, positions, count, expected in cases:
            with self.subTest(exists=exists):
                result = cls.find_all_continuous_hit_positions(exists, positions, count, 48, 16)
                self.assertEqual(result, expected)

    def test_find_all_discontinuous_hit_positions(self):
        cls = self._make_worker_class()
        positions = [16, 32, 48, 64, 80, 96]
        cases = [
            ([[0, 0, 1, 0, 0, 1], [0, 0, 1, 0, 0, 1]], 128, [48, 96]),
            ([[0, 0, 1, 0, 0, 1], [0, 0, 1, 0, 0, 0]], 128, [48]),
            ([[0, 0, 1, 0, 0, 1], [0, 0, 1, 0, 0, 1]], 64, [48]),
        ]
        for exists, token_len, expected in cases:
            with self.subTest(exists=exists, token_len=token_len):
                result = cls.find_all_discontinuous_hit_positions(exists, positions, 6, token_len, 16)
                self.assertEqual(result, expected)

    def test_find_all_continuous_hit_positions_all_one(self):
        cls = self._make_worker_class()
        arr = [[1, 1, 1], [1, 1, 1]]
        result = cls.find_all_continuous_hit_positions(arr, [16, 32, 48], 3, 48, 16)
        self.assertEqual(result, [16, 32, 48])

    def test_find_all_continuous_hit_positions_first_pos(self):
        cls = self._make_worker_class()
        arr = [[0, 1], [1, 0]]
        result = cls.find_all_continuous_hit_positions(arr, [16, 32], 2, 48, 16)
        self.assertEqual(result, [])

    def test_find_all_continuous_hit_positions_empty(self):
        cls = self._make_worker_class()
        result = cls.find_all_continuous_hit_positions([], [], 0, 48, 16)
        self.assertEqual(result, [])

    def test_wait_for_layer_load_fallback_waits_for_reuse(self):
        cls = self._make_worker_class()
        worker = cls.__new__(cls)
        worker.current_layer = 0
        worker.num_layers = 1
        worker.layer_load_tasks = [[]]
        worker.prefetch_layer_map = {}
        worker.layer_load_finished_events = [threading.Event()]
        worker.kv_recv_thread = MagicMock()
        worker.external_slot_release_waiter = MagicMock()
        worker._submit_ready_layer_loads = MagicMock()

        worker.wait_for_layer_load()

        worker.external_slot_release_waiter.assert_called_once_with(0)

    def test_find_all_discontinuous_hit_positions_all_tp_hits(self):
        cls = self._make_worker_class()
        arr = [[0, 0, 1, 0, 0, 1], [0, 0, 1, 0, 0, 1]]
        result = cls.find_all_discontinuous_hit_positions(arr, [16, 32, 48, 64, 80, 96], 6, 128, 16)
        self.assertEqual(result, [48, 96])

    def test_find_all_discontinuous_hit_positions_some_tp_hits(self):
        cls = self._make_worker_class()
        arr = [[0, 0, 1, 0, 0, 1], [0, 0, 1, 0, 0, 0]]
        result = cls.find_all_discontinuous_hit_positions(arr, [16, 32, 48, 64, 80, 96], 6, 128, 16)
        self.assertEqual(result, [48])

    def test_partial_prefill_block_index_boundaries(self):
        self.assertEqual(get_partial_block_index(20, 16, 1, True), 1)
        self.assertEqual(get_partial_block_index(32, 16, 1, True), 1)
        self.assertIsNone(get_partial_block_index(32, 16, 2, True))
        self.assertIsNone(get_partial_block_index(20, 16, 1, False))

    def test_find_all_discontinuous_hit_positions_all_tp_hits_with_limits(self):
        cls = self._make_worker_class()
        arr = [[0, 0, 1, 0, 0, 1], [0, 0, 1, 0, 0, 1]]
        result = cls.find_all_discontinuous_hit_positions(arr, [16, 32, 48, 64, 80, 96], 6, 64, 16)
        self.assertEqual(result, [48])

    def test_max_intersection_hit_position_single_group(self):
        cls = self._make_worker_class()
        hits = [[16, 32, 48]]
        self.assertEqual(48, cls._max_intersection_hit_position(hits))

    def test_max_intersection_hit_position_empty_group(self):
        cls = self._make_worker_class()
        hits: list[list[int]] = []
        self.assertEqual(0, cls._max_intersection_hit_position(hits))

    def test_max_intersection_hit_position_multi_group(self):
        cls = self._make_worker_class()
        hits = [[16, 32, 48], [32, 48], [16, 32], [32, 48, 64]]
        self.assertEqual(32, cls._max_intersection_hit_position(hits))

    def test_external_coordinator_lookup_uses_only_lookup_mask(self):
        cls = self._make_worker_class()
        worker = object.__new__(cls)
        worker.hash_block_size = 128
        worker.num_kv_cache_groups = 1
        worker.cache_coordinator = MagicMock()
        worker.cache_coordinator.lcm_block_size = 128
        worker.cache_coordinator.lookup_mask.return_value = ([True],)
        worker.cache_coordinator.store_mask.return_value = ([False],)
        worker.cache_coordinator.find_longest_cache_hit.return_value = ((), 128)
        worker.m_store = MagicMock()
        worker.m_store.exists.return_value = [1]

        worker.token_database = MagicMock()
        worker.token_database.get_block_size.return_value = 128
        worker.token_database.group_cache_families = {"kv": {0: "default"}}
        worker.token_database.process_token_key_strings.side_effect = lambda *args, chunk_filter, **kwargs: (
            [(0, 128, "key", "ab" * 32)] if chunk_filter(0) else []
        )

        hit = worker._lookup_with_coordinator(
            128,
            [b"h0"],
            [0],
            use_layerwise=False,
            include_all_ranks=False,
        )

        self.assertEqual(hit, 128)
        worker.cache_coordinator.lookup_mask.assert_called_once_with(128)
        worker.cache_coordinator.store_mask.assert_not_called()
        worker.m_store.exists.assert_called_once_with(["key"])
        worker.cache_coordinator.find_longest_cache_hit.assert_called_once()
        self.assertFalse(worker.cache_coordinator.find_longest_cache_hit.call_args.kwargs["apply_eagle"])
        worker.token_database.process_tokens.assert_not_called()

    def test_layerwise_multi_group_layout_includes_mtp(self):
        import torch
        from vllm.v1.kv_cache_interface import FullAttentionSpec

        cls = self._make_worker_class()
        worker = object.__new__(cls)
        worker.num_layers = 4
        worker._base_num_layers = 4
        worker.num_kv_cache_groups = 2
        worker.hf_config = SimpleNamespace(num_hidden_layers=4)
        worker.use_layerwise_transfer = True
        worker._extra_config = {"layerwise_num_shared_buffers": 2}
        main_spec = FullAttentionSpec(
            block_size=2,
            num_kv_heads=1,
            head_size=8,
            dtype=torch.float16,
        )
        indexer_spec = FullAttentionSpec(
            block_size=2,
            num_kv_heads=1,
            head_size=4,
            dtype=torch.float16,
        )
        worker.kv_cache_config = SimpleNamespace(
            kv_cache_groups=[
                SimpleNamespace(
                    layer_names=[
                        *(f"model.layers.{layer}.self_attn.attn" for layer in range(4)),
                        "model.mtp.0.self_attn.attn",
                    ],
                    kv_cache_spec=main_spec,
                ),
                SimpleNamespace(
                    layer_names=[
                        *(f"model.layers.{layer}.self_attn.indexer.k_cache" for layer in range(4)),
                    ],
                    kv_cache_spec=indexer_spec,
                ),
            ]
        )

        worker._init_layerwise_config()

        self.assertEqual(worker.num_layers, 5)
        self.assertEqual(worker.physical_layer_to_group_layers[4], [(0, 4)])
        self.assertTrue(worker.layerwise_offload)
        self.assertEqual(worker.independent_layers, [0])
        self.assertEqual(len(worker.layer_load_tasks), 5)
        self.assertEqual(len(worker.layer_save_tasks), 5)


class TestKVPoolWorkerInit(unittest.TestCase):
    """Test KVPoolWorker initialization with mocked dependencies."""

    def _make_vllm_config(self, kv_role="kv_producer", extra_config=None, block_size=16):
        config = MagicMock()
        config.model_config.model = "org/llama-7b"
        config.model_config.use_mla = False
        config.model_config.hf_text_config = MagicMock(spec=[])  # no index_topk
        config.model_config.get_num_layers.return_value = 32
        config.model_config.get_total_num_kv_heads.return_value = 8
        config.model_config.max_model_len = 1024
        config.parallel_config.data_parallel_rank = 0
        config.parallel_config.rank = 0
        config.parallel_config.pipeline_parallel_size = 1
        config.kv_transfer_config.kv_role = kv_role
        config.kv_transfer_config.kv_connector_extra_config = extra_config or {"backend": "mooncake"}
        config.cache_config.block_size = block_size
        config.kv_events_config = None
        return config

    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.importlib")
    @patch(
        "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.get_decode_context_model_parallel_rank"
    )
    @patch(
        "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.get_decode_context_model_parallel_world_size"
    )
    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.get_pcp_group")
    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.get_tensor_model_parallel_world_size")
    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.get_tensor_model_parallel_rank")
    def test_init_basic(self, mock_tp_rank, mock_tp_size, mock_pcp_group, mock_dcp_ws, mock_dcp_rank, mock_importlib):
        mock_tp_rank.return_value = 0
        mock_tp_size.return_value = 1
        pcp_group = MagicMock()
        pcp_group.world_size = 1
        pcp_group.rank_in_group = 0
        mock_pcp_group.return_value = pcp_group
        mock_dcp_ws.return_value = 1
        mock_dcp_rank.return_value = 0

        mock_backend = MagicMock()
        mock_importlib.import_module.return_value = mock_backend

        config = self._make_vllm_config()
        from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker import KVPoolWorker

        worker = KVPoolWorker(config, use_layerwise=False)

        self.assertEqual(worker.block_size, 16)
        self.assertEqual(worker.num_layers, 32)
        self.assertFalse(worker.use_layerwise)
        self.assertFalse(worker.use_mla)
        self.assertEqual(worker.tp_rank, 0)

    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.importlib")
    @patch(
        "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.get_decode_context_model_parallel_rank"
    )
    @patch(
        "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.get_decode_context_model_parallel_world_size"
    )
    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.get_pcp_group")
    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.get_tensor_model_parallel_world_size")
    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.get_tensor_model_parallel_rank")
    def test_init_mla(self, mock_tp_rank, mock_tp_size, mock_pcp_group, mock_dcp_ws, mock_dcp_rank, mock_importlib):
        mock_tp_rank.return_value = 0
        mock_tp_size.return_value = 1
        pcp_group = MagicMock()
        pcp_group.world_size = 1
        mock_pcp_group.return_value = pcp_group
        mock_dcp_ws.return_value = 1
        mock_dcp_rank.return_value = 0
        mock_importlib.import_module.return_value = MagicMock()

        config = self._make_vllm_config()
        config.model_config.use_mla = True
        from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker import KVPoolWorker

        worker = KVPoolWorker(config, use_layerwise=False)
        self.assertTrue(worker.use_mla)
        self.assertEqual(worker.num_kv_head, 1)

    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.importlib")
    @patch(
        "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.get_decode_context_model_parallel_rank"
    )
    @patch(
        "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.get_decode_context_model_parallel_world_size"
    )
    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.get_pcp_group")
    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.get_tensor_model_parallel_world_size")
    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.get_tensor_model_parallel_rank")
    def test_init_kv_head_less_than_tp(
        self, mock_tp_rank, mock_tp_size, mock_pcp_group, mock_dcp_ws, mock_dcp_rank, mock_importlib
    ):
        mock_tp_rank.return_value = 2
        mock_tp_size.return_value = 8
        pcp_group = MagicMock()
        pcp_group.world_size = 1
        mock_pcp_group.return_value = pcp_group
        mock_dcp_ws.return_value = 1
        mock_dcp_rank.return_value = 0
        mock_importlib.import_module.return_value = MagicMock()

        config = self._make_vllm_config()
        config.model_config.get_total_num_kv_heads.return_value = 4  # < tp_size=8
        from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker import KVPoolWorker

        worker = KVPoolWorker(config, use_layerwise=False)
        self.assertEqual(worker.put_step, 2)  # 8 / 4
        self.assertEqual(worker.head_or_tp_rank, 1)  # 2 // 2

    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.importlib")
    @patch(
        "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.get_decode_context_model_parallel_rank"
    )
    @patch(
        "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.get_decode_context_model_parallel_world_size"
    )
    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.get_pcp_group")
    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.get_tensor_model_parallel_world_size")
    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.get_tensor_model_parallel_rank")
    def test_get_kv_events_empty(
        self, mock_tp_rank, mock_tp_size, mock_pcp_group, mock_dcp_ws, mock_dcp_rank, mock_importlib
    ):
        mock_tp_rank.return_value = 0
        mock_tp_size.return_value = 1
        pcp_group = MagicMock()
        pcp_group.world_size = 1
        mock_pcp_group.return_value = pcp_group
        mock_dcp_ws.return_value = 1
        mock_dcp_rank.return_value = 0
        mock_importlib.import_module.return_value = MagicMock()

        config = self._make_vllm_config()
        from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker import KVPoolWorker

        worker = KVPoolWorker(config, use_layerwise=False)
        events = worker.get_kv_events()
        self.assertEqual(events, [])

    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.importlib")
    @patch(
        "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.get_decode_context_model_parallel_rank"
    )
    @patch(
        "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.get_decode_context_model_parallel_world_size"
    )
    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.get_pcp_group")
    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.get_tensor_model_parallel_world_size")
    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.get_tensor_model_parallel_rank")
    def test_get_kv_events_with_send_thread(
        self, mock_tp_rank, mock_tp_size, mock_pcp_group, mock_dcp_ws, mock_dcp_rank, mock_importlib
    ):
        mock_tp_rank.return_value = 0
        mock_tp_size.return_value = 1
        pcp_group = MagicMock()
        pcp_group.world_size = 1
        mock_pcp_group.return_value = pcp_group
        mock_dcp_ws.return_value = 1
        mock_dcp_rank.return_value = 0
        mock_importlib.import_module.return_value = MagicMock()

        config = self._make_vllm_config()
        config.kv_events_config = MagicMock()
        config.kv_events_config.enable_kv_cache_events = True
        from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker import KVPoolWorker

        worker = KVPoolWorker(config, use_layerwise=False)
        worker.kv_send_thread = MagicMock()
        worker.kv_send_thread.get_kv_events.return_value = [MagicMock()]
        events = worker.get_kv_events()
        self.assertEqual(len(events), 1)

    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.importlib")
    @patch(
        "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.get_decode_context_model_parallel_rank"
    )
    @patch(
        "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.get_decode_context_model_parallel_world_size"
    )
    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.get_pcp_group")
    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.get_tensor_model_parallel_world_size")
    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.get_tensor_model_parallel_rank")
    def test_consumer_partition_config(
        self, mock_tp_rank, mock_tp_size, mock_pcp_group, mock_dcp_ws, mock_dcp_rank, mock_importlib
    ):
        mock_tp_rank.return_value = 0
        mock_tp_size.return_value = 1
        pcp_group = MagicMock()
        pcp_group.world_size = 1
        mock_pcp_group.return_value = pcp_group
        mock_dcp_ws.return_value = 1
        mock_dcp_rank.return_value = 0
        mock_importlib.import_module.return_value = MagicMock()

        config = self._make_vllm_config(
            kv_role="kv_consumer",
            extra_config={
                "backend": "mooncake",
                "consumer_is_to_put": True,
                "prefill_pp_layer_partition": "16,16",
                "prefill_pp_size": "2",
            },
        )
        config.model_config.hf_text_config.num_hidden_layers = 32
        from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker import KVPoolWorker

        worker = KVPoolWorker(config, use_layerwise=False)
        self.assertIsNotNone(worker.token_database.partitions)
        self.assertEqual(worker.token_database.partitions, [16, 16])


class TestKVPoolWorkerRegisterAndTransfer(unittest.TestCase):
    """Test register_kv_caches, start_load_kv, wait_for_save, get_finished, lookup_scheduler."""

    def _patch_all(self):
        """Return a dict of started patches."""
        patches = {
            "tp_rank": patch(
                "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.get_tensor_model_parallel_rank",
                return_value=0,
            ),
            "tp_size": patch(
                "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.get_tensor_model_parallel_world_size",
                return_value=1,
            ),
            "pcp_group": patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.get_pcp_group"),
            "dcp_ws": patch(
                "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.get_decode_context_model_parallel_world_size",
                return_value=1,
            ),
            "dcp_rank": patch(
                "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.get_decode_context_model_parallel_rank",
                return_value=0,
            ),
            "importlib": patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.importlib"),
        }
        mocks = {}
        for name, p in patches.items():
            mocks[name] = p.start()
        pcp_group = MagicMock()
        pcp_group.world_size = 1
        mocks["pcp_group"].return_value = pcp_group
        mocks["importlib"].import_module.return_value = MagicMock()
        self._patches = patches
        return mocks

    def _stop_all(self):
        for p in self._patches.values():
            p.stop()

    def _make_config(self, kv_role="kv_producer", extra_config=None, block_size=16):
        config = MagicMock()
        config.model_config.model = "org/llama-7b"
        config.model_config.use_mla = False
        config.model_config.hf_text_config = MagicMock(spec=[])
        config.model_config.max_model_len = 1024
        config.model_config.get_num_layers.return_value = 2
        config.model_config.get_total_num_kv_heads.return_value = 1
        config.parallel_config.data_parallel_rank = 0
        config.parallel_config.rank = 0
        config.parallel_config.pipeline_parallel_size = 1
        config.kv_transfer_config.kv_role = kv_role
        config.kv_transfer_config.kv_connector_extra_config = extra_config or {"backend": "mooncake"}
        config.cache_config.block_size = block_size
        config.kv_events_config = None
        return config

    def _make_worker(self, kv_role="kv_producer", extra_config=None, use_layerwise=False):
        self._patch_all()
        config = self._make_config(kv_role=kv_role, extra_config=extra_config)
        from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker import KVPoolWorker

        worker = KVPoolWorker(config, use_layerwise=use_layerwise)
        return worker

    def setUp(self):
        self._patches = {}

    def tearDown(self):
        self._stop_all()

    def test_register_kv_caches_non_mla(self):
        worker = self._make_worker()
        fake_cache = MagicMock()
        fake_cache.shape = [100, 16, 8, 64]
        fake_cache.element_size.return_value = 2
        fake_cache.data_ptr.return_value = 10000
        kv_caches = {"layer.0": (fake_cache, fake_cache)}
        # init_store + register_buffer now happen directly in register_kv_caches
        # (no separate init_backend handshake). Mark threads as already started
        # so we only exercise the buffer-registration path.
        worker._transfer_threads_started = True
        worker.register_kv_caches(kv_caches)
        self.assertEqual(len(worker.group_kv_caches_base_addr[0]), 2)
        worker.m_store.register_buffer.assert_called_once()

    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.threading.Event")
    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.KVCacheStoreRecvingThread")
    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.KVCacheStoreSendingThread")
    def test_transfer_threads_use_grouped_block_sizes(self, send_thread, recv_thread, event):
        worker = self._make_worker(kv_role="kv_both", extra_config={"backend": "mooncake", "load_async": True})
        worker.grouped_block_size = [128, 128, 128, 128, 8, 32]

        worker._start_kv_transfer_threads()

        self.assertEqual(send_thread.call_args.args[2], worker.grouped_block_size)
        self.assertEqual(recv_thread.call_args.args[2], worker.grouped_block_size)
        event.return_value.wait.assert_called()

    def test_register_kv_caches_initializes_layerwise_memcache(self):
        worker = self._make_worker(extra_config={"backend": "memcache"}, use_layerwise=True)
        fake_cache = MagicMock()
        fake_cache.shape = [100, 16, 8, 64]
        fake_cache.element_size.return_value = 2
        fake_cache.data_ptr.return_value = 10000
        worker._transfer_threads_started = True

        worker.register_kv_caches({"layer.0": (fake_cache, fake_cache)})

        worker.m_store.ensure_initialized.assert_called_once_with()
        worker.m_store.register_buffer.assert_called_once()

    def test_start_load_kv_sync(self):
        worker = self._make_worker()
        worker.m_store.get = MagicMock()
        # Setup token database
        worker.token_database.set_group_buffers({0: [1000, 2000]}, {0: [160]})

        load_spec = LoadSpec(vllm_cached_tokens=0, kvpool_cached_tokens=16, can_load=True, token_len=16)
        req = ReqMeta(
            req_id="r1",
            token_len_chunk=16,
            block_ids=[0],
            block_hashes=["h0"],
            load_spec=load_spec,
        )
        meta = AscendConnectorMetadata(set(), set())
        meta.add_request(req)
        worker.start_load_kv(meta)
        worker.m_store.get.assert_called_once()

    @patch(
        "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.KVCacheStoreRecvingThread.start",
        autospec=True,
    )
    def test_async_load_failure_is_reported_by_worker(self, start_thread):
        worker = self._make_worker(kv_role="kv_consumer", extra_config={"load_async": True})
        worker.m_store.get = MagicMock()
        worker.token_database.set_group_buffers({0: [1000]}, {0: [160]})
        worker.m_store.get.return_value = [1]
        start_thread.side_effect = lambda thread: thread.ready_event.set()
        worker._start_kv_transfer_threads()

        req = ReqMeta(
            req_id="r1",
            token_len_chunk=16,
            block_ids=[7],
            block_hashes=["h0"],
            load_spec=LoadSpec(0, 16, can_load=True, token_len=16),
        )
        meta = AscendConnectorMetadata(set())
        meta.add_request(req)
        worker.start_load_kv(meta)

        recv_thread = worker.kv_recv_thread
        recv_thread._handle_request(recv_thread.request_queue.get_nowait())
        self.assertEqual(worker.get_block_ids_with_load_errors(), {7})
        self.assertEqual(worker.get_block_ids_with_load_errors(), set())

    def test_start_load_kv(self):
        cases = [
            (16, [0], ["h0"], LoadSpec(0, 16, True, token_len=16), True),
            (64, [99], ["h0", "h1", "h2", "h3"], LoadSpec(0, 64, True, token_len=64), True),
            (16, [0], ["h0"], None, False),
        ]
        for token_len, block_ids, hashes, load_spec, should_load in cases:
            with self.subTest(token_len=token_len, block_ids=block_ids, load_spec=load_spec):
                worker = self._make_worker()
                worker.m_store.get = MagicMock()
                worker.token_database.set_group_buffers({0: [1000]}, {0: [160]})
                req = ReqMeta(
                    req_id="r1",
                    token_len_chunk=token_len,
                    block_ids=block_ids,
                    block_hashes=hashes,
                    load_spec=load_spec,
                )
                meta = AscendConnectorMetadata(set())
                meta.add_request(req)
                worker.start_load_kv(meta)
                self.assertEqual(worker.m_store.get.called, should_load)
                if block_ids == [99]:
                    _, addrs, sizes = worker.m_store.get.call_args.args
                    self.assertEqual(addrs, [[1000 + 99 * 160]])
                    self.assertEqual(sizes, [[160]])

    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.KVCacheStoreRecvingThread")
    def test_async_recv_thread_shares_invalid_block_state(self, recv_thread_cls):
        worker = self._make_worker(
            kv_role="kv_consumer",
            extra_config={"backend": "mooncake", "load_async": True},
        )
        recv_thread = MagicMock()

        def create_recv_thread(*args, **kwargs):
            args[6].set()
            return recv_thread

        recv_thread_cls.side_effect = create_recv_thread

        worker._start_kv_transfer_threads()

        kwargs = recv_thread_cls.call_args.kwargs
        self.assertIs(kwargs["invalid_block_ids"], worker._invalid_block_ids)
        self.assertIs(
            kwargs["invalid_block_ids_lock"],
            worker._invalid_block_ids_lock,
        )
        kwargs["invalid_block_ids"].add(7)
        self.assertEqual(worker.get_block_ids_with_load_errors(), {7})

    def test_wait_for_save_waits_for_save(self):
        worker = self._make_worker()
        worker.kv_send_thread = MagicMock()

        req = ReqMeta(
            req_id="r1",
            token_len_chunk=16,
            block_ids=[0],
            block_hashes=["h0"],
            can_save=True,
        )
        meta = AscendConnectorMetadata(set(), set())
        meta.add_request(req)
        worker.wait_for_save(meta)
        worker.kv_send_thread.add_stored_request.assert_called_with("r1")
        worker.kv_send_thread.add_request.assert_called_once()
        worker.kv_send_thread.request_queue.join.assert_called_once()

    def test_wait_for_save_skip_non_save(self):
        worker = self._make_worker()
        worker.kv_send_thread = MagicMock()

        req = ReqMeta(
            req_id="r1",
            token_len_chunk=16,
            block_ids=[0],
            block_hashes=["h0"],
            can_save=False,
        )
        meta = AscendConnectorMetadata(set(), set())
        meta.add_request(req)
        worker.wait_for_save(meta)
        worker.kv_send_thread.add_stored_request.assert_not_called()
        worker.kv_send_thread.request_queue.join.assert_not_called()

    def test_get_finished_producer(self):
        worker = self._make_worker(kv_role="kv_producer")

        send_thread = MagicMock()
        send_thread.get_and_clear_finished_requests.return_value = {"r1"}
        worker.kv_send_thread = send_thread

        meta = AscendConnectorMetadata(set(), set())
        done_s, done_r = worker.get_finished({"r1"}, meta)
        self.assertIn("r1", done_s)
        self.assertEqual(done_r, set())

    def test_get_finished_consumer(self):
        worker = self._make_worker(kv_role="kv_consumer")
        meta = AscendConnectorMetadata(set(), set())
        done_s, done_r = worker.get_finished(set(), meta)
        self.assertEqual(done_s, set())

    def test_lookup_scheduler_all_cached(self):
        worker = self._make_worker()
        worker.m_store.exists.return_value = [1, 1]
        result = worker.lookup_scheduler(32, ["h0", "h1"], use_layerwise=False)
        self.assertEqual(result, 32)

    def test_lookup_scheduler_partial(self):
        worker = self._make_worker()
        worker.m_store.exists.return_value = [1, 0]
        result = worker.lookup_scheduler(32, ["h0", "h1"], use_layerwise=False)
        self.assertEqual(result, 16)

    def test_lookup_scheduler_exception(self):
        worker = self._make_worker()
        worker.m_store.exists.side_effect = Exception("fail")
        result = worker.lookup_scheduler(32, ["h0", "h1"], use_layerwise=False)
        self.assertEqual(result, 0)

    def test_lookup_all_cached(self):
        worker = self._make_worker()
        worker.m_store.exists.return_value = [1, 1]
        result = worker.lookup(32, ["h0", "h1"], use_layerwise=False)
        self.assertEqual(result, 32)

    def test_lookup_partial(self):
        worker = self._make_worker()
        worker.m_store.exists.return_value = [1, 0]
        result = worker.lookup(32, ["h0", "h1"], use_layerwise=False)
        self.assertEqual(result, 16)

    def test_lookup_exception(self):
        worker = self._make_worker()
        worker.m_store.exists.side_effect = Exception("fail")
        result = worker.lookup(32, ["h0", "h1"], use_layerwise=False)
        self.assertEqual(result, 0)

    def test_lookup_layerwise(self):
        worker = self._make_worker()
        worker.m_store.exists.return_value = [1, 1, 1, 1]
        for method in (worker.lookup, worker.lookup_scheduler):
            with self.subTest(method=method.__name__):
                self.assertEqual(method(32, ["h0", "h1"], use_layerwise=True), 32)

    def test_lookup_scheduler_multi_tp(self):
        self._stop_all()
        patches = {
            "tp_rank": patch(
                "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.get_tensor_model_parallel_rank",
                return_value=0,
            ),
            "tp_size": patch(
                "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.get_tensor_model_parallel_world_size",
                return_value=2,
            ),
            "pcp_group": patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.get_pcp_group"),
            "dcp_ws": patch(
                "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.get_decode_context_model_parallel_world_size",
                return_value=1,
            ),
            "dcp_rank": patch(
                "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.get_decode_context_model_parallel_rank",
                return_value=0,
            ),
            "importlib": patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.importlib"),
        }
        mocks = {}
        for name, p in patches.items():
            mocks[name] = p.start()
        pcp_group = MagicMock()
        pcp_group.world_size = 1
        mocks["pcp_group"].return_value = pcp_group
        mocks["importlib"].import_module.return_value = MagicMock()
        self._patches = patches

        config = self._make_config()
        config.model_config.get_total_num_kv_heads.return_value = 2
        from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker import KVPoolWorker

        worker = KVPoolWorker(config, use_layerwise=False)
        # 2 blocks * 2 tp_ranks = 4 keys
        worker.m_store.exists.return_value = [1, 1, 1, 1]
        result = worker.lookup_scheduler(32, ["h0", "h1"], use_layerwise=False)
        self.assertEqual(result, 32)


class TestKVPoolWorkerBuildConnectorWorkerMeta(unittest.TestCase):
    """Test build_connector_worker_meta method."""

    def _make_worker(self):
        return make_worker(self)

    def test_build_connector_worker_meta(self):
        from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.kv_transfer import KVCacheStoreSendingThread

        cases = [(False, None, None), (True, None, None), (True, {}, None), (True, {1: 2}, {1: 2})]
        for use_mamba, events, expected in cases:
            with self.subTest(use_mamba=use_mamba, events=events):
                worker = self._make_worker()
                worker.use_mamba = use_mamba
                if events is not None:
                    worker.kv_send_thread = MagicMock(spec=KVCacheStoreSendingThread)
                    worker.kv_send_thread.get_completed_events.return_value = events
                else:
                    worker.kv_send_thread = None
                result = worker.build_connector_worker_meta()
                self.assertEqual(None if result is None else result.completed_events, expected)


class TestKVPoolWorkerGetFinishedAsync(unittest.TestCase):
    """Test get_finished with async recv thread."""

    def _make_worker(self, kv_role="kv_consumer"):
        return make_worker(self, kv_role=kv_role, extra_config={"load_async": True})

    def test_get_finished_async_recv_thread(self):
        worker = self._make_worker(kv_role="kv_consumer")
        worker.load_async = True

        recv_thread = MagicMock()
        recv_thread.get_and_clear_finished_requests.return_value = {"r1"}
        worker.kv_recv_thread = recv_thread
        worker.kv_send_thread = None

        loading_req_ids = {"r1"}
        meta = AscendConnectorMetadata(set(), loading_req_ids=loading_req_ids)
        done_s, done_r = worker.get_finished(set(), meta)
        self.assertEqual(done_s, set())
        self.assertEqual(done_r, {"r1"})
        recv_thread.get_and_clear_finished_requests.assert_called_once_with(loading_req_ids)

        recv_thread.reset_mock()
        recv_thread.get_and_clear_finished_requests.return_value = set()
        meta = AscendConnectorMetadata({"r_preempted"}, loading_req_ids=set())
        worker.get_finished(set(), meta)
        recv_thread.discard_finished_requests.assert_called_once_with({"r_preempted"})

    def test_get_finished_layerwise_send_thread(self):
        worker = self._make_worker(kv_role="kv_producer")
        worker.use_layerwise = True

        send_thread = MagicMock()
        send_thread.get_and_clear_finished_requests.return_value = set()
        worker.kv_send_thread = send_thread
        worker.kv_recv_thread = None

        meta = AscendConnectorMetadata(set())
        done_s, done_r = worker.get_finished(set(), meta)
        self.assertEqual(done_s, set())
        self.assertEqual(done_r, set())
        send_thread.get_and_clear_finished_requests.assert_called_once_with()


class TestKVPoolWorkerStartLoadKVAsync(unittest.TestCase):
    """Test start_load_kv with load_async=True."""

    def _make_worker(self):
        worker = make_worker(self, kv_role="kv_consumer", extra_config={"load_async": True})
        worker.load_async = True
        return worker

    def test_start_load_kv_async(self):
        worker = self._make_worker()
        recv_thread = MagicMock()
        worker.kv_recv_thread = recv_thread

        load_spec = LoadSpec(vllm_cached_tokens=0, kvpool_cached_tokens=16, can_load=True, token_len=16)
        req = ReqMeta(
            req_id="r1",
            token_len_chunk=16,
            block_ids=[0],
            block_hashes=["h0"],
            load_spec=load_spec,
        )
        meta = AscendConnectorMetadata(set())
        meta.add_request(req)
        worker.start_load_kv(meta)
        recv_thread.add_request.assert_called_once_with(req)

        recv_thread.reset_mock()
        worker = self._make_worker()
        worker.kv_recv_thread = recv_thread
        worker.start_load_kv(AscendConnectorMetadata(set()))
        recv_thread.add_request.assert_not_called()


class TestKVPoolWorkerProcessLayerData(unittest.TestCase):
    """Test process_layer_data and related layerwise methods."""

    def _make_worker(self):
        return make_worker(self)

    def _make_gva_worker(self, num_groups=1):
        worker = make_worker(self, extra_config={"backend": "memcache"}, use_layerwise=True)
        worker.layerwise_offload = True
        worker.num_kv_cache_groups = num_groups
        worker.grouped_block_size = [16] * num_groups
        worker.kv_cache_group_families = ["default"] * num_groups
        worker.group_block_len = {group_id: [64] for group_id in range(num_groups)}
        worker.group_num_layers = {group_id: 1 for group_id in range(num_groups)}
        worker.hash_block_size = 16
        worker.page_size_bytes = 64
        worker.head_or_tp_rank = 0
        worker.m_store = MagicMock()
        return worker

    @staticmethod
    def _make_gva_request(num_groups=1, load_spec=None, can_save=None):
        block_ids_by_group = [[7 + group_id] for group_id in range(num_groups)]
        return ReqMeta(
            req_id="r1",
            token_len_chunk=16,
            save_start_token=0,
            save_end_token=16,
            target_token_len=16,
            block_ids=block_ids_by_group[0],
            block_ids_by_group=block_ids_by_group,
            block_hashes=["h0"],
            can_save=can_save,
            load_spec=load_spec,
            block_ids_np=np.asarray(block_ids_by_group[0], dtype=np.int64),
            block_ids_by_group_np=[np.asarray(block_ids, dtype=np.int64) for block_ids in block_ids_by_group],
        )

    def test_set_external_slot_release_waiter_gated_on_layerwise_transfer(self):
        waiter = MagicMock()

        worker = self._make_worker()
        worker.use_layerwise_transfer = False
        self.assertFalse(worker.set_external_slot_release_waiter(waiter))
        self.assertIsNone(worker.external_slot_release_waiter)

        worker.use_layerwise_transfer = True
        self.assertTrue(worker.set_external_slot_release_waiter(waiter))
        self.assertIs(worker.external_slot_release_waiter, waiter)

    def test_set_external_slot_release_waiter_updates_running_recv_thread(self):
        from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.kv_transfer import (
            KVCacheStoreLayerRecvingThread,
        )

        waiter = MagicMock()

        worker = self._make_worker()
        worker.use_layerwise_transfer = True
        worker.kv_recv_thread = MagicMock(spec=KVCacheStoreLayerRecvingThread)
        self.assertTrue(worker.set_external_slot_release_waiter(waiter))
        # A waiter registered after the receive thread started is handed
        # over to the thread directly, not just stored on the worker.
        self.assertIs(worker.kv_recv_thread.external_slot_release_waiter, waiter)

    def test_process_layer_data_empty_requests(self):
        worker = self._make_worker()
        worker.process_layer_data([])
        for layer_tasks in worker.layer_save_tasks:
            self.assertEqual(layer_tasks, [])
        for layer_tasks in worker.layer_load_tasks:
            self.assertEqual(layer_tasks, [])

    def test_empty_layerwise_step_reowns_task_lists(self):
        worker = self._make_worker()
        worker.use_layerwise = True
        old_save_tasks = worker.layer_save_tasks
        old_load_tasks = worker.layer_load_tasks

        worker.start_load_kv(AscendConnectorMetadata(set(), set()))

        for layer_id in range(worker.num_layers):
            self.assertIsNot(worker.layer_save_tasks[layer_id], old_save_tasks[layer_id])
            self.assertIsNot(worker.layer_load_tasks[layer_id], old_load_tasks[layer_id])

    def test_layerwise_load_is_prepared_before_next_save_allocation(self):
        worker = self._make_worker()
        worker.num_layers = 0
        call_order = []
        worker._prepare_load_gvas = MagicMock(side_effect=lambda requests: call_order.append("load"))
        worker._alloc_gvas_for_save = MagicMock(side_effect=lambda requests: call_order.append("save"))
        worker._build_shared_save_data = MagicMock()
        worker._build_shared_load_data = MagicMock()

        worker.process_layer_data([MagicMock()])

        self.assertEqual(call_order, ["load", "save"])

    def test_process_layer_data_reowns_task_lists_before_populating(self):
        worker = self._make_worker()
        old_save_tasks = worker.layer_save_tasks
        old_load_tasks = worker.layer_load_tasks
        save_marker = MagicMock()
        load_marker = MagicMock()
        worker._process_save_for_layer_batch = MagicMock(
            side_effect=lambda _requests, layer_id, *_args: worker.layer_save_tasks[layer_id].append(save_marker)
        )
        worker._process_load_for_layer_batch = MagicMock(
            side_effect=lambda _requests, layer_id, *_args: worker.layer_load_tasks[layer_id].append(load_marker)
        )
        worker._prepare_load_gvas = MagicMock()
        worker._alloc_gvas_for_save = MagicMock()
        worker._build_shared_save_data = MagicMock()
        worker._build_shared_load_data = MagicMock()

        worker.process_layer_data([MagicMock()])

        for layer_id in range(worker.num_layers):
            self.assertIsNot(worker.layer_save_tasks[layer_id], old_save_tasks[layer_id])
            self.assertIsNot(worker.layer_load_tasks[layer_id], old_load_tasks[layer_id])
            old_save_tasks[layer_id].clear()
            old_load_tasks[layer_id].clear()
            self.assertEqual(worker.layer_save_tasks[layer_id], [save_marker])
            self.assertEqual(worker.layer_load_tasks[layer_id], [load_marker])

    def test_build_shared_save_data_marks_last_actual_task(self):
        from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.kv_transfer import (
            KVCacheStoreLayerSendingThread,
        )

        worker = self._make_worker()
        worker.num_layers = 3
        worker.num_kv_cache_groups = 1
        first_task = LayerTransferTask(layer_id=0, block_ranges=[])
        last_task = LayerTransferTask(layer_id=1, block_ranges=[])
        worker.layer_save_tasks = [[first_task], [last_task], []]
        shared = SharedBlockData(
            block_ids_arr=np.asarray([0]),
            block_gvas_arr=np.asarray([100]),
            req_ids=["r1"],
            is_last_chunks=[True],
            save_keys=["k0"],
        )
        send_thread = object.__new__(KVCacheStoreLayerSendingThread)
        send_thread.build_shared_data = MagicMock(return_value=shared)
        worker.kv_send_thread = send_thread

        worker._build_shared_save_data()

        self.assertEqual(first_task.write_finish_keys, [])
        self.assertEqual(last_task.write_finish_keys, ["k0"])

    def test_process_save_for_layer_batch_skip_no_save(self):
        worker = self._make_worker()
        req = ReqMeta(req_id="r1", token_len_chunk=32, block_ids=[0, 1], block_hashes=["h0", "h1"], can_save=False)
        worker._process_save_for_layer_batch([req], 0)
        self.assertEqual(len(worker.layer_save_tasks[0]), 0)

    def test_process_save_for_layer_batch_skip_zero_range(self):
        worker = self._make_worker()
        req = ReqMeta(
            req_id="r1",
            token_len_chunk=32,
            block_ids=[0, 1],
            block_hashes=["h0", "h1"],
            can_save=True,
            save_start_token=16,
            save_end_token=16,
        )
        worker._process_save_for_layer_batch([req], 0)
        self.assertEqual(len(worker.layer_save_tasks[0]), 0)

    def test_process_load_for_layer_batch_skips(self):
        for load_spec in (None, LoadSpec(0, 0, can_load=False, token_len=0)):
            with self.subTest(load_spec=load_spec):
                worker = self._make_worker()
                req = ReqMeta(
                    req_id="r1",
                    token_len_chunk=32,
                    block_ids=[0, 1],
                    block_hashes=["h0", "h1"],
                    load_spec=load_spec,
                )
                worker._process_load_for_layer_batch([req], 0)
                self.assertEqual(worker.layer_load_tasks[0], [])

    def test_reused_layer_loads_full_cached_prefix(self):
        worker = self._make_worker()
        worker.layerwise_offload = True
        worker.independent_layers = [0]
        request = ReqMeta(
            req_id="r1",
            token_len_chunk=32,
            block_ids=[0, 1],
            block_hashes=["h0", "h1"],
            load_spec=LoadSpec(
                vllm_cached_tokens=16,
                kvpool_cached_tokens=32,
                can_load=True,
                token_len=32,
            ),
        )

        worker._process_load_for_layer_batch([request], 0)
        worker._process_load_for_layer_batch([request], 1)

        independent_range = worker.layer_load_tasks[0][0].block_ranges[0]
        reused_range = worker.layer_load_tasks[1][0].block_ranges[0]
        self.assertEqual((independent_range.start_block, independent_range.end_block), (1, 2))
        self.assertEqual((reused_range.start_block, reused_range.end_block), (0, 2))

    def test_mtp_load_uses_safe_extent_not_store_skip_extent(self):
        worker = self._make_worker()
        worker.use_eagle = True
        worker.layerwise_offload = True
        worker.independent_layers = [0, 1]
        request = ReqMeta(
            req_id="r1",
            token_len_chunk=32,
            block_ids=[0, 1],
            block_hashes=["h0", "h1"],
            load_spec=LoadSpec(
                vllm_cached_tokens=16,
                kvpool_cached_tokens=16,
                can_load=True,
                kvpool_store_skip_tokens=32,
            ),
        )

        worker._process_load_for_layer_batch([request], 1)

        self.assertEqual(worker.layer_load_tasks[1], [])

    def test_mtp_gva_prepare_uses_safe_extent_not_store_skip_extent(self):
        worker = self._make_gva_worker()
        worker.use_eagle = True
        key_info = MagicMock()
        key_info.size.return_value = 64
        key_info.gva_list.return_value = [201]
        worker.m_store.batch_get_key_info.return_value = [key_info]
        worker.m_store.batch_add_lease.return_value = [0]
        request = ReqMeta(
            req_id="r1",
            token_len_chunk=32,
            block_ids_by_group=[[0, 1]],
            block_ids_by_group_np=[np.asarray([0, 1], dtype=np.int64)],
            block_hashes=["h0", "h1"],
            load_spec=LoadSpec(
                vllm_cached_tokens=16,
                kvpool_cached_tokens=16,
                can_load=True,
                kvpool_store_skip_tokens=32,
            ),
        )

        worker._prepare_load_gvas([request])

        queried_keys = worker.m_store.batch_get_key_info.call_args.args[0]
        self.assertEqual(len(queried_keys), 1)

    def test_full_pool_hit_uses_verified_extent(self):
        worker = self._make_gva_worker()
        worker.independent_layers = [0]
        key_info = MagicMock()
        key_info.size.return_value = 64
        key_info.gva_list.return_value = [201]
        worker.m_store.batch_get_key_info.return_value = [key_info]
        worker.m_store.batch_add_lease.return_value = [0]
        request = self._make_gva_request(
            load_spec=LoadSpec(
                vllm_cached_tokens=0,
                kvpool_cached_tokens=15,
                can_load=True,
                kvpool_store_skip_tokens=16,
            ),
            can_save=True,
        )

        worker._prepare_load_gvas([request])
        worker._alloc_gvas_for_save([request])
        worker._process_load_for_layer_batch([request], 1)
        worker._process_save_for_layer_batch([request], 1)

        queried_keys = worker.m_store.batch_get_key_info.call_args.args[0]
        self.assertEqual(len(queried_keys), 1)
        self.assertNotIn("@partial@", queried_keys[0])
        worker.m_store.batch_alloc.assert_not_called()
        load_range = worker.layer_load_tasks[1][0].block_ranges[0]
        self.assertEqual((load_range.start_block, load_range.end_block), (0, 1))
        self.assertIsNone(load_range.partial_block_index)
        self.assertEqual(worker.layer_save_tasks[1], [])

    def test_partial_prefill_is_saved_and_loaded_for_reused_layer(self):
        from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store import (
            pool_worker as _pool_worker,
        )

        self.assertIsNotNone(_pool_worker)
        worker = make_worker(self, extra_config={"backend": "memcache"}, use_layerwise=True)
        worker.layerwise_offload = True
        worker.independent_layers = [0]
        worker.num_kv_cache_groups = 1
        worker.grouped_block_size = [16]
        worker.kv_cache_group_families = ["default"]
        worker.group_block_len = {0: [64]}
        worker.group_num_layers = {0: 1}
        worker.hash_block_size = 16
        worker.page_size_bytes = 64
        worker.head_or_tp_rank = 0
        worker._allocated_gvas = {}
        worker.m_store = MagicMock()
        worker.m_store.batch_alloc.return_value = [101]

        save_request = ReqMeta(
            req_id="r1",
            token_len_chunk=16,
            save_start_token=16,
            save_end_token=16,
            target_token_len=20,
            num_prompt_tokens=32,
            block_ids=[0, 1],
            block_hashes=["h0"],
            can_save=True,
            block_ids_np=np.asarray([0, 1], dtype=np.int64),
            block_ids_by_group_np=[np.asarray([0, 1], dtype=np.int64)],
        )
        worker._alloc_gvas_for_save([save_request])
        worker._process_save_for_layer_batch([save_request], 1)

        self.assertIsNotNone(save_request.save_keys)
        assert save_request.save_keys is not None
        partial_key = save_request.save_keys[0]
        self.assertIn("@partial@r1@0@1@20@", partial_key)
        self.assertEqual(save_request.partial_save_gva_per_group, [101])
        save_range = worker.layer_save_tasks[1][0].block_ranges[0]
        self.assertEqual(save_range.partial_block_index, 1)

        normal_info = MagicMock()
        normal_info.size.return_value = 64
        normal_info.gva_list.return_value = [201]
        partial_info = MagicMock()
        partial_info.size.return_value = 64
        partial_info.gva_list.return_value = [202]
        worker.m_store.batch_get_key_info.return_value = [
            normal_info,
            partial_info,
        ]
        worker.m_store.batch_add_lease.return_value = [0, 0]

        load_request = ReqMeta(
            req_id="r1",
            token_len_chunk=16,
            target_token_len=24,
            num_prompt_tokens=32,
            block_ids=[0, 1],
            block_hashes=["h0"],
            load_spec=LoadSpec(
                vllm_cached_tokens=20,
                kvpool_cached_tokens=20,
                can_load=True,
            ),
            block_ids_np=np.asarray([0, 1], dtype=np.int64),
            block_ids_by_group_np=[np.asarray([0, 1], dtype=np.int64)],
        )
        worker._prepare_load_gvas([load_request])
        worker._process_load_for_layer_batch([load_request], 0)
        worker._process_load_for_layer_batch([load_request], 1)

        queried_keys = worker.m_store.batch_get_key_info.call_args.args[0]
        self.assertIn(partial_key, queried_keys)
        self.assertNotIn(partial_key, worker._allocated_gvas)
        self.assertEqual(load_request.partial_load_gva_per_group, [202])
        self.assertEqual(worker.layer_load_tasks[0], [])
        block_range = worker.layer_load_tasks[1][0].block_ranges[0]
        self.assertEqual(
            (
                block_range.start_block,
                block_range.end_block,
                block_range.partial_block_index,
            ),
            (0, 1, 1),
        )

    def test_layerwise_lease_failure_is_not_copied(self):
        worker = self._make_gva_worker()
        key_info = MagicMock()
        key_info.size.return_value = 64
        key_info.gva_list.return_value = [201]
        worker.m_store.batch_get_key_info.return_value = [key_info]
        worker.m_store.batch_add_lease.return_value = [-1]
        request = self._make_gva_request(
            load_spec=LoadSpec(
                vllm_cached_tokens=16,
                kvpool_cached_tokens=16,
                can_load=True,
            ),
        )

        worker._prepare_load_gvas([request])

        self.assertEqual(request.load_block_gvas_by_group_np[0].tolist(), [0])
        self.assertEqual(request.load_keys, [])
        self.assertEqual(worker.get_block_ids_with_load_errors(), {7})

    def test_partial_lease_retries_until_snapshot_is_readable(self):
        worker = self._make_gva_worker()
        full_info = MagicMock()
        full_info.size.return_value = 64
        full_info.gva_list.return_value = [201]
        partial_info = MagicMock()
        partial_info.size.return_value = 64
        partial_info.gva_list.return_value = [202]
        worker.m_store.batch_get_key_info.return_value = [
            full_info,
            partial_info,
        ]
        worker.m_store.batch_add_lease.side_effect = [
            [0, -3101],
            [0],
        ]
        request = ReqMeta(
            req_id="r1",
            token_len_chunk=16,
            target_token_len=24,
            block_ids=[7, 8],
            block_hashes=["h0"],
            load_spec=LoadSpec(
                vllm_cached_tokens=20,
                kvpool_cached_tokens=20,
                can_load=True,
            ),
            block_ids_np=np.asarray([7, 8], dtype=np.int64),
            block_ids_by_group_np=[np.asarray([7, 8], dtype=np.int64)],
        )

        with patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.time.sleep") as sleep:
            worker._prepare_load_gvas([request])

        partial_key = worker._make_layerwise_partial_key(request, 0, 1, 20)
        self.assertEqual(
            worker.m_store.batch_add_lease.call_args_list[1].args[0],
            [partial_key],
        )
        sleep.assert_called_once()
        self.assertEqual(request.load_keys, [worker._make_layerwise_full_key(0, "h0"), partial_key])
        self.assertEqual(request.partial_load_gva_per_group, [202])
        self.assertEqual(worker.get_block_ids_with_load_errors(), set())

    def test_multi_group_load_failure_stops_before_forward(self):
        worker = self._make_gva_worker(2)
        valid_info = MagicMock()
        valid_info.size.return_value = 64
        valid_info.gva_list.return_value = [201]
        missing_info = MagicMock()
        missing_info.size.return_value = 0
        missing_info.gva_list.return_value = []
        worker.m_store.batch_get_key_info.side_effect = [
            [valid_info],
            [missing_info],
        ]
        worker.m_store.batch_add_lease.return_value = [0]
        request = self._make_gva_request(
            num_groups=2,
            load_spec=LoadSpec(
                vllm_cached_tokens=16,
                kvpool_cached_tokens=16,
                can_load=True,
            ),
        )

        with self.assertRaisesRegex(
            RuntimeError,
            "multi-group KV load failed",
        ):
            worker._prepare_load_gvas([request])

        group0_key = worker._make_layerwise_full_key(0, "h0")
        worker.m_store.batch_remove_lease.assert_called_once_with([group0_key])

    def test_worker_physical_layer_index_supports_mtp_layers_namespace(self):
        worker = self._make_worker()

        self.assertEqual(
            worker._extract_physical_layer_index(
                "mtp.layers.0.self_attn",
            ),
            worker.num_layers,
        )

    def test_evicted_allocated_gva_is_reallocated(self):
        worker = self._make_gva_worker()
        key = worker._make_layerwise_full_key(0, "h0")
        worker._allocated_gvas[key] = 101
        worker.m_store.batch_is_exist.return_value = [0]
        worker.m_store.batch_alloc.return_value = [202]
        request = self._make_gva_request(can_save=True)

        worker._alloc_gvas_for_save([request])

        worker.m_store.batch_alloc.assert_called_once_with([key], [64])
        self.assertEqual(worker._allocated_gvas[key], 202)
        self.assertEqual(request.block_gvas_by_group_np[0].tolist(), [202])

    def test_partial_decode_is_saved_and_loaded_for_reused_layer(self):
        worker = self._make_worker()
        worker.layerwise_offload = True
        worker.independent_layers = [0]
        request = ReqMeta(
            req_id="r1",
            token_len_chunk=32,
            save_start_token=32,
            save_end_token=32,
            target_token_len=34,
            num_prompt_tokens=32,
            block_ids=[0, 1, 2],
            block_hashes=["h0", "h1"],
            can_save=True,
            load_spec=LoadSpec(
                vllm_cached_tokens=33,
                kvpool_cached_tokens=33,
                can_load=True,
            ),
            partial_save_gva_per_group=[301],
            partial_load_gva_per_group=[302],
        )

        worker._process_save_for_layer_batch([request], 1)
        worker._process_load_for_layer_batch([request], 1)

        save_range = worker.layer_save_tasks[1][0].block_ranges[0]
        load_range = worker.layer_load_tasks[1][0].block_ranges[0]
        self.assertEqual(save_range.partial_block_index, 2)
        self.assertEqual(load_range.partial_block_index, 2)


class TestKVPoolWorkerTpMismatch(unittest.TestCase):
    """Tests for TP-asymmetric prefill/decode strided KV transfer.

    Scenario: decode node (tp2) stores KV, prefill node (tp4) loads/hits.
    Qwen3-8B GQA: num_kv_heads=8 -> decode tp2 holds 4 heads/rank, prefill tp4
    holds 2 heads/rank; effective_tp=4, decode num_sub_keys=2.
    """

    def _make_vllm_config(self, kv_role="kv_consumer", extra_config=None, num_kv_heads=8, use_sparse=False):
        config = MagicMock()
        config.model_config.model = "qwen/qwen3-8b"
        config.model_config.use_mla = False
        if use_sparse:
            config.model_config.hf_text_config = MagicMock()
            config.model_config.hf_text_config.index_topk = 32
        else:
            config.model_config.hf_text_config = MagicMock(spec=[])  # no index_topk
        config.model_config.get_num_layers.return_value = 36
        config.model_config.get_total_num_kv_heads.return_value = num_kv_heads
        config.model_config.max_model_len = 4096
        config.parallel_config.data_parallel_rank = 0
        config.parallel_config.rank = 0
        config.parallel_config.pipeline_parallel_size = 1
        config.kv_transfer_config.kv_role = kv_role
        config.kv_transfer_config.kv_connector_extra_config = extra_config or {"backend": "mooncake"}
        config.cache_config.block_size = 16
        config.kv_events_config = None
        return config

    def _patches(self, tp_rank=0, tp_size=2):
        return [
            patch(
                "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.get_tensor_model_parallel_rank",
                return_value=tp_rank,
            ),
            patch(
                "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.get_tensor_model_parallel_world_size",
                return_value=tp_size,
            ),
            patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.get_pcp_group"),
            patch(
                "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.get_decode_context_model_parallel_world_size",
                return_value=1,
            ),
            patch(
                "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.get_decode_context_model_parallel_rank",
                return_value=0,
            ),
            patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.importlib"),
        ]

    def _start(self, patches):
        mocks = [p.start() for p in patches]
        pcp_group = MagicMock()
        pcp_group.world_size = 1
        mocks[2].return_value = pcp_group  # get_pcp_group -> pcp_group
        mocks[5].import_module.return_value = MagicMock()  # importlib.import_module
        return mocks

    def _make_worker(
        self,
        *,
        tp_size=2,
        tp_rank=0,
        kv_role="kv_consumer",
        extra_config=None,
        num_kv_heads=8,
        use_sparse=False,
        use_layerwise=False,
        use_mla=False,
    ):
        patches = self._patches(tp_rank=tp_rank, tp_size=tp_size)
        self._start(patches)
        try:
            cfg = self._make_vllm_config(
                kv_role=kv_role, extra_config=extra_config, num_kv_heads=num_kv_heads, use_sparse=use_sparse
            )
            cfg.model_config.use_mla = use_mla
            from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker import KVPoolWorker

            return KVPoolWorker(cfg, use_layerwise=use_layerwise)
        finally:
            for p in patches:
                p.stop()

    def _make_strided_worker(self, tp_rank=0):
        worker = self._make_worker(
            tp_rank=tp_rank,
            extra_config={"backend": "mooncake", "prefill_tp_size": 4},
        )
        worker.block_size = 4
        worker.group_kv_caches_base_addr = {0: [0]}
        worker.group_block_len = {0: [16]}
        worker.group_block_stride = {0: [16]}
        worker.sub_size_bytes = 2
        worker.token_database.block_size = [4]
        worker.token_database.hash_block_size = 4
        return worker

    def test_tp_mismatch_detected_decode_tp2_prefill_tp4(self):
        worker = self._make_worker(
            tp_size=2, kv_role="kv_consumer", extra_config={"backend": "mooncake", "prefill_tp_size": 4}, num_kv_heads=8
        )
        self.assertTrue(worker.tp_mismatch)
        self.assertEqual(worker.peer_tp_size, 4)
        self.assertEqual(worker.effective_tp_size, 4)
        self.assertEqual(worker.local_heads_per_rank, 4)
        self.assertEqual(worker.effective_heads_per_rank, 2)
        self.assertEqual(worker.num_sub_keys, 2)

    def test_register_kv_caches_initializes_tp_mismatch_strides(self):
        worker = self._make_worker(
            tp_size=2, kv_role="kv_consumer", extra_config={"backend": "mooncake", "prefill_tp_size": 4}, num_kv_heads=8
        )
        fake_cache = MagicMock()
        fake_cache.shape = [100, 16, 4, 64]
        fake_cache.__getitem__.return_value.numel.return_value = 16 * 4 * 64
        fake_cache.element_size.return_value = 2
        fake_cache.stride.return_value = 16 * 4 * 64
        fake_cache.data_ptr.return_value = 10000
        fake_cache.untyped_storage.return_value.data_ptr.return_value = 10000
        worker._transfer_threads_started = True

        worker.register_kv_caches({"layers.0": (fake_cache, fake_cache)})

        self.assertEqual(worker.per_token_bytes, 512)
        self.assertEqual(worker.sub_size_bytes, 256)

    def test_tp_mismatch_disabled(self):
        cases = [
            ({"backend": "mooncake"}, False),
            ({"backend": "mooncake", "prefill_tp_size": 2}, False),
            ({"backend": "mooncake", "prefill_tp_size": 4}, True),
        ]
        for extra_config, use_mla in cases:
            with self.subTest(extra_config=extra_config, use_mla=use_mla):
                worker = self._make_worker(extra_config=extra_config, use_mla=use_mla)
                self.assertFalse(worker.tp_mismatch)
                self.assertEqual(worker.num_sub_keys, 1)

    def test_tp_mismatch_rejects_incompatible_layouts(self):
        for options in ({"use_sparse": True}, {"use_layerwise": True}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self._make_worker(
                    extra_config={"backend": "mooncake", "prefill_tp_size": 4},
                    **options,
                )

    def test_build_strided_addrs_uses_stride(self):
        worker = self._make_worker(extra_config={"backend": "mooncake", "prefill_tp_size": 4}, num_kv_heads=8)
        # Simulate register_kv_caches outputs (group-0 dict structure).
        worker.block_size = 4
        worker.group_kv_caches_base_addr = {0: [1000]}
        worker.group_block_len = {0: [64]}  # bytes per block
        worker.group_block_stride = {0: [128]}  # padded stride (> block_len)
        worker.sub_size_bytes = 8
        addrs, sizes = worker._build_strided_addrs(block_id=2, token_count=3, sub_idx=1)
        # per_token_bytes = 64 // 4 = 16; block_base = 1000 + 2*128 = 1256
        # sub_idx=1 -> head_offset = 8
        # addrs = [1256+0*16+8, 1256+1*16+8, 1256+2*16+8] = [1264, 1280, 1296]
        self.assertEqual(addrs, [1264, 1280, 1296])
        self.assertEqual(sizes, [8, 8, 8])

    def test_build_tp_mismatch_keys_and_addrs(self):
        worker = self._make_strided_worker(tp_rank=1)

        keys, addrs, sizes, block_ids = worker._build_tp_mismatch_keys_and_addrs(
            block_hashes=[b"h0", b"h1"], block_ids=[10, 11], token_len=8, mask_num=0
        )
        self.assertEqual(len(keys), 4)
        self.assertEqual(len(addrs), 4)
        self.assertEqual(len(sizes), 4)
        self.assertEqual(len(block_ids), 4)
        self.assertIn("@head_or_tp_rank:2", keys[0])
        self.assertIn("@head_or_tp_rank:3", keys[1])

        keys, addrs, sizes, block_ids = worker._build_tp_mismatch_keys_and_addrs(
            block_hashes=[b"h0", b"h1"], block_ids=[10], token_len=8, mask_num=0
        )
        self.assertEqual(len(keys), 2)
        self.assertEqual(len(addrs), 2)
        self.assertEqual(len(sizes), 2)
        self.assertEqual(block_ids, [10, 10])
        self.assertTrue(keys[0].endswith(f"@{b'h1'.hex()}"))

    def test_load_kv_tp_mismatch_calls_backend_get(self):
        worker = self._make_strided_worker()
        worker.m_store = MagicMock()
        worker.m_store.get.return_value = [0]  # success

        worker._load_kv_tp_mismatch(block_hashes=[b"h0"], block_ids=[5], token_len=4, mask_num=0)
        worker.m_store.get.assert_called_once()

    def test_store_kv_tp_mismatch_skips_when_not_stored(self):
        worker = self._make_worker(extra_config={"backend": "mooncake", "prefill_tp_size": 4}, num_kv_heads=8)
        worker.kv_send_thread = MagicMock()
        worker.kv_send_thread.is_stored_request.return_value = False
        req = ReqMeta(
            req_id="r1", token_len_chunk=4, block_ids_by_group=[[5]], block_hashes=[b"h0"], current_event=None
        )
        worker._store_kv_tp_mismatch(req)
        worker.kv_send_thread.dec_stored_request.assert_not_called()

    def test_store_kv_tp_mismatch_decrements_on_success_and_error(self):
        for put_error in (None, RuntimeError("put failed")):
            with self.subTest(put_error=put_error):
                worker = self._make_strided_worker()
                worker.m_store = MagicMock()
                worker.m_store.put.side_effect = put_error
                worker.enable_kv_events = False
                send_thread = MagicMock()
                send_thread.is_stored_request.return_value = True
                send_thread.lookup.return_value = [False, True]
                worker.kv_send_thread = send_thread
                req = ReqMeta(
                    req_id="r1",
                    token_len_chunk=4,
                    block_ids_by_group=[[5]],
                    block_hashes=[b"h0"],
                    current_event=None,
                )

                if put_error:
                    with self.assertRaises(RuntimeError):
                        worker._store_kv_tp_mismatch(req)
                else:
                    worker._store_kv_tp_mismatch(req)
                    self.assertEqual(len(worker.m_store.put.call_args.args[0]), 1)
                send_thread.dec_stored_request.assert_called_once_with("r1")


if __name__ == "__main__":
    unittest.main()
