from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

import tests.ut.distributed.ascend_store._mock_deps  # noqa: F401
from tests.ut.distributed.ascend_store.test_kv_transfer import TestGVALayerTransferFailures as _GVALayerTransferFailures
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store import attention_fence
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend import get_layerwise_protocol
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise_transfer import (
    LayerTransferArrayBuilder,
    LayerwiseTransferPreparer,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import (
    LayerBlockRange,
    LayerSaveTask,
    LayerTransferTask,
    LayerwisePreparation,
    LoadSpec,
    ReqMeta,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker import KVPoolWorker


def make_preparer(backend, groups=1):
    return LayerwiseTransferPreparer(
        backend,
        "model",
        0,
        16,
        enabled=True,
        can_allocate=True,
        block_sizes=[16] * groups,
        group_block_len={group: [16, 16] for group in range(groups)},
        page_size_bytes=16,
        layerwise_offload=True,
        protocol=get_layerwise_protocol("memcache"),
        record_invalid_blocks=MagicMock(),
    )


def make_request(req_id="r1", groups=1, hashes=None):
    hashes = [b"a", b"b"] if hashes is None else hashes
    ids = [np.arange(1, len(hashes) + 1, dtype=np.int64) for _ in range(groups)]
    return ReqMeta(
        req_id,
        token_len_chunk=16 * len(hashes),
        can_save=True,
        block_ids_by_group=[group.tolist() for group in ids],
        block_ids_by_group_np=ids,
        block_hashes=hashes,
        load_spec=LoadSpec(0, 16 * len(hashes), can_load=True),
        is_last_chunk=True,
    )


def key_info(gva):
    return SimpleNamespace(size=lambda: 32, gva_list=lambda: [gva])


def test_variable_layout_array_offsets():
    database = SimpleNamespace(
        group_block_len={0: [10, 2, 20]},
        group_kv_caches_base_addr={0: [100, 200, 300]},
        group_block_stride={0: [1000, 2000, 3000]},
        group_layer_cache_entry_offsets={0: [0, 2, 3]},
    )
    builder = LayerTransferArrayBuilder(database, 2)
    ids, gvas = np.asarray([1]), np.asarray([10000])
    addresses, sizes, remote = builder._build_transfer_arrays(ids, gvas, 0)
    np.testing.assert_array_equal(addresses, [1100, 2200])
    np.testing.assert_array_equal(sizes, [10, 2])
    np.testing.assert_array_equal(remote, [10000, 10010])
    np.testing.assert_array_equal(builder._build_transfer_arrays(ids, gvas, 1)[2], [10012])
    database.group_layer_cache_entry_offsets = {0: [0, 1, 2]}
    with pytest.raises(ValueError, match="Invalid layerwise offsets"):
        LayerTransferArrayBuilder(database, 2)


def test_preparation_runs_once_for_competing_transfer_threads():
    callback = MagicMock()
    preparation = LayerwisePreparation(callback)
    with ThreadPoolExecutor(max_workers=2) as executor:
        list(executor.map(lambda _: preparation.ensure_ready(), range(8)))
    callback.assert_called_once()


def test_preparation_replays_failure_without_retrying_backend():
    callback = MagicMock(side_effect=RuntimeError("failed"))
    preparation = LayerwisePreparation(callback)
    for _ in range(2):
        with pytest.raises(RuntimeError, match="failed"):
            preparation.ensure_ready()
    callback.assert_called_once()


def test_load_queries_once_and_shared_lease_survives_first_completion():
    backend = MagicMock()
    backend.batch_get_key_info.return_value = [key_info(100)]
    backend.batch_add_lease.return_value = [0]
    backend.batch_remove_lease.return_value = 0
    preparer = make_preparer(backend)
    preparer._prepare_load_gvas([make_request("a", hashes=[b"x"]), make_request("b", hashes=[b"x"])])
    backend.batch_get_key_info.assert_called_once()
    assert len(backend.batch_get_key_info.call_args.args[0]) == 1
    backend.batch_add_lease.assert_called_once()
    preparer.release_finished_load_leases({"a"})
    backend.batch_remove_lease.assert_not_called()
    preparer.release_finished_load_leases({"b"})
    backend.batch_remove_lease.assert_called_once()
    assert not preparer.load_lease_refcounts


@pytest.mark.parametrize("results", [[0], [0, 9]])
def test_lease_failure_rolls_back_acquired_keys(results):
    backend = MagicMock()
    backend.batch_get_key_info.return_value = [key_info(100), key_info(200)]
    backend.batch_add_lease.return_value = results
    backend.batch_remove_lease.return_value = 0
    preparer = make_preparer(backend, groups=2)
    with pytest.raises(RuntimeError):
        preparer._prepare_load_gvas([make_request(groups=2, hashes=[b"x"])])
    backend.batch_remove_lease.assert_called_once()
    assert len(backend.batch_remove_lease.call_args.args[0]) == 1
    assert not preparer.load_lease_refcounts


@pytest.mark.parametrize("infos", [[], [SimpleNamespace(size=lambda: 32, gva_list=lambda: [])]])
def test_malformed_metadata_fails_before_lease_acquisition(infos):
    backend = MagicMock()
    backend.batch_get_key_info.return_value = infos
    preparer = make_preparer(backend)
    with pytest.raises(RuntimeError):
        preparer._prepare_load_gvas([make_request(hashes=[b"x"])])
    backend.batch_add_lease.assert_not_called()


def test_saves_allocate_one_batch_and_deduplicate_shared_keys():
    backend = MagicMock()
    backend.batch_alloc.return_value = [100, 200]
    preparer = make_preparer(backend, groups=2)
    requests = [make_request("a", groups=2, hashes=[b"x"]), make_request("b", groups=2, hashes=[b"x"])]
    for request in requests:
        request.load_spec = None
    preparer._alloc_gvas_for_save(requests)
    backend.batch_alloc.assert_called_once()
    assert len(backend.batch_alloc.call_args.args[0]) == 2
    assert requests[0].block_gvas_by_group_np[0].tolist() == [100]
    assert requests[0].block_gvas_by_group_np[1].tolist() == [200]
    assert requests[1].block_gvas_by_group_np[0].tolist() == [0]
    assert requests[1].save_keys == []


def test_invalid_allocation_never_prepares_a_successful_save():
    backend = MagicMock()
    backend.batch_alloc.return_value = []
    preparer = make_preparer(backend)
    request = make_request(hashes=[b"x"])
    request.load_spec = None
    with pytest.raises(RuntimeError, match="batch_alloc returned invalid GVAs"):
        preparer._alloc_gvas_for_save([request])
    assert request.save_keys is None


def test_indexed_gate_records_mtp_mapping_and_resets():
    attention_fence.reset_attention_compute_start_gates(3, {"mtp.layers.0.attn": 2})
    try:
        gate = attention_fence.get_attention_compute_start_gate(2)
        gate.record = MagicMock()
        attention_fence.record_attention_compute_start("mtp.layers.0.attn")
        gate.record.assert_called_once()
        assert attention_fence.get_attention_compute_start_gate(0)._event is None
        attention_fence.reset_attention_compute_start_gates(3)
        assert attention_fence.get_attention_compute_start_gate(2) is not gate
    finally:
        attention_fence.reset_attention_compute_start_gate()


def test_adjacent_reused_layer_does_not_wait_on_its_own_attention():
    worker = KVPoolWorker.__new__(KVPoolWorker)
    worker.kv_recv_thread = MagicMock()
    worker.use_layerwise_transfer = True
    worker.current_layer = 0
    worker.num_layers = 2
    worker.next_layer_to_submit = 0
    worker.num_prefetch_layers = 2
    worker.prefetch_layer_map = {1: 0}
    worker.layer_load_tasks = [[], [LayerTransferTask(1, [])]]
    worker._layer_load_preparation = None
    worker._submit_ready_layer_loads()
    task = worker.kv_recv_thread.add_request.call_args.args[0]
    assert task.layer_id == 1
    assert task.wait_for_save_layer == 0
    assert task.attention_start_gate is None


def test_last_actual_group_task_owns_request_completion():
    request = make_request()
    first = LayerTransferTask(0, [LayerBlockRange(request, 0, 2)])
    last = LayerTransferTask(1, [LayerBlockRange(request, 0, 2)], group_id=1)
    KVPoolWorker._mark_last_layer_tasks([[first], [last], []])
    assert first.finished_req_ids == set()
    assert last.finished_req_ids == {request.req_id}


def test_control_only_save_waits_for_attention_before_slot_reuse():
    thread, _, finished, _ = _GVALayerTransferFailures()._make_sending_thread()
    thread._wait_attention_done = MagicMock(side_effect=lambda _: pytest.fail("attention failed"))
    with pytest.raises(pytest.fail.Exception):
        thread._handle_request(LayerSaveTask(0, []))
    assert not finished.is_set()
    thread._wait_attention_done = MagicMock()
    thread._handle_request(LayerSaveTask(0, []))
    thread._wait_attention_done.assert_called_once_with(0)
    assert finished.is_set()


def test_failed_copy_keeps_request_count_and_slot_owned():
    thread, _, finished, task = _GVALayerTransferFailures()._make_sending_thread()
    thread._batch_copy_with_limits = MagicMock(return_value=7)
    with pytest.raises(RuntimeError, match="batch_copy failed"):
        thread._handle_request(LayerSaveTask(0, [task]))
    assert not finished.is_set()
    assert thread.stored_requests["r1"] == 1
    assert not thread.get_and_clear_finished_requests()


def test_worker_prepares_load_before_save_and_only_once():
    worker = KVPoolWorker.__new__(KVPoolWorker)
    worker.backend_name = "memcache"
    worker.use_block_key_layerwise = False
    worker.use_layerwise_transfer = True
    worker.use_eagle = False
    worker.num_layers = 1
    worker.physical_layer_to_group_layers = {0: [(0, 0)]}
    worker.kv_send_thread = MagicMock()
    worker.kv_recv_thread = MagicMock()
    worker._compute_reachable_store_masks = MagicMock(return_value=None)
    worker._compute_reachable_load_masks = MagicMock(return_value=None)
    worker._build_shared_save_data = MagicMock()
    worker._build_shared_load_data = MagicMock()
    request = make_request()
    block_range = LayerBlockRange(request, 0, 2)
    worker._process_save_for_layer_batch = lambda *args: worker.layer_save_tasks[0].append(
        LayerTransferTask(0, [block_range])
    )
    worker._process_load_for_layer_batch = lambda *args: worker.layer_load_tasks[0].append(
        LayerTransferTask(0, [block_range])
    )
    calls = []
    preparer = MagicMock()
    preparer._prepare_load_gvas.side_effect = lambda _: calls.append("lease")
    preparer._alloc_gvas_for_save.side_effect = lambda _: calls.append("allocate")
    worker._get_layerwise_transfer_preparer = lambda: preparer
    worker.process_layer_data([request])
    assert calls == []
    save_preparation = worker.kv_send_thread.add_request.call_args.args[0]
    load_preparation = worker.kv_recv_thread.add_request.call_args.args[0]
    save_preparation.ensure_ready()
    load_preparation.ensure_ready()
    assert calls == ["lease", "allocate"]
    worker._build_shared_save_data.assert_called_once()
    worker._build_shared_load_data.assert_called_once()


def test_full_and_hbm_tail_loads_have_distinct_shared_arrays():
    from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.kv_transfer import KVCacheStoreLayerRecvingThread

    worker = KVPoolWorker.__new__(KVPoolWorker)
    worker.kv_recv_thread = MagicMock(spec=KVCacheStoreLayerRecvingThread)
    full, tail = object(), object()
    worker.kv_recv_thread.build_shared_data.side_effect = [tail, full]
    first = LayerTransferTask(0, [], uses_hbm_tail=True)
    second = LayerTransferTask(1, [], uses_hbm_tail=False)
    third = LayerTransferTask(2, [], uses_hbm_tail=False)
    worker.layer_load_tasks = [[first], [second], [third]]
    worker._build_shared_load_data()
    assert first.shared_block_data is tail
    assert second.shared_block_data is third.shared_block_data is full
    assert worker.kv_recv_thread.build_shared_data.call_count == 2
