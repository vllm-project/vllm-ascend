# SPDX-License-Identifier: Apache-2.0
"""Regression coverage for the two-stage P-side prefetch pipeline."""

import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import tests.ut.distributed.ascend_store._mock_deps  # noqa: F401
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.kv_transfer import KVCacheStoreLayerRecvingThread
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker import KVPoolWorker


def make_worker():
    worker = KVPoolWorker.__new__(KVPoolWorker)
    worker.kvpp_offload = True
    worker.use_layerwise = True
    worker.num_layers = 6
    worker.current_layer = 0
    worker.next_layer_to_submit = 0
    worker.prefetch_layer_map = {4: 2, 5: 3}
    worker.layer_load_tasks = [[object()], [object()], [], [], [], []]
    worker.layer_save_tasks = [[] for _ in range(6)]
    worker.layer_load_finished_events = [threading.Event() for _ in range(6)]
    worker.kv_recv_thread = MagicMock(spec=KVCacheStoreLayerRecvingThread)
    return worker


def test_startup_primes_first_two_layers_without_attention_gate():
    worker = make_worker()
    worker.process_layer_data = MagicMock()
    worker.start_load_kv(SimpleNamespace(requests=[object()]))
    queued = [call.args[0] for call in worker.kv_recv_thread.add_request.call_args_list]
    assert [task.layer_id for task in queued] == [0, 1]
    assert all(task.attention_start_gate is None for task in queued)


def test_empty_metadata_still_primes_reuse_lifetimes():
    worker = make_worker()
    worker.process_layer_data = MagicMock()
    worker.start_load_kv(SimpleNamespace(requests=[]))
    assert worker.next_layer_to_submit == 2


@pytest.mark.parametrize("owner", [False, True])
def test_h2d_advances_by_global_layer_not_number_of_owner_tasks(owner):
    worker = make_worker()
    worker.layer_load_tasks[2] = [object()] if owner else []
    worker._submit_ready_layer_loads(startup=True)
    worker.layer_load_finished_events[0].set()
    worker.wait_for_layer_load()
    queued = [call.args[0] for call in worker.kv_recv_thread.add_request.call_args_list]
    assert [task.layer_id for task in queued] == [0, 1, 2]
    assert (queued[-1].attention_start_gate is not None) == owner
    assert worker.layer_load_finished_events[0].is_set()
    # Do not wait for future H2D on the compute thread.
    assert not worker.layer_load_finished_events[2].is_set()
    worker.current_layer = 1
    worker.layer_load_finished_events[1].set()
    worker.wait_for_layer_load()
    assert worker.next_layer_to_submit == 4


def test_peer_without_h2d_still_waits_for_previous_buffer_save():
    worker = make_worker()
    worker.next_layer_to_submit = 4
    worker._submit_ready_layer_loads()
    task = worker.kv_recv_thread.add_request.call_args.args[0]
    assert task.transfer_tasks == []
    assert task.wait_for_save_layer == 2


def test_kvpp_and_attention_can_both_observe_load_completion():
    worker = make_worker()
    worker._extract_physical_layer_index = lambda _: 0
    worker.layer_load_finished_events[0].set()
    worker.wait_for_layer_load()
    worker.wait_for_kvpp_cache("layer.0")
    assert worker.layer_load_finished_events[0].is_set()


def test_completed_event_does_not_hide_transfer_failure():
    worker = make_worker()
    worker._extract_physical_layer_index = lambda _: 0
    worker.layer_load_finished_events[0].set()
    worker.kv_recv_thread.raise_if_failed.side_effect = RuntimeError("H2D failed")
    with pytest.raises(RuntimeError, match="H2D failed"):
        worker.wait_for_kvpp_cache("layer.0")


def test_only_owner_keeps_h2d_tasks_but_single_writer_saves_are_preserved():
    worker = make_worker()
    worker.kvpp_layer_owners = {i: i // 3 for i in range(6)}
    worker.kvpp_rank = 1
    worker.physical_layer_to_group_layers = {}
    worker.backend_name = "memcache"
    worker._compute_reachable_store_masks = MagicMock(return_value=None)
    worker._process_save_for_layer_batch = lambda requests, layer, *args: worker.layer_save_tasks[layer].append("save")
    worker._process_load_for_layer_batch = lambda requests, layer, *args: worker.layer_load_tasks[layer].append("load")
    worker._prepare_load_gvas = MagicMock()
    worker._alloc_gvas_for_save = MagicMock()
    worker._build_shared_save_data = MagicMock()
    worker._build_shared_load_data = MagicMock()
    worker.process_layer_data([SimpleNamespace()])
    assert worker.layer_load_tasks == [[], [], [], ["load"], ["load"], ["load"]]
    assert worker.layer_save_tasks == [["save"] for _ in range(6)]


def test_end_of_forward_releases_leases_but_only_completes_last_chunks():
    worker = make_worker()
    worker.current_layer = worker.num_layers - 1
    worker.sync_save_events = [MagicMock() for _ in range(worker.num_layers)]
    worker.layer_save_finished_events = [threading.Event() for _ in range(worker.num_layers)]
    worker.kv_send_thread = MagicMock()
    worker.m_store = MagicMock()
    requests = [
        SimpleNamespace(
            req_id="partial", is_last_chunk=False, load_keys=["shared"], load_spec=SimpleNamespace(can_load=True)
        ),
        SimpleNamespace(
            req_id="done", is_last_chunk=True, load_keys=["shared", "other"], load_spec=SimpleNamespace(can_load=True)
        ),
        SimpleNamespace(req_id="no_load", is_last_chunk=True, load_keys=None, load_spec=None),
    ]
    worker.save_kv_layer(SimpleNamespace(requests=requests))
    worker.m_store.batch_remove_lease.assert_called_once_with(["shared", "other"])
    worker.kv_recv_thread.set_finished_request.assert_called_once_with("done")
    assert worker.current_layer == worker.num_layers


def test_start_load_resets_previous_forward_events_before_priming():
    worker = make_worker()
    for event in worker.layer_load_finished_events:
        event.set()
    worker.process_layer_data = MagicMock()
    worker.kv_recv_thread.add_request.side_effect = lambda task: (
        None
        if not worker.layer_load_finished_events[task.layer_id].is_set()
        else pytest.fail("stale completion observed")
    )
    worker.start_load_kv(SimpleNamespace(requests=[]))
    assert not any(event.is_set() for event in worker.layer_load_finished_events)
    assert worker.kv_recv_thread.final_layer_id == -1


@pytest.mark.parametrize("kvpp", [False, True])
def test_next_forward_waits_for_single_writer_publication(monkeypatch, kvpp):
    from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store import pool_worker

    worker = make_worker()
    worker.kvpp_offload = kvpp
    worker.layerwise_offload = True
    worker.backend_name = "memcache"
    worker.put_step = 2
    reached_barrier = threading.Event()
    published = threading.Event()
    loaded = threading.Event()

    def barrier():
        reached_barrier.set()
        assert published.wait(timeout=5)

    monkeypatch.setattr(pool_worker, "get_tp_group", lambda: SimpleNamespace(barrier=barrier))
    worker.process_layer_data = lambda requests: loaded.set()
    thread = threading.Thread(target=worker.start_load_kv, args=(SimpleNamespace(requests=[object()]),))
    thread.start()
    try:
        assert reached_barrier.wait(timeout=5)
        assert not loaded.is_set()
    finally:
        published.set()
        thread.join(timeout=5)
    assert not thread.is_alive()
    assert loaded.is_set()
