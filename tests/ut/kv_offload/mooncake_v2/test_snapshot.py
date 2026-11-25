# SPDX-License-Identifier: Apache-2.0
"""Snapshot lifecycle contracts; native transport and NPU calls are mocked."""

import queue
import threading
import weakref
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake import base_worker, pull_scheduler, pull_worker
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.connector import MooncakePullConnector
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.pull_scheduler import MooncakePullConnectorScheduler
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.pull_worker import (
    MooncakePullConnectorWorker,
    MooncakePullRecvingThread,
)
from vllm_ascend.distributed.kv_transfer.utils.mooncake_transfer_engine import GlobalTE

from .helpers import make_transfer_metadata
from .test_pull_scheduler import make_pull_scheduler


def make_worker(monkeypatch):
    worker = MooncakePullConnectorWorker.__new__(MooncakePullConnectorWorker)
    worker._recving_thread = None
    worker._registered_regions = SimpleNamespace(ptrs=[100, 100, 200], lengths=[8, 8, 8])
    worker.kv_caches = {"layer": object()}
    worker.kv_transfer_config = SimpleNamespace(kv_port=6000, engine_id="old")
    worker.vllm_config = SimpleNamespace(
        kv_transfer_config=worker.kv_transfer_config,
        parallel_config=SimpleNamespace(data_parallel_rank=1),
    )
    worker.tp_size, worker.pp_size, worker.pcp_size = 2, 2, 1
    worker.tp_rank, worker.pp_rank, worker.pcp_rank = 1, 1, 0
    worker.engine_id = "old"
    worker.xfer_handshake_metadata = object()
    worker.engine = Mock()
    worker.engine.unregister_memory.return_value = 0
    global_te = GlobalTE()
    global_te.transfer_engine = worker.engine
    global_te.is_register_buffer = True
    monkeypatch.setattr(base_worker, "global_te", global_te)
    return worker, global_te


def test_teardown_stops_thread_before_unregister_and_is_repeatable(monkeypatch):
    worker, global_te = make_worker(monkeypatch)
    calls = []
    engine = worker.engine
    thread = Mock()
    thread.shutdown.side_effect = lambda: calls.append("thread")
    engine.unregister_memory.side_effect = lambda ptr: calls.append(ptr) or 0
    worker._recving_thread = thread
    worker.prepare_for_snapshot_restore()
    worker.prepare_for_snapshot_restore()
    assert calls == ["thread", 100, 200]
    assert worker.engine is None and worker._recving_thread is None
    assert worker.xfer_handshake_metadata is None
    assert global_te.transfer_engine is None and not global_te.is_register_buffer


def test_thread_failure_does_not_release_engine(monkeypatch):
    worker, global_te = make_worker(monkeypatch)
    thread = Mock()
    thread.shutdown.side_effect = RuntimeError("did not stop")
    worker._recving_thread = thread
    with pytest.raises(RuntimeError, match="did not stop"):
        worker.prepare_for_snapshot_restore()
    assert worker.engine is global_te.transfer_engine
    worker.engine.unregister_memory.assert_not_called()
    assert worker._recving_thread is thread


def test_unregister_failure_is_not_ignored(monkeypatch):
    worker, global_te = make_worker(monkeypatch)
    worker.engine.unregister_memory.return_value = 7
    with pytest.raises(RuntimeError, match="failed: 7"):
        worker.prepare_for_snapshot_restore()
    assert worker.engine is global_te.transfer_engine


def test_shared_engine_released_after_all_owners_prepare(monkeypatch):
    first, global_te = make_worker(monkeypatch)
    second = MooncakePullConnectorWorker.__new__(MooncakePullConnectorWorker)
    second._recving_thread = None
    second.engine = first.engine
    second._registered_regions = first._registered_regions
    reference = weakref.ref(first.engine)
    first.prepare_for_snapshot_restore()
    assert reference() is second.engine
    second.prepare_for_snapshot_restore()
    assert reference() is None
    assert global_te.transfer_engine is None


def test_rebuild_refreshes_identity_group_and_registers_existing_storage(monkeypatch):
    worker, global_te = make_worker(monkeypatch)
    reference = weakref.ref(worker.engine)
    group = object()
    replacement = Mock()
    replacement.get_rpc_port.return_value = 9999
    caches = worker.kv_caches

    def create(host, device_name):
        assert reference() is None  # No old engine overlaps the replacement.
        assert host == "10.0.0.2" and device_name == "3"
        global_te.transfer_engine = replacement
        return replacement

    monkeypatch.setattr(global_te, "get_transfer_engine", create)
    monkeypatch.setattr(base_worker, "get_tp_group", lambda: group)
    monkeypatch.setattr(base_worker.torch.npu, "current_device", lambda: 3)
    worker.register_kv_caches = Mock()
    worker.rebuild_kv_transfer_endpoint("10.0.0.2", "new")
    assert worker.engine is replacement and worker.tp_group is group
    assert worker.engine_id == worker.kv_transfer_config.engine_id == "new"
    assert worker.side_channel_host == "10.0.0.2"
    assert worker.side_channel_port == 6004 and worker.handshake_port == 6007
    assert worker.te_rpc_port == 9999
    worker.register_kv_caches.assert_called_once_with(caches)


def test_rebuild_requires_registration(monkeypatch):
    worker, global_te = make_worker(monkeypatch)
    worker._registered_regions = None
    with pytest.raises(RuntimeError, match="registered KV caches"):
        worker.rebuild_kv_transfer_endpoint("10.0.0.2", "new")
    assert worker.engine is global_te.transfer_engine


def test_pull_thread_can_exit_blocked_queue_and_release_engine(monkeypatch):
    thread = MooncakePullRecvingThread.__new__(MooncakePullRecvingThread)
    threading.Thread.__init__(thread, daemon=True)
    thread.request_queue = queue.Queue()
    thread.stop_event = threading.Event()
    thread.ready_event = threading.Event()
    thread.executor = ThreadPoolExecutor(max_workers=1)
    thread.device = 0
    thread.engine = object()
    monkeypatch.setattr(pull_worker.torch.npu, "set_device", lambda device: None)
    thread.start()
    assert thread.ready_event.wait(timeout=5)
    try:
        thread.shutdown()
        thread.shutdown()
        assert not thread.is_alive() and thread.engine is None
        assert thread.request_queue.unfinished_tasks == 0
    finally:
        thread.shutdown()


def test_pull_thread_timeout_preserves_engine_and_executor():
    thread = MooncakePullRecvingThread.__new__(MooncakePullRecvingThread)
    threading.Thread.__init__(thread)
    thread._ident = 1
    thread.stop_event = threading.Event()
    thread.request_queue = queue.Queue()
    thread.join = Mock()
    thread.is_alive = Mock(return_value=True)
    thread.executor = Mock()
    engine = thread.engine = object()
    with pytest.raises(RuntimeError, match="did not stop"):
        thread.shutdown()
    assert thread.engine is engine
    thread.executor.shutdown.assert_not_called()


@pytest.mark.parametrize("consumer", [False, True])
def test_scheduler_rebuild_discards_old_threads_and_publishes_fresh_metadata(monkeypatch, consumer):
    scheduler = make_pull_scheduler()
    scheduler.kv_transfer_config = SimpleNamespace(
        is_kv_consumer=consumer,
        is_kv_producer=not consumer,
        engine_id="old",
        kv_port=6000,
    )
    scheduler.vllm_config = SimpleNamespace(parallel_config=SimpleNamespace(data_parallel_rank=1))
    scheduler.max_device_id = 8
    scheduler.tp_size = scheduler.pp_size = 1
    old = Mock()
    scheduler._sending_thread = old
    scheduler._reqs_need_recv["stale"] = object()
    scheduler._reqs_need_send["stale"] = 1
    scheduler._reqs_recv_info["stale"] = object()
    scheduler._reqs_in_batch.add("stale")
    created = []

    def start_receiver(ready):
        fresh = Mock()
        fresh.start.side_effect = ready.set
        fresh.is_alive.return_value = True
        created.append(fresh)
        return fresh

    monkeypatch.setattr(pull_scheduler, "MooncakeSchedulerRecvingThread", start_receiver)
    scheduler.rebuild_kv_transfer_endpoint("10.0.0.2", "new")
    old.shutdown.assert_called_once_with(timeout=10)
    assert scheduler._sending_thread is None
    assert not any(
        (scheduler._reqs_need_recv, scheduler._reqs_need_send, scheduler._reqs_in_batch, scheduler._reqs_recv_info)
    )
    assert scheduler.side_channel_host == "10.0.0.2" and scheduler.side_channel_port == 6009
    assert scheduler.engine_id == "new"
    if consumer:
        assert scheduler._recving_thread is created[0]
    else:

        def start_sender(*args):
            assert args[0:3] == ("10.0.0.2", 6009, "new")
            assert args[3][0].engine_id == "new"
            return start_receiver(args[-1])

        monkeypatch.setattr(pull_scheduler, "MooncakeSchedulerSendingThread", start_sender)
        metadata = replace(make_transfer_metadata(), engine_id="new")
        scheduler.set_xfer_handshake_metadata_from_workers({0: metadata})
        assert scheduler._sending_thread is created[0]


def test_facade_uses_existing_lifecycle_hooks():
    connector = MooncakePullConnector.__new__(MooncakePullConnector)
    connector.connector_worker = Mock(spec=MooncakePullConnectorWorker)
    connector.connector_scheduler = Mock(spec=MooncakePullConnectorScheduler)
    connector.kv_transfer_config = SimpleNamespace(engine_id="old")
    connector._connector_metadata = object()
    connector.prepare_for_snapshot_restore()
    connector.connector_worker.prepare_for_snapshot_restore.assert_called_once_with()
    connector.connector_scheduler.prepare_for_snapshot_restore.assert_called_once_with()
    assert connector._connector_metadata is None
    connector.rebuild_kv_transfer_endpoint("10.0.0.2", "new")
    assert connector.engine_id == connector.kv_transfer_config.engine_id == "new"
    connector.connector_worker.rebuild_kv_transfer_endpoint.assert_called_once_with("10.0.0.2", "new")
    connector.connector_scheduler.rebuild_kv_transfer_endpoint.assert_called_once_with("10.0.0.2", "new")
