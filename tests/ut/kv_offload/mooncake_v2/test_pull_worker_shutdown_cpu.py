# SPDX-License-Identifier: Apache-2.0
"""CPU coverage for the real Pull connector, receiver and executor shutdown chain."""

import queue
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake import pull_worker
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.connector import MooncakePullConnector
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.metadata import MooncakeConnectorMetadata
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.pull_worker import MooncakePullConnectorWorker

from .helpers import make_metadata_groups, make_pp_metadata
from .test_pull_worker import make_req_meta, make_thread

WAIT_SECONDS = 5


class AbortableQueue(queue.Queue):
    """Unblock the test receiver during teardown if production shutdown fails."""

    abort_marker = object()

    def get(self, *args, **kwargs):
        work = super().get(*args, **kwargs)
        if work is self.abort_marker:
            self.task_done()
            raise SystemExit
        return work


@pytest.fixture(params=("batch", "request"))
def pull_shutdown(request, monkeypatch):
    monkeypatch.setattr(pull_worker.torch.npu, "set_device", lambda _device: None)
    with ThreadPoolExecutor(max_workers=1) as executor:
        request_queue = AbortableQueue()
        receiver = make_thread(
            pull_transfer_mode=request.param,
            request_queue=request_queue,
            ready_event=threading.Event(),
            device="cpu",
            executor=executor,
            engine=MagicMock(),
            _shutdown_lock=threading.Lock(),
            _executor_closed=False,
            remote_tp_rank_groups={"engine-p": {0: {(0, 0): [[0]]}}},
            remote_layer_index_pairs={"engine-p": {0: [(0, 0)]}},
        )
        threading.Thread.__init__(receiver, daemon=True)
        # Supply remote metadata in memory. The receiver still builds and submits
        # the real transfer buckets and takes its normal completion path.
        monkeypatch.setattr(
            receiver, "_get_remote_metadata", lambda *_: make_metadata_groups(pp_metadata=make_pp_metadata())
        )
        worker = MooncakePullConnectorWorker.__new__(MooncakePullConnectorWorker)
        worker.kv_transfer_config = SimpleNamespace(is_kv_consumer=True)
        worker._recving_thread = receiver
        connector = MooncakePullConnector.__new__(MooncakePullConnector)
        connector.connector_worker = worker
        connector.connector_scheduler = None
        try:
            yield connector, worker, receiver
        finally:
            if receiver.is_alive():
                request_queue.put(AbortableQueue.abort_marker)
                receiver.join(timeout=WAIT_SECONDS)
                assert not receiver.is_alive(), "test receiver could not be cleaned up"


def test_idle_shutdown_is_repeatable(pull_shutdown):
    connector, worker, receiver = pull_shutdown
    receiver.start()
    assert receiver.ready_event.wait(timeout=WAIT_SECONDS)
    with patch.object(receiver.executor, "shutdown", wraps=receiver.executor.shutdown) as close:
        connector.shutdown()
        assert not receiver.is_alive(), "public shutdown left the idle receiver alive"
        assert worker._recving_thread is None
        assert receiver.request_queue.unfinished_tasks == 0
        connector.shutdown()
        close.assert_called_once_with(wait=True)


def test_shutdown_waits_for_accepted_reads(pull_shutdown, monkeypatch):
    connector, worker, receiver = pull_shutdown
    first_read_started = threading.Event()
    release_read = threading.Event()
    warned = threading.Event()
    errors: list[BaseException] = []

    def read(*_args):
        if first_read_started.is_set():
            return -1
        first_read_started.set()
        if not release_read.wait(timeout=WAIT_SECONDS * 2):
            raise TimeoutError("test did not release the READ")
        return 0

    def stop():
        try:
            connector.shutdown()
        except BaseException as error:
            errors.append(error)

    receiver.engine.batch_transfer_sync_read.side_effect = read
    monkeypatch.setattr(pull_worker, "SHUTDOWN_WARNING_INTERVAL_S", 0.02, raising=False)
    warning = MagicMock(side_effect=lambda *_args: warned.set())
    monkeypatch.setattr(pull_worker.logger, "warning", warning)
    active = MooncakeConnectorMetadata()
    active.requests = {"active": make_req_meta(local=([1, 2],), remote=([3, 4],))}
    queued = MooncakeConnectorMetadata()
    queued.requests = {"queued": make_req_meta(local=([5, 6],), remote=([7, 8],))}
    worker.start_load_kv(active)
    worker.start_load_kv(queued)
    shutdown = threading.Thread(target=stop, daemon=True)
    try:
        receiver.start()
        assert first_read_started.wait(timeout=WAIT_SECONDS)
        shutdown.start()
        assert warned.wait(timeout=WAIT_SECONDS), "the blocked drain did not emit a warning"
        assert shutdown.is_alive()
        assert worker._recving_thread is receiver
        if receiver.pull_transfer_mode == "batch":
            assert receiver.is_alive()
            assert "receiver still draining" in warning.call_args.args[0]
        else:
            assert not receiver.is_alive()
            assert "executor still draining" in warning.call_args.args[0]
        assert receiver.finished_requests.empty(), "READs still in flight cannot be published as finished"
        with pytest.raises(RuntimeError, match="shutting down"):
            worker.start_load_kv(active)
        with pytest.raises(RuntimeError, match="shutting down"):
            receiver.add_not_transfer_request("late-virtual")
    finally:
        release_read.set()
        if shutdown.ident is not None:
            shutdown.join(timeout=WAIT_SECONDS)
    assert not shutdown.is_alive()
    assert not errors
    assert not receiver.is_alive()
    assert worker._recving_thread is None
    assert receiver.request_queue.unfinished_tasks == 0
    assert receiver.engine.batch_transfer_sync_read.call_count == 2
    results = receiver.get_and_clear_finished_requests()
    assert results.finished_recving == {"active", "queued"}
    assert results.failed_recving == {"queued"}
    assert receiver.get_and_clear_invalid_block_ids() == {5, 6}
    with pytest.raises(RuntimeError, match="shutdown"):
        receiver.executor.submit(lambda: None)


def test_shutdown_before_start(pull_shutdown):
    connector, worker, receiver = pull_shutdown
    with patch.object(receiver.executor, "shutdown", wraps=receiver.executor.shutdown) as close:
        connector.shutdown()
        close.assert_called_once_with(wait=True)
    assert worker._recving_thread is None
    assert receiver.request_queue.unfinished_tasks == 0
    with pytest.raises(RuntimeError, match="shutting down"):
        receiver.add_requests("engine-p", {"late": make_req_meta()})


def test_failed_executor_cleanup_preserves_owner_and_can_retry(pull_shutdown):
    connector, worker, receiver = pull_shutdown
    close = receiver.executor.shutdown
    attempts = 0

    def fail_once(*, wait):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("cleanup failed")
        close(wait=wait)

    with patch.object(receiver.executor, "shutdown", side_effect=fail_once):
        with pytest.raises(RuntimeError, match="cleanup failed"):
            connector.shutdown()
        assert worker._recving_thread is receiver
        with pytest.raises(RuntimeError, match="shutting down"):
            receiver.add_not_transfer_request("late-virtual")
        connector.shutdown()
        assert attempts == 2
    assert worker._recving_thread is None
