import contextlib
import multiprocessing
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.mp.process import TransferProcess

STUB = str(Path(__file__).with_name("_transfer_process_stub.py"))


def open_process(timeout=5):
    process = TransferProcess(timeout=timeout, command=[sys.executable, STUB])
    process.client.call("init", {})
    return process


@pytest.fixture
def transfer_process():
    process = open_process()
    try:
        yield process
    finally:
        with contextlib.suppress(RuntimeError):
            process.close()


def test_concurrent_callers_and_out_of_order_completion(transfer_process):
    client = transfer_process.client
    with ThreadPoolExecutor(max_workers=4) as executor:
        assert list(executor.map(lambda i: client.call("echo", i), range(20))) == list(range(20))
    slow = client.submit("store", {"delay": 0.2, "value": "slow"})
    fast = client.submit("load", {"delay": 0, "value": "fast"})
    assert client.wait(fast) == "fast"
    assert not slow.done()
    assert client.wait(slow) == "slow"


def test_close_drains_accepted_work_and_reaps_child(transfer_process):
    client = transfer_process.client
    pending = client.submit("store", {"delay": 0.1, "value": 42})
    transfer_process.close()
    assert pending.result() == 42
    assert transfer_process.process.returncode == 0
    assert not client._io.is_alive()
    with pytest.raises(RuntimeError, match="closed"):
        client.submit("echo", 1)


def test_remote_exception_reaches_caller(transfer_process):
    client = transfer_process.client
    with pytest.raises(RuntimeError, match="backend operation failed"):
        client.call("fail")
    assert client.call("echo", "healthy") == "healthy"


def test_child_death_fails_all_waiters(transfer_process):
    client = transfer_process.client
    futures = [client.submit("store", {"delay": 3, "value": i}) for i in range(4)]
    transfer_process.process.kill()
    for future in futures:
        with pytest.raises(RuntimeError, match="channel stopped"):
            client.wait(future)


def test_timeout_does_not_replay_or_leave_waiters_blocked():
    process = open_process()
    client = process.client
    client.timeout = 1
    try:
        future = client.submit("store", {"delay": 5, "value": 42})
        with pytest.raises(RuntimeError):
            client.wait(future)
        with pytest.raises(RuntimeError):
            client.submit("echo")
    finally:
        with contextlib.suppress(RuntimeError):
            process.close()
    assert process.process.poll() is not None


def test_parent_lifetime_eof_stops_child_during_io(transfer_process):
    transfer_process.client.submit("store", {"delay": 30, "value": 42})
    os.close(transfer_process._parent_write)
    transfer_process._parent_write = None
    assert transfer_process.process.wait(timeout=3) == 1


def _daemon_worker(connection):
    try:
        process = open_process()
        try:
            connection.send(process.client.call("echo", "started from daemon"))
        finally:
            process.close()
    finally:
        connection.close()


def test_model_worker_can_itself_be_a_multiprocessing_daemon():
    context = multiprocessing.get_context("fork")
    reader, writer = context.Pipe(duplex=False)
    process = context.Process(target=_daemon_worker, args=(writer,), daemon=True)
    process.start()
    writer.close()
    try:
        assert reader.poll(10)
        assert reader.recv() == "started from daemon"
        process.join(5)
        assert process.exitcode == 0
    finally:
        if process.is_alive():
            process.kill()
            process.join(5)
        reader.close()


def test_failed_initialization_can_be_closed():
    process = TransferProcess(timeout=5, command=[sys.executable, STUB])
    try:
        with pytest.raises(RuntimeError, match="initialization failed"):
            process.client.call("init", {"fail_init": True})
    finally:
        process.close()
    assert process.process.returncode == 0


def test_socket_initialization_failure_wakes_startup_and_reaps_child():
    client_module = "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.mp.client"
    process_module = "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.mp.process"
    error = RuntimeError("context initialization failed")
    process = MagicMock()
    with (
        patch(f"{client_module}.Context", side_effect=error),
        patch(f"{process_module}.subprocess.Popen", return_value=process),
        pytest.raises(RuntimeError) as raised,
    ):
        TransferProcess(timeout=1, command=[sys.executable, STUB])
    assert raised.value.__cause__ is error
    process.wait.assert_called_once()
