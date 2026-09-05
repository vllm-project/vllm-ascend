import contextlib
import multiprocessing
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.mp.transfer import TransferChannel

STUB = str(Path(__file__).with_name("_transfer_process_stub.py"))


def open_channel(timeout=5):
    channel = TransferChannel(timeout=timeout, command=[sys.executable, STUB])
    channel.call("init", {})
    return channel


@pytest.fixture
def channel():
    channel = open_channel()
    try:
        yield channel
    finally:
        with contextlib.suppress(RuntimeError):
            channel.close()


def test_concurrent_callers_and_out_of_order_completion(channel):
    with ThreadPoolExecutor(max_workers=4) as executor:
        assert list(executor.map(lambda i: channel.call("echo", i), range(20))) == list(range(20))
    slow = channel.submit("store", {"delay": 0.2, "value": "slow"})
    fast = channel.submit("load", {"delay": 0, "value": "fast"})
    assert channel.wait(fast) == "fast"
    assert not slow.done()
    assert channel.wait(slow) == "slow"


def test_close_drains_accepted_work_and_reaps_child(channel):
    pending = channel.submit("store", {"delay": 0.1, "value": 42})
    channel.close()
    assert pending.result() == 42
    assert channel.process.returncode == 0
    assert not channel._io.is_alive()
    with pytest.raises(RuntimeError, match="closed"):
        channel.submit("echo", 1)


def test_remote_exception_reaches_caller(channel):
    with pytest.raises(RuntimeError, match="backend operation failed"):
        channel.call("fail")
    assert channel.call("echo", "healthy") == "healthy"


def test_child_death_fails_all_waiters(channel):
    futures = [channel.submit("store", {"delay": 3, "value": i}) for i in range(4)]
    channel.process.kill()
    for future in futures:
        with pytest.raises(RuntimeError, match="channel stopped"):
            channel.wait(future)


def test_timeout_does_not_replay_or_leave_waiters_blocked():
    channel = open_channel(timeout=1)
    try:
        future = channel.submit("store", {"delay": 5, "value": 42})
        with pytest.raises(RuntimeError):
            channel.wait(future)
        with pytest.raises(RuntimeError):
            channel.submit("echo")
    finally:
        with contextlib.suppress(RuntimeError):
            channel.close()
    assert channel.process.poll() is not None


def test_parent_lifetime_eof_stops_child_during_io(channel):
    channel.submit("store", {"delay": 30, "value": 42})
    os.close(channel._parent_write)
    channel._parent_write = None
    assert channel.process.wait(timeout=3) == 1


def _daemon_worker(connection):
    try:
        channel = open_channel()
        try:
            connection.send(channel.call("echo", "started from daemon"))
        finally:
            channel.close()
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
    channel = TransferChannel(timeout=5, command=[sys.executable, STUB])
    try:
        with pytest.raises(RuntimeError, match="initialization failed"):
            channel.call("init", {"fail_init": True})
    finally:
        channel.close()
    assert channel.process.returncode == 0


def test_socket_initialization_failure_wakes_startup_and_reaps_child():
    module = "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.mp.transfer"
    error = RuntimeError("context initialization failed")
    process = MagicMock()
    with (
        patch(f"{module}.Context", side_effect=error),
        patch(f"{module}.subprocess.Popen", return_value=process),
        pytest.raises(RuntimeError) as raised,
    ):
        TransferChannel(timeout=1, command=[sys.executable, STUB])
    assert raised.value.__cause__ is error
    process.wait.assert_called_once()
