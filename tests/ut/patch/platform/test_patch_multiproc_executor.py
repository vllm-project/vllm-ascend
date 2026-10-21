# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import runpy
import sys
from multiprocessing import Manager, current_process, get_context
from pathlib import Path

import pytest

PATCH_PATH = Path(__file__).resolve().parents[4] / "vllm_ascend/patch/platform/patch_multiproc_executor.py"


def _start_manager(ready_pipe, death_pipe, **kwargs):
    with Manager() as manager:
        shared_dict = manager.dict({"expert_map": None})
        ready_pipe.send((current_process().daemon, dict(shared_dict)))
    ready_pipe.close()
    death_pipe.close()


@pytest.fixture
def executor_modules():
    # Keep vLLM imports out of spawned workers that only exercise Manager().
    import vllm.v1.executor.multiproc_executor as multiproc_executor
    from vllm.entrypoints.cli import serve

    return multiproc_executor, serve


def test_patch_updates_cached_headless_executor(monkeypatch, executor_modules):
    multiproc_executor, serve = executor_modules
    # Reproduce serve.py importing the original class before the EPLB patch.
    monkeypatch.setattr(serve, "MultiprocExecutor", multiproc_executor.MultiprocExecutor)
    monkeypatch.setattr(multiproc_executor, "MultiprocExecutor", multiproc_executor.MultiprocExecutor)

    patched = runpy.run_path(str(PATCH_PATH))

    assert serve.MultiprocExecutor is patched["AscendMultiprocExecutor"]
    assert multiproc_executor.MultiprocExecutor is serve.MultiprocExecutor


def test_patch_does_not_import_cli(monkeypatch, executor_modules):
    multiproc_executor, _ = executor_modules
    monkeypatch.delitem(sys.modules, "vllm.entrypoints.cli.serve")
    monkeypatch.setattr(multiproc_executor, "MultiprocExecutor", multiproc_executor.MultiprocExecutor)

    runpy.run_path(str(PATCH_PATH))

    assert "vllm.entrypoints.cli.serve" not in sys.modules


def test_eplb_worker_can_start_manager(monkeypatch, executor_modules):
    multiproc_executor, serve = executor_modules
    monkeypatch.setattr(multiproc_executor, "MultiprocExecutor", multiproc_executor.MultiprocExecutor)
    monkeypatch.setattr(serve, "MultiprocExecutor", serve.MultiprocExecutor)
    patched = runpy.run_path(str(PATCH_PATH))
    context = get_context("spawn")
    monkeypatch.setitem(patched["AscendWorkerProc"].make_worker_process.__globals__, "get_mp_context", lambda: context)
    monkeypatch.setattr(multiproc_executor.WorkerProc, "worker_main", _start_manager)
    worker_lock = context.Lock()

    handle = patched["AscendWorkerProc"].make_worker_process(
        vllm_config=None,
        local_rank=0,
        rank=0,
        distributed_init_method="",
        input_shm_handle=None,
        shared_worker_lock=worker_lock,
    )
    try:
        assert handle.ready_pipe.poll(30), "Worker did not start its EPLB manager"
        assert handle.ready_pipe.recv() == (False, {"expert_map": None})
        handle.proc.join(timeout=30)
        assert handle.proc.exitcode == 0
    finally:
        if handle.proc.is_alive():
            handle.proc.terminate()
            handle.proc.join(timeout=30)
        handle.ready_pipe.close()
        handle.death_writer.close()


def _make_executor(mq, rank):
    """Build an AscendMultiprocExecutor without starting any worker."""
    from collections import deque
    from unittest.mock import MagicMock

    import vllm_ascend.patch.platform.patch_multiproc_executor as patch_module

    executor = object.__new__(patch_module.AscendMultiprocExecutor)
    executor.is_failed = False
    executor.rpc_broadcast_mq = MagicMock()
    executor.response_mqs = [mq]
    executor.futures_queue = deque()
    return executor


def test_collective_rpc_retries_corrupted_reply():
    from unittest.mock import MagicMock

    from vllm.v1.executor.multiproc_executor import WorkerProc

    import vllm_ascend.patch.platform.patch_multiproc_executor as patch_module

    mq = MagicMock()
    mq.dequeue.side_effect = [
        patch_module.MessageQueueFrameCorruptionError("corrupt"),
        (WorkerProc.ResponseStatus.SUCCESS, "value"),
    ]
    executor = _make_executor(mq, 0)

    assert executor.collective_rpc("execute_model", args=(7,), unique_reply_rank=0) == "value"

    # The same request is re-sent, not a different one.
    assert executor.rpc_broadcast_mq.enqueue.call_count == 2
    first, second = executor.rpc_broadcast_mq.enqueue.call_args_list
    assert first.args[0] == second.args[0]


def test_collective_rpc_does_not_retry_multi_reply_rpc():
    from unittest.mock import MagicMock

    import vllm_ascend.patch.platform.patch_multiproc_executor as patch_module

    mq = MagicMock()
    mq.dequeue.side_effect = patch_module.MessageQueueFrameCorruptionError("corrupt")
    executor = _make_executor(mq, 0)

    with pytest.raises(patch_module.MessageQueueFrameCorruptionError):
        executor.collective_rpc("take_draft_token_ids")

    assert executor.rpc_broadcast_mq.enqueue.call_count == 1


def test_collective_rpc_gives_up_after_one_retry():
    from unittest.mock import MagicMock

    import vllm_ascend.patch.platform.patch_multiproc_executor as patch_module

    mq = MagicMock()
    mq.dequeue.side_effect = [
        patch_module.MessageQueueFrameCorruptionError("corrupt"),
        patch_module.MessageQueueFrameCorruptionError("corrupt again"),
    ]
    executor = _make_executor(mq, 0)

    with pytest.raises(patch_module.MessageQueueFrameCorruptionError):
        executor.collective_rpc("execute_model", unique_reply_rank=0)

    assert executor.rpc_broadcast_mq.enqueue.call_count == 2
