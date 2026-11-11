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
