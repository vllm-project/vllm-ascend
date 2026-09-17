# SPDX-License-Identifier: Apache-2.0
"""EPLB must use non-daemon workers on both CLI import orders."""

import ast
import multiprocessing
from collections import namedtuple
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


def _load_patch_nodes(*names, namespace):
    path = Path(__file__).parents[4] / "vllm_ascend/patch/platform/patch_multiproc_executor.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    nodes = [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names]
    assert len(nodes) == len(names)
    source = "from __future__ import annotations\n" + "\n".join(ast.unparse(node) for node in nodes)
    exec(compile(source, str(path), "exec"), namespace)
    return namespace


@pytest.mark.parametrize("import_order", ["serve_first", "patch_first", "serve_initializing", "custom_executor"])
def test_eplb_executor_updates_headless_alias(import_order):
    original = type("MultiprocExecutor", (), {})
    patched = type("AscendMultiprocExecutor", (original,), {})
    custom = type("CustomExecutor", (), {})
    executor_module = SimpleNamespace(MultiprocExecutor=original)
    modules = {}
    if import_order != "patch_first":
        serve = SimpleNamespace()
        if import_order == "serve_first":
            serve.MultiprocExecutor = original
        elif import_order == "custom_executor":
            serve.MultiprocExecutor = custom
        modules["vllm.entrypoints.cli.serve"] = serve
    namespace = _load_patch_nodes(
        "_patch_multiproc_executor",
        namespace={
            "sys": SimpleNamespace(modules=modules),
            "vllm": SimpleNamespace(v1=SimpleNamespace(executor=SimpleNamespace(multiproc_executor=executor_module))),
            "MultiprocExecutor": original,
            "AscendMultiprocExecutor": patched,
        },
    )
    namespace["_patch_multiproc_executor"]()
    assert executor_module.MultiprocExecutor is patched
    if import_order == "patch_first":
        # No eager CLI import: its eventual `from ... import` gets the patch.
        assert modules == {}
        serve = SimpleNamespace(MultiprocExecutor=executor_module.MultiprocExecutor)
    elif import_order == "serve_initializing":
        assert not hasattr(serve, "MultiprocExecutor")
        serve.MultiprocExecutor = executor_module.MultiprocExecutor
    assert serve.MultiprocExecutor is (custom if import_order == "custom_executor" else patched)
    namespace["_patch_multiproc_executor"]()
    assert serve.MultiprocExecutor is (custom if import_order == "custom_executor" else patched)


@pytest.mark.parametrize("inherited_fds", [None, [7, 8]])
def test_eplb_worker_factory_preserves_lifecycle(inherited_fds):
    ready_reader, ready_writer, death_reader, death_writer = (Mock() for _ in range(4))
    ready_reader.fileno.return_value = 11
    death_writer.fileno.return_value = 12
    context = Mock()
    context.Pipe.side_effect = [(ready_reader, ready_writer), (death_reader, death_writer)]
    worker = type("WorkerProc", (), {"worker_main": staticmethod(Mock())})
    namespace = _load_patch_nodes(
        "AscendWorkerProc",
        namespace={
            "WorkerProc": worker,
            "get_mp_context": lambda: context,
            "UnreadyWorkerProcHandle": namedtuple("UnreadyWorkerProcHandle", "proc rank ready_pipe death_writer"),
        },
    )
    config, lock, shm = object(), object(), object()
    handle = namespace["AscendWorkerProc"].make_worker_process(
        config, 0, 8, "tcp://test", shm, lock, True, inherited_fds
    )
    kwargs = context.Process.call_args.kwargs
    assert kwargs["daemon"] is False
    assert kwargs["target"] is worker.worker_main
    assert kwargs["name"] == "VllmWorker-8"
    assert kwargs["kwargs"] == {
        "vllm_config": config,
        "local_rank": 0,
        "rank": 8,
        "distributed_init_method": "tcp://test",
        "input_shm_handle": shm,
        "ready_pipe": ready_writer,
        "death_pipe": death_reader,
        "shared_worker_lock": lock,
        "is_driver_worker": True,
        "inherited_fds": [] if inherited_fds is None else [7, 8, 11, 12],
    }
    assert inherited_fds is None or inherited_fds == [7, 8]
    context.Process.return_value.start.assert_called_once_with()
    ready_writer.close.assert_called_once_with()
    death_reader.close.assert_called_once_with()
    ready_reader.close.assert_not_called()
    death_writer.close.assert_not_called()
    assert handle.proc is context.Process.return_value
    assert handle.rank == 8


def _manager_worker_main(ready_pipe, **kwargs):
    with multiprocessing.Manager() as manager:
        result = manager.dict(value=3)
        ready_pipe.send((multiprocessing.current_process().daemon, result["value"]))
    ready_pipe.close()


def test_eplb_worker_can_start_manager_process():
    worker = type("WorkerProc", (), {"worker_main": staticmethod(_manager_worker_main)})
    namespace = _load_patch_nodes(
        "AscendWorkerProc",
        namespace={
            "WorkerProc": worker,
            "get_mp_context": lambda: multiprocessing.get_context("spawn"),
            "UnreadyWorkerProcHandle": namedtuple("UnreadyWorkerProcHandle", "proc rank ready_pipe death_writer"),
        },
    )
    handle = namespace["AscendWorkerProc"].make_worker_process(None, 0, 8, "tcp://test", None, None)
    try:
        assert handle.ready_pipe.poll(20), "EPLB Manager did not start in the worker"
        assert handle.ready_pipe.recv() == (False, 3)
        handle.proc.join(20)
        assert handle.proc.exitcode == 0
    finally:
        if handle.proc.is_alive():
            handle.proc.terminate()
            handle.proc.join(5)
        handle.ready_pipe.close()
        handle.death_writer.close()
