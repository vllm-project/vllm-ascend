# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Unit tests for the Ascend sleep-mode backend."""

from types import SimpleNamespace
from unittest.mock import MagicMock, call

import pytest
from vllm.device_allocator.sleep_mode_backend import SleepModeBackendFactory

from vllm_ascend.device_allocator.sleep_mode_backend import CaMemBackend

SLEEP_TAGS = ("weights", "runtime", "cudagraph")


@pytest.fixture
def allocator(monkeypatch):
    import vllm_ascend.device_allocator.sleep_mode_backend as backend_module

    mock_allocator = MagicMock()
    monkeypatch.setattr(backend_module.CaMemAllocator, "get_instance", lambda: mock_allocator)
    return mock_allocator


def test_backend_suspend_forwards_discardable(allocator):
    backend = CaMemBackend()

    backend.suspend({0x1000, 0x2000})

    allocator.sleep.assert_called_once_with(SLEEP_TAGS, {0x1000, 0x2000})
    assert backend.state() == "SUSPENDED"


def test_backend_suspend_defaults_to_no_discardable(allocator):
    backend = CaMemBackend()

    backend.suspend()

    allocator.sleep.assert_called_once_with(SLEEP_TAGS, ())


def test_backend_resume_and_discard(allocator):
    backend = CaMemBackend()

    backend.resume(["weights"])
    allocator.wake_up.assert_called_once_with(["weights"])
    assert backend.state() == "RUNNING"

    backend.discard(("kv_cache",))
    allocator.discard.assert_called_once_with(("kv_cache",))
    assert backend.state() == "SUSPENDED"


def test_backend_accepts_allocator_factory(allocator):
    backend = CaMemBackend(lambda: allocator)

    backend.suspend({0x1000})

    allocator.sleep.assert_called_once_with(SLEEP_TAGS, {0x1000})


def test_backend_registered_with_upstream_factory():
    assert SleepModeBackendFactory.get_backend_class("camem") is CaMemBackend


def test_worker_import_registers_backend():
    import vllm_ascend.worker.worker  # noqa: F401

    assert SleepModeBackendFactory.get_backend_class("camem") is CaMemBackend


def test_factory_creates_camem_backend():
    model_config = SimpleNamespace(sleep_mode_backend="camem")

    backend = SleepModeBackendFactory.create_backend(model_config)

    assert isinstance(backend, CaMemBackend)


def test_graph_resume_waits_for_model_tags(allocator):
    backend = CaMemBackend()
    reuse = MagicMock(reusing=True)
    reuse.complete_after.side_effect = [False, True]
    backend.resume(["weights"], graph_reuse=reuse)
    allocator.wake_up.assert_called_once_with(["weights"], defer_cudagraph=True)
    assert backend.state() == "RESUMING"
    backend.resume(["kv_cache"], graph_reuse=reuse)
    assert allocator.wake_up.call_args_list[-2:] == [
        call(["kv_cache"], defer_cudagraph=True),
        call(["cudagraph"]),
    ]
    assert backend.state() == "RESUMING"
    backend.finish_graph_recovery()
    assert backend.state() == "RUNNING"


def test_graph_resume_reports_recoverable_graph_failure(allocator):
    backend = CaMemBackend()
    allocator.wake_up.side_effect = [None, RuntimeError("restore failed")]
    reuse = MagicMock(reusing=True)
    backend.resume(None, graph_reuse=reuse)
    assert reuse.reason == "graph_pool_restore_failed"
    assert reuse.signature is None
    assert backend.state() == "RESUMING"
    allocator.pointer_to_data = {1: SimpleNamespace(is_asleep=True)}
    with pytest.raises(RuntimeError, match="asleep allocations"):
        backend.finish_graph_recovery()
    allocator.pointer_to_data = {}
    backend.finish_graph_recovery()
    assert backend.state() == "RUNNING"


def test_graph_resume_rejects_indeterminate_native_state(allocator):
    backend = CaMemBackend()
    allocator.wake_up.side_effect = [None, RuntimeError("restore failed")]
    allocator._check_memory_state.side_effect = RuntimeError("restart the worker")
    with pytest.raises(RuntimeError, match="restart the worker"):
        backend.resume(None, graph_reuse=MagicMock(reusing=True))
