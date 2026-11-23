# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Unit tests for the Ascend sleep-mode backend."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from vllm.device_allocator.sleep_mode_backend import SleepModeBackendFactory

from vllm_ascend.device_allocator.sleep_mode_backend import CaMemBackend


@pytest.fixture
def allocator(monkeypatch):
    import vllm_ascend.device_allocator.sleep_mode_backend as backend_module

    mock_allocator = MagicMock()
    monkeypatch.setattr(backend_module.CaMemAllocator, "get_instance", lambda: mock_allocator)
    return mock_allocator


def test_backend_suspend_level_one(allocator):
    backend = CaMemBackend()

    backend.suspend(1)

    allocator.sleep.assert_called_once_with(("weights",))
    assert backend.state() == "SUSPENDED"


def test_backend_suspend_level_two(allocator):
    backend = CaMemBackend()

    backend.suspend(2)

    allocator.sleep.assert_called_once_with(())


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

    backend.suspend(1)

    allocator.sleep.assert_called_once_with(("weights",))


def test_backend_registered_with_upstream_factory():
    assert SleepModeBackendFactory.get_backend_class("camem") is CaMemBackend


def test_worker_import_registers_backend():
    import vllm_ascend.worker.worker  # noqa: F401

    assert SleepModeBackendFactory.get_backend_class("camem") is CaMemBackend


def test_factory_creates_camem_backend():
    model_config = SimpleNamespace(sleep_mode_backend="camem")

    backend = SleepModeBackendFactory.create_backend(model_config)

    assert isinstance(backend, CaMemBackend)
