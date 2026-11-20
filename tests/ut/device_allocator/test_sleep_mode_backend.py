# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Unit tests for the Ascend sleep-mode backend."""

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


def test_backend_rejects_unknown_level(allocator):
    backend = CaMemBackend()

    with pytest.raises(ValueError, match="Unsupported sleep level"):
        backend.suspend(0)

    allocator.sleep.assert_not_called()


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


def test_backend_is_supported_checks_extension(monkeypatch):
    import vllm_ascend.device_allocator.sleep_mode_backend as backend_module

    monkeypatch.setattr(backend_module, "camem_available", True)
    assert CaMemBackend.is_supported()

    monkeypatch.setattr(backend_module, "camem_available", False)
    assert not CaMemBackend.is_supported()


def test_backend_registered_with_upstream_factory():
    assert SleepModeBackendFactory.get_backend_class("camem") is CaMemBackend
