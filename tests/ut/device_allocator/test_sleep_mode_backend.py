# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Unit tests for the Ascend sleep-mode backend."""

from types import SimpleNamespace
from unittest.mock import MagicMock

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


@pytest.mark.parametrize("last_tags", [None, ["kv_cache"]])
def test_partial_resume_waits_for_live_allocations(allocator, last_tags):
    weights = SimpleNamespace(tag="weights", is_asleep=True)
    kv = SimpleNamespace(tag="kv_cache", is_asleep=True)
    allocator.pointer_to_data = {1: weights, 2: kv}

    def wake_matching(tags):
        for data in allocator.pointer_to_data.values():
            if tags is None or data.tag in tags:
                data.is_asleep = False

    allocator.wake_up.side_effect = wake_matching
    backend = CaMemBackend()
    backend.suspend()
    backend.resume(["weights"])
    assert not weights.is_asleep and kv.is_asleep
    assert backend.state() == "RESUMING"

    backend.resume(last_tags)
    assert backend.state() == "RUNNING"


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
