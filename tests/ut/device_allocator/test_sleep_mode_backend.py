# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from concurrent.futures import ThreadPoolExecutor
from threading import Event
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from vllm.device_allocator.sleep_mode_backend import SleepModeBackendFactory

from vllm_ascend.device_allocator.sleep_mode_backend import (
    AscendLegacySleepModeBackend,
    AscendSleepModeBackend,
    guard_sleep_execution,
    normalize_sleep_mode_backend,
    register_sleep_mode_backends,
)


@pytest.fixture
def backend():
    runtime = MagicMock()
    runtime.allocation_tags.return_value = {"weights", "kv_cache"}
    instance = AscendSleepModeBackend()
    instance.bind_runtime(runtime)
    return instance


def stages(backend):
    return [call.args[0] for call in backend.runtime.agree.call_args_list if ":" not in call.args[0]]


def test_registration_is_idempotent_and_keeps_gpu_cumem(monkeypatch):
    monkeypatch.setattr(SleepModeBackendFactory, "_registry", SleepModeBackendFactory._registry.copy())
    original = SleepModeBackendFactory.get_backend_class("cumem")
    register_sleep_mode_backends()
    register_sleep_mode_backends()
    assert SleepModeBackendFactory.get_backend_class("ascend") is AscendSleepModeBackend
    assert SleepModeBackendFactory.get_backend_class("ascend_legacy") is AscendLegacySleepModeBackend
    assert SleepModeBackendFactory.get_backend_class("cumem") is original


def test_registration_conflict_is_not_swallowed(monkeypatch):
    monkeypatch.setattr(SleepModeBackendFactory, "_registry", {"ascend": lambda: object})
    with pytest.raises(ValueError, match="already registered"):
        register_sleep_mode_backends()


@pytest.mark.parametrize("name", ["cumem", "ascend", "ascend_legacy"])
def test_selection(name, monkeypatch):
    monkeypatch.setattr(SleepModeBackendFactory, "_registry", SleepModeBackendFactory._registry.copy())
    config = SimpleNamespace(sleep_mode_backend=name)
    normalize_sleep_mode_backend(config)
    assert config.sleep_mode_backend == ("ascend_legacy" if name == "cumem" else name)


def test_unknown_selection_fails(monkeypatch):
    monkeypatch.setattr(SleepModeBackendFactory, "_registry", SleepModeBackendFactory._registry.copy())
    with pytest.raises(ValueError, match="Unsupported"):
        normalize_sleep_mode_backend(SimpleNamespace(sleep_mode_backend="not_registered"))


def test_unbound_and_rebinding_fail(backend):
    with pytest.raises(RuntimeError, match="no worker"):
        AscendSleepModeBackend().suspend()
    with pytest.raises(RuntimeError, match="already bound"):
        backend.bind_runtime(MagicMock())


@pytest.mark.parametrize("level", [1, 2])
def test_repeated_cycles_and_stage_order(backend, level):
    for _ in range(10):
        backend.runtime.reset_mock()
        backend.suspend(level)
        assert backend.state() == "SUSPENDED"
        backend.suspend(level)
        assert stages(backend) == ["drain", "backup", "inventory", "cleanup", "quiesce", "unregister", "unmap"]
        backend.resume()
        assert backend.state() == "RUNNING"
        assert stages(backend)[7:] == ["validate_wake", "remap", "reregister", "restore", "rebuild", "ready"]
        backend.resume()
        backend.runtime.wake_allocator.assert_called_once_with(None)


def test_staged_wake_does_not_allow_execution(backend):
    backend.suspend(2)
    backend.resume(["weights"])
    assert backend.state() == "RESUMING"
    backend.runtime.rebuild.assert_called_once_with(["weights"])
    backend.runtime.reregister.assert_not_called()
    with pytest.raises(RuntimeError, match="RESUMING"), backend.execution():
        pytest.fail("unmapped KV storage must not be used")
    backend.resume(["weights"])
    backend.resume(["kv_cache"])
    assert backend.state() == "RUNNING"
    backend.runtime.restore.assert_called_once()


def test_kv_first_is_rejected_before_remap(backend):
    backend.suspend(2)
    with pytest.raises(ValueError, match="weights before"):
        backend.resume(["kv_cache"])
    backend.runtime.wake_allocator.assert_not_called()
    backend.resume()
    assert backend.state() == "RUNNING"


def test_other_allocation_pool_delays_readiness(backend):
    backend.runtime.allocation_tags.return_value.add("runtime")
    backend.suspend()
    backend.resume(["weights", "runtime"])
    assert backend.state() == "RESUMING"
    backend.resume(["runtime"])
    backend.resume()
    assert backend.state() == "RUNNING"


def test_empty_and_invalid_tags_do_not_touch_storage(backend):
    backend.suspend()
    backend.resume([])
    with pytest.raises(ValueError, match="Unknown"):
        backend.resume(["typo"])
    backend.runtime.wake_allocator.assert_not_called()


@pytest.mark.parametrize("level", [0, 3, -1])
def test_invalid_sleep_level(backend, level):
    with pytest.raises(ValueError, match="level"):
        backend.suspend(level)
    backend.runtime.synchronize.assert_not_called()


def test_different_level_cannot_replace_active_cycle(backend):
    backend.suspend(1)
    with pytest.raises(RuntimeError, match="current sleep"):
        backend.suspend(2)


def test_backup_failure_rolls_back_without_unmap(backend):
    backend.runtime.backup.side_effect = ValueError("backup failed")
    with pytest.raises(ValueError, match="backup failed"):
        backend.suspend(2)
    assert backend.state() == "RUNNING"
    backend.runtime.clear_backup.assert_called_once()
    backend.runtime.sleep_allocator.assert_not_called()


@pytest.mark.parametrize("method", ["synchronize", "cleanup", "unregister", "sleep_allocator"])
def test_unsafe_suspend_failure_is_closed(backend, method):
    getattr(backend.runtime, method).side_effect = RuntimeError("injected")
    with pytest.raises(RuntimeError, match="injected"):
        backend.suspend()
    assert backend.state() != "RUNNING"
    with pytest.raises(RuntimeError, match="failed"):
        backend.resume()


@pytest.mark.parametrize("method", ["wake_allocator", "reregister", "restore", "rebuild", "synchronize"])
def test_resume_failure_never_advertises_running(backend, method):
    backend.suspend()
    getattr(backend.runtime, method).side_effect = RuntimeError("injected")
    with pytest.raises(RuntimeError, match="injected"):
        backend.resume()
    assert backend.state() == "RESUMING"
    with pytest.raises(RuntimeError, match="failed"), backend.execution():
        pass


def test_other_rank_failure_prevents_next_stage(backend):
    def agree(stage, error):
        if stage == "cleanup":
            raise RuntimeError("rank 1 failed")

    backend.runtime.agree.side_effect = agree
    with pytest.raises(RuntimeError, match="rank 1"):
        backend.suspend()
    backend.runtime.sleep_allocator.assert_not_called()
    assert backend.state() == "SUSPENDED"


def test_rollback_failure_is_closed(backend):
    backend.runtime.backup.side_effect = RuntimeError("backup")
    backend.runtime.clear_backup.side_effect = RuntimeError("rollback")
    with pytest.raises(RuntimeError, match="backup"):
        backend.suspend()
    with pytest.raises(RuntimeError, match="failed"):
        backend.resume()


def test_close_is_idempotent_and_does_not_wake(backend):
    runtime = backend.runtime
    backend.suspend(2)
    backend.close()
    backend.close()
    runtime.wake_allocator.assert_not_called()
    with pytest.raises(RuntimeError, match="closed"):
        backend.resume()


def test_execution_finishes_before_sleep(backend):
    entered, release, sleep_started = Event(), Event(), Event()

    class Worker:
        sleep_mode_backend = backend

        @guard_sleep_execution
        def execute(self):
            entered.set()
            assert release.wait(5)
            assert not sleep_started.is_set()

    backend.runtime.synchronize.side_effect = sleep_started.set
    with ThreadPoolExecutor(max_workers=2) as pool:
        inference = pool.submit(Worker().execute)
        assert entered.wait(5)
        sleeping = pool.submit(backend.suspend)
        release.set()
        inference.result(5)
        sleeping.result(5)
    assert sleep_started.is_set()


def test_legacy_adapter_calls_existing_path():
    runtime = MagicMock()
    backend = AscendLegacySleepModeBackend()
    backend.bind_runtime(runtime)
    backend.suspend(2)
    backend.resume(["weights"])
    runtime.suspend_legacy.assert_called_once_with(2)
    runtime.resume_legacy.assert_called_once_with(["weights"])


def test_unsupported_capabilities_and_discard(backend):
    assert not backend.preserves_communicators()
    assert not backend.preserves_compiled_artifacts()
    assert not backend.preserves_graphs_with_communicators()
    assert not backend.supports_durable_storage()
    with pytest.raises(RuntimeError, match="selective discard"):
        backend.discard(("weights",))
