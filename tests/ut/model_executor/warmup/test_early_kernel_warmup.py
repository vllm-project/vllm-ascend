# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Unit tests for ``early_kernel_warmup``."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from vllm_ascend.model_executor.warmup import early_kernel_warmup as ek


class _FakeThread:
    """Runs the target inline so the test stays deterministic."""

    instances: list["_FakeThread"] = []

    def __init__(self, target=None, name=None, daemon=False):
        self.target = target
        self.name = name
        self.daemon = daemon
        self.started = False
        self.joined_with = None
        _FakeThread.instances.append(self)

    def start(self):
        self.started = True
        self.target()

    def is_alive(self):
        return False

    def join(self, timeout=None):
        self.joined_with = timeout


class _RunningThread:
    def __init__(self, alive_after_join):
        self.alive_after_join = alive_after_join
        self.joined_with = None

    def is_alive(self):
        return self.joined_with is None or self.alive_after_join

    def join(self, timeout=None):
        self.joined_with = timeout


def _config(enabled):
    warmup = SimpleNamespace(enable_early_kernel_warmup=enabled)
    return patch.object(ek, "get_ascend_config", return_value=SimpleNamespace(ascend_warmup_config=warmup))


@pytest.fixture(autouse=True)
def reset_state():
    _FakeThread.instances = []
    ek._STATE.update({"started": False, "thread": None, "done": False})
    yield
    ek._STATE.update({"started": False, "thread": None, "done": False})


def test_config_switches_default_to_off():
    from vllm_ascend.ascend_config import AscendWarmupConfig

    config = AscendWarmupConfig()
    assert config.enable_early_kernel_warmup is False
    assert config.enable_early_nz_warmup is False


def test_uninitialized_config_counts_as_disabled():
    with patch.object(ek, "get_ascend_config", side_effect=RuntimeError("not initialized")):
        assert ek._enabled() is False


def test_start_is_noop_when_disabled():
    with _config(False), patch.object(ek, "_preimport") as preimport:
        ek.start_early_kernel_warmup()

    preimport.assert_not_called()
    assert ek._STATE["started"] is False
    assert ek._STATE["thread"] is None


def test_start_without_vllm_config_does_not_latch():
    """Construct can run before the config context; a later call must still start."""
    with (
        _config(True),
        patch("vllm.config.get_current_vllm_config", return_value=None),
        patch.object(ek, "_preimport") as preimport,
    ):
        ek.start_early_kernel_warmup()

    preimport.assert_not_called()
    assert ek._STATE["started"] is False
    assert ek._STATE["thread"] is None

    with (
        _config(True),
        patch("vllm.config.get_current_vllm_config", return_value=MagicMock()),
        patch.object(ek.torch.npu, "current_device", return_value=0),
        patch.object(ek, "_preimport"),
        patch.object(ek, "_run"),
        patch.object(ek.threading, "Thread", _FakeThread),
    ):
        ek.start_early_kernel_warmup()

    assert ek._STATE["started"] is True
    assert ek._STATE["thread"] is not None


def test_start_runs_warmup_on_a_daemon_thread():
    vllm_config = MagicMock()

    with (
        _config(True),
        patch("vllm.config.get_current_vllm_config", return_value=vllm_config),
        patch.object(ek.torch.npu, "current_device", return_value=0),
        patch.object(ek, "_preimport") as preimport,
        patch.object(ek, "_run") as run,
        patch.object(ek.threading, "Thread", _FakeThread),
    ):
        ek.start_early_kernel_warmup()

    preimport.assert_called_once()
    run.assert_called_once()
    shim, device_index = run.call_args[0]
    assert shim.vllm_config is vllm_config
    assert device_index == 0

    thread = _FakeThread.instances[0]
    assert thread.daemon is True
    assert thread.name == "coldstart-early-kernel-warmup"
    assert ek._STATE["done"] is True


def test_start_is_idempotent():
    with (
        _config(True),
        patch("vllm.config.get_current_vllm_config", return_value=MagicMock()),
        patch.object(ek.torch.npu, "current_device", return_value=0),
        patch.object(ek, "_preimport"),
        patch.object(ek, "_run"),
        patch.object(ek.threading, "Thread", _FakeThread),
    ):
        ek.start_early_kernel_warmup()
        ek.start_early_kernel_warmup()

    assert len(_FakeThread.instances) == 1


def test_start_never_raises():
    with (
        _config(True),
        patch("vllm.config.get_current_vllm_config", side_effect=RuntimeError("boom")),
        patch.object(ek, "_preimport"),
    ):
        ek.start_early_kernel_warmup()

    assert ek._STATE["thread"] is None


def test_join_is_noop_without_thread():
    ek.join_early_kernel_warmup("rms")

    assert ek._STATE["thread"] is None


def test_join_skips_a_finished_thread():
    thread = _FakeThread(target=lambda: None, name="t")
    ek._STATE["thread"] = thread

    ek.join_early_kernel_warmup("rejection_sampler")

    assert thread.joined_with is None


def test_join_waits_for_a_running_thread():
    thread = _RunningThread(alive_after_join=False)
    ek._STATE["thread"] = thread

    with patch.object(ek.logger, "warning") as warning:
        ek.join_early_kernel_warmup("load_model")

    assert thread.joined_with == ek._JOIN_TIMEOUT_S
    warning.assert_not_called()


def test_join_warns_when_the_thread_outlives_the_timeout():
    ek._STATE["thread"] = _RunningThread(alive_after_join=True)

    with patch.object(ek.logger, "warning") as warning:
        ek.join_early_kernel_warmup("load_model")

    warning.assert_called_once()
