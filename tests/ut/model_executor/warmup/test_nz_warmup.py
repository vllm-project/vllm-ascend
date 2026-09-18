# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Unit tests for ``nz_warmup``."""

from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm_ascend.model_executor.warmup import nz_warmup


@pytest.fixture(autouse=True)
def reset_state():
    nz_warmup._NZ_WARMED = False
    nz_warmup._NZ_THREAD_STARTED = False
    nz_warmup._NZ_THREAD = None
    yield
    nz_warmup._NZ_WARMED = False
    nz_warmup._NZ_THREAD_STARTED = False
    nz_warmup._NZ_THREAD = None


class _FakeThread:
    instances: list["_FakeThread"] = []

    def __init__(self, target=None, name=None, daemon=False):
        self.target = target
        self.name = name
        self.daemon = daemon
        _FakeThread.instances.append(self)

    def start(self):
        pass


def test_warm_nz_format_cast_runs_once():
    with patch.object(nz_warmup, "maybe_trans_nz") as mock_trans, patch.object(torch.npu, "synchronize"):
        nz_warmup.warm_nz_format_cast("test")
        first_calls = mock_trans.call_count
        nz_warmup.warm_nz_format_cast("test")

    assert mock_trans.call_count == first_calls
    assert nz_warmup._NZ_WARMED is True


def test_warm_nz_format_cast_swallows_failures():
    with patch.object(nz_warmup, "maybe_trans_nz", side_effect=RuntimeError("boom")):
        nz_warmup.warm_nz_format_cast("test")

    assert nz_warmup._NZ_WARMED is True


def test_thread_is_started_only_once():
    _FakeThread.instances = []
    with (
        patch.object(nz_warmup.threading, "Thread", _FakeThread),
        patch.object(torch.npu, "current_device", return_value=0),
    ):
        nz_warmup.start_nz_warm_thread("thread")
        nz_warmup.start_nz_warm_thread("thread")

    assert len(_FakeThread.instances) == 1
    assert _FakeThread.instances[0].daemon is True
    assert _FakeThread.instances[0].name == "coldstart-nz-warm"


def test_thread_is_skipped_when_already_warm():
    _FakeThread.instances = []
    nz_warmup._NZ_WARMED = True
    with patch.object(nz_warmup.threading, "Thread", _FakeThread):
        nz_warmup.start_nz_warm_thread("thread")

    assert _FakeThread.instances == []


def test_join_is_noop_without_thread():
    nz_warmup.join_nz_warm_thread()

    assert nz_warmup._NZ_THREAD is None


def test_join_waits_for_a_running_thread():
    thread = MagicMock()
    thread.is_alive.side_effect = [True, False]
    nz_warmup._NZ_THREAD = thread

    nz_warmup.join_nz_warm_thread()

    thread.join.assert_called_once_with(timeout=nz_warmup._NZ_JOIN_TIMEOUT_S)


def test_join_skips_a_finished_thread():
    thread = MagicMock()
    thread.is_alive.return_value = False
    nz_warmup._NZ_THREAD = thread

    nz_warmup.join_nz_warm_thread()

    thread.join.assert_not_called()
