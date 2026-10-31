# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import signal
from unittest.mock import MagicMock

import pytest

from vllm_ascend.device_allocator import sleep_signals


@pytest.mark.parametrize("fail", [False, True])
def test_sdk_import_cannot_replace_callers_handlers(monkeypatch, fail):
    original = {number: object() for number in (signal.SIGTERM, signal.SIGINT)}
    handlers = original.copy()
    monkeypatch.setattr(signal, "getsignal", handlers.get)
    monkeypatch.setattr(signal, "signal", lambda number, handler: handlers.__setitem__(number, handler))
    try:
        with sleep_signals.preserve_worker_signal_handlers():
            handlers[signal.SIGTERM] = object()
            handlers[signal.SIGINT] = object()
            if fail:
                raise ValueError("SDK initialization failed")
    except ValueError:
        assert fail
    assert handlers == original


def test_unchanged_handlers_are_not_reinstalled(monkeypatch):
    handler = object()
    monkeypatch.setattr(signal, "getsignal", lambda number: handler)
    setter = MagicMock()
    monkeypatch.setattr(signal, "signal", setter)
    with sleep_signals.preserve_worker_signal_handlers():
        pass
    setter.assert_not_called()


def test_background_thread_does_not_modify_signal_handlers(monkeypatch):
    monkeypatch.setattr(sleep_signals.threading, "current_thread", lambda: object())
    getter = MagicMock()
    setter = MagicMock()
    monkeypatch.setattr(signal, "getsignal", getter)
    monkeypatch.setattr(signal, "signal", setter)
    with sleep_signals.preserve_worker_signal_handlers():
        pass
    getter.assert_not_called()
    setter.assert_not_called()
