# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Compiler imports must not consume the engine's termination request."""

import signal
from contextlib import nullcontext
from unittest.mock import patch

import pytest

from vllm_ascend.worker.signal_utils import preserve_termination_handlers


@pytest.mark.parametrize("raises", [False, True])
def test_restores_engine_handlers_after_backend_import(raises):
    original = {signal.SIGTERM: object(), signal.SIGINT: object()}
    handlers = original.copy()
    with (
        patch("signal.getsignal", side_effect=handlers.__getitem__),
        patch("signal.signal", side_effect=handlers.__setitem__) as install,
    ):
        error_context = pytest.raises(RuntimeError) if raises else nullcontext()
        with error_context, preserve_termination_handlers():
            handlers.update({signal.SIGTERM: object(), signal.SIGINT: object()})
            if raises:
                raise RuntimeError("backend import failed")
        assert handlers == original
        assert install.call_count == 2


def test_preserves_default_and_ignored_handlers():
    handlers: dict[signal.Signals, object] = {signal.SIGTERM: signal.SIG_DFL, signal.SIGINT: signal.SIG_IGN}
    with (
        patch("signal.getsignal", side_effect=handlers.__getitem__),
        patch("signal.signal", side_effect=handlers.__setitem__),
        preserve_termination_handlers(),
    ):
        handlers[signal.SIGTERM] = object()
        handlers[signal.SIGINT] = object()
    assert handlers == {signal.SIGTERM: signal.SIG_DFL, signal.SIGINT: signal.SIG_IGN}


def test_leaves_unchanged_handlers_and_unknown_native_handlers_alone():
    handlers = {signal.SIGTERM: object(), signal.SIGINT: None}
    with patch("signal.getsignal", side_effect=handlers.__getitem__), patch("signal.signal") as install:
        with preserve_termination_handlers():
            handlers[signal.SIGINT] = object()
        install.assert_not_called()


def test_non_main_thread_does_not_touch_process_handlers():
    with (
        patch("threading.current_thread", return_value=object()),
        patch("signal.getsignal") as get,
        patch("signal.signal") as install,
    ):
        with preserve_termination_handlers():
            pass
        get.assert_not_called()
        install.assert_not_called()
