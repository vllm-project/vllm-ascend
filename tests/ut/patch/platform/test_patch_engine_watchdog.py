# SPDX-License-Identifier: Apache-2.0
"""Guards for the engine-core watchdog patch.

``patch_engine_watchdog.py`` no longer self-patches at import. Everything is
installed by ``patch_watchdog_for_engine_core()``, which the consolidated
``patch_engine_core._run_engine_core_patch_func`` calls at engine-core entry
(and therefore runs in the engine-core child process). What is guarded here
(everything reachable from CPU UT):

* ``patch_watchdog_for_engine_core`` installs the ``_process_engine_step`` /
  ``_process_input_queue`` feed patches, the ``SignalCallback.trigger``
  dump-on-signal patch, and configures + starts the watchdog;
* ``patched_signal_callback`` dumps the stacks with reason ``"signal"`` and
  delegates to the pristine upstream ``trigger``;
* ``_patched_process_engine_step`` feeds the watchdog and delegates to the
  pristine upstream ``_process_engine_step``;
* ``_patched_process_input_queue`` feeds the watchdog when the input-queue
  wait times out (the idle-is-alive contract) and skips the feed when work is
  already pending.
"""

import queue
from types import SimpleNamespace
from unittest.mock import MagicMock

from vllm.v1.engine.core import EngineCoreProc
from vllm.v1.engine.utils import SignalCallback

import vllm_ascend.patch.platform.patch_engine_watchdog as _watchdog_patch


def test_patch_watchdog_for_engine_core_installs_all_patches(monkeypatch):
    # Register restore of everything the installer rebinds.
    monkeypatch.setattr(SignalCallback, "trigger", SignalCallback.trigger)
    monkeypatch.setattr(EngineCoreProc, "_process_engine_step", EngineCoreProc._process_engine_step)
    monkeypatch.setattr(EngineCoreProc, "_process_input_queue", EngineCoreProc._process_input_queue)
    watchdog = MagicMock()
    monkeypatch.setattr(_watchdog_patch, "_watchdog", watchdog)

    _watchdog_patch.patch_watchdog_for_engine_core()

    assert SignalCallback.trigger is _watchdog_patch.patched_signal_callback
    assert EngineCoreProc._process_engine_step is _watchdog_patch._patched_process_engine_step
    assert EngineCoreProc._process_input_queue is _watchdog_patch._patched_process_input_queue
    watchdog.setup.assert_called_once_with("engine")
    watchdog.start.assert_called_once_with()


def test_patched_signal_callback_dumps_stack_with_signal_reason(monkeypatch):
    watchdog = MagicMock()
    monkeypatch.setattr(_watchdog_patch, "_watchdog", watchdog)
    original = MagicMock()
    monkeypatch.setattr(_watchdog_patch, "_original_signal_callback", original)

    fake_self = object()
    result = _watchdog_patch.patched_signal_callback(fake_self)

    watchdog.dump_stack.assert_called_once_with("signal")
    original.assert_called_once_with(fake_self)
    assert result is original.return_value


def test_patched_process_engine_step_feeds_and_delegates(monkeypatch):
    watchdog = MagicMock()
    monkeypatch.setattr(_watchdog_patch, "_watchdog", watchdog)
    original = MagicMock(return_value=True)
    monkeypatch.setattr(_watchdog_patch, "_original_process_engine_step", original)

    fake_self = object()
    result = _watchdog_patch._patched_process_engine_step(fake_self)

    watchdog.feed.assert_called_once_with()
    original.assert_called_once_with(fake_self)
    assert result is True


def test_patched_process_input_queue_feeds_watchdog_when_waiting(monkeypatch):
    """The patched copy replaces the upstream blocking get with a 5 s
    timeout get and feeds the watchdog on each queue.Empty -- an idle
    engine waiting for work must count as alive."""
    watchdog = MagicMock()
    monkeypatch.setattr(_watchdog_patch, "_watchdog", watchdog)

    input_queue = MagicMock()
    # Queue stays empty: both the wait loop and the trailing drain loop see
    # an empty queue, so no client request is ever handled.
    input_queue.empty.side_effect = [True, True]
    input_queue.get.side_effect = queue.Empty

    engine = SimpleNamespace(
        has_work=MagicMock(return_value=False),
        is_running=MagicMock(side_effect=[True, False]),
        input_queue=input_queue,
        aborts_queue=SimpleNamespace(mutex=MagicMock(), queue=[]),
        process_input_queue_block=True,
        _handle_client_request=MagicMock(),
        _notify_idle_state_callbacks=MagicMock(),
    )

    _watchdog_patch._patched_process_input_queue(engine)

    watchdog.feed.assert_called_once_with()
    input_queue.get.assert_called_once_with(block=True, timeout=5)
    engine._handle_client_request.assert_not_called()


def test_patched_process_input_queue_skips_feed_when_work_pending(monkeypatch):
    """When work is already pending the wait loop (and its watchdog feed) is
    skipped; the pending request is handled straight from the drain loop."""
    watchdog = MagicMock()
    monkeypatch.setattr(_watchdog_patch, "_watchdog", watchdog)

    input_queue = MagicMock()
    # One pending request: the drain loop enters once, then sees the queue
    # empty and exits.
    input_queue.empty.side_effect = [False, True]
    input_queue.get_nowait.return_value = ("add_request", "request")

    handle_client_request = MagicMock()
    engine = SimpleNamespace(
        has_work=MagicMock(return_value=True),
        is_running=MagicMock(return_value=True),
        input_queue=input_queue,
        _handle_client_request=handle_client_request,
    )

    _watchdog_patch._patched_process_input_queue(engine)

    watchdog.feed.assert_not_called()
    input_queue.get.assert_not_called()
    handle_client_request.assert_called_once_with("add_request", "request")
