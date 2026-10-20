# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import itertools
import os
import time
from unittest.mock import MagicMock, patch

import pytest

import vllm_ascend.watch_dog as watch_dog_module
from vllm_ascend.watch_dog import (
    NoopWatchDog,
    ThreadedWatchDog,
    get_watch_dog,
    start_watch_dog,
)


@pytest.fixture(autouse=True)
def _reset_global_watch_dog():
    original = watch_dog_module._watch_dog
    yield
    watch_dog_module._watch_dog = original


def _make_watch_dog(dump_dir, **kwargs) -> ThreadedWatchDog:
    # Overwrite attributes (e.g. timeout) that are normally configured via
    # module-level defaults so tests can exercise them directly.
    wd = ThreadedWatchDog("engine_0", str(dump_dir), MagicMock())
    for key, value in kwargs.items():
        setattr(wd, key, value)
    return wd


def _run_check_loop(wd: ThreadedWatchDog, ticks: int) -> int:
    """Run the check loop for ``ticks`` checks; return the dump count."""
    with (
        patch.object(wd.stop_event, "wait", side_effect=[False] * ticks + [True]),
        patch.object(wd, "handle_failure") as handle_failure,
    ):
        wd._check_loop()
    return handle_failure.call_count


def test_check_loop_dumps_once_per_stall(tmp_path):
    wd = _make_watch_dog(tmp_path)
    # Use a controlled clock instead of time.monotonic() offsets: monotonic()
    # measures host uptime and can be smaller than the offset on freshly
    # booted CI hosts, which would make last_feed_time negative and the
    # dump-once guard skip the dump.
    clock = itertools.count(1000)
    with patch("vllm_ascend.watch_dog.time.monotonic", side_effect=lambda: next(clock)):
        # Pretend the watchdog has not been fed for a while so a stall triggers.
        wd.last_feed_time = 100.0
        assert _run_check_loop(wd, ticks=3) == 1
        # A new feed resets the window; a later stall dumps anew.
        wd.feed()
        wd.last_timeout_log = 0.0
        wd.last_feed_time = 200.0
        assert _run_check_loop(wd, ticks=3) == 1


def test_check_loop_stays_quiet_while_fed(tmp_path):
    wd = _make_watch_dog(tmp_path)
    wd.feed()
    assert _run_check_loop(wd, ticks=3) == 0


def test_disable_suspends_detection_and_restarts_window(tmp_path):
    wd = _make_watch_dog(tmp_path, timeout=60)
    wd.last_feed_time = time.monotonic() - 3600
    with wd.disable():
        assert _run_check_loop(wd, ticks=3) == 0
    assert _run_check_loop(wd, ticks=3) == 0


def test_handle_failure_appends_numbered_dumps(tmp_path):
    wd = _make_watch_dog(tmp_path / "dumps")
    wd.handle_failure("feed timeout")
    wd.handle_failure("SIGTERM")

    with open(wd.dump_file) as f:
        content = f.read()
    assert "dump #1" in content and "due to feed timeout" in content
    assert "dump #2" in content and "due to SIGTERM" in content
    assert test_handle_failure_appends_numbered_dumps.__name__ in content


@pytest.mark.timeout(10)
def test_handle_failure_is_reentrant(tmp_path):
    # A second SIGTERM can re-enter handle_failure() while a dump holds the lock.
    wd = _make_watch_dog(tmp_path)
    with wd.lock:
        wd.handle_failure("SIGTERM")
    with open(wd.dump_file) as f:
        assert "due to SIGTERM" in f.read()


def test_handle_failure_does_not_raise_on_error(tmp_path):
    logger = MagicMock()
    wd = ThreadedWatchDog("engine_0", str(tmp_path), logger)
    with patch("builtins.open", side_effect=OSError("boom")):
        wd.handle_failure("feed timeout")
    logger.warning.assert_called_once()


def test_noop_watch_dog_does_nothing():
    wd = NoopWatchDog()
    wd.feed()
    with wd.disable():
        pass
    wd.handle_failure("SIGTERM")


def test_start_watch_dog_starts_threaded_watch_dog(tmp_path):
    logger = MagicMock()
    with (
        patch("vllm_ascend.watch_dog.get_ascend_log_dir", return_value=str(tmp_path)),
        patch("faulthandler.enable") as enable,
    ):
        wd = start_watch_dog("worker_1", logger)
    assert isinstance(wd, ThreadedWatchDog)
    try:
        assert wd is get_watch_dog()
        enable.assert_called_once_with(all_threads=True)
        assert wd.thread is not None
        assert wd.thread.is_alive() and wd.thread.daemon
        wd.handle_failure("shutdown")
        assert os.path.exists(tmp_path / f"VLLM_STACK_DUMP_for_worker_1_{os.getpid()}.log")
    finally:
        wd.stop()
