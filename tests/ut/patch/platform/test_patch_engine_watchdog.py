# SPDX-License-Identifier: Apache-2.0
"""Upstream-drift guards for the engine-core watchdog patch.

``patch_engine_watchdog.py`` keeps the process-wide watchdog alive from the
engine-core loop: ``_process_engine_step`` feeds the watchdog once per step,
and ``_process_input_queue`` -- a verbatim copy of the upstream method --
disables the watchdog while the loop blocks on idle input so an idle engine
never trips the feed timeout. What is guarded here (everything reachable
from CPU UT):

* ``_original_process_engine_step`` stashes the pristine upstream method at
  import, before ``patch_watchdog_for_engine_core`` rebinds the class attr;
* ``_patched_process_engine_step`` feeds the watchdog and delegates to that
  pristine original;
* ``_patched_process_input_queue`` differs from the pinned upstream source
  in exactly three places -- the def name, the ``logging.DEBUG`` spelling,
  and the ``with get_watch_dog().disable():`` wrap of the blocking
  ``input_queue.get``; the source-diff guard fails loudly on any other
  upstream drift (the pristine upstream is read via AST from the vllm
  module file, since the class attr may already be rebound in CI);
* ``patch_watchdog_for_engine_core`` installs both patches and starts the
  watchdog named ``engine_<dp_rank>`` through the vllm engine-core logger.

The watchdog implementation itself is covered by ``tests/ut/test_watch_dog.py``;
only the engine-core patch is guarded here.
"""

import ast
import inspect
import queue
import textwrap
from difflib import unified_diff
from types import SimpleNamespace
from unittest.mock import MagicMock

import vllm.v1.engine.core as _engine_core_mod
from vllm.v1.engine.core import EngineCoreProc

import vllm_ascend.patch.platform.patch_engine_watchdog as _watchdog_patch


def _fake_engine(*, has_work, is_running, input_queue, block=True):
    """Assemble the engine-core state the patched method touches."""
    return SimpleNamespace(
        has_work=has_work,
        is_running=is_running,
        input_queue=input_queue,
        aborts_queue=queue.Queue(),
        process_input_queue_block=block,
        _handle_client_request=MagicMock(),
        _notify_idle_state_callbacks=MagicMock(),
    )


# ---------------------------------------------------------------------------
# 1. import-time stash + upstream-copy drift guards
# ---------------------------------------------------------------------------


def test_original_process_engine_step_is_pristine_upstream():
    """Stashed at import, before ``patch_watchdog_for_engine_core`` rebinds
    the class attribute, so it must be the genuine upstream method."""
    assert _watchdog_patch._original_process_engine_step is not _watchdog_patch._patched_process_engine_step
    assert _watchdog_patch._original_process_engine_step.__qualname__ == "EngineCoreProc._process_engine_step"
    src = inspect.getsource(_watchdog_patch._original_process_engine_step)
    assert "self.step_fn()" in src, (
        "upstream _process_engine_step no longer steps the engine via "
        "self.step_fn(); the feed patch would silently stop guarding it."
    )


_EXPECTED_INPUT_QUEUE_DELTA = [
    "-def _process_input_queue(self):",
    "+def _patched_process_input_queue(self):",
    "-            if logger.isEnabledFor(DEBUG):",
    "+            if logger.isEnabledFor(logging.DEBUG):",
    "-            req = self.input_queue.get(block=block)",
    "+            with get_watch_dog().disable():",
    "+                req = self.input_queue.get(block=block)",
]


def _upstream_method_source(name: str) -> str:
    """Pristine upstream method source, read from the vllm module file via
    AST: in CI the platform plugin runs ``patch_watchdog_for_engine_core``
    at pytest startup (before this test module is imported), so
    ``EngineCoreProc``'s class attrs may already be rebound to the patched
    functions and the pristine method must not be read from them."""
    src = inspect.getsource(_engine_core_mod)
    tree = ast.parse(src)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "EngineCoreProc")
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)
    # ast.get_source_segment strips the leading indent of the first line
    # only; slice the method's column offset off every line so the method
    # reads as a uniformly indented top-level function (matching the
    # patched module-level copy).
    lines = src.splitlines()
    return "\n".join(line[fn.col_offset :] for line in lines[fn.lineno - 1 : fn.end_lineno])


def _source_lines(src: str) -> list[str]:
    """Normalized source lines: dedented, blank lines dropped."""
    return [line for line in textwrap.dedent(src).splitlines() if line.strip()]


def test_patched_process_input_queue_is_upstream_copy_plus_disable():
    """The patched method must stay a verbatim upstream copy whose only
    behavioral change is disabling the watchdog around the blocking
    ``input_queue.get``. Any other divergence -- upstream or local -- fails
    here with the offending lines."""
    delta = [
        line.rstrip()
        for line in unified_diff(
            _source_lines(_upstream_method_source("_process_input_queue")),
            _source_lines(inspect.getsource(_watchdog_patch._patched_process_input_queue)),
            n=0,
        )
        if line[:1] in "+-" and line[:3] not in ("+++", "---")
    ]
    assert delta == _EXPECTED_INPUT_QUEUE_DELTA, (
        "patched _process_input_queue diverged from upstream beyond the intended disable() wrap:\n" + "\n".join(delta)
    )


# ---------------------------------------------------------------------------
# 2. the patched wrappers
# ---------------------------------------------------------------------------


def test_patched_process_engine_step_feeds_and_delegates(monkeypatch):
    watchdog = SimpleNamespace(feed=MagicMock())
    monkeypatch.setattr(_watchdog_patch, "get_watch_dog", lambda: watchdog)
    original = MagicMock(return_value=True)
    monkeypatch.setattr(_watchdog_patch, "_original_process_engine_step", original)

    fake_self = object()
    result = _watchdog_patch._patched_process_engine_step(fake_self)

    watchdog.feed.assert_called_once_with()
    original.assert_called_once_with(fake_self)
    assert result is True


def test_patched_process_input_queue_disables_watchdog_while_blocking(monkeypatch):
    """An idle engine blocked on input must not trip the feed timeout: the
    blocking get runs with the watchdog disabled, and no feed happens here
    (feeding is ``_process_engine_step``'s job)."""
    watchdog = SimpleNamespace(feed=MagicMock(), disable=MagicMock())
    monkeypatch.setattr(_watchdog_patch, "get_watch_dog", lambda: watchdog)

    input_queue = MagicMock()
    input_queue.empty.side_effect = [True, True]
    input_queue.get.side_effect = queue.Empty

    engine = _fake_engine(
        has_work=MagicMock(return_value=False),
        is_running=MagicMock(side_effect=[True, False]),
        input_queue=input_queue,
    )
    engine.aborts_queue.put("abort-1")
    engine.aborts_queue.put("abort-2")

    _watchdog_patch._patched_process_input_queue(engine)

    watchdog.feed.assert_not_called()
    watchdog.disable.assert_called_once_with()
    watchdog.disable.return_value.__enter__.assert_called_once_with()
    input_queue.get.assert_called_once_with(block=True)
    engine._notify_idle_state_callbacks.assert_called_once_with()
    engine._handle_client_request.assert_not_called()
    assert engine.aborts_queue.empty()


def test_patched_process_input_queue_drains_requests_after_wait(monkeypatch):
    """Requests arriving while blocked are handled by the trailing
    'Handle any more client requests' pass after the wait loop exits."""
    watchdog = SimpleNamespace(disable=MagicMock())
    monkeypatch.setattr(_watchdog_patch, "get_watch_dog", lambda: watchdog)

    input_queue = MagicMock()
    input_queue.empty.side_effect = [True, False, True]
    input_queue.get.side_effect = queue.Empty
    request = ("add_request", "request")
    input_queue.get_nowait.return_value = request

    engine = _fake_engine(
        has_work=MagicMock(return_value=False),
        is_running=MagicMock(side_effect=[True, False]),
        input_queue=input_queue,
    )

    _watchdog_patch._patched_process_input_queue(engine)

    watchdog.disable.assert_called_once_with()
    engine._handle_client_request.assert_called_once_with(*request)


def test_patched_process_input_queue_skips_wait_when_work_pending(monkeypatch):
    """With work already pending, the wait loop (and the watchdog disable)
    is skipped entirely; the request is handled straight from the drain."""
    watchdog = SimpleNamespace(disable=MagicMock())
    monkeypatch.setattr(_watchdog_patch, "get_watch_dog", lambda: watchdog)

    input_queue = MagicMock()
    input_queue.empty.side_effect = [False, True]
    request = ("add_request", "request")
    input_queue.get_nowait.return_value = request

    engine = _fake_engine(
        has_work=MagicMock(return_value=True),
        is_running=MagicMock(return_value=True),
        input_queue=input_queue,
    )

    _watchdog_patch._patched_process_input_queue(engine)

    watchdog.disable.assert_not_called()
    input_queue.get.assert_not_called()
    engine._handle_client_request.assert_called_once_with(*request)


def test_patched_process_input_queue_nonblocking_get(monkeypatch):
    """``process_input_queue_block=False``: one non-blocking get is
    attempted, the request is handled, and the wait loop exits at once."""
    watchdog = SimpleNamespace(disable=MagicMock())
    monkeypatch.setattr(_watchdog_patch, "get_watch_dog", lambda: watchdog)

    input_queue = MagicMock()
    input_queue.empty.side_effect = [False, True]
    request = ("add_request", "request")
    input_queue.get.return_value = request

    engine = _fake_engine(
        has_work=MagicMock(return_value=False),
        is_running=MagicMock(return_value=True),
        input_queue=input_queue,
        block=False,
    )

    _watchdog_patch._patched_process_input_queue(engine)

    watchdog.disable.assert_called_once_with()
    input_queue.get.assert_called_once_with(block=False)
    engine._handle_client_request.assert_called_once_with(*request)


# ---------------------------------------------------------------------------
# 3. the installer
# ---------------------------------------------------------------------------


def test_patch_watchdog_for_engine_core_installs_and_starts(monkeypatch):
    # Register restore of everything the installer rebinds.
    monkeypatch.setattr(EngineCoreProc, "_process_engine_step", EngineCoreProc._process_engine_step)
    monkeypatch.setattr(EngineCoreProc, "_process_input_queue", EngineCoreProc._process_input_queue)
    start_watch_dog = MagicMock()
    monkeypatch.setattr(_watchdog_patch, "start_watch_dog", start_watch_dog)

    _watchdog_patch.patch_watchdog_for_engine_core(dp_rank=3)

    assert EngineCoreProc._process_engine_step is _watchdog_patch._patched_process_engine_step
    assert EngineCoreProc._process_input_queue is _watchdog_patch._patched_process_input_queue
    start_watch_dog.assert_called_once_with("engine_3", _engine_core_mod.logger)
