# SPDX-License-Identifier: Apache-2.0
"""Upstream-drift guards for the worker-side watchdog patch.

``patch_worker_watchdog.py`` installs two patches at its module level
(worker patches are applied via ``adapt_patch()`` when each worker starts,
and by the test conftest at collection):

* ``WorkerProc.worker_busy_loop`` starts the per-worker watchdog
  (``worker_<rank>``) before entering the upstream loop, feeds it once per
  RPC, and disables it while blocked on the broadcast queue;
* ``WorkerProc.monitor_death_pipe`` replaces the upstream death-pipe monitor
  with a copy that additionally dumps the stacks (reason ``"shutdown"``)
  when the parent process exits.

Guarded here (everything reachable from CPU UT):

* both patches took effect at import;
* each patched copy differs from the pinned upstream source only by the
  intended watchdog lines (source-diffed against the pristine upstream,
  extracted via AST because the class attrs are already rebound at test
  time);
* the busy loop starts the watchdog with the rank-derived name, disables it
  around the blocking dequeue, feeds it per RPC, and executes the dequeued
  RPC;
* the death-pipe monitor dumps stacks and shuts the queues down on parent
  exit, swallows unexpected monitor errors, and is a no-op without a pipe;
* the upstream seams the copied death-pipe monitor depends on still exist
  (``WorkerProc.__init__`` initializes ``rpc_broadcast_mq`` /
  ``worker_response_mq`` via ``_init_message_queues``).

The watchdog implementation itself is covered by ``tests/ut/test_watch_dog.py``;
only the worker-side patch is guarded here.
"""

import ast
import inspect
import textwrap
import threading
from difflib import unified_diff
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import vllm.v1.executor.multiproc_executor as _multiproc_mod
from vllm.v1.executor.multiproc_executor import WorkerProc, logger

import vllm_ascend.patch.worker.patch_worker_watchdog as _worker_watchdog_patch


def _join_death_pipe_monitor(timeout: float = 5.0) -> None:
    for thread in threading.enumerate():
        if thread.name == "DeathPipeMonitor":
            thread.join(timeout=timeout)


def _upstream_method_source(name: str) -> str:
    """Pristine upstream method source, read from the module file via AST:
    the class attrs are already rebound to the patches at test time."""
    src = inspect.getsource(_multiproc_mod)
    tree = ast.parse(src)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "WorkerProc")
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


def _diff_delta(upstream_src: str, patched_src: str) -> list[str]:
    return [
        line.rstrip()
        for line in unified_diff(_source_lines(upstream_src), _source_lines(patched_src), n=0)
        if line[:1] in "+-" and line[:3] not in ("+++", "---")
    ]


# ---------------------------------------------------------------------------
# 1. patches installed at import
# ---------------------------------------------------------------------------


def test_worker_watchdog_patches_installed_at_import():
    """The patch module rebinds both methods at module level; the test
    conftest imports every worker patch module, so the rebind is live."""
    assert WorkerProc.worker_busy_loop is _worker_watchdog_patch._patched_worker_busy_loop
    assert WorkerProc.monitor_death_pipe is _worker_watchdog_patch._patched_monitor_death_pipe


# ---------------------------------------------------------------------------
# 2. upstream-copy drift guards
# ---------------------------------------------------------------------------

_EXPECTED_BUSY_LOOP_DELTA = [
    "-def worker_busy_loop(self):",
    '-    """Main busy loop for Multiprocessing Workers"""',
    "+def _patched_worker_busy_loop(self, *args, **kwargs):",
    "+    rank = kwargs.get(\"rank\", 0)",
    "+    start_watch_dog(f\"worker_{rank}\", logger)",
    '+    """Main busy loop for Multiprocessing Workers."""',
    "-        self._execute_worker_rpc(self.rpc_broadcast_mq.dequeue(indefinite=True))",
    "+        with get_watch_dog().disable():",
    "+            rpc_request = self.rpc_broadcast_mq.dequeue(indefinite=True)",
    "+        get_watch_dog().feed()",
    "+        self._execute_worker_rpc(rpc_request)",
    "+        del rpc_request",
]

_EXPECTED_DEATH_PIPE_DELTA = [
    "-def monitor_death_pipe(self, death_pipe, shutdown_requested: threading.Event):",
    "+def _patched_monitor_death_pipe(self, death_pipe, shutdown_requested):",
    "-    def death_pipe_monitor(queues_to_shutdown: list[MessageQueue]):",
    "+    def death_pipe_monitor(queues_to_shutdown):",
    "+            get_watch_dog().handle_failure(\"shutdown\")",
]


def test_patched_worker_busy_loop_is_upstream_copy_plus_watchdog():
    """The patched loop must stay a verbatim upstream copy whose only
    changes are the watchdog start/disable/feed lines. Any other divergence
    -- upstream or local -- fails here with the offending lines."""
    delta = _diff_delta(
        _upstream_method_source("worker_busy_loop"),
        inspect.getsource(_worker_watchdog_patch._patched_worker_busy_loop),
    )
    assert delta == _EXPECTED_BUSY_LOOP_DELTA, (
        "patched worker_busy_loop diverged from upstream beyond the "
        "intended watchdog lines:\n" + "\n".join(delta)
    )


def test_patched_monitor_death_pipe_is_upstream_copy_plus_dump():
    """The patched monitor must stay a verbatim upstream copy whose only
    change is the shutdown stack dump."""
    delta = _diff_delta(
        _upstream_method_source("monitor_death_pipe"),
        inspect.getsource(_worker_watchdog_patch._patched_monitor_death_pipe),
    )
    assert delta == _EXPECTED_DEATH_PIPE_DELTA, (
        "patched monitor_death_pipe diverged from upstream beyond the "
        "intended handle_failure() line:\n" + "\n".join(delta)
    )


def test_upstream_worker_proc_seams_for_patched_copies():
    """The patched copies depend on upstream seams that live outside the
    copied method bodies; guard them so a rename/removal upstream does not
    break the copies silently."""
    init_src = inspect.getsource(WorkerProc.__init__)
    assert "_init_message_queues" in init_src, (
        "upstream WorkerProc.__init__ no longer initializes message queues "
        "via _init_message_queues; the death-pipe monitor in "
        "patch_worker_watchdog.py would silently break."
    )
    queues_src = inspect.getsource(WorkerProc._init_message_queues)
    assert "rpc_broadcast_mq" in queues_src, (
        "upstream _init_message_queues no longer creates rpc_broadcast_mq; "
        "the death-pipe monitor in patch_worker_watchdog.py would silently break."
    )
    assert "worker_response_mq" in queues_src, (
        "upstream _init_message_queues no longer creates worker_response_mq; "
        "the death-pipe monitor in patch_worker_watchdog.py would silently break."
    )
    assert hasattr(logger, "info_once"), (
        "upstream executor logger lost info_once; the death-pipe monitor "
        "would silently break."
    )


# ---------------------------------------------------------------------------
# 3. the patched busy loop
# ---------------------------------------------------------------------------


def test_patched_worker_busy_loop_starts_watchdog_and_runs_rpcs(monkeypatch):
    """The patched loop starts the per-worker watchdog before entering the
    upstream loop, disables the watchdog while blocked on the broadcast
    queue, feeds it per RPC, and executes the dequeued RPC."""
    watchdog = SimpleNamespace(feed=MagicMock(), disable=MagicMock())
    monkeypatch.setattr(_worker_watchdog_patch, "get_watch_dog", lambda: watchdog)
    start_watch_dog = MagicMock()
    monkeypatch.setattr(_worker_watchdog_patch, "start_watch_dog", start_watch_dog)

    request = object()
    mq = MagicMock()
    mq.dequeue.side_effect = [request, StopIteration]
    worker = SimpleNamespace(rpc_broadcast_mq=mq, _execute_worker_rpc=MagicMock())

    with pytest.raises(StopIteration):
        _worker_watchdog_patch._patched_worker_busy_loop(worker, rank=2)

    start_watch_dog.assert_called_once_with("worker_2", logger)
    assert mq.dequeue.call_count == 2
    mq.dequeue.assert_called_with(indefinite=True)
    assert watchdog.disable.call_count == 2
    watchdog.feed.assert_called_once_with()
    worker._execute_worker_rpc.assert_called_once_with(request)


def test_patched_worker_busy_loop_defaults_rank_to_zero(monkeypatch):
    watchdog = SimpleNamespace(feed=MagicMock(), disable=MagicMock())
    monkeypatch.setattr(_worker_watchdog_patch, "get_watch_dog", lambda: watchdog)
    start_watch_dog = MagicMock()
    monkeypatch.setattr(_worker_watchdog_patch, "start_watch_dog", start_watch_dog)

    worker = SimpleNamespace(rpc_broadcast_mq=MagicMock(), _execute_worker_rpc=MagicMock())
    worker.rpc_broadcast_mq.dequeue.side_effect = StopIteration

    with pytest.raises(StopIteration):
        _worker_watchdog_patch._patched_worker_busy_loop(worker)

    start_watch_dog.assert_called_once_with("worker_0", logger)


def test_patched_worker_busy_loop_asserts_mq_after_watchdog_start(monkeypatch):
    """The watchdog starts before the upstream ``rpc_broadcast_mq`` assert,
    so even a worker whose queues failed to initialize gets monitored."""
    watchdog = SimpleNamespace(feed=MagicMock(), disable=MagicMock())
    monkeypatch.setattr(_worker_watchdog_patch, "get_watch_dog", lambda: watchdog)
    start_watch_dog = MagicMock()
    monkeypatch.setattr(_worker_watchdog_patch, "start_watch_dog", start_watch_dog)

    worker = SimpleNamespace(rpc_broadcast_mq=None, _execute_worker_rpc=MagicMock())

    with pytest.raises(AssertionError):
        _worker_watchdog_patch._patched_worker_busy_loop(worker)

    start_watch_dog.assert_called_once_with("worker_0", logger)


# ---------------------------------------------------------------------------
# 4. the patched death-pipe monitor
# ---------------------------------------------------------------------------


def test_patched_monitor_death_pipe_noop_without_pipe(monkeypatch):
    monkeypatch.setattr(_worker_watchdog_patch, "Thread", MagicMock())

    result = _worker_watchdog_patch._patched_monitor_death_pipe(object(), None, MagicMock())

    assert result is None
    _worker_watchdog_patch.Thread.assert_not_called()


def test_patched_monitor_death_pipe_dumps_stack_on_parent_exit(monkeypatch):
    watchdog = MagicMock()
    monkeypatch.setattr(_worker_watchdog_patch, "get_watch_dog", lambda: watchdog)

    death_pipe = MagicMock()
    death_pipe.recv.side_effect = EOFError
    shutdown_requested = MagicMock()
    rpc_mq = MagicMock()
    worker_mq = MagicMock()
    fake_self = SimpleNamespace(rpc_broadcast_mq=rpc_mq, worker_response_mq=worker_mq)

    _worker_watchdog_patch._patched_monitor_death_pipe(fake_self, death_pipe, shutdown_requested)

    _join_death_pipe_monitor()
    watchdog.handle_failure.assert_called_once_with("shutdown")
    shutdown_requested.set.assert_called_once_with()
    rpc_mq.shutdown.assert_called_once_with()
    worker_mq.shutdown.assert_called_once_with()


def test_patched_monitor_death_pipe_swallows_unexpected_errors(monkeypatch):
    watchdog = MagicMock()
    monkeypatch.setattr(_worker_watchdog_patch, "get_watch_dog", lambda: watchdog)

    death_pipe = MagicMock()
    death_pipe.recv.side_effect = RuntimeError("boom")
    shutdown_requested = MagicMock()
    rpc_mq = MagicMock()
    worker_mq = MagicMock()
    fake_self = SimpleNamespace(rpc_broadcast_mq=rpc_mq, worker_response_mq=worker_mq)

    _worker_watchdog_patch._patched_monitor_death_pipe(fake_self, death_pipe, shutdown_requested)

    _join_death_pipe_monitor()
    watchdog.handle_failure.assert_not_called()
    shutdown_requested.set.assert_not_called()
    rpc_mq.shutdown.assert_not_called()
    worker_mq.shutdown.assert_not_called()
