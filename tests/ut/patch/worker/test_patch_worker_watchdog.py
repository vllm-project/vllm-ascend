# SPDX-License-Identifier: Apache-2.0
"""Guards for the worker-side watchdog patch.

``patch_worker_watchdog.py`` installs three patches at its module level
(worker patches are applied via ``adapt_patch(is_global_patch=False)`` when
each worker starts):

* ``WorkerProc.worker_busy_loop`` configures and starts the watchdog before
  entering the upstream loop;
* ``WorkerProc.monitor_death_pipe`` replaces the upstream death-pipe monitor
  with a copy that additionally dumps the stacks (reason ``"shutdown"``) when
  the parent process exits;
* ``shm_broadcast.MessageQueue.acquire_read`` feeds the watchdog while
  blocked waiting for a written block.

Guarded here (everything reachable from CPU UT):

* all three patches took effect at import;
* the death-pipe monitor dumps stacks and shuts the queues down on parent
  exit, and is a no-op without a death pipe;
* the upstream seams the copied death-pipe monitor depends on still exist
  (``WorkerProc.__init__`` initializes ``rpc_broadcast_mq`` /
  ``worker_response_mq`` via ``_init_message_queues``).
"""

import inspect
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

from vllm.distributed.device_communicators import shm_broadcast
from vllm.v1.executor.multiproc_executor import WorkerProc, logger

import vllm_ascend.patch.worker.patch_worker_watchdog as _worker_watchdog_patch


def _join_death_pipe_monitor(timeout: float = 5.0) -> None:
    for thread in threading.enumerate():
        if thread.name == "DeathPipeMonitor":
            thread.join(timeout=timeout)


def test_worker_watchdog_patches_installed_at_import():
    assert WorkerProc.worker_busy_loop is _worker_watchdog_patch._patched_worker_busy_loop
    assert WorkerProc.monitor_death_pipe is _worker_watchdog_patch._patched_monitor_death_pipe
    assert shm_broadcast.MessageQueue.acquire_read is _worker_watchdog_patch._patched_queue_acquire_read


def test_patched_worker_busy_loop_sets_up_and_delegates(monkeypatch):
    watchdog = MagicMock()
    monkeypatch.setattr(_worker_watchdog_patch, "_watchdog", watchdog)
    original = MagicMock(return_value="done")
    monkeypatch.setattr(_worker_watchdog_patch, "_original_worker_busy_loop", original)

    result = _worker_watchdog_patch._patched_worker_busy_loop("arg", key="val")

    watchdog.setup.assert_called_once_with("worker")
    watchdog.start.assert_called_once_with()
    original.assert_called_once_with("arg", key="val")
    assert result == "done"


def test_monitor_death_pipe_noop_without_pipe():
    result = _worker_watchdog_patch._patched_monitor_death_pipe(object(), None, MagicMock())

    assert result is None
    assert "DeathPipeMonitor" not in {t.name for t in threading.enumerate()}


def test_monitor_death_pipe_dumps_stack_on_parent_exit(monkeypatch):
    watchdog = MagicMock()
    monkeypatch.setattr(_worker_watchdog_patch, "_watchdog", watchdog)

    death_pipe = MagicMock()
    death_pipe.recv.side_effect = EOFError
    shutdown_requested = MagicMock()
    rpc_mq = MagicMock()
    worker_mq = MagicMock()
    fake_self = SimpleNamespace(rpc_broadcast_mq=rpc_mq, worker_response_mq=worker_mq)

    _worker_watchdog_patch._patched_monitor_death_pipe(fake_self, death_pipe, shutdown_requested)

    _join_death_pipe_monitor()
    watchdog.dump_stack.assert_called_once_with("shutdown")
    shutdown_requested.set.assert_called_once_with()
    rpc_mq.shutdown.assert_called_once_with()
    worker_mq.shutdown.assert_called_once_with()


def test_upstream_worker_proc_seams_for_death_pipe_monitor():
    """The patched death-pipe monitor is a copy of upstream's flow; guard the
    upstream seams it depends on so a rename/removal upstream does not break
    it silently."""
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
        "upstream executor logger lost info_once; the death-pipe monitor would silently break."
    )
