# SPDX-License-Identifier: Apache-2.0
"""Bounded shutdown for worker-owned QoS engines.

A timeout leaves registrations alive. The caller must fail the worker; it must
not recycle buffers whose transfer completion is unknown.
"""

import time

QOS_SHUTDOWN_TIMEOUT = 60.0


def drain_queue(thread, work_queue, timeout=QOS_SHUTDOWN_TIMEOUT):
    deadline = time.monotonic() + timeout
    with work_queue.all_tasks_done:
        while work_queue.unfinished_tasks:
            if not thread.is_alive():
                raise RuntimeError("QoS transfer thread exited with unfinished transfers")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("QoS transfers did not finish; buffers remain registered")
            work_queue.all_tasks_done.wait(min(remaining, 1.0))


def stop_queue(thread, work_queue, timeout=QOS_SHUTDOWN_TIMEOUT):
    if thread is None:
        return
    drain_queue(thread, work_queue, timeout)
    if thread.is_alive():
        work_queue.put(None)
        thread.join(timeout)
        if thread.is_alive():
            raise TimeoutError("QoS transfer thread did not stop; buffers remain registered")
    executor = getattr(thread, "executor", None)
    if executor is not None:
        executor.shutdown(wait=True)


def stop_listener(thread, timeout=QOS_SHUTDOWN_TIMEOUT):
    if thread is None:
        return
    thread.qos_stop_event.set()
    thread.join(timeout)
    if thread.is_alive():
        raise TimeoutError("QoS metadata listener did not stop; buffers remain registered")


def wait_remote_reads(sender, timeout=QOS_SHUTDOWN_TIMEOUT):
    if sender is None:
        return
    deadline = time.monotonic() + timeout
    tracker = sender.task_tracker
    while True:
        with tracker.done_task_lock:
            if not tracker.reqs_to_process:
                return
        if not sender.is_alive():
            raise RuntimeError("QoS peer listener died before remote READ completion")
        if time.monotonic() >= deadline:
            raise TimeoutError("remote READ completion unknown; buffers remain registered")
        sender.qos_stop_event.wait(0.05)


def wait_remote_writes(receiver, timeout=QOS_SHUTDOWN_TIMEOUT):
    """A D worker must retain its registrations until all P writes complete."""
    if receiver is None:
        return
    deadline = time.monotonic() + timeout
    while True:
        with receiver.lock:
            if receiver.qos_receive_error:
                raise RuntimeError("remote WRITE failed; completion unknown, buffers remain registered")
            if not receiver.qos_pending_requests:
                return
        if not receiver.is_alive():
            raise RuntimeError("QoS peer listener died before remote WRITE completion")
        if time.monotonic() >= deadline:
            raise TimeoutError("remote WRITE completion unknown; buffers remain registered")
        receiver.qos_stop_event.wait(0.05)
