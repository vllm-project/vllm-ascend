# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vllm-ascend project
"""Bound SHM notification mailbox growth during sustained fast-path reads."""

import time
from contextlib import suppress

import zmq
from vllm.distributed.device_communicators.shm_broadcast import SpinCondition

_NOTIFY_POLL_INTERVAL_S = 1.0


def record_read(self: SpinCondition) -> None:
    now = time.monotonic()
    self.last_read = now
    if now - getattr(self, "_ascend_last_notify_poll", 0.0) < _NOTIFY_POLL_INTERVAL_S:
        return

    # CONFLATE bounds payloads, not the internal activate_read command queue.
    # Service it on the owning reader thread even when SHM reads bypass wait().
    if self.local_notify_socket.poll(timeout=0, flags=zmq.POLLIN):
        with suppress(zmq.Again):
            self.local_notify_socket.recv(flags=zmq.NOBLOCK, copy=False)
    self._ascend_last_notify_poll = time.monotonic()


SpinCondition.record_read = record_read
