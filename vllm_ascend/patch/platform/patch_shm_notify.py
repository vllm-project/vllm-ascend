# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vllm-ascend project
"""Bound SHM notification mailbox growth during sustained fast-path reads."""

import time
from contextlib import suppress

from vllm.distributed.device_communicators.shm_broadcast import SpinCondition
from zmq.constants import Flag, PollEvent
from zmq.error import Again

# Match upstream vLLM PR #63165.
_NOTIFY_POLL_INTERVAL_S = 2.0


def record_read(self: SpinCondition) -> None:
    now = time.monotonic()
    self.last_read = now
    if now - getattr(self, "_ascend_last_notify_poll", 0.0) < _NOTIFY_POLL_INTERVAL_S:
        return

    # CONFLATE bounds payloads, not the internal activate_read command queue.
    # Service it on the owning reader thread even when SHM reads bypass wait().
    if self.local_notify_socket.poll(timeout=0, flags=PollEvent.POLLIN):
        with suppress(Again):
            self.local_notify_socket.recv(flags=Flag.DONTWAIT, copy=False)
    self._ascend_last_notify_poll = time.monotonic()


SpinCondition.record_read = record_read
