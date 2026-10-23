# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vllm-ascend project
"""Backport the SHM notification fix merged in upstream vLLM PR #63165."""

import time

from vllm.distributed.device_communicators.shm_broadcast import SpinCondition
from zmq.constants import Flag
from zmq.sugar.context import Context

SHM_NOTIFY_POLL_INTERVAL_S = 2.0
_original_init = SpinCondition.__init__


def init(
    self: SpinCondition,
    is_reader: bool,
    context: Context,
    notify_address: str,
    busy_loop_s: float = 1,
) -> None:
    _original_init(self, is_reader, context, notify_address, busy_loop_s)
    if is_reader:
        self.last_notify_poll = self.last_read


def record_read(self: SpinCondition) -> None:
    self.last_read = time.monotonic()
    if self.last_read - self.last_notify_poll >= SHM_NOTIFY_POLL_INTERVAL_S:
        self.last_notify_poll = self.last_read
        # poll() makes libzmq process the queued commands; a bare recv()
        # returns the pending message without doing so.
        if self.local_notify_socket.poll(timeout=0):
            self.local_notify_socket.recv(flags=Flag.DONTWAIT, copy=False)


SpinCondition.__init__ = init
SpinCondition.record_read = record_read
