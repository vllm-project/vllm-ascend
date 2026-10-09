# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import time
from contextlib import contextmanager

from vllm.distributed.device_communicators import shm_broadcast

from vllm_ascend.common.utils.watch_dog import get_watch_dog

_watchdog = get_watch_dog()

MessageQueue = shm_broadcast.MessageQueue

# Cap on how long an idle reader parks before re-reading the authoritative SHM
# written-flag. Bounds lost-notify recovery latency to ~5s while the periodic
# wakeup stays negligible (one flag check per reader every 5s).
SHM_READER_RECHECK_INTERVAL_MS = 5000


def timeout_ms(self) -> int:
    """Returns a timeout, capped at the recheck interval, that is:
    - min(time to deadline, time to next warning) if we're logging warnings
    - time to deadline, if we're not logging warnings
    - recheck interval if the timeout is None and we're not logging warnings
    - raise TimeoutError if we are past the deadline
    """
    wait_ms = SHM_READER_RECHECK_INTERVAL_MS
    if self.warning_wait_time_ms is not None:
        wait_ms = min(wait_ms, self.warning_wait_time_ms)
    if self.timeout is None:
        return wait_ms
    time_left_ms = int((self.deadline - time.monotonic()) * 1000)
    if time_left_ms <= 0:
        raise TimeoutError
    return min(wait_ms, time_left_ms)


@contextmanager
def acquire_read(
    self,
    timeout: float | None = None,
    indefinite: bool = False,
):
    assert self._is_local_reader, "Only readers can acquire read"
    read_timeout = self.ReadTimeoutWithWarnings(timeout=timeout, should_warn=not indefinite)
    with self.buffer.get_metadata(self.current_idx) as metadata_buffer:
        while True:

            def check():
                shm_broadcast.memory_fence()
                read_flag = metadata_buffer[self.local_reader_rank + 1]
                written_flag = metadata_buffer[0]
                return not (not written_flag or read_flag)

            if shm_broadcast.SPINLOOP_EXT_ENABLED and not check():
                shm_broadcast.spinloop(
                    metadata_buffer[0 : self.local_reader_rank + 1],
                    check,
                    timeout=shm_broadcast.SPINLOOP_TIMEOUT_SECONDS,
                )

            if not check():
                # this block is either
                # (1) not written
                # (2) already read by this reader

                # for readers, `self.current_idx` is the next block to read
                # if this block is not ready,
                # we need to wait until it is written
                self._spin_condition.wait(timeout_ms=read_timeout.timeout_ms())
                _watchdog.feed()

                if self.shutting_down:
                    raise RuntimeError("cancelled")

                # if we wait for a long time, log a message
                if read_timeout.should_warn():
                    shm_broadcast.logger.info(
                        shm_broadcast.LONG_WAIT_TIME_LOG_MSG,
                        shm_broadcast.VLLM_RINGBUFFER_WARNING_INTERVAL,
                    )

                continue

            # found a block that is not read by this reader
            # let caller read from the buffer
            with self.buffer.get_data(self.current_idx) as buf:
                try:
                    yield buf
                finally:
                    # caller has read from the buffer; set the read flag.
                    metadata_buffer[self.local_reader_rank + 1] = 1
                    # Memory fence ensures the read flag is visible to the writer.
                    # Without this, writer may not see our read completion and
                    # could wait indefinitely for all readers to finish.
                    shm_broadcast.memory_fence()
                    next_idx = self.current_idx + 1
                    self.current_idx = next_idx % self.buffer.max_chunks
                    self._spin_condition.record_read()
            break


MessageQueue.ReadTimeoutWithWarnings.timeout_ms = timeout_ms
MessageQueue.acquire_read = acquire_read
