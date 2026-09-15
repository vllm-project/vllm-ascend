#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#
from contextlib import contextmanager

from vllm.distributed.device_communicators import shm_broadcast
from vllm.v1.executor.multiproc_executor import WorkerProc

from vllm_ascend.common.utils.watch_dog import get_watch_dog

_watchdog = get_watch_dog()

_original_worker_busy_loop = WorkerProc.worker_busy_loop


def _patched_worker_busy_loop(*args, **kwargs):
    _watchdog.setup("worker")
    _watchdog.start()
    return _original_worker_busy_loop(*args, **kwargs)


@contextmanager
def _patched_queue_acquire_read(self, timeout: float | None = None, indefinite: bool = False):
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
                # patched watchdog:
                _watchdog.feed()

                if self.shutting_down:
                    raise RuntimeError("cancelled")

                # if we wait for a long time, log a message
                if read_timeout.should_warn():
                    shm_broadcast.logger.info(
                        shm_broadcast.LONG_WAIT_TIME_LOG_MSG, shm_broadcast.VLLM_RINGBUFFER_WARNING_INTERVAL
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


WorkerProc.worker_busy_loop = _patched_worker_busy_loop
shm_broadcast.MessageQueue.acquire_read = _patched_queue_acquire_read
