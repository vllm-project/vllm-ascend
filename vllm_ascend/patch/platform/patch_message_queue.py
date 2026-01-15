import time
from contextlib import contextmanager

from vllm.distributed.device_communicators.shm_broadcast import (
    VLLM_RINGBUFFER_WARNING_INTERVAL, MessageQueue, long_wait_time_msg,
    memory_fence)
from vllm.distributed.utils import sched_yield
from vllm.logger import logger


@contextmanager
def acquire_write(self, timeout: float | None = None):
    assert self._is_writer, "Only writers can acquire write"
    start_time = time.monotonic()
    n_warning = 1
    while True:
        with self.buffer.get_metadata(self.current_idx) as metadata_buffer:
            # Memory fence ensures we see the latest read flags from readers.
            # Without this, we may read stale flags from our CPU cache and
            # spin indefinitely even though readers have completed.
            memory_fence()
            read_count = sum(metadata_buffer[1:])
            written_flag = metadata_buffer[0]
            if written_flag and read_count != self.buffer.n_reader:
                # this block is written and not read by all readers
                # for writers, `self.current_idx` is the next block to write
                # if this block is not ready to write,
                # we need to wait until it is read by all readers

                # Release the processor to other threads
                sched_yield()

                # if we time out, raise an exception
                elapsed = time.monotonic() - start_time
                if timeout is not None and elapsed > timeout:
                    raise TimeoutError

                # if we wait for a long time, log a message
                if elapsed > VLLM_RINGBUFFER_WARNING_INTERVAL * n_warning:
                    logger.info(
                        long_wait_time_msg(VLLM_RINGBUFFER_WARNING_INTERVAL))
                    n_warning += 1

                continue
            # found a block that is either
            # (1) not written
            # (2) read by all readers

            # mark the block as not written
            metadata_buffer[0] = 0
            # let caller write to the buffer
            with self.buffer.get_data(self.current_idx) as buf:
                yield buf

            # caller has written to the buffer
            # NOTE: order is important here
            # first set the read flags to 0
            # then set the written flag to 1
            # otherwise, the readers may think they already read the block
            for i in range(1, self.buffer.n_reader + 1):
                # set read flag to 0, meaning it is not read yet
                metadata_buffer[i] = 0
            # Memory fence here ensures the order of the buffer and flag writes.
            # This guarantees that when `metadata_buffer[0] = 1` is visible to
            # readers, `buf` can be completely ready. Without this, some CPU
            # architectures with weak ordering may incur memory inconsistency.
            memory_fence()
            # mark the block as written
            metadata_buffer[0] = 1
            # Memory fence ensures the write is visible to readers on other cores
            # before we proceed. Without this, readers may spin indefinitely
            # waiting for a write that's stuck in our CPU's store buffer.
            memory_fence()
            self.current_idx = (self.current_idx + 1) % self.buffer.max_chunks
            break


MessageQueue.acquire_write = acquire_write
