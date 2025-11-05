import time
from contextlib import contextmanager
from typing import Optional

import vllm.envs as envs
from vllm.distributed.device_communicators.shm_broadcast import (Handle,
                                                                 MessageQueue,
                                                                 ShmRingBuffer,
                                                                 SpinTimer)
from vllm.distributed.utils import sched_yield
from vllm.logger import logger
from vllm.utils import (get_ip, get_mp_context, get_open_port,
                        get_open_zmq_ipc_path, is_valid_ipv6_address)
from zmq import IPV6, XPUB, XPUB_VERBOSE, Context  # type: ignore

VLLM_RINGBUFFER_WARNING_INTERVAL = envs.VLLM_RINGBUFFER_WARNING_INTERVAL


def __init__(
    self,
    n_reader,  # number of all readers
    n_local_reader,  # number of local readers through shared memory
    local_reader_ranks: Optional[list[int]] = None,
    max_chunk_bytes: int = 1024 * 1024 * 10,
    max_chunks: int = 10,
    connect_ip: Optional[str] = None,
):
    if local_reader_ranks is None:
        local_reader_ranks = list(range(n_local_reader))
    else:
        assert len(local_reader_ranks) == n_local_reader
    self.n_local_reader = n_local_reader
    n_remote_reader = n_reader - n_local_reader
    self.n_remote_reader = n_remote_reader

    context = Context()

    if n_local_reader > 0:
        # for local readers, we will:
        # 1. create a shared memory ring buffer to communicate small data
        # 2. create a publish-subscribe socket to communicate large data
        self.buffer = ShmRingBuffer(n_local_reader, max_chunk_bytes,
                                    max_chunks)

        # XPUB is very similar to PUB,
        # except that it can receive subscription messages
        # to confirm the number of subscribers
        self.local_socket = context.socket(XPUB)
        # set the verbose option so that we can receive every subscription
        # message. otherwise, we will only receive the first subscription
        # see http://api.zeromq.org/3-3:zmq-setsockopt for more details
        self.local_socket.setsockopt(XPUB_VERBOSE, True)
        local_subscribe_addr = get_open_zmq_ipc_path()
        logger.debug("Binding to %s", local_subscribe_addr)
        self.local_socket.bind(local_subscribe_addr)

        self.current_idx = 0
        self.writer_lock = get_mp_context().Lock()
    else:
        self.buffer = None  # type: ignore
        local_subscribe_addr = None
        self.local_socket = None
        self.current_idx = -1

    remote_addr_ipv6 = False
    if n_remote_reader > 0:
        # for remote readers, we will:
        # create a publish-subscribe socket to communicate large data
        if not connect_ip:
            connect_ip = get_ip()
        self.remote_socket = context.socket(XPUB)
        self.remote_socket.setsockopt(XPUB_VERBOSE, True)
        remote_subscribe_port = get_open_port()
        if is_valid_ipv6_address(connect_ip):
            self.remote_socket.setsockopt(IPV6, 1)
            remote_addr_ipv6 = True
            connect_ip = f"[{connect_ip}]"
        socket_addr = f"tcp://{connect_ip}:{remote_subscribe_port}"
        self.remote_socket.bind(socket_addr)
        remote_subscribe_addr = f"tcp://{connect_ip}:{remote_subscribe_port}"
    else:
        remote_subscribe_addr = None
        self.remote_socket = None

    self._is_writer = True
    self._is_local_reader = False
    self.local_reader_rank = -1
    # rank does not matter for remote readers
    self._is_remote_reader = False
    self._read_spin_timer = SpinTimer()

    self.handle = Handle(
        local_reader_ranks=local_reader_ranks,
        buffer_handle=self.buffer.handle()
        if self.buffer is not None else None,
        local_subscribe_addr=local_subscribe_addr,
        remote_subscribe_addr=remote_subscribe_addr,
        remote_addr_ipv6=remote_addr_ipv6,
    )

    logger.info("vLLM message queue communication handle: %s", self.handle)


@contextmanager
def acquire_write(self, timeout: Optional[float] = None):
    assert self._is_writer, "Only writers can acquire write"
    start_time = time.monotonic()
    n_warning = 1
    while True:
        with self.buffer.get_metadata(self.current_idx) as metadata_buffer:
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
                        "No available shared memory broadcast block found"
                        " in %s seconds. This typically happens when some"
                        " processes are hanging or doing some"
                        " time-consuming work (e.g. compilation)",
                        VLLM_RINGBUFFER_WARNING_INTERVAL)
                    n_warning += 1

                continue
            # found a block that is either
            # (1) not written
            # (2) read by all readers

            with self.writer_lock:
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
            # mark the block as written
            metadata_buffer[0] = 1
            self.current_idx = (self.current_idx + 1) % self.buffer.max_chunks
            break


MessageQueue.__init__ = __init__
MessageQueue.acquire_write = acquire_write
