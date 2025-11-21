import os
import pickle
from threading import Event
from typing import Optional

from vllm.distributed.device_communicators.shm_broadcast import MessageQueue
from vllm.logger import init_logger
from zmq import IPV6, SUB, SUBSCRIBE, XPUB, XPUB_VERBOSE, Context

logger = init_logger(__name__)


def enqueue(self, obj, timeout: Optional[float] = None):
    """ Write to message queue with optional timeout (in seconds) """
    serialized_obj = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    if self.n_local_reader > 0:
        with self.acquire_write(timeout) as buf:
            buf[0] = 1  # overflow
        self.local_socket.send(serialized_obj)
    if self.n_remote_reader > 0:
        self.remote_socket.send(serialized_obj)


def dequeue(self,
            timeout: Optional[float] = None,
            cancel: Optional[Event] = None):
    """ Read from message queue with optional timeout (in seconds) """
    if self._is_local_reader:
        with self.acquire_read(timeout, cancel) as buf:
            overflow = buf[0] == 1
            if overflow:
                obj = MessageQueue.recv(self.local_socket, timeout)
            else:
                use_zmq_broadcast = bool(
                    int(os.environ.get("USE_ZMQ_BROADCAST", "1")))
                logger.warning(f"dequeue not overflow, {use_zmq_broadcast=}")
                obj = MessageQueue.recv(self.local_socket, timeout)
    elif self._is_remote_reader:
        obj = MessageQueue.recv(self.remote_socket, timeout)
    else:
        raise RuntimeError("Only readers can dequeue")
    return obj


def patch_shm_to_zmq():
    use_zmq_broadcast = bool(int(os.environ.get("USE_ZMQ_BROADCAST", "1")))
    if use_zmq_broadcast:
        MessageQueue.enqueue = enqueue
        MessageQueue.dequeue = dequeue
