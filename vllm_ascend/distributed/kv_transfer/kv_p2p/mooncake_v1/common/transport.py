# SPDX-License-Identifier: Apache-2.0
"""Transport helpers shared by the mooncake v1 KV connectors.

The helpers in this module are byte-for-byte copies of the definitions that used
to live in ``mooncake_connector``. They are collected here so that both V1
connectors (pull and layerwise) share one implementation and the unit tests have
a single patch target.

They are intentionally **not** imported from the V2 ``mooncake/utils.py``:

* V2 ``mooncake/__init__.py`` eagerly imports the whole V2 package
  (base_scheduler/base_worker/connector/metadata/pull_scheduler/pull_worker/
  stats), so reusing its utils would drag V2 into the V1 import graph;
* the two copies have already diverged: ``zmq_ctx`` here accepts ``DEALER``
  while the V2 version only accepts ``ROUTER``/``REQ``, and V2 ``SizedDict`` is
  a generic version;
* keeping the copy local gives the V1 unit tests a stable ``mooncake_v1.common``
  patch target instead of coupling them to the V2 module.

``RemoteSocketPool.close()`` is wired into the pull connector worker shutdown
and is terminal: once it runs, ``get()`` raises and ``return_socket()``
releases a checked-out socket in place instead of pooling it. Sockets that are
checked out while ``close()`` runs stay owned by their borrower, so their ZMQ
contexts are kept until they are returned or discarded.

This module must not import any ``vllm_ascend.*`` module.
"""

import contextlib
import hashlib
import struct
import threading
import time
from collections import OrderedDict, defaultdict, deque
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
import zmq
from vllm.logger import logger
from vllm.utils.network_utils import make_zmq_path, make_zmq_socket


@dataclass
class SizedDict(OrderedDict):
    def __init__(self, max_size=16000, *args, **kwargs):
        self.max_size = max_size
        super().__init__(*args, **kwargs)

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        if len(self) > self.max_size:
            self.popitem(last=False)

    def __getitem__(self, key):
        try:
            return super().__getitem__(key)
        except KeyError:
            value: dict[int, list[int]] = {}
            self[key] = value
            return value


@contextlib.contextmanager
def zmq_ctx(socket_type: Any, addr: str) -> Iterator[Any]:  # type: ignore
    """Context manager for a ZMQ socket"""

    if socket_type not in (zmq.ROUTER, zmq.REQ, zmq.DEALER):  # type: ignore
        raise ValueError(f"Unexpected socket type: {socket_type}")

    ctx: zmq.Context | None = None  # type: ignore
    try:
        ctx = zmq.Context()  # type: ignore
        yield make_zmq_socket(ctx=ctx, path=addr, socket_type=socket_type, bind=socket_type == zmq.ROUTER)  # type: ignore
    finally:
        if ctx is not None:
            ctx.destroy(linger=0)


def group_concurrent_contiguous(
    src: list[int],
    dst: list[int],
    src_block_stride: int = 1,
    dst_block_stride: int = 1,
    block_len: int = 1,
) -> tuple[list[list[int]], list[list[int]]]:
    """Group block ids that are contiguous in both id space and memory."""
    src_indices: npt.NDArray[np.int64] = np.array(src, dtype=np.int64)
    dst_indices: npt.NDArray[np.int64] = np.array(dst, dtype=np.int64)

    if src_indices.size == 0:
        return [], []

    src_byte_contiguous = np.diff(src_indices) * src_block_stride == block_len
    dst_byte_contiguous = np.diff(dst_indices) * dst_block_stride == block_len
    brk = np.where(~(src_byte_contiguous & dst_byte_contiguous))[0] + 1
    src_groups = np.split(src_indices, brk)
    dst_groups = np.split(dst_indices, brk)

    src_groups = [g.tolist() for g in src_groups]
    dst_groups = [g.tolist() for g in dst_groups]

    return src_groups, dst_groups


def split_if_not_byte_contiguous(
    src_groups: list[list[int]],
    dst_groups: list[list[int]],
    src_block_stride: int,
    dst_block_stride: int,
    block_len: int,
) -> tuple[list[list[int]], list[list[int]]]:
    if src_block_stride == block_len and dst_block_stride == block_len:
        return src_groups, dst_groups

    src = [bid for group in src_groups for bid in group]
    dst = [bid for group in dst_groups for bid in group]
    return group_concurrent_contiguous(
        src,
        dst,
        src_block_stride=src_block_stride,
        dst_block_stride=dst_block_stride,
        block_len=block_len,
    )


def string_to_int64_hash(input_str):
    """
    Hash the string using SHA-256 and convert it into an int64 integer.
    """
    hashed_bytes = hashlib.sha256(input_str.encode("utf-8")).digest()
    trunked_bytes = hashed_bytes[:8]
    uint64_value = struct.unpack("<Q", trunked_bytes)[0]
    return uint64_value


def ensure_zmq_send(
    socket: Any,  # type: ignore
    data: bytes,
    path: str,
    max_retries: int = 3,
):
    retries_left = max_retries
    while True:
        try:
            socket.send(data)
            return
        except zmq.ZMQError as e:  # type: ignore
            retries_left -= 1
            if retries_left > 0:
                logger.warning("Send failed. error=%s, attempts_left=%d. ", e, retries_left)
                time.sleep(0.1)
            else:
                logger.error("Send failed after all retries. error=%s. ", e)
                raise RuntimeError(f"Failed to send data to {path} after {max_retries} retries: {e}")


def ensure_zmq_recv(
    socket: Any,  # type: ignore
    path: str,
    max_retries: int = 3,
) -> bytes:
    retries_left = max_retries
    while True:
        try:
            return socket.recv()
        except zmq.ZMQError as e:  # type: ignore
            retries_left -= 1
            if retries_left > 0:
                logger.warning("Receive failed. error=%s, attempts_left=%d. ", e, retries_left)
                time.sleep(0.1)
            else:
                logger.error("Receive failed after all retries. source=%s, error=%s. ", path, e)
                raise RuntimeError(f"Failed to receive data after {max_retries} retries: {e}")


class RemoteSocketPool:
    """Reusable REQ sockets keyed by the remote ``host:port`` path.

    Extracted from ``KVCacheRecvingThread`` without behaviour changes: sockets
    are created lazily per path, reused through a FIFO queue and bounded by
    ``timeout`` seconds (``SNDTIMEO``/``RCVTIMEO``).

    ``close`` releases the sockets pooled by this instance together with the ZMQ
    context each of them created and puts the pool in its terminal state. The
    pull connector worker calls it during shutdown. After ``close``, ``get``
    raises and a checked-out socket returned with ``return_socket`` is closed in
    place. Sockets still checked out when ``close`` runs are not closed (their
    borrower owns them); their contexts are kept until they are returned with
    ``return_socket`` or released with ``discard``. A borrowed socket that is
    neither returned nor discarded keeps its context alive with the pool, so
    that responsibility stays with the borrower.
    """

    def __init__(self, timeout: float = 1.0):
        self.timeout = timeout
        self.lock = threading.Lock()
        self.sockets: dict[str, deque[Any]] = defaultdict(deque)
        # Sockets created by ``get`` own the context stored here; it is used by
        # ``close``/``return_socket`` to destroy the contexts this pool created.
        # Contexts of checked-out sockets stay here until they are returned.
        self._contexts: dict[Any, Any] = {}
        # Terminal state; set by ``close``.
        self._closed = False

    def get(self, remote_host: str, remote_handshake_port: int) -> Any:  # type: ignore
        """Get a socket to the remote host."""
        remote_path = make_zmq_path("tcp", remote_host, remote_handshake_port)
        with self.lock:
            if self._closed:
                raise RuntimeError("RemoteSocketPool is closed")
            if self.sockets[remote_path]:
                return self.sockets[remote_path].popleft()

            ctx = zmq.Context()  # type: ignore
            sock = make_zmq_socket(
                ctx=ctx,
                path=remote_path,
                socket_type=zmq.REQ,  # type: ignore
                bind=False,
            )
            sock.setsockopt(
                zmq.SNDTIMEO,  # type: ignore
                int(self.timeout * 1000),
            )
            sock.setsockopt(
                zmq.RCVTIMEO,  # type: ignore
                int(self.timeout * 1000),
            )
            self._contexts[sock] = ctx
            return sock

    def return_socket(
        self,
        sock: Any,  # type: ignore
        remote_host: str,
        remote_handshake_port: int,
    ) -> None:
        """Return the remote socket to the pool, or release it after close.

        Best effort: once the pool is closed the checked-out socket is closed and
        its context destroyed in place instead of being pooled, so a failure here
        only logs a warning and never raises.
        """
        remote_path = make_zmq_path("tcp", remote_host, remote_handshake_port)
        with self.lock:
            if not self._closed:
                self.sockets[remote_path].append(sock)
                return
            ctx = self._contexts.pop(sock, None)
        self._close_socket(sock, ctx, remote_path)

    def close(self) -> None:
        """Close the pooled sockets and destroy the contexts this pool created.

        Terminal and idempotent: the first call closes the pooled sockets and
        destroys their contexts, later calls are no-ops. Sockets checked out at
        that moment are left open and their contexts stay in ``_contexts`` until
        ``return_socket`` or ``discard`` releases them; ``close`` never touches a
        borrowed socket.
        """
        with self.lock:
            if self._closed:
                return
            self._closed = True
            pooled = [
                (remote_path, sock, self._contexts.pop(sock, None))
                for remote_path, sock_queue in self.sockets.items()
                for sock in sock_queue
            ]
            self.sockets = defaultdict(deque)
        for remote_path, sock, ctx in pooled:
            self._close_socket(sock, ctx, remote_path)

    def discard(self, sock: Any) -> None:  # type: ignore
        """Release one checked-out socket that will not be returned to the pool.

        Best effort: the socket is closed and its ZMQ context destroyed in place,
        logging a warning instead of raising.
        """
        with self.lock:
            ctx = self._contexts.pop(sock, None)
        self._close_socket(sock, ctx, "<discarded>")

    def _close_socket(self, sock: Any, ctx: Any, remote_path: str) -> None:  # type: ignore
        """Close one socket and destroy its context, logging failures."""
        try:
            sock.close(linger=0)
        except Exception:
            logger.warning("Failed to close pooled socket. path=%s. ", remote_path)
        if ctx is None:
            return
        try:
            ctx.destroy(linger=0)
        except Exception:
            logger.warning("Failed to destroy socket context. path=%s. ", remote_path)


__all__ = [
    "RemoteSocketPool",
    "SizedDict",
    "ensure_zmq_recv",
    "ensure_zmq_send",
    "group_concurrent_contiguous",
    "split_if_not_byte_contiguous",
    "string_to_int64_hash",
    "zmq_ctx",
]


def ensure_zmq_recv_with_poller(
    socket: zmq.Socket,  # type: ignore
    poller: zmq.Poller,  # type: ignore
    path: str,
    timeout: float = 1.0,
    max_retries: int = 3,
) -> bytes:
    retries_left = max_retries
    while True:
        try:
            if dict(poller.poll(int(timeout * 1000))):  # milliseconds
                data = socket.recv()
                return data
            else:
                raise zmq.ZMQError("Receive timeout")  # type: ignore
        except zmq.ZMQError as e:  # type: ignore
            retries_left -= 1
            if retries_left > 0:
                logger.warning("Receive failed. error=%s, attempts_left=%d. ", e, retries_left)
                time.sleep(0.1)
            else:
                logger.error("Receive failed after all retries. error=%s. ", e)
                raise RuntimeError(f"Failed to receive data from {path} after {max_retries} retries: {e}")
