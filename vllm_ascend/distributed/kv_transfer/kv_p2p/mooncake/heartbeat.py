# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Scheduler-independent lease heartbeats for Mooncake pull requests."""

import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any

import msgspec
import zmq
from vllm.logger import logger
from vllm.utils.network_utils import make_zmq_path, make_zmq_socket

HEARTBEAT_MSG = b"heartbeat_msg"
DEFAULT_KV_LEASE_DURATION = 30
MAX_HEARTBEAT_REQUESTS = 512
HEARTBEAT_ACK_TIMEOUT = 1.0


@dataclass
class _Peer:
    socket: Any
    pending: deque[bytes] = field(default_factory=deque)
    ack_deadline: float = 0.0


class MooncakeHeartbeatThread(threading.Thread):
    """Renew outstanding requests even when the scheduler waits on a worker.

    Only this thread owns sockets. The scheduler changes the request registry
    in O(1); snapshots and network work happen once per heartbeat interval.
    Each producer has at most one unacknowledged message, independently of
    other producers. Failed sockets are retried at the next interval.
    """

    def __init__(self, interval: float) -> None:
        super().__init__(daemon=True, name="MooncakeHeartbeatThread")
        self.interval = interval
        self.requests: dict[str, tuple[str, int, str]] = {}
        self.state_lock = threading.Lock()
        self.stop_event = threading.Event()
        self.wake_event = threading.Event()

    def add_request(self, local_id: str, host: str, port: int, remote_id: str) -> None:
        with self.state_lock:
            was_empty = not self.requests
            self.requests[local_id] = (host, port, remote_id)
        if was_empty:
            self.wake_event.set()

    def remove_request(self, local_id: str) -> tuple[str, int, str] | None:
        with self.state_lock:
            return self.requests.pop(local_id, None)

    def shutdown(self) -> None:
        self.stop_event.set()
        self.wake_event.set()
        if self.ident is not None:
            self.join()
        with self.state_lock:
            self.requests.clear()

    def run(self) -> None:
        peers: dict[tuple[str, int], _Peer] = {}
        poller = zmq.Poller()  # type: ignore[attr-defined]
        encoder = msgspec.msgpack.Encoder()
        next_send = 0.0
        with zmq.Context() as context:  # type: ignore[attr-defined]
            try:
                while not self.stop_event.is_set():
                    now = time.monotonic()
                    if now >= next_send:
                        with self.state_lock:
                            requests = list(self.requests.values())
                        groups: dict[tuple[str, int], set[str]] = defaultdict(set)
                        for host, port, remote_id in requests:
                            groups[host, port].add(remote_id)
                        for endpoint in peers.keys() - groups.keys():
                            self._close_peer(peers.pop(endpoint), poller)
                        for endpoint, ids in groups.items():
                            peer = peers.get(endpoint)
                            if peer is None:
                                peer = self._connect_peer(context, endpoint, poller)
                                if peer is None:
                                    continue
                                peers[endpoint] = peer
                            # Drain the current snapshot before starting another;
                            # slow ACKs must not repeatedly discard its tail.
                            if not peer.pending:
                                ids_list = tuple(ids)
                                peer.pending = deque(
                                    encoder.encode((HEARTBEAT_MSG, ids_list[i : i + MAX_HEARTBEAT_REQUESTS]))
                                    for i in range(0, len(ids_list), MAX_HEARTBEAT_REQUESTS)
                                )
                        next_send = now + self.interval
                    for endpoint, peer in list(peers.items()):
                        try:
                            if peer.ack_deadline:
                                try:
                                    response = peer.socket.recv(flags=zmq.NOBLOCK)  # type: ignore[attr-defined]
                                except zmq.Again:  # type: ignore[attr-defined]
                                    if now < peer.ack_deadline:
                                        continue
                                    raise TimeoutError("heartbeat ACK timed out") from None
                                if response != b"ACK":
                                    raise ValueError(f"Unexpected heartbeat ACK: {response!r}")
                                peer.ack_deadline = 0.0
                            if peer.pending:
                                peer.socket.send(peer.pending[0], flags=zmq.NOBLOCK)  # type: ignore[attr-defined]
                                peer.pending.popleft()
                                peer.ack_deadline = now + HEARTBEAT_ACK_TIMEOUT
                        except zmq.Again:  # type: ignore[attr-defined]
                            continue
                        except (zmq.ZMQError, TimeoutError, ValueError):  # type: ignore[attr-defined]
                            logger.warning("Mooncake heartbeat failed for %s:%d; retrying next interval", *endpoint)
                            self._close_peer(peers.pop(endpoint), poller)
                    if peers:
                        poller.poll(100)
                    else:
                        # An idle connector needs no periodic wakeup.
                        with self.state_lock:
                            idle = not self.requests
                        self.wake_event.wait(None if idle else max(0.0, next_send - time.monotonic()))
                        self.wake_event.clear()
                        next_send = 0.0
            except Exception:
                logger.exception("Mooncake heartbeat thread failed")
            finally:
                for peer in peers.values():
                    self._close_peer(peer, poller)

    @staticmethod
    def _connect_peer(context: Any, endpoint: tuple[str, int], poller: Any) -> _Peer | None:
        sock = None
        try:
            path = make_zmq_path("tcp", *endpoint)
            sock = make_zmq_socket(context, path, zmq.REQ, bind=False)  # type: ignore[attr-defined]
            sock.setsockopt(zmq.LINGER, 0)  # type: ignore[attr-defined]
            poller.register(sock, zmq.POLLIN)  # type: ignore[attr-defined]
            return _Peer(sock)
        except Exception:
            if sock is not None:
                sock.close(linger=0)
            logger.warning(
                "Cannot connect Mooncake heartbeat peer %s:%s; retrying next interval", *endpoint, exc_info=True
            )
            return None

    @staticmethod
    def _close_peer(peer: _Peer, poller: Any) -> None:
        poller.unregister(peer.socket)
        peer.socket.close(linger=0)
