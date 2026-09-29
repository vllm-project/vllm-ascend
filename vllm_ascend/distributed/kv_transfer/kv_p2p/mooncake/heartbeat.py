# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Request lease tracking and the independent scheduler heartbeat thread."""

import math
import threading
import time
from collections import defaultdict
from typing import Any

import msgspec
import zmq
from vllm.logger import logger
from vllm.utils.network_utils import make_zmq_path

from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.utils import ensure_zmq_recv, ensure_zmq_send, zmq_ctx

HEARTBEAT_MSG = b"heartbeat_v1"
HEARTBEAT_VERSION = 1
DEFAULT_KV_LEASE_DURATION = 480
LEASE_IO_TIMEOUT_MS = 1000
HEARTBEAT_MAX_ATTEMPTS = 3


def validate_positive_int(name: str, value: Any) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def validate_lease_duration(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("kv_lease_duration must be a finite number of at least 6 seconds")
    duration = float(value)
    if not math.isfinite(duration) or duration < 6:
        raise ValueError("kv_lease_duration must be a finite number of at least 6 seconds")
    return duration


def renew_heartbeat_leases(deadlines: dict[str, float], request_ids: Any, duration: float) -> bool:
    """Renew live leases under the caller's producer-state lock."""
    if not isinstance(request_ids, (tuple, list)):
        return False
    if not all(isinstance(req_id, str) for req_id in request_ids):
        return False
    now = time.monotonic()
    for req_id in request_ids:
        old_expiry = deadlines.get(req_id)
        if old_expiry is not None and old_expiry > now:
            deadlines[req_id] = max(old_expiry, now + duration * 2 / 3)
    return True


class MooncakeHeartbeatThread(threading.Thread):
    """Renew active requests independently of DONE; never hold the lock during I/O."""

    def __init__(
        self,
        ready_event: threading.Event,
        *,
        io_timeout_ms: int = LEASE_IO_TIMEOUT_MS,
        max_attempts: int = HEARTBEAT_MAX_ATTEMPTS,
    ) -> None:
        super().__init__(daemon=True, name="MooncakeHeartbeatThread")
        self.ready_event = ready_event
        self.io_timeout_ms = io_timeout_ms
        self.max_attempts = max_attempts
        self._lock = threading.Lock()
        self._wakeup = threading.Event()
        self.requests: dict[str, tuple[str, str, int, str, float]] = {}
        self.last_sent: dict[str, float] = {}
        self._failures: dict[str, int] = {}

    def start_request(self, request_id: str, params: dict[str, Any] | None) -> None:
        if not params or not params.get("do_remote_prefill") or not params.get("remote_block_ids"):
            return
        if params.get("kv_lease_version") is None:
            return
        if params["kv_lease_version"] != HEARTBEAT_VERSION:
            raise ValueError("Unsupported Mooncake KV lease protocol")
        duration = validate_lease_duration(params.get("kv_lease_duration"))
        engine = params.get("remote_engine_id")
        host = params.get("remote_host")
        port = params.get("remote_port")
        remote_id = params.get("remote_request_id")
        if not isinstance(engine, str) or not engine or not isinstance(host, str) or not host:
            raise ValueError("Invalid Mooncake KV lease engine or host")
        if not isinstance(remote_id, str) or not remote_id or type(port) is not int or not 0 < port < 65536:
            raise ValueError("Invalid Mooncake KV lease port or request ID")
        with self._lock:
            for other_engine, other_host, other_port, _, other_duration in self.requests.values():
                if other_engine == engine and (host, port, duration) != (other_host, other_port, other_duration):
                    raise ValueError("Inconsistent Mooncake KV lease endpoint or duration for one engine")
            self.requests[request_id] = (engine, host, port, remote_id, duration)
            self._failures.pop(request_id, None)
            self._wakeup.set()

    def stop_request(self, request_id: str) -> None:
        with self._lock:
            remote = self.requests.pop(request_id, None)
            self._failures.pop(request_id, None)
            if remote is not None and not any(r[0] == remote[0] for r in self.requests.values()):
                self.last_sent.pop(remote[0], None)
            self._wakeup.set()

    def _next_heartbeat(self) -> tuple[tuple[str, str, int, tuple[str, ...]] | None, float | None]:
        """Select one due engine and its next deadline under the caller's lock."""
        now = time.monotonic()
        grouped: dict[str, set[str]] = defaultdict(set)
        endpoints: dict[str, tuple[str, int, float]] = {}
        for engine, host, port, remote_id, duration in self.requests.values():
            grouped[engine].add(remote_id)
            endpoints[engine] = (host, port, duration)
        timeout = None
        for engine in sorted(grouped, key=lambda key: self.last_sent.get(key, float("-inf"))):
            host, port, duration = endpoints[engine]
            last_sent = self.last_sent.get(engine)
            if last_sent is None or now - last_sent >= duration / 6:
                self.last_sent[engine] = now
                return (engine, host, port, tuple(sorted(grouped[engine]))), 0.0
            delay = max(0.0, last_sent + duration / 6 - now)
            timeout = delay if timeout is None else min(timeout, delay)
        return None, timeout

    def run(self) -> None:
        self.ready_event.set()
        while True:
            self._run_once()

    def _run_once(self) -> None:
        with self._lock:
            # Clear before reading state under the same lock as mutations, so
            # START/STOP between unlock and wait cannot lose their wakeup.
            self._wakeup.clear()
            snapshot, timeout = self._next_heartbeat()
            attempted = {
                req_id: state
                for req_id, state in self.requests.items()
                if snapshot is not None and state[0] == snapshot[0]
            }
        if snapshot is None:
            self._wakeup.wait(timeout)
            return
        engine, host, port, request_ids = snapshot
        succeeded = False
        try:
            self._send_control(host, port, (HEARTBEAT_MSG, engine, request_ids))
            succeeded = True
        except Exception:
            logger.exception("Failed Mooncake heartbeat for engine %s", engine)
        with self._lock:
            for req_id, state in attempted.items():
                # Ignore requests removed or registered again during network I/O.
                if self.requests.get(req_id) is not state:
                    continue
                if succeeded:
                    self._failures.pop(req_id, None)
                    continue
                failures = self._failures.get(req_id, 0) + 1
                if failures < self.max_attempts:
                    self._failures[req_id] = failures
                    continue
                del self.requests[req_id]
                self._failures.pop(req_id, None)
                # Stop renewal only; do not report transfer failure or send DONE.
                logger.warning(
                    "Stopping Mooncake heartbeat for request %s (engine %s) after %d consecutive failures",
                    req_id,
                    engine,
                    failures,
                )
            if not any(state[0] == engine for state in self.requests.values()):
                self.last_sent.pop(engine, None)

    def _send_control(self, host: str, port: int, message: tuple[Any, ...]) -> None:
        path = make_zmq_path("tcp", host, port)
        # Only this thread touches these sockets; DONE owns a separate pool.
        with zmq_ctx(zmq.REQ, path) as sock:  # type: ignore[attr-defined]
            sock.setsockopt(zmq.SNDTIMEO, self.io_timeout_ms)  # type: ignore[attr-defined]
            sock.setsockopt(zmq.RCVTIMEO, self.io_timeout_ms)  # type: ignore[attr-defined]
            ensure_zmq_send(sock, msgspec.msgpack.encode(message), path, max_retries=1)
            response = ensure_zmq_recv(sock, path, max_retries=1)
            if response != b"ACK":
                raise RuntimeError(f"Mooncake heartbeat rejected: {response!r}")
