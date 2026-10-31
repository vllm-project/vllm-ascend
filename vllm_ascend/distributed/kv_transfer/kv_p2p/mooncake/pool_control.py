# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Small JSON control channel for committed pool manifests.

KV bytes stay in the pool. Worker sockets and their contexts are exclusively
owned by the server thread; the client uses short-lived, bounded REQ sockets.
This channel also works when the source engine no longer schedules requests.
"""

import json
import math
import threading
import time
from collections.abc import Callable
from typing import Any

import zmq
from vllm.logger import logger

from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.pool_metadata import PoolWorkerReport


class PoolManifestRegistry:
    """Published reports outlive source HBM and expire independently of steps.

    The publisher owns native object protection before publish; release also
    cleans any partial protection left by a failed attempt. A failed release preserves the record so
    cleanup can be retried rather than silently losing lease ownership.
    """

    def __init__(
        self,
        incarnation: str,
        release: Callable[[PoolWorkerReport], None],
        *,
        retention_seconds: float,
        clock: Callable[[], float] = time.monotonic,
    ):
        if not incarnation or not math.isfinite(retention_seconds) or retention_seconds <= 0:
            raise ValueError("Pool manifests require incarnation and positive retention")
        self.incarnation = incarnation
        self.release = release
        self.retention_seconds = retention_seconds
        self.clock = clock
        self._lock = threading.RLock()
        self._reports: dict[tuple[str, str], tuple[PoolWorkerReport, float]] = {}
        self._closed = False

    def publish(self, report: PoolWorkerReport, *, retention_started_at: float | None = None) -> None:
        key = (report.transfer_id, report.attempt_id)
        with self._lock:
            if self._closed:
                raise RuntimeError("Pool manifest registry is closed")
            previous = self._reports.get(key)
            if previous is not None:
                if previous[0] != report:
                    raise ValueError("Conflicting pool manifest publication")
                return  # Duplicate publication cannot extend source retention.
            now = self.clock()
            start = now if retention_started_at is None else retention_started_at
            if not math.isfinite(start) or start > now:
                raise ValueError("Invalid pool retention start time")
            deadline = start + self.retention_seconds
            if deadline <= now:
                raise TimeoutError("Pool retention expired before manifest publication")
            self._reports[key] = (report, deadline)

    def _remove(self, key: tuple[str, str]) -> None:
        entry = self._reports.get(key)
        if entry is None:
            return  # Idempotent acknowledgement and expiry.
        report, _ = entry
        self.release(report)
        del self._reports[key]

    def expire(self) -> None:
        with self._lock:
            for key, (_, deadline) in list(self._reports.items()):
                if self.clock() >= deadline:
                    try:
                        self._remove(key)
                    except Exception:
                        # Retain failed ownership for retry, but still reclaim
                        # unrelated attempts and keep the control endpoint live.
                        logger.exception("Pool manifest expiry cleanup failed: attempt=%s", key[1])

    def handle(self, request: dict[str, Any]) -> dict[str, Any]:
        if request.keys() != {"op", "incarnation", "transfer_id", "attempt_id"}:
            raise ValueError("Invalid pool control fields")
        if request["incarnation"] != self.incarnation:
            raise ValueError("Stale pool worker incarnation")
        if any(not isinstance(request[name], str) or not request[name] for name in request):
            raise ValueError("Invalid pool control identity")
        key = (request["transfer_id"], request["attempt_id"])
        with self._lock:
            if self._closed:
                raise RuntimeError("Pool manifest registry is closed")
            if request["op"] == "release":
                self._remove(key)
                return {"released": True}
            if request["op"] != "manifest":
                raise ValueError("Unsupported pool control operation")
            entry = self._reports.get(key)
            if entry is None:
                raise ValueError("Pool attempt is not committed or has expired")
            report, deadline = entry
            if self.clock() >= deadline:
                self._remove(key)
                raise ValueError("Pool attempt retention expired")
            return report.to_wire()

    def close(self) -> None:
        with self._lock:
            self._closed = True
            errors = []
            for key in list(self._reports):
                try:
                    self._remove(key)
                except Exception as exc:
                    errors.append(exc)
                    logger.exception("Pool manifest close cleanup failed: attempt=%s", key[1])
            if errors:
                raise RuntimeError(f"Pool manifest cleanup failed for {len(errors)} attempts") from errors[0]


class PoolControlServer:
    def __init__(
        self,
        host: str,
        handler: Callable[[dict[str, Any]], dict[str, Any]],
        *,
        maintenance: Callable[[], None] | None = None,
        maintenance_interval_s: float = 1.0,
    ):
        if not math.isfinite(maintenance_interval_s) or maintenance_interval_s <= 0:
            raise ValueError("Pool control maintenance interval must be finite and positive")
        self.handler = handler
        self.host = host
        self.maintenance = maintenance
        self.maintenance_interval_s = maintenance_interval_s
        self.endpoint = ""
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._startup_error: BaseException | None = None
        self._thread = threading.Thread(target=self._run, name="pool-manifest-control", daemon=True)
        self._thread.start()
        if not self._ready.wait(timeout=10):
            self._stop.set()
            raise RuntimeError("Timed out starting pool manifest control")
        if self._startup_error is not None:
            raise RuntimeError("Failed to bind pool manifest control") from self._startup_error

    def _run(self) -> None:
        context = zmq.Context()
        socket = context.socket(zmq.REP)
        socket.setsockopt(zmq.LINGER, 0)
        socket.setsockopt(zmq.SNDTIMEO, 1000)
        socket.setsockopt(zmq.RCVTIMEO, 100)
        try:
            host = f"[{self.host}]" if ":" in self.host else self.host
            if ":" in self.host:
                socket.setsockopt(zmq.IPV6, 1)
            port = socket.bind_to_random_port(f"tcp://{host}")
            self.endpoint = f"tcp://{host}:{port}"
            self._ready.set()
            next_maintenance = time.monotonic() + self.maintenance_interval_s
            while not self._stop.is_set():
                if self.maintenance is not None and time.monotonic() >= next_maintenance:
                    try:
                        self.maintenance()
                    except Exception:
                        logger.exception("Pool control maintenance failed; retrying without stopping the endpoint")
                    next_maintenance = time.monotonic() + self.maintenance_interval_s
                try:
                    payload = socket.recv()
                except zmq.Again:
                    continue
                try:
                    request = json.loads(payload)
                    if not isinstance(request, dict):
                        raise ValueError("Pool control request must be an object")
                    response = {"ok": True, "result": self.handler(request)}
                except Exception as exc:
                    response = {"ok": False, "error": str(exc)}
                try:
                    socket.send_json(response)
                except zmq.Again:
                    # REP cannot receive another request after a failed send.
                    raise RuntimeError("Pool control response timed out") from None
        except BaseException as exc:
            self._startup_error = exc
            self._ready.set()
            logger.exception("Pool control server stopped unexpectedly")
        finally:
            socket.close()
            context.term()

    def close(self) -> None:
        self._stop.set()
        # A callback may still own a native lease operation. Never allow the
        # worker to close its backend while that call is executing.
        self._thread.join()

    def check_health(self) -> None:
        if not self._thread.is_alive() or self._startup_error is not None:
            raise RuntimeError("Pool manifest control server is unavailable") from self._startup_error


def pool_control_request(endpoint: str, request: dict[str, Any], *, timeout_ms: int = 5000) -> dict[str, Any]:
    if not endpoint.startswith("tcp://") or type(timeout_ms) is not int or timeout_ms <= 0:
        raise ValueError("Pool control requires a TCP endpoint and positive timeout")
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.setsockopt(zmq.LINGER, 0)
    socket.setsockopt(zmq.SNDTIMEO, timeout_ms)
    socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
    if "[" in endpoint:
        socket.setsockopt(zmq.IPV6, 1)
    try:
        socket.connect(endpoint)
        socket.send_json(request)
        response = socket.recv_json()
        if not isinstance(response, dict) or response.get("ok") is not True:
            raise RuntimeError(f"Pool control rejected request: {response}")
        result = response.get("result")
        if not isinstance(result, dict):
            raise ValueError("Malformed pool control response")
        return result
    finally:
        socket.close()
        context.term()
