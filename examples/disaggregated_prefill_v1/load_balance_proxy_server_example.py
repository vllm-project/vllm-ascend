# Adapted from https://github.com/vllm-project/vllm/tests/v1/kv_connector/nixl_integration/toy_proxy_server.py

# SPDX-License-Identifier: Apache-2.0
#
# Tutorial: Using the Load Balance Proxy Server Example
#
# This proxy server is designed to distribute requests between multiple
# "prefiller" and "decoder" backend servers for large language model inference.
# It is useful for scaling out inference workloads and balancing load across
# multiple backend instances.
#
# Features:
# - Load balances requests to multiple prefiller and decoder servers.
# - Supports OpenAI-compatible /v1/completions and /v1/chat/completions endpoints.
# - Streams responses from backend servers to clients.
#
# Prerequisites:
# - Python 3.10+
# - Install dependencies:
#     pip install fastapi<0.124.0 httpx uvicorn vllm
#
# Step 1: Start Your Backend Servers
# ----------------------------------
# You need to have at least one prefiller and one decoder backend running.
# These can be mock servers or actual vLLM servers.
#
# For testing, you can use the provided mock server:
#
#   vllm serve --host 0.0.0.0 --port 8100 ... # Prefiller 1
#   vllm serve --host 0.0.0.0 --port 8101 ... # Prefiller 2
#   vllm serve --host 0.0.0.0 --port 8200 ... # Decoder 1
#   vllm serve --host 0.0.0.0 --port 8201 ... # Decoder 2
#
# Step 2: Start the Proxy Server
# ------------------------------
# Run the proxy server, specifying the host/port for each prefiller and decoder:
#
#   python load_balance_proxy_server_example.py \
#     --host 0.0.0.0 --port 9000 --workers 2 \
#     --prefiller-hosts 127.0.0.1 127.0.0.1 \
#     --prefiller-ports 8100 8101 \
#     --decoder-hosts 127.0.0.1 127.0.0.1 \
#     --decoder-ports 8200 8201
#
# This will start the proxy on port 9000, load balancing between two prefiller
# and two decoder servers.
#
# Step 3: Send a Request to the Proxy
# -----------------------------------
# You can now send OpenAI-compatible requests to the proxy. For example:
#
#   curl -X POST http://localhost:9000/v1/completions \
#     -H "Content-Type: application/json" \
#     -d '{
#           "model": "your-model",
#           "prompt": "The quick brown fox jumps over the lazy dog",
#           "max_tokens": 16
#         }'
#
# Or for chat completions:
#
#   curl -X POST http://localhost:9000/v1/chat/completions \
#     -H "Content-Type: application/json" \
#     -d '{
#           "model": "your-model",
#           "messages": [{"role": "user", "content": "Hello!"}],
#           "max_tokens": 16
#         }'
#
# Step 4: Health Check
# --------------------
# To check if the proxy is running and see how many backend instances are
# connected, use:
#
#   curl http://localhost:9000/healthcheck
#
# This will return a JSON object with the status and the number of prefiller
# and decoder instances.
#
# Step 5: Add or Remove Prefiller or Decoder Instances (Optional)
# ---------------------------------------------------------------
# You can add or remove prefiller or decoder instances after the proxy is started.
# For example, add 2 prefiller instances:
#
#   curl -X POST http://localhost:9000/instances/add \
#     -H "Content-Type: application/json" \
#     -d '{
#           "type": "prefill",
#           "instances": ["127.0.0.1:8102", "127.0.0.1:8103"]
#         }'
#
# or remove 1 decoder instance:
#
#   curl -X POST http://localhost:9000/instances/remove \
#     -H "Content-Type: application/json" \
#     -d '{
#           "type": "decode",
#           "instances": "127.0.0.1:8201"
#         }'
#
# This will return a JSON object with the adding or removing info
# and the current prefiller and decoder instances.
#
# When adding instances, if the instances are not started,
# the proxy will wait and try until the instances to be started
# or exceeding the number of attempts
#
# Notes:
# - You can scale the number of prefiller and decoder servers as needed.
# - The proxy will round-robin requests to balance load.
# - For production, ensure your backend servers are robust and secure.
#
# For more details, see the code and comments in this file.

import argparse
import asyncio
import base64
import functools
import heapq
import ipaddress
import json
import logging
import os
import sys
import tempfile
import threading
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field, replace
from enum import Enum, IntEnum
from multiprocessing.managers import BaseManager
from pathlib import Path
from typing import Any, cast

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

logger = logging.getLogger(__name__)

try:
    import uvloop  # type: ignore[import-not-found]

    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
except ImportError:
    pass


class ServerRole(str, Enum):
    PREFILL = "prefill"
    DECODE = "decode"


@dataclass
class InstanceInfo:
    request_id: str
    prefiller_key: str
    prefiller_score: float
    decoder_key: str
    decoder_score: float
    decoder_host: str
    decoder_port: int
    prefiller_cached_tokens: int | None = None


TAINT_PRIORITY = 1e15


class CBState(IntEnum):
    """Circuit-breaker states (mirrors vllm-router 0.1.15)."""

    CLOSED = 0  # normal, requests allowed
    OPEN = 1  # tripped, can_execute() == False
    HALF_OPEN = 2  # trial period, limited requests allowed


_CB_NAME = {CBState.CLOSED: "closed", CBState.OPEN: "open", CBState.HALF_OPEN: "half_open"}


def _cb_name(state) -> str:
    if state is None:
        return "disabled"
    return _CB_NAME.get(state, str(state))


class CircuitBreaker:
    """Three-state circuit breaker mirroring vllm-router 0.1.15 semantics.

    State machine:
      CLOSED    --(consecutive failures >= failure_threshold)--> OPEN
      OPEN      --(after recovery_timeout, lazily on can_execute)--> HALF_OPEN
      HALF_OPEN --(any failure)--> OPEN
      HALF_OPEN --(consecutive successes >= success_threshold)--> CLOSED

    4xx client errors are NOT counted as failures (only 5xx / network errors).

    Thread-safety: a threading.Lock guards all mutations. In normal operation
    everything runs on the single event-loop thread, so contention is nil; the
    lock is defensive (keeps correctness if called from another thread, e.g.
    the NodeListener probe path). The on_transition/on_outcome callbacks run
    while the lock is held and must NOT re-acquire the scheduler RLock (to
    avoid lock-order inversion: callers always take RLock -> cb lock).
    """

    def __init__(
        self,
        failure_threshold: int = 3,
        success_threshold: int = 2,
        recovery_timeout: float = 10.0,
        enabled: bool = True,
        name: str = "",
        on_transition=None,  # callback(name, old_state, new_state)
        on_outcome=None,  # callback(name, outcome_str)
    ):
        self.failure_threshold = failure_threshold
        self.success_threshold = success_threshold
        self.recovery_timeout = recovery_timeout
        self.enabled = enabled
        self.name = name
        self._state = CBState.CLOSED
        self._consecutive_failures = 0
        self._consecutive_successes = 0
        self._last_state_change = time.monotonic()
        self._lock = threading.Lock()
        self._on_transition = on_transition
        self._on_outcome = on_outcome

    def _transition(self, new_state: CBState) -> None:
        old_state = self._state
        self._state = new_state
        self._last_state_change = time.monotonic()
        if new_state == CBState.OPEN:
            self._consecutive_failures = 0
        elif new_state == CBState.CLOSED:
            self._consecutive_successes = 0
        if old_state != new_state and self._on_transition is not None:
            self._on_transition(self.name, old_state, new_state)

    def can_execute(self) -> bool:
        """Whether a request may be sent to this node. Lazily transitions
        OPEN -> HALF_OPEN once recovery_timeout has elapsed."""
        if not self.enabled:
            return True
        with self._lock:
            if self._state == CBState.OPEN:
                if time.monotonic() - self._last_state_change >= self.recovery_timeout:
                    self._transition(CBState.HALF_OPEN)
                    return True  # allow one trial request
                if self._on_outcome is not None:
                    self._on_outcome(self.name, "rejected")
                return False
            return True

    def record_success(self) -> None:
        if not self.enabled:
            return
        with self._lock:
            if self._on_outcome is not None:
                self._on_outcome(self.name, "success")
            self._consecutive_failures = 0
            if self._state == CBState.HALF_OPEN:
                self._consecutive_successes += 1
                if self._consecutive_successes >= self.success_threshold:
                    self._transition(CBState.CLOSED)

    def record_failure(self, status_code: int | None = None) -> None:
        if not self.enabled:
            return
        # 4xx client errors are not failures (mirror vllm-router).
        if status_code is not None and 400 <= status_code < 500:
            return
        with self._lock:
            if self._on_outcome is not None:
                self._on_outcome(self.name, "failure")
            self._consecutive_successes = 0
            self._consecutive_failures += 1
            if self._state == CBState.HALF_OPEN:
                self._transition(CBState.OPEN)
            elif self._state == CBState.CLOSED and self._consecutive_failures >= self.failure_threshold:
                self._transition(CBState.OPEN)

    @property
    def state(self) -> CBState:
        with self._lock:
            return self._state


class NoAvailableNodeError(Exception):
    """Raised when no healthy/circuit-closed node is available for selection."""


class DecodeEarlyError(Exception):
    """Decode failed before any chunk reached the client (failover-able)."""

    def __init__(self, message: str = "", status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


class DecodeLateError(Exception):
    """Decode failed after chunks were already sent (cannot failover)."""

    def __init__(self, message: str = "", status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


def extract_cached_tokens(response_json: dict) -> int | None:
    usage = response_json.get("usage") or {}
    prompt_tokens_details = usage.get("prompt_tokens_details") or {}
    cached_tokens = prompt_tokens_details.get("cached_tokens")
    return cached_tokens if isinstance(cached_tokens, int) else None


def update_cached_tokens_in_chunk(chunk_json: dict, cached_tokens: int | None) -> bool:
    if cached_tokens is None:
        return False
    usage = chunk_json.get("usage")
    if not isinstance(usage, dict):
        return False
    prompt_tokens_details = usage.get("prompt_tokens_details")
    if not isinstance(prompt_tokens_details, dict):
        prompt_tokens_details = {}
    usage["prompt_tokens_details"] = prompt_tokens_details
    prompt_tokens_details["cached_tokens"] = cached_tokens
    return True


def encode_response_chunk(chunk_json: dict, is_sse: bool) -> bytes:
    chunk = json.dumps(chunk_json, ensure_ascii=False).encode("utf-8")
    return b"data: " + chunk + b"\n\n" if is_sse else chunk


global_args: argparse.Namespace | None = None
shared_scheduler: "SharedProxyScheduler | None" = None
runtime: "WorkerRuntime | None" = None


@dataclass
class BackendServer:
    host: str
    port: int
    ordinal: int
    active_tokens: float = 0.0
    active_kv_cache: float = 0.0
    heap_seq: int = 0
    # --- dual-layer health state (mirrors vllm-router 0.1.15) ---
    # Layer 1: passive /health probe result (maintained by NodeListener)
    healthy: bool = True
    consecutive_health_failures: int = 0
    consecutive_health_successes: int = 0
    # Layer 2: active circuit breaker driven by real request outcomes
    circuit_breaker: CircuitBreaker | None = None

    def is_available(self) -> bool:
        """Dual-layer availability: passive health AND active circuit breaker.

        Used by _pop_valid to hard-filter nodes out of the candidate pool.
        ``can_execute`` lazily transitions OPEN -> HALF_OPEN past recovery_timeout.
        """
        if not self.healthy:
            return False
        if self.circuit_breaker is None:
            return True
        return self.circuit_breaker.can_execute()


@dataclass
class RolePools:
    """Per-role scheduling state: live servers, priority heap, and drain-isolated keys."""

    servers: dict[str, BackendServer] = field(default_factory=dict)
    heap: list[tuple[float, int, int, str]] = field(default_factory=list)
    tainted: set[str] = field(default_factory=set)


def setup_logging(log_level: str) -> None:
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        force=True,
    )
    logger.setLevel(getattr(logging, log_level.upper()))


def next_req_id() -> str:
    return str(uuid.uuid4())


def calculate_prefill_score(request_length: int) -> float:
    length_score = request_length / 4.0
    return length_score * 0.0345 + 120.0745


def calculate_decode_score(request_length: int) -> float:
    return request_length


def normalize_host(host: str) -> str:
    return host.replace("localhost", "0.0.0.0").replace("127.0.0.1", "0.0.0.0")


def server_key(host: str, port: int) -> str:
    return f"{normalize_host(host)}:{int(port)}"


def build_server_url(host: str, port: int) -> str:
    url = f"http://{host}:{port}"
    try:
        ip = ipaddress.ip_address(host)
        if isinstance(ip, ipaddress.IPv6Address):
            url = f"http://[{host}]:{port}"
    except Exception:
        pass
    return url


def build_base_url(host: str, port: int) -> str:
    return f"{build_server_url(host, port)}/v1"


class MetricsCollector:
    """Zero-dependency Prometheus metrics collector living in the scheduler
    process so counters aggregate across all uvicorn workers (which reach it
    via RPC). /metrics renders the current snapshot as Prometheus text.
    """

    def __init__(self, scheduler=None):
        self.scheduler = scheduler
        self._cb_transitions: dict[tuple, int] = {}
        self._cb_outcomes: dict[tuple, int] = {}
        self._prefill_errors: dict[tuple, int] = {}
        self._decode_errors: dict[tuple, int] = {}
        self._failovers: dict[str, int] = {}
        self._requests = 0

    def record_cb_transition(self, role: str, instance: str, old_state, new_state) -> None:
        key = (role, instance, _cb_name(old_state), _cb_name(new_state))
        self._cb_transitions[key] = self._cb_transitions.get(key, 0) + 1

    def record_cb_outcome(self, role: str, instance: str, outcome: str) -> None:
        key = (role, instance, outcome)
        self._cb_outcomes[key] = self._cb_outcomes.get(key, 0) + 1

    def record_health_transition(self, role: str, instance: str, healthy: bool) -> None:
        # health is a live gauge rendered from ServerState; nothing to accumulate
        pass

    def record_prefill_error(self, instance: str, reason: str) -> None:
        key = (instance, reason or "unknown")
        self._prefill_errors[key] = self._prefill_errors.get(key, 0) + 1

    def record_decode_error(self, instance: str, reason: str) -> None:
        key = (instance, reason or "unknown")
        self._decode_errors[key] = self._decode_errors.get(key, 0) + 1

    def record_failover(self, role: str) -> None:
        self._failovers[role] = self._failovers.get(role, 0) + 1

    def record_request(self) -> None:
        self._requests += 1

    def render(self) -> str:
        lines: list[str] = []
        if self.scheduler is not None:
            snapshot = self.scheduler.get_health_snapshot()
            lines.append("# HELP pd_proxy_worker_health 1 if node passed /health probe, else 0")
            lines.append("# TYPE pd_proxy_worker_health gauge")
            lines.append("# HELP pd_proxy_cb_state Circuit breaker state: 0=closed,1=open,2=half_open")
            lines.append("# TYPE pd_proxy_cb_state gauge")
            lines.append("# HELP pd_proxy_worker_available 1 if node is selectable, else 0")
            lines.append("# TYPE pd_proxy_worker_available gauge")
            for role, inst, healthy, cb_state, available in snapshot:
                lines.append(
                    f'pd_proxy_worker_health{{role="{role}",instance="{inst}"}} {1 if healthy else 0}'
                )
                lines.append(
                    f'pd_proxy_cb_state{{role="{role}",instance="{inst}"}} {int(cb_state)}'
                )
                lines.append(
                    f'pd_proxy_worker_available{{role="{role}",instance="{inst}"}} {1 if available else 0}'
                )
        if self._cb_transitions:
            lines.append("# HELP pd_proxy_cb_state_transitions_total Circuit breaker state transitions")
            lines.append("# TYPE pd_proxy_cb_state_transitions_total counter")
            for (role, inst, frm, to), cnt in sorted(self._cb_transitions.items()):
                lines.append(
                    f'pd_proxy_cb_state_transitions_total{{role="{role}",instance="{inst}",from="{frm}",to="{to}"}} {cnt}'
                )
        if self._cb_outcomes:
            lines.append("# HELP pd_proxy_cb_outcomes_total Circuit breaker outcomes")
            lines.append("# TYPE pd_proxy_cb_outcomes_total counter")
            for (role, inst, outcome), cnt in sorted(self._cb_outcomes.items()):
                lines.append(
                    f'pd_proxy_cb_outcomes_total{{role="{role}",instance="{inst}",outcome="{outcome}"}} {cnt}'
                )
        if self._prefill_errors:
            lines.append("# HELP pd_proxy_prefill_errors_total Prefill failures")
            lines.append("# TYPE pd_proxy_prefill_errors_total counter")
            for (inst, reason), cnt in sorted(self._prefill_errors.items()):
                lines.append(
                    f'pd_proxy_prefill_errors_total{{prefiller="{inst}",reason="{reason}"}} {cnt}'
                )
        if self._decode_errors:
            lines.append("# HELP pd_proxy_decode_errors_total Decode failures")
            lines.append("# TYPE pd_proxy_decode_errors_total counter")
            for (inst, reason), cnt in sorted(self._decode_errors.items()):
                lines.append(
                    f'pd_proxy_decode_errors_total{{decoder="{inst}",reason="{reason}"}} {cnt}'
                )
        if self._failovers:
            lines.append("# HELP pd_proxy_failover_attempts_total Node-level failover attempts")
            lines.append("# TYPE pd_proxy_failover_attempts_total counter")
            for role, cnt in sorted(self._failovers.items()):
                lines.append(f'pd_proxy_failover_attempts_total{{role="{role}"}} {cnt}')
        lines.append("# HELP pd_proxy_requests_total Total requests handled")
        lines.append("# TYPE pd_proxy_requests_total counter")
        lines.append(f"pd_proxy_requests_total {self._requests}")
        return "\n".join(lines) + "\n"


class SharedProxyScheduler:
    """Centralized mutable scheduling state shared by all uvicorn workers.

    Uses lazy-deletion min-heap: on priority change, push a new entry and
    bump the server's ``heap_seq`` counter; stale entries (whose seq does
    not match) are skipped on pop.
    """

    def __init__(self, prefiller_instances, decoder_instances):
        self._lock = threading.RLock()
        self.request_num = 0
        self.waiting_nodes: dict[str, tuple[str, tuple[str, int], int]] = {}
        self._pools: dict[ServerRole, RolePools] = {
            ServerRole.PREFILL: RolePools(),
            ServerRole.DECODE: RolePools(),
        }
        self._ordinal = 0
        self.metrics = MetricsCollector(self)

        for host, port in prefiller_instances:
            self._add_server_no_lock(ServerRole.PREFILL, host, port)
        for host, port in decoder_instances:
            self._add_server_no_lock(ServerRole.DECODE, host, port)

    def _wire_circuit_breaker(self, role: ServerRole, entry: BackendServer) -> None:
        """Attach a configured CircuitBreaker with metrics hooks to a server."""
        args = get_global_args()
        entry.circuit_breaker = CircuitBreaker(
            failure_threshold=args.cb_failure_threshold,
            success_threshold=args.cb_success_threshold,
            recovery_timeout=args.cb_recovery_timeout_secs,
            enabled=not args.disable_circuit_breaker,
            name=f"{role.value}:{server_key(entry.host, entry.port)}",
            on_transition=self._on_cb_transition,
            on_outcome=self._on_cb_outcome,
        )

    def _on_cb_transition(self, name: str, old_state, new_state) -> None:
        try:
            role, instance = name.split(":", 1)
        except ValueError:
            return
        self.metrics.record_cb_transition(role, instance, old_state, new_state)

    def _on_cb_outcome(self, name: str, outcome: str) -> None:
        try:
            role, instance = name.split(":", 1)
        except ValueError:
            return
        self.metrics.record_cb_outcome(role, instance, outcome)

    def _pool(self, role: ServerRole) -> RolePools:
        return self._pools[role]

    @property
    def prefillers(self) -> dict[str, BackendServer]:
        return self._pool(ServerRole.PREFILL).servers

    @property
    def decoders(self) -> dict[str, BackendServer]:
        return self._pool(ServerRole.DECODE).servers

    def _next_ordinal(self) -> int:
        ordinal = self._ordinal
        self._ordinal += 1
        return ordinal

    def _priority(self, role: ServerRole, entry: BackendServer, key: str) -> float:
        if key in self._pool(role).tainted:
            return TAINT_PRIORITY
        if role is ServerRole.PREFILL:
            return entry.active_tokens + entry.active_kv_cache * 0.3
        return entry.active_tokens

    def _push_heap(self, role: ServerRole, key: str) -> None:
        pool = self._pool(role)
        entry = pool.servers[key]
        entry.heap_seq += 1
        heapq.heappush(pool.heap, (self._priority(role, entry, key), entry.ordinal, entry.heap_seq, key))
        if len(pool.heap) > 2 * len(pool.servers):
            self._reset_heap(role)

    def _pop_valid(self, role: ServerRole, *, exclude: set[str] | None = None) -> str:
        """Pop the lowest-priority *available* server key.

        Hard-isolates nodes that are not ``is_available()`` (unhealthy or
        circuit open) and any key in ``exclude`` (per-request failover
        exclusions): such entries are collected and pushed back so recovered
        nodes rejoin the pool automatically. Tainted (draining) nodes are NOT
        skipped here -- the taint mechanism is for manual /instances/remove.
        """
        pool = self._pool(role)
        exclude = exclude or set()
        skipped: list[tuple[float, int, int, str]] = []
        key: str | None = None
        while pool.heap:
            entry_tuple = heapq.heappop(pool.heap)
            _, _, seq, k = entry_tuple
            if k not in pool.servers:
                continue
            entry = pool.servers[k]
            if entry.heap_seq != seq:
                continue
            if k in exclude or not entry.is_available():
                skipped.append(entry_tuple)
                continue
            key = k
            break
        for item in skipped:
            heapq.heappush(pool.heap, item)
        if key is None:
            raise RuntimeError(f"No available {role.value} servers")
        return key

    def _reset_heap(self, role: ServerRole, *, bump_seq: bool = False) -> None:
        pool = self._pool(role)
        heap = []
        for key, entry in pool.servers.items():
            if bump_seq:
                entry.heap_seq += 1
            heap.append((self._priority(role, entry, key), entry.ordinal, entry.heap_seq, key))
        heapq.heapify(heap)
        pool.heap = heap

    def _add_server_no_lock(self, role: ServerRole, host: str, port: int) -> bool:
        key = server_key(host, port)
        pool = self._pool(role)
        if key in pool.servers:
            return False
        entry = BackendServer(host, int(port), self._next_ordinal())
        self._wire_circuit_breaker(role, entry)
        pool.servers[key] = entry
        self._push_heap(role, key)
        return True

    def get_snapshot(self) -> dict[str, list[dict[str, Any]]]:
        with self._lock:
            return {
                "prefill_instances": [
                    {"host": e.host, "port": e.port}
                    for _, e in sorted(self.prefillers.items(), key=lambda item: item[1].ordinal)
                ],
                "decode_instances": [
                    {"host": e.host, "port": e.port}
                    for _, e in sorted(self.decoders.items(), key=lambda item: item[1].ordinal)
                ],
            }

    def log_status(self, msg: str) -> None:
        snapshot = self.get_snapshot()
        logger.info(
            "%s prefill=%s decode=%s",
            msg,
            [f"{s['host']}:{s['port']}" for s in snapshot["prefill_instances"]],
            [f"{s['host']}:{s['port']}" for s in snapshot["decode_instances"]],
        )

    def _node_info(self, role: ServerRole, key: str, entry: BackendServer) -> dict[str, Any]:
        cb_state = entry.circuit_breaker.state if entry.circuit_breaker is not None else None
        return {
            "instance": key,
            "healthy": entry.healthy,
            "cb": _cb_name(cb_state),
            "available": entry.is_available(),
        }

    def get_health_snapshot(self) -> list[tuple[str, str, bool, CBState | None, bool]]:
        """Return per-node health/cb/availability for /metrics rendering.

        Tuple: (role_value, key, healthy, cb_state_or_None, available).
        Runs under the lock so the snapshot is consistent.
        """
        with self._lock:
            out: list[tuple[str, str, bool, CBState | None, bool]] = []
            for role in ServerRole:
                for key, entry in sorted(self._pool(role).servers.items(), key=lambda i: i[1].ordinal):
                    cb_state = entry.circuit_breaker.state if entry.circuit_breaker is not None else None
                    out.append((role.value, key, entry.healthy, cb_state, entry.is_available()))
            return out

    def healthcheck(self) -> dict[str, Any]:
        with self._lock:
            prefillers = [
                self._node_info(ServerRole.PREFILL, k, e)
                for k, e in sorted(self.prefillers.items(), key=lambda i: i[1].ordinal)
            ]
            decoders = [
                self._node_info(ServerRole.DECODE, k, e)
                for k, e in sorted(self.decoders.items(), key=lambda i: i[1].ordinal)
            ]
            total = len(prefillers) + len(decoders)
            avail = sum(1 for n in prefillers + decoders if n["available"])
            status = "unavailable" if avail == 0 else ("degraded" if avail < total else "ok")
            return {
                "status": status,
                "prefill_instances": len(self.prefillers),
                "decode_instances": len(self.decoders),
                "request_num": self.request_num,
                "prefillers": prefillers,
                "decoders": decoders,
            }

    def _pick_server(
        self,
        role: ServerRole,
        load: float,
        *,
        active_tokens: bool = False,
        kv_cache: bool = False,
        exclude: set[str] | None = None,
    ) -> dict[str, Any]:
        key = self._pop_valid(role, exclude=exclude)
        entry = self._pool(role).servers[key]
        if active_tokens:
            entry.active_tokens += load
        if kv_cache:
            entry.active_kv_cache += load
        self._push_heap(role, key)
        return {"key": key, "host": entry.host, "port": entry.port}

    def _release_load(
        self,
        role: ServerRole,
        key: str | None,
        load: float,
        *,
        active_tokens: bool = False,
        kv_cache: bool = False,
    ) -> None:
        if not key or key not in self._pool(role).servers:
            return
        entry = self._pool(role).servers[key]
        if active_tokens:
            entry.active_tokens -= load
        if kv_cache:
            entry.active_kv_cache = max(0.0, entry.active_kv_cache - load)
        self._push_heap(role, key)

    def begin_request(self, load: float) -> dict[str, Any]:
        """Pick a prefiller, reserve KV pressure, and count this as an active request."""
        with self._lock:
            picked = self._pick_server(ServerRole.PREFILL, load, kv_cache=True)
            self.request_num += 1
            return picked

    def reserve_prefill_kv(self, load: float) -> dict[str, Any]:
        """Pick a prefiller for recompute without bumping the active request count."""
        with self._lock:
            return self._pick_server(ServerRole.PREFILL, load, kv_cache=True)

    def pick_decoder(self, load: float) -> dict[str, Any]:
        with self._lock:
            return self._pick_server(ServerRole.DECODE, load, active_tokens=True)

    def pick_prefiller_excluding(self, load: float, exclude_keys: list[str]) -> dict[str, Any]:
        """Pick an available prefiller for recompute/failover, excluding given keys."""
        with self._lock:
            return self._pick_server(
                ServerRole.PREFILL, load, kv_cache=True, exclude=set(exclude_keys)
            )

    def pick_decoder_excluding(self, load: float, exclude_keys: list[str]) -> dict[str, Any]:
        """Pick an available decoder, excluding given (failed) keys for failover."""
        with self._lock:
            return self._pick_server(
                ServerRole.DECODE, load, active_tokens=True, exclude=set(exclude_keys)
            )

    def is_node_available(self, role: ServerRole, key: str) -> bool:
        with self._lock:
            entry = self._pool(role).servers.get(key)
            if entry is None:
                return False
            return entry.is_available()

    def record_outcome(
        self, role: ServerRole, key: str, success: bool, status_code: int = 0
    ) -> None:
        """Feed a real request outcome into a node's circuit breaker.

        Called by workers via runtime.schedule (RPC in multi-process mode).
        Only scalar args cross the wire; the CircuitBreaker object never leaves
        the scheduler process.
        """
        with self._lock:
            entry = self._pool(role).servers.get(key)
            if entry is None or entry.circuit_breaker is None:
                return
            if success:
                entry.circuit_breaker.record_success()
            else:
                entry.circuit_breaker.record_failure(status_code=status_code or None)

    def apply_health_result(self, role: ServerRole, key: str, healthy: bool) -> None:
        """Apply a passive /health probe result, flipping ``healthy`` at the
        configured success/failure thresholds. Called by NodeListener in the
        scheduler's own process."""
        args = get_global_args()
        with self._lock:
            entry = self._pool(role).servers.get(key)
            if entry is None:
                return
            if healthy:
                entry.consecutive_health_failures = 0
                entry.consecutive_health_successes += 1
                if (
                    not entry.healthy
                    and entry.consecutive_health_successes >= args.health_success_threshold
                ):
                    entry.healthy = True
                    logger.info("[health] %s %s recovered (healthy=True)", role.value, key)
                    self.metrics.record_health_transition(role.value, key, True)
            else:
                entry.consecutive_health_successes = 0
                entry.consecutive_health_failures += 1
                if (
                    entry.healthy
                    and entry.consecutive_health_failures >= args.health_failure_threshold
                ):
                    entry.healthy = False
                    logger.warning("[health] %s %s marked unhealthy (healthy=False)", role.value, key)
                    self.metrics.record_health_transition(role.value, key, False)

    def record_prefill_error(self, key: str, reason: str) -> None:
        self.metrics.record_prefill_error(key, reason)

    def record_decode_error(self, key: str, reason: str) -> None:
        self.metrics.record_decode_error(key, reason)

    def record_failover(self, role: str) -> None:
        self.metrics.record_failover(role)

    def record_request(self) -> None:
        self.metrics.record_request()

    def render_metrics(self) -> str:
        return self.metrics.render()

    def release_prefill_kv(self, key: str, load: float) -> None:
        with self._lock:
            self._release_load(ServerRole.PREFILL, key, load, kv_cache=True)

    def release_decoder(self, key: str, load: float) -> None:
        with self._lock:
            self._release_load(ServerRole.DECODE, key, load, active_tokens=True)

    def finish_request(
        self,
        prefiller_key: str | None,
        prefiller_load: float,
        decoder_key: str | None,
        decoder_load: float,
        release_prefill_kv: bool,
    ) -> None:
        with self._lock:
            if release_prefill_kv:
                self._release_load(ServerRole.PREFILL, prefiller_key, prefiller_load, kv_cache=True)
            self._release_load(ServerRole.DECODE, decoder_key, decoder_load, active_tokens=True)
            self.request_num = max(0, self.request_num - 1)

    def get_waiting_nodes(self) -> dict[str, tuple[str, tuple[str, int], int]]:
        with self._lock:
            return dict(self.waiting_nodes)

    def add_instances(self, role: ServerRole, instances: list[tuple[str, int]]) -> list[str]:
        waiting_nodes: list[str] = []
        with self._lock:
            servers = self._pool(role).servers
            for host, port in instances:
                key = server_key(host, port)
                if key in servers or key in self.waiting_nodes:
                    continue
                self.waiting_nodes[key] = (role.value, (host, int(port)), 0)
                waiting_nodes.append(f"{host}:{port}")
        return waiting_nodes

    def mark_waiting_retry(self, key: str, retry_count: int) -> None:
        with self._lock:
            if key not in self.waiting_nodes:
                return
            instance_type, server, _ = self.waiting_nodes[key]
            self.waiting_nodes[key] = (instance_type, server, retry_count)

    def activate_waiting_instance(self, role: ServerRole, host: str, port: int) -> None:
        with self._lock:
            key = server_key(host, port)
            self.waiting_nodes.pop(key, None)
            pool = self._pool(role)
            if key in pool.tainted:
                pool.tainted.discard(key)
                self._push_heap(role, key)
                return
            if self._add_server_no_lock(role, host, port):
                self.log_status(f"Add {role.value} instance: {host}:{port}.")

    def drop_waiting_instance(self, key: str) -> None:
        with self._lock:
            self.waiting_nodes.pop(key, None)

    def remove_instances(self, role: ServerRole, instances: list[tuple[str, int]]) -> bool:
        if not instances:
            return False
        keys = {server_key(host, port) for host, port in instances}
        with self._lock:
            pool = self._pool(role)
            if self.request_num > 0:
                pool.tainted.update(keys)
                self._reset_heap(role, bump_seq=True)
                logger.warning("Start to taint %s instances %s.", role.value, sorted(keys))
                return True

            removed = False
            for key in keys:
                removed = pool.servers.pop(key, None) is not None or removed
                self.waiting_nodes.pop(key, None)
            pool.tainted.difference_update(keys)
            if removed:
                self._reset_heap(role, bump_seq=True)
                self.log_status(f"Remove {role.value} instances: {sorted(keys)}.")
            return False

    def finalize_tainted_instances(self) -> None:
        with self._lock:
            if self.request_num != 0:
                return
            for role in ServerRole:
                pool = self._pool(role)
                if not pool.tainted:
                    continue
                keys = list(pool.tainted)
                for key in keys:
                    pool.servers.pop(key, None)
                pool.tainted.clear()
                self._reset_heap(role, bump_seq=True)
                self.log_status(f"Remove {role.value} instances after drain: {keys}.")


class SchedulerManager(BaseManager):
    """Multiprocessing RPC bridge; body is empty but required by BaseManager."""


def _shared_scheduler_proxy() -> "SharedProxyScheduler":
    if shared_scheduler is None:
        raise RuntimeError("shared scheduler is not initialized")
    return shared_scheduler


SchedulerManager.register("get_scheduler", callable=_shared_scheduler_proxy)


class WorkerRuntime:
    def __init__(self, scheduler: Any):
        self.scheduler = scheduler
        self._clients: dict[ServerRole, dict[str, httpx.AsyncClient]] = {
            ServerRole.PREFILL: {},
            ServerRole.DECODE: {},
        }
        self._async_lock = asyncio.Lock()

    async def schedule(self, method: str, /, *args, **kwargs) -> Any:
        async with self._async_lock:
            return getattr(self.scheduler, method)(*args, **kwargs)

    async def get_client(self, role: ServerRole, key: str) -> httpx.AsyncClient:
        clients = self._clients[role]
        if key not in clients:
            await self.sync_clients()
        return clients[key]

    async def sync_clients(self) -> None:
        snapshot = self.scheduler.get_snapshot()
        role_targets = {
            ServerRole.PREFILL: {
                server_key(s["host"], s["port"]): (s["host"], s["port"]) for s in snapshot["prefill_instances"]
            },
            ServerRole.DECODE: {
                server_key(s["host"], s["port"]): (s["host"], s["port"]) for s in snapshot["decode_instances"]
            },
        }
        for role, targets in role_targets.items():
            await self._sync_clients(role, targets)

    async def _sync_clients(self, role: ServerRole, targets: dict[str, tuple[str, int]]) -> None:
        clients = self._clients[role]
        for key in [key for key in clients if key not in targets]:
            await clients.pop(key).aclose()
        for key, (host, port) in targets.items():
            if key in clients:
                continue
            clients[key] = httpx.AsyncClient(
                timeout=None,
                base_url=build_base_url(host, port),
                limits=httpx.Limits(max_connections=100000, max_keepalive_connections=100000),
            )

    async def close(self) -> None:
        for role in ServerRole:
            for client in list(self._clients[role].values()):
                await client.aclose()
            self._clients[role].clear()


def get_runtime() -> WorkerRuntime:
    if runtime is None:
        raise RuntimeError("worker runtime is not initialized")
    return runtime


class NodeListener:
    def __init__(self, scheduler):
        self.scheduler = scheduler
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self) -> None:
        while True:
            args = get_global_args()
            # Passive /health probing of active nodes (mirrors vllm-router).
            if not args.disable_health_check:
                try:
                    asyncio.run(self._probe_all())
                except Exception as e:  # noqa: BLE001
                    logger.error("health probe cycle error: %s", e)

            for key, (instance_type, server, retries) in list(self.scheduler.get_waiting_nodes().items()):
                host, port = server
                is_valid = asyncio.run(self.check_instance_status(host, port))
                print(f"Checking instance {key}...")
                retries += 1
                if is_valid:
                    self.scheduler.activate_waiting_instance(ServerRole(instance_type), host, port)
                elif retries >= args.max_waiting_retries:
                    print(f"Instance {key} was not added to the proxy.")
                    self.scheduler.drop_waiting_instance(key)
                else:
                    self.scheduler.mark_waiting_retry(key, retries)

            self.scheduler.finalize_tainted_instances()
            interval = (
                args.health_check_interval
                if not args.disable_health_check
                else args.waiting_retry_interval
            )
            time.sleep(interval)

    async def _probe_all(self) -> None:
        """Concurrently probe /health of every active prefiller/decoder and
        feed results into ``apply_health_result``. Uses a bare-host URL (NOT
        the server client's base_url=.../v1, which would turn /health into
        /v1/health -> 404 on mock/real vLLM)."""
        snapshot = self.scheduler.get_snapshot()
        targets: list[tuple[ServerRole, str, str, int]] = []
        for s in snapshot["prefill_instances"]:
            targets.append((ServerRole.PREFILL, server_key(s["host"], s["port"]), s["host"], s["port"]))
        for s in snapshot["decode_instances"]:
            targets.append((ServerRole.DECODE, server_key(s["host"], s["port"]), s["host"], s["port"]))

        async def probe(role: ServerRole, key: str, host: str, port: int) -> None:
            ok = await self.check_instance_status(host, port)
            self.scheduler.apply_health_result(role, key, ok)

        await asyncio.gather(*[probe(*t) for t in targets], return_exceptions=True)

    @staticmethod
    async def check_instance_status(host: str, port: int) -> bool:
        """Probe the node's health endpoint at the ROOT path (not /v1/...).

        2xx => healthy. Uses a bare-host client so /health resolves correctly
        instead of being mangled into /v1/health (which 404s).
        """
        args = get_global_args()
        endpoint = args.health_check_endpoint
        url = f"{build_server_url(host, port)}{endpoint}"
        headers = {"Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}"}
        try:
            async with httpx.AsyncClient(timeout=args.health_check_timeout) as client:
                response = await client.get(url, headers=headers)
                return response.status_code < 400
        except (httpx.RequestError, httpx.HTTPStatusError):
            return False


def manager_config_path(proxy_port: int) -> Path:
    return Path(tempfile.gettempdir()) / f"vllm_lb_proxy_manager_{proxy_port}.json"


def write_manager_config(proxy_port: int, host: str, manager_port: int, authkey: bytes) -> None:
    manager_config_path(proxy_port).write_text(
        json.dumps(
            {
                "host": host,
                "port": manager_port,
                "authkey": base64.b64encode(authkey).decode("ascii"),
            }
        ),
        encoding="utf-8",
    )


def read_manager_config(proxy_port: int) -> dict[str, Any]:
    path = manager_config_path(proxy_port)
    if not path.is_file():
        raise RuntimeError(
            f"Manager config not found at {path}. "
            "Start the proxy from __main__ with --workers > 1 before worker processes connect."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def cleanup_manager_config(proxy_port: int) -> None:
    manager_config_path(proxy_port).unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--prefiller-hosts", type=str, nargs="+", default=["localhost"])
    parser.add_argument("--prefiller-ports", type=int, nargs="+", default=[8001])
    parser.add_argument("--decoder-hosts", type=str, nargs="+", default=["localhost"])
    parser.add_argument("--decoder-ports", type=int, nargs="+", default=[8002])
    parser.add_argument("--max-retries", type=int, default=3, help="Maximum number of retries for HTTP requests")
    parser.add_argument(
        "--retry-delay", type=float, default=0.001, help="Base delay (seconds) for exponential backoff retries"
    )
    parser.add_argument(
        "--max-waiting-retries", type=int, default=3, help="Maximum number of retries for waiting nodes to be started"
    )
    parser.add_argument(
        "--waiting-retry-interval",
        type=float,
        default=10,
        help="Check interval (seconds) for waiting nodes to be started",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of uvicorn worker processes. Scheduling state is shared across workers.",
    )
    # --- health check (passive /health probing; mirrors vllm-router) ---
    parser.add_argument(
        "--health-check-interval",
        type=float,
        default=5,
        help="Seconds between /health probes of active nodes",
    )
    parser.add_argument(
        "--health-check-timeout",
        type=float,
        default=2,
        help="Timeout (seconds) for a single /health probe",
    )
    parser.add_argument(
        "--health-failure-threshold",
        type=int,
        default=2,
        help="Consecutive probe failures to mark a node unhealthy",
    )
    parser.add_argument(
        "--health-success-threshold",
        type=int,
        default=2,
        help="Consecutive probe successes to mark a node healthy again",
    )
    parser.add_argument(
        "--health-check-endpoint",
        type=str,
        default="/health",
        help="Health probe endpoint (root path, not /v1/...)",
    )
    parser.add_argument(
        "--disable-health-check",
        action="store_true",
        help="Disable passive /health probing of active nodes",
    )
    # --- circuit breaker (active, driven by request outcomes) ---
    parser.add_argument(
        "--cb-failure-threshold",
        type=int,
        default=3,
        help="Consecutive request failures to open a node's circuit",
    )
    parser.add_argument(
        "--cb-success-threshold",
        type=int,
        default=2,
        help="Half-open consecutive successes to close a circuit",
    )
    parser.add_argument(
        "--cb-recovery-timeout-secs",
        type=float,
        default=10,
        help="Seconds in OPEN before lazily transitioning to HALF_OPEN",
    )
    parser.add_argument(
        "--disable-circuit-breaker",
        action="store_true",
        help="Disable the circuit breaker (can_execute always True)",
    )
    # --- request-level failover (the gap vllm-router leaves open in PD mode) ---
    parser.add_argument(
        "--failover-max-retries",
        type=int,
        default=3,
        help="Max distinct prefiller nodes tried on prefill failure",
    )
    parser.add_argument(
        "--failover-max-decoders",
        type=int,
        default=3,
        help="Max distinct decoder nodes tried on decode failure",
    )
    parser.add_argument(
        "--disable-failover",
        action="store_true",
        help="Disable node-level failover (fall back to same-node retries)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Log level for the proxy server.",
    )
    args = parser.parse_args()
    if len(args.prefiller_hosts) != len(args.prefiller_ports):
        raise ValueError("Number of prefiller hosts must match number of prefiller ports")
    if len(args.decoder_hosts) != len(args.decoder_ports):
        raise ValueError("Number of decoder hosts must match number of decoder ports")
    args.prefiller_instances = list(zip(args.prefiller_hosts, args.prefiller_ports))
    args.decoder_instances = list(zip(args.decoder_hosts, args.decoder_ports))
    return args


def get_global_args() -> argparse.Namespace:
    global global_args
    if global_args is None:
        global_args = parse_args()
    return global_args


def connect_shared_scheduler(proxy_port: int):
    manager_cfg = read_manager_config(proxy_port)
    manager = SchedulerManager(
        address=(manager_cfg["host"], manager_cfg["port"]),
        authkey=base64.b64decode(manager_cfg["authkey"]),
    )
    manager.connect()
    return manager.get_scheduler()  # type: ignore[attr-defined]


def bootstrap_parent_process(args: argparse.Namespace) -> None:
    """Initialize cross-worker shared state in the parent process before uvicorn spawns workers."""
    global shared_scheduler
    if args.workers <= 1:
        return

    shared_scheduler = SharedProxyScheduler(args.prefiller_instances, args.decoder_instances)
    NodeListener(shared_scheduler)

    authkey = os.urandom(16)
    manager = SchedulerManager(address=("127.0.0.1", 0), authkey=authkey)
    server = manager.get_server()
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = cast(tuple[str, int], server.address)
    write_manager_config(args.port, host, port, authkey)


def _ensure_scheduler(args) -> SharedProxyScheduler:
    global shared_scheduler
    if shared_scheduler is not None:
        return shared_scheduler
    shared_scheduler = SharedProxyScheduler(args.prefiller_instances, args.decoder_instances)
    NodeListener(shared_scheduler)
    return shared_scheduler


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global runtime
    args = get_global_args()
    if args.workers > 1:
        scheduler = connect_shared_scheduler(args.port)
    else:
        scheduler = _ensure_scheduler(args)
    runtime = WorkerRuntime(scheduler)
    await runtime.sync_clients()
    snapshot = scheduler.get_snapshot()
    logger.info(
        "Initialized %s prefill clients and %s decode clients in worker %s.",
        len(snapshot["prefill_instances"]),
        len(snapshot["decode_instances"]),
        os.getpid(),
    )
    yield
    await runtime.close()
    runtime = None


app = FastAPI(lifespan=lifespan)


def create_app():
    setup_logging(get_global_args().log_level)
    return app


async def listen_for_disconnect(request: Request) -> None:
    while True:
        message = await request.receive()
        if message["type"] == "http.disconnect":
            break


def with_cancellation(handler_func):
    @functools.wraps(handler_func)
    async def wrapper(*args, **kwargs):
        request = kwargs["request"]
        handler_task = asyncio.create_task(handler_func(*args, **kwargs))
        cancellation_task = asyncio.create_task(listen_for_disconnect(request))
        done, pending = await asyncio.wait([handler_task, cancellation_task], return_when=asyncio.FIRST_COMPLETED)
        for task in pending:
            task.cancel()
        if handler_task in done:
            return handler_task.result()
        return None

    return wrapper


def auth_headers(request_id: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}",
        "X-Request-Id": request_id,
    }


def build_prefill_request(req_data: dict) -> dict:
    payload = req_data.copy()
    payload["kv_transfer_params"] = {
        "do_remote_decode": True,
        "do_remote_prefill": False,
        "remote_engine_id": None,
        "remote_block_ids": None,
        "remote_host": None,
        "remote_port": None,
    }
    payload["stream"] = False
    payload["max_tokens"] = 1
    payload["min_tokens"] = 1
    if "max_completion_tokens" in payload:
        payload["max_completion_tokens"] = 1
    payload.pop("stream_options", None)
    return payload


async def send_request_to_service(
    client: httpx.AsyncClient,
    endpoint: str,
    req_data: dict,
    request_id: str,
    max_retries: int = 3,
    base_delay: float = 0.2,
):
    req_data = build_prefill_request(req_data)
    headers = auth_headers(request_id)
    last_exc = None
    for attempt in range(1, max_retries + 1):
        try:
            response = await client.post(endpoint, json=req_data, headers=headers)
            response.raise_for_status()
            return response
        except (httpx.RequestError, httpx.HTTPStatusError) as exc:
            logger.warning("Attempt %s failed for %s: %s", attempt, endpoint, exc)
            last_exc = exc
            if attempt < max_retries:
                await asyncio.sleep(base_delay * (2 ** (attempt - 1)))
            else:
                logger.error("All %s attempts failed for %s.", max_retries, endpoint)
                raise last_exc


async def stream_service_response_with_retry(
    client: httpx.AsyncClient,
    endpoint: str,
    req_data: dict,
    request_id: str,
    max_retries: int = 3,
    base_delay: float = 0.2,
):
    """Stream a decode response, classifying failures as early vs late.

    - DecodeEarlyError: failed before any chunk reached the caller -> the
      caller (generate_stream) may fail over to another decoder.
    - DecodeLateError: failed after chunks were already yielded -> cannot
      fail over (would duplicate tokens); the caller truncates the stream.

    When node-level failover is enabled the caller passes max_retries=1 so
    retries do not waste time on a node the circuit breaker is already
    counting; with --disable-failover the caller passes the full max_retries
    to retain the original same-node retry behavior.
    """
    headers = auth_headers(request_id)
    for attempt in range(1, max_retries + 1):
        first_chunk_sent = False
        try:
            async with client.stream("POST", endpoint, json=req_data, headers=headers) as response:
                response.raise_for_status()
                async for chunk in response.aiter_bytes():
                    first_chunk_sent = True
                    yield chunk
                return
        except (httpx.RequestError, httpx.HTTPStatusError) as exc:
            status_code = getattr(getattr(exc, "response", None), "status_code", None)
            if first_chunk_sent:
                raise DecodeLateError(str(exc), status_code=status_code) from exc
            if attempt < max_retries:
                logger.warning("Attempt %s failed for streaming %s: %s", attempt, endpoint, exc)
                await asyncio.sleep(base_delay * (2 ** (attempt - 1)))
                continue
            raise DecodeEarlyError(str(exc), status_code=status_code) from exc
        except Exception as exc:
            if first_chunk_sent:
                raise DecodeLateError(str(exc)) from exc
            if attempt < max_retries:
                logger.warning("Attempt %s failed for streaming %s: %s", attempt, endpoint, exc)
                await asyncio.sleep(base_delay * (2 ** (attempt - 1)))
                continue
            raise DecodeEarlyError(str(exc)) from exc


async def _finish_instance(runtime: WorkerRuntime, info: InstanceInfo, *, release_prefill_kv: bool) -> None:
    await runtime.schedule(
        "finish_request",
        info.prefiller_key,
        info.prefiller_score,
        info.decoder_key,
        info.decoder_score,
        release_prefill_kv,
    )


class DeferredStreamingResponse(StreamingResponse):
    """A StreamingResponse that defers committing the HTTP status line until
    the first chunk is produced.

    Why: Starlette's StreamingResponse sends ``http.response.start`` (status
    200) BEFORE iterating the body iterator. So if the generator raises before
    the first chunk, the client already has a 200 and ends up with an empty
    body -- the "silent 200" failure mode. This subclass pulls the first chunk
    first: if the generator raises (or yields nothing) before producing it, we
    send a 503 + JSON error body instead. Once the first chunk is out,
    behavior matches StreamingResponse (a later error can only truncate).
    """

    def __init__(
        self,
        content,
        error_status: int = 503,
        error_body: bytes = b'{"error":"upstream decode failed"}',
        **kwargs,
    ):
        super().__init__(content, **kwargs)
        self._error_status = error_status
        self._error_body = error_body

    async def stream_response(self, send) -> None:
        aiter = self.body_iterator.__aiter__()
        sent_start = False
        try:
            try:
                first = await aiter.__anext__()
            except StopAsyncIteration:
                # Empty stream -> treat as error to avoid 200 + empty body.
                await send(
                    {"type": "http.response.start", "status": self._error_status, "headers": []}
                )
                await send(
                    {"type": "http.response.body", "body": self._error_body, "more_body": False}
                )
                return
            # First chunk produced OK -> now commit the real status + headers.
            await send(
                {"type": "http.response.start", "status": self.status_code, "headers": self.raw_headers}
            )
            sent_start = True
            if not isinstance(first, (bytes, memoryview)):
                first = first.encode(self.charset)
            await send({"type": "http.response.body", "body": first, "more_body": True})
            async for chunk in aiter:
                if not isinstance(chunk, (bytes, memoryview)):
                    chunk = chunk.encode(self.charset)
                await send({"type": "http.response.body", "body": chunk, "more_body": True})
            await send({"type": "http.response.body", "body": b"", "more_body": False})
        except Exception:
            if not sent_start:
                # Generator failed before first chunk -> return an explicit error.
                await send(
                    {"type": "http.response.start", "status": self._error_status, "headers": []}
                )
                await send(
                    {"type": "http.response.body", "body": self._error_body, "more_body": False}
                )
            else:
                # Headers already committed -> can only end the stream.
                await send({"type": "http.response.body", "body": b"", "more_body": False})


async def assign_instances(
    api: str,
    req_data: Any,
    request_length: int,
    *,
    is_initial_request: bool,
    excluded_prefillers: set[str] | None = None,
) -> InstanceInfo:
    """Pick a prefiller, run the prefill request, then pick a decoder.

    Request-level failover (PD mode): when failover is enabled the inner HTTP
    retry count is reduced to 1 and on prefill failure we exclude the failed
    prefiller and select a different one, up to ``failover_max_retries``
    distinct nodes. Each outcome is fed to the node's circuit breaker. With
    ``--disable-failover`` the original same-node retry behavior is preserved.
    """
    runtime = get_runtime()
    args = get_global_args()
    prefiller_score = calculate_prefill_score(request_length)
    decoder_score = calculate_decode_score(request_length)
    request_id = next_req_id()

    failover_enabled = not args.disable_failover
    max_nodes = 1 if not failover_enabled else max(1, args.failover_max_retries)
    inner_retries = args.max_retries if not failover_enabled else 1
    excluded = set(excluded_prefillers) if excluded_prefillers else set()
    began = False  # did begin_request bump request_num?

    response = None
    prefiller_key: str | None = None
    for attempt in range(max_nodes + 1):
        # --- pick a prefiller (with exclusion on failover attempts) ---
        try:
            if attempt == 0:
                if is_initial_request:
                    prefiller = await runtime.schedule("begin_request", prefiller_score)
                    began = True
                else:
                    prefiller = await runtime.schedule("reserve_prefill_kv", prefiller_score)
            else:
                prefiller = await runtime.schedule(
                    "pick_prefiller_excluding", prefiller_score, list(excluded)
                )
        except RuntimeError:
            # No available prefiller in the pool (all unhealthy / circuit open).
            break
        prefiller_key = prefiller["key"]

        # --- send the prefill request ---
        try:
            response = await send_request_to_service(
                await runtime.get_client(ServerRole.PREFILL, prefiller_key),
                api,
                req_data,
                request_id,
                max_retries=inner_retries,
                base_delay=args.retry_delay,
            )
            await runtime.schedule("record_outcome", ServerRole.PREFILL, prefiller_key, True, 200)
            break  # success
        except (httpx.RequestError, httpx.HTTPStatusError) as exc:
            status_code = getattr(getattr(exc, "response", None), "status_code", None) or 0
            await runtime.schedule("record_outcome", ServerRole.PREFILL, prefiller_key, False, status_code)
            await runtime.schedule("release_prefill_kv", prefiller_key, prefiller_score)
            response = None
            if not failover_enabled:
                prefiller_key = None
                break
            excluded.add(prefiller_key)
            await runtime.schedule("record_failover", "prefill")
            await runtime.schedule(
                "record_prefill_error", prefiller_key, f"http{status_code}" if status_code else "network"
            )
            logger.warning(
                "[failover] prefiller %s failed (%s); trying another", prefiller_key,
                f"http{status_code}" if status_code else "network",
            )
            prefiller_key = None
            continue
        except Exception as exc:
            await runtime.schedule("record_outcome", ServerRole.PREFILL, prefiller_key, False, 0)
            await runtime.schedule("release_prefill_kv", prefiller_key, prefiller_score)
            response = None
            if not failover_enabled:
                prefiller_key = None
                break
            excluded.add(prefiller_key)
            await runtime.schedule("record_failover", "prefill")
            await runtime.schedule("record_prefill_error", prefiller_key, str(exc)[:40])
            logger.warning("[failover] prefiller %s failed (%s); trying another", prefiller_key, str(exc)[:40])
            prefiller_key = None
            continue

    if response is None:
        # All prefillers exhausted: undo the request_num bump from begin_request.
        if began:
            await runtime.schedule("finish_request", None, 0.0, None, 0.0, False)
        if excluded:
            raise NoAvailableNodeError(f"prefill failed on all tried prefillers {sorted(excluded)}")
        raise NoAvailableNodeError("no available prefiller")

    response_json = response.json()
    kv_transfer_params = response_json.get("kv_transfer_params", {})
    if kv_transfer_params:
        req_data["kv_transfer_params"] = kv_transfer_params
    prefiller_cached_tokens = extract_cached_tokens(response_json)

    try:
        decoder = await runtime.schedule("pick_decoder", decoder_score)
    except RuntimeError:
        # No available decoder.
        await runtime.schedule("release_prefill_kv", prefiller_key, prefiller_score)
        if began:
            await runtime.schedule("finish_request", None, 0.0, None, 0.0, False)
        raise NoAvailableNodeError("no available decoder")

    prefiller_client = await runtime.get_client(ServerRole.PREFILL, prefiller_key)
    decoder_client = await runtime.get_client(ServerRole.DECODE, decoder["key"])
    logger.debug("Using %s %s", prefiller_client.base_url, decoder_client.base_url)
    return InstanceInfo(
        request_id=request_id,
        prefiller_key=prefiller_key,
        prefiller_score=prefiller_score,
        decoder_key=decoder["key"],
        decoder_score=decoder_score,
        decoder_host=decoder["host"],
        decoder_port=decoder["port"],
        prefiller_cached_tokens=prefiller_cached_tokens,
    )


async def reassign_instances(
    api: str,
    req_data: Any,
    request_length: int,
    previous_instance: InstanceInfo,
) -> InstanceInfo:
    runtime = get_runtime()
    await runtime.schedule("release_prefill_kv", previous_instance.prefiller_key, previous_instance.prefiller_score)
    await runtime.schedule("release_decoder", previous_instance.decoder_key, previous_instance.decoder_score)
    return await assign_instances(api, req_data, request_length, is_initial_request=False)


async def handle_completions_impl(api: str, request: Request):
    runtime = get_runtime()
    args = get_global_args()
    request_released = False
    try:
        await runtime.schedule("record_request")
        req_data = await request.json()
        req_body = await request.body()
        request_length = len(req_body)
        instance_info = await assign_instances(api, req_data, request_length, is_initial_request=True)
        stream_flag = bool(req_data.get("stream", False))
        chat_flag = "messages" in req_data

        if "prompt" in req_data:
            origin_prompt = req_data["prompt"]
        elif chat_flag:
            messages = req_data["messages"]
            origin_prompt = messages[0].get("content", "")
        else:
            origin_prompt = ""
        origin_max_tokens = req_data.get("max_tokens", 16)

        async def generate_stream():
            nonlocal instance_info
            nonlocal request_released
            generated_token = ""
            released_kv = False
            retry_count = 0
            completion_tokens = 0
            reported_prefiller_cached_tokens = instance_info.prefiller_cached_tokens
            # Per-request failover state for the decode leg.
            failover_enabled = not args.disable_failover
            decode_retries = args.max_retries if not failover_enabled else 1
            excluded_decoders: set[str] = set()

            async def release_prefill_kv_once() -> None:
                nonlocal released_kv
                if not released_kv:
                    await runtime.schedule(
                        "release_prefill_kv", instance_info.prefiller_key, instance_info.prefiller_score
                    )
                    released_kv = True

            try:
                # Outer loop: decode-failover. The inner `while retry` loop
                # handles the recomputed-triggered re-prefill path.
                while True:
                    retry = True
                    try:
                        while retry:
                            retry = False
                            decoder_client = await runtime.get_client(
                                ServerRole.DECODE, instance_info.decoder_key
                            )
                            async for chunk in stream_service_response_with_retry(
                                decoder_client,
                                api,
                                req_data,
                                request_id=instance_info.request_id,
                                max_retries=decode_retries,
                                base_delay=args.retry_delay,
                            ):
                                if not released_kv and chunk:
                                    await release_prefill_kv_once()
                                try:
                                    chunk_str = chunk.decode("utf-8").strip()
                                except UnicodeDecodeError:
                                    logger.debug("Skipping chunk: %s", chunk)
                                    yield chunk
                                    continue
                                if not chunk_str:
                                    continue
                                is_sse = chunk_str.startswith("data: ")
                                if is_sse:
                                    chunk_str = chunk_str[len("data: ") :]
                                try:
                                    chunk_json = json.loads(chunk_str)
                                except json.JSONDecodeError:
                                    logger.debug("Skipping chunk: %s", chunk_str)
                                    yield chunk
                                    continue
                                choices = chunk_json.get("choices", [])
                                if not choices:
                                    if update_cached_tokens_in_chunk(
                                        chunk_json, reported_prefiller_cached_tokens
                                    ):
                                        chunk = encode_response_chunk(chunk_json, is_sse)
                                    yield chunk
                                    continue

                                choice = choices[0]
                                delta = choice.get("delta") or {}
                                message = choice.get("message") or {}
                                content = (
                                    delta.get("content")
                                    or message.get("content")
                                    or choice.get("text")
                                    or ""
                                )
                                generated_token += content

                                stop_reason = choice.get("stop_reason")
                                usage = chunk_json.get("usage", {})
                                completion_tokens = (
                                    (completion_tokens + 1)
                                    if stream_flag
                                    else (completion_tokens + usage.get("completion_tokens", 0))
                                )
                                if stop_reason == "recomputed":
                                    retry = True
                                    retry_count += 1
                                    if chat_flag:
                                        messages[0]["content"] = origin_prompt + generated_token
                                    else:
                                        req_data["prompt"] = origin_prompt + generated_token
                                    req_data["max_tokens"] = (
                                        origin_max_tokens - completion_tokens + retry_count
                                    )
                                    tmp_request_length = len(json.dumps(req_data).encode("utf-8"))
                                    instance_info = await reassign_instances(
                                        api, req_data, tmp_request_length, instance_info
                                    )
                                    released_kv = False
                                    break
                                if retry_count > 0 and not stream_flag:
                                    if chat_flag:
                                        choice["message"]["content"] = generated_token
                                    else:
                                        choice["text"] = generated_token
                                    chunk = encode_response_chunk(chunk_json, is_sse)
                                yield chunk
                        # Decode stream completed normally.
                        await runtime.schedule(
                            "record_outcome", ServerRole.DECODE, instance_info.decoder_key, True, 200
                        )
                        break
                    except DecodeLateError as exc:
                        # Chunks already sent to the client -> cannot fail over.
                        await runtime.schedule(
                            "record_outcome",
                            ServerRole.DECODE,
                            instance_info.decoder_key,
                            False,
                            exc.status_code or 0,
                        )
                        await runtime.schedule(
                            "record_decode_error", instance_info.decoder_key, "late"
                        )
                        logger.error(
                            "[decode] late failure from %s:%s: %s; stream truncated",
                            instance_info.decoder_host,
                            instance_info.decoder_port,
                            exc,
                        )
                        raise
                    except DecodeEarlyError as exc:
                        # No chunk reached the client yet -> failover-able.
                        await runtime.schedule(
                            "record_outcome",
                            ServerRole.DECODE,
                            instance_info.decoder_key,
                            False,
                            exc.status_code or 0,
                        )
                        await runtime.schedule(
                            "release_decoder", instance_info.decoder_key, instance_info.decoder_score
                        )
                        if not failover_enabled:
                            await runtime.schedule(
                                "record_decode_error", instance_info.decoder_key, "early"
                            )
                            logger.error(
                                "[decode] early failure from %s:%s: %s; failover disabled",
                                instance_info.decoder_host,
                                instance_info.decoder_port,
                                exc,
                            )
                            raise
                        excluded_decoders.add(instance_info.decoder_key)
                        await runtime.schedule(
                            "record_decode_error", instance_info.decoder_key, "early"
                        )
                        await runtime.schedule("record_failover", "decode")
                        if len(excluded_decoders) > args.failover_max_decoders:
                            logger.error(
                                "[decode] failover exhausted, excluded=%s",
                                sorted(excluded_decoders),
                            )
                            raise NoAvailableNodeError("decode failover exhausted")
                        logger.warning(
                            "[decode] early failure from %s:%s (%s); trying another decoder",
                            instance_info.decoder_host,
                            instance_info.decoder_port,
                            f"http{exc.status_code}" if exc.status_code else "network",
                        )
                        # If the prefiller is still available, reuse its KV and
                        # only swap the decoder; otherwise re-prefill on a
                        # fresh prefiller/decoder pair (KV lost).
                        prefiller_ok = await runtime.schedule(
                            "is_node_available", ServerRole.PREFILL, instance_info.prefiller_key
                        )
                        if prefiller_ok:
                            try:
                                new_decoder = await runtime.schedule(
                                    "pick_decoder_excluding",
                                    instance_info.decoder_score,
                                    list(excluded_decoders),
                                )
                            except RuntimeError:
                                raise NoAvailableNodeError("no available decoder after failover")
                            instance_info = replace(
                                instance_info,
                                decoder_key=new_decoder["key"],
                                decoder_host=new_decoder["host"],
                                decoder_port=new_decoder["port"],
                            )
                            # released_kv stays as-is: KV release is per-prefiller.
                        else:
                            tmp_request_length = len(json.dumps(req_data).encode("utf-8"))
                            instance_info = await reassign_instances(
                                api, req_data, tmp_request_length, instance_info
                            )
                            released_kv = False
                        continue
            except asyncio.CancelledError:
                logger.warning(
                    "Streaming from decoder %s:%s was cancelled; releasing request %s resources",
                    instance_info.decoder_host,
                    instance_info.decoder_port,
                    instance_info.request_id,
                )
                raise
            except NoAvailableNodeError:
                # Re-raise so DeferredStreamingResponse surfaces a 503.
                raise
            except Exception as exc:
                logger.error(
                    "Error during streaming from decoder %s:%s: %s while handling request %s; "
                    "releasing prefiller KV",
                    instance_info.decoder_host,
                    instance_info.decoder_port,
                    exc,
                    instance_info.request_id,
                )
                raise
            finally:
                await _finish_instance(runtime, instance_info, release_prefill_kv=not released_kv)
                released_kv = True
                request_released = True

        media_type = "text/event-stream; charset=utf-8" if stream_flag else "application/json"
        # DeferredStreamingResponse delays committing the status line until the
        # first chunk, so a decode failure before any chunk surfaces as a 503
        # instead of a silent 200 + empty body.
        return DeferredStreamingResponse(
            generate_stream(),
            media_type=media_type,
            error_body=b'{"error":{"message":"upstream decode failed","type":"decode_failed"}}',
        )
    except NoAvailableNodeError as e:
        logger.error("No available node for %s: %s", api, e)
        return JSONResponse(
            status_code=503,
            content={
                "error": {
                    "message": f"no available backend: {e}",
                    "type": "no_available_node",
                }
            },
        )
    except Exception:
        import traceback

        exc_info = sys.exc_info()
        print(f"Error occurred in disagg prefill proxy server - {api} endpoint")
        print("".join(traceback.format_exception(*exc_info)))
        if not request_released and "instance_info" in locals():
            await _finish_instance(runtime, instance_info, release_prefill_kv=True)
            request_released = True
        raise


async def adjust_instances_impl(adjust_mode: str, request: Request):
    req_data = await request.json()
    instance_type = req_data.get("type", "")
    instances = req_data.get("instances", [])
    if isinstance(instances, str):
        instances = [instances]
    parsed_instances = parse_server_addresses(instances)
    all_msg = f"{adjust_mode} {instance_type} instances: {[f'{host}:{port}' for host, port in parsed_instances]}."

    try:
        role = ServerRole(instance_type)
    except ValueError:
        return {
            "error": (
                f"Instance type {instance_type!r} is not supported. "
                f"Only '{ServerRole.PREFILL.value}' and '{ServerRole.DECODE.value}' are allowed."
            )
        }

    scheduler = get_runtime().scheduler

    if adjust_mode == "add":
        waiting_nodes = scheduler.add_instances(role, parsed_instances)
        if waiting_nodes:
            all_msg = f"Instances {waiting_nodes} are waiting to be added."
    elif adjust_mode == "remove":
        need_waiting = scheduler.remove_instances(role, parsed_instances)
        if need_waiting:
            all_msg = (
                f"Instances {[f'{host}:{port}' for host, port in parsed_instances]} "
                "are isolated and waiting to be removed."
            )

    snapshot = scheduler.get_snapshot()
    return {
        "message": all_msg,
        "current_prefill_instances": [f"{server['host']}:{server['port']}" for server in snapshot["prefill_instances"]],
        "current_decode_instances": [f"{server['host']}:{server['port']}" for server in snapshot["decode_instances"]],
    }


def parse_server_addresses(instances: list[str]) -> list[tuple[str, int]]:
    return [(host, int(port)) for host, port in (instance.split(":") for instance in instances)]


@app.post("/v1/completions")
@with_cancellation
async def handle_completions(request: Request):
    return await handle_completions_impl("/completions", request)


@app.post("/v1/chat/completions")
@with_cancellation
async def handle_chat_completions(request: Request):
    return await handle_completions_impl("/chat/completions", request)


@app.post("/reset_prefix_cache")
async def reset_prefix_cache(request: Request):
    params = dict(request.query_params)
    runtime = get_runtime()
    await runtime.sync_clients()
    snapshot = runtime.scheduler.get_snapshot()
    backend_instances = [(ServerRole.PREFILL, server) for server in snapshot["prefill_instances"]] + [
        (ServerRole.DECODE, server) for server in snapshot["decode_instances"]
    ]
    failures: list[str] = []
    for role, server in backend_instances:
        base_url = build_server_url(server["host"], server["port"])
        try:
            client = await runtime.get_client(role, server_key(server["host"], server["port"]))
            resp = await client.post(f"{base_url}/reset_prefix_cache", params=params)
            resp.raise_for_status()
        except Exception as e:
            logger.error("reset_prefix_cache failed for %s: %s", base_url, e)
            failures.append(base_url)
    if failures:
        return JSONResponse(status_code=500, content={"failed": failures})
    return Response(status_code=200)


@app.get("/metrics")
async def metrics():
    return Response(
        content=get_runtime().scheduler.render_metrics(),
        media_type="text/plain; version=0.0.4",
    )


@app.get("/healthcheck")
async def healthcheck():
    return get_runtime().scheduler.healthcheck()


@app.post("/instances/add")
async def handle_add_instances(request: Request):
    return await adjust_instances_impl("add", request)


@app.post("/instances/remove")
async def handle_remove_instances(request: Request):
    return await adjust_instances_impl("remove", request)


if __name__ == "__main__":
    global_args = parse_args()
    setup_logging(global_args.log_level)
    bootstrap_parent_process(global_args)
    import uvicorn

    module_name = Path(__file__).stem
    try:
        uvicorn.run(
            f"{module_name}:create_app",
            host=global_args.host,
            port=global_args.port,
            workers=global_args.workers,
            factory=True,
            app_dir=str(Path(__file__).resolve().parent),
        )
    finally:
        cleanup_manager_config(global_args.port)
