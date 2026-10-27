# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""P/D proxy with session-affine Decoder-local Append-Prefill.

Run this standalone example with httpx, fastapi, anyio and uvicorn installed::

    python decoder_append_prefill_proxy_server.py \
        --host 127.0.0.1 --port 9000 --workers 1 \
        --prefiller-hosts 127.0.0.1 --prefiller-ports 8100 \
        --decoder-hosts 127.0.0.1 --decoder-ports 8200 \
        --enable-decoder-append-prefill --decoder-data-parallel-size 2

Send a stable X-Session-ID on each turn. The feature is disabled by default.
When enabled, cold sessions use P/D and warm sessions use their pinned Decoder.
Enable scheduler_config.decoder_append_prefill_enable in each Decoder's
--additional-config and enable prefix caching. The platform selects the matching
append-prefill admission Scheduler; see the Decoder Append-Prefill feature guide.

Decoder-local routing requires Decoder prefix caching and backend support for
vllm_xargs.require_decoder_cache_hit: reject a zero-hit request before prefill
with stop_reason=decoder_cache_miss and no generated output. Streaming backends
must preserve this zero-token terminal event. The proxy then retries through
P/D with the original history and generation budget. Cache loss after output
must report decoder_cache_lost_after_output; it must never replay the request.
Session affinity alone does not prove that a prefix remains resident.
Decoder-local append-prefill is currently unsupported with DyntraLB, which
requires Decoder instances to execute decode-only work.

The operator must change --decoder-engine-epoch and restart this proxy after
restarting Decoders. Each Decoder endpoint's DP size must match the CLI setting.
Single-worker operation is the validated configuration; the inherited shared
scheduler supports multiple workers but needs separate integration validation.
SSE framing and cancellation cleanup are built in; no experiment modules or
runtime function replacement are needed. Based on the Ascend P/D proxy example
and vLLM's nixl_integration/toy_proxy_server.py.
"""

import argparse
import asyncio
import base64
import functools
import json
import logging
import os
import sys
import threading
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from enum import Enum
from multiprocessing.managers import BaseManager
from pathlib import Path
from typing import Any, cast

import anyio
import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

if __package__:
    from examples.disaggregated_prefill_v1 import load_balance_proxy_server_example as base_proxy
    from examples.disaggregated_prefill_v1.load_balance_proxy_server_example import ServerRole, WorkerRuntime
else:
    import load_balance_proxy_server_example as base_proxy
    from load_balance_proxy_server_example import ServerRole, WorkerRuntime

# Share stateless helpers and runtime types, while retaining this entry point's
# own app, arguments and cross-worker scheduler state.
extract_cached_tokens = base_proxy.extract_cached_tokens
update_cached_tokens_in_chunk = base_proxy.update_cached_tokens_in_chunk
encode_response_chunk = base_proxy.encode_response_chunk
next_req_id = base_proxy.next_req_id
calculate_prefill_score = base_proxy.calculate_prefill_score
calculate_decode_score = base_proxy.calculate_decode_score
server_key = base_proxy.server_key
build_server_url = base_proxy.build_server_url
build_base_url = base_proxy.build_base_url
manager_config_path = base_proxy.manager_config_path
write_manager_config = base_proxy.write_manager_config
read_manager_config = base_proxy.read_manager_config
cleanup_manager_config = base_proxy.cleanup_manager_config
listen_for_disconnect = base_proxy.listen_for_disconnect
auth_headers = base_proxy.auth_headers
send_request_to_service = base_proxy.send_request_to_service
_abort_prefill_selection = base_proxy._abort_prefill_selection
parse_server_addresses = base_proxy.parse_server_addresses

logger = logging.getLogger(__name__)


class SSEFramer:
    """Reassemble OpenAI JSON SSE events independently of HTTP chunk boundaries."""

    def __init__(self, limit: int = 8 * 1024 * 1024) -> None:
        if limit <= 0:
            raise ValueError("SSE frame size limit must be positive")
        self.buffer = b""
        self.data: list[bytes] = []
        self.done = False
        self.limit = limit
        self._event_size = 0

    def feed(self, chunk: bytes) -> list[bytes]:
        self.buffer += chunk
        frames = []
        while b"\n" in self.buffer:
            line, self.buffer = self.buffer.split(b"\n", 1)
            self._event_size += len(line) + 1
            if self._event_size > self.limit:
                raise ValueError("SSE event exceeds frame size limit")
            line = line.removesuffix(b"\r")
            if not line:
                self._event_size = 0
                if not self.data:
                    continue
                payload = b"\n".join(self.data)
                self.data.clear()
                if self.done:
                    raise ValueError("SSE data received after DONE")
                if payload == b"[DONE]":
                    self.done = True
                    frames.append(b"data: [DONE]\n\n")
                else:
                    obj = json.loads(payload.decode("utf-8"))
                    if not isinstance(obj, dict):
                        raise ValueError("SSE JSON event must be an object")
                    frames.append(b"data: " + json.dumps(obj, ensure_ascii=False).encode() + b"\n\n")
            elif line.startswith(b"data:"):
                self.data.append(line[5:].removeprefix(b" "))
        if self._event_size + len(self.buffer) > self.limit:
            raise ValueError("SSE event exceeds frame size limit")
        return frames

    def finish(self) -> None:
        if self.buffer.strip() or self.data or not self.done:
            raise ValueError("Truncated SSE stream or missing DONE")


class RouteMode(str, Enum):
    PD = "pd"
    DECODER_LOCAL = "decoder_local"


class SessionState(str, Enum):
    READY = "ready"
    IN_FLIGHT = "in_flight"


@dataclass
class InstanceInfo:
    route_mode: RouteMode
    route_reason: str
    request_id: str
    session_id: str | None
    decoder_key: str
    decoder_score: float
    decoder_host: str
    decoder_port: int
    decoder_dp_rank: int = 0
    decoder_engine_epoch: str = ""
    prefiller_key: str | None = None
    prefiller_score: float = 0.0
    prefiller_cached_tokens: int | None = None


@dataclass
class RoutePlan:
    mode: RouteMode
    reason: str
    session_id: str | None = None
    pinned_decoder_key: str | None = None
    pinned_decoder_dp_rank: int | None = None
    session_reserved: bool = False


class SessionBusyError(RuntimeError):
    pass


class BackendContractError(RuntimeError):
    """The backend violated the routing contract; do not count this response as success."""

    def __init__(self, message: str, code: str):
        super().__init__(message)
        self.code = code


SESSION_ID_HEADER = "X-Session-ID"
DP_RANK_HEADER = "X-data-parallel-rank"


global_args: argparse.Namespace | None = None
shared_scheduler: "SharedProxyScheduler | None" = None
runtime: "WorkerRuntime | None" = None


@dataclass
class SessionAffinity:
    decoder_key: str | None
    turn_count: int
    last_access_time: float
    state: SessionState
    decoder_dp_rank: int | None = None
    decoder_engine_epoch: str = ""
    cache_generation: int = 0


def setup_logging(log_level: str) -> None:
    base_proxy.setup_logging(log_level)
    logger.setLevel(getattr(logging, log_level.upper()))


class SharedProxyScheduler(base_proxy.SharedProxyScheduler):
    """Add session ownership to the shared P/D proxy's load scheduler."""

    def __init__(
        self,
        prefiller_instances,
        decoder_instances,
        decoder_data_parallel_size: int = 1,
        decoder_engine_epoch: str | None = None,
    ):
        if decoder_data_parallel_size < 1:
            raise ValueError("Decoder data parallel size must be greater than zero")
        super().__init__(prefiller_instances, decoder_instances)
        self.decoder_data_parallel_size = decoder_data_parallel_size
        # An operator-supplied deployment label, NOT a backend-validated lease.
        # Affinities live only in this scheduler and are empty on proxy restart.
        self.decoder_engine_epoch = decoder_engine_epoch or str(uuid.uuid4())
        self._next_decoder_dp_rank: dict[str, int] = {}
        self.session_affinities: dict[str, SessionAffinity] = {}

    def healthcheck(self) -> dict[str, Any]:
        with self._lock:
            return {
                **super().healthcheck(),
                "active_sessions": len(self.session_affinities),
                "decoder_data_parallel_size": self.decoder_data_parallel_size,
                "decoder_engine_epoch": self.decoder_engine_epoch,
                "affinity_kind": "proxy_hint_no_backend_lease",
                "in_flight_sessions": sum(
                    affinity.state is SessionState.IN_FLIGHT for affinity in self.session_affinities.values()
                ),
            }

    def pick_decoder(self, load: float) -> dict[str, Any]:
        with self._lock:
            decoder = super().pick_decoder(load)
            return self._with_decoder_owner(decoder, None)

    def _with_decoder_owner(self, decoder: dict[str, Any], dp_rank: int | None) -> dict[str, Any]:
        key = decoder["key"]
        if dp_rank is None:
            dp_rank = self._next_decoder_dp_rank.get(key, 0)
            self._next_decoder_dp_rank[key] = (dp_rank + 1) % self.decoder_data_parallel_size
        decoder["dp_rank"] = dp_rank
        decoder["engine_epoch"] = self.decoder_engine_epoch
        return decoder

    def _validate_decoder_dp_rank(self, dp_rank: int) -> None:
        if not 0 <= dp_rank < self.decoder_data_parallel_size:
            raise ValueError(
                f"Decoder DP rank {dp_rank} is outside configured DP size {self.decoder_data_parallel_size}"
            )

    def reserve_decoder(self, key: str, load: float, dp_rank: int | None = None) -> dict[str, Any]:
        """Reserve load on a specific, non-tainted Decoder."""
        with self._lock:
            if dp_rank is not None:
                self._validate_decoder_dp_rank(dp_rank)
            pool = self._pool(ServerRole.DECODE)
            if key not in pool.servers or key in pool.tainted:
                raise RuntimeError(f"Decoder {key} is unavailable")
            entry = pool.servers[key]
            entry.active_tokens += load
            self._push_heap(ServerRole.DECODE, key)
            return self._with_decoder_owner({"key": key, "host": entry.host, "port": entry.port}, dp_rank)

    def begin_decoder_request(self, key: str, load: float, dp_rank: int | None = None) -> dict[str, Any]:
        """Start a D-local request and reserve its pinned Decoder."""
        with self._lock:
            decoder = self.reserve_decoder(key, load, dp_rank)
            self.request_num += 1
            return decoder

    def reserve_session(self, session_id: str, ttl_seconds: float) -> dict[str, Any]:
        """Atomically reserve a session for one mutating request.

        The returned affinity only predicts where prefix KV may exist. It is
        not proof of a prefix-cache hit; the Decoder Scheduler remains the
        authority for actual matched tokens.
        """
        with self._lock:
            now = time.monotonic()
            affinity = self.session_affinities.get(session_id)
            if affinity is not None and affinity.state is SessionState.IN_FLIGHT:
                return {"status": "busy"}
            if affinity is not None and now - affinity.last_access_time >= ttl_seconds:
                self.session_affinities.pop(session_id, None)
                affinity = None

            decoder_pool = self._pool(ServerRole.DECODE)
            if affinity is not None and (
                affinity.decoder_key not in decoder_pool.servers
                or affinity.decoder_key in decoder_pool.tainted
                or affinity.decoder_engine_epoch != self.decoder_engine_epoch
            ):
                self.session_affinities.pop(session_id, None)
                affinity = None

            if affinity is None:
                self.session_affinities[session_id] = SessionAffinity(
                    decoder_key=None,
                    turn_count=0,
                    last_access_time=now,
                    state=SessionState.IN_FLIGHT,
                    decoder_engine_epoch=self.decoder_engine_epoch,
                )
                return {"status": "cold", "decoder_key": None, "turn_count": 0}

            affinity.state = SessionState.IN_FLIGHT
            affinity.last_access_time = now
            return {
                "status": "warm",
                "decoder_key": affinity.decoder_key,
                "turn_count": affinity.turn_count,
                "decoder_dp_rank": affinity.decoder_dp_rank,
                "decoder_engine_epoch": affinity.decoder_engine_epoch,
            }

    def bind_session_decoder(self, session_id: str, decoder_key: str, dp_rank: int = 0) -> None:
        """Bind an accepted request to its Decoder and rank; retain proxy serialization."""
        with self._lock:
            self._validate_decoder_dp_rank(dp_rank)
            affinity = self.session_affinities.get(session_id)
            if affinity is None or affinity.state is not SessionState.IN_FLIGHT:
                raise RuntimeError(f"Session {session_id} is not reserved")
            pool = self._pool(ServerRole.DECODE)
            if decoder_key not in pool.servers or decoder_key in pool.tainted:
                raise RuntimeError(f"Decoder {decoder_key} is unavailable")
            affinity.decoder_key = decoder_key
            affinity.decoder_dp_rank = dp_rank
            affinity.decoder_engine_epoch = self.decoder_engine_epoch
            affinity.last_access_time = time.monotonic()

    def reset_session_binding(self, session_id: str) -> None:
        """Clear a stale Decoder binding while preserving proxy serialization."""
        with self._lock:
            affinity = self.session_affinities.get(session_id)
            if affinity is not None and affinity.state is SessionState.IN_FLIGHT:
                affinity.decoder_key = None
                affinity.decoder_dp_rank = None
                affinity.cache_generation += 1

    def finish_session(self, session_id: str, success: bool) -> None:
        """Commit or roll back a reserved session after response completion."""
        with self._lock:
            affinity = self.session_affinities.get(session_id)
            if affinity is None:
                return
            decoder_pool = self._pool(ServerRole.DECODE)
            decoder_available = (
                affinity.decoder_key in decoder_pool.servers and affinity.decoder_key not in decoder_pool.tainted
            )
            if not decoder_available:
                self.session_affinities.pop(session_id, None)
                return
            if success:
                affinity.turn_count += 1
            elif affinity.turn_count == 0:
                self.session_affinities.pop(session_id, None)
                return
            affinity.state = SessionState.READY
            affinity.last_access_time = time.monotonic()

    def invalidate_session(self, session_id: str) -> None:
        with self._lock:
            self.session_affinities.pop(session_id, None)

    def invalidate_all_sessions(self) -> None:
        with self._lock:
            self.session_affinities.clear()

    def _invalidate_decoder_sessions_no_lock(self, decoder_keys: set[str]) -> None:
        for session_id, affinity in list(self.session_affinities.items()):
            if affinity.decoder_key in decoder_keys:
                self.session_affinities.pop(session_id, None)

    def remove_instances(self, role: ServerRole, instances: list[tuple[str, int]]) -> bool:
        if not instances:
            return False
        keys = {server_key(host, port) for host, port in instances}
        with self._lock:
            if role is ServerRole.DECODE:
                self._invalidate_decoder_sessions_no_lock(keys)
            return super().remove_instances(role, instances)


class SchedulerManager(BaseManager):
    """Multiprocessing RPC bridge; body is empty but required by BaseManager."""


def _shared_scheduler_proxy() -> "SharedProxyScheduler":
    if shared_scheduler is None:
        raise RuntimeError("shared scheduler is not initialized")
    return shared_scheduler


SchedulerManager.register("get_scheduler", callable=_shared_scheduler_proxy)


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
            time.sleep(args.waiting_retry_interval)

    @staticmethod
    async def check_instance_status(host: str, port: int) -> bool:
        endpoint = "/models"
        headers = {"Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}"}
        try:
            async with httpx.AsyncClient(timeout=5.0, base_url=build_base_url(host, port)) as client:
                response = await client.get(endpoint, headers=headers)
                response.raise_for_status()
                return True
        except (httpx.RequestError, httpx.HTTPStatusError):
            return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--prefiller-hosts", type=str, nargs="+", default=["localhost"])
    parser.add_argument("--prefiller-ports", type=int, nargs="+", default=[8001])
    parser.add_argument("--decoder-hosts", type=str, nargs="+", default=["localhost"])
    parser.add_argument("--decoder-ports", type=int, nargs="+", default=[8002])
    parser.add_argument(
        "--decoder-data-parallel-size",
        type=int,
        default=1,
        help="Internal attention DP size of each Decoder endpoint.",
    )
    parser.add_argument(
        "--decoder-engine-epoch",
        type=str,
        default=None,
        help=(
            "Operator-provided Decoder deployment label; change it and restart the proxy after a backend restart. "
            "A random label is used when omitted. This does not validate backend cache residency or lease."
        ),
    )
    parser.add_argument(
        "--disable-recomputed-retry",
        action="store_true",
        help=(
            "Reject Decoder recomputation or missing Prefiller KV-transfer metadata. "
            "Non-streaming requests return HTTP 502 instead of silently performing fallback work."
        ),
    )
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
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Log level for the proxy server.",
    )
    parser.add_argument(
        "--enable-decoder-append-prefill",
        action="store_true",
        help="Route warm sessions to their pinned Decoder; cold sessions and cache misses use P/D.",
    )
    parser.add_argument(
        "--session-affinity-ttl",
        type=float,
        default=3600.0,
        help="Seconds before an idle session-to-Decoder affinity expires.",
    )
    args = parser.parse_args()
    if len(args.prefiller_hosts) != len(args.prefiller_ports):
        raise ValueError("Number of prefiller hosts must match number of prefiller ports")
    if len(args.decoder_hosts) != len(args.decoder_ports):
        raise ValueError("Number of decoder hosts must match number of decoder ports")
    if args.session_affinity_ttl <= 0:
        raise ValueError("Session affinity TTL must be greater than zero")
    if args.decoder_data_parallel_size <= 0:
        raise ValueError("Decoder data parallel size must be greater than zero")
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

    shared_scheduler = SharedProxyScheduler(
        args.prefiller_instances,
        args.decoder_instances,
        args.decoder_data_parallel_size,
        args.decoder_engine_epoch,
    )
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
    shared_scheduler = SharedProxyScheduler(
        args.prefiller_instances,
        args.decoder_instances,
        args.decoder_data_parallel_size,
        args.decoder_engine_epoch,
    )
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
    try:
        await runtime.sync_clients()
        snapshot = scheduler.get_snapshot()
        logger.info(
            "Initialized %s prefill clients and %s decode clients in worker %s.",
            len(snapshot["prefill_instances"]),
            len(snapshot["decode_instances"]),
            os.getpid(),
        )
        yield
    finally:
        await runtime.close()
        runtime = None


app = FastAPI(lifespan=lifespan)


def create_app():
    setup_logging(get_global_args().log_level)
    return app


def with_cancellation(handler_func):
    @functools.wraps(handler_func)
    async def wrapper(*args, **kwargs):
        request = kwargs["request"]
        handler_task = asyncio.create_task(handler_func(*args, **kwargs))
        cancellation_task = asyncio.create_task(listen_for_disconnect(request))
        tasks = (handler_task, cancellation_task)
        delivered = False
        try:
            done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for task in pending:
                task.cancel()
            with anyio.CancelScope(shield=True):
                await asyncio.gather(*pending, return_exceptions=True)
            if handler_task in done:
                result = handler_task.result()
                delivered = True
                return result
            cancellation_task.result()
            return None
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            with anyio.CancelScope(shield=True):
                await asyncio.gather(*tasks, return_exceptions=True)
                if not delivered and handler_task.done() and not handler_task.cancelled():
                    if handler_task.exception() is None:
                        result = handler_task.result()
                        if isinstance(result, CleanupStreamingResponse):
                            try:
                                await result.body_iterator.aclose()
                            finally:
                                await result.finalize()

    return wrapper


def decoder_headers(request_id: str, decoder_dp_rank: int) -> dict[str, str]:
    """Create trusted backend headers; no incoming client rank/header is forwarded."""
    return {**auth_headers(request_id), DP_RANK_HEADER: str(decoder_dp_rank)}


def routing_response_headers(info: InstanceInfo) -> dict[str, str]:
    return {
        "X-Request-Id": info.request_id,
        "X-Proxy-Route": info.route_mode.value,
        "X-Proxy-Route-Reason": info.route_reason,
        "X-Decoder-Owner": info.decoder_key,
        "X-Decoder-DP-Rank": str(info.decoder_dp_rank),
        "X-Decoder-Engine-Epoch": info.decoder_engine_epoch,
    }


async def plan_request_route(request: Request, req_data: dict) -> RoutePlan:
    """Reserve session affinity and choose PD or Decoder-local routing."""
    args = get_global_args()
    body_session_id = req_data.pop("conversation_id", None)
    # Client-supplied transfer metadata can refer to expired P allocations. Only
    # assign_pd may attach the metadata returned by the current P request.
    req_data.pop("kv_transfer_params", None)
    # Admission is controlled by this proxy's chosen route, not by the client.
    if req_data.get("vllm_xargs") is not None:
        req_data["vllm_xargs"] = dict(req_data["vllm_xargs"])
        req_data["vllm_xargs"].pop("require_decoder_cache_hit", None)
    if not args.enable_decoder_append_prefill:
        return RoutePlan(RouteMode.PD, "feature_disabled")

    session_id = request.headers.get(SESSION_ID_HEADER)
    if not session_id and isinstance(body_session_id, str) and body_session_id:
        session_id = body_session_id
    if not session_id:
        return RoutePlan(RouteMode.PD, "no_session_id")

    reservation = await get_runtime().schedule("reserve_session", session_id, args.session_affinity_ttl)
    if reservation["status"] == "busy":
        raise SessionBusyError(f"Session {session_id} already has an in-flight request")

    if reservation["status"] == "cold":
        return RoutePlan(
            RouteMode.PD,
            "cold_session",
            session_id=session_id,
            session_reserved=True,
        )

    return RoutePlan(
        RouteMode.DECODER_LOCAL,
        "affinity_hit",
        session_id=session_id,
        pinned_decoder_key=reservation["decoder_key"],
        pinned_decoder_dp_rank=reservation["decoder_dp_rank"],
        session_reserved=True,
    )


async def stream_service_response(
    client: httpx.AsyncClient,
    endpoint: str,
    req_data: dict,
    request_id: str,
    decoder_dp_rank: int,
    max_retries: int = 3,
    base_delay: float = 0.2,
):
    headers = decoder_headers(request_id, decoder_dp_rank)
    max_attempts = max(1, max_retries)
    for attempt in range(1, max_attempts + 1):
        first_chunk_sent = False
        try:
            async with client.stream("POST", endpoint, json=req_data, headers=headers) as response:
                if response.status_code >= 400:
                    # Buffer the upstream error body so callers can forward the real
                    # cause (e.g. vLLM's "maximum context length" message) instead of
                    # dropping it into an empty 200. The body is unreadable once this
                    # streaming context exits, so read it before raising.
                    await response.aread()
                    response.raise_for_status()
                if not req_data.get("stream", False):
                    # A non-streaming JSON document may cross HTTP chunk
                    # boundaries. Buffer it before committing proxy headers so
                    # route metrics and fail-closed checks see a whole document.
                    body = await response.aread()
                    first_chunk_sent = True
                    yield body
                    return
                framer = SSEFramer()
                async for chunk in response.aiter_bytes():
                    # Never retry after receiving bytes, even an incomplete event.
                    # The backend may already have generated output at that point.
                    first_chunk_sent = first_chunk_sent or bool(chunk)
                    for frame in framer.feed(chunk):
                        yield frame
                framer.finish()
                return
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 400 or attempt == max_attempts:
                raise
            logger.warning("Attempt %s failed for streaming %s: %s", attempt, endpoint, exc)
        except httpx.RequestError as exc:
            if first_chunk_sent:
                logger.error("Streaming to client interrupted after response started: %s", exc)
                raise
            if attempt == max_attempts:
                raise
            logger.warning("Attempt %s failed for streaming %s: %s", attempt, endpoint, exc)
        except Exception as exc:
            if first_chunk_sent:
                logger.error("Streaming to client interrupted after response started: %s", exc)
                raise
            if attempt == max_attempts:
                raise
            logger.warning("Attempt %s failed for streaming %s: %s", attempt, endpoint, exc)
        await asyncio.sleep(base_delay * (2 ** (attempt - 1)))


async def _finish_instance(runtime: WorkerRuntime, info: InstanceInfo, *, release_prefill_kv: bool) -> None:
    await runtime.schedule(
        "finish_request",
        info.prefiller_key,
        info.prefiller_score,
        info.decoder_key,
        info.decoder_score,
        release_prefill_kv,
    )


async def _reserve_pd_decoder(
    runtime: WorkerRuntime,
    decoder_score: float,
    pinned_decoder_key: str | None,
    pinned_decoder_dp_rank: int | None,
    session_id: str | None,
) -> tuple[dict[str, Any], str]:
    if pinned_decoder_key is None:
        return await runtime.schedule("pick_decoder", decoder_score), "load_balanced_decoder"
    try:
        return await runtime.schedule(
            "reserve_decoder", pinned_decoder_key, decoder_score, pinned_decoder_dp_rank
        ), "pinned_decoder"
    except RuntimeError:
        logger.warning("Pinned Decoder %s is unavailable; rebinding session %s", pinned_decoder_key, session_id)
        if session_id is not None:
            await runtime.schedule("reset_session_binding", session_id)
        return await runtime.schedule("pick_decoder", decoder_score), "decoder_unavailable_rebind"


async def assign_pd(
    api: str,
    req_data: Any,
    request_length: int,
    *,
    is_initial_request: bool,
    session_id: str | None,
    pinned_decoder_key: str | None,
    pinned_decoder_dp_rank: int | None,
    route_reason: str,
) -> InstanceInfo:
    runtime = get_runtime()
    args = get_global_args()
    prefiller_score = calculate_prefill_score(request_length)
    decoder_score = calculate_decode_score(request_length)
    request_id = next_req_id()
    pick_prefill = "begin_request" if is_initial_request else "reserve_prefill_kv"
    prefiller = await runtime.schedule(pick_prefill, prefiller_score)
    prefiller_key = prefiller["key"]

    decoder = None
    try:
        response = await send_request_to_service(
            await runtime.get_client(ServerRole.PREFILL, prefiller_key),
            api,
            req_data,
            request_id,
            max_retries=args.max_retries,
            base_delay=args.retry_delay,
        )
        response_json = response.json()
        kv_transfer_params = response_json.get("kv_transfer_params", {})
        if (getattr(args, "disable_recomputed_retry", False) or route_reason == "decoder_cache_miss_fallback") and not (
            isinstance(kv_transfer_params, dict) and kv_transfer_params
        ):
            raise BackendContractError(
                "Prefiller returned no KV-transfer metadata; refusing an unverified PD decode leg",
                "missing_prefiller_kv_transfer",
            )
        req_data.pop("kv_transfer_params", None)
        if kv_transfer_params:
            req_data["kv_transfer_params"] = kv_transfer_params
        prefiller_cached_tokens = extract_cached_tokens(response_json)
        decoder, decoder_reason = await _reserve_pd_decoder(
            runtime,
            decoder_score,
            pinned_decoder_key,
            pinned_decoder_dp_rank,
            session_id,
        )
        # No cancellable operation between D reservation and ownership handoff.
        return InstanceInfo(
            route_mode=RouteMode.PD,
            route_reason=route_reason if decoder_reason != "decoder_unavailable_rebind" else decoder_reason,
            request_id=request_id,
            session_id=session_id,
            prefiller_key=prefiller_key,
            prefiller_score=prefiller_score,
            decoder_key=decoder["key"],
            decoder_score=decoder_score,
            decoder_host=decoder["host"],
            decoder_port=decoder["port"],
            decoder_dp_rank=decoder["dp_rank"],
            decoder_engine_epoch=decoder["engine_epoch"],
            prefiller_cached_tokens=prefiller_cached_tokens,
        )
    except (Exception, asyncio.CancelledError):
        with anyio.CancelScope(shield=True):
            if decoder is not None:
                await runtime.schedule("release_decoder", decoder["key"], decoder_score)
            await _abort_prefill_selection(
                runtime, prefiller_key, prefiller_score, is_initial_request=is_initial_request
            )
        raise


async def assign_decoder_local(
    api: str,
    req_data: dict,
    request_length: int,
    route: RoutePlan,
) -> InstanceInfo:
    del api
    if route.pinned_decoder_key is None or route.pinned_decoder_dp_rank is None:
        raise RuntimeError("Decoder-local route requires a pinned Decoder and DP rank")
    runtime = get_runtime()
    decoder_score = calculate_decode_score(request_length)
    decoder = await runtime.schedule(
        "begin_decoder_request", route.pinned_decoder_key, decoder_score, route.pinned_decoder_dp_rank
    )
    req_data.pop("kv_transfer_params", None)
    req_data["vllm_xargs"] = dict(req_data.get("vllm_xargs") or {})
    req_data["vllm_xargs"]["require_decoder_cache_hit"] = 1
    return InstanceInfo(
        route_mode=RouteMode.DECODER_LOCAL,
        route_reason=route.reason,
        request_id=next_req_id(),
        session_id=route.session_id,
        decoder_key=decoder["key"],
        decoder_score=decoder_score,
        decoder_host=decoder["host"],
        decoder_port=decoder["port"],
        decoder_dp_rank=decoder["dp_rank"],
        decoder_engine_epoch=decoder["engine_epoch"],
    )


async def assign_request(api: str, req_data: dict, request_length: int, route: RoutePlan) -> InstanceInfo:
    if route.mode is RouteMode.DECODER_LOCAL:
        try:
            return await assign_decoder_local(api, req_data, request_length, route)
        except BackendContractError:
            raise
        except RuntimeError:
            logger.warning(
                "Decoder-local route for session %s lost its pinned Decoder; falling back to PD",
                route.session_id,
            )
            if route.session_id is not None:
                await get_runtime().schedule("reset_session_binding", route.session_id)
            route.pinned_decoder_key = None
            route.pinned_decoder_dp_rank = None
            route.reason = "decoder_unavailable_fallback"
    return await assign_pd(
        api,
        req_data,
        request_length,
        is_initial_request=True,
        session_id=route.session_id,
        pinned_decoder_key=route.pinned_decoder_key,
        pinned_decoder_dp_rank=route.pinned_decoder_dp_rank,
        route_reason=route.reason,
    )


async def reassign_instances(
    api: str,
    req_data: Any,
    request_length: int,
    previous_instance: InstanceInfo,
    *,
    route_reason: str = "recomputed_fallback",
) -> InstanceInfo:
    runtime = get_runtime()
    if previous_instance.prefiller_key is not None:
        await runtime.schedule("release_prefill_kv", previous_instance.prefiller_key, previous_instance.prefiller_score)
        previous_instance.prefiller_key = None
    await runtime.schedule("release_decoder", previous_instance.decoder_key, previous_instance.decoder_score)
    # If P admission fails, the caller still owns the old InstanceInfo and
    # finishes its request count. Its D reservation has already been released.
    previous_instance.decoder_score = 0.0
    return await assign_pd(
        api,
        req_data,
        request_length,
        is_initial_request=False,
        session_id=previous_instance.session_id,
        pinned_decoder_key=previous_instance.decoder_key,
        pinned_decoder_dp_rank=previous_instance.decoder_dp_rank,
        route_reason=route_reason,
    )


async def _replay_first_chunk(first_chunk: bytes, gen: Any):
    """Replay a pre-read first chunk, then stream the rest of ``gen``.

    ``handle_completions_impl`` pre-reads the first chunk of the decode stream
    before committing the ASGI response head so an initial decode error can be
    returned with the real HTTP status; this replays that chunk so the client
    still receives the full stream.
    """
    try:
        if first_chunk:
            yield first_chunk
        async for chunk in gen:
            yield chunk
    finally:
        await gen.aclose()


class CleanupStreamingResponse(StreamingResponse):
    """Retain cleanup ownership even when the body iterator never starts."""

    def __init__(self, content, *, finalize, **kwargs):
        super().__init__(content, **kwargs)
        self.finalize = finalize

    async def __call__(self, scope, receive, send):
        try:
            await super().__call__(scope, receive, send)
        finally:
            # Starlette may cancel its body task before its first iteration.
            # Closing an unstarted async generator does not execute its finally.
            with anyio.CancelScope(shield=True):
                try:
                    await self.body_iterator.aclose()
                finally:
                    await self.finalize()


async def handle_completions_impl(api: str, request: Request):
    runtime = get_runtime()
    args = get_global_args()
    request_released = False
    session_finalized = False
    released_kv = False
    try:
        req_data = await request.json()
        req_body = await request.body()
        request_length = len(req_body)
        try:
            route = await plan_request_route(request, req_data)
        except SessionBusyError as exc:
            return JSONResponse(
                status_code=409,
                content={
                    "error": {
                        "message": str(exc),
                        "type": "session_conflict",
                        "code": "session_busy",
                    }
                },
            )

        async def finish_session_once(success: bool) -> None:
            nonlocal session_finalized
            if route.session_reserved and route.session_id is not None and not session_finalized:
                await runtime.schedule("finish_session", route.session_id, success)
                session_finalized = True

        instance_info = await assign_request(api, req_data, request_length, route)
        logger.info(
            "Request %s session=%s route=%s reason=%s decoder=%s dp_rank=%s engine_epoch=%s affinity=proxy_hint",
            instance_info.request_id,
            instance_info.session_id,
            instance_info.route_mode.value,
            instance_info.route_reason,
            instance_info.decoder_key,
            instance_info.decoder_dp_rank,
            instance_info.decoder_engine_epoch,
        )
        stream_flag = bool(req_data.get("stream", False))
        initial_route = instance_info.route_mode.value
        chat_flag = "messages" in req_data

        if "prompt" in req_data:
            origin_prompt = req_data["prompt"]
        elif chat_flag:
            messages = req_data["messages"]
            origin_prompt = messages[0].get("content", "")
        else:
            origin_prompt = ""
        origin_max_tokens = req_data.get("max_tokens", 16)

        # Pre-open the decode response BEFORE committing the ASGI 200 head so an
        # initial decode 4xx/5xx/connection failure can be returned with the real
        # HTTP status code instead of an empty 200. On success the opened stream
        # (first chunk already read) is handed to generate_stream via these vars.
        preopened_gen = None
        preopened_first = b""
        decoder_client = await runtime.get_client(ServerRole.DECODE, instance_info.decoder_key)
        preopened_gen = stream_service_response(
            decoder_client,
            api,
            req_data,
            request_id=instance_info.request_id,
            decoder_dp_rank=instance_info.decoder_dp_rank,
            max_retries=args.max_retries,
            base_delay=args.retry_delay,
        )
        try:
            preopened_first = await preopened_gen.__anext__()
        except asyncio.CancelledError:
            await _finish_instance(runtime, instance_info, release_prefill_kv=True)
            request_released = True
            await finish_session_once(False)
            raise
        except StopAsyncIteration:
            preopened_first = b""
        except httpx.HTTPStatusError as exc:
            await _finish_instance(runtime, instance_info, release_prefill_kv=True)
            request_released = True
            await finish_session_once(False)
            return Response(
                content=exc.response.content, status_code=exc.response.status_code, media_type="application/json"
            )
        except httpx.RequestError as exc:
            await _finish_instance(runtime, instance_info, release_prefill_kv=True)
            request_released = True
            await finish_session_once(False)
            return JSONResponse(
                status_code=502,
                content={
                    "error": {
                        "message": f"Decode backend request to {api} failed: {exc}",
                        "type": "upstream_error",
                        "code": "decode_backend_unavailable",
                    }
                },
            )

        if getattr(args, "disable_recomputed_retry", False) and not stream_flag and preopened_first:
            result = json.loads(preopened_first)
            if any(choice.get("stop_reason") == "recomputed" for choice in result.get("choices", [])):
                await preopened_gen.aclose()
                raise BackendContractError(
                    "Decoder requested recomputation; recomputation retry is disabled",
                    "decoder_recomputed_retry_disabled",
                )

        if route.session_reserved and route.session_id is not None:
            try:
                await runtime.schedule(
                    "bind_session_decoder", route.session_id, instance_info.decoder_key, instance_info.decoder_dp_rank
                )
            except RuntimeError as exc:
                await _finish_instance(runtime, instance_info, release_prefill_kv=True)
                request_released = True
                await finish_session_once(False)
                return JSONResponse(
                    status_code=503,
                    content={
                        "error": {
                            "message": str(exc),
                            "type": "session_affinity_error",
                            "code": "decoder_affinity_unavailable",
                        }
                    },
                )

        async def generate_stream():
            nonlocal instance_info
            nonlocal request_released, released_kv
            nonlocal preopened_gen, preopened_first
            generated_token = ""
            emitted_output = False
            gen = None
            retry_count = 0
            retry = True
            completion_tokens = 0
            reported_prefiller_cached_tokens = instance_info.prefiller_cached_tokens
            stream_succeeded = False

            async def release_prefill_kv_once() -> None:
                nonlocal released_kv
                if not released_kv and instance_info.prefiller_key is not None:
                    await runtime.schedule(
                        "release_prefill_kv", instance_info.prefiller_key, instance_info.prefiller_score
                    )
                    released_kv = True

            released_kv = instance_info.prefiller_key is None

            try:
                while retry:
                    retry = False
                    if preopened_gen is not None:
                        # First iteration: consume the decode stream pre-opened by the
                        # caller so initial 4xx/5xx could be returned with the real status.
                        gen = _replay_first_chunk(preopened_first, preopened_gen)
                        preopened_gen = None
                    else:
                        decoder_client = await runtime.get_client(ServerRole.DECODE, instance_info.decoder_key)
                        gen = stream_service_response(
                            decoder_client,
                            api,
                            req_data,
                            request_id=instance_info.request_id,
                            decoder_dp_rank=instance_info.decoder_dp_rank,
                            max_retries=args.max_retries,
                            base_delay=args.retry_delay,
                        )
                    async for chunk in gen:
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
                        # HTTP headers describe the initial route and cannot be
                        # changed after streaming starts. The final JSON (or
                        # parsed SSE chunks) records any recompute recovery so
                        # clients can distinguish P fallback from D-local reuse.
                        chunk_json["proxy_metrics"] = {
                            "initial_route": initial_route,
                            "final_route": instance_info.route_mode.value,
                            "recovered": retry_count > 0,
                            "recovery_count": retry_count,
                            "admission_fallback": instance_info.route_reason == "decoder_cache_miss_fallback",
                            "fallback_reason": (
                                "decoder_cache_miss"
                                if instance_info.route_reason == "decoder_cache_miss_fallback"
                                else None
                            ),
                            "decoder_owner": instance_info.decoder_key,
                            "decoder_dp_rank": instance_info.decoder_dp_rank,
                            "decoder_engine_epoch": instance_info.decoder_engine_epoch,
                            "cached_tokens_source": (
                                "prefiller" if instance_info.prefiller_key is not None else "decoder"
                            ),
                        }
                        chunk = encode_response_chunk(chunk_json, is_sse)
                        choices = chunk_json.get("choices", [])
                        if not choices or not stream_flag:
                            if update_cached_tokens_in_chunk(chunk_json, reported_prefiller_cached_tokens):
                                chunk = encode_response_chunk(chunk_json, is_sse)
                            if not choices:
                                yield chunk
                                continue

                        choice = choices[0]
                        delta = choice.get("delta") or {}
                        message = choice.get("message") or {}
                        content = delta.get("content") or message.get("content") or choice.get("text") or ""
                        generated_token += content
                        emitted_output = emitted_output or bool(
                            content
                            or choice.get("token_ids")
                            or delta.get("reasoning")
                            or delta.get("reasoning_content")
                            or delta.get("tool_calls")
                            or message.get("tool_calls")
                            or message.get("reasoning")
                            or message.get("reasoning_content")
                        )

                        stop_reason = choice.get("stop_reason")
                        if stop_reason == "decoder_cache_lost_after_output":
                            raise BackendContractError(
                                "Decoder lost KV after output; replay would duplicate tokens",
                                "decoder_cache_lost_after_output",
                            )
                        if stop_reason == "decoder_cache_miss":
                            if (
                                instance_info.route_mode is not RouteMode.DECODER_LOCAL
                                or retry_count > 0
                                or emitted_output
                            ):
                                raise BackendContractError(
                                    "Invalid Decoder cache-miss admission response",
                                    "invalid_decoder_cache_miss",
                                )
                            # This is an admission fallback, not recovery after
                            # Decoder prefill. Preserve the entire original
                            # prompt/history and generation budget.
                            await gen.aclose()
                            req_data.get("vllm_xargs", {}).pop("require_decoder_cache_hit", None)
                            req_data["max_tokens"] = origin_max_tokens
                            completion_tokens = 0
                            retry_count += 1
                            retry = True
                            try:
                                instance_info = await reassign_instances(
                                    api,
                                    req_data,
                                    request_length,
                                    instance_info,
                                    route_reason="decoder_cache_miss_fallback",
                                )
                            except BackendContractError:
                                raise
                            except Exception as exc:
                                raise BackendContractError(
                                    "P-to-D fallback failed after Decoder cache miss",
                                    "decoder_cache_miss_fallback_failed",
                                ) from exc
                            instance_info.route_reason = "decoder_cache_miss_fallback"
                            released_kv = False
                            reported_prefiller_cached_tokens = instance_info.prefiller_cached_tokens
                            if route.session_reserved and route.session_id is not None:
                                await runtime.schedule(
                                    "bind_session_decoder",
                                    route.session_id,
                                    instance_info.decoder_key,
                                    instance_info.decoder_dp_rank,
                                )
                            break
                        usage = chunk_json.get("usage", {})
                        completion_tokens = (
                            (completion_tokens + 1)
                            if stream_flag
                            else (completion_tokens + usage.get("completion_tokens", 0))
                        )
                        if stop_reason == "recomputed":
                            if getattr(args, "disable_recomputed_retry", False):
                                raise BackendContractError(
                                    "Decoder requested recomputation; recomputation retry is disabled",
                                    "decoder_recomputed_retry_disabled",
                                )
                            retry = True
                            retry_count += 1
                            if chat_flag:
                                messages[0]["content"] = (
                                    origin_prompt
                                    + ([{"type": "text", "text": generated_token}] if generated_token else [])
                                    if isinstance(origin_prompt, list)
                                    else (origin_prompt or "") + generated_token
                                )
                            else:
                                req_data["prompt"] = origin_prompt + generated_token
                            req_data["max_tokens"] = origin_max_tokens - completion_tokens + retry_count
                            (req_data.get("vllm_xargs") or {}).pop("require_decoder_cache_hit", None)
                            tmp_request_length = len(json.dumps(req_data).encode("utf-8"))
                            instance_info = await reassign_instances(api, req_data, tmp_request_length, instance_info)
                            reported_prefiller_cached_tokens = instance_info.prefiller_cached_tokens
                            if route.session_reserved and route.session_id is not None:
                                await runtime.schedule(
                                    "bind_session_decoder",
                                    route.session_id,
                                    instance_info.decoder_key,
                                    instance_info.decoder_dp_rank,
                                )
                            logger.info(
                                "Request %s session=%s route=%s reason=%s decoder=%s dp_rank=%s engine_epoch=%s",
                                instance_info.request_id,
                                instance_info.session_id,
                                instance_info.route_mode.value,
                                instance_info.route_reason,
                                instance_info.decoder_key,
                                instance_info.decoder_dp_rank,
                                instance_info.decoder_engine_epoch,
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
                stream_succeeded = True
            except asyncio.CancelledError:
                logger.warning(
                    "Streaming from decoder %s:%s was cancelled; releasing request %s resources",
                    instance_info.decoder_host,
                    instance_info.decoder_port,
                    instance_info.request_id,
                )
                raise
            except BackendContractError as exc:
                logger.error("Request %s failed routing gate: %s", instance_info.request_id, exc)
                yield encode_response_chunk(
                    {
                        "error": {"message": str(exc), "type": "proxy_routing_gate", "code": exc.code},
                        "proxy_metrics": {
                            "initial_route": initial_route,
                            "final_route": instance_info.route_mode.value,
                            "recovered": retry_count > 0,
                            "recovery_count": retry_count,
                            "gate_failed": True,
                        },
                    },
                    stream_flag,
                )
            except Exception as exc:
                logger.error(
                    "Error during streaming from decoder %s:%s: %s while handling request %s; releasing prefiller KV",
                    instance_info.decoder_host,
                    instance_info.decoder_port,
                    exc,
                    instance_info.request_id,
                )
                yield encode_response_chunk(
                    {
                        "error": {"message": "Decoder response stream failed", "type": "proxy_stream_error"},
                    },
                    stream_flag,
                )
            finally:
                with anyio.CancelScope(shield=True):
                    try:
                        try:
                            if gen is not None:
                                await gen.aclose()
                        finally:
                            await _finish_instance(runtime, instance_info, release_prefill_kv=not released_kv)
                            released_kv = True
                            request_released = True
                    finally:
                        await finish_session_once(stream_succeeded)
                        logger.info(
                            "Request %s session=%s completed=%s decoder=%s dp_rank=%s engine_epoch=%s",
                            instance_info.request_id,
                            instance_info.session_id,
                            stream_succeeded,
                            instance_info.decoder_key,
                            instance_info.decoder_dp_rank,
                            instance_info.decoder_engine_epoch,
                        )

        media_type = "text/event-stream; charset=utf-8" if stream_flag else "application/json"

        async def finalize_response_once():
            nonlocal request_released, released_kv
            try:
                if preopened_gen is not None:
                    await preopened_gen.aclose()
            finally:
                try:
                    if not request_released:
                        await _finish_instance(runtime, instance_info, release_prefill_kv=not released_kv)
                        request_released = True
                        released_kv = True
                finally:
                    await finish_session_once(False)

        return CleanupStreamingResponse(
            generate_stream(),
            finalize=finalize_response_once,
            media_type=media_type,
            headers=routing_response_headers(instance_info),
        )
    except asyncio.CancelledError:
        with anyio.CancelScope(shield=True):
            try:
                if "preopened_gen" in locals() and preopened_gen is not None:
                    await preopened_gen.aclose()
            finally:
                try:
                    if not request_released and "instance_info" in locals():
                        await _finish_instance(runtime, instance_info, release_prefill_kv=not released_kv)
                        request_released = True
                finally:
                    if "finish_session_once" in locals():
                        await finish_session_once(False)
        raise
    except Exception as e:
        import traceback

        exc_info = sys.exc_info()
        print(f"Error occurred in disagg prefill proxy server - {api} endpoint")
        print("".join(traceback.format_exception(*exc_info)))
        if not request_released and "instance_info" in locals():
            await _finish_instance(runtime, instance_info, release_prefill_kv=True)
            request_released = True
        if "finish_session_once" in locals():
            await finish_session_once(False)
        if isinstance(e, BackendContractError):
            return JSONResponse(
                status_code=502,
                content={
                    "error": {"message": str(e), "type": "proxy_routing_gate", "code": e.code},
                    "proxy_metrics": {"gate_failed": True},
                },
            )
        if isinstance(e, httpx.HTTPStatusError) and e.response is not None:
            try:
                err_body = e.response.json()
            except Exception:
                err_body = {"error": {"message": e.response.text or "upstream error"}}
            return JSONResponse(err_body, status_code=e.response.status_code)
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


@app.post("/v1/completions")
@with_cancellation
async def handle_completions(request: Request):
    return await handle_completions_impl("/completions", request)


@app.post("/v1/chat/completions")
@with_cancellation
async def handle_chat_completions(request: Request):
    return await handle_completions_impl("/chat/completions", request)


async def handle_models_impl():
    runtime = get_runtime()
    request_id = next_req_id()
    await runtime.sync_clients()
    snapshot = runtime.scheduler.get_snapshot()
    candidates = [(ServerRole.PREFILL, s) for s in snapshot["prefill_instances"]] + [
        (ServerRole.DECODE, s) for s in snapshot["decode_instances"]
    ]
    if not candidates:
        return JSONResponse(status_code=503, content={"error": "No available backend instances"})
    failures: list[str] = []
    for role, server in candidates:
        host, port = server["host"], server["port"]
        key = server_key(host, port)
        try:
            client = await runtime.get_client(role, key)
            response = await client.get("/models", headers=auth_headers(request_id), timeout=3.0)
            response.raise_for_status()
            return Response(content=response.content, media_type="application/json")
        except (httpx.RequestError, httpx.HTTPStatusError) as exc:
            logger.warning("GET /models from %s:%s failed: %s", host, port, exc)
            failures.append(f"{host}:{port}")
    return JSONResponse(
        status_code=503,
        content={
            "error": "All backend instances failed to serve /v1/models",
            "failed": failures,
        },
    )


@app.get("/v1/models")
async def handle_models():
    return await handle_models_impl()


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
    runtime.scheduler.invalidate_all_sessions()
    if failures:
        return JSONResponse(status_code=500, content={"failed": failures})
    return Response(status_code=200)


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

    module_name = f"{__package__}.{Path(__file__).stem}" if __package__ else Path(__file__).stem
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
