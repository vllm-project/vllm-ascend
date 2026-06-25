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
from dataclasses import dataclass, field
from enum import Enum
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


TAINT_PRIORITY = 1e15

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

        for host, port in prefiller_instances:
            self._add_server_no_lock(ServerRole.PREFILL, host, port)
        for host, port in decoder_instances:
            self._add_server_no_lock(ServerRole.DECODE, host, port)

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

    def _pop_valid(self, role: ServerRole) -> str:
        pool = self._pool(role)
        while pool.heap:
            _, _, seq, key = heapq.heappop(pool.heap)
            if key not in pool.servers:
                continue
            entry = pool.servers[key]
            if entry.heap_seq == seq:
                return key
        raise RuntimeError(f"No available {role.value} servers")

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
        pool.servers[key] = BackendServer(host, int(port), self._next_ordinal())
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

    def healthcheck(self) -> dict[str, Any]:
        with self._lock:
            return {
                "status": "ok",
                "prefill_instances": len(self.prefillers),
                "decode_instances": len(self.decoders),
                "request_num": self.request_num,
            }

    def _pick_server(
        self,
        role: ServerRole,
        load: float,
        *,
        active_tokens: bool = False,
        kv_cache: bool = False,
    ) -> dict[str, Any]:
        key = self._pop_valid(role)
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
    headers = auth_headers(request_id)
    for attempt in range(1, max_retries + 1):
        try:
            async with client.stream("POST", endpoint, json=req_data, headers=headers) as response:
                response.raise_for_status()
                first_chunk_sent = False
                async for chunk in response.aiter_bytes():
                    first_chunk_sent = True
                    yield chunk
                return
        except (httpx.RequestError, httpx.HTTPStatusError) as exc:
            if attempt < max_retries:
                logger.warning("Attempt %s failed for streaming %s: %s", attempt, endpoint, exc)
                await asyncio.sleep(base_delay * (2 ** (attempt - 1)))
            else:
                logger.error("All %s attempts failed for streaming %s.", max_retries, endpoint)
                raise exc
        except Exception as exc:
            if "first_chunk_sent" in locals() and first_chunk_sent:
                logger.error("Streaming to client interrupted after response started: %s", exc)
                return
            if attempt < max_retries:
                logger.warning("Attempt %s failed for streaming %s: %s", attempt, endpoint, exc)
                await asyncio.sleep(base_delay * (2 ** (attempt - 1)))
            else:
                logger.error("All %s attempts failed for streaming %s.", max_retries, endpoint)
                raise exc


async def _abort_prefill_selection(
    runtime: WorkerRuntime,
    prefiller_key: str,
    prefiller_score: float,
    *,
    is_initial_request: bool,
) -> None:
    if is_initial_request:
        await runtime.schedule("finish_request", prefiller_key, prefiller_score, None, 0.0, release_prefill_kv=True)
    else:
        await runtime.schedule("release_prefill_kv", prefiller_key, prefiller_score)


async def _finish_instance(runtime: WorkerRuntime, info: InstanceInfo, *, release_prefill_kv: bool) -> None:
    await runtime.schedule(
        "finish_request",
        info.prefiller_key,
        info.prefiller_score,
        info.decoder_key,
        info.decoder_score,
        release_prefill_kv,
    )


async def assign_instances(
    api: str,
    req_data: Any,
    request_length: int,
    *,
    is_initial_request: bool,
) -> InstanceInfo:
    runtime = get_runtime()
    args = get_global_args()
    prefiller_score = calculate_prefill_score(request_length)
    decoder_score = calculate_decode_score(request_length)
    request_id = next_req_id()
    pick_prefill = "begin_request" if is_initial_request else "reserve_prefill_kv"
    prefiller = await runtime.schedule(pick_prefill, prefiller_score)
    prefiller_key = prefiller["key"]

    try:
        response = await send_request_to_service(
            await runtime.get_client(ServerRole.PREFILL, prefiller_key),
            api,
            req_data,
            request_id,
            max_retries=args.max_retries,
            base_delay=args.retry_delay,
        )
    except Exception:
        await _abort_prefill_selection(runtime, prefiller_key, prefiller_score, is_initial_request=is_initial_request)
        raise

    kv_transfer_params = response.json().get("kv_transfer_params", {})
    if kv_transfer_params:
        req_data["kv_transfer_params"] = kv_transfer_params

    try:
        decoder = await runtime.schedule("pick_decoder", decoder_score)
    except Exception:
        await _abort_prefill_selection(runtime, prefiller_key, prefiller_score, is_initial_request=is_initial_request)
        raise

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
            retry = True
            completion_tokens = 0

            async def release_prefill_kv_once() -> None:
                nonlocal released_kv
                if not released_kv:
                    await runtime.schedule(
                        "release_prefill_kv", instance_info.prefiller_key, instance_info.prefiller_score
                    )
                    released_kv = True

            try:
                while retry:
                    retry = False
                    decoder_client = await runtime.get_client(ServerRole.DECODE, instance_info.decoder_key)
                    async for chunk in stream_service_response_with_retry(
                        decoder_client,
                        api,
                        req_data,
                        request_id=instance_info.request_id,
                        max_retries=args.max_retries,
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
                        if chunk_str.startswith("data: "):
                            chunk_str = chunk_str[len("data: ") :]
                        try:
                            chunk_json = json.loads(chunk_str)
                        except json.JSONDecodeError:
                            logger.debug("Skipping chunk: %s", chunk_str)
                            yield chunk
                            continue
                        choices = chunk_json.get("choices", [])
                        if not choices:
                            yield chunk
                            continue

                        choice = choices[0]
                        delta = choice.get("delta") or {}
                        message = choice.get("message") or {}
                        content = delta.get("content") or message.get("content") or choice.get("text") or ""
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
                            req_data["max_tokens"] = origin_max_tokens - completion_tokens + retry_count
                            tmp_request_length = len(json.dumps(req_data).encode("utf-8"))
                            instance_info = await reassign_instances(api, req_data, tmp_request_length, instance_info)
                            released_kv = False
                            break
                        if retry_count > 0 and not stream_flag:
                            if chat_flag:
                                choice["message"]["content"] = generated_token
                            else:
                                choice["text"] = generated_token
                            chunk = json.dumps(chunk_json).encode("utf-8")
                        yield chunk
            except asyncio.CancelledError:
                logger.warning(
                    "Streaming from decoder %s:%s was cancelled; releasing request %s resources",
                    instance_info.decoder_host,
                    instance_info.decoder_port,
                    instance_info.request_id,
                )
                raise
            except Exception as exc:
                logger.error(
                    "Error during streaming from decoder %s:%s: %s while handling request %s; releasing prefiller KV",
                    instance_info.decoder_host,
                    instance_info.decoder_port,
                    exc,
                    instance_info.request_id,
                )
            finally:
                await _finish_instance(runtime, instance_info, release_prefill_kv=not released_kv)
                released_kv = True
                request_released = True

        media_type = "text/event-stream; charset=utf-8" if stream_flag else "application/json"
        return StreamingResponse(generate_stream(), media_type=media_type)
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
