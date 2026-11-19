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
# - Forwards native Anthropic /v1/messages and OpenAI /v1/responses APIs.
# - Responses requests must be synchronous and supply their own conversation
#   history; background and previous_response_id are not supported.
# - Use Messages/Responses forwarding in this example for evaluation and
#   testing. Responses requests can hit vLLM #64009 on backends that cannot
#   enforce min_tokens=1 during prefill.
# - Native recompute needs the matching Ascend decode scheduler and API server.
#   On recompute the proxy selects a fresh P/D pair, up to --max-retries times.
#   It continues visible text or retries the original input with the remaining
#   token budget. Structured stream events are held until the attempt finishes.
#   A retry can run backend tools again.
# - Streams responses from backend servers to clients.
#
# Prerequisites:
# - Python 3.10+
# - Both backends must expose any Messages/Responses API used through the proxy.
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
import copy
import functools
import hashlib
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
from collections.abc import AsyncGenerator, AsyncIterator, Iterator
from contextlib import aclosing, asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum
from multiprocessing.managers import BaseManager
from pathlib import Path
from typing import Any, cast

import httpx
from anyio import CancelScope
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from starlette.types import Receive, Scope, Send

logger = logging.getLogger(__name__)

PD_RECOMPUTE_ARG = "ascend_pd_recompute"
PD_RECOMPUTE_VERSION = "v1"
PD_RECOMPUTE_RESUME = "ascend_pd_recompute_resume"
PD_CONTINUE_MESSAGE = "ascend_pd_continue_message"
PD_RECOMPUTE_RETRY = "ascend_pd_recompute_before_output"
PD_RECOMPUTE_INTERRUPTED = "ascend_pd_recompute_after_output"
NATIVE_STREAM_START_EVENTS = ("message_start", "response.created", "response.in_progress", "ping")
NATIVE_STREAM_END_EVENTS = ("message_stop", "response.completed", "response.incomplete", "response.failed", "error")
MAX_DEFERRED_STREAM_BYTES = 64 * 1024

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
    decoder_released: bool = False


TAINT_PRIORITY = 1e15


def extract_cached_tokens(response_json: dict) -> int | None:
    usage = response_json.get("usage") or {}
    prompt_tokens_details = usage.get("prompt_tokens_details") or {}
    cached_tokens = prompt_tokens_details.get("cached_tokens")
    return cached_tokens if isinstance(cached_tokens, int) else 0


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


def conversation_affinity_key(req_data: dict) -> str | None:
    """Identify a conversation without hashing its growing message history."""
    user = req_data.get("user")
    if isinstance(user, str) and user:
        identity = [req_data.get("model"), "session", user]
    else:
        messages = req_data.get("messages") or []
        first_user_index = next((i for i, message in enumerate(messages) if message.get("role") == "user"), None)
        if first_user_index is None:
            return None
        instructions = [m for m in messages[:first_user_index] if m.get("role") in ("system", "developer")]
        identity = [req_data.get("model"), instructions, messages[first_user_index]]
    encoded = json.dumps(identity, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


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
        affinity_key: str | None = None,
    ) -> dict[str, Any]:
        if affinity_key and role is ServerRole.PREFILL:
            pool = self._pool(role)
            candidates = pool.servers.keys() - pool.tainted
            if not candidates:
                raise RuntimeError("No available prefill servers for session affinity")
            # Rendezvous hashing is stable across workers/restarts and only
            # remaps conversations whose chosen endpoint is removed.
            key = max(
                candidates,
                key=lambda k: (hashlib.sha256((affinity_key + "\0" + k).encode("utf-8")).digest(), k),
            )
            logger.debug("Prefill affinity key=%s backend=%s", affinity_key[:16], key)
        else:
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

    def begin_request(self, load: float, affinity_key: str | None = None) -> dict[str, Any]:
        """Pick a prefiller, reserve KV pressure, and count this as an active request."""
        with self._lock:
            picked = self._pick_server(ServerRole.PREFILL, load, kv_cache=True, affinity_key=affinity_key)
            self.request_num += 1
            return picked

    def reserve_prefill_kv(self, load: float, affinity_key: str | None = None) -> dict[str, Any]:
        """Pick a prefiller for recompute without bumping the active request count."""
        with self._lock:
            return self._pick_server(ServerRole.PREFILL, load, kv_cache=True, affinity_key=affinity_key)

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
    parser.add_argument(
        "--enable-prefill-affinity",
        action="store_true",
        help="Route conversations to stable prefill endpoints to reuse prefix cache; decode remains load balanced.",
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


def _api_kind(api: str) -> str:
    path = "/" + api.strip("/")
    if path.endswith("/responses"):
        return "responses"
    if path.endswith("/messages"):
        return "messages"
    return "openai"


def build_prefill_request_for_api(api: str, req_data: dict) -> dict:
    if _api_kind(api) != "responses":
        return build_prefill_request(req_data)
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
    payload["max_output_tokens"] = 1
    payload.pop("max_tokens", None)
    payload.pop("min_tokens", None)
    payload.pop("stream_options", None)
    payload.pop("max_completion_tokens", None)
    return payload


def extract_cached_tokens_for_api(api: str, response_json: dict) -> int | None:
    kind = _api_kind(api)
    usage = response_json.get("usage") or {}
    if kind == "messages":
        cached = usage.get("cache_read_input_tokens")
        return cached if isinstance(cached, int) else 0
    if kind == "responses":
        details = usage.get("input_tokens_details") or {}
        cached = details.get("cached_tokens")
        return cached if isinstance(cached, int) else 0
    return extract_cached_tokens(response_json)


def write_cached_tokens_for_api(api: str, chunk_json: dict, cached_tokens: int | None) -> bool:
    kind = _api_kind(api)
    if kind == "openai":
        return update_cached_tokens_in_chunk(chunk_json, cached_tokens)
    if cached_tokens is None:
        return False
    usage = chunk_json.get("usage")
    if not isinstance(usage, dict):
        response_obj = chunk_json.get("response" if kind == "responses" else "message")
        usage = response_obj.get("usage") if isinstance(response_obj, dict) else None
    if not isinstance(usage, dict):
        return False
    if kind == "messages":
        input_tokens = usage.get("input_tokens")
        previous_cached = usage.get("cache_read_input_tokens", 0)
        if isinstance(input_tokens, int) and isinstance(previous_cached, int):
            # Messages input_tokens excludes cache reads; preserve total input.
            usage["input_tokens"] = input_tokens + previous_cached - cached_tokens
        usage["cache_read_input_tokens"] = cached_tokens
        return True
    details = usage.get("input_tokens_details")
    if not isinstance(details, dict):
        details = {}
    details["cached_tokens"] = cached_tokens
    cached_per_turn = details.get("cached_tokens_per_turn")
    # An empty list means per-turn accounting is unavailable; preserve it.
    if isinstance(cached_per_turn, list) and cached_per_turn:
        # Only the first turn is prefetched by P. Later tool turns run on D.
        details["cached_tokens_per_turn"] = [cached_tokens, *cached_per_turn[1:]]
        details["cached_tokens"] = sum(details["cached_tokens_per_turn"])
    usage["input_tokens_details"] = details
    return True


def _json_object(payload: bytes) -> dict | None:
    try:
        obj = json.loads(payload)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None
    return obj if isinstance(obj, dict) else None


@dataclass(frozen=True)
class SSEFrame:
    raw: bytes
    lines: list[bytes]
    data_indices: list[int]
    data: dict | None
    event: bytes | None

    @classmethod
    def parse(cls, frame: bytes) -> "SSEFrame":
        lines = frame.splitlines(keepends=True)
        data_indices = [index for index, line in enumerate(lines) if line.startswith(b"data:")]
        payload = b"\n".join(lines[index][5:].strip() for index in data_indices)
        event = next((line[6:].strip() for line in reversed(lines) if line.startswith(b"event:")), None)
        return cls(frame, lines, data_indices, _json_object(payload), event)

    def encode(self) -> bytes:
        first = self.data_indices[0]
        ending = self.lines[first][len(self.lines[first].rstrip(b"\r\n")) :]
        patched = b"data: " + json.dumps(self.data).encode("utf-8") + ending
        return b"".join(
            patched if index == first else line
            for index, line in enumerate(self.lines)
            if index == first or index not in self.data_indices
        )


def patch_cached_tokens_in_sse_frame(api: str, frame: bytes | SSEFrame, cached_tokens: int | None) -> bytes:
    parsed = SSEFrame.parse(frame) if isinstance(frame, bytes) else frame
    if parsed.data is not None and write_cached_tokens_for_api(api, parsed.data, cached_tokens):
        return parsed.encode()
    return parsed.raw


@dataclass(frozen=True)
class RecomputeProgress:
    output_tokens: int
    remaining_tokens: int
    input_tokens: int

    @classmethod
    def parse(cls, data: dict | None) -> "RecomputeProgress | None":
        if data is None:
            return None
        used = data.get("output_tokens")
        remaining = data.get("remaining_tokens")
        prompt = data.get("input_tokens")
        if type(used) is not int or type(remaining) is not int or type(prompt) is not int:
            return None
        if used <= 0 or remaining < 0 or prompt < 0:
            return None
        return cls(used, remaining, prompt)


class DecodeRecompute(Exception):
    def __init__(self, signal: str, progress: RecomputeProgress | None = None, partial: dict | None = None):
        super().__init__(signal)
        self.signal = signal
        self.progress = progress
        self.partial = partial


def _decode_recompute(obj: dict | None) -> DecodeRecompute | None:
    if obj is None:
        return None
    transfer = obj.get("kv_transfer_params")
    if isinstance(transfer, dict) and transfer.get(PD_RECOMPUTE_ARG) == PD_RECOMPUTE_INTERRUPTED:
        return DecodeRecompute(PD_RECOMPUTE_INTERRUPTED, RecomputeProgress.parse(transfer), obj)
    error = obj.get("error")
    if isinstance(error, dict):
        message = error.get("message")
    elif obj.get("type") == "error":
        message = obj.get("message")
        code = obj.get("code")
        if code in (PD_RECOMPUTE_RETRY, PD_RECOMPUTE_INTERRUPTED) and (
            not isinstance(message, str) or not message.startswith(f"{code}:")
        ):
            message = code
    else:
        return None
    if isinstance(message, str):
        signal, _, detail = message.partition(":")
        if signal in (PD_RECOMPUTE_RETRY, PD_RECOMPUTE_INTERRUPTED):
            progress = RecomputeProgress.parse(_json_object(detail.encode())) if detail else None
            return DecodeRecompute(signal, progress)
    return None


class RetryPrefillMissingKV(Exception):
    """A recompute attempt must use fresh KV transfer parameters."""


class IncompleteDecodeResponse(Exception):
    """The native backend ended a non-streaming response without a JSON object."""


async def _split_sse_frames(chunks: AsyncIterator[bytes]) -> AsyncGenerator[tuple[bytes, bool], None]:
    stream_buf = bytearray()
    async for chunk in chunks:
        stream_buf += chunk
        while True:
            boundaries = [
                (offset, len(separator))
                for separator in (b"\n\n", b"\r\n\r\n", b"\r\r")
                if (offset := stream_buf.find(separator)) >= 0
            ]
            if not boundaries:
                break
            offset, size = min(boundaries)
            frame = bytes(stream_buf[: offset + size])
            del stream_buf[: offset + size]
            yield frame, True
    if stream_buf:
        yield bytes(stream_buf), False


async def send_request_to_service(
    client: httpx.AsyncClient,
    endpoint: str,
    req_data: dict,
    request_id: str,
    max_retries: int = 3,
    base_delay: float = 0.2,
):
    req_data = build_prefill_request_for_api(endpoint, req_data)
    headers = auth_headers(request_id)
    max_attempts = max(1, max_retries)
    for attempt in range(1, max_attempts + 1):
        try:
            response = await client.post(endpoint, json=req_data, headers=headers)
            response.raise_for_status()
            return response
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 400 or attempt == max_attempts:
                raise
            logger.warning("Attempt %s failed for %s: %s", attempt, endpoint, exc)
        except httpx.RequestError as exc:
            if attempt == max_attempts:
                raise
            logger.warning("Attempt %s failed for %s: %s", attempt, endpoint, exc)
        await asyncio.sleep(base_delay * (2 ** (attempt - 1)))


async def stream_service_response(
    client: httpx.AsyncClient,
    endpoint: str,
    req_data: dict,
    request_id: str,
    max_retries: int = 3,
    base_delay: float = 0.2,
    *,
    native: bool = False,
) -> AsyncGenerator[bytes, None]:
    headers = auth_headers(request_id)
    max_attempts = max(1, max_retries)
    req_data = req_data.copy()
    if not req_data.get("stream", False):
        req_data.pop("stream_options", None)
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
                async for chunk in response.aiter_bytes():
                    first_chunk_sent = True
                    yield chunk
                return
        except httpx.HTTPStatusError as exc:
            if native and (recompute := _decode_recompute(_json_object(exc.response.content))):
                raise recompute from exc
            if exc.response.status_code == 400 or attempt == max_attempts:
                raise
            logger.warning("Attempt %s failed for streaming %s: %s", attempt, endpoint, exc)
        except httpx.RequestError as exc:
            if first_chunk_sent:
                if native:
                    raise
                logger.error("Streaming to client interrupted after response started: %s", exc)
                return
            if attempt == max_attempts:
                raise
            logger.warning("Attempt %s failed for streaming %s: %s", attempt, endpoint, exc)
        except Exception as exc:
            if first_chunk_sent:
                if native:
                    raise
                logger.error("Streaming to client interrupted after response started: %s", exc)
                return
            if attempt == max_attempts:
                raise
            logger.warning("Attempt %s failed for streaming %s: %s", attempt, endpoint, exc)
        await asyncio.sleep(base_delay * (2 ** (attempt - 1)))


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
        None if info.decoder_released else info.decoder_key,
        info.decoder_score,
        release_prefill_kv,
    )
    info.decoder_released = True


async def _release_decoder_once(runtime: WorkerRuntime, info: InstanceInfo) -> None:
    if info.decoder_released:
        return
    await runtime.schedule("release_decoder", info.decoder_key, info.decoder_score)
    info.decoder_released = True


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
    request_id = req_data.get("request_id") or next_req_id()
    pick_prefill = "begin_request" if is_initial_request else "reserve_prefill_kv"
    if args.enable_prefill_affinity and (is_initial_request or _api_kind(api) == "openai"):
        prefiller = await runtime.schedule(pick_prefill, prefiller_score, conversation_affinity_key(req_data))
    else:
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
        response_json = response.json()
        kv_transfer_params = response_json.get("kv_transfer_params")
        if kv_transfer_params:
            req_data["kv_transfer_params"] = kv_transfer_params
        elif not is_initial_request and _api_kind(api) != "openai":
            raise RetryPrefillMissingKV("Recompute prefill backend did not return KV transfer parameters.")
        prefiller_cached_tokens = extract_cached_tokens_for_api(api, response_json)
    except (Exception, asyncio.CancelledError):
        with CancelScope(shield=True):
            await _abort_prefill_selection(
                runtime, prefiller_key, prefiller_score, is_initial_request=is_initial_request
            )
        raise

    decoder_key = None
    try:
        decoder = await runtime.schedule("pick_decoder", decoder_score)
        decoder_key = decoder["key"]
        prefiller_client = await runtime.get_client(ServerRole.PREFILL, prefiller_key)
        decoder_client = await runtime.get_client(ServerRole.DECODE, decoder_key)
    except (Exception, asyncio.CancelledError):
        with CancelScope(shield=True):
            await _abort_prefill_selection(
                runtime, prefiller_key, prefiller_score, is_initial_request=is_initial_request
            )
            if decoder_key is not None:
                await runtime.schedule("release_decoder", decoder_key, decoder_score)
        raise

    logger.debug("Using %s %s", prefiller_client.base_url, decoder_client.base_url)
    return InstanceInfo(
        request_id=request_id,
        prefiller_key=prefiller_key,
        prefiller_score=prefiller_score,
        decoder_key=decoder_key,
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
    *,
    previous_prefiller_kv_released: bool,
) -> InstanceInfo:
    runtime = get_runtime()
    if not previous_prefiller_kv_released:
        await runtime.schedule("release_prefill_kv", previous_instance.prefiller_key, previous_instance.prefiller_score)
    await _release_decoder_once(runtime, previous_instance)
    return await assign_instances(api, req_data, request_length, is_initial_request=False)


async def _replay_first_chunk(first_chunk: bytes, gen: Any):
    """Replay a pre-read first chunk, then stream the rest of ``gen``.

    ``handle_completions_impl`` pre-reads the first chunk of the decode stream
    before committing the ASGI response head so an initial decode error can be
    returned with the real HTTP status; this replays that chunk so the client
    still receives the full stream.
    """
    if first_chunk:
        yield first_chunk
    async for chunk in gen:
        yield chunk


def _encode_sse_event(data: dict) -> bytes:
    return b"event: " + data["type"].encode() + b"\n" + encode_response_chunk(data, True)


def _native_stream_error(api: str, signal: str, sequence: int) -> bytes:
    message = _recompute_failure_message(signal)
    if _api_kind(api) == "messages":
        obj = _native_decode_error(api, signal, message)
    else:
        obj = {
            "type": "error",
            "code": signal,
            "message": message,
            "param": None,
            "sequence_number": sequence,
        }
    return _encode_sse_event(obj)


async def _native_sse_frames(chunks: AsyncIterator[bytes], state: "NativeStream") -> AsyncGenerator[bytes, None]:
    pending: list[bytes] = []
    pending_size = 0
    deferred: list[SSEFrame] = []
    attempt_committed = False
    decode = state.decode
    continuing = bool(decode.prefix)
    defer_attempt = decode.buffer_output
    finished_attempt = False
    async with aclosing(_split_sse_frames(chunks)) as frames:
        async for raw_frame, complete in frames:
            if finished_attempt:
                yield raw_frame
                continue
            frame = SSEFrame.parse(raw_frame)
            data = frame.data
            event = data.get("type") if data is not None else None
            if (event == "error" or frame.event == b"error") and (recompute := _decode_recompute(data)):
                if attempt_committed and recompute.signal == PD_RECOMPUTE_RETRY:
                    raise DecodeRecompute(PD_RECOMPUTE_INTERRUPTED)
                raise recompute
            preamble = (
                event in NATIVE_STREAM_START_EVENTS
                if data is not None
                else all(not line.strip() or line.startswith(b":") for line in frame.lines)
            )
            if continuing and preamble:
                continue
            if defer_attempt or state.defer_event(data) or (continuing and event in NATIVE_STREAM_END_EVENTS):
                if not complete:
                    raise DecodeRecompute(PD_RECOMPUTE_INTERRUPTED)
                defer_attempt = True
                deferred.append(frame)
                if event in NATIVE_STREAM_END_EVENTS:
                    for start in pending:
                        yield start
                    pending.clear()
                    if continuing:
                        for outgoing in state.finish_continued(deferred):
                            yield outgoing
                        return
                    for held in deferred:
                        yield patch_cached_tokens_in_sse_frame(decode.api, held, decode.info.prefiller_cached_tokens)
                    deferred.clear()
                    finished_attempt = True
                continue

            if continuing:
                if data is None or not complete:
                    raise DecodeRecompute(PD_RECOMPUTE_INTERRUPTED)
                for outgoing in state.continue_event(frame):
                    attempt_committed = True
                    yield outgoing
                continue

            state.observe(data, continuing=False)
            patched = (
                patch_cached_tokens_in_sse_frame(decode.api, frame, decode.info.prefiller_cached_tokens)
                if complete
                else raw_frame
            )
            if not attempt_committed and (preamble or state.text_setup(data)):
                if (data is None or event == "ping") and pending_size + len(patched) > MAX_DEFERRED_STREAM_BYTES:
                    continue
                pending.append(patched)
                pending_size += len(patched)
                continue
            for start_frame in pending:
                yield start_frame
            pending.clear()
            attempt_committed = True
            yield patched
    if continuing or deferred:
        raise DecodeRecompute(PD_RECOMPUTE_INTERRUPTED)
    for start_frame in pending:
        yield start_frame


def _native_decode_error(api: str, signal: str, message: str) -> dict:
    if _api_kind(api) == "messages":
        error_type = "overloaded_error" if signal == PD_RECOMPUTE_RETRY else "api_error"
        return {"type": "error", "error": {"type": error_type, "message": message}}
    return {"error": {"type": "server_error", "code": signal, "message": message, "param": None}}


def _recompute_failure_message(signal: str) -> str:
    if signal == PD_RECOMPUTE_INTERRUPTED:
        return "Decode recompute interrupted generation; the proxy cannot resume the partial response."
    return "Decode recompute could not restart this request. Retry the original request if it is safe to repeat."


def _native_text_block(kind: str, obj: dict) -> dict | None:
    blocks = obj.get("content")
    if kind == "responses":
        output = obj.get("output")
        if not output or len(output) != 1 or output[0].get("type") != "message":
            return None
        blocks = output[0].get("content")
    block_type = "text" if kind == "messages" else "output_text"
    if blocks and len(blocks) == 1:
        block = blocks[0]
        if block.get("type") == block_type and isinstance(block.get("text"), str):
            return block
    return None


def _continue_native_prompt(kind: str, history: Any, prefix: str) -> list | None:
    if isinstance(history, str):
        history = [{"role": "user", "content": history}]
    if not isinstance(history, list):
        return None
    last = history[-1] if history else {}
    if kind == "messages":
        if isinstance(last, dict) and last.get("role") == "assistant":
            content = last.get("content")
            if isinstance(content, str):
                return [*history[:-1], {**last, "content": content + prefix}]
            if isinstance(content, list) and all(block.get("type") == "text" for block in content):
                text = "".join(block["text"] for block in content)
                return [*history[:-1], {**last, "content": text + prefix}]
            return None
        return [*history, {"role": "assistant", "content": prefix}]

    if (
        isinstance(last, dict)
        and last.get("role") == "assistant"
        and last.get("status") in ("in_progress", "incomplete")
    ):
        block = _native_text_block(kind, {"output": [last]})
        if block is None:
            return None
        return [*history[:-1], {**last, "content": [{**block, "text": block["text"] + prefix}]}]
    return [
        *history,
        {
            "type": "message",
            "role": "assistant",
            "id": f"msg_{uuid.uuid4().hex}",
            "status": "in_progress",
            "content": [{"type": "output_text", "text": prefix, "annotations": []}],
        },
    ]


class NativeStream:
    """Keep the client-visible envelope and collect text for a decode retry."""

    def __init__(self, decode: "NativeDecode"):
        self.decode = decode
        self.sent = False
        self.sequence = -1
        self.envelope: dict | None = None
        self.item: dict | None = None
        self.part: dict | None = None
        self.block_open = False
        self.resumable = True
        self.text: list[str] = []
        self.merged_attempt_text = False
        self.attempt_item_start: SSEFrame | None = None

    def text_setup(self, data: dict | None) -> bool:
        if data is None:
            return False
        event = data.get("type")
        if event == "content_block_start":
            block = data.get("content_block") or {}
            return data.get("index") == 0 and block.get("type") == "text" and not block.get("text")
        if event == "response.output_item.added":
            item = data.get("item") or {}
            return data.get("output_index") == 0 and item.get("type") == "message" and isinstance(item.get("id"), str)
        if event == "response.content_part.added":
            part = data.get("part") or {}
            return (
                data.get("output_index") == 0
                and data.get("content_index") == 0
                and part.get("type") == "output_text"
                and not part.get("text")
            )
        return False

    def defer_event(self, data: dict | None) -> bool:
        if data is None:
            return False
        event = data.get("type")
        if event in ("content_block_start", "response.output_item.added", "response.content_part.added"):
            return not self.text_setup(data)
        if event == "content_block_delta":
            return data.get("index") != 0 or (data.get("delta") or {}).get("type") != "text_delta"
        if event == "response.output_text.delta":
            return data.get("output_index", 0) != 0 or data.get("content_index", 0) != 0
        if event in ("content_block_stop", "message_delta"):
            return event == "content_block_stop" or bool(self.decode.prefix)
        if isinstance(event, str) and event.startswith("response."):
            return event not in NATIVE_STREAM_START_EVENTS and event not in NATIVE_STREAM_END_EVENTS
        return False

    def observe(self, data: dict | None, *, continuing: bool) -> None:
        if data is None:
            return
        if not continuing and isinstance(data.get("sequence_number"), int):
            self.sequence = data["sequence_number"]
        if not self.resumable:
            return
        if self.decode.kind == "messages":
            self.observe_message(data, continuing)
        else:
            self.observe_response(data, continuing)

    def observe_message(self, data: dict, continuing: bool) -> None:
        event = data.get("type")
        if event == "message_start":
            self.envelope = data.get("message")
        elif event == "content_block_start":
            self.resumable = self.text_setup(data)
            self.block_open = self.resumable
        elif event == "content_block_delta":
            delta = data.get("delta") or {}
            self.resumable = (
                self.block_open
                and data.get("index") == 0
                and delta.get("type") == "text_delta"
                and isinstance(delta.get("text"), str)
            )
            if self.resumable:
                self.text.append(delta["text"])
        elif not continuing and event in ("message_delta", "message_stop"):
            self.resumable = False

    def observe_response(self, data: dict, continuing: bool) -> None:
        event = data.get("type")
        if event == "response.created":
            self.envelope = data.get("response")
        elif event == "response.output_item.added":
            self.resumable = self.text_setup(data)
            if not continuing and self.resumable:
                self.item = data["item"]
        elif event == "response.content_part.added":
            self.resumable = self.text_setup(data)
            if not continuing and self.resumable:
                self.part = data["part"]
        elif event == "response.output_text.delta":
            self.resumable = (
                self.part is not None
                and data.get("output_index") == 0
                and data.get("content_index") == 0
                and isinstance(data.get("delta"), str)
            )
            if self.resumable:
                self.text.append(data["delta"])
        elif isinstance(event, str) and event.startswith("response.") and not continuing:
            self.resumable = event == "response.in_progress"

    def continue_event(self, frame: SSEFrame) -> Iterator[bytes]:
        data = frame.data
        if data is None:
            return
        event = data.get("type")
        self.observe(data, continuing=True)
        if not self.resumable:
            raise DecodeRecompute(PD_RECOMPUTE_INTERRUPTED)
        if self.decode.kind == "messages":
            if event == "content_block_start":
                self.merged_attempt_text = True
            elif event == "content_block_delta":
                self.merged_attempt_text = True
                yield frame.raw
            else:
                raise DecodeRecompute(PD_RECOMPUTE_INTERRUPTED)
            return

        if event == "response.output_item.added":
            self.attempt_item_start = frame
        elif event == "response.content_part.added":
            self.merged_attempt_text = True
        elif event == "response.output_text.delta" and self.item is not None:
            self.merged_attempt_text = True
            data["item_id"] = self.item["id"]
            self.number(data)
            yield frame.encode()
        else:
            raise DecodeRecompute(PD_RECOMPUTE_INTERRUPTED)

    def finish_continued(self, frames: list[SSEFrame]) -> Iterator[bytes]:
        terminal = frames[-1]
        data = cast(dict, terminal.data)
        if data["type"] == "error":
            if self.decode.kind == "responses":
                self.number(data)
                yield terminal.encode()
            else:
                yield terminal.raw
            return
        if self.decode.kind == "messages":
            yield _encode_sse_event({"type": "content_block_stop", "index": 0})
            for frame in frames:
                event = frame.data
                if event is None:
                    yield frame.raw
                    continue
                index = event.get("index")
                if self.merged_attempt_text and event["type"] == "content_block_stop" and index == 0:
                    continue
                if isinstance(index, int) and not self.merged_attempt_text:
                    event["index"] = index + 1
                if event["type"] == "message_delta":
                    self.decode.patch_usage(event, (self.envelope or {}).get("usage"))
                yield frame.encode()
            return
        yield from self.finish_responses(frames, data)

    def finish_responses(self, frames: list[SSEFrame], terminal: dict) -> Iterator[bytes]:
        if self.envelope is None or self.item is None or self.part is None:
            raise DecodeRecompute(PD_RECOMPUTE_INTERRUPTED)
        response = terminal["response"]
        response["id"] = self.envelope["id"]
        if "created_at" in self.envelope:
            response["created_at"] = self.envelope["created_at"]
        if terminal["type"] == "response.failed":
            self.number(terminal)
            yield frames[-1].encode()
            return
        output = response.get("output")
        if not isinstance(output, list):
            raise DecodeRecompute(PD_RECOMPUTE_INTERRUPTED)
        native_item = output[0] if self.merged_attempt_text and output else {}
        native_parts = native_item.get("content", [])
        if native_item and (not native_parts or native_parts[0].get("type") != "output_text"):
            raise DecodeRecompute(PD_RECOMPUTE_INTERRUPTED)
        text = self.decode.prefix + "".join(self.text)
        part = {**self.part, **(native_parts[0] if native_parts else {}), "text": text}
        status = native_item.get("status") or ("incomplete" if response["status"] == "incomplete" else "completed")
        item = {
            **self.item,
            **native_item,
            "id": self.item["id"],
            "content": [part, *native_parts[1:]],
            "status": status,
        }
        response["output"] = [item, *output[int(self.merged_attempt_text) :]]
        self.decode.patch_usage(response)
        indices = {"output_index": 0, "content_index": 0, "item_id": item["id"]}
        for event in (
            {"type": "response.output_text.done", **indices, "text": text, "logprobs": part.get("logprobs") or []},
            {"type": "response.content_part.done", **indices, "part": part},
        ):
            self.number(event)
            yield _encode_sse_event(event)
        item_done = {"type": "response.output_item.done", "output_index": 0, "item": item}
        finished_item = False
        buffered = frames[:-1]
        if not self.merged_attempt_text and self.attempt_item_start is not None:
            buffered.insert(0, self.attempt_item_start)
        for frame in buffered:
            data = frame.data
            if data is None:
                yield frame.raw
                continue
            index = data.get("output_index")
            event_type = data.get("type")
            if self.merged_attempt_text and index == 0:
                if (
                    event_type in ("response.output_text.done", "response.content_part.done")
                    and data.get("content_index") == 0
                ):
                    continue
                if event_type == "response.output_item.done":
                    continue
                if "item_id" in data:
                    data["item_id"] = item["id"]
            elif isinstance(index, int):
                if not finished_item:
                    self.number(item_done)
                    yield _encode_sse_event(item_done)
                    finished_item = True
                if not self.merged_attempt_text:
                    data["output_index"] = index + 1
            self.number(data)
            yield frame.encode()
        if not finished_item:
            self.number(item_done)
            yield _encode_sse_event(item_done)
        self.number(terminal)
        yield frames[-1].encode()

    def reset_attempt(self) -> None:
        self.text.clear()
        self.merged_attempt_text = False
        self.attempt_item_start = None

    def number(self, data: dict) -> None:
        self.sequence += 1
        data["sequence_number"] = self.sequence

    def generated_text(self) -> str:
        if not self.resumable or not isinstance((self.envelope or {}).get("id"), str):
            return ""
        if self.decode.kind == "responses" and (self.item is None or self.part is None):
            return ""
        return "".join(self.text)


class NativeStreamingResponse(StreamingResponse):
    def __init__(self, first_frame: bytes, frames: AsyncGenerator[bytes, None], decode: "NativeDecode"):
        self.frames = frames
        self.decode = decode
        self.forward = self._forward(first_frame)
        super().__init__(self.forward, media_type="text/event-stream; charset=utf-8")

    async def _forward(self, first_frame: bytes) -> AsyncGenerator[bytes, None]:
        try:
            yield first_frame
            async for frame in self.frames:
                yield frame
        finally:
            await self._finish()

    async def _finish(self) -> None:
        with CancelScope(shield=True):
            try:
                await self.frames.aclose()
            finally:
                await self.decode.finish()

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            with CancelScope(shield=True):
                try:
                    await self.forward.aclose()
                finally:
                    await self._finish()


class NativeDecode:
    """Own one logical request and its current prefill/decode reservation."""

    def __init__(self, api: str, req_data: dict, request_length: int, info: InstanceInfo):
        self.api = api
        self.kind = _api_kind(api)
        self.req_data = req_data
        self.request_length = request_length
        self.info = info
        self.runtime = get_runtime()
        self.args = get_global_args()
        self.prefill_released = False
        self.retries = 0
        self.finished = False
        tools = req_data.get("tools") or []
        has_server_tools = self.kind == "responses" and (
            not isinstance(tools, list)
            or any(not isinstance(tool, dict) or tool.get("type") not in ("function", "custom") for tool in tools)
        )
        if self.kind == "messages":
            thinking = req_data.get("thinking") or {}
            output_config = req_data.get("output_config") or {}
            self.buffer_output = bool(thinking.get("type") in ("enabled", "adaptive") or output_config.get("format"))
        else:
            text_format = (req_data.get("text") or {}).get("format") or {}
            self.buffer_output = bool(
                has_server_tools
                or text_format.get("type") not in (None, "text")
                or req_data.get("include")
                or req_data.get("structured_outputs")
                or req_data.get("enable_response_messages")
            )
        self.history_key = "messages" if self.kind == "messages" else "input"
        self.history = req_data.get(self.history_key)
        self.prefix = ""
        self.output_tokens = 0
        self.first_progress: RecomputeProgress | None = None
        self.first_cached_tokens: int | None = None
        self.partial: dict | None = None

    async def release_prefill_once(self) -> None:
        if not self.prefill_released:
            await self.runtime.schedule("release_prefill_kv", self.info.prefiller_key, self.info.prefiller_score)
            self.prefill_released = True

    async def chunks(self) -> AsyncGenerator[bytes, None]:
        client = await self.runtime.get_client(ServerRole.DECODE, self.info.decoder_key)
        payload = self.req_data.copy()
        payload["vllm_xargs"] = {**(payload.get("vllm_xargs") or {}), PD_RECOMPUTE_ARG: PD_RECOMPUTE_VERSION}
        payload["vllm_xargs"].pop(PD_RECOMPUTE_RESUME, None)
        if not self.buffer_output:
            payload["vllm_xargs"][PD_RECOMPUTE_RESUME] = PD_RECOMPUTE_VERSION
        async with aclosing(
            stream_service_response(
                client,
                self.api,
                payload,
                request_id=self.info.request_id,
                max_retries=self.args.max_retries,
                base_delay=self.args.retry_delay,
                native=True,
            )
        ) as upstream:
            async for chunk in upstream:
                if chunk:
                    await self.release_prefill_once()
                yield chunk

    async def retry(self, recompute: DecodeRecompute, text: str | None = None) -> bool:
        if self.retries >= self.args.max_retries:
            return False
        if recompute.signal == PD_RECOMPUTE_INTERRUPTED:
            progress = recompute.progress
            if progress is None or not progress.remaining_tokens:
                return False
            budget_key = "max_tokens" if self.kind == "messages" else "max_output_tokens"
            budget = self.req_data.get(budget_key)
            if isinstance(budget, int) and progress.output_tokens + progress.remaining_tokens > budget:
                return False
            prefix = self.prefix + text if text is not None else ""
            history = _continue_native_prompt(self.kind, self.history, prefix) if prefix else None
            if self.first_progress is None:
                self.first_progress = progress
                self.first_cached_tokens = self.info.prefiller_cached_tokens
            if not prefix:
                self.partial = None
            elif self.partial is None:
                self.partial = recompute.partial
            if history is not None:
                self.req_data[self.history_key] = history
            elif self.prefix:
                self.req_data[self.history_key] = self.history
            self.prefix = prefix
            self.output_tokens += progress.output_tokens
            self.req_data[budget_key] = progress.remaining_tokens
            if isinstance(minimum := self.req_data.get("min_tokens"), int):
                self.req_data["min_tokens"] = max(0, minimum - progress.output_tokens)
            if self.kind == "messages":
                xargs = self.req_data.get("vllm_xargs") or {}
                if history is not None:
                    self.req_data["vllm_xargs"] = {**xargs, PD_CONTINUE_MESSAGE: PD_RECOMPUTE_VERSION}
                elif PD_CONTINUE_MESSAGE in xargs:
                    self.req_data["vllm_xargs"] = {k: v for k, v in xargs.items() if k != PD_CONTINUE_MESSAGE}
        self.retries += 1
        await self.release_prefill_once()
        self.req_data.pop("kv_transfer_params", None)
        self.req_data["request_id"] = next_req_id()
        if self.first_progress is not None:
            self.request_length = len(json.dumps(self.req_data).encode("utf-8"))
        self.info = await reassign_instances(
            self.api, self.req_data, self.request_length, self.info, previous_prefiller_kv_released=True
        )
        self.prefill_released = False
        return True

    def patch_usage(self, obj: dict, initial_usage: dict | None = None) -> bool:
        first = self.first_progress
        if first is None or not self.prefix:
            return write_cached_tokens_for_api(self.api, obj, self.info.prefiller_cached_tokens)
        usage = obj.get("usage")
        if not isinstance(usage, dict):
            return False
        usage["output_tokens"] = self.output_tokens + (usage.get("output_tokens") or 0)
        if self.kind == "messages" and "input_tokens" in usage:
            usage["input_tokens"] = first.input_tokens
            usage["cache_read_input_tokens"] = 0
            if "cache_creation_input_tokens" in usage or initial_usage:
                created = (initial_usage or {}).get("cache_creation_input_tokens") or 0
                usage["cache_creation_input_tokens"] = created
                usage["input_tokens"] -= created
        elif self.kind == "responses":
            usage["input_tokens"] = first.input_tokens
            usage["total_tokens"] = first.input_tokens + usage["output_tokens"]
            details = usage.get("input_tokens_details") or {}
            if details.get("input_tokens_per_turn"):
                details["input_tokens_per_turn"][0] = first.input_tokens
            obj["max_output_tokens"] = first.remaining_tokens + first.output_tokens
        write_cached_tokens_for_api(self.api, obj, self.first_cached_tokens)
        return True

    def merge_nonstream(self, obj: dict) -> None:
        if self.partial is None:
            return
        block = _native_text_block(self.kind, obj)
        if block is not None:
            block["text"] = self.prefix + block["text"]
            return
        key = "content" if self.kind == "messages" else "output"
        if not isinstance(obj.get(key), list):
            raise DecodeRecompute(PD_RECOMPUTE_INTERRUPTED)
        partial = copy.deepcopy(self.partial[key][0])
        if self.kind == "messages":
            partial["text"] = self.prefix
        else:
            partial["content"][0]["text"] = self.prefix
            partial["status"] = "incomplete" if obj["status"] == "incomplete" else "completed"
        obj[key].insert(0, partial)

    async def finish(self) -> None:
        if not self.finished:
            with CancelScope(shield=True):
                await _finish_instance(self.runtime, self.info, release_prefill_kv=not self.prefill_released)
            self.finished = True

    async def nonstream(self) -> Response:
        while True:
            try:
                async with aclosing(self.chunks()) as chunks:
                    body = b"".join([chunk async for chunk in chunks])
                obj = _json_object(body)
                if recompute := _decode_recompute(obj):
                    raise recompute
                if obj is None:
                    raise IncompleteDecodeResponse("Decode backend did not return a complete JSON response.")
                self.merge_nonstream(obj)
                if self.patch_usage(obj, (self.partial or {}).get("usage")) or self.partial is not None:
                    body = encode_response_chunk(obj, False)
                return Response(body, media_type="application/json")
            except DecodeRecompute as exc:
                block = (
                    _native_text_block(self.kind, exc.partial)
                    if exc.partial is not None and not self.buffer_output
                    else None
                )
                if not await self.retry(exc, block["text"] if block else None):
                    raise

    async def stream(self) -> AsyncGenerator[bytes, None]:
        state = NativeStream(self)
        try:
            while True:
                try:
                    async with (
                        aclosing(self.chunks()) as chunks,
                        aclosing(_native_sse_frames(chunks, state)) as frames,
                    ):
                        async for frame in frames:
                            state.sent = True
                            yield frame
                    return
                except DecodeRecompute as exc:
                    text = state.generated_text()
                    if state.sent and not (self.prefix or text):
                        raise
                    if not await self.retry(exc, text):
                        raise
                    if state.sent:
                        state.reset_attempt()
                    else:
                        state = NativeStream(self)
        except DecodeRecompute as exc:
            if not state.sent:
                raise
            signal = PD_RECOMPUTE_INTERRUPTED if self.prefix else exc.signal
            yield _native_stream_error(self.api, signal, state.sequence + 1)
        except (httpx.RequestError, httpx.HTTPStatusError, RetryPrefillMissingKV):
            if not state.sent or self.first_progress is None:
                raise
            yield _native_stream_error(self.api, PD_RECOMPUTE_INTERRUPTED, state.sequence + 1)


async def _handle_native_decode(api: str, req_data: dict, request_length: int, instance_info: InstanceInfo) -> Response:
    decode = NativeDecode(api, req_data, request_length, instance_info)
    if not req_data.get("stream", False):
        try:
            return await decode.nonstream()
        finally:
            await decode.finish()

    frames = decode.stream()
    try:
        first_frame = await anext(frames, None)
    except (Exception, asyncio.CancelledError):
        await decode.finish()
        raise
    if first_frame is None:
        await decode.finish()
        return Response(b"", media_type="text/event-stream; charset=utf-8")

    return NativeStreamingResponse(first_frame, frames, decode)


async def handle_completions_impl(api: str, request: Request):
    runtime = get_runtime()
    args = get_global_args()
    request_released = False
    try:
        req_data = await request.json()
        if _api_kind(api) == "responses":
            # Reject before scheduling: P must finish before D starts, and
            # response history is local to each backend instance.
            unsupported = None
            if req_data.get("background"):
                unsupported = "background"
            elif req_data.get("previous_response_id") is not None:
                unsupported = "previous_response_id"
            if unsupported is not None:
                return JSONResponse(
                    {
                        "error": {
                            "message": (
                                f"The PD proxy does not support {unsupported}. "
                                "Use synchronous requests with explicit input history."
                            ),
                            "type": "invalid_request_error",
                            "param": unsupported,
                            "code": "unsupported_parameter",
                        }
                    },
                    status_code=400,
                )
        if args.enable_prefill_affinity and (session_id := request.headers.get("X-Session-Id")):
            req_data["user"] = session_id
        if request_id := request.headers.get("X-Request-Id"):
            req_data["request_id"] = request_id
        req_body = await request.body()
        request_length = len(req_body)
        if _api_kind(api) != "openai":
            native_info = await assign_instances(api, req_data, request_length, is_initial_request=True)
            try:
                return await _handle_native_decode(api, req_data, request_length, native_info)
            except DecodeRecompute as exc:
                return JSONResponse(
                    _native_decode_error(api, exc.signal, _recompute_failure_message(exc.signal)),
                    status_code=503 if exc.signal == PD_RECOMPUTE_RETRY else 502,
                )
            except (httpx.RequestError, RetryPrefillMissingKV, IncompleteDecodeResponse) as exc:
                logger.warning("Native decode failed for %s: %s", native_info.request_id, exc)
                return JSONResponse(
                    _native_decode_error(api, "backend_unavailable", "The backend did not complete the request."),
                    status_code=502,
                )
            except httpx.HTTPStatusError as exc:
                return Response(
                    content=exc.response.content, status_code=exc.response.status_code, media_type="application/json"
                )
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
            max_retries=args.max_retries,
            base_delay=args.retry_delay,
        )
        try:
            preopened_first = await preopened_gen.__anext__()
        except asyncio.CancelledError:
            await _finish_instance(runtime, instance_info, release_prefill_kv=True)
            request_released = True
            raise
        except StopAsyncIteration:
            preopened_first = b""
        except httpx.HTTPStatusError as exc:
            await _finish_instance(runtime, instance_info, release_prefill_kv=True)
            request_released = True
            return Response(
                content=exc.response.content, status_code=exc.response.status_code, media_type="application/json"
            )
        except httpx.RequestError as exc:
            await _finish_instance(runtime, instance_info, release_prefill_kv=True)
            request_released = True
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

        async def generate_stream():
            nonlocal instance_info
            nonlocal request_released
            nonlocal preopened_gen, preopened_first
            generated_token = ""
            released_kv = False
            retry_count = 0
            retry = True
            completion_tokens = 0
            reported_prefiller_cached_tokens = instance_info.prefiller_cached_tokens

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
                                messages[0]["content"] = (
                                    origin_prompt
                                    + ([{"type": "text", "text": generated_token}] if generated_token else [])
                                    if isinstance(origin_prompt, list)
                                    else (origin_prompt or "") + generated_token
                                )
                            else:
                                req_data["prompt"] = origin_prompt + generated_token
                            req_data["max_tokens"] = origin_max_tokens - completion_tokens + retry_count
                            tmp_request_length = len(json.dumps(req_data).encode("utf-8"))
                            instance_info = await reassign_instances(
                                api,
                                req_data,
                                tmp_request_length,
                                instance_info,
                                previous_prefiller_kv_released=released_kv,
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
    except Exception as e:
        import traceback

        exc_info = sys.exc_info()
        print(f"Error occurred in disagg prefill proxy server - {api} endpoint")
        print("".join(traceback.format_exception(*exc_info)))
        if not request_released and "instance_info" in locals():
            await _finish_instance(runtime, instance_info, release_prefill_kv=True)
            request_released = True
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


@app.post("/v1/messages")
@with_cancellation
async def handle_anthropic_messages(request: Request):
    return await handle_completions_impl("/messages", request)


@app.post("/v1/responses")
@with_cancellation
async def handle_openai_responses(request: Request):
    return await handle_completions_impl("/responses", request)


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
