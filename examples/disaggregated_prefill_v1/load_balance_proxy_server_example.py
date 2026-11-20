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
#   On recompute the proxy selects P/D again with the same scheduling policy as Chat.
#   It retries with the remaining token budget and continues plain text.
#   After visible output, retries forward a new native event sequence with backend-local IDs and indices.
#   Only visible text is fed back to the model.
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
PD_CONTINUE_MESSAGE = "ascend_pd_continue_message"
PD_RECOMPUTE_RETRY = "ascend_pd_recompute_before_output"
PD_RECOMPUTE_INTERRUPTED = "ascend_pd_recompute_after_output"
NATIVE_STREAM_END_EVENTS = ("message_stop", "response.completed", "response.incomplete", "response.failed", "error")

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


def extract_cached_tokens(response_json: dict, *, details_key: str = "prompt_tokens_details") -> int | None:
    usage = response_json.get("usage") or {}
    details = usage.get(details_key) or {}
    cached = details.get("cached_tokens")
    return cached if isinstance(cached, int) else 0


def _write_cached_details(usage: dict, key: str, cached_tokens: int) -> dict:
    details = usage.get(key)
    if not isinstance(details, dict):
        details = usage[key] = {}
    details["cached_tokens"] = cached_tokens
    return details


def update_cached_tokens_in_chunk(chunk_json: dict, cached_tokens: int | None) -> bool:
    usage = chunk_json.get("usage")
    if cached_tokens is None or not isinstance(usage, dict):
        return False
    _write_cached_details(usage, "prompt_tokens_details", cached_tokens)
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
    endpoint = api.rstrip("/").rsplit("/", 1)[-1]
    return endpoint if endpoint in ("messages", "responses") else "openai"


def build_prefill_request_for_api(api: str, req_data: dict) -> dict:
    payload = build_prefill_request(req_data)
    if _api_kind(api) == "responses":
        payload["max_output_tokens"] = 1
        for name in ("max_tokens", "min_tokens", "max_completion_tokens"):
            payload.pop(name, None)
    return payload


def extract_cached_tokens_for_api(api: str, response_json: dict) -> int | None:
    kind = _api_kind(api)
    if kind == "messages":
        usage = response_json.get("usage") or {}
        cached = usage.get("cache_read_input_tokens")
        return cached if isinstance(cached, int) else 0
    key = "input_tokens_details" if kind == "responses" else "prompt_tokens_details"
    return extract_cached_tokens(response_json, details_key=key)


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
    details = _write_cached_details(usage, "input_tokens_details", cached_tokens)
    cached_per_turn = details.get("cached_tokens_per_turn")
    # An empty list means per-turn accounting is unavailable; preserve it.
    if isinstance(cached_per_turn, list) and cached_per_turn:
        # Only the first turn is prefetched by P. Later tool turns run on D.
        details["cached_tokens_per_turn"] = [cached_tokens, *cached_per_turn[1:]]
        details["cached_tokens"] = sum(details["cached_tokens_per_turn"])
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
    complete: bool = True

    @classmethod
    def parse(cls, frame: bytes, *, complete: bool = True) -> "SSEFrame":
        lines = frame.splitlines(keepends=True)
        data_indices = [index for index, line in enumerate(lines) if line.startswith(b"data:")]
        payload = b"\n".join(lines[index][5:].strip() for index in data_indices)
        event = next((line[6:].strip() for line in reversed(lines) if line.startswith(b"event:")), None)
        return cls(frame, lines, data_indices, _json_object(payload), event, complete)

    def encode(self) -> bytes:
        first = self.data_indices[0]
        ending = self.lines[first][len(self.lines[first].rstrip(b"\r\n")) :]
        patched = b"data: " + json.dumps(self.data).encode("utf-8") + ending
        return b"".join(
            patched if index == first else line
            for index, line in enumerate(self.lines)
            if index == first or index not in self.data_indices
        )


class DecodeRecompute(Exception):
    def __init__(self, signal: str, progress: dict | None = None, partial: dict | None = None):
        super().__init__(signal)
        self.signal = signal
        self.progress = progress
        self.partial = partial


def _decode_recompute(obj: dict | None) -> DecodeRecompute | None:
    if obj is None:
        return None
    transfer = obj.get("kv_transfer_params")
    if isinstance(transfer, dict) and transfer.get(PD_RECOMPUTE_ARG) == PD_RECOMPUTE_INTERRUPTED:
        return DecodeRecompute(PD_RECOMPUTE_INTERRUPTED, transfer, obj)
    error = obj.get("error")
    signals = (PD_RECOMPUTE_RETRY, PD_RECOMPUTE_INTERRUPTED)
    if isinstance(error, dict):
        message = error.get("message")
    elif obj.get("type") == "error":
        message = obj.get("message")
        code = obj.get("code")
        if code in signals and (not isinstance(message, str) or not message.startswith(f"{code}:")):
            message = code
    else:
        return None
    if isinstance(message, str):
        signal, _, detail = message.partition(":")
        if signal in signals:
            progress = _json_object(detail.encode()) if detail else None
            return DecodeRecompute(signal, progress)
    return None


class IncompleteBackendResponse(Exception):
    """A P/D backend response lacks required data."""


async def _split_sse_frames(chunks: AsyncIterator[bytes]) -> AsyncGenerator[SSEFrame, None]:
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
            yield SSEFrame.parse(frame)
    if stream_buf:
        yield SSEFrame.parse(bytes(stream_buf), complete=False)


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
    if args.enable_prefill_affinity:
        prefiller = await runtime.schedule(pick_prefill, prefiller_score, conversation_affinity_key(req_data))
    else:
        prefiller = await runtime.schedule(pick_prefill, prefiller_score)
    prefiller_key = prefiller["key"]
    decoder_key = None

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
        prefiller_cached_tokens = extract_cached_tokens_for_api(api, response_json)
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


def _encode_sse_event(data: dict) -> bytes:
    return b"event: " + data["type"].encode() + b"\n" + encode_response_chunk(data, True)


def _native_stream_error(api: str, signal: str, sequence: int) -> bytes:
    error = _native_decode_error(api, signal, _recompute_failure_message(signal))
    if _api_kind(api) != "messages":
        error = {**error["error"], "type": "error", "sequence_number": sequence}
    return _encode_sse_event(error)


def _delta_text(event: dict) -> str:
    delta = event.get("delta")
    if event.get("type") == "content_block_delta" and isinstance(delta, dict) and delta.get("type") == "text_delta":
        delta = delta.get("text")
    elif event.get("type") != "response.output_text.delta":
        return ""
    return delta if isinstance(delta, str) else ""


def _native_decode_error(api: str, signal: str, message: str) -> dict:
    if _api_kind(api) == "messages":
        error_type = "overloaded_error" if signal == PD_RECOMPUTE_RETRY else "api_error"
        return {"type": "error", "error": {"type": error_type, "message": message}}
    return {"error": {"type": "server_error", "code": signal, "message": message, "param": None}}


def _recompute_failure_message(signal: str) -> str:
    if signal == PD_RECOMPUTE_INTERRUPTED:
        return "Decode recompute interrupted generation; the proxy cannot resume the partial response."
    return "Decode recompute could not restart this request. Retry the original request if it is safe to repeat."


def _new_response_message(status: str, content: list[dict]) -> dict:
    return {
        "type": "message",
        "id": f"msg_{uuid.uuid4().hex}",
        "role": "assistant",
        "status": status,
        "content": content,
    }


def _visible_blocks(kind: str, obj: dict) -> Iterator[dict]:
    if kind == "messages":
        blocks = obj.get("content", [])
        expected = "text"
    else:
        blocks = [
            part for item in obj.get("output", []) if item.get("type") == "message" for part in item.get("content", [])
        ]
        expected = "output_text"
    for block in blocks:
        if block.get("type") == expected and isinstance(block.get("text"), str):
            yield block


def _continue_native_prompt(kind: str, history: Any, prefix: str) -> list | None:
    if isinstance(history, str):
        history = [{"role": "user", "content": history}]
    if not isinstance(history, list):
        return None
    last = history[-1] if history and isinstance(history[-1], dict) else {}
    assistant = last.get("role") == "assistant"
    if kind == "messages":
        if not assistant:
            return [*history, {"role": "assistant", "content": prefix}]
        content = last.get("content")
        if isinstance(content, str):
            return [*history[:-1], {**last, "content": content + prefix}]
        if isinstance(content, list):
            return [*history[:-1], {**last, "content": [*content, {"type": "text", "text": prefix}]}]
        return None

    part = {"type": "output_text", "text": prefix, "annotations": []}
    if assistant and last.get("type") == "message" and last.get("status") in ("in_progress", "incomplete"):
        content = last.get("content")
        if isinstance(content, list) and len(content) == 1 and content[0].get("type") == "output_text":
            block = content[0]
            return [*history[:-1], {**last, "content": [{**block, "text": block["text"] + prefix}]}]
    # The Responses renderer reads the first text part of a partial assistant item.
    return [*history, _new_response_message("in_progress", [part])]


class NativeStream:
    """Forward backend frames and collect only visible text for the retry input."""

    def __init__(self, decode: "DecodeRequest"):
        self.decode = decode
        self.sent = False
        self.sequence = -1

    async def frames(self, chunks: AsyncIterator[bytes]) -> AsyncGenerator[bytes, None]:
        resumed = self.sent
        pending: list[bytes] = []
        pending_bytes = 0
        committed = False
        terminal = False
        async with aclosing(_split_sse_frames(chunks)) as frames:
            async for frame in frames:
                data = frame.data or {}
                event = data.get("type")
                if (event == "error" or frame.event == b"error") and (exc := _decode_recompute(data)):
                    if committed and exc.signal == PD_RECOMPUTE_RETRY:
                        raise DecodeRecompute(PD_RECOMPUTE_INTERRUPTED)
                    raise exc
                if event == "content_block_start" and data.get("content_block", {}).get("type") == "text":
                    self.decode.prefix += data["content_block"].get("text", "")
                self.decode.prefix += _delta_text(data)
                patched = write_cached_tokens_for_api(self.decode.api, data, self.decode.info.prefiller_cached_tokens)
                raw = frame.encode() if patched else frame.raw
                keepalive = event == "ping" or (
                    frame.data is None and all(not line.strip() or line.startswith(b":") for line in frame.lines)
                )
                if not committed and (
                    keepalive or event in ("message_start", "response.created", "response.in_progress")
                ):
                    if not keepalive or pending_bytes + len(raw) <= 64 * 1024:
                        pending.append(raw)
                        pending_bytes += len(raw)
                    continue
                for header in pending:
                    yield header
                pending.clear()
                committed = True
                self.sequence = data.get("sequence_number", self.sequence)
                terminal |= event in NATIVE_STREAM_END_EVENTS
                yield raw
        if resumed and not terminal:
            raise IncompleteBackendResponse("Decode stream ended without a terminal event.")
        for header in pending:
            yield header


class DecodeStreamingResponse(StreamingResponse):
    def __init__(self, first_frame: bytes, frames: AsyncGenerator[bytes, None], media_type: str):
        self.frames = frames
        self.forward = self._forward(first_frame)
        super().__init__(self.forward, media_type=media_type)

    async def _forward(self, first_frame: bytes) -> AsyncGenerator[bytes, None]:
        async with aclosing(self.frames):
            if first_frame:
                yield first_frame
            async for frame in self.frames:
                yield frame

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            with CancelScope(shield=True):
                async with aclosing(self.frames):
                    await self.forward.aclose()


class ChatOutput:
    """Decode Chat/Completions choices and prepare their continuation prompt."""

    def __init__(self, req_data: dict, cached_tokens: int | None):
        self.request = req_data
        self.messages = req_data.get("messages")
        self.prompt = req_data.get("prompt", self.messages[0].get("content", "") if self.messages else "")
        self.max_tokens = req_data.get("max_tokens", 16)
        self.cached_tokens = cached_tokens
        self.text = ""
        self.tokens = 0
        self.retries = 0
        self.started = False

    async def frames(self, chunks: AsyncIterator[bytes], stream: bool) -> AsyncGenerator[bytes, None]:
        async for chunk in chunks:
            if not self.started:
                # Let the HTTP handler commit the response after the first upstream read,
                # even when that chunk is a recompute marker and produces no client text.
                self.started = True
                yield b""
            try:
                line = chunk.decode("utf-8").strip()
                if not line:
                    continue
                is_sse = line.startswith("data: ")
                obj = json.loads(line.removeprefix("data: ") if is_sse else line)
            except (UnicodeDecodeError, json.JSONDecodeError):
                yield chunk
                continue
            choices = obj.get("choices", [])
            if (not choices or not stream) and update_cached_tokens_in_chunk(obj, self.cached_tokens):
                chunk = encode_response_chunk(obj, is_sse)
            if not choices:
                yield chunk
                continue
            choice = choices[0]
            self.text += (
                (choice.get("delta") or {}).get("content")
                or (choice.get("message") or {}).get("content")
                or choice.get("text")
                or ""
            )
            self.tokens += 1 if stream else obj.get("usage", {}).get("completion_tokens", 0)
            if choice.get("stop_reason") == "recomputed":
                self.retries += 1
                if self.messages is not None:
                    self.messages[0]["content"] = (
                        self.prompt + ([{"type": "text", "text": self.text}] if self.text else [])
                        if isinstance(self.prompt, list)
                        else (self.prompt or "") + self.text
                    )
                else:
                    self.request["prompt"] = self.prompt + self.text
                self.request["max_tokens"] = self.max_tokens - self.tokens + self.retries
                raise DecodeRecompute(PD_RECOMPUTE_RETRY)
            if self.retries and not stream:
                if self.messages is not None:
                    choice["message"]["content"] = self.text
                else:
                    choice["text"] = self.text
                chunk = encode_response_chunk(obj, is_sse)
            yield chunk


class DecodeRequest:
    """Keep one scheduled request alive across decode attempts."""

    def __init__(self, api: str, req_data: dict, info: InstanceInfo):
        self.api = api
        self.kind = _api_kind(api)
        self.req_data = req_data
        self.info = info
        self.runtime = get_runtime()
        self.args = get_global_args()
        self.history_key = "messages" if self.kind == "messages" else "input"
        self.history = req_data.get(self.history_key)
        self.prefill_released = False
        self.prefix = ""

    async def release_prefill_once(self) -> None:
        if not self.prefill_released:
            await self.runtime.schedule("release_prefill_kv", self.info.prefiller_key, self.info.prefiller_score)
            self.prefill_released = True

    async def chunks(self) -> AsyncGenerator[bytes, None]:
        client = await self.runtime.get_client(ServerRole.DECODE, self.info.decoder_key)
        payload = self.req_data
        if self.kind != "openai":
            payload = {
                **payload,
                "vllm_xargs": {**(payload.get("vllm_xargs") or {}), PD_RECOMPUTE_ARG: PD_RECOMPUTE_VERSION},
            }
        async with aclosing(
            stream_service_response(
                client,
                self.api,
                payload,
                request_id=self.info.request_id,
                max_retries=self.args.max_retries,
                base_delay=self.args.retry_delay,
                native=self.kind != "openai",
            )
        ) as upstream:
            async for chunk in upstream:
                if chunk:
                    await self.release_prefill_once()
                yield chunk

    async def retry(self, recompute: DecodeRecompute) -> bool:
        if recompute.signal == PD_RECOMPUTE_INTERRUPTED:
            progress = recompute.progress or {}
            remaining = progress.get("remaining_tokens")
            if type(remaining) is not int or remaining <= 0:
                return False
            budget_key = "max_tokens" if self.kind == "messages" else "max_output_tokens"
            if self.prefix:
                history = _continue_native_prompt(self.kind, self.history, self.prefix)
                if history is None:
                    return False
                self.req_data[self.history_key] = history
                if self.kind == "messages":
                    self.req_data["vllm_xargs"] = {
                        **(self.req_data.get("vllm_xargs") or {}),
                        PD_CONTINUE_MESSAGE: PD_RECOMPUTE_VERSION,
                    }
            self.req_data[budget_key] = remaining

        await self.release_prefill_once()
        if self.kind != "openai":
            self.req_data.pop("kv_transfer_params", None)
        request_length = len(json.dumps(self.req_data).encode("utf-8"))
        self.info = await reassign_instances(
            self.api, self.req_data, request_length, self.info, previous_prefiller_kv_released=True
        )
        self.prefill_released = False
        return True

    def json_body(self, body: bytes) -> bytes:
        obj = _json_object(body)
        if recompute := _decode_recompute(obj):
            raise recompute
        if obj is None:
            raise IncompleteBackendResponse("Decode backend did not return a complete JSON response.")
        if self.prefix:
            if self.kind == "messages":
                blocks = obj["content"]
                if not blocks or blocks[0].get("type") != "text":
                    blocks.insert(0, {"type": "text", "text": ""})
            else:
                output = obj["output"]
                if not output or output[0].get("type") != "message":
                    status = "incomplete" if obj.get("status") == "incomplete" else "completed"
                    output.insert(0, _new_response_message(status, []))
                blocks = output[0]["content"]
                if not blocks or blocks[0].get("type") != "output_text":
                    blocks.insert(0, {"type": "output_text", "text": "", "annotations": []})
            blocks[0]["text"] = self.prefix + blocks[0]["text"]
        if write_cached_tokens_for_api(self.api, obj, self.info.prefiller_cached_tokens) or self.prefix:
            return encode_response_chunk(obj, False)
        return body

    async def response_chunks(self, stream: bool) -> AsyncGenerator[bytes, None]:
        state = NativeStream(self) if stream and self.kind != "openai" else None
        chat = None
        try:
            if self.kind == "openai":
                chat = ChatOutput(self.req_data, self.info.prefiller_cached_tokens)
            while True:
                try:
                    async with aclosing(self.chunks()) as chunks:
                        if chat is not None:
                            output = chat.frames(chunks, stream)
                        elif state is not None:
                            output = state.frames(chunks)
                        else:
                            body = b"".join([chunk async for chunk in chunks])
                            yield self.json_body(body)
                            return
                        async with aclosing(output) as frames:
                            async for frame in frames:
                                if state is not None:
                                    state.sent = True
                                yield frame
                        return
                except DecodeRecompute as exc:
                    if exc.partial:
                        self.prefix += "".join(block["text"] for block in _visible_blocks(self.kind, exc.partial))
                    if not await self.retry(exc):
                        raise
                    if state is not None and not state.sent:
                        state = NativeStream(self)
        except Exception as exc:
            if chat is not None and chat.started:
                logger.exception("Chat decode interrupted after response started")
                return
            if (
                state is None
                or not state.sent
                or not isinstance(
                    exc, (DecodeRecompute, httpx.RequestError, httpx.HTTPStatusError, IncompleteBackendResponse)
                )
            ):
                raise
            yield _native_stream_error(self.api, PD_RECOMPUTE_INTERRUPTED, state.sequence + 1)
        finally:
            with CancelScope(shield=True):
                await _finish_instance(self.runtime, self.info, release_prefill_kv=not self.prefill_released)


async def _handle_decode(api: str, req_data: dict, instance_info: InstanceInfo) -> Response:
    decode = DecodeRequest(api, req_data, instance_info)
    stream = bool(req_data.get("stream", False))
    frames = decode.response_chunks(stream)
    if not stream and decode.kind != "openai":
        async with aclosing(frames):
            return Response(await anext(frames), media_type="application/json")

    first_frame = await anext(frames, None)
    media_type = "text/event-stream; charset=utf-8" if stream else "application/json"
    if first_frame is None:
        return Response(b"", media_type=media_type)
    return DecodeStreamingResponse(first_frame, frames, media_type)


async def handle_completions_impl(api: str, request: Request):
    args = get_global_args()
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
        info = await assign_instances(api, req_data, request_length, is_initial_request=True)
        try:
            return await _handle_decode(api, req_data, info)
        except DecodeRecompute as exc:
            return JSONResponse(
                _native_decode_error(api, exc.signal, _recompute_failure_message(exc.signal)),
                status_code=503 if exc.signal == PD_RECOMPUTE_RETRY else 502,
            )
        except (httpx.RequestError, IncompleteBackendResponse) as exc:
            logger.warning("Decode failed for %s: %s", info.request_id, exc)
            if _api_kind(api) == "openai":
                error = {
                    "error": {
                        "message": f"Decode backend request to {api} failed: {exc}",
                        "type": "upstream_error",
                        "code": "decode_backend_unavailable",
                    }
                }
            else:
                error = _native_decode_error(api, "backend_unavailable", "The backend did not complete the request.")
            return JSONResponse(error, status_code=502)
        except httpx.HTTPStatusError as exc:
            return Response(
                content=exc.response.content, status_code=exc.response.status_code, media_type="application/json"
            )
    except Exception as e:
        import traceback

        exc_info = sys.exc_info()
        print(f"Error occurred in disagg prefill proxy server - {api} endpoint")
        print("".join(traceback.format_exception(*exc_info)))
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
