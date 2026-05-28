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
#     --host 0.0.0.0 --port 9000 --workers 8 \
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
import threading
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from multiprocessing.managers import BaseManager
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

logger = logging.getLogger(__name__)

try:
    import uvloop

    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
except ImportError:
    pass


@dataclass
class InstanceType:
    PREFILL: str = "prefill"
    DECODE: str = "decode"


@dataclass(frozen=True)
class ServerState:
    host: str
    port: int

    def __str__(self) -> str:
        return f"{self.host}:{self.port}"


@dataclass
class InstanceInfo:
    request_id: str
    prefiller_key: str
    prefiller_score: float
    prefiller: dict[str, Any]
    decoder_key: str
    decoder_score: float
    decoder: dict[str, Any]


TAINT_PRIORITY = 1e15
MANAGER_CONFIG_ENV = "LB_PROXY_MANAGER_CONFIG"
ARGS_CONFIG_ENV = "LB_PROXY_ARGS"

global_args = None
shared_scheduler = None
runtime = None


def setup_logging(log_level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        force=True,
    )


def normalize_host(host: str) -> str:
    return host.replace("localhost", "0.0.0.0").replace("127.0.0.1", "0.0.0.0")


def server_key(host: str, port: int) -> str:
    return f"{normalize_host(host)}:{int(port)}"


def build_base_url(host: str, port: int) -> str:
    url = f"http://{host}:{port}/v1"
    try:
        ip = ipaddress.ip_address(host)
        if isinstance(ip, ipaddress.IPv6Address):
            url = f"http://[{host}]:{port}/v1"
    except Exception:
        pass
    return url


class SharedProxyScheduler:
    """Centralized mutable scheduling state shared by all uvicorn workers."""

    def __init__(self, prefiller_instances, decoder_instances):
        self._lock = threading.RLock()
        self.request_num = 0
        self.waiting_nodes: dict[str, tuple[str, tuple[str, int], int]] = {}
        self.tainted_prefillers: set[str] = set()
        self.tainted_decoders: set[str] = set()
        self.prefillers: dict[str, dict[str, Any]] = {}
        self.decoders: dict[str, dict[str, Any]] = {}
        self.prefiller_heap: list[tuple[float, int, str]] = []
        self.decoder_heap: list[tuple[float, int, str]] = []
        self._ordinal = 0

        for host, port in prefiller_instances:
            self._add_server_no_lock(InstanceType.PREFILL, host, port)
        for host, port in decoder_instances:
            self._add_server_no_lock(InstanceType.DECODE, host, port)

    def _server_map(self, instance_type: str) -> dict[str, dict[str, Any]]:
        return self.prefillers if instance_type == InstanceType.PREFILL else self.decoders

    def _heap(self, instance_type: str) -> list[tuple[float, int, str]]:
        return self.prefiller_heap if instance_type == InstanceType.PREFILL else self.decoder_heap

    def _next_ordinal(self) -> int:
        ordinal = self._ordinal
        self._ordinal += 1
        return ordinal

    def _priority_no_lock(self, instance_type: str, key: str) -> float:
        server = self._server_map(instance_type)[key]
        if instance_type == InstanceType.PREFILL:
            if key in self.tainted_prefillers:
                return TAINT_PRIORITY
            return float(server["active_tokens"]) + float(server["active_kv_cache"]) * 0.3
        if key in self.tainted_decoders:
            return TAINT_PRIORITY
        return float(server["active_tokens"])

    def _rebuild_heap_no_lock(self, instance_type: str) -> None:
        heap = []
        for key, state in self._server_map(instance_type).items():
            heap.append((self._priority_no_lock(instance_type, key), state["ordinal"], key))
        heapq.heapify(heap)
        if instance_type == InstanceType.PREFILL:
            self.prefiller_heap = heap
        else:
            self.decoder_heap = heap

    def _add_server_no_lock(self, instance_type: str, host: str, port: int) -> bool:
        key = server_key(host, port)
        servers = self._server_map(instance_type)
        if key in servers:
            return False
        servers[key] = {
            "host": host,
            "port": int(port),
            "active_tokens": 0.0,
            "active_kv_cache": 0.0,
            "ordinal": self._next_ordinal(),
        }
        heapq.heappush(self._heap(instance_type), (0.0, servers[key]["ordinal"], key))
        return True

    def get_snapshot(self) -> dict[str, list[dict[str, Any]]]:
        with self._lock:
            return {
                "prefill_instances": [
                    {"host": state["host"], "port": state["port"]}
                    for _, state in sorted(self.prefillers.items(), key=lambda item: item[1]["ordinal"])
                ],
                "decode_instances": [
                    {"host": state["host"], "port": state["port"]}
                    for _, state in sorted(self.decoders.items(), key=lambda item: item[1]["ordinal"])
                ],
            }

    def print_status(self, msg: str) -> None:
        snapshot = self.get_snapshot()
        status = {
            "prefill_instances": [f"{s['host']}:{s['port']}" for s in snapshot["prefill_instances"]],
            "decode_instances": [f"{s['host']}:{s['port']}" for s in snapshot["decode_instances"]],
        }
        print(f"{msg} Status: {status}")

    def healthcheck(self) -> dict[str, Any]:
        with self._lock:
            return {
                "status": "ok",
                "prefill_instances": len(self.prefillers),
                "decode_instances": len(self.decoders),
                "request_num": self.request_num,
            }

    def request_started(self) -> None:
        with self._lock:
            self.request_num += 1

    def request_finished(self) -> None:
        with self._lock:
            self.request_num = max(0, self.request_num - 1)

    def next_req_id(self) -> str:
        return str(uuid.uuid4())

    def calculate_prefill_scores(self, request_length: int) -> float:
        length_score = request_length / 4.0
        return length_score * 0.0345 + 120.0745

    def calculate_decode_scores(self, request_length: int) -> float:
        return request_length

    def select_prefiller(self, token_count: float) -> dict[str, Any]:
        with self._lock:
            if not self.prefiller_heap:
                raise RuntimeError("No prefiller servers available")
            _, _, key = heapq.heappop(self.prefiller_heap)
            state = self.prefillers[key]
            state["active_tokens"] += token_count
            state["active_kv_cache"] += token_count
            heapq.heappush(
                self.prefiller_heap,
                (self._priority_no_lock(InstanceType.PREFILL, key), state["ordinal"], key),
            )
            return {"key": key, "host": state["host"], "port": state["port"]}

    def release_prefiller(self, key: str, token_count: float) -> None:
        with self._lock:
            if key not in self.prefillers:
                return
            self.prefillers[key]["active_tokens"] -= token_count
            self._rebuild_heap_no_lock(InstanceType.PREFILL)

    def release_prefiller_kv(self, key: str, token_count: float) -> None:
        with self._lock:
            if key not in self.prefillers:
                return
            state = self.prefillers[key]
            state["active_kv_cache"] = max(0.0, float(state["active_kv_cache"]) - token_count)
            self._rebuild_heap_no_lock(InstanceType.PREFILL)

    def select_decoder(self, token_count: float) -> dict[str, Any]:
        with self._lock:
            if not self.decoder_heap:
                raise RuntimeError("No decoder servers available")
            _, _, key = heapq.heappop(self.decoder_heap)
            state = self.decoders[key]
            state["active_tokens"] += token_count
            heapq.heappush(
                self.decoder_heap,
                (self._priority_no_lock(InstanceType.DECODE, key), state["ordinal"], key),
            )
            return {"key": key, "host": state["host"], "port": state["port"]}

    def release_decoder(self, key: str, token_count: float) -> None:
        with self._lock:
            if key not in self.decoders:
                return
            self.decoders[key]["active_tokens"] -= token_count
            self._rebuild_heap_no_lock(InstanceType.DECODE)

    def get_waiting_nodes(self) -> dict[str, tuple[str, tuple[str, int], int]]:
        with self._lock:
            return dict(self.waiting_nodes)

    def add_instances(self, instance_type: str, instances: list[tuple[str, int]]) -> tuple[list[str], list[str]]:
        added_nodes: list[str] = []
        waiting_nodes: list[str] = []
        with self._lock:
            servers = self._server_map(instance_type)
            for host, port in instances:
                key = server_key(host, port)
                if key in servers or key in self.waiting_nodes:
                    continue
                self.waiting_nodes[key] = (instance_type, (host, int(port)), 0)
                waiting_nodes.append(f"{host}:{port}")
        return added_nodes, waiting_nodes

    def mark_waiting_retry(self, key: str, retry_count: int) -> None:
        with self._lock:
            if key not in self.waiting_nodes:
                return
            instance_type, server, _ = self.waiting_nodes[key]
            self.waiting_nodes[key] = (instance_type, server, retry_count)

    def activate_waiting_instance(self, instance_type: str, host: str, port: int) -> None:
        with self._lock:
            key = server_key(host, port)
            self.waiting_nodes.pop(key, None)
            if instance_type == InstanceType.PREFILL and key in self.tainted_prefillers:
                self.tainted_prefillers.discard(key)
                self._rebuild_heap_no_lock(InstanceType.PREFILL)
                return
            if instance_type == InstanceType.DECODE and key in self.tainted_decoders:
                self.tainted_decoders.discard(key)
                self._rebuild_heap_no_lock(InstanceType.DECODE)
                return
            if self._add_server_no_lock(instance_type, host, port):
                self.print_status(f"Add {instance_type} instance: {host}:{port}.")

    def drop_waiting_instance(self, key: str) -> None:
        with self._lock:
            self.waiting_nodes.pop(key, None)

    def remove_prefillers(self, instances: list[tuple[str, int]]) -> bool:
        return self._remove_instances(InstanceType.PREFILL, instances)

    def remove_decoders(self, instances: list[tuple[str, int]]) -> bool:
        return self._remove_instances(InstanceType.DECODE, instances)

    def _remove_instances(self, instance_type: str, instances: list[tuple[str, int]]) -> bool:
        if not instances:
            return False
        keys = {server_key(host, port) for host, port in instances}
        with self._lock:
            if self.request_num > 0:
                if instance_type == InstanceType.PREFILL:
                    self.tainted_prefillers.update(keys)
                else:
                    self.tainted_decoders.update(keys)
                self._rebuild_heap_no_lock(instance_type)
                logger.warning("Start to taint %s instances %s.", instance_type, sorted(keys))
                return True

            removed = False
            servers = self._server_map(instance_type)
            for key in keys:
                removed = servers.pop(key, None) is not None or removed
                self.waiting_nodes.pop(key, None)
                if instance_type == InstanceType.PREFILL:
                    self.tainted_prefillers.discard(key)
                else:
                    self.tainted_decoders.discard(key)
            if removed:
                self._rebuild_heap_no_lock(instance_type)
                self.print_status(f"Remove {instance_type} instances: {sorted(keys)}.")
            return False

    def finalize_tainted_instances(self) -> None:
        with self._lock:
            if self.request_num != 0:
                return
            if self.tainted_prefillers:
                keys = list(self.tainted_prefillers)
                for key in keys:
                    self.prefillers.pop(key, None)
                self.tainted_prefillers.clear()
                self._rebuild_heap_no_lock(InstanceType.PREFILL)
                self.print_status(f"Remove prefiller instances after drain: {keys}.")
            if self.tainted_decoders:
                keys = list(self.tainted_decoders)
                for key in keys:
                    self.decoders.pop(key, None)
                self.tainted_decoders.clear()
                self._rebuild_heap_no_lock(InstanceType.DECODE)
                self.print_status(f"Remove decoder instances after drain: {keys}.")


class SchedulerManager(BaseManager):
    pass


def get_shared_scheduler():
    if shared_scheduler is None:
        raise RuntimeError("shared scheduler is not initialized")
    return shared_scheduler


SchedulerManager.register("get_scheduler", callable=get_shared_scheduler)


class WorkerRuntime:
    def __init__(self, scheduler):
        self.scheduler = scheduler
        self.prefiller_clients: dict[str, httpx.AsyncClient] = {}
        self.decoder_clients: dict[str, httpx.AsyncClient] = {}

    async def sync_clients(self) -> None:
        snapshot = self.scheduler.get_snapshot()
        prefiller_targets = {
            server_key(s["host"], s["port"]): (s["host"], s["port"]) for s in snapshot["prefill_instances"]
        }
        decoder_targets = {
            server_key(s["host"], s["port"]): (s["host"], s["port"]) for s in snapshot["decode_instances"]
        }
        await self._sync_group(self.prefiller_clients, prefiller_targets)
        await self._sync_group(self.decoder_clients, decoder_targets)

    async def _sync_group(
        self,
        client_group: dict[str, httpx.AsyncClient],
        targets: dict[str, tuple[str, int]],
    ) -> None:
        stale = [key for key in client_group if key not in targets]
        for key in stale:
            client = client_group.pop(key)
            await client.aclose()
        for key, (host, port) in targets.items():
            if key in client_group:
                continue
            client_group[key] = httpx.AsyncClient(
                timeout=None,
                base_url=build_base_url(host, port),
                limits=httpx.Limits(max_connections=100000, max_keepalive_connections=100000),
            )

    def get_prefiller_client(self, key: str) -> httpx.AsyncClient:
        return self.prefiller_clients[key]

    def get_decoder_client(self, key: str) -> httpx.AsyncClient:
        return self.decoder_clients[key]

    async def close(self) -> None:
        for client in list(self.prefiller_clients.values()):
            await client.aclose()
        for client in list(self.decoder_clients.values()):
            await client.aclose()
        self.prefiller_clients.clear()
        self.decoder_clients.clear()


class NodeListener:
    def __init__(self, scheduler):
        self.scheduler = scheduler
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self) -> None:
        while True:
            for key, (instance_type, server, retries) in list(self.scheduler.get_waiting_nodes().items()):
                host, port = server
                is_valid = asyncio.run(self.check_instance_status(host, port))
                print(f"Checking instance {key}...")
                retries += 1
                if is_valid:
                    self.scheduler.activate_waiting_instance(instance_type, host, port)
                elif retries >= global_args.max_waiting_retries:
                    print(f"Instance {key} was not added to the proxy.")
                    self.scheduler.drop_waiting_instance(key)
                else:
                    self.scheduler.mark_waiting_retry(key, retries)

            self.scheduler.finalize_tainted_instances()
            time.sleep(global_args.waiting_retry_interval)

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


def serialize_args(args) -> dict[str, Any]:
    return {
        "port": args.port,
        "host": args.host,
        "prefiller_hosts": args.prefiller_hosts,
        "prefiller_ports": args.prefiller_ports,
        "decoder_hosts": args.decoder_hosts,
        "decoder_ports": args.decoder_ports,
        "max_retries": args.max_retries,
        "retry_delay": args.retry_delay,
        "max_waiting_retries": args.max_waiting_retries,
        "waiting_retry_interval": args.waiting_retry_interval,
        "workers": args.workers,
        "log_level": args.log_level,
    }


def deserialize_args(raw_args: dict[str, Any]):
    args = argparse.Namespace(**raw_args)
    args.prefiller_instances = list(zip(args.prefiller_hosts, args.prefiller_ports))
    args.decoder_instances = list(zip(args.decoder_hosts, args.decoder_ports))
    return args


def parse_args():
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


def load_global_args_from_env():
    global global_args
    if global_args is None:
        raw = os.environ.get(ARGS_CONFIG_ENV)
        if raw is None:
            raise RuntimeError(f"{ARGS_CONFIG_ENV} is not set")
        global_args = deserialize_args(json.loads(raw))
    return global_args


def connect_shared_scheduler():
    config = os.environ.get(MANAGER_CONFIG_ENV)
    if config is None:
        raise RuntimeError(f"{MANAGER_CONFIG_ENV} is not set")
    manager_cfg = json.loads(config)
    manager = SchedulerManager(
        address=(manager_cfg["host"], manager_cfg["port"]),
        authkey=base64.b64decode(manager_cfg["authkey"]),
    )
    manager.connect()
    return manager.get_scheduler()


def start_shared_scheduler(args) -> None:
    global shared_scheduler
    shared_scheduler = SharedProxyScheduler(args.prefiller_instances, args.decoder_instances)
    NodeListener(shared_scheduler)
    authkey = os.urandom(16)
    manager = SchedulerManager(address=("127.0.0.1", 0), authkey=authkey)
    server = manager.get_server()
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.address
    os.environ[MANAGER_CONFIG_ENV] = json.dumps(
        {"host": host, "port": port, "authkey": base64.b64encode(authkey).decode("ascii")}
    )
    os.environ[ARGS_CONFIG_ENV] = json.dumps(serialize_args(args))


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global runtime
    scheduler = connect_shared_scheduler()
    runtime = WorkerRuntime(scheduler)
    await runtime.sync_clients()
    snapshot = scheduler.get_snapshot()
    print(
        f"Initialized {len(snapshot['prefill_instances'])} prefill clients and "
        f"{len(snapshot['decode_instances'])} decode clients in worker {os.getpid()}."
    )
    yield
    await runtime.close()
    runtime = None


app = FastAPI(lifespan=lifespan)


def create_app():
    args = load_global_args_from_env()
    setup_logging(args.log_level)
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


async def send_request_to_service(
    client: httpx.AsyncClient,
    prefiller_key: str,
    endpoint: str,
    req_data: dict,
    request_id: str,
    max_retries: int = 3,
    base_delay: float = 0.2,
):
    req_data = req_data.copy()
    req_data["kv_transfer_params"] = {
        "do_remote_decode": True,
        "do_remote_prefill": False,
        "remote_engine_id": None,
        "remote_block_ids": None,
        "remote_host": None,
        "remote_port": None,
    }
    req_data["stream"] = False
    req_data["max_tokens"] = 1
    req_data["min_tokens"] = 1
    if "max_completion_tokens" in req_data:
        req_data["max_completion_tokens"] = 1
    if "stream_options" in req_data:
        del req_data["stream_options"]
    headers = {"Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}", "X-Request-Id": request_id}
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
    headers = {"Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}", "X-Request-Id": request_id}
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


async def handle_select_instance(api: str, req_data: Any, request_length: int) -> InstanceInfo:
    await runtime.sync_clients()
    scheduler = runtime.scheduler
    prefiller_score = scheduler.calculate_prefill_scores(request_length)
    request_id = scheduler.next_req_id()
    prefiller = scheduler.select_prefiller(prefiller_score)
    prefiller_key = prefiller["key"]

    try:
        response = await send_request_to_service(
            runtime.get_prefiller_client(prefiller_key),
            prefiller_key,
            api,
            req_data,
            request_id,
            max_retries=global_args.max_retries,
            base_delay=global_args.retry_delay,
        )
    finally:
        scheduler.release_prefiller(prefiller_key, prefiller_score)

    response_json = response.json()
    kv_transfer_params = response_json.get("kv_transfer_params", {})
    if kv_transfer_params:
        req_data["kv_transfer_params"] = kv_transfer_params

    decoder_score = scheduler.calculate_decode_scores(request_length)
    decoder = scheduler.select_decoder(decoder_score)
    decoder_key = decoder["key"]
    logger.debug(
        "Using %s %s",
        runtime.get_prefiller_client(prefiller_key).base_url,
        runtime.get_decoder_client(decoder_key).base_url,
    )
    return InstanceInfo(
        request_id=request_id,
        prefiller_key=prefiller_key,
        prefiller_score=prefiller_score,
        prefiller=prefiller,
        decoder_key=decoder_key,
        decoder_score=decoder_score,
        decoder=decoder,
    )


async def select_recompute_instance(
    api: str,
    req_data: Any,
    request_length: int,
    previous_instance: InstanceInfo,
) -> InstanceInfo:
    """Release the old decoder before assigning a new instance for recompute."""
    runtime.scheduler.release_decoder(previous_instance.decoder_key, previous_instance.decoder_score)
    return await handle_select_instance(api, req_data, request_length)


async def handle_completions_impl(api: str, request: Request):
    scheduler = runtime.scheduler
    scheduler.request_started()
    request_released = False
    try:
        req_data = await request.json()
        req_body = await request.body()
        request_length = len(req_body)
        instance_info = await handle_select_instance(api, req_data, request_length)
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
            try:
                while retry:
                    retry = False
                    async for chunk in stream_service_response_with_retry(
                        runtime.get_decoder_client(instance_info.decoder_key),
                        api,
                        req_data,
                        request_id=instance_info.request_id,
                        max_retries=global_args.max_retries,
                        base_delay=global_args.retry_delay,
                    ):
                        if not released_kv and chunk:
                            scheduler.release_prefiller_kv(instance_info.prefiller_key, instance_info.prefiller_score)
                            released_kv = True
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
                            instance_info = await select_recompute_instance(
                                api,
                                req_data,
                                tmp_request_length,
                                instance_info,
                            )
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
                    instance_info.decoder["host"],
                    instance_info.decoder["port"],
                    instance_info.request_id,
                )
                scheduler.release_prefiller_kv(instance_info.prefiller_key, instance_info.prefiller_score)
                raise
            except Exception as exc:
                logger.error(
                    "Error during streaming from decoder %s:%s: %s "
                    "while handling request %s; releasing prefiller KV",
                    instance_info.decoder["host"],
                    instance_info.decoder["port"],
                    exc,
                    instance_info.request_id,
                )
                scheduler.release_prefiller_kv(instance_info.prefiller_key, instance_info.prefiller_score)
            finally:
                scheduler.release_decoder(instance_info.decoder_key, instance_info.decoder_score)
                scheduler.request_finished()
                request_released = True

        media_type = "text/event-stream; charset=utf-8" if stream_flag else "application/json"
        return StreamingResponse(generate_stream(), media_type=media_type)
    except Exception:
        import traceback

        exc_info = sys.exc_info()
        print(f"Error occurred in disagg prefill proxy server - {api} endpoint")
        print("".join(traceback.format_exception(*exc_info)))
        raise
    finally:
        if not request_released:
            scheduler.request_finished()


async def adjust_instances_impl(adjust_mode: str, request: Request):
    req_data = await request.json()
    instance_type = req_data.get("type", "")
    instances = req_data.get("instances", [])
    if isinstance(instances, str):
        instances = [instances]
    instances = trans_instances(instances)
    all_msg = f"{adjust_mode} {instance_type} instances: {[str(server) for server in instances]}."

    if instance_type not in [InstanceType.PREFILL, InstanceType.DECODE]:
        return {
            "error": f"Instance type {instance_type} is not supported. "
            f"Only support '{InstanceType.PREFILL}' and '{InstanceType.DECODE}'."
        }

    raw_instances = [(server.host, server.port) for server in instances]
    scheduler = runtime.scheduler

    if adjust_mode == "add":
        added_nodes, waiting_nodes = scheduler.add_instances(instance_type, raw_instances)
        if waiting_nodes:
            all_msg = (
                f"{adjust_mode} {instance_type} instances: {added_nodes}. "
                f"Instances {waiting_nodes} are waiting to be added."
            )
    elif adjust_mode == "remove":
        if instance_type == InstanceType.PREFILL:
            need_waiting = scheduler.remove_prefillers(raw_instances)
        else:
            need_waiting = scheduler.remove_decoders(raw_instances)
        if need_waiting:
            all_msg = f"Instances {[str(server) for server in instances]} are isolated and waiting to be removed."

    snapshot = scheduler.get_snapshot()
    return {
        "message": all_msg,
        "current_prefill_instances": [f"{server['host']}:{server['port']}" for server in snapshot["prefill_instances"]],
        "current_decode_instances": [f"{server['host']}:{server['port']}" for server in snapshot["decode_instances"]],
    }


def trans_instances(instances: list[str]) -> list[ServerState]:
    result = []
    for instance in instances:
        host, port = instance.split(":")
        result.append(ServerState(host, int(port)))
    return result


@app.post("/v1/completions")
@with_cancellation
async def handle_completions(request: Request):
    return await handle_completions_impl("/completions", request)


@app.post("/v1/chat/completions")
@with_cancellation
async def handle_chat_completions(request: Request):
    return await handle_completions_impl("/chat/completions", request)


@app.get("/healthcheck")
async def healthcheck():
    return runtime.scheduler.healthcheck()


@app.post("/instances/add")
async def handle_add_instances(request: Request):
    return await adjust_instances_impl("add", request)


@app.post("/instances/remove")
async def handle_remove_instances(request: Request):
    return await adjust_instances_impl("remove", request)


if __name__ == "__main__":
    global_args = parse_args()
    setup_logging(global_args.log_level)
    start_shared_scheduler(global_args)
    import uvicorn

    module_name = Path(__file__).stem
    uvicorn.run(
        f"{module_name}:create_app",
        host=global_args.host,
        port=global_args.port,
        workers=global_args.workers,
        factory=True,
        app_dir=str(Path(__file__).resolve().parent),
    )
