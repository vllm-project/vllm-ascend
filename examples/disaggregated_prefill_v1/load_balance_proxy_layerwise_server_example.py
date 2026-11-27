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
#     pip install fastapi<0.124.0 httpx uvicorn
# - Copy this script alone; no other repository Python files are required.
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
#   python load_balance_proxy_layerwise_server_example.py \
#     --host 127.0.0.1 --port 9000 --workers 4 \
#     --prefiller-hosts 127.0.0.1 127.0.0.1 \
#     --prefiller-ports 8100 8101 \
#     --decoder-hosts 127.0.0.1 127.0.0.1 \
#     --decoder-ports 8200 8201
#
# This will start the proxy on port 9000, load balancing between two prefiller
# and two decoder servers. D must also reach callback ports 9001 through 9004.
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
# Notes:
# - You can scale the number of prefiller and decoder servers as needed.
# - The proxy will round-robin requests to balance load.
# - For production, ensure your backend servers are robust and secure.
#
# For more details, see the code and comments in this file.

import argparse
import asyncio
import copy
import functools
import heapq
import ipaddress
import json
import logging
import multiprocessing
import os
import signal
import socket
import sys
import threading
import time
import uuid
from contextlib import ExitStack, asynccontextmanager
from dataclasses import dataclass
from enum import Enum
from multiprocessing.managers import BaseManager

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse

logger = logging.getLogger(__name__)

# Wire key consumed by the connector's per-prompt metadata selection.
BATCH_KV_TRANSFER_PARAMS = "batch_kv_transfer_params"
WORKER_START_TIMEOUT = 30
WORKER_STOP_TIMEOUT = 10
WORKER_POLL_INTERVAL = 0.1
LISTEN_BACKLOG = 2048
MAX_PORT = 65535
PREFILL_KV_WEIGHT = 0.3
SSE_FRAME_SEPARATOR = b"\n\n"
SSE_DONE_SENTINEL = b"[DONE]"

# Add uvloop for faster event loop if available
try:
    import uvloop

    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
except ImportError:
    pass


class SSEFrameParser:
    """Buffer incomplete SSE frames while the handler forwards the original bytes."""

    def __init__(self):
        self._buffer = b""

    def feed(self, chunk: bytes) -> list[dict]:
        self._buffer += chunk
        frames = []
        while SSE_FRAME_SEPARATOR in self._buffer:
            raw, _, self._buffer = self._buffer.partition(SSE_FRAME_SEPARATOR)
            text = raw.strip()
            if text.startswith(b"data:"):
                text = text[len(b"data:") :].strip()
            if text == SSE_DONE_SENTINEL:
                self._buffer = b""
                break
            try:
                payload, _ = json.JSONDecoder().raw_decode(text.decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            if isinstance(payload, dict):
                frames.append(payload)
        return frames


class ServerRole(str, Enum):
    PREFILL = "prefill"
    DECODE = "decode"


def server_key(host: str, port: int) -> str:
    host = host.replace("localhost", "0.0.0.0").replace("127.0.0.1", "0.0.0.0")
    return f"{host}:{int(port)}"


@dataclass
class BackendLoad:
    ordinal: int
    active_tokens: float = 0.0
    active_kv_cache: float = 0.0
    heap_seq: int = 0


class ServerState:
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.url = f"http://{host}:{port}/v1"
        # Auto-completion for ipv6
        try:
            ip = ipaddress.ip_address(self.host)
            if isinstance(ip, ipaddress.IPv6Address):
                self.url = f"http://[{host}]:{port}/v1"
        except Exception:
            pass
        self.client = httpx.AsyncClient(
            timeout=None,
            base_url=self.url,
            limits=httpx.Limits(max_connections=100000, max_keepalive_connections=100000),
        )
        self.active_tokens = 0
        self.active_kv_cache = 0  # Only for prefiller
        self.active_requests = 0  # Number of active requests
        self.aborted_requests = set()  # Track aborted requests
        # Removed individual server lock - will use global locks instead


class LayerwiseScheduler:
    """Share backend load, retaining ownership so a dead worker can release it."""

    def __init__(self, prefiller_instances, decoder_instances):
        self._lock = threading.RLock()
        self.prefillers = {server_key(h, p): BackendLoad(i) for i, (h, p) in enumerate(prefiller_instances)}
        self.decoders = {server_key(h, p): BackendLoad(i) for i, (h, p) in enumerate(decoder_instances)}
        self._servers = {ServerRole.PREFILL: self.prefillers, ServerRole.DECODE: self.decoders}
        self._heaps = {role: [] for role in ServerRole}
        for role, servers in self._servers.items():
            for key in servers:
                self._push_heap(role, key)
        self.worker_loads = {}

    def register_worker(self):
        """Give each process incarnation an identity independent of its port slot."""
        with self._lock:
            worker_token = uuid.uuid4().hex
            self.worker_loads[worker_token] = {}
            return worker_token

    def _heap_entry(self, role, key):
        server = self._servers[role][key]
        priority = server.active_tokens
        if role is ServerRole.PREFILL:
            priority += server.active_kv_cache * PREFILL_KV_WEIGHT
        return priority, server.ordinal, server.heap_seq, key

    def _push_heap(self, role, key):
        servers = self._servers[role]
        servers[key].heap_seq += 1
        heapq.heappush(self._heaps[role], self._heap_entry(role, key))
        # Discard stale entries periodically so completed requests do not grow the heap.
        if len(self._heaps[role]) > 2 * len(servers):
            self._heaps[role] = [self._heap_entry(role, key) for key in servers]
            heapq.heapify(self._heaps[role])

    def _release_load(self, role, key, load, *, active_tokens=False, kv_cache=False):
        server = self._servers[role][key]
        if active_tokens:
            server.active_tokens -= load
        if kv_cache:
            server.active_kv_cache = max(0.0, server.active_kv_cache - load)
        self._push_heap(role, key)

    def reserve(self, worker_token, role, load):
        role = ServerRole(role)
        with self._lock:
            worker_loads = self.worker_loads.get(worker_token)
            if worker_loads is None:
                raise RuntimeError("Worker incarnation is no longer active")
            heap = self._heaps[role]
            while heap:
                _, _, seq, key = heapq.heappop(heap)
                server = self._servers[role][key]
                if seq == server.heap_seq:
                    break
            else:
                raise RuntimeError(f"No available {role.value} servers")
            server.active_tokens += load
            if role is ServerRole.PREFILL:
                server.active_kv_cache += load
            self._push_heap(role, key)
            loads = worker_loads.setdefault((role, key), [0.0, 0.0])
            loads[0] += load
            if role is ServerRole.PREFILL:
                loads[1] += load
            return key

    def release(self, worker_token, role, key, load, kv_cache=False):
        role = ServerRole(role)
        with self._lock:
            worker_loads = self.worker_loads.get(worker_token)
            if worker_loads is None:
                return  # Cleanup already released this incarnation's reservations.
            loads = worker_loads[role, key]
            loads[int(kv_cache)] -= load
            self._release_load(role, key, load, active_tokens=not kv_cache, kv_cache=kv_cache)

    def release_worker(self, worker_token):
        with self._lock:
            # Removing the identity under the same lock fences already-received
            # Manager RPCs that execute after cleanup, even after process.join().
            for (role, key), (tokens, kv_cache) in self.worker_loads.pop(worker_token, {}).items():
                self._release_load(role, key, tokens, active_tokens=True)
                self._release_load(role, key, kv_cache, kv_cache=True)


class LayerwiseManager(BaseManager):
    """Own the scheduler without moving HTTP clients or asyncio state across processes."""


LayerwiseManager.register("Scheduler", LayerwiseScheduler)


class ProxyState:
    def __init__(self, prefiller_instances, decoder_instances, scheduler=None, worker_token=None):
        self.prefillers: list[ServerState] = [ServerState(h, p) for h, p in prefiller_instances]
        self.decoders: list[ServerState] = [ServerState(h, p) for h, p in decoder_instances]
        self.scheduler = (
            scheduler if scheduler is not None else LayerwiseScheduler(prefiller_instances, decoder_instances)
        )
        self.worker_token = worker_token if worker_token is not None else self.scheduler.register_worker()
        self.prefiller_indices = {server_key(h, p): i for i, (h, p) in enumerate(prefiller_instances)}
        self.decoder_indices = {server_key(h, p): i for i, (h, p) in enumerate(decoder_instances)}
        self.req_to_prefiller = {}
        self.req_id_lock = asyncio.Lock()
        # Removed selection locks - no longer needed for synchronous methods

        self.req_id_future = {}
        self.req_data_dict = {}
        self.metaserver_expected_ids = {}
        self.metaserver_params = {}
        self.metaserver_ready_events = {}
        self.metaserver_dispatch_tasks = {}
        self.metaserver_lock = asyncio.Lock()

    def abort_prefiller_request(self, server_idx: int, request_id):  # Changed to synchronous
        """
        Mark a request as aborted. This will helps to release kv cache in
        prefiller node.
        """
        # No lock needed - atomic operation
        self.prefillers[server_idx].aborted_requests.add(request_id)

    def acquire_aborted_prefiller_requests(self, server_idx: int):  # Changed to synchronous
        """
        Get the set of aborted requests and clear it.
        This is used to release kv cache in prefiller node.
        """
        # No lock needed - atomic operation
        aborted_requests = self.prefillers[server_idx].aborted_requests.copy()
        self.prefillers[server_idx].aborted_requests.clear()
        return aborted_requests

    async def next_req_id(self):
        async with self.req_id_lock:
            return str(uuid.uuid4())

    def register_request_batch(self, parent_request_id, request_ids, req_data, request_length, api):
        request_record = (copy.deepcopy(req_data), request_length, api, parent_request_id)
        for request_id in request_ids:
            self.req_data_dict[request_id] = request_record
        self.metaserver_expected_ids[parent_request_id] = tuple(request_ids)
        self.metaserver_params[parent_request_id] = {}
        self.metaserver_ready_events[parent_request_id] = asyncio.Event()

    async def cleanup_request_batch(self, parent_request_id):
        async with self.metaserver_lock:
            request_ids = self.metaserver_expected_ids.pop(parent_request_id, ())
            for request_id in request_ids:
                self.req_data_dict.pop(request_id, None)
            self.metaserver_params.pop(parent_request_id, None)
            self.metaserver_ready_events.pop(parent_request_id, None)
            self.metaserver_dispatch_tasks.pop(parent_request_id, None)

    def select_prefiller(self, token_count):
        key = self.scheduler.reserve(self.worker_token, ServerRole.PREFILL, token_count)
        return self.prefiller_indices[key]

    def release_prefiller(self, idx, token_count):
        server = self.prefillers[idx]
        self.scheduler.release(self.worker_token, ServerRole.PREFILL, server_key(server.host, server.port), token_count)

    def release_prefiller_kv(self, idx, token_count):
        server = self.prefillers[idx]
        self.scheduler.release(
            self.worker_token, ServerRole.PREFILL, server_key(server.host, server.port), token_count, kv_cache=True
        )

    def select_decoder(self, token_count):
        key = self.scheduler.reserve(self.worker_token, ServerRole.DECODE, token_count)
        return self.decoder_indices[key]

    def release_decoder(self, idx, token_count):
        server = self.decoders[idx]
        self.scheduler.release(self.worker_token, ServerRole.DECODE, server_key(server.host, server.port), token_count)

    # Omni_infer's calculate_input_scores function
    def calculate_prefill_scores(self, request_length: int) -> float:
        length_score = request_length / 4.0
        input_score = length_score * 0.0345 + 120.0745
        return input_score

    def calculate_decode_scores(self, request_length: int) -> float:
        return request_length


proxy_state = None


def _validate_host(host: str, arg_name: str, index: int) -> None:
    """Validate that a host argument is a well-formed IP address or hostname.

    Catches shell-escaping typos where a missing space before a line-continuation
    backslash concatenates adjacent values (e.g. ``192.0.0.3\\n192.0.0.3``).
    """
    # A valid IP (IPv4/IPv6) is accepted as-is.
    try:
        ipaddress.ip_address(host)
        return
    except ValueError:
        pass
    # A non-IP value whose labels are all digits is almost always a shell-escaping
    # typo that concatenated two hosts (e.g. "192.0.0.3192.0.0.3"); anything else
    # is treated as a hostname.
    if host and not all(label.isdigit() for label in host.split(".")):
        return
    hint = (
        "looks like a concatenated IP address; check for missing whitespace before line-continuation backslashes"
        if host
        else "must be a valid IP or hostname"
    )
    raise ValueError(f"Invalid host {host!r} in {arg_name}[{index}]: {hint}")


def _validate_hosts(hosts: list, arg_name: str) -> None:
    """Validate every host in *hosts*, raising ValueError on the first
    malformed entry."""
    for i, host in enumerate(hosts):
        _validate_host(host, arg_name, i)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Worker processes; metaserver ports are --port + 1 + worker_id when workers > 1.",
    )
    parser.add_argument("--prefiller-hosts", type=str, nargs="+", default=["localhost"])
    parser.add_argument("--prefiller-ports", type=int, nargs="+", default=[8001])
    parser.add_argument("--decoder-hosts", type=str, nargs="+", default=["localhost"])
    parser.add_argument("--decoder-ports", type=int, nargs="+", default=[8002])
    parser.add_argument("--max-retries", type=int, default=3, help="Maximum number of retries for HTTP requests")
    parser.add_argument(
        "--retry-delay", type=float, default=0.001, help="Base delay (seconds) for exponential backoff retries"
    )
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.workers > 1 and not 0 < args.port <= MAX_PORT - args.workers:
        parser.error("Multi-worker mode requires 1 <= --port and --port + --workers <= 65535")
    logger.info(
        "Decoder hosts %s must reach %s on metaserver port(s) %s at /v1/metaserver",
        set(args.decoder_hosts),
        args.host,
        list(range(args.port + 1, args.port + args.workers + 1)) if args.workers > 1 else [args.port],
    )
    # Wildcard address is not allowed for layerwise connector
    if args.host in ["0.0.0.0", "::", "0:0:0:0:0:0:0:0"]:
        raise ValueError(
            f"Decoder hosts will access Proxy host:port/metaserver, to avoid configuration errors, "
            f"the Wildcard Address {args.host} is not allowed for proxy"
        )
    # Validate host arguments so that shell-escaping typos
    # (e.g. "192.0.0.3\\n192.0.0.3") are caught early with a clear
    # message rather than producing a confusing runtime failure.
    _validate_hosts(args.prefiller_hosts, "--prefiller-hosts")
    _validate_hosts(args.decoder_hosts, "--decoder-hosts")
    if len(args.prefiller_hosts) != len(args.prefiller_ports):
        raise ValueError("Number of prefiller hosts must match number of prefiller ports")
    if len(args.decoder_hosts) != len(args.decoder_ports):
        raise ValueError("Number of decoder hosts must match number of decoder ports")
    args.prefiller_instances = list(zip(args.prefiller_hosts, args.prefiller_ports))
    args.decoder_instances = list(zip(args.decoder_hosts, args.decoder_ports))
    return args


@asynccontextmanager
async def lifespan(app: FastAPI):
    global proxy_state
    proxy_state = ProxyState(
        global_args.prefiller_instances,
        global_args.decoder_instances,
        getattr(app.state, "scheduler", None),
        getattr(app.state, "worker_token", None),
    )
    print(f"Initialized {len(proxy_state.prefillers)} prefill clients and {len(proxy_state.decoders)} decode clients.")
    try:
        yield
    finally:
        tasks = list(proxy_state.metaserver_dispatch_tasks.values())
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        for server in proxy_state.prefillers + proxy_state.decoders:
            await server.client.aclose()
        proxy_state.scheduler.release_worker(proxy_state.worker_token)


async def listen_for_disconnect(request: Request) -> None:
    """Return if a disconnect message is received"""
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


app = FastAPI(lifespan=lifespan)


async def send_request_to_service(
    client: httpx.AsyncClient,
    prefiller_id: int,
    endpoint: str,
    req_data: dict,
    request_id: str,
    max_retries: int = 3,
    base_delay: float = 0.2,
):
    proxy_state.acquire_aborted_prefiller_requests(prefiller_id)
    req_data = req_data.copy()
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
            if request_id in proxy_state.req_id_future:
                result_future = proxy_state.req_id_future[request_id]
                result_future.set_result(response.json()["kv_transfer_params"])
            return
        except (httpx.RequestError, httpx.HTTPStatusError) as e:
            logger.warning("Attempt %s failed for %s: %s", attempt, endpoint, e)
            last_exc = e
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
                return  # Success, exit after streaming
        except (httpx.RequestError, httpx.HTTPStatusError) as e:
            if attempt < max_retries:
                logger.warning("Attempt %s failed for streaming %s: %s", attempt, endpoint, e)
                await asyncio.sleep(base_delay * (2 ** (attempt - 1)))
            else:
                logger.error("All %s attempts failed for streaming %s.", max_retries, endpoint)
                raise e
        except Exception as e:
            # If any chunk has been sent, do not retry, just log and drop
            if "first_chunk_sent" in locals() and first_chunk_sent:
                logger.error("Streaming to client interrupted after response started: %s", e)
                return
            else:
                if attempt < max_retries:
                    logger.warning("Attempt %s failed for streaming %s: %s", attempt, endpoint, e)
                    await asyncio.sleep(base_delay * (2 ** (attempt - 1)))
                else:
                    logger.error("All %s attempts failed for streaming %s.", max_retries, endpoint)
                    raise e


def get_api_request_id(api, req_id, prompt_idx=0):
    if api == "/completions":
        return f"cmpl-{req_id}-{prompt_idx}"
    elif api == "/chat/completions":
        return "chatcmpl-" + req_id


def get_api_request_ids(api, req_id, req_data):
    if api != "/completions":
        return [get_api_request_id(api, req_id)]

    prompt = req_data.get("prompt")
    # A list[int] is one tokenized prompt. A list[str] or list[list[int]] is
    # expanded by vLLM into one child request per entry.
    is_token_ids = isinstance(prompt, list) and all(isinstance(token, int) for token in prompt)
    prompt_count = len(prompt) if isinstance(prompt, list) and prompt and not is_token_ids else 1
    return [get_api_request_id(api, req_id, prompt_idx) for prompt_idx in range(prompt_count)]


async def _handle_completions(api: str, request: Request):
    try:
        req_data = await request.json()
        req_body = await request.body()
        request_length = len(req_body)
        request_id = await proxy_state.next_req_id()
        request_ids_api = get_api_request_ids(api, request_id, req_data)
        proxy_state.register_request_batch(request_id, request_ids_api, req_data, request_length, api)
        req_data["kv_transfer_params"] = {
            "do_remote_decode": False,
            "do_remote_prefill": True,
            "metaserver": metaserver_url(global_args),
        }
        # Select decoder
        decoder_score = proxy_state.calculate_decode_scores(request_length)
        logger.debug("Decoder score: %f", decoder_score)
        # Use the prefiller's kv_transfer_params to select decoder
        decoder_idx = proxy_state.select_decoder(decoder_score)
        decoder = proxy_state.decoders[decoder_idx]
        # logger.debug("Using %s %s", prefiller.url, decoder.url)
        # Stream response from decoder
        released_kv = False

        # Record request info for recompute
        stream_flag = bool(req_data.get("stream", False))
        chat_flag = "messages" in req_data
        if "prompt" in req_data:
            origin_prompt = req_data["prompt"]
        elif chat_flag:
            messages = req_data["messages"]
            origin_prompt = messages[0].get("content", "")
        else:
            origin_prompt = ""
        # refer to vLLM sampling_params: max_token default value
        origin_max_tokens = req_data.get("max_tokens", 16)

        async def generate_stream():
            nonlocal released_kv
            generated_token = ""
            released_kv = False
            retry_count = 0
            retry = True
            completion_tokens = 0
            frame_parser = SSEFrameParser()
            # Only one await per chunk, minimal logic in loop
            try:
                while retry:
                    retry = False
                    # Each decoder attempt emits a self-contained stream, so the
                    # frame buffer must be reset at every attempt boundary.
                    frame_parser = SSEFrameParser()
                    async for chunk in stream_service_response_with_retry(
                        decoder.client,
                        api,
                        req_data,
                        request_id=request_id,
                        max_retries=global_args.max_retries,
                        base_delay=global_args.retry_delay,
                    ):
                        # The decode backend speaks SSE, but the network does
                        # not respect frame boundaries: one frame can arrive
                        # split across two chunks, and two frames can be
                        # coalesced into one chunk. Reassemble frames before
                        # decoding them, so no frame is dropped.
                        frames = frame_parser.feed(chunk)
                        if not frames:
                            # No complete frame yet; forward the raw bytes so
                            # the client still sees them, and wait for the
                            # rest of the frame.
                            if chunk:
                                yield chunk
                            continue
                        for chunk_json in frames:
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
                                else (completion_tokens + usage.get("completion_tokens"))
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
                                break
                            if retry_count > 0 and not stream_flag:
                                if chat_flag:
                                    choice["message"]["content"] = generated_token
                                else:
                                    choice["text"] = generated_token
                                chunk = json.dumps(chunk_json).encode("utf-8")
                            yield chunk
                        if retry:
                            break
            except Exception as e:
                logger.error(
                    "Error during streaming from decoder %s: %s the aborted request %s "
                    "will be routing to the target prefiller when new request is ready to dispatch to it",
                    decoder.url,
                    e,
                    request_id,
                )
            finally:
                # After streaming done, release tokens
                proxy_state.release_decoder(decoder_idx, decoder_score)
                await proxy_state.cleanup_request_batch(request_id)

        if stream_flag:
            return StreamingResponse(generate_stream(), media_type="text/event-stream")
        else:
            return StreamingResponse(generate_stream(), media_type="application/json")
    except Exception as e:
        import traceback

        exc_info = sys.exc_info()
        print(f"Error occurred in disagg prefill proxy server - {api} endpoint")
        print(e)
        print("".join(traceback.format_exception(*exc_info)))
        raise


@app.post("/v1/completions")
@with_cancellation
async def handle_completions(request: Request):
    return await _handle_completions("/completions", request)


@app.post("/v1/chat/completions")
@with_cancellation
async def handle_chat_completions(request: Request):
    return await _handle_completions("/chat/completions", request)


@app.get("/healthcheck")
async def healthcheck():
    return {
        "status": "ok",
        "prefill_instances": len(proxy_state.prefillers),
        "decode_instances": len(proxy_state.decoders),
    }


@app.post("/reset_prefix_cache")
async def reset_prefix_cache(request: Request):
    params = dict(request.query_params)
    failures = []
    for client, base_url in [
        (s.client, f"http://{s.host}:{s.port}") for s in proxy_state.prefillers + proxy_state.decoders
    ]:
        try:
            resp = await client.post(f"{base_url}/reset_prefix_cache", params=params)
            resp.raise_for_status()
        except Exception as e:
            logger.error("reset_prefix_cache failed for %s: %s", base_url, e)
            failures.append(base_url)
    if failures:
        from fastapi.responses import JSONResponse

        return JSONResponse(status_code=500, content={"failed": failures})
    from fastapi.responses import Response as FastAPIResponse

    return FastAPIResponse(status_code=200)


async def dispatch_prefill_batch(
    req_data,
    request_length,
    api,
    parent_request_id,
    expected_request_ids,
    params_by_request_id,
):
    prefiller_score = proxy_state.calculate_prefill_scores(request_length)
    logger.debug("Request length: %s, Prefiller score: %s", request_length, prefiller_score)
    prefiller_idx = proxy_state.select_prefiller(prefiller_score)
    try:
        prefiller = proxy_state.prefillers[prefiller_idx]
        req_data = copy.deepcopy(req_data)
        if len(expected_request_ids) == 1:
            req_data["kv_transfer_params"] = params_by_request_id[expected_request_ids[0]]
        else:
            # vLLM expands a completion prompt list into cmpl-<parent>-N child
            # requests. Preserve the per-child D endpoint/cache metadata while
            # issuing the original prompt list to P exactly once.
            kv_transfer_params = copy.deepcopy(params_by_request_id[expected_request_ids[0]])
            kv_transfer_params[BATCH_KV_TRANSFER_PARAMS] = {
                request_id: params_by_request_id[request_id] for request_id in expected_request_ids
            }
            req_data["kv_transfer_params"] = kv_transfer_params
        logger.debug("Using prefill prefiller.url=%r req_data=%r", prefiller.url, req_data)
        await send_request_to_service(
            prefiller.client,
            prefiller_idx,
            api,
            req_data,
            parent_request_id,
            max_retries=global_args.max_retries,
            base_delay=global_args.retry_delay,
        )
    finally:
        proxy_state.release_prefiller(prefiller_idx, prefiller_score)
        proxy_state.release_prefiller_kv(prefiller_idx, prefiller_score)


@app.post("/v1/metaserver")
async def metaserver(request: Request):
    kv_transfer_params = await request.json()
    request_id = kv_transfer_params.get("request_id")
    request_record = proxy_state.req_data_dict.get(request_id)
    if request_record is None:
        raise HTTPException(status_code=404, detail=f"Unknown metaserver request_id: {request_id!r}")

    req_data, request_length, api, parent_request_id = request_record
    async with proxy_state.metaserver_lock:
        params_by_request_id = proxy_state.metaserver_params[parent_request_id]
        params_by_request_id[request_id] = kv_transfer_params
        expected_request_ids = proxy_state.metaserver_expected_ids[parent_request_id]
        dispatch_task = proxy_state.metaserver_dispatch_tasks.get(parent_request_id)
        if dispatch_task is not None and dispatch_task.done():
            if dispatch_task.cancelled() or dispatch_task.exception() is not None:
                dispatch_task = None
        if dispatch_task is None and all(
            expected_request_id in params_by_request_id for expected_request_id in expected_request_ids
        ):
            dispatch_task = asyncio.create_task(
                dispatch_prefill_batch(
                    req_data,
                    request_length,
                    api,
                    parent_request_id,
                    expected_request_ids,
                    copy.deepcopy(params_by_request_id),
                )
            )
            proxy_state.metaserver_dispatch_tasks[parent_request_id] = dispatch_task
            proxy_state.metaserver_ready_events[parent_request_id].set()
        ready_event = proxy_state.metaserver_ready_events[parent_request_id]

    if dispatch_task is None:
        await ready_event.wait()
        async with proxy_state.metaserver_lock:
            dispatch_task = proxy_state.metaserver_dispatch_tasks[parent_request_id]

    try:
        await asyncio.shield(dispatch_task)
    except Exception as error:
        logger.exception("Post metaserver failed for parent request %s", parent_request_id)
        raise HTTPException(status_code=502, detail=str(error)) from error
    return {"status": "ok"}


def metaserver_url(args):
    host = f"[{args.host}]" if ":" in args.host else args.host
    return f"http://{host}:{getattr(args, 'metaserver_port', args.port)}/v1/metaserver"


def bind_listener(host, port, *, reuse_port=False):
    family, socktype, protocol, _, address = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)[0]
    sock = socket.socket(family, socktype, protocol)
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        if reuse_port:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
        sock.bind(address)
        sock.listen(LISTEN_BACKLOG)
        sock.setblocking(False)
        return sock
    except BaseException:
        sock.close()
        raise


class WorkerServer(uvicorn.Server):
    def __init__(self, config, ready):
        super().__init__(config)
        self.ready = ready

    async def startup(self, sockets=None):
        await super().startup(sockets=sockets)
        if self.started:
            self.ready.set()


def run_worker(args, scheduler, worker_id, worker_token, ready):
    """Both listeners run on this worker's event loop and share its request state."""
    global global_args
    global_args = args
    args.metaserver_port = args.port + 1 + worker_id
    app.state.scheduler = scheduler
    app.state.worker_token = worker_token
    logging.basicConfig(level=logging.INFO)
    with ExitStack() as stack:
        public = stack.enter_context(bind_listener(args.host, args.port, reuse_port=True))
        metadata = stack.enter_context(bind_listener(args.host, args.metaserver_port))
        logger.info(
            "Worker %s (pid %s, Uvicorn %s) metaserver: %s",
            worker_id,
            os.getpid(),
            uvicorn.__version__,
            metaserver_url(args),
        )
        server = WorkerServer(
            uvicorn.Config(app, host=args.host, port=args.port, timeout_graceful_shutdown=WORKER_STOP_TIMEOUT), ready
        )
        server.run(sockets=[public, metadata])


def start_worker(ctx, args, scheduler, worker_id, stopping):
    ready = ctx.Event()
    worker_token = scheduler.register_worker()
    process = ctx.Process(target=run_worker, args=(args, scheduler, worker_id, worker_token, ready))
    process.start()
    deadline = time.monotonic() + WORKER_START_TIMEOUT
    while not ready.wait(WORKER_POLL_INTERVAL):
        if process.exitcode is not None or stopping.is_set() or time.monotonic() >= deadline:
            stop_workers([process])
            scheduler.release_worker(worker_token)
            raise RuntimeError(f"Layerwise proxy worker {worker_id} failed to start")
    return process, worker_token


def stop_workers(processes):
    for process in processes:
        if process.is_alive():
            process.terminate()
    deadline = time.monotonic() + WORKER_STOP_TIMEOUT
    for process in processes:
        process.join(timeout=max(0, deadline - time.monotonic()))
        if process.is_alive():
            process.kill()
            process.join()
        process.close()


def run_workers(args):
    if not hasattr(socket, "SO_REUSEPORT"):
        raise RuntimeError("Multi-worker layerwise proxy requires SO_REUSEPORT")
    # Check the whole range without SO_REUSEPORT: do not join another proxy's
    # listener group. Children still bind their own sockets and surface races.
    with ExitStack() as stack:
        for port in range(args.port, args.port + args.workers + 1):
            stack.enter_context(bind_listener(args.host, port))

    ctx = multiprocessing.get_context("spawn")
    stopping = threading.Event()
    previous_handlers = {sig: signal.signal(sig, lambda *_: stopping.set()) for sig in (signal.SIGINT, signal.SIGTERM)}
    processes = []
    try:
        with LayerwiseManager(ctx=ctx) as manager:
            scheduler = manager.Scheduler(args.prefiller_instances, args.decoder_instances)
            try:
                for worker_id in range(args.workers):
                    processes.append(start_worker(ctx, args, scheduler, worker_id, stopping))
                while not stopping.wait(WORKER_POLL_INTERVAL):
                    for worker_id, (process, worker_token) in enumerate(processes):
                        if process.exitcode is not None:
                            process.join()
                            scheduler.release_worker(worker_token)
                            replacement = start_worker(ctx, args, scheduler, worker_id, stopping)
                            process.close()
                            processes[worker_id] = replacement
            finally:
                stop_workers([process for process, _ in processes])
    finally:
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    global_args = parse_args()
    if global_args.workers > 1:
        run_workers(global_args)
    else:
        uvicorn.run(app, host=global_args.host, port=global_args.port)
