# Adapted from https://github.com/vllm-project/vllm/tests/v1/kv_connector/nixl_integration/toy_proxy_server.py

# SPDX-License-Identifier: Apache-2.0
#
# Dynamic bucketing-based hybrid load balance proxy server.
# See README.md in this directory for the tutorial and usage.

import argparse
import asyncio
import functools
import heapq
import os
import sys
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

import httpx
from dynamic_bucket_load_balancer import DynamicBucketLoadBalancer, ServerInfo, Task
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

try:
    from vllm.logger import init_logger

    logger = init_logger(__name__)
except ImportError:
    import logging

    logger = logging.getLogger(__name__)

# Use uvloop for a faster event loop if available
try:
    import uvloop

    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
except ImportError:
    pass


class ServerState:
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.url = f"http://{host}:{port}/v1"
        self.client = httpx.AsyncClient(
            timeout=None,
            base_url=self.url,
            limits=httpx.Limits(max_connections=100000, max_keepalive_connections=100000),
        )
        self.active_tokens = 0
        self.aborted_requests = set()

    def __eq__(self, other):
        self_host = self.host.replace("localhost", "0.0.0.0").replace("127.0.0.1", "0.0.0.0")
        other_host = other.host.replace("localhost", "0.0.0.0").replace("127.0.0.1", "0.0.0.0")
        return self_host == other_host and str(self.port) == str(other.port)

    def __hash__(self):
        self_host = self.host.replace("localhost", "0.0.0.0").replace("127.0.0.1", "0.0.0.0")
        return hash((self_host, str(self.port)))

    def __repr__(self):
        return f"{self.host}:{self.port}"


@dataclass(order=True)
class ServerHeapItem:
    priority: float
    server_idx: int
    server: ServerState


class ProxyState:
    def __init__(self, server_instances):
        self.infer_servers: list[ServerState] = [ServerState(h, p) for h, p in server_instances]
        self.req_id_lock = asyncio.Lock()

        # Dynamic bucket load balancer
        self.bucket_load_balancer = None

        if global_args.enable_dynamic_bucket:
            self.num_buckets = 2  # Two buckets (short/long) when dynamic bucketing is enabled
            self.server_group_threshold = global_args.server_group_threshold
            buckets = [(0, self.server_group_threshold), (self.server_group_threshold, global_args.max_request_tokens)]

            self.bucket_load_balancer = DynamicBucketLoadBalancer(buckets=buckets)
        else:
            self.num_buckets = 1  # No bucketing by default

        # Priority queue per group; smaller score = higher priority (lower load)
        server_heap_items = [ServerHeapItem(0.0, i, server) for i, server in enumerate(self.infer_servers)]
        self.server_heaps: list[list[ServerHeapItem]] = self._group_servers(server_heap_items, self.num_buckets)
        self.server_idx_to_group_idx = {}

        # Heapify each group
        for idx, cur_heap in enumerate(self.server_heaps):
            for server_item in cur_heap:
                self.server_idx_to_group_idx[server_item.server_idx] = idx
            heapq.heapify(cur_heap)

        logger.info(
            "Dynamic bucket enabled: %s, number of groups: %s",
            global_args.enable_dynamic_bucket,
            len(self.server_heaps),
        )
        for group_idx, cur_heap in enumerate(self.server_heaps):
            logger.info("Group %s: %s", group_idx, cur_heap)

    @staticmethod
    def _group_servers(servers: list[ServerHeapItem], num_groups: int):
        """
        Split servers into num_groups groups.

        Args:
            servers (list): the server list to group.
            num_groups (int): the number of groups.

        Returns:
            list[list]: the grouped server list.

        Raises:
            ValueError: when num_groups <= 0.
        """
        if num_groups <= 0:
            raise ValueError("Num of group is illegal")

        if len(servers) < num_groups:
            raise ValueError("Number of servers must greater than or equal to number of groups")

        n = len(servers)
        if n == 0:
            return [[] for _ in range(num_groups)]
        elif n == 1:
            return [servers]

        base_size = n // num_groups
        remainder = n % num_groups

        groups = []
        start_index = 0
        for i in range(num_groups):
            group_size = base_size + 1 if i < remainder else base_size
            end_index = start_index + group_size
            groups.append(servers[start_index:end_index])
            start_index = end_index

        return groups

    def _update_server_priority(self, server_idx: int):
        """Update the priority of a server in the heap."""
        server = self.infer_servers[server_idx]
        priority = server.active_tokens
        # Remove the old entry, then add the new one
        group_idx = self.server_idx_to_group_idx[server_idx]

        self.server_heaps[group_idx] = [
            server_heap_item
            for server_heap_item in self.server_heaps[group_idx]
            if server_heap_item.server_idx != server_idx
        ]
        self.server_heaps[group_idx].append(ServerHeapItem(priority, server_idx, server))
        heapq.heapify(self.server_heaps[group_idx])

    async def next_req_id(self):
        async with self.req_id_lock:
            return str(uuid.uuid4())

    def select_server(self, token_count, group_idx: int):
        if not self.infer_servers:
            raise RuntimeError("No inference servers available")

        server_heap_item: ServerHeapItem = heapq.heappop(self.server_heaps[group_idx])
        chosen_server_idx = server_heap_item.server_idx

        # Update the chosen server (accumulate load)
        self.infer_servers[chosen_server_idx].active_tokens += token_count

        # Update priority and re-add to the heap
        self._update_server_priority(chosen_server_idx)

        return chosen_server_idx

    def release_server(self, idx: int, token_count, req_id):
        self.infer_servers[idx].active_tokens -= token_count
        if global_args.enable_dynamic_bucket and req_id is not None and self.bucket_load_balancer is not None:
            self.bucket_load_balancer.release_task(req_id)
        # Update the priority queue after release
        self._update_server_priority(idx)

    def calculate_request_score(self, request_length: int, max_tokens: int = 16, ignore_eos: bool = False) -> float:
        if ignore_eos:
            return request_length + max_tokens
        else:
            # Note that 0.5 is an empirical value here because we don't know
            # the actual number of tokens generated before EOS.
            return request_length + 0.5 * max_tokens

    def calculate_request_tokens(self, request_length: int) -> float:
        return request_length / 4.0

    def select_server_group(self, req_id: str, request_tokens, priority_score) -> tuple[int, Task | None]:
        """Pick the best group given the request length and the current load of each group."""
        if global_args.enable_dynamic_bucket and self.bucket_load_balancer is not None:
            group_idx, task = self.bucket_load_balancer.dispatch_single_task(req_id, request_tokens, priority_score)
            return group_idx, task
        else:
            return 0, None


proxy_state = None


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--server-hosts", type=str, nargs="+", default=["localhost"])
    parser.add_argument("--server-ports", type=int, nargs="+", default=[8001])
    parser.add_argument("--max-retries", type=int, default=3, help="Maximum number of retries for HTTP requests")
    parser.add_argument(
        "--retry-delay", type=float, default=0.001, help="Base delay (seconds) for exponential backoff retries"
    )

    parser.add_argument("--server-group-threshold", type=int, default=32 * 1024, help="Threshold of server groups")
    parser.add_argument("--max-request-tokens", type=int, default=128 * 1024, help="Max tokens of request")
    parser.add_argument(
        "--enable-dynamic-bucket", action="store_true", default=False, help="Enable dynamic bucket load Balancer"
    )

    args = parser.parse_args()
    if len(args.server_hosts) != len(args.server_ports):
        raise ValueError("Number of dp hosts must match number of dp ports")
    args.server_instances = list(zip(args.server_hosts, args.server_ports))
    return args


@asynccontextmanager
async def lifespan(app: FastAPI):
    global proxy_state
    proxy_state = ProxyState(global_args.server_instances)
    logger.debug("Initialized %s dp server clients.", len(proxy_state.infer_servers))
    yield
    for p in proxy_state.infer_servers:
        await p.client.aclose()


async def listen_for_disconnect(request: Request) -> None:
    """Return when a disconnect message is received."""
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
        # Reset per retry to avoid leaking a stale True from a previous iteration
        first_chunk_sent = False
        try:
            async with client.stream("POST", endpoint, json=req_data, headers=headers) as response:
                response.raise_for_status()
                async for chunk in response.aiter_bytes():
                    first_chunk_sent = True
                    yield chunk
                return  # Success; exit after streaming completes
        except (httpx.RequestError, httpx.HTTPStatusError) as e:
            # After the first chunk is forwarded, retry is forbidden (would duplicate/corrupt the stream).
            if first_chunk_sent:
                logger.error("Streaming to client interrupted after response started: %s", str(e))
                return
            if attempt < max_retries:
                logger.warning("Attempt %s failed for streaming %s: %s", attempt, endpoint, str(e))
                await asyncio.sleep(base_delay * (2 ** (attempt - 1)))
            else:
                logger.error("All %s attempts failed for streaming %s.", max_retries, endpoint)
                raise e
        except Exception as e:
            # Same guard as above for non-HTTP exceptions
            if first_chunk_sent:
                logger.error("Streaming to client interrupted after response started: %s", str(e))
                return
            if attempt < max_retries:
                logger.warning("Attempt %s failed for streaming %s: %s", attempt, endpoint, str(e))
                await asyncio.sleep(base_delay * (2 ** (attempt - 1)))
            else:
                logger.error("All %s attempts failed for streaming %s.", max_retries, endpoint)
                raise e


async def _select_instance(api: str, req_data: Any, request_length: int):
    # refer to vLLM sampling_params: max_token default value
    max_tokens = req_data.get("max_tokens", 16)
    ignore_eos = req_data.get("ignore_eos", False)
    priority_score = 0.0
    if global_args.enable_dynamic_bucket:
        priority_score = proxy_state.calculate_request_tokens(request_length)
    else:
        priority_score = proxy_state.calculate_request_score(
            request_length, max_tokens=max_tokens, ignore_eos=ignore_eos
        )

    logger.debug(
        "Request length: %s, max tokens: %s, ignore_eos: %s, Priority score: %s",
        request_length,
        max_tokens,
        ignore_eos,
        priority_score,
    )
    request_id = await proxy_state.next_req_id()
    # Select server based on priority score
    request_tokens = proxy_state.calculate_request_tokens(request_length)
    group_idx, task = proxy_state.select_server_group(request_id, request_tokens, priority_score)

    try:
        server_idx = proxy_state.select_server(priority_score, group_idx)
    except Exception:
        if global_args.enable_dynamic_bucket and task is not None and proxy_state.bucket_load_balancer is not None:
            proxy_state.bucket_load_balancer.release_task(task.id)
        raise

    if global_args.enable_dynamic_bucket and task is not None:
        task.server_info = ServerInfo("DP", server_idx)

    chosen_server = proxy_state.infer_servers[server_idx]
    logger.debug(
        "[group_idx=%s, server_idx=%s] Choose server %s to process request %s",
        group_idx,
        server_idx,
        chosen_server.url,
        request_id,
    )
    return InstanceInfo(
        request_id=request_id, server_idx=server_idx, priority_score=priority_score, server_state=chosen_server
    )


@dataclass
class InstanceInfo:
    request_id: str
    server_idx: int
    priority_score: float
    server_state: ServerState


async def _handle_completions(api: str, request: Request):
    # streaming_started ensures release_server runs exactly once: in
    # generate_stream's finally on the normal path, or below if it never started.
    instance_info = None
    streaming_started = False
    try:
        req_data = await request.json()
        req_body = await request.body()
        request_length = len(req_body)
        instance_info = await _select_instance(api, req_data, request_length)

        async def generate_stream():
            nonlocal instance_info
            try:
                async for chunk in stream_service_response_with_retry(
                    instance_info.server_state.client,  # type: ignore
                    api,
                    req_data,
                    request_id=instance_info.request_id,  # type: ignore
                    max_retries=global_args.max_retries,
                    base_delay=global_args.retry_delay,
                ):
                    yield chunk
            except Exception as e:
                logger.error(
                    "Error during streaming from server %s: %s, the aborted request is: %s.",
                    instance_info.server_state.url,  # type: ignore
                    str(e),
                    instance_info.request_id,  # type: ignore
                )
            finally:
                # Release load after streaming completes
                proxy_state.release_server(  # type: ignore
                    instance_info.server_idx,  # type: ignore
                    instance_info.priority_score,  # type: ignore
                    instance_info.request_id,  # type: ignore
                )

        streaming_started = True
        return StreamingResponse(generate_stream(), media_type="application/json")
    except Exception as e:
        import traceback

        exc_info = sys.exc_info()
        print(f"Error occurred in external dp proxy server - {api} endpoint")
        print(e)
        print("".join(traceback.format_exception(*exc_info)))
        raise
    finally:
        # If streaming never started (client disconnect or selection error),
        # release here to avoid leaking active_tokens / the bucket task; the
        # normal path already released in generate_stream.
        if instance_info is not None and not streaming_started:
            proxy_state.release_server(instance_info.server_idx, instance_info.priority_score, instance_info.request_id)


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
        "server_instances": len(proxy_state.infer_servers),
    }


if __name__ == "__main__":
    global global_args
    global_args = parse_args()
    import uvicorn

    uvicorn.run(app, host=global_args.host, port=global_args.port)
