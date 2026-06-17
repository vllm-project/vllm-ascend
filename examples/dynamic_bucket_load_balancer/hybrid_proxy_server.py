# Adapted from https://github.com/vllm-project/vllm/tests/v1/kv_connector/nixl_integration/toy_proxy_server.py

# SPDX-License-Identifier: Apache-2.0
#
# Tutorial: Dynamic Bucketing-Based Hybrid Load Balance Proxy Server
#
# This proxy server distributes requests across multiple vLLM servers running
# for large language model inference. For each request it estimates a load
# score, picks the least-loaded backend instance, and can optionally split
# the backend pool into a short-request group and a long-request
# group (dynamic bucket load balancing).
#
# Prerequisites:
# - Python 3.10+
# - Install dependencies:
#     pip install "fastapi<0.124.0" httpx uvicorn
#
# Step 1: Start Your Backend Servers
# ----------------------------------
# Start at least two vLLM servers , each as a separate process on its own port.
# The proxy also works with a single backend, but load balancing is only
# meaningful with two or more.
#
#   vllm serve --host 0.0.0.0 --port 8100 ... # vLLM Server0
#   vllm serve --host 0.0.0.0 --port 8101 ... # vLLM Server1
#
# Step 2: Start the Proxy Server
# ------------------------------
# From examples/dynamic_bucket_load_balancer/, point the proxy at each backend
# with --server-hosts / --server-ports:
#
#   python hybrid_proxy_server.py \
#     --host 0.0.0.0 --port 8000 \
#     --server-hosts 127.0.0.1 127.0.0.1 \
#     --server-ports 8100 8101
#
# This starts the proxy on port 8000 and load balances across the two backends.
#
# To enable dynamic bucket load balancing (split the pool into short/long groups),
# add --enable-dynamic-bucket. The server count must be >= 2 so each bucket has at
# least one instance:
#
#   python hybrid_proxy_server.py \
#     --host 0.0.0.0 --port 8000 \
#     --server-hosts 127.0.0.1 127.0.0.1 127.0.0.1 127.0.0.1 \
#     --server-ports 8100 8101 8102 8103 \
#     --enable-dynamic-bucket \
#     --server-group-threshold 32768
#
# Step 3: Send a Request to the Proxy
# -----------------------------------
# Send OpenAI-compatible requests to the proxy. For example:
#
#   curl -X POST http://localhost:8000/v1/completions \
#     -H "Content-Type: application/json" \
#     -d '{
#           "model": "your-model",
#           "prompt": "The quick brown fox jumps over the lazy dog",
#           "max_tokens": 16
#         }'
#
# Or for chat completions:
#
#   curl -X POST http://localhost:8000/v1/chat/completions \
#     -H "Content-Type: application/json" \
#     -d '{
#           "model": "your-model",
#           "messages": [{"role": "user", "content": "Hello!"}],
#           "max_tokens": 16
#         }'
#
# Step 4: Health Check
# --------------------
# Check that the proxy is running and how many backends it fronts:
#
#   curl http://localhost:8000/healthcheck
#
# Returns a JSON object, e.g.:
#   {"status": "ok", "server_instances": 2}

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

# 如果可用，引入 uvloop 以获得更快的事件循环（event loop）
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
        self.aborted_requests = set()  # 记录已中止（aborted）的请求

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

        # 动态分桶负载均衡器
        self.bucket_load_balancer: DynamicBucketLoadBalancer | None = None

        if global_args.enable_dynamic_bucket:
            self.num_buckets = 2  # 启用动态分桶时的分长短两个桶
            self.server_group_threshold = global_args.server_group_threshold
            buckets = [(0, self.server_group_threshold), (self.server_group_threshold, global_args.max_request_tokens)]

            self.bucket_load_balancer = DynamicBucketLoadBalancer(buckets=buckets)
        else:
            self.num_buckets = 1  # 默认不分桶

        # 初始化优先级队列以高效地选择 server, 每个元素为 (优先级分数, server 索引, server 引用)
        # 优先级分数越小 = 优先级越高（负载越低）
        server_heap_items = [ServerHeapItem(0.0, i, server) for i, server in enumerate(self.infer_servers)]
        # 对Server进行分组
        self.server_heaps: list[list[ServerHeapItem]] = self._group_servers(server_heap_items, self.num_buckets)
        self.server_idx_to_group_idx = {}

        # 堆化每一分组
        for idx, cur_heap in enumerate(self.server_heaps):
            for server_item in cur_heap:
                self.server_idx_to_group_idx[server_item.server_idx] = idx
            heapq.heapify(cur_heap)

        # 记录动态分桶状态、分组数量及各分组完整实例信息
        logger.info(
            f"Dynamic bucket enabled: {global_args.enable_dynamic_bucket}, number of groups: {len(self.server_heaps)}"
        )
        for group_idx, cur_heap in enumerate(self.server_heaps):
            logger.info(f"Group {group_idx}: {cur_heap}")

    @staticmethod
    def _group_servers(servers: list[ServerHeapItem], num_groups: int):
        """
        将 servers 划分为 num_groups 个分组。

        Args:
            servers (list): 待分组的 server 列表。
            num_groups (int): 分组数量。

        Returns:
            list[list]: 分组后的 server 列表。

        Raises:
            ValueError: 当 num_groups <= 0 时抛出。
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
        """更新堆中某个 server 的优先级。"""
        server = self.infer_servers[server_idx]
        priority = server.active_tokens
        # 先移除旧条目，再加入新条目
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

        # 更新被选中的 server（累加负载）
        self.infer_servers[chosen_server_idx].active_tokens += token_count

        # 更新优先级并重新加入堆
        self._update_server_priority(chosen_server_idx)

        return chosen_server_idx

    def release_server(self, idx: int, token_count, req_id):
        self.infer_servers[idx].active_tokens -= token_count
        if global_args.enable_dynamic_bucket and req_id is not None:
            self.bucket_load_balancer.release_task(req_id)
        # 释放后更新优先级队列
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
        """根据请求长度与各分组当前负载，选择最优分组。"""
        if global_args.enable_dynamic_bucket:
            group_idx, task = self.bucket_load_balancer.dispatch_single_task(req_id, request_tokens, priority_score)
            return group_idx, task
        else:
            return 0, None


proxy_state: ProxyState | None = None


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
    logger.debug(f"Initialized {len(proxy_state.infer_servers)} dp server clients.")
    yield
    for p in proxy_state.infer_servers:
        await p.client.aclose()


async def listen_for_disconnect(request: Request) -> None:
    """收到 disconnect 消息时返回。"""
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
        # 每次重试都重置，避免上一次迭代遗留的 True 跨重试泄漏
        first_chunk_sent = False
        try:
            async with client.stream("POST", endpoint, json=req_data, headers=headers) as response:
                response.raise_for_status()
                async for chunk in response.aiter_bytes():
                    first_chunk_sent = True
                    yield chunk
                return  # 成功，流式结束后退出
        except (httpx.RequestError, httpx.HTTPStatusError) as e:
            # 一旦已经向客户端转发过任何 chunk，就不能再重试，否则客户端会收到重复/损坏的流。
            if first_chunk_sent:
                logger.error(f"Streaming to client interrupted after response started: {str(e)}")
                return
            if attempt < max_retries:
                logger.warning(f"Attempt {attempt} failed for streaming {endpoint}: {str(e)}")
                await asyncio.sleep(base_delay * (2 ** (attempt - 1)))
            else:
                logger.error(f"All {max_retries} attempts failed for streaming {endpoint}.")
                raise e
        except Exception as e:
            # 对非 HTTP 异常沿用与上相同的防护
            if first_chunk_sent:
                logger.error(f"Streaming to client interrupted after response started: {str(e)}")
                return
            if attempt < max_retries:
                logger.warning(f"Attempt {attempt} failed for streaming {endpoint}: {str(e)}")
                await asyncio.sleep(base_delay * (2 ** (attempt - 1)))
            else:
                logger.error(f"All {max_retries} attempts failed for streaming {endpoint}.")
                raise e


async def _select_instance(api: str, req_data: Any, request_length: int):
    # refer to vLLM sampling_params: max_token default value
    max_tokens = req_data.get("max_tokens", 16)
    ignore_eos = req_data.get("ignore_eos", False)
    priority_score = 0
    if global_args.enable_dynamic_bucket:
        priority_score = proxy_state.calculate_request_tokens(request_length)
    else:
        priority_score = proxy_state.calculate_request_score(
            request_length, max_tokens=max_tokens, ignore_eos=ignore_eos
        )

    logger.debug(
        f"Request length: {request_length}, max tokens: {max_tokens}, "
        f"ignore_eos: {ignore_eos}, Priority score: {priority_score}"
    )
    request_id = await proxy_state.next_req_id()
    # Select server based on priority score
    request_tokens = proxy_state.calculate_request_tokens(request_length)
    group_idx, task = proxy_state.select_server_group(request_id, request_tokens, priority_score)

    try:
        server_idx = proxy_state.select_server(priority_score, group_idx)
    except Exception:
        if global_args.enable_dynamic_bucket and task is not None:
            proxy_state.bucket_load_balancer.release_task(task.id)
        raise

    if global_args.enable_dynamic_bucket and task is not None:
        task.server_info = ServerInfo("DP", server_idx)

    chosen_server = proxy_state.infer_servers[server_idx]
    logger.debug(
        f"[group_idx={group_idx}, server_idx={server_idx}] Choose server {chosen_server.url} to process request {request_id}"
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
    # streaming_started 确保 release_server 只执行一次：要么在 generate_stream 的
    # finally 中（正常路径），要么在流未启动的路径。
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
                    instance_info.server_state.client,
                    api,
                    req_data,
                    request_id=instance_info.request_id,
                    max_retries=global_args.max_retries,
                    base_delay=global_args.retry_delay,
                ):
                    yield chunk
            except Exception as e:
                logger.error(
                    f"Error during streaming from server {instance_info.server_state.url}: {str(e)}, "
                    f"the aborted request is: {instance_info.request_id}."
                )
            finally:
                # 流式结束后，释放负载
                proxy_state.release_server(
                    instance_info.server_idx, instance_info.priority_score, instance_info.request_id
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
        # 流从未启动（例如客户端断连，或实例选择阶段出错）：在这里释放，
        # 以免 active_tokens 和桶任务泄漏。正常路径下 generate_stream 已释放，故跳过。
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
