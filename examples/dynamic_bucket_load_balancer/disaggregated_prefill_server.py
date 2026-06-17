# Adapted from https://github.com/vllm-project/vllm/tests/v1/kv_connector/nixl_integration/toy_proxy_server.py

# SPDX-License-Identifier: Apache-2.0
#
# Tutorial: Dynamic Bucketing-Based Disaggregated Prefill Proxy Server
#
# This proxy server implements disaggregated prefill (PD disaggregation): each
# request is prefilled on a "prefiller" backend (KV producer), its KV cache is
# transferred to a "decoder" backend (KV consumer), and the response is decoded
# and streamed from the decoder. It load balances across multiple prefiller and
# decoder instances, and can optionally split the prefiller pool into a
# short-request group and a long-request group (dynamic bucket load balancing).
#
# Prerequisites:
# - Python 3.10+
# - Install dependencies:
#     pip install "fastapi<0.124.0" httpx uvicorn vllm
#
# Step 1: Start Your Backend Servers
# ----------------------------------
# Start prefiller (KV producer) and decoder (KV consumer) vLLM servers, each on
# its own port, configured as a disaggregated-prefill pair via --kv-transfer-config
# (e.g. MooncakeConnectorV1 with kv_role "kv_producer"/"kv_consumer"). See
# examples/disaggregated_prefill_v1/mooncake_connector_deployment_guide.md for the
# full config. You need at least one prefiller and one decoder.
#
#   # Prefiller (kv_role: kv_producer)
#   vllm serve <model> --host 0.0.0.0 --port 8100 \
#     --kv-transfer-config '{"kv_connector":"MooncakeConnectorV1","kv_role":"kv_producer",...}'
#   # Decoder (kv_role: kv_consumer)
#   vllm serve <model> --host 0.0.0.0 --port 8200 \
#     --kv-transfer-config '{"kv_connector":"MooncakeConnectorV1","kv_role":"kv_consumer",...}'
#
# Step 2: Start the Proxy Server
# ------------------------------
# From examples/dynamic_bucket_load_balancer/, point the proxy at each prefiller
# and decoder with --prefiller-hosts/--prefiller-ports and
# --decoder-hosts/--decoder-ports:
#
#   python disaggregated_prefill_server.py \
#     --host 0.0.0.0 --port 8000 \
#     --prefiller-hosts 127.0.0.1 127.0.0.1 \
#     --prefiller-ports 8100 8101 \
#     --decoder-hosts 127.0.0.1 127.0.0.1 \
#     --decoder-ports 8200 8201
#
# This starts the proxy on port 8000 across two prefillers and two decoders.
#
# To enable dynamic bucket load balancing (split the prefiller pool into
# short/long groups), add --enable-dynamic-bucket. The prefiller count must be
# >= 2 so each bucket has at least one instance:
#
#   python disaggregated_prefill_server.py \
#     --host 0.0.0.0 --port 8000 \
#     --prefiller-hosts 127.0.0.1 127.0.0.1 127.0.0.1 127.0.0.1 \
#     --prefiller-ports 8100 8101 8102 8103 \
#     --decoder-hosts 127.0.0.1 127.0.0.1 \
#     --decoder-ports 8200 8201 \
#     --enable-dynamic-bucket \
#     --prefill-group-threshold 32768
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
# Check that the proxy is running and how many prefiller/decoder instances it
# fronts:
#
#   curl http://localhost:8000/healthcheck
#
# Returns a JSON object, e.g.:
#   {"status": "ok", "prefill_instances": 2, "decode_instances": 2}
#
# Step 5: Add or Remove Prefiller or Decoder Instances (Optional)
# ---------------------------------------------------------------
# You can add or remove prefiller or decoder instances after the proxy is started.
# For example, add 2 prefiller instances (dynamic bucketing off):
#
#   curl -X POST http://localhost:8000/instances/add \
#     -H "Content-Type: application/json" \
#     -d '{
#           "type": "prefill",
#           "instances": ["127.0.0.1:8102", "127.0.0.1:8103"]
#         }'
#
# When dynamic bucketing is enabled (--enable-dynamic-bucket), adding a prefiller
# REQUIRES a "bucket" field that selects which prefill bucket it joins: "short"
# (short-request bucket) or "long" (long-request bucket). Omitting it for a
# prefiller is rejected while dynamic bucketing is on. "bucket" only applies to
# prefillers and is ignored when dynamic bucketing is off or for decode instances:
#
#   curl -X POST http://localhost:8000/instances/add \
#     -H "Content-Type: application/json" \
#     -d '{
#           "type": "prefill",
#           "instances": ["127.0.0.1:8102"],
#           "bucket": "long"
#         }'
#
# Remove 1 decoder instance ("instances" may be a string or a list):
#
#   curl -X POST http://localhost:8000/instances/remove \
#     -H "Content-Type: application/json" \
#     -d '{
#           "type": "decode",
#           "instances": "127.0.0.1:8201"
#         }'

import argparse
import asyncio
import functools
import heapq
import ipaddress
import json
import os
import sys
import threading
import time
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


@dataclass
class InstanceType:
    PREFILL: str = "prefill"
    DECODE: str = "decode"


TAINT_PRIORITY = 1e15

# 桶名 -> 分组索引。仅在开启动态分桶时有意义：此时 prefiller 池被拆分为两个桶，
# 分组 0 服务短请求，分组 1 服务长请求。
BUCKET_SHORT = "short"
BUCKET_LONG = "long"
BUCKET_NAME_TO_GROUP = {BUCKET_SHORT: 0, BUCKET_LONG: 1}


class ServerState:
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.url = f"http://{host}:{port}/v1"
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
        self.active_kv_cache = 0  # 仅用于 prefiller
        self.active_requests = 0  # 当前活跃请求数
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
    def __init__(self, prefiller_instances, decoder_instances):
        self.request_num = 0
        self.tainted_prefillers: list[ServerState] = []
        self.tainted_decoders: list[ServerState] = []
        self.node_listener = NodeListener(self)

        self.prefillers: list[ServerState] = [ServerState(h, p) for h, p in prefiller_instances]
        self.decoders: list[ServerState] = [ServerState(h, p) for h, p in decoder_instances]
        self.req_to_prefiller = {}
        self.req_id_lock = asyncio.Lock()
        # 关闭已移除实例 httpx client 的后台任务；保留引用避免 aclose() task在执行完成前被gc回收, 导致连接关闭中途丢失
        self._pending_closes: set = set()

        # 动态分桶负载均衡器
        self.bucket_load_balancer: DynamicBucketLoadBalancer | None = None

        if global_args.enable_dynamic_bucket:
            self.num_prefill_buckets = 2  # 启用动态分桶时的分组数量
            self.prefill_group_threshold = global_args.prefill_group_threshold
            prefill_buckets = [
                (0, self.prefill_group_threshold),
                (self.prefill_group_threshold, global_args.max_request_tokens),
            ]

            self.bucket_load_balancer = DynamicBucketLoadBalancer(buckets=prefill_buckets)
        else:
            self.num_prefill_buckets = 1  # 默认不分组

        # 初始化优先级队列以高效地选择 server, 每个元素为 (优先级分数, server 索引, server 引用)
        # 优先级分数越小 = 优先级越高（负载越低）
        prefiller_heap_items = [ServerHeapItem(0.0, i, server) for i, server in enumerate(self.prefillers)]
        # 对Server进行分组
        self.prefiller_heaps: list[list[ServerHeapItem]] = self._group_servers(
            prefiller_heap_items, self.num_prefill_buckets
        )
        self.server_idx_to_group_idx = {}
        self.prefill_group_load = {}
        # 堆化每一分组
        for idx, cur_heap in enumerate(self.prefiller_heaps):
            for server_item in cur_heap:
                self.server_idx_to_group_idx[server_item.server_idx] = idx
            heapq.heapify(cur_heap)
            self.prefill_group_load[idx] = 0

        # 记录动态分桶状态、分组数量及各分组完整实例信息
        logger.info(
            f"Dynamic bucket enabled: {global_args.enable_dynamic_bucket}, "
            f"number of prefill groups: {len(self.prefiller_heaps)}"
        )
        for group_idx, cur_heap in enumerate(self.prefiller_heaps):
            logger.info(f"Prefill Group {group_idx}: {cur_heap}")

        self.decoder_heap: list[ServerHeapItem] = [
            ServerHeapItem(0.0, i, server) for i, server in enumerate(self.decoders)
        ]
        heapq.heapify(self.decoder_heap)

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

    @staticmethod
    def _add_new_server_to_heaps(server_heaps: list[list[ServerHeapItem]], new_server_heap_item: ServerHeapItem):
        """
        添加新的 Server 到 Server堆分组 中，并返回新的 Server堆分组 和 添加的Server所在堆分组索引
        """
        num_heaps = len(server_heaps)

        # 仅一个分组
        if num_heaps == 1:
            heapq.heappush(server_heaps[0], new_server_heap_item)
            return server_heaps, 0

        for idx, group in enumerate(server_heaps):
            # 当前分组比下一分组元素多，将新Server添加到下一分组
            if idx < num_heaps - 1 and len(server_heaps[idx]) > len(server_heaps[idx + 1]):
                heapq.heappush(server_heaps[idx + 1], new_server_heap_item)
                return server_heaps, idx + 1

        # 所有分组元素个数相同，添加到第一分组
        heapq.heappush(server_heaps[0], new_server_heap_item)
        return server_heaps, 0

    def _update_prefiller_priority(self, server_idx: int):
        """更新堆中某个 prefiller server 的优先级。"""
        server = self.prefillers[server_idx]
        # 优先级由 active_tokens 与 active_kv_cache 计算得到
        priority = server.active_tokens + server.active_kv_cache * 0.3
        # 先移除旧条目，再加入新条目
        group_idx = self.server_idx_to_group_idx[server_idx]

        self.prefiller_heaps[group_idx] = [
            server_heap_item
            for server_heap_item in self.prefiller_heaps[group_idx]
            if server_heap_item.server_idx != server_idx
        ]
        self.prefiller_heaps[group_idx].append(ServerHeapItem(priority, server_idx, server))
        heapq.heapify(self.prefiller_heaps[group_idx])

    def _update_decoder_priority(self, server_idx: int):
        """更新堆中某个 decoder server 的优先级。"""
        server = self.decoders[server_idx]
        priority = server.active_tokens
        # 先移除旧条目，再加入新条目
        self.decoder_heap = [
            server_heap_item for server_heap_item in self.decoder_heap if server_heap_item.server_idx != server_idx
        ]
        self.decoder_heap.append(ServerHeapItem(priority, server_idx, server))
        heapq.heapify(self.decoder_heap)

    def _restore_prefiller_priorities(self, servers) -> None:
        """重新计算并应用给定 prefiller 在堆中的优先级。

        用于移除尝试未能驱逐的 server，使其重新可被选择，而不是被卡在 TAINT_PRIORITY 上。
        """
        for idx, server in enumerate(self.prefillers):
            if server in servers:
                self._update_prefiller_priority(idx)

    def _restore_decoder_priorities(self, servers) -> None:
        """重新计算并应用给定 decoder 在堆中的优先级"""
        for idx, server in enumerate(self.decoders):
            if server in servers:
                self._update_decoder_priority(idx)

    def _schedule_close(self, server: "ServerState") -> None:
        """关闭已移除实例的 httpx client，且不阻塞 server 选择。

        在主事件循环上以"即发即忘（fire-and-forget）"方式执行（remove_* 总是在主循环中运行）。
        保留任务引用是为了避免它在 aclose() 完成前被 GC 回收。
        仅在 request_num == 0 时才会被调用，因此该 client 没有在途请求。
        """
        task = asyncio.create_task(server.client.aclose())
        self._pending_closes.add(task)
        task.add_done_callback(self._pending_closes.discard)

    def abort_prefiller_request(self, server_idx: int, request_id):
        """
        将某个请求标记为已中止（aborted）。这有助于在 prefiller 节点上
        释放对应的 KV cache。
        """
        if server_idx >= len(self.prefillers):
            return
        self.prefillers[server_idx].aborted_requests.add(request_id)

    def acquire_aborted_prefiller_requests(self, server_idx: int):
        """
        获取已中止请求的集合，并将其清空。
        该集合用于在 prefiller 节点上释放 KV cache。
        """
        if server_idx >= len(self.prefillers):
            return set()
        aborted_requests = self.prefillers[server_idx].aborted_requests.copy()
        self.prefillers[server_idx].aborted_requests.clear()
        return aborted_requests

    async def next_req_id(self):
        async with self.req_id_lock:
            return str(uuid.uuid4())

    def select_prefiller(self, token_count, group_idx):
        if not self.prefillers:
            raise RuntimeError("No prefiller servers available")

        if not self.prefiller_heaps[group_idx]:
            raise RuntimeError(f"No prefiller servers available in bucket group {group_idx}")

        server_heap_item: ServerHeapItem = heapq.heappop(self.prefiller_heaps[group_idx])
        chosen_server_idx = server_heap_item.server_idx

        # 更新被选中的 server（累加负载）
        self.prefillers[chosen_server_idx].active_tokens += token_count
        self.prefillers[chosen_server_idx].active_kv_cache += token_count
        self.prefill_group_load[group_idx] += token_count

        # 更新优先级并重新加入堆
        self._update_prefiller_priority(chosen_server_idx)

        return chosen_server_idx

    def release_prefiller(self, idx, token_count, task):
        if idx >= len(self.prefillers):
            raise ValueError(f"No prefiller servers with idx = {idx}")
        self.prefillers[idx].active_tokens -= token_count
        group_idx = self.server_idx_to_group_idx[idx]
        self.prefill_group_load[group_idx] -= token_count
        if global_args.enable_dynamic_bucket and task is not None:
            self.bucket_load_balancer.release_task(task.id)
        # 释放后更新优先级队列
        self._update_prefiller_priority(idx)

    def release_prefiller_kv(self, idx, token_count):
        if idx >= len(self.prefillers):
            return
        if self.prefillers[idx].active_kv_cache > 0:
            self.prefillers[idx].active_kv_cache -= token_count
        # 释放后更新优先级队列
        self._update_prefiller_priority(idx)

    def select_decoder(self, token_count):
        if not self.decoder_heap:
            raise RuntimeError("No decoder servers available")

        server_heap_item = heapq.heappop(self.decoder_heap)
        chosen_server_idx = server_heap_item.server_idx

        # 更新被选中的 server（累加负载）
        self.decoders[chosen_server_idx].active_tokens += token_count

        # 更新优先级并重新加入堆
        self._update_decoder_priority(chosen_server_idx)

        return chosen_server_idx

    def release_decoder(self, idx, token_count):
        if idx >= len(self.decoders):
            return
        self.decoders[idx].active_tokens -= token_count
        # 释放后更新优先级队列
        self._update_decoder_priority(idx)

    # Omni_infer's calculate_input_scores function
    def calculate_prefill_scores(self, request_length: int) -> float:
        length_score = request_length / 4.0
        input_score = length_score * 0.0345 + 120.0745
        return input_score

    def calculate_decode_scores(self, request_length: int) -> float:
        return request_length

    def calculate_prefill_tokens(self, request_length: int) -> float:
        return request_length / 4.0

    def select_prefill_group(self, req_id: str, request_tokens, prefill_score) -> tuple[int, Task | None]:
        """根据请求长度与各分组当前负载，选择最优分组。"""
        if global_args.enable_dynamic_bucket:
            group_idx, task = self.bucket_load_balancer.dispatch_single_task(req_id, request_tokens, prefill_score)
            return group_idx, task
        else:
            return 0, None

    async def add_instances(
        self,
        instance_type: str,
        instances: list[ServerState],
        bucket: str | None = None,
    ) -> tuple[list[str], list[str]]:
        added_nodes, waiting_nodes = [], []
        for server in instances:
            is_valid = await self.node_listener.check_instance_status(server.client)
            if is_valid and instance_type == InstanceType.PREFILL:
                self.add_prefillers([server], bucket)
                added_nodes.append(str(server))
            elif is_valid and instance_type == InstanceType.DECODE:
                self.add_decoders([server])
                added_nodes.append(str(server))
            else:
                node = str(server)
                # 对于新增prefill实例，记录请求的桶，以便后续重试（实例健康后）时把它放入同一个桶。
                self.node_listener.waiting_nodes[node] = (instance_type, server, 0, bucket)
                waiting_nodes.append(node)
        return added_nodes, waiting_nodes

    def add_prefillers(self, instances: list[ServerState], bucket: str | None = None) -> None:
        # 为新增的 prefiller 确定目标桶。显式指定的 "short"/"long" 仅在开启动态分桶（两个桶）时生效
        if bucket is not None and self.num_prefill_buckets > 1:
            target_group_idx = BUCKET_NAME_TO_GROUP.get(bucket)
            if target_group_idx is None:
                logger.warning(f"Unsupported bucket '{bucket}'; falling back to automatic placement.")
        else:
            if bucket is not None and self.num_prefill_buckets <= 1:
                logger.warning(
                    f"Bucket '{bucket}' requested but dynamic bucketing is off; "
                    f"the instance is added to the single prefill bucket."
                )
            target_group_idx = None  # 通过 _add_new_server_to_heaps 自动放置

        for server in instances:
            if server in self.tainted_prefillers:
                self.tainted_prefillers.remove(server)

                # 适配动态分桶
                for group_idx, heap in enumerate(self.prefiller_heaps):
                    re_heapify_flag = False
                    for server_heap_item in heap:
                        if server_heap_item.server == server:
                            server_heap_item.priority = 0
                            self.server_idx_to_group_idx[server_heap_item.server_idx] = group_idx
                            re_heapify_flag = True
                    if re_heapify_flag:
                        heapq.heapify(heap)
            elif server not in self.prefillers:
                self.prefillers.append(server)

                new_prefiller_server_idx = len(self.prefillers) - 1
                new_server_heap_item = ServerHeapItem(0, new_prefiller_server_idx, server)
                if target_group_idx is not None:
                    heapq.heappush(self.prefiller_heaps[target_group_idx], new_server_heap_item)
                    group_idx = target_group_idx
                else:
                    self.prefiller_heaps, group_idx = self._add_new_server_to_heaps(
                        self.prefiller_heaps, new_server_heap_item
                    )
                self.server_idx_to_group_idx[new_prefiller_server_idx] = group_idx

        self.print_status(f"Add prefiller instances: {instances}.")

    def add_decoders(self, instances: list[ServerState]) -> None:
        for server in instances:
            if server in self.tainted_decoders:
                self.tainted_decoders.remove(server)

                for server_heap_item in self.decoder_heap:
                    if server_heap_item.server == server:
                        server_heap_item.priority = 0
                heapq.heapify(self.decoder_heap)

            elif server not in self.decoders:
                self.decoders.append(server)
                # decoder_heap: [(priority_0, 0, server_0)] -> [(priority_0, 0, server_0), (0, 1, server_1)]（添加后 heap 多出一项）
                heapq.heappush(self.decoder_heap, ServerHeapItem(0, len(self.decoders) - 1, server))
        self.print_status(f"Add decoder instances: {instances}.")

    def remove_prefillers(self, instances: list[ServerState]) -> bool:
        if not instances:
            return False

        if self.request_num > 0:
            logger.warning(f"Start to taint prefill instances {instances}.")
            self._taint_prefillers(instances)
            return True

        instances_to_remove = set(instances)

        # 每个 prefiller 必须留在其所属桶内：桶反映了其 server 应服务的请求长度类别，
        # 因此移除时不能跨桶重新均衡 server。相应地，拒绝任何会使某个桶变空的移除操作
        # （否则路由到空桶的请求会在 select_prefiller 中触发 IndexError）。
        for group_idx, heap in enumerate(self.prefiller_heaps):
            if all(item.server in instances_to_remove for item in heap):
                logger.warning(f"Refusing removal: prefill bucket group {group_idx} would be left empty")
                # 解除污染，使这些 server 保持可用，避免在调用方清空列表后被卡在 TAINT_PRIORITY 上。
                self._restore_prefiller_priorities(instances_to_remove)
                return False

        new_prefillers = []
        old_idx_to_new_idx = {}
        for idx, server in enumerate(self.prefillers):
            if server not in instances_to_remove:
                new_prefillers.append(server)
                old_idx_to_new_idx[idx] = len(new_prefillers) - 1
            else:
                if server.active_tokens != 0 or server.active_kv_cache != 0 or server.active_requests != 0:
                    logger.warning("Prefill server is not empty, please wait for all requests to be completed")
                    # 与上文同理：在拒绝前解除污染。
                    self._restore_prefiller_priorities(instances_to_remove)
                    return False

        removed_servers = [s for s in self.prefillers if s in instances_to_remove]
        self.prefillers = new_prefillers

        # 重建堆，保留每个存活 server 所属的桶，并将其索引重映射到新的（紧凑化后的）prefiller 列表中。
        new_prefiller_heaps: list[list[ServerHeapItem]] = []
        for group_idx, heap in enumerate(self.prefiller_heaps):
            new_heap = []
            for server_item in heap:
                if server_item.server not in instances_to_remove:
                    old_idx = server_item.server_idx
                    server_item.server_idx = old_idx_to_new_idx[old_idx]
                    new_heap.append(server_item)

            new_prefiller_heaps.append(new_heap)

        self.prefiller_heaps = new_prefiller_heaps

        for group_idx, heap in enumerate(self.prefiller_heaps):
            for server_item in heap:
                self.server_idx_to_group_idx[server_item.server_idx] = group_idx
            heapq.heapify(heap)

        for group_idx, heap in enumerate(self.prefiller_heaps):
            self.prefill_group_load[group_idx] = 0
            for server_item in heap:
                self.prefill_group_load[group_idx] += server_item.server.active_tokens

        # 关闭已移除实例的 httpx client，避免连接泄漏。
        for server in removed_servers:
            self._schedule_close(server)

        self.print_status(f"Remove prefiller instances: {instances}.")
        return False

    def remove_decoders(self, instances: list[ServerState]) -> bool:
        if not instances:
            return False

        if self.request_num > 0:
            logger.warning(f"Start to taint decode instances {instances}.")
            self._taint_decoders(instances)
            return True

        instances_to_remove = set(instances)

        # 至少保留一个 decoder：当池为空时 select_decoder 会抛异常。
        if all(server in instances_to_remove for server in self.decoders):
            logger.warning("Refusing removal: no decoder would be left after removal")
            # [A1] 解除污染，使这些 decoder 保持可选，而不是被卡住。
            self._restore_decoder_priorities(instances_to_remove)
            return False

        # 拒绝移除仍有在途请求的 decoder，与 remove_prefillers 保持一致； 否则下方的重建+重新索引会使活跃流持有的 decoder_idx 失效
        # （release_decoder 会作用到错误的 decoder 上）。
        for server in self.decoders:
            if server in instances_to_remove and server.active_tokens != 0:
                logger.warning("Decode server is not empty, please wait for all requests to be completed")
                # 解除污染，使该 decoder 保持可选，避免在调用方清空列表后被卡在 TAINT_PRIORITY 上。
                self._restore_decoder_priorities(instances_to_remove)
                return False
        removed_servers = [server for server in self.decoders if server in instances_to_remove]
        self.decoders = [server for server in self.decoders if server not in instances_to_remove]
        decoder_heap_copy: list[ServerHeapItem] = self.decoder_heap.copy()
        decoder_heap_copy.sort(key=lambda x: x.server_idx)  # 按键（decoder_idx）排序
        decoder_heap = []
        idx = 0
        for server_heap_item in decoder_heap_copy:
            if server_heap_item.server not in instances_to_remove:
                decoder_heap.append(ServerHeapItem(server_heap_item.priority, idx, server_heap_item.server))
                idx += 1

        self.decoder_heap = decoder_heap
        heapq.heapify(self.decoder_heap)
        # 关闭已移除实例的 httpx client，避免连接泄漏。
        for server in removed_servers:
            self._schedule_close(server)

        self.print_status(f"Remove decoder instances: {instances}.")
        return False

    def _drain_tainted_instances(self) -> None:
        """当没有在途请求时，尝试驱逐被污染的实例。"""
        if self.tainted_prefillers and not self.request_num:
            need_waiting = self.remove_prefillers(self.tainted_prefillers)
            if not need_waiting:
                self.tainted_prefillers.clear()

        if self.tainted_decoders and not self.request_num:
            need_waiting = self.remove_decoders(self.tainted_decoders)
            if not need_waiting:
                self.tainted_decoders.clear()

    def _taint_prefillers(self, instances: list[ServerState]) -> None:
        instances_to_taint = set(instances)
        for server in self.prefillers:
            if server in instances_to_taint and server not in self.tainted_prefillers:
                self.tainted_prefillers.append(server)

        for group_idx, heap in enumerate(self.prefiller_heaps):
            re_heapify_flag = False
            for server_item in heap:
                if server_item.server in instances_to_taint:
                    server_item.priority = TAINT_PRIORITY
                    re_heapify_flag = True
            if re_heapify_flag:
                heapq.heapify(heap)

    def _taint_decoders(self, instances: list[ServerState]) -> None:
        instances_to_taint = set(instances)
        for server in self.decoders:
            if server in instances_to_taint and server not in self.tainted_decoders:
                self.tainted_decoders.append(server)

        re_heapify_flag = False
        for server_item in self.decoder_heap:
            if server_item.server in instances_to_taint:
                server_item.priority = TAINT_PRIORITY
                re_heapify_flag = True
        if re_heapify_flag:
            heapq.heapify(self.decoder_heap)

    def print_status(self, msg: str) -> None:
        status = {
            "prefill_instances": [str(server) for server in self.prefillers],
            "decode_instances": [str(server) for server in self.decoders],
        }
        print(f"{msg} Status: {status}")


proxy_state: ProxyState | None = None


class NodeListener:
    def __init__(self, proxy):
        self.proxy_state = proxy
        # value（字典值）: (instance_type, server, check_times, requested_bucket)
        self.waiting_nodes: dict[str, tuple[str, Any, int, str | None]] = {}
        # 捕获主事件循环。后台线程从不直接操作共享的 prefiller/decoder 结构，
        # 而是把整个轮询周期都编排（marshal）到这个循环上执行。这样所有对共享状态
        # （堆、列表、request_num、waiting_nodes）的修改都保持在单线程内，使得
        # 同步的 add/remove/select/release 方法无需加锁也不会有竞态；request_num
        # 的驱逐防护相对于请求选择成为原子操作；就绪探针也复用每个 server 自身的
        # httpx client，运行在拥有它的循环上（不会出现跨循环使用 client 的情况）。
        self.loop = asyncio.get_running_loop()
        self.listening_thread = threading.Thread(target=self._node_listener, daemon=True)
        self.listening_thread.start()

    def _node_listener(self) -> None:
        while True:
            try:
                # 在主循环上运行整个周期，并阻塞直到其完成。
                future = asyncio.run_coroutine_threadsafe(self._cycle(), self.loop)
                future.result()
            except Exception as e:
                logger.error(f"Node listener cycle failed: {e}")
            time.sleep(global_args.waiting_retry_interval)

    async def _cycle(self) -> None:
        """一次轮询周期；在主事件循环上执行。"""
        for node, (instance_type, server, check_times, bucket) in list(self.waiting_nodes.items()):
            print(f"Checking instance {node}...")
            check_times += 1
            is_valid = await self.check_instance_status(server.client)
            if is_valid:
                if instance_type == InstanceType.PREFILL:
                    self.proxy_state.add_prefillers([server], bucket)
                else:
                    self.proxy_state.add_decoders([server])
                self.waiting_nodes.pop(node)
            elif check_times == global_args.max_waiting_retries:
                print(f"Instance {node} was not added to the proxy.")
                self.waiting_nodes.pop(node)
            else:
                self.waiting_nodes[node] = (instance_type, server, check_times, bucket)

        self.proxy_state._drain_tainted_instances()

    @staticmethod
    async def check_instance_status(client: httpx.AsyncClient) -> bool:
        endpoint = "/models"
        headers = {"Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}"}
        try:
            response = await client.get(endpoint, headers=headers)
            response.raise_for_status()
            return True
        except (httpx.RequestError, httpx.HTTPStatusError):
            return False


def parse_args(args_list=None):
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
    parser.add_argument("--prefill-group-threshold", type=int, default=32 * 1024, help="Threshold of prefill groups")
    parser.add_argument("--max-request-tokens", type=int, default=128 * 1024, help="Max tokens of request")
    parser.add_argument(
        "--enable-dynamic-bucket", action="store_true", default=False, help="Enable dynamic bucket load Balancer"
    )
    args = parser.parse_args(args_list)
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
    proxy_state = ProxyState(global_args.prefiller_instances, global_args.decoder_instances)
    print(f"Initialized {len(proxy_state.prefillers)} prefill clients and {len(proxy_state.decoders)} decode clients.")
    yield
    for p in proxy_state.prefillers:
        await p.client.aclose()
    for d in proxy_state.decoders:
        await d.client.aclose()


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


async def send_request_to_service(
    client: httpx.AsyncClient,
    prefiller_id: int,
    endpoint: str,
    req_data: dict,
    request_id: str,
    max_retries: int = 3,
    base_delay: float = 0.2,
):
    aborted_requests = proxy_state.acquire_aborted_prefiller_requests(prefiller_id)
    req_data = req_data.copy()
    req_data["kv_transfer_params"] = {
        "do_remote_decode": True,
        "do_remote_prefill": False,
        "remote_engine_id": None,
        "remote_block_ids": None,
        "remote_host": None,
        "remote_port": None,
        "aborted_request": list(aborted_requests),
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
        except (httpx.RequestError, httpx.HTTPStatusError) as e:
            logger.warning(f"Attempt {attempt} failed for {endpoint}: {str(e)}")
            last_exc = e
            if attempt < max_retries:
                await asyncio.sleep(base_delay * (2 ** (attempt - 1)))
            else:
                logger.error(f"All {max_retries} attempts failed for {endpoint}.")
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
                return  # 成功，流式结束后退出
        except (httpx.RequestError, httpx.HTTPStatusError) as e:
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
            # 如果已经向客户端发送过任何 chunk，则不再重试，只记录日志并丢弃
            if "first_chunk_sent" in locals() and first_chunk_sent:
                logger.error(f"Streaming to client interrupted after response started: {str(e)}")
                return
            else:
                if attempt < max_retries:
                    logger.warning(f"Attempt {attempt} failed for streaming {endpoint}: {str(e)}")
                    await asyncio.sleep(base_delay * (2 ** (attempt - 1)))
                else:
                    logger.error(f"All {max_retries} attempts failed for streaming {endpoint}.")
                    raise e


async def _handle_select_instance(api: str, req_data: Any, request_length: int):
    prefiller_score = 0
    if proxy_state.num_prefill_buckets > 1:
        prefiller_score = proxy_state.calculate_prefill_tokens(request_length)
    else:
        prefiller_score = proxy_state.calculate_prefill_scores(request_length)
    logger.debug(f"Request length: {request_length}, Prefiller score: {prefiller_score}")
    request_id = await proxy_state.next_req_id()
    # 选择 prefiller
    request_tokens = proxy_state.calculate_prefill_tokens(request_length)
    group_idx, task = proxy_state.select_prefill_group(request_id, request_tokens, prefiller_score)

    logger.debug(f"Selected group_idx: {group_idx}")

    try:
        prefiller_idx = proxy_state.select_prefiller(prefiller_score, group_idx)
    except Exception:
        if global_args.enable_dynamic_bucket and task is not None:
            proxy_state.bucket_load_balancer.release_task(task.id)
        raise

    if global_args.enable_dynamic_bucket and task is not None:
        task.server_info = ServerInfo(InstanceType.PREFILL, prefiller_idx)

    prefiller = proxy_state.prefillers[prefiller_idx]
    # 向 prefiller 发送请求
    try:
        response = await send_request_to_service(
            prefiller.client,
            prefiller_idx,
            api,
            req_data,
            request_id,
            max_retries=global_args.max_retries,
            base_delay=global_args.retry_delay,
        )
    except Exception:
        # prefill 失败：在重新抛出异常前，回滚 select_prefiller 所获取的资源
        # （active_tokens、active_kv_cache、prefill_group_load 以及桶任务 task）
        proxy_state.release_prefiller(prefiller_idx, prefiller_score, task)
        proxy_state.release_prefiller_kv(prefiller_idx, prefiller_score)
        raise
    proxy_state.release_prefiller(prefiller_idx, prefiller_score, task)

    try:
        response_json = response.json()
        kv_transfer_params = response_json.get("kv_transfer_params", {})
        if kv_transfer_params:
            req_data["kv_transfer_params"] = kv_transfer_params
        # 选择 decoder
        decoder_score = proxy_state.calculate_decode_scores(request_length)
        logger.debug("Decoder score: %f", decoder_score)
        # 使用 prefiller 返回的 kv_transfer_params 来选择 decoder
        decoder_idx = proxy_state.select_decoder(decoder_score)
        decoder = proxy_state.decoders[decoder_idx]
        logger.debug("Using %s %s", prefiller.url, decoder.url)
        return InstanceInfo(
            request_id=request_id,
            prefiller_idx=prefiller_idx,
            prefiller_score=prefiller_score,
            prefiller=prefiller,
            decoder=decoder,
            decoder_idx=decoder_idx,
            decoder_score=decoder_score,
        )
    except Exception:
        proxy_state.release_prefiller_kv(prefiller_idx, prefiller_score)
        raise


@dataclass
class InstanceInfo:
    request_id: str
    prefiller_idx: int
    prefiller_score: float
    prefiller: ServerState
    decoder_idx: int
    decoder_score: float
    decoder: ServerState


async def _handle_completions(api: str, request: Request):
    # request_num 必须覆盖整个流式生命周期，而不仅仅是实例选择阶段。
    # streaming_started 用于区分"选择失败、流从未启动"与"流已启动并通过其自身的
    # finally 释放资源"这两条路径。
    streaming_started = False
    try:
        proxy_state.request_num += 1
        req_data = await request.json()
        req_body = await request.body()
        request_length = len(req_body)
        instance_info = await _handle_select_instance(api, req_data, request_length)
        stream_flag = bool(req_data.get("stream", False))
        chat_flag = "messages" in req_data

        if "prompt" in req_data:
            origin_prompt = req_data["prompt"]
        elif chat_flag:
            messages = req_data["messages"]
            origin_prompt = messages[0].get("content", "")
        else:
            origin_prompt = ""
        # 参考 vLLM sampling_params：max_tokens 的默认值
        origin_max_tokens = req_data.get("max_tokens", 16)

        async def generate_stream():
            nonlocal instance_info
            generated_token = ""
            released_kv = False
            retry_count = 0
            retry = True
            completion_tokens = 0
            # 每个 chunk 只 await 一次，循环内逻辑尽量精简
            try:
                while retry:
                    retry = False
                    # 每次 (prefiller, decoder) 重试都重置，确保新选中的 prefiller 的
                    # KV 在其首个 decode chunk 时被释放
                    released_kv = False
                    async for chunk in stream_service_response_with_retry(
                        instance_info.decoder.client,
                        api,
                        req_data,
                        request_id=instance_info.request_id,
                        max_retries=global_args.max_retries,
                        base_delay=global_args.retry_delay,
                    ):
                        if not released_kv and chunk:
                            proxy_state.release_prefiller_kv(instance_info.prefiller_idx, instance_info.prefiller_score)
                            released_kv = True
                        try:
                            chunk_str = chunk.decode("utf-8").strip()
                        except UnicodeDecodeError:
                            logger.debug(f"Skipping chunk: {chunk}")
                            yield chunk
                            continue
                        if not chunk_str:
                            continue
                        if chunk_str.startswith("data: "):
                            chunk_str = chunk_str[len("data: ") :]
                        try:
                            chunk_json = json.loads(chunk_str)
                        except json.JSONDecodeError:
                            # 如果 chunk 是 [done]，跳过它。
                            logger.debug(f"Skipping chunk: {chunk_str}")
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
                            else (completion_tokens + usage.get("completion_tokens"))
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
                            # 在选择新 decoder 之前，先释放即将丢弃的 decoder，否则其
                            # active_tokens 会泄漏（下方的 finally 只会释放最终的
                            # instance_info.decoder）
                            proxy_state.release_decoder(instance_info.decoder_idx, instance_info.decoder_score)
                            # 防止在选择失败时于 finally 中重复释放
                            instance_info.decoder_idx = len(proxy_state.decoders)
                            instance_info.decoder_score = 0
                            instance_info = await _handle_select_instance(api, req_data, tmp_request_length)
                            break
                        if retry_count > 0 and not stream_flag:
                            if chat_flag:
                                choice["message"]["content"] = generated_token
                            else:
                                choice["text"] = generated_token
                            chunk = json.dumps(chunk_json).encode("utf-8")
                        yield chunk
            except Exception as e:
                logger.error(
                    f"Error during streaming from decoder {instance_info.decoder.url}: {str(e)} "
                    f"the aborted request {instance_info.request_id} will be routing to the target "
                    "prefiller when new request is ready to dispatch to it"
                )
                proxy_state.abort_prefiller_request(instance_info.prefiller_idx, instance_info.request_id)
            finally:
                # 将 KV 释放移到这里，是为了也能覆盖客户端提前断连的情况
                # （CancelledError 属于 BaseException，不会被 `except Exception` 捕获）。
                # 一旦首个 decode chunk 已经触发过释放，released_kv 即为 True。
                if not released_kv:
                    proxy_state.release_prefiller_kv(instance_info.prefiller_idx, instance_info.prefiller_score)
                # 流式结束后，释放 tokens
                proxy_state.release_decoder(instance_info.decoder_idx, instance_info.decoder_score)
                # 请求只有在其流真正结束后才算完成；在此处递减，使 request_num 能反映在途的流（用作扩缩容的防护基准）。
                proxy_state.request_num -= 1

        # 根据是否流式，决定正确的 media type
        media_type = "text/event-stream; charset=utf-8" if stream_flag else "application/json"
        streaming_started = True
        return StreamingResponse(generate_stream(), media_type=media_type)
    except Exception as e:
        import traceback

        exc_info = sys.exc_info()
        print(f"Error occurred in disagg prefill proxy server - {api} endpoint")
        print(e)
        print("".join(traceback.format_exception(*exc_info)))
        raise
    finally:
        # 仅在流从未启动（例如实例选择失败）时在此处递减；成功路径由 generate_stream 的 finally 处理
        if not streaming_started:
            proxy_state.request_num -= 1


async def _handle_adjust_instances(adjust_mode: str, request: Request):
    try:
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

        if adjust_mode == "add":
            # "bucket" 指定新增 prefiller 加入哪个 prefill 桶
            # （"short" 或 "long"）。开启动态分桶时，对 prefiller 它是必填项；
            # 否则会被忽略（decode 类型，或动态分桶关闭时）。
            bucket = req_data.get("bucket")
            if instance_type == InstanceType.PREFILL and proxy_state.num_prefill_buckets > 1:
                if bucket not in BUCKET_NAME_TO_GROUP:
                    return {
                        "error": f"'bucket' is required for prefill instances when "
                        f"dynamic bucketing is enabled. Expected one of "
                        f"{list(BUCKET_NAME_TO_GROUP)} (got {bucket!r})."
                    }
            elif bucket is not None:
                if instance_type == InstanceType.DECODE:
                    logger.warning("'bucket' is ignored for decode instances.")
                else:
                    logger.warning("'bucket' is ignored because dynamic bucketing is off.")
                bucket = None
            added_nodes, waiting_nodes = await proxy_state.add_instances(instance_type, instances, bucket)
            if waiting_nodes:
                all_msg = (
                    f"{adjust_mode} {instance_type} instances: {added_nodes}. "
                    f"Instances {waiting_nodes} are waiting to be added."
                )
        elif adjust_mode == "remove":
            if instance_type == InstanceType.PREFILL:
                need_waiting = proxy_state.remove_prefillers(instances)
                current_pool = proxy_state.prefillers
            else:
                need_waiting = proxy_state.remove_decoders(instances)
                current_pool = proxy_state.decoders

            if need_waiting:
                all_msg = f"Instances {instances} are isolated and waiting to be removed."
            else:
                # 区分"真正移除"与"被拒绝（桶/池变空 或 实例仍忙）"：
                # 被拒绝的实例在调用结束后仍留在池中。
                refused = [s for s in instances if s in current_pool]
                if refused:
                    all_msg = (
                        f"Refused to remove {instance_type} instances "
                        f"{[str(s) for s in refused]}: still busy or their removal "
                        f"would empty a bucket/pool."
                    )
                else:
                    all_msg = f"Removed {instance_type} instances: {[str(s) for s in instances]}."
        return {
            "message": all_msg,
            "current_prefill_instances": [str(prefiller) for prefiller in proxy_state.prefillers],
            "current_decode_instances": [str(decoder) for decoder in proxy_state.decoders],
        }
    except Exception as e:
        logger.error(f"Failed to {adjust_mode} instances: {e}")
        raise e


def trans_instances(instances: list[str]) -> list[ServerState]:
    server_list = []
    for instance in instances:
        h, p = instance.split(":")
        server_list.append(ServerState(h, int(p)))
    return server_list


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


@app.post("/instances/add")
async def handle_add_instances(request: Request):
    return await _handle_adjust_instances("add", request)


@app.post("/instances/remove")
async def handle_remove_instances(request: Request):
    return await _handle_adjust_instances("remove", request)


if __name__ == "__main__":
    global global_args
    global_args = parse_args()
    import uvicorn

    uvicorn.run(app, host=global_args.host, port=global_args.port)
