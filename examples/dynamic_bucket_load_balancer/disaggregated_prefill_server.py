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
import copy
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
from typing import List, Optional

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from dynamic_bucket_load_balancer import DynamicBucketLoadBalancer, Task, Bucket, ServerInfo

try:
    from vllm.logger import init_logger

    logger = init_logger(__name__)
except ImportError:
    import logging

    logger = logging.getLogger(__name__)

# Add uvloop for faster event loop if available
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

# Bucket name -> group index. Only meaningful when dynamic bucketing is on, in
# which case the prefiller pool is split into two buckets: group 0 serves short
# requests ([0, threshold)) and group 1 serves long requests ([threshold, max)).
# Used to place an explicitly added prefiller into a chosen bucket.
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
        self.active_kv_cache = 0  # Only for prefiller
        self.active_requests = 0  # Number of active requests
        self.aborted_requests = set()  # Track aborted requests

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
        # Background tasks closing httpx clients of removed instances; kept so they
        # are not garbage-collected before completion.
        self._pending_closes: set = set()

        # 动态分桶负载均衡器
        self.bucket_load_balancer: Optional[DynamicBucketLoadBalancer] = None

        if global_args.enable_dynamic_bucket:
            self.num_prefill_buckets = 2 # 启用动态分桶时的分组数量
            self.prefill_group_threshold = global_args.prefill_group_threshold
            prefill_buckets = [(0, self.prefill_group_threshold),
                               (self.prefill_group_threshold, global_args.max_request_tokens)]

            self.bucket_load_balancer = DynamicBucketLoadBalancer(buckets=prefill_buckets)
        else:
            self.num_prefill_buckets = 1 # 默认不分组

        # Initialize priority queues for efficient server selection
        # Each entry is (priority_score, server_index, server_reference)
        # Lower priority score = higher priority (less loaded)
        prefiller_heap_items = [ServerHeapItem(0.0, i, server) for i, server in enumerate(self.prefillers)]
        # 对Server进行分组
        self.prefiller_heaps: List[List[ServerHeapItem]] = self._group_servers(prefiller_heap_items, self.num_prefill_buckets)
        self.server_idx_to_group_idx = {}
        self.prefill_group_load = {}
        # 堆化每一分组
        for idx, cur_heap in enumerate(self.prefiller_heaps):
            for server_item in cur_heap:
                self.server_idx_to_group_idx[server_item.server_idx] = idx
            heapq.heapify(cur_heap)
            self.prefill_group_load[idx] = 0

        # 记录动态分桶状态、分组数量及各分组完整实例信息
        logger.info(f"Dynamic bucket enabled: {global_args.enable_dynamic_bucket}, "
                    f"number of prefill groups: {len(self.prefiller_heaps)}")
        for group_idx, cur_heap in enumerate(self.prefiller_heaps):
            logger.info(f"Prefill Group {group_idx}: {cur_heap}")

        self.decoder_heap:List[ServerHeapItem] = [ServerHeapItem(0.0, i, server) for i, server in enumerate(self.decoders)]
        heapq.heapify(self.decoder_heap)

    @staticmethod
    def _group_servers(servers: List[ServerHeapItem], num_groups: int):
        """
        Group servers into num_groups groups.

        Args:
            servers (list): servers to be grouped.
            num_groups (int): num of groups.

        Returns:
            list[list]: grouped list of servers.

        Raises:
            ValueError: if num_groups <= 0.
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
    def _add_new_server_to_heaps(server_heaps: List[List[ServerHeapItem]], new_server_heap_item:ServerHeapItem):
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
        """Update the priority of a prefiller server in the heap."""
        server = self.prefillers[server_idx]
        # Priority based on active_tokens and active_kv_cache
        priority = server.active_tokens + server.active_kv_cache * 0.3
        # Remove old entry and add new one
        group_idx = self.server_idx_to_group_idx[server_idx]

        self.prefiller_heaps[group_idx] = [server_heap_item for server_heap_item in self.prefiller_heaps[group_idx] if server_heap_item.server_idx != server_idx]
        self.prefiller_heaps[group_idx].append(ServerHeapItem(priority, server_idx, server))
        heapq.heapify(self.prefiller_heaps[group_idx])

    def _update_decoder_priority(self, server_idx: int):
        """Update the priority of a decoder server in the heap."""
        server = self.decoders[server_idx]
        priority = server.active_tokens
        # Remove old entry and add new one
        self.decoder_heap = [server_heap_item for server_heap_item in self.decoder_heap if server_heap_item.server_idx != server_idx]
        self.decoder_heap.append(ServerHeapItem(priority, server_idx, server))
        heapq.heapify(self.decoder_heap)

    def _restore_prefiller_priorities(self, servers) -> None:
        """Recompute and re-apply heap priority for the given prefillers.

        Used to un-taint servers that a removal attempt could not evict, so they
        become selectable again instead of being stranded at TAINT_PRIORITY
        (fix: orphaned taint). Harmless for servers that were never tainted.
        """
        for idx, server in enumerate(self.prefillers):
            if server in servers:
                self._update_prefiller_priority(idx)

    def _restore_decoder_priorities(self, servers) -> None:
        """Recompute and re-apply heap priority for the given decoders (un-taint)."""
        for idx, server in enumerate(self.decoders):
            if server in servers:
                self._update_decoder_priority(idx)

    def _schedule_close(self, server: "ServerState") -> None:
        """Close a removed instance's httpx client without blocking selection.

        Fire-and-forget on the main loop (remove_* always runs there). The task
        reference is retained so it is not GC'd before aclose() finishes (fix:
        connection leak on instance removal). Only reached when request_num == 0,
        so the client has no in-flight request.
        """
        task = asyncio.create_task(server.client.aclose())
        self._pending_closes.add(task)
        task.add_done_callback(self._pending_closes.discard)

    def abort_prefiller_request(self, server_idx: int, request_id):  # Changed to synchronous
        """
        Mark a request as aborted. This will helps to release kv cache in
        prefiller node.
        """
        if server_idx >= len(self.prefillers):
            return
        self.prefillers[server_idx].aborted_requests.add(request_id)

    def acquire_aborted_prefiller_requests(self, server_idx: int):  # Changed to synchronous
        """
        Get the set of aborted requests and clear it.
        This is used to release kv cache in prefiller node.
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
        # Defensive: removal is refused if it would empty a bucket, but guard
        # against a cryptic IndexError if that invariant ever breaks.
        if not self.prefiller_heaps[group_idx]:
            raise RuntimeError(
                f"No prefiller servers available in bucket group {group_idx}")

        server_heap_item:ServerHeapItem = heapq.heappop(self.prefiller_heaps[group_idx])
        chosen_server_idx = server_heap_item.server_idx

        # Update the chosen server atomically
        self.prefillers[chosen_server_idx].active_tokens += token_count
        self.prefillers[chosen_server_idx].active_kv_cache += token_count
        self.prefill_group_load[group_idx] += token_count

        # Update priority and re-add to heap
        self._update_prefiller_priority(chosen_server_idx)

        return chosen_server_idx

    def release_prefiller(self, idx, token_count, task):
        if idx >= len(self.prefillers):
            raise ValueError(f'No prefiller servers with idx = {idx}')
        self.prefillers[idx].active_tokens -= token_count
        group_idx = self.server_idx_to_group_idx[idx]
        self.prefill_group_load[group_idx] -= token_count
        if global_args.enable_dynamic_bucket and task is not None:
            self.bucket_load_balancer.release_task(task.id)
        # Update priority queue after releasing
        self._update_prefiller_priority(idx)

    def release_prefiller_kv(self, idx, token_count):
        if idx >= len(self.prefillers):
            return
        if self.prefillers[idx].active_kv_cache > 0:
            self.prefillers[idx].active_kv_cache -= token_count
        # Update priority queue after releasing
        self._update_prefiller_priority(idx)

    def select_decoder(self, token_count):
        if not self.decoder_heap:
            raise RuntimeError("No decoder servers available")

        server_heap_item = heapq.heappop(self.decoder_heap)
        chosen_server_idx = server_heap_item.server_idx

        # Update the chosen server atomically
        self.decoders[chosen_server_idx].active_tokens += token_count

        # Update priority and re-add to heap
        self._update_decoder_priority(chosen_server_idx)

        return chosen_server_idx

    def release_decoder(self, idx, token_count):
        if idx >= len(self.decoders):
            return
        self.decoders[idx].active_tokens -= token_count
        # Update priority queue after releasing
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
        """
        Find Best group by request length and current load of groups
        """
        if global_args.enable_dynamic_bucket:
            group_idx, task = self.bucket_load_balancer.dispatch_single_task(req_id, request_tokens, prefill_score)
            return group_idx, task
        else:
            return 0, None

    async def add_instances(
        self,
        instance_type: str,
        instances: list[ServerState],
        bucket: Optional[str] = None,
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
                # Remember the requested bucket so a later retry (once the
                # instance is healthy) places it into the same bucket.
                self.node_listener.waiting_nodes[node] = (instance_type, server, 0, bucket)
                waiting_nodes.append(node)
        return added_nodes, waiting_nodes

    def add_prefillers(self, instances: list[ServerState], bucket: Optional[str] = None) -> None:
        # Resolve the target bucket for genuinely new prefillers. An explicit
        # "short"/"long" only applies under dynamic bucketing (two buckets); with
        # a single bucket every prefiller lives in group 0, so the request is
        # ignored with a warning. A prefiller that is merely being un-tainted
        # keeps its existing bucket (we never move servers across buckets on add,
        # mirroring the removal policy: a bucket reflects a server's capability).
        if bucket is not None and self.num_prefill_buckets > 1:
            target_group_idx = BUCKET_NAME_TO_GROUP.get(bucket)
            if target_group_idx is None:
                logger.warning(
                    f"Unsupported bucket '{bucket}'; falling back to automatic "
                    f"placement.")
        else:
            if bucket is not None and self.num_prefill_buckets <= 1:
                logger.warning(
                    f"Bucket '{bucket}' requested but dynamic bucketing is off; "
                    f"the instance is added to the single prefill bucket.")
            target_group_idx = None  # automatic placement via _add_new_server_to_heaps

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
                    self.prefiller_heaps, group_idx = self._add_new_server_to_heaps(self.prefiller_heaps, new_server_heap_item)
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
                # decoder_heap: [(priority_0, 0, server_0)] -> [(priority_0, 0, server_0), (0, 1, server_1)]
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

        # Each prefiller must stay in its assigned bucket: a bucket reflects
        # the request-length class its servers are meant to serve, so we must not
        # rebalance servers across buckets on removal. Instead, refuse any removal
        # that would empty a bucket (a request routed to an empty bucket would
        # otherwise IndexError in select_prefiller).
        for group_idx, heap in enumerate(self.prefiller_heaps):
            if all(item.server in instances_to_remove for item in heap):
                logger.warning(
                    f"Refusing removal: prefill bucket group {group_idx} would be "
                    f"left empty")
                # un-taint so the servers stay usable and are not stranded at
                # TAINT_PRIORITY after the caller clears the list.
                self._restore_prefiller_priorities(instances_to_remove)
                return False

        new_prefillers = []
        old_idx_to_new_idx = {}
        for idx, server in enumerate(self.prefillers):
            if server not in instances_to_remove:
                new_prefillers.append(server)
                old_idx_to_new_idx[idx] = len(new_prefillers) - 1
            else:
                if server.active_tokens!=0 or server.active_kv_cache!=0 or server.active_requests!=0:
                    logger.warning("Prefill server is not empty, please wait for all requests to be completed")
                    # same as above: un-taint before refusing.
                    self._restore_prefiller_priorities(instances_to_remove)
                    return False

        removed_servers = [s for s in self.prefillers if s in instances_to_remove]
        self.prefillers = new_prefillers

        # Rebuild heaps preserving each surviving server's bucket membership and
        # remapping its index into the new (compacted) prefiller list.
        new_prefiller_heaps: List[List[ServerHeapItem]] = []
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

        # Close httpx clients of the removed instances to avoid leaks.
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

        # Keep at least one decoder: select_decoder raises if the pool is empty.
        if all(server in instances_to_remove for server in self.decoders):
            logger.warning("Refusing removal: no decoder would be left after removal")
            # [A1] un-taint so the decoders stay selectable instead of being stranded.
            self._restore_decoder_priorities(instances_to_remove)
            return False

        # refuse to remove a decoder that still has in-flight requests, mirroring
        # remove_prefillers; otherwise the rebuild + reindex below invalidates in-flight
        # decoder_idx held by active streams (release_decoder would hit the wrong decoder).
        for server in self.decoders:
            if server in instances_to_remove and server.active_tokens != 0:
                logger.warning("Decode server is not empty, please wait for all requests to be completed")
                # un-taint so the decoder stays selectable instead of being
                # stranded at TAINT_PRIORITY after the caller clears the list.
                self._restore_decoder_priorities(instances_to_remove)
                return False
        removed_servers = [server for server in self.decoders if server in instances_to_remove]
        self.decoders = [server for server in self.decoders if server not in instances_to_remove]
        decoder_heap_copy:List[ServerHeapItem] = self.decoder_heap.copy()
        decoder_heap_copy.sort(key=lambda x: x.server_idx)  # sorted by key: decoder_idx
        decoder_heap = []
        idx = 0
        for server_heap_item in decoder_heap_copy:
            if server_heap_item.server not in instances_to_remove:
                decoder_heap.append(ServerHeapItem(server_heap_item.priority, idx, server_heap_item.server))
                idx += 1

        self.decoder_heap = decoder_heap
        heapq.heapify(self.decoder_heap)
        # Close httpx clients of the removed instances to avoid leaks.
        for server in removed_servers:
            self._schedule_close(server)

        self.print_status(f"Remove decoder instances: {instances}.")
        return False

    def _drain_tainted_instances(self) -> None:
        """Try to evict tainted instances once no requests are in flight.

        Runs on the main event loop (scheduled by NodeListener), so the
        request_num check is atomic w.r.t. request selection (fix: cross-thread
        TOCTOU on the drain guard). remove_* un-taints whatever it cannot evict,
        so clearing the tainted lists afterwards never strands a server.
        """
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

proxy_state: Optional[ProxyState] = None


class NodeListener:
    def __init__(self, proxy):
        self.proxy_state = proxy
        # value: (instance_type, server, check_times, requested_bucket)
        self.waiting_nodes: dict[str, tuple[str, Any, int, Optional[str]]] = {}
        # Capture the main event loop. The background thread never touches the
        # shared prefiller/decoder structures directly; it marshals the whole poll
        # cycle onto this loop instead. This keeps every mutation of shared state
        # (heaps, lists, request_num, waiting_nodes) single-threaded, so the
        # synchronous add/remove/select/release methods stay race-free without
        # locking, the request_num drain-guard becomes atomic w.r.t. selection,
        # and the readiness probe reuses each server's httpx client on the loop
        # that owns it (no cross-loop client usage).
        self.loop = asyncio.get_running_loop()
        self.listening_thread = threading.Thread(target=self._node_listener, daemon=True)
        self.listening_thread.start()

    def _node_listener(self) -> None:
        while True:
            try:
                # Run the entire cycle on the main loop and block until it finishes.
                future = asyncio.run_coroutine_threadsafe(self._cycle(), self.loop)
                future.result()
            except Exception as e:
                logger.error(f"Node listener cycle failed: {e}")
            time.sleep(global_args.waiting_retry_interval)

    async def _cycle(self) -> None:
        """One poll cycle; executed on the main event loop."""
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


def parse_args(args_list = None):
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
    parser.add_argument("--prefill-group-threshold",
                        type=int,
                        default=32 * 1024,
                        help="Threshold of prefill groups")
    parser.add_argument("--max-request-tokens",
                        type=int,
                        default=128 * 1024,
                        help="Max tokens of request")
    parser.add_argument("--enable-dynamic-bucket",
                        action="store_true",
                        default=False,
                        help="Enable dynamic bucket load Balancer")
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
                return  # Success, exit after streaming
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
            # If any chunk has been sent, do not retry, just log and drop
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
    # Select prefiller
    request_tokens = proxy_state.calculate_prefill_tokens(request_length)
    group_idx, task = proxy_state.select_prefill_group(request_id, request_tokens, prefiller_score)

    logger.debug(f'Selected group_idx: {group_idx}')

    try:
        prefiller_idx = proxy_state.select_prefiller(prefiller_score, group_idx)
    except Exception:
        if global_args.enable_dynamic_bucket and task is not None:
            proxy_state.bucket_load_balancer.release_task(task.id)
        raise

    if global_args.enable_dynamic_bucket and task is not None:
        task.server_info = ServerInfo(InstanceType.PREFILL,prefiller_idx)

    prefiller = proxy_state.prefillers[prefiller_idx]
    # Send request to prefiller
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
        # prefill failed: roll back what select_prefiller acquired (active_tokens,
        # active_kv_cache, prefill_group_load and the bucket task) before re-raising
        proxy_state.release_prefiller(prefiller_idx, prefiller_score,task)
        proxy_state.release_prefiller_kv(prefiller_idx, prefiller_score)
        raise
    proxy_state.release_prefiller(prefiller_idx, prefiller_score,task)

    try:
        response_json = response.json()
        kv_transfer_params = response_json.get("kv_transfer_params", {})
        if kv_transfer_params:
            req_data["kv_transfer_params"] = kv_transfer_params
        # Select decoder
        decoder_score = proxy_state.calculate_decode_scores(request_length)
        logger.debug("Decoder score: %f", decoder_score)
        # Use the prefiller's kv_transfer_params to select decoder
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
    # request_num must cover the whole streaming lifetime, not just instance
    # selection. streaming_started distinguishes the "selection failed, stream never
    # ran" path from the "stream ran and releases via its own finally" path.
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
        # refer to vLLM sampling_params: max_token default value
        origin_max_tokens = req_data.get("max_tokens", 16)

        async def generate_stream():
            nonlocal instance_info
            generated_token = ""
            released_kv = False
            retry_count = 0
            retry = True
            completion_tokens = 0
            # Only one await per chunk, minimal logic in loop
            try:
                while retry:
                    retry = False
                    # reset per (prefiller, decoder) attempt so the KV of the
                    # newly selected prefiller gets released on its first decode chunk
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
                            chunk_str = chunk_str[len("data: "):]
                        try:
                            chunk_json = json.loads(chunk_str)
                        except json.JSONDecodeError:
                            # if chunk is [done], skip it.
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
                            # release the decoder we are abandoning before picking a
                            # new one, otherwise its active_tokens leak (the finally below
                            # only releases the final instance_info.decoder)
                            proxy_state.release_decoder(instance_info.decoder_idx, instance_info.decoder_score)
                            # Prevent double-release in finally block if selection fails
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
                # KV release moved here so it also covers early client disconnect
                # (CancelledError is a BaseException and is not caught by `except Exception`).
                # released_kv is True once the first decode chunk already triggered release.
                if not released_kv:
                    proxy_state.release_prefiller_kv(instance_info.prefiller_idx, instance_info.prefiller_score)
                # After streaming done, release tokens
                proxy_state.release_decoder(instance_info.decoder_idx, instance_info.decoder_score)
                # the request is only truly done once its stream finishes; decrement
                # here so request_num reflects in-flight streams (used as a scaling guard rail)
                proxy_state.request_num -= 1

        # Determine the correct media type based on stream flag
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
        # only decrement here when the stream never started (e.g. instance
        # selection failed); the successful path is handled by generate_stream's finally
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
            # "bucket" selects which prefill bucket a newly added prefiller joins
            # ("short" or "long"). When dynamic bucketing is on it is REQUIRED for
            # prefillers; otherwise it is ignored (decode, or dynamic bucketing off).
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
                    logger.warning(
                        "'bucket' is ignored because dynamic bucketing is off.")
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
                # Distinguish actually-removed from refused (empty-bucket / busy
                # guards): refused instances are still in the pool after the call.
                refused = [s for s in instances if s in current_pool]
                if refused:
                    all_msg = (
                        f"Refused to remove {instance_type} instances "
                        f"{[str(s) for s in refused]}: still busy or their removal "
                        f"would empty a bucket/pool.")
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