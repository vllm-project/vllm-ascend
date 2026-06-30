# SPDX-License-Identifier: Apache-2.0
import copy
import hashlib
import math
import queue
import random
import struct
import threading
import time
from collections import OrderedDict, defaultdict, deque
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

import msgspec
import numpy as np
import numpy.typing as npt
import torch
import zmq
from mooncake.engine import TransferEngine  # type: ignore
from vllm import envs
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.utils import BlockIds
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorHandshakeMetadata,
    KVConnectorMetadata,
    KVConnectorRole,
    SupportsHMA,
)
from vllm.distributed.utils import get_pp_indices
from vllm.logger import logger
from vllm.utils.network_utils import get_ip, make_zmq_path, make_zmq_socket
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import KVCacheConfig, MambaSpec
from contextlib import contextmanager

from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_connector import (
    GroupPull,
    KVCacheSendingThread,
    MooncakeAgentMetadata,
    MooncakeConnectorMetadata,
    SizedDict,
    split_if_not_byte_contiguous,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_connector import (
    KVCacheRecvingThread as BaseKVCacheRecvingThread,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_connector import (
    MooncakeConnectorScheduler as BaseMooncakeConnectorScheduler,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_connector import (
    MooncakeConnectorWorker as BaseMooncakeConnectorWorker,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_connector import (
    group_concurrent_contiguous as base_group_concurrent_contiguous,
)
from vllm_ascend.distributed.kv_transfer.utils.mooncake_transfer_engine import global_te
from vllm_ascend.distributed.kv_transfer.utils.utils import (
    RegisterRegions,
    collect_storage_merged_register_regions,
    validate_register_region_count,
)

# isort: off
if TYPE_CHECKING:
    from vllm.v1.attention.backend import AttentionMetadata  # type: ignore
    from vllm.forward_context import ForwardContext
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.request import Request
# isort: on

GET_META_MSG = b"get_meta_msg"
DONE_RECVING_MSG = b"done_recving_msg"
READY_SCHEDULER = b"ready_scheduler"
STAGING_FULL = b"staging_full"
START_PULL = b"START_PULL"

# 2 MB huge-page granularity.  D2H (device-to-host) transfers on Ascend require
# the registered host memory size to be 2M-aligned so that UBMem uses 2M huge
# pages instead of falling back to 4K small pages.  aclrtMallocHost (used by
# torch pin_memory) already tries to back the allocation with 2M huge pages;
# pairing that with a 2M-aligned size lets UBMem operate in 2M-page mode.
HUGEPAGE_SIZE_2M = 2 * 1024 * 1024

StagingBlockKey = tuple[int, int, int]
LegacyStagingBlockKey = tuple[int, int]

# ZMQ ports for D2RH (hop1) and scheduler ready signaling (hop1 done).
# Layout matches side_channel_port + device_index used by KV handshake:
#   port = BASE + dp_rank * tp_size * pp_size * pcp_size + (pp_rank + pcp_rank) * tp_size + tp_rank
# TP=1 / DP0 / PP0 / PCP0 鈫?D2RH=8100, READY=8200 (same as legacy hardcoded values).
D2RH_ZMQ_PORT_BASE = 8100
SCHEDULER_READY_ZMQ_PORT_BASE = 8200


class KVCacheTaskTracker:
    def __init__(self):
        super().__init__()

        self.done_task_lock = threading.Lock()
        self.finished_requests: set[str] = set()
        # Only used in prefill node. Tracks requests whose kv blocks freeing is
        # intentionally delayed. Each entry is a tuple of (request_id,
        # timestamp). If a request remains in this queue for too long, it will
        # be force-freed.
        self.delayed_free_requests: OrderedDict[str, float] = OrderedDict()
        self.reqs_to_process: set[str] = set()

    def add_req_to_process(self, request_id: str):
        self.reqs_to_process.add(request_id)

    def add_not_transfer_request(self, request_id: str):
        with self.done_task_lock:
            self.finished_requests.add(request_id)
            self.reqs_to_process.discard(request_id)

    def update_done_task_count(self, request_id: str):
        with self.done_task_lock:
            if request_id in self.reqs_to_process:
                self.finished_requests.add(request_id)
                self.reqs_to_process.discard(request_id)
                self.delayed_free_requests.pop(request_id, None)
            else:
                logger.error(
                    "MooncakeConnector finish req not in reqs to process."
                    "If it is a P node, this request may have been force freed."
                )

    def get_and_clear_finished_requests(self) -> set[str]:
        """
        Get and clear the requests that have been completed.
        Returns:
            A set of request IDs that have been completed.
        """
        with self.done_task_lock:
            finished_requests = self.finished_requests.copy()
            expired_requests = self._retrieve_expired_requests()
            finished_requests.update(expired_requests)
            self.finished_requests.clear()
        return finished_requests

    def add_delayed_request(self, request_id: str, delay_start_time: float):
        """Add a delayed free request."""
        with self.done_task_lock:
            if request_id in self.reqs_to_process:
                self.delayed_free_requests[request_id] = delay_start_time

    def _retrieve_expired_requests(self):
        """Retrieve all expired delayed requests."""
        expired_requests: set[str] = set()
        # Free delayed requests if they exceed the timeout
        current_time = time.time()
        while self.delayed_free_requests:
            request_id = next(iter(self.delayed_free_requests))
            delay_start_time = self.delayed_free_requests[request_id]
            if current_time - delay_start_time > envs.VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT:
                self.delayed_free_requests.popitem(last=False)
                self.reqs_to_process.discard(request_id)
                expired_requests.add(request_id)
                logger.info("Force freed request: %s", request_id)
            else:
                break
        return expired_requests


class HostListeningThread(threading.Thread):
    def __init__(
        self,
        all_requests: set[str],
        decode_tp_size: int,
        scheduler_ready_port: int,
    ):
        super().__init__(daemon=True, name="HostListeningThread")
        self.port_send_num: dict[str, int] = {}
        self.all_requests = all_requests
        self.decode_tp_size = max(decode_tp_size, 1)
        self.scheduler_ready_port = scheduler_ready_port
        self.ready_count: dict[str, int] = defaultdict(int)

        self.task_tracker = KVCacheTaskTracker()
        self.ready_request = set()
        self.ready_lock = threading.Lock()
        self.host_ip = get_ip()
        self.encoder = msgspec.msgpack.Encoder()
        self.decoder = msgspec.msgpack.Decoder(MooncakeAgentMetadata)
        self.remote_sockets_lock = threading.Lock()
        self.remote_sockets: dict[  # type: ignore
            str, deque[zmq.Socket]
        ] = defaultdict(  # type: ignore
            deque
        )
        self.remote_poller = zmq.Poller()  # type: ignore
        self.timeout = 1.0  # seconds

    def get_and_clear_finished_requests(self) -> set[str]:
        """
        Get and clear the requests that have been completed.
        Returns:
            A set of request IDs that have been completed.
        """
        return self.task_tracker.get_and_clear_finished_requests()

    def run(self):
        """Run the thread to handle KV cache transfer requests."""
        try:
            # Listen for new requests for metadata. NOTE(rob): we need each rank
            # to have a unique port. This hack to keeps us moving. We will
            # switch when moving to etcd or where we have a single ZMQ socket in
            # the scheduler.
            handshake_port = self.scheduler_ready_port
            path = make_zmq_path("tcp", self.host_ip, handshake_port)
            logger.info("Starting scheduler ready listener on path: %s", path)
            with zmq_ctx(zmq.ROUTER, path) as sock:  # type: ignore
                self.run_busy_loop(sock)
        except Exception as e:
            logger.exception("Mooncake KVCacheSendingThread exception: %s", e)

    def run_busy_loop(self, sock: zmq.Socket):  # type: ignore
        # encoder = msgspec.msgpack.Encoder()
        # encoded_data = encoder.encode(self.metadata)
        # size_in_bytes = len(encoded_data)
        # logger.debug("Size of encoded MooncakeAgentMetadata: %s bytes", str(size_in_bytes))

        decoder = msgspec.msgpack.Decoder(type=tuple)
        while True:
            try:
                frames = sock.recv_multipart()
                if len(frames) < 2:
                    logger.error("Invalid message format: %s", frames)
                    continue

                identity = frames[0]
                payload = [f for f in frames[1:] if f != b""]
                if len(payload) != 1:
                    logger.error("Invalid message format: %s", frames)
                    continue

                msg = decoder.decode(payload[0])
                if msg[0] == READY_SCHEDULER:
                    logger.debug("Got READY_SCHEDULER for request %s", msg[1])
                    request_id = msg[1]
                    with self.ready_lock:
                        self.ready_count[request_id] += 1
                        if self.ready_count[request_id] >= self.decode_tp_size:
                            self.ready_request.add(request_id)
                            del self.ready_count[request_id]
                    # Acknowledge the request completion.
                    while True:
                        try:
                            # Send ACK to the sender.
                            sock.send_multipart((identity, b"", b"ACK"), flags=zmq.NOBLOCK)  # type: ignore
                            break
                        except zmq.Again:  # type: ignore
                            # If the socket is not ready, retry sending.
                            logger.debug("Socket not ready, retrying to send ACK for request %s", msg[1])
                            time.sleep(0.01)
                elif msg[0] == STAGING_FULL:
                    request_id = msg[1]
                    logger.info(
                        "CPU staging full for request %s, remove from all_requests to allow retry",
                        request_id,
                    )
                    with self.ready_lock:
                        self.all_requests.discard(request_id)
                        self.ready_count.pop(request_id, None)
                    while True:
                        try:
                            sock.send_multipart((identity, b"", b"ACK"), flags=zmq.NOBLOCK)  # type: ignore
                            break
                        except zmq.Again:  # type: ignore
                            logger.debug("Socket not ready, retrying to send ACK for STAGING_FULL %s", msg[1])
                            time.sleep(0.01)
                else:
                    logger.error("Connection listener got unexpected message %s", msg)
            except Exception as e:
                logger.error("Connection listener got exception %s: %s", type(e), e)


@contextmanager
def zmq_ctx(socket_type: Any, addr: str) -> Iterator[zmq.Socket]:  # type: ignore
    """Context manager for a ZMQ socket"""

    if socket_type not in (zmq.ROUTER, zmq.REQ, zmq.DEALER):  # type: ignore
        raise ValueError(f"Unexpected socket type: {socket_type}")

    ctx: zmq.Context | None = None  # type: ignore
    try:
        ctx = zmq.Context()  # type: ignore
        yield make_zmq_socket(ctx=ctx, path=addr, socket_type=socket_type, bind=socket_type == zmq.ROUTER)  # type: ignore
    finally:
        if ctx is not None:
            ctx.destroy(linger=0)


def group_concurrent_contiguous(
    src: list[int], dst: list[int]
) -> tuple[list[npt.NDArray[np.int64]], list[npt.NDArray[np.int64]]]:
    """Vectorised NumPy implementation."""
    src_indices: npt.NDArray[np.int64] = np.array(src, dtype=np.int64)
    dst_indices: npt.NDArray[np.int64] = np.array(dst, dtype=np.int64)

    if src_indices.size == 0:
        return [], []

    brk = np.where((np.diff(src_indices) != 1) | (np.diff(dst_indices) != 1))[0] + 1
    src_groups = np.split(src_indices, brk)
    dst_groups = np.split(dst_indices, brk)

    src_groups = [g.tolist() for g in src_groups]
    dst_groups = [g.tolist() for g in dst_groups]

    return src_groups, dst_groups


def string_to_int64_hash(input_str):
    """
    Hash the string using SHA-256 and convert it into an int64 integer.
    """
    hashed_bytes = hashlib.sha256(input_str.encode("utf-8")).digest()
    trunked_bytes = hashed_bytes[:8]
    uint64_value = struct.unpack("<Q", trunked_bytes)[0]
    return uint64_value


def ensure_zmq_send(
    socket: zmq.Socket,  # type: ignore
    data: bytes,
    path: str,
    max_retries: int = 3,
):
    retries_left = max_retries
    while True:
        try:
            socket.send(data)
            return
        except zmq.ZMQError as e:  # type: ignore
            retries_left -= 1
            if retries_left > 0:
                logger.warning("Send failed: %s, retrying... (%s attempts left)", e, retries_left)
                time.sleep(0.1)
            else:
                logger.error("Send failed after all retries: %s", e)
                raise RuntimeError(f"Failed to send data to {path} after {max_retries} retries: {e}")


def ensure_zmq_recv(
    socket: zmq.Socket,  # type: ignore
    poller: zmq.Poller,  # type: ignore
    path: str,
    timeout: float = 1.0,
    max_retries: int = 3,
) -> bytes:
    retries_left = max_retries
    while True:
        try:
            if dict(poller.poll(int(timeout * 1000))):  # milliseconds
                data = socket.recv()
                return data
            else:
                raise zmq.ZMQError("Receive timeout")  # type: ignore
        except zmq.ZMQError as e:  # type: ignore
            retries_left -= 1
            if retries_left > 0:
                logger.warning("Receive failed: %s, retrying... (%s attempts left)", e, retries_left)
                time.sleep(0.1)
            else:
                logger.error("Receive failed from %s after all retries: %s", path, e)
                raise RuntimeError(f"Failed to receive data after {max_retries} retries: {e}")


# decode node should know pp_partition_layer in prefill node,
# it is configured in kv_transfer_config by partition_list_str,
# default using vllm layer split algorithm.
def get_prefill_pp_indices(
    num_hidden_layers: int, pp_rank: int, pp_size: int, partition_list_str: str | None = None
) -> tuple[int, int]:
    if partition_list_str is None:
        return get_pp_indices(num_hidden_layers, pp_rank, pp_size)
    else:
        try:
            partitions = [int(layer) for layer in partition_list_str.split(",")]
        except ValueError as err:
            raise ValueError("Invalid partition string: {}".format(partition_list_str)) from err
        if len(partitions) != pp_size:
            raise ValueError(f"{len(partitions)=} does not match {pp_size=}.")
        if sum(partitions) != num_hidden_layers:
            raise ValueError(f"{sum(partitions)=} does not match {num_hidden_layers=}.")
        start_layer = sum(partitions[:pp_rank])
        end_layer = start_layer + partitions[pp_rank]
        return (start_layer, end_layer)


def get_parallel_device_index(
    tp_rank: int,
    tp_size: int,
    pp_rank: int = 0,
    pcp_rank: int = 0,
) -> int:
    return (pp_rank + pcp_rank) * tp_size + tp_rank


def get_dp_port_offset(
    dp_rank: int,
    tp_size: int,
    pp_size: int,
    pcp_size: int,
) -> int:
    return dp_rank * tp_size * pp_size * pcp_size


def get_d2rh_zmq_port(
    vllm_config: VllmConfig,
    tp_rank: int,
    pp_rank: int = 0,
    pcp_rank: int = 0,
) -> int:
    parallel_config = vllm_config.parallel_config
    tp_size = parallel_config.tensor_parallel_size
    pp_size = parallel_config.pipeline_parallel_size
    pcp_size = parallel_config.prefill_context_parallel_size
    dp_rank = parallel_config.data_parallel_rank
    device_index = get_parallel_device_index(tp_rank, tp_size, pp_rank, pcp_rank)
    dp_offset = get_dp_port_offset(dp_rank, tp_size, pp_size, pcp_size)
    return D2RH_ZMQ_PORT_BASE + dp_offset + device_index


def get_scheduler_ready_zmq_port(vllm_config: VllmConfig) -> int:
    parallel_config = vllm_config.parallel_config
    tp_size = parallel_config.tensor_parallel_size
    pp_size = parallel_config.pipeline_parallel_size
    pcp_size = parallel_config.prefill_context_parallel_size
    dp_rank = parallel_config.data_parallel_rank
    dp_offset = get_dp_port_offset(dp_rank, tp_size, pp_size, pcp_size)
    return SCHEDULER_READY_ZMQ_PORT_BASE + dp_offset


def compute_tp_num_need_pulls(
    num_key_value_heads: int,
    decode_tp_size: int,
    prefill_tp_size: int,
    is_deepseek_mla: bool,
) -> int:
    if is_deepseek_mla:
        return 1
    num_d_block_heads = max(1, num_key_value_heads // decode_tp_size)
    num_p_block_heads = max(1, num_key_value_heads // prefill_tp_size)
    return num_d_block_heads // num_p_block_heads


def get_remote_tp_ranks(
    tp_ori_data: np.ndarray,
    rand_group_index: list[int],
    num_groups: int,
    prefill_tp_size: int,
    decode_tp_size: int,
    num_key_value_heads: int,
    is_deepseek_mla: bool,
    use_sparse: bool,
) -> list[list[int]]:
    tp_num_need_pulls = compute_tp_num_need_pulls(num_key_value_heads, decode_tp_size, prefill_tp_size, is_deepseek_mla)
    tp_sampled_nums: list[list[int]] = []
    if prefill_tp_size > num_key_value_heads or is_deepseek_mla or use_sparse:
        tp_ori_data = tp_ori_data.reshape(-1, num_groups)
        chosen_group = tp_ori_data[:, rand_group_index]
        flattened = chosen_group.reshape(-1).tolist()
        tp_sampled_nums = [flattened[i : i + tp_num_need_pulls] for i in range(0, len(flattened), tp_num_need_pulls)]
    else:
        group_size = prefill_tp_size // decode_tp_size
        for i in range(decode_tp_size):
            slice_data = tp_ori_data[i * group_size : (i + 1) * group_size]
            tp_sampled_nums.append(slice_data.tolist())
    return tp_sampled_nums


def get_remote_ranks_for_req(
    req_id: str,
    prefill_tp_size: int,
    decode_tp_size: int,
    prefill_pp_size: int,
    num_key_value_heads: int,
    is_deepseek_mla: bool,
    use_sparse: bool,
) -> list[list[int]]:
    if prefill_tp_size == decode_tp_size:
        return [[tp + pp * prefill_tp_size for pp in range(prefill_pp_size)] for tp in range(prefill_tp_size)]

    if is_deepseek_mla or use_sparse:
        num_kv_head = 1
    else:
        num_kv_head = num_key_value_heads
    ori_data = np.arange(prefill_tp_size * prefill_pp_size)
    seed = string_to_int64_hash(req_id)
    rand = random.Random(seed)
    ori_data = ori_data.reshape(prefill_pp_size, -1)
    num_groups = max(1, len(ori_data[0]) // num_kv_head)
    rand_group_index = rand.sample(range(num_groups), max(decode_tp_size // num_kv_head, 1))
    all_results = [
        get_remote_tp_ranks(
            ori_data[pp_index],
            rand_group_index,
            num_groups,
            prefill_tp_size,
            decode_tp_size,
            num_key_value_heads,
            is_deepseek_mla,
            use_sparse,
        )
        for pp_index in range(prefill_pp_size)
    ]
    sampled_nums: list[list[int]] = []
    for group_index in range(len(all_results[0])):
        group: list[int] = []
        for pp_index in range(prefill_pp_size):
            group.extend(all_results[pp_index][group_index])
        sampled_nums.append(group)
    return sampled_nums


def resolve_remote_host_for_handshake_port(
    base_port: int,
    remote_handshake_port: int,
    remote_host: str,
    remote_engine_id: str,
    remote_multi_nodes_meta_mapping: dict[str, dict[str, Any]] | None,
) -> tuple[str, str]:
    rank = str(remote_handshake_port - base_port)
    if remote_multi_nodes_meta_mapping is None or remote_multi_nodes_meta_mapping.get(rank) is None:
        return remote_host, remote_engine_id
    info = remote_multi_nodes_meta_mapping[rank]
    return info.get("host", remote_host), info.get("engine_id", remote_engine_id)


CPU_STAGING_ENGINE_ID = "__d2rh_cpu_staging__"
CPU_STAGING_HANDSHAKE_PORT = -1


def _build_block_map(remote_block_ids: BlockIds, local_block_ids: BlockIds) -> dict[tuple[int, int], int]:
    return {
        (group_id, remote_block_id): local_block_id
        for group_id, (remote_group_block_ids, local_group_block_ids) in enumerate(
            zip(remote_block_ids, local_block_ids)
        )
        for remote_block_id, local_block_id in zip(remote_group_block_ids, local_group_block_ids)
    }


def _group_block_map_values(block_map: dict[tuple[int, int], int]) -> BlockIds:
    max_group_id = max((group_id for group_id, _ in block_map), default=-1)
    grouped_block_ids: list[list[int]] = [[] for _ in range(max_group_id + 1)]
    seen_block_ids: list[set[int]] = [set() for _ in range(max_group_id + 1)]
    for (group_id, _), block_id in block_map.items():
        if block_id in seen_block_ids[group_id]:
            continue
        seen_block_ids[group_id].add(block_id)
        grouped_block_ids[group_id].append(block_id)
    return tuple(grouped_block_ids)


def _get_group_pull_field(group_pull: GroupPull | dict[str, Any], field: str) -> Any:
    if isinstance(group_pull, dict):
        return group_pull[field]
    return getattr(group_pull, field)


class D2RHCPUCacheManager:
    def __init__(self, num_blocks: int):
        self.num_blocks = num_blocks
        self.free_queue = deque(range(num_blocks))
        self.used_set: set[int] = set()
        self.lock = threading.Lock()

    def alloc_block_map(self, remote_block_ids: BlockIds) -> dict[LegacyStagingBlockKey, int] | None:
        with self.lock:
            # Deduplicate (group_id, remote_block_id) to avoid over-allocation
            # when compressed/shared layouts contain repeated remote block ids.
            remote_keys: list[tuple[int, int]] = []
            seen_keys: set[tuple[int, int]] = set()
            for group_id, group_block_ids in enumerate(remote_block_ids):
                for remote_block_id in group_block_ids:
                    key = (group_id, remote_block_id)
                    if key in seen_keys:
                        continue
                    seen_keys.add(key)
                    remote_keys.append(key)

            num_blocks = len(remote_keys)
            if num_blocks > len(self.free_queue):
                logger.info(
                    "CPU staging does not have enough blocks, need=%d free=%d",
                    num_blocks,
                    len(self.free_queue),
                )
                return None

            block_map: dict[LegacyStagingBlockKey, int] = {}
            for key in remote_keys:
                local_block_id = self.free_queue.popleft()
                self.used_set.add(local_block_id)
                block_map[key] = local_block_id
            return block_map

    def alloc_sharded_block_map(
        self,
        remote_block_ids: BlockIds,
        group_pulls_by_port: list[list[GroupPull | dict[str, Any]]],
    ) -> dict[StagingBlockKey, int] | None:
        offsets_by_group: dict[int, set[int]] = defaultdict(set)
        for group_pulls in group_pulls_by_port:
            for group_pull in group_pulls:
                group_id = int(_get_group_pull_field(group_pull, "group_id"))
                remote_tp_offset = int(_get_group_pull_field(group_pull, "remote_tp_offset"))
                offsets_by_group[group_id].add(remote_tp_offset)

        with self.lock:
            remote_keys: list[StagingBlockKey] = []
            seen_keys: set[StagingBlockKey] = set()
            for group_id, group_block_ids in enumerate(remote_block_ids):
                offsets = offsets_by_group.get(group_id) or {0}
                for remote_block_id in group_block_ids:
                    for remote_tp_offset in sorted(offsets):
                        key = (group_id, remote_block_id, remote_tp_offset)
                        if key in seen_keys:
                            continue
                        seen_keys.add(key)
                        remote_keys.append(key)

            num_blocks = len(remote_keys)
            if num_blocks > len(self.free_queue):
                logger.info(
                    "CPU staging does not have enough blocks, need=%d free=%d",
                    num_blocks,
                    len(self.free_queue),
                )
                return None

            block_map: dict[StagingBlockKey, int] = {}
            for key in remote_keys:
                local_block_id = self.free_queue.popleft()
                self.used_set.add(local_block_id)
                block_map[key] = local_block_id
            logger.info(f"[===] after_allco{self.used_set=} {self.free_queue =}{block_map=}")
            return block_map

    def free_block_map(self, block_map: dict[tuple[int, ...], int]) -> None:
        with self.lock:
            for block_id in set(block_map.values()):
                if block_id in self.used_set:
                    self.used_set.remove(block_id)
                    self.free_queue.append(block_id)
            logger.info(f"[===] free {self.used_set=} {self.free_queue =}{block_map=}")


class D2RHThread(threading.Thread):
    def __init__(
        self,
        cpu_kv_caches_base_addr: list[list[int]],
        cpu_block_len_per_addr: list[list[int]],
        cpu_block_stride_per_addr: list[list[int]],
        cpu_block_size_scale: list[list[int]],
        kv_group2layeridx: dict[int, tuple[dict[str, Any], list[int]]],
        engine: TransferEngine,
        cpu_kvcache_manager: D2RHCPUCacheManager,
        remote_local_block_map: dict[str, dict[tuple[int, ...], int]],
        vllm_config: VllmConfig,
        d2rh_handshake_port: int,
        scheduler_ready_port: int,
        tp_rank: int = 0,
    ):
        super().__init__(daemon=True, name=f"D2RHThread-TP{tp_rank}")
        self.cpu_kv_caches_base_addr = cpu_kv_caches_base_addr
        self.cpu_block_len_per_addr = cpu_block_len_per_addr
        self.cpu_block_stride_per_addr = cpu_block_stride_per_addr
        self.cpu_block_size_scale = cpu_block_size_scale
        self.kv_group2layeridx = kv_group2layeridx
        self.group_compress_ratios: dict[int, int] = {}
        for group_id, (group_spec, _) in self.kv_group2layeridx.items():
            compress_ratio = 1
            kv_cache_spec = group_spec.get("kv_cache_spec")
            if isinstance(kv_cache_spec, dict):
                for spec in kv_cache_spec.values():
                    if isinstance(spec, dict) and isinstance(spec.get("compress_ratio"), int):
                        compress_ratio = max(1, spec["compress_ratio"])
                        break
            self.group_compress_ratios[group_id] = compress_ratio

        self.kv_caches_base_addr: dict[str, dict[int, list[list[int]]]] = SizedDict()
        self.remote_te_port: dict[str, dict[int, int]] = SizedDict()
        self.remote_block_size_scale: dict[str, dict[int, list[list[int]]]] = SizedDict()
        self.remote_block_stride_per_addr: dict[str, dict[int, list[list[int]]]] = SizedDict()
        self.remote_kv_group2layeridx: dict[str, dict[int, dict[int, tuple[dict[str, Any], list[int]]]]] = SizedDict()

        self.request_queue: queue.Queue[Any] = queue.Queue()
        self.remote_sockets_lock = threading.Lock()
        self.remote_sockets: dict[str, deque[zmq.Socket]] = defaultdict(deque)  # type: ignore[type-arg]
        self.remote_poller = zmq.Poller()  # type: ignore
        self.timeout = 1.0
        self.encoder = msgspec.msgpack.Encoder()
        self.decoder = msgspec.msgpack.Decoder(MooncakeAgentMetadata)
        self.engine = engine
        self.host_ip = get_ip()
        self.cpu_kvcache_manager = cpu_kvcache_manager
        self.remote_local_block_map = remote_local_block_map
        self.vllm_config = vllm_config
        self.block_size = vllm_config.cache_config.block_size
        self.d2rh_handshake_port = d2rh_handshake_port
        self.scheduler_ready_port = scheduler_ready_port
        self.tp_rank = tp_rank

    def add_request(self, **kwargs: Any) -> None:
        self.request_queue.put(kwargs)

    def run(self) -> None:
        try:
            path = make_zmq_path("tcp", self.host_ip, self.d2rh_handshake_port)

            def zmq_listener_worker() -> None:
                try:
                    with zmq_ctx(zmq.ROUTER, path) as sock:  # type: ignore
                        self.run_busy_loop(sock)
                except Exception as e:
                    logger.exception("D2RH ZMQ listener crashed: %s", e)

            threading.Thread(
                target=zmq_listener_worker,
                name=f"D2RH-ZMQListener-TP{self.tp_rank}",
                daemon=True,
            ).start()
        except Exception as e:
            logger.exception("Failed to initialize D2RH listener: %s", e)
            return

        while True:
            request_data = self.request_queue.get()
            try:
                if request_data is not None:
                    self._handle_request(request_data)
            except Exception as e:
                logger.exception("D2RH request processing failed: %s", e)
            finally:
                self.request_queue.task_done()

    def run_busy_loop(self, sock: zmq.Socket) -> None:  # type: ignore[type-arg]
        decoder = msgspec.msgpack.Decoder(type=tuple)
        while True:
            try:
                frames = sock.recv_multipart()
                if len(frames) < 2:
                    logger.error("Invalid D2RH message format: %s", frames)
                    continue
                identity = frames[0]
                payload = [f for f in frames[1:] if f != b""]
                if len(payload) != 1:
                    logger.error("Invalid D2RH message payload: %s", frames)
                    continue

                msg = decoder.decode(payload[0])
                pull_ack = b"ACK"
                if msg[0] == START_PULL:
                    request_id = msg[1]
                    params: dict[str, Any] | None = None
                    try:
                        params = msg[2]
                        remote_block_ids: BlockIds = tuple(params.get("remote_block_ids") or ())
                        block_map = self.cpu_kvcache_manager.alloc_sharded_block_map(
                            remote_block_ids,
                            params["group_pulls_by_port"],
                        )
                        if block_map is None:
                            pull_ack = STAGING_FULL
                        else:
                            remote_request_id = params.get("remote_request_id", request_id)
                            self.remote_local_block_map[request_id] = block_map
                            self.remote_local_block_map[remote_request_id] = block_map
                            self.add_request(
                                request_id=request_id,
                                remote_request_id=remote_request_id,
                                remote_host=params["remote_host"],
                                remote_engine_id=params["remote_engine_id"],
                                remote_port=params["remote_port"],
                                remote_multi_nodes_meta_mapping=params.get("remote_multi_nodes_meta_mapping"),
                                remote_block_ids=remote_block_ids,
                                remote_handshake_ports=params["remote_handshake_ports"],
                                group_pulls_by_port=params["group_pulls_by_port"],
                                remote_port_send_num=params.get("remote_port_send_num"),
                                num_computed_tokens=params.get("num_computed_tokens", 0),
                            )
                    except Exception as e:
                        # Release any partially created mapping to prevent CPU
                        # staging leaks on handshake/queueing failures.
                        remote_request_id = params.get("remote_request_id", request_id) if params is not None else request_id
                        block_map = self.remote_local_block_map.pop(remote_request_id, None)
                        self.remote_local_block_map.pop(request_id, None)
                        if block_map:
                            self.cpu_kvcache_manager.free_block_map(block_map)
                        logger.exception("Failed to handle D2RH START_PULL for request %s: %s", request_id, e)
                        pull_ack = STAGING_FULL
                else:
                    logger.error("D2RH listener got unexpected message %s", msg)
                    pull_ack = STAGING_FULL

                while True:
                    try:
                        sock.send_multipart((identity, b"", pull_ack), flags=zmq.NOBLOCK)  # type: ignore
                        break
                    except zmq.Again:  # type: ignore
                        time.sleep(0.01)
            except Exception as e:
                logger.error("D2RH listener exception %s: %s", type(e), e)

    def _handle_request(self, req_meta: dict[str, Any]) -> None:
        request_id = req_meta["request_id"]
        remote_request_id = req_meta["remote_request_id"]
        try:
            self._transfer_kv_cache_all_groups(req_meta)
            self.send_pull_done(request_id)
        except Exception:
            # Ensure staged CPU blocks are reclaimed if hop1 transfer fails.
            block_map = self.remote_local_block_map.pop(remote_request_id, None)
            self.remote_local_block_map.pop(request_id, None)
            if block_map:
                self.cpu_kvcache_manager.free_block_map(block_map)
            raise

    def _transfer_kv_cache_all_groups(self, req_meta: dict[str, Any]) -> None:
        remote_request_id = req_meta["remote_request_id"]
        remote_engine_id = req_meta["remote_engine_id"]
        remote_host = req_meta["remote_host"]
        remote_port = req_meta["remote_port"]
        remote_multi_nodes_meta_mapping = req_meta.get("remote_multi_nodes_meta_mapping")
        remote_block_ids: BlockIds = req_meta["remote_block_ids"]
        remote_handshake_ports: list[int] = req_meta["remote_handshake_ports"]
        group_pulls_by_port: list[list[GroupPull]] = req_meta["group_pulls_by_port"]
        remote_port_send_num = req_meta.get("remote_port_send_num")
        block_map = self.remote_local_block_map[remote_request_id]
        logger.info("[D2RH Thread] block_map: %s", block_map)

        if not any(remote_block_ids):
            return

        for remote_handshake_port, group_pulls in zip(remote_handshake_ports, group_pulls_by_port):
            port_host, port_engine_id = resolve_remote_host_for_handshake_port(
                remote_port,
                remote_handshake_port,
                remote_host,
                remote_engine_id,
                remote_multi_nodes_meta_mapping,
            )
            if (
                port_engine_id not in self.kv_caches_base_addr
                or remote_handshake_port not in self.kv_caches_base_addr[port_engine_id]
            ):
                self._get_remote_metadata(port_host, remote_handshake_port)

            remote_base_addrs = self.kv_caches_base_addr[port_engine_id][remote_handshake_port]
            remote_block_size_scale = self.remote_block_size_scale[port_engine_id][remote_handshake_port]
            remote_block_stride_per_addr = self.remote_block_stride_per_addr[port_engine_id][remote_handshake_port]
            session_id = f"{port_host}:{self.remote_te_port[port_engine_id][remote_handshake_port]}"
            src_list: list[int] = []
            dst_list: list[int] = []
            length_list: list[int] = []

            def expand_block_ids(block_ids: list[int], scale: int) -> list[int]:
                return [bid * scale + offset for bid in block_ids for offset in range(scale)]

            for group_pull in group_pulls:
                group_id = _get_group_pull_field(group_pull, "group_id")
                group_spec, layer_indices = self.kv_group2layeridx[group_id]
                num_group_pulls = _get_group_pull_field(group_pull, "num_group_pulls")
                remote_tp_offset = _get_group_pull_field(group_pull, "remote_tp_offset")
                local_group_block_ids: list[int] = []
                for bid in remote_block_ids[group_id]:
                    key_with_offset = (group_id, bid, remote_tp_offset)
                    key_legacy = (group_id, bid)
                    if key_with_offset in block_map:
                        local_group_block_ids.append(block_map[key_with_offset])
                    elif key_legacy in block_map:
                        local_group_block_ids.append(block_map[key_legacy])
                    else:
                        raise RuntimeError(
                            f"CPU staging block map missing key {(group_id, bid)} for request {remote_request_id}."
                        )
                remote_group_block_ids = list(remote_block_ids[group_id])
                if not local_group_block_ids:
                    continue

                is_mamba_group = group_spec["kv_cache_spec_type"] == "MambaSpec"
                if is_mamba_group:
                    if len(local_group_block_ids) != len(remote_group_block_ids):
                        raise RuntimeError("For MambaSpec num block should equal on P node and CPU staging.")
                    grouped_remote_block_ids = [[remote_group_block_ids[-1]]]
                    grouped_local_block_ids = [[local_group_block_ids[0]]]
                else:
                    local_scale = self.cpu_block_size_scale[layer_indices[0]][0]
                    remote_scale = remote_block_size_scale[layer_indices[0]][0]
                    kernel_local_block_ids = expand_block_ids(local_group_block_ids, local_scale)
                    kernel_remote_block_ids = expand_block_ids(remote_group_block_ids, remote_scale)
                    remote_kernel_block_size = self.block_size // remote_scale
                    remote_kernel_token_size = remote_kernel_block_size * self.group_compress_ratios[group_id]
                    remote_start_idx = req_meta.get("num_computed_tokens", 0) // remote_kernel_token_size
                    kernel_remote_block_ids = kernel_remote_block_ids[remote_start_idx:]
                    num_kernel_blocks = min(len(kernel_remote_block_ids), len(kernel_local_block_ids))
                    kernel_remote_block_ids = kernel_remote_block_ids[:num_kernel_blocks]
                    kernel_local_block_ids = kernel_local_block_ids[:num_kernel_blocks]
                    if num_group_pulls == 1:
                        grouped_remote_block_ids, grouped_local_block_ids = base_group_concurrent_contiguous(
                            kernel_remote_block_ids, kernel_local_block_ids
                        )
                    else:
                        grouped_remote_block_ids = [[block_id] for block_id in kernel_remote_block_ids]
                        grouped_local_block_ids = [[block_id] for block_id in kernel_local_block_ids]

                for layer_idx in layer_indices:
                    for cache_idx in range(len(self.cpu_kv_caches_base_addr[layer_idx])):
                        src_layer_base_addr = self.cpu_kv_caches_base_addr[layer_idx][cache_idx]
                        dst_layer_base_addr = remote_base_addrs[layer_idx][cache_idx]
                        block_len = self.cpu_block_len_per_addr[layer_idx][cache_idx]
                        block_stride = self.cpu_block_stride_per_addr[layer_idx][cache_idx]
                        remote_block_stride = remote_block_stride_per_addr[layer_idx][cache_idx]
                        inner_block_len = block_len // num_group_pulls
                        transfer_remote_block_ids, transfer_local_block_ids = split_if_not_byte_contiguous(
                            grouped_remote_block_ids,
                            grouped_local_block_ids,
                            src_block_stride=remote_block_stride,
                            dst_block_stride=block_stride,
                            block_len=inner_block_len,
                        )
                        for remote_block_id, local_block_id in zip(transfer_remote_block_ids, transfer_local_block_ids):
                            src_list.append(
                                src_layer_base_addr
                                + local_block_id[0] * block_stride
                            )
                            dst_list.append(dst_layer_base_addr + remote_block_id[0] * remote_block_stride)
                            length_list.append(inner_block_len * len(local_block_id))

            if src_list:
                # 统计 batch_transfer_sync_read 调用耗时
                _bt_start = time.perf_counter()
                ret = self.engine.batch_transfer_sync_read(session_id, src_list, dst_list, length_list)
                _bt_end = time.perf_counter()
                logger.info(
                    "[batch_transfer_sync_read] 耗时: %.6f s, src_list 长度: %d, "
                    "总传输字节数: %d",
                    _bt_end - _bt_start, len(src_list), sum(length_list))
                if ret < 0:
                    raise RuntimeError(f"D2RH hop1 transfer failed, ret: {ret}")
            self._send_done_recv_signal(
                remote_request_id,
                port_host,
                remote_handshake_port,
                remote_port_send_num,
            )

    def _get_remote_metadata(self, remote_host: str, remote_handshake_port: int) -> None:
        sock: zmq.Socket | None = None  # type: ignore[type-arg]
        try:
            sock = self._get_remote_socket(remote_host, remote_handshake_port)
            ensure_zmq_send(sock, self.encoder.encode((GET_META_MSG, "")), f"{remote_host}:{remote_handshake_port}")
            metadata_bytes = ensure_zmq_recv(sock, self.remote_poller, f"{remote_host}:{remote_handshake_port}")
            agent_meta = self.decoder.decode(metadata_bytes)
            self.kv_caches_base_addr[agent_meta.engine_id][remote_handshake_port] = agent_meta.kv_caches_base_addr
            self.remote_te_port[agent_meta.engine_id][remote_handshake_port] = agent_meta.te_rpc_port
            self.remote_block_size_scale[agent_meta.engine_id][remote_handshake_port] = agent_meta.block_size_scale
            self.remote_block_stride_per_addr[agent_meta.engine_id][remote_handshake_port] = agent_meta.block_strides
            self.remote_kv_group2layeridx[agent_meta.engine_id][remote_handshake_port] = agent_meta.kv_group2layeridx
        finally:
            if sock is not None:
                self._return_remote_socket(sock, remote_host, remote_handshake_port)

    def _send_done_recv_signal(
        self,
        request_id: str,
        remote_host: str,
        remote_handshake_port: int,
        remote_port_send_num: dict[int, Any] | None = None,
    ) -> None:
        sock: zmq.Socket | None = None  # type: ignore[type-arg]
        try:
            sock = self._get_remote_socket(remote_host, remote_handshake_port)
            remote_path = f"{remote_host}:{remote_handshake_port}"
            ensure_zmq_send(
                sock,
                self.encoder.encode((DONE_RECVING_MSG, request_id, remote_port_send_num or {})),
                remote_path,
            )
            resp = ensure_zmq_recv(sock, self.remote_poller, remote_path, timeout=self.timeout)
            if resp != b"ACK":
                raise RuntimeError(f"Failed to receive ACK, resp: {resp.decode('utf-8')}")
        finally:
            if sock is not None:
                self._return_remote_socket(sock, remote_host, remote_handshake_port)

    def send_pull_done(self, request_id: str) -> None:
        sock: zmq.Socket | None = None  # type: ignore[type-arg]
        try:
            sock = self._get_remote_socket(self.host_ip, self.scheduler_ready_port)
            scheduler_path = f"{self.host_ip}:{self.scheduler_ready_port}"
            ensure_zmq_send(sock, self.encoder.encode((READY_SCHEDULER, request_id)), scheduler_path)
            resp = ensure_zmq_recv(sock, self.remote_poller, scheduler_path, timeout=self.timeout)
            if resp != b"ACK":
                raise RuntimeError(f"Failed to receive ACK, resp: {resp.decode('utf-8')}")
        finally:
            if sock is not None:
                self._return_remote_socket(sock, self.host_ip, self.scheduler_ready_port)

    def _get_remote_socket(self, remote_host: str, remote_handshake_port: int) -> zmq.Socket:  # type: ignore[type-arg]
        remote_path = make_zmq_path("tcp", remote_host, remote_handshake_port)
        with self.remote_sockets_lock:
            if self.remote_sockets[remote_path]:
                return self.remote_sockets[remote_path].popleft()
            ctx = zmq.Context()  # type: ignore
            sock = make_zmq_socket(ctx=ctx, path=remote_path, socket_type=zmq.REQ, bind=False)  # type: ignore
            sock.setsockopt(zmq.SNDTIMEO, int(self.timeout * 1000))  # type: ignore
            self.remote_poller.register(sock, zmq.POLLIN)  # type: ignore
            return sock

    def _return_remote_socket(self, sock: zmq.Socket, remote_host: str, remote_handshake_port: int) -> None:  # type: ignore[type-arg]
        remote_path = make_zmq_path("tcp", remote_host, remote_handshake_port)
        with self.remote_sockets_lock:
            self.remote_sockets[remote_path].append(sock)


class KVCacheRecvingThread(BaseKVCacheRecvingThread):
    def __init__(
        self,
        *args: Any,
        cpu_kv_caches_base_addr: list[list[int]],
        cpu_block_len_per_addr: list[list[int]],
        cpu_block_stride_per_addr: list[list[int]],
        cpu_block_size_scale: list[list[int]],
        cpu_te_rpc_port: int,
        cpu_kvcache_manager: D2RHCPUCacheManager,
        remote_local_block_map: dict[str, dict[tuple[int, ...], int]],
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self.cpu_kv_caches_base_addr = cpu_kv_caches_base_addr
        self.cpu_block_len_per_addr = cpu_block_len_per_addr
        self.cpu_block_stride_per_addr = cpu_block_stride_per_addr
        self.cpu_block_size_scale = cpu_block_size_scale
        self.cpu_te_rpc_port = cpu_te_rpc_port
        self.cpu_kvcache_manager = cpu_kvcache_manager
        self.remote_local_block_map = remote_local_block_map
        self.cpu_host = get_ip()
        self.kv_caches_base_addr[CPU_STAGING_ENGINE_ID][CPU_STAGING_HANDSHAKE_PORT] = cpu_kv_caches_base_addr
        self.remote_te_port[CPU_STAGING_ENGINE_ID][CPU_STAGING_HANDSHAKE_PORT] = cpu_te_rpc_port
        self.remote_block_size_scale[CPU_STAGING_ENGINE_ID][CPU_STAGING_HANDSHAKE_PORT] = cpu_block_size_scale
        self.remote_block_stride_per_addr[CPU_STAGING_ENGINE_ID][CPU_STAGING_HANDSHAKE_PORT] = cpu_block_stride_per_addr
        self.remote_kv_group2layeridx[CPU_STAGING_ENGINE_ID][CPU_STAGING_HANDSHAKE_PORT] = self.kv_group2layeridx

    def _transfer_kv_cache_all_groups(self, req_meta: dict[str, Any]) -> None:
        request_id = req_meta["request_id"]
        remote_request_id = req_meta["remote_request_id"]
        block_map = self.remote_local_block_map.get(remote_request_id) or self.remote_local_block_map.get(request_id)
        if block_map is None:
            raise RuntimeError(f"CPU staging block map missing for request {remote_request_id}.")
        logger.info("[H2D Thread] block_map: %s", block_map)

        remote_block_ids: BlockIds = req_meta["remote_block_ids"]
        group_pulls: list[GroupPull] = req_meta.get("group_pulls", [])
        offset_by_group: dict[int, int] = {group_pull.group_id: group_pull.remote_tp_offset for group_pull in group_pulls}
        cpu_remote_block_ids_groups: list[list[int]] = []
        for group_id, group_block_ids in enumerate(remote_block_ids):
            group_remote_tp_offset = offset_by_group.get(group_id, 0)
            mapped_group_block_ids: list[int] = []
            for block_id in group_block_ids:
                key_with_offset = (group_id, block_id, group_remote_tp_offset)
                key_legacy = (group_id, block_id)
                if key_with_offset in block_map:
                    mapped_group_block_ids.append(block_map[key_with_offset])
                elif key_legacy in block_map:
                    mapped_group_block_ids.append(block_map[key_legacy])
                else:
                    raise RuntimeError(
                        f"CPU staging block map missing key {(group_id, block_id)} for request {remote_request_id}."
                    )
            cpu_remote_block_ids_groups.append(mapped_group_block_ids)
        cpu_remote_block_ids: BlockIds = tuple(cpu_remote_block_ids_groups)
        cpu_req_meta = dict(req_meta)
        cpu_req_meta["remote_block_ids"] = cpu_remote_block_ids
        cpu_req_meta["remote_engine_id"] = CPU_STAGING_ENGINE_ID
        cpu_req_meta["remote_host"] = self.cpu_host
        cpu_req_meta["remote_handshake_port"] = CPU_STAGING_HANDSHAKE_PORT
        super()._transfer_kv_cache_all_groups(cpu_req_meta)

    def _handle_request(self, req_meta: dict[str, Any]) -> None:
        try:
            super()._handle_request(req_meta)
        finally:
            if req_meta.get("all_task_done"):
                request_id = req_meta["request_id"]
                remote_request_id = req_meta["remote_request_id"]
                block_map = self.remote_local_block_map.pop(remote_request_id, None)
                self.remote_local_block_map.pop(request_id, None)
                if block_map:
                    self.cpu_kvcache_manager.free_block_map(block_map)


class MooncakeConnectorScheduler(BaseMooncakeConnectorScheduler):
    def __init__(self, vllm_config: VllmConfig, engine_id: str, kv_cache_config: KVCacheConfig):
        super().__init__(vllm_config, engine_id, kv_cache_config)
        self.kv_role = vllm_config.kv_transfer_config.kv_role
        prefill_parallel_config: dict[str, Any] = vllm_config.kv_transfer_config.get_from_extra_config("prefill", {})
        decode_parallel_config: dict[str, Any] = vllm_config.kv_transfer_config.get_from_extra_config("decode", {})
        self._prefill_tp_size = prefill_parallel_config["tp_size"]
        self._prefill_pp_size = prefill_parallel_config.get("pp_size", 1)
        self._decode_tp_size = decode_parallel_config["tp_size"]
        self.num_key_value_heads = vllm_config.model_config.hf_text_config.num_key_value_heads
        self.is_deepseek_mla = vllm_config.model_config.is_deepseek_mla
        self.use_sparse = False
        self.encoder = msgspec.msgpack.Encoder()
        self.decoder = msgspec.msgpack.Decoder(MooncakeAgentMetadata)
        self.remote_sockets: dict[str, deque[zmq.Socket]] = defaultdict(deque)  # type: ignore[type-arg]
        self.remote_poller = zmq.Poller()  # type: ignore
        self.timeout = 1.0
        self.remote_sockets_lock = threading.Lock()
        self.local_host = get_ip()
        if self.kv_role == "kv_consumer":
            self.all_requests: set[str] = set()
            self.listeningthread = HostListeningThread(
                self.all_requests,
                self._decode_tp_size,
                get_scheduler_ready_zmq_port(vllm_config),
            )
            self.listeningthread.start()

    def _get_remote_ranks_for_req(self, req_id: str, prefill_tp_size: int | None = None) -> list[list[int]]:
        if prefill_tp_size is None:
            prefill_tp_size = self._prefill_tp_size
        return get_remote_ranks_for_req(
            req_id,
            prefill_tp_size,
            self._decode_tp_size,
            self._prefill_pp_size,
            self.num_key_value_heads,
            self.is_deepseek_mla,
            self.use_sparse,
        )

    def _get_remote_rank(self, req_id: str, prefill_tp_size: int | None = None) -> list[int]:
        return self._get_remote_ranks_for_req(req_id, prefill_tp_size)[0]

    def _get_attention_group_num_need_pulls(self, group_spec: dict[str, Any], prefill_tp_size: int) -> int:
        kv_cache_spec = group_spec.get("kv_cache_spec", {})
        num_key_value_heads = self.num_key_value_heads
        if isinstance(kv_cache_spec, dict):
            for key in ("num_kv_heads", "num_key_value_heads"):
                if isinstance(kv_cache_spec.get(key), int):
                    num_key_value_heads = kv_cache_spec[key]
                    break
        num_d_block_heads = max(1, num_key_value_heads // self.tp_size)
        num_p_block_heads = max(1, num_key_value_heads // prefill_tp_size)
        return num_d_block_heads // num_p_block_heads

    def _build_group_pulls_by_port(
        self,
        remote_handshake_ports: list[int],
        remote_base_port: int,
        prefill_tp_size: int,
        decode_tp_rank: int,
    ) -> list[list[GroupPull]]:
        pulls_by_port: list[list[GroupPull]] = []
        for rank_idx, port in enumerate(remote_handshake_ports):
            remote_rank = (port - remote_base_port) % (prefill_tp_size * self._prefill_pp_size)
            prefill_pp_rank = remote_rank // prefill_tp_size
            group_pulls: list[GroupPull] = []
            for group_id, group in enumerate(self.kv_cache_groups):
                if isinstance(group.kv_cache_spec, MambaSpec):
                    num_group_pulls = max(1, prefill_tp_size // self._decode_tp_size)
                else:
                    group_spec = self._serialize_group_for_scheduler(group)
                    num_group_pulls = self._get_attention_group_num_need_pulls(group_spec, prefill_tp_size)
                if len(remote_handshake_ports) % num_group_pulls != 0:
                    raise RuntimeError(
                        "Invalid remote handshake ports and group pulls mapping: "
                        f"len(remote_handshake_ports)={len(remote_handshake_ports)}, "
                        f"num_group_pulls={num_group_pulls}"
                    )
                remote_tp_offset = rank_idx % num_group_pulls
                group_pulls.append(
                    GroupPull(
                        group_id=group_id,
                        remote_tp_offset=remote_tp_offset,
                        num_group_pulls=num_group_pulls,
                        prefill_pp_rank=prefill_pp_rank,
                        is_group_transfer_end=remote_tp_offset == num_group_pulls - 1,
                    )
                )
            pulls_by_port.append(group_pulls)
        return pulls_by_port

    @staticmethod
    def _serialize_group_for_scheduler(group: Any) -> dict[str, Any]:
        kv_cache_spec = group.kv_cache_spec
        return {
            "kv_cache_spec_type": type(kv_cache_spec).__name__,
            "kv_cache_spec": {
                "num_kv_heads": getattr(kv_cache_spec, "num_kv_heads", None),
                "num_key_value_heads": getattr(kv_cache_spec, "num_key_value_heads", None),
            },
            "layer_names": list(group.layer_names),
        }

    def _build_start_pull_params(self, request_id: str, params: dict[str, Any], decode_tp_rank: int) -> dict[str, Any]:
        prefill_tp_size = params.get("remote_ptp_size") or self._prefill_tp_size
        remote_request_id = params.get("remote_request_id", request_id)
        remote_ranks_per_decode = get_remote_ranks_for_req(
            remote_request_id,
            prefill_tp_size,
            self._decode_tp_size,
            self._prefill_pp_size,
            self.num_key_value_heads,
            self.is_deepseek_mla,
            self.use_sparse,
        )
        p_ranks = remote_ranks_per_decode[decode_tp_rank]
        base_port = params["remote_port"]
        remote_handshake_ports = [base_port + p_rank for p_rank in p_ranks]
        remote_host, remote_engine_id = resolve_remote_host_for_handshake_port(
            base_port,
            remote_handshake_ports[0],
            params["remote_host"],
            params["remote_engine_id"],
            params.get("remote_multi_nodes_meta_mapping"),
        )
        pull_params = copy.copy(params)
        pull_params["remote_host"] = remote_host
        pull_params["remote_engine_id"] = remote_engine_id
        pull_params["remote_handshake_ports"] = remote_handshake_ports
        pull_params["group_pulls_by_port"] = self._build_group_pulls_by_port(
            remote_handshake_ports,
            base_port,
            prefill_tp_size,
            decode_tp_rank,
        )
        pull_params["decode_tp_rank"] = decode_tp_rank
        return pull_params

    def _send_start_pull(self, request_id: str, params: dict[str, Any], d2rh_port: int) -> bytes:
        sock: zmq.Socket | None = None  # type: ignore[type-arg]
        try:
            sock = self._get_remote_socket(self.local_host, d2rh_port)
            d2rh_path = f"{self.local_host}:{d2rh_port}"
            ensure_zmq_send(sock, self.encoder.encode((START_PULL, request_id, params)), d2rh_path)
            return ensure_zmq_recv(sock, self.remote_poller, d2rh_path, timeout=self.timeout)
        finally:
            if sock is not None:
                self._return_remote_socket(sock, self.local_host, d2rh_port)

    def _get_remote_socket(self, remote_host: str, remote_handshake_port: int) -> zmq.Socket:  # type: ignore[type-arg]
        remote_path = make_zmq_path("tcp", remote_host, remote_handshake_port)
        with self.remote_sockets_lock:
            if self.remote_sockets[remote_path]:
                return self.remote_sockets[remote_path].popleft()
            ctx = zmq.Context()  # type: ignore
            sock = make_zmq_socket(ctx=ctx, path=remote_path, socket_type=zmq.REQ, bind=False)  # type: ignore
            sock.setsockopt(zmq.SNDTIMEO, int(self.timeout * 1000))  # type: ignore
            self.remote_poller.register(sock, zmq.POLLIN)  # type: ignore
            return sock

    def _return_remote_socket(self, sock: zmq.Socket, remote_host: str, remote_handshake_port: int) -> None:  # type: ignore[type-arg]
        remote_path = make_zmq_path("tcp", remote_host, remote_handshake_port)
        with self.remote_sockets_lock:
            self.remote_sockets[remote_path].append(sock)

    def get_num_new_matched_tokens(self, request: "Request", num_computed_tokens: int) -> tuple[int, bool]:
        params = request.kv_transfer_params
        if params is not None and params.get("do_remote_prefill"):
            if request.request_id not in self.all_requests:
                got_staging_full = False
                for decode_tp_rank in range(self._decode_tp_size):
                    d2rh_port = get_d2rh_zmq_port(self.vllm_config, decode_tp_rank)
                    pull_params = self._build_start_pull_params(request.request_id, params, decode_tp_rank)
                    resp = self._send_start_pull(request.request_id, pull_params, d2rh_port)
                    if resp == STAGING_FULL:
                        got_staging_full = True
                        break
                    if resp != b"ACK":
                        raise RuntimeError(f"Failed to receive ACK, resp: {resp.decode('utf-8')}")
                if got_staging_full:
                    with self.listeningthread.ready_lock:
                        self.all_requests.discard(request.request_id)
                        self.listeningthread.ready_count.pop(request.request_id, None)
                    return None, False  # type: ignore[return-value]
                self.all_requests.add(request.request_id)

            with self.listeningthread.ready_lock:
                if request.request_id not in self.listeningthread.ready_request:
                    return None, False  # type: ignore[return-value]
                # self.all_requests.remove(request.request_id)
                token_ids = request.prompt_token_ids or []
                actual = self._state_prefill_token_count(len(token_ids))
                params["num_computed_tokens"] = num_computed_tokens
                count = max(actual - num_computed_tokens, 0)
                return count, count > 0

        if params is not None and params.get("do_remote_decode") and self.need_truncate:
            self._truncate_request_for_prefill(request)
        return 0, False

    def update_state_after_alloc(self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int):
        if self.kv_role == "kv_consumer":
            with self.listeningthread.ready_lock:
                self.all_requests.discard(request.request_id)
                self.listeningthread.ready_request.discard(request.request_id)
                self.listeningthread.ready_count.pop(request.request_id, None)
        super().update_state_after_alloc(request, blocks, num_external_tokens)
        
class MooncakeConnectorWorker(BaseMooncakeConnectorWorker):
    def __init__(self, vllm_config: VllmConfig, engine_id: str, kv_cache_config: KVCacheConfig):
        super().__init__(vllm_config, engine_id, kv_cache_config)
        self.remote_local_block_map: dict[str, dict[tuple[int, ...], int]] = {}

    def _make_cpu_staging_caches(self, kv_caches: dict[str, torch.Tensor]) -> dict[str, list[torch.Tensor]]:
        """Allocate CPU staging buffers with 2M-aligned sizes for UBMem 2M-page mode.

        For each KV cache tensor, a flat pinned buffer whose byte size is rounded
        up to a 2M boundary is allocated via ``torch.empty(..., pin_memory=True)``
        (which internally calls ``aclrtMallocHost``).  A view with the original
        shape/dtype is created from the first ``num_blocks`` blocks; the tail
        padding is never accessed by block addressing.

        The flat buffer (owning the 2M-aligned storage) is kept in
        ``self.cpu_caches_hold`` for GC; the view is returned in *cpu_caches*
        for metadata extraction.  Registration info ``(ptr, aligned_size)`` is
        collected in ``self._cpu_register_ptrs`` / ``self._cpu_register_lengths``
        so that ``register_buffer`` sees the full 2M-aligned region.
        """
        cpu_caches: dict[str, list[torch.Tensor]] = {}
        self.cpu_caches_hold: list[torch.Tensor] = []
        self._cpu_register_ptrs: list[int] = []
        self._cpu_register_lengths: list[int] = []
        for layer_name, kv_cache_tuple in kv_caches.items():
            cpu_caches[layer_name] = []
            for cache in self._as_kv_cache_tuple(kv_cache_tuple):
                raw_size = cache.numel() * cache.element_size()
                aligned_size = (
                    (raw_size + HUGEPAGE_SIZE_2M - 1) // HUGEPAGE_SIZE_2M
                ) * HUGEPAGE_SIZE_2M
                # Allocate flat pinned buffer with 2M-aligned byte size.
                # Use the original dtype so element count = aligned_size / element_size.
                aligned_num_elements = aligned_size // cache.element_size()
                flat = torch.empty(
                    aligned_num_elements, dtype=cache.dtype, device="cpu", pin_memory=True
                )
                # Create a view with the original shape (first num_blocks blocks).
                cpu_cache = flat[: cache.numel()].view(cache.shape)
                # Keep the flat buffer alive — it owns the 2M-aligned storage.
                self.cpu_caches_hold.append(flat)
                cpu_caches[layer_name].append(cpu_cache)
                # Record 2M-aligned registration info.
                self._cpu_register_ptrs.append(flat.data_ptr())
                self._cpu_register_lengths.append(aligned_size)
        return cpu_caches

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]) -> None:
        self.use_mla = self.vllm_config.model_config.is_deepseek_mla
        self.use_sparse = hasattr(self.vllm_config.model_config.hf_text_config, "index_topk")
        self.num_blocks = self.kv_cache_config.num_blocks
        # Print config info for kvcaches  
        self.kv_caches = kv_caches
        self.kv_group2layeridx = self._build_kv_group2layeridx()
        has_mamba_group = self._has_mamba_group()
        layer_name_to_idx = {
            layer_name: layer_idx
            for _, (group_spec, layer_indices) in self.kv_group2layeridx.items()
            for layer_name, layer_idx in zip(group_spec["layer_names"], layer_indices)
        }
        metadata_layers = max(layer_name_to_idx.values(), default=-1) + 1
        self.kv_caches_base_addr: list[list[int]] = [[] for _ in range(metadata_layers)]
        self.block_size_scale: list[list[int]] = [[] for _ in range(metadata_layers)]
        self.block_len_per_addr: list[list[int]] = [[] for _ in range(metadata_layers)]
        self.block_shape_per_addr: list[list[int]] = [[] for _ in range(metadata_layers)]
        self.block_stride_per_addr: list[list[int]] = [[] for _ in range(metadata_layers)]

        for layer_name, kv_cache_tuple in kv_caches.items():
            layer_idx = layer_name_to_idx[layer_name]
            for single_kv_cache in self._as_kv_cache_tuple(kv_cache_tuple):
                tensor_num_blocks = single_kv_cache.shape[0]
                block_size_scale = tensor_num_blocks // self.num_blocks
                block_shape = single_kv_cache.shape[1:]
                self.block_len_per_addr[layer_idx].append(single_kv_cache.element_size() * math.prod(block_shape))
                self.block_stride_per_addr[layer_idx].append(single_kv_cache.stride(0) * single_kv_cache.element_size())
                self.block_shape_per_addr[layer_idx].append(single_kv_cache.shape)
                self.block_size_scale[layer_idx].append(block_size_scale)
                self.kv_caches_base_addr[layer_idx].append(single_kv_cache.data_ptr())

        if has_mamba_group:
            ptrs, lengths = self._get_registered_kv_tensor_buffers(kv_caches)
            register_regions = RegisterRegions(ptrs=ptrs, lengths=lengths)
        elif self.use_hybrid:
            ptrs, lengths = self._get_registered_kv_tensor_buffers_hybrid(kv_caches)
            register_regions = RegisterRegions(ptrs=ptrs, lengths=lengths)
        else:
            register_regions = collect_storage_merged_register_regions(kv_caches)

        cpu_kv_caches_base_addr: list[list[int]] = []
        cpu_block_len_per_addr: list[list[int]] = []
        cpu_block_stride_per_addr: list[list[int]] = []
        cpu_block_size_scale: list[list[int]] = []
        cpu_kvcache_manager: D2RHCPUCacheManager | None = None
        self.d2rh_thread: D2RHThread | None = None

        if self.kv_role == "kv_consumer":
            cpu_kvcache_manager = D2RHCPUCacheManager(self.num_blocks)
            cpu_caches = self._make_cpu_staging_caches(kv_caches)
            cpu_kv_caches_base_addr = [[] for _ in range(metadata_layers)]
            cpu_block_len_per_addr = [[] for _ in range(metadata_layers)]
            cpu_block_stride_per_addr = [[] for _ in range(metadata_layers)]
            cpu_block_size_scale = [[] for _ in range(metadata_layers)]
            for layer_name, cpu_cache_tuple in cpu_caches.items():
                layer_idx = layer_name_to_idx[layer_name]
                for single_cpu_cache in cpu_cache_tuple:
                    tensor_num_blocks = single_cpu_cache.shape[0]
                    block_shape = single_cpu_cache.shape[1:]
                    block_len = single_cpu_cache.element_size() * math.prod(block_shape)
                    cpu_kv_caches_base_addr[layer_idx].append(single_cpu_cache.data_ptr())
                    cpu_block_len_per_addr[layer_idx].append(block_len)
                    cpu_block_stride_per_addr[layer_idx].append(
                        single_cpu_cache.stride(0) * single_cpu_cache.element_size()
                    )
                    cpu_block_size_scale[layer_idx].append(tensor_num_blocks // self.num_blocks)
            # CPU staging caches are freshly allocated independent tensors with
            # 2M-aligned storage.  Register the full 2M-aligned regions directly
            # (collected in _make_cpu_staging_caches) so that UBMem uses 2M huge
            # pages instead of falling back to 4K small pages.
            # cpu_register_regions = collect_storage_merged_register_regions(cpu_caches)
            register_regions.ptrs.extend(self._cpu_register_ptrs)
            register_regions.lengths.extend(self._cpu_register_lengths)

        validate_register_region_count(register_regions)
        global_te.register_buffer(register_regions.ptrs, register_regions.lengths)

        metadata = MooncakeAgentMetadata(
            engine_id=self.engine_id,
            te_rpc_port=self.te_rpc_port,
            kv_group2layeridx=self.kv_group2layeridx,
            block_size=self.block_size,
            kv_caches_base_addr=self.kv_caches_base_addr,
            block_size_scale=self.block_size_scale,
            num_blocks=self.num_blocks,
            block_lens=self.block_len_per_addr,
            block_strides=self.block_stride_per_addr,
            local_ip=get_ip(),
        )
        self.xfer_handshake_metadata = metadata

        ready_event = threading.Event()
        if self.kv_role == "kv_producer":
            self.kv_send_thread = KVCacheSendingThread(
                self.vllm_config,
                self.tp_rank,
                self._prefill_tp_size,
                self.engine_id,
                self.side_channel_host,
                self.side_channel_port,
                metadata,
                ready_event,
                self.kv_caches,
                self.pcp_rank,
            )
            self.kv_send_thread.start()
        else:
            assert cpu_kvcache_manager is not None
            d2rh_port = get_d2rh_zmq_port(self.vllm_config, self.tp_rank, self.pp_rank, self.pcp_rank)
            scheduler_ready_port = get_scheduler_ready_zmq_port(self.vllm_config)
            self.d2rh_thread = D2RHThread(
                cpu_kv_caches_base_addr=cpu_kv_caches_base_addr,
                cpu_block_len_per_addr=cpu_block_len_per_addr,
                cpu_block_stride_per_addr=cpu_block_stride_per_addr,
                cpu_block_size_scale=cpu_block_size_scale,
                kv_group2layeridx=self.kv_group2layeridx,
                engine=self.engine,
                cpu_kvcache_manager=cpu_kvcache_manager,
                remote_local_block_map=self.remote_local_block_map,
                vllm_config=self.vllm_config,
                d2rh_handshake_port=d2rh_port,
                scheduler_ready_port=scheduler_ready_port,
                tp_rank=self.tp_rank,
            )
            self.d2rh_thread.start()
            self.kv_recv_thread = KVCacheRecvingThread(
                self.tp_rank,
                self.tp_size,
                self._prefill_pp_size,
                self.engine,
                self.engine_id,
                self.handshake_port,
                self.side_channel_port,
                self.kv_caches_base_addr,
                self.block_len_per_addr,
                self.block_stride_per_addr,
                self._is_hma_required,
                ready_event,
                self.vllm_config,
                self.kv_caches,
                self._prefill_pp_layer_partition,
                self.kv_group2layeridx,
                self.block_size_scale,
                cpu_kv_caches_base_addr=cpu_kv_caches_base_addr,
                cpu_block_len_per_addr=cpu_block_len_per_addr,
                cpu_block_stride_per_addr=cpu_block_stride_per_addr,
                cpu_block_size_scale=cpu_block_size_scale,
                cpu_te_rpc_port=self.te_rpc_port,
                cpu_kvcache_manager=cpu_kvcache_manager,
                remote_local_block_map=self.remote_local_block_map,
            )
            self.kv_recv_thread.start()

        start_wait_time = time.time()
        thread = self.kv_send_thread if self.kv_role == "kv_producer" else self.kv_recv_thread
        assert thread is not None
        while not ready_event.is_set():
            if not thread.is_alive():
                raise RuntimeError("KV Cache sending/receiving thread failed to start.")
            if time.time() - start_wait_time > 5 * 60:
                raise RuntimeError("Timeout waiting for KV Cache thread to be ready.")
            time.sleep(3)


class MooncakeConnector(KVConnectorBase_V1, SupportsHMA):
    def __init__(self, vllm_config: VllmConfig, role: KVConnectorRole, kv_cache_config: KVCacheConfig | None = None):
        assert vllm_config.kv_transfer_config is not None
        assert kv_cache_config is not None
        self.engine_id = vllm_config.kv_transfer_config.engine_id
        self._connector_metadata = MooncakeConnectorMetadata()

        if role == KVConnectorRole.SCHEDULER:
            self.connector_scheduler: MooncakeConnectorScheduler | None = MooncakeConnectorScheduler(
                vllm_config, str(self.engine_id), kv_cache_config
            )
            self.connector_worker: MooncakeConnectorWorker | None = None
        elif role == KVConnectorRole.WORKER:
            self.connector_scheduler = None
            self.connector_worker = MooncakeConnectorWorker(vllm_config, str(self.engine_id), kv_cache_config)

    def get_num_new_matched_tokens(self, request: "Request", num_computed_tokens: int) -> tuple[int, bool]:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.get_num_new_matched_tokens(request, num_computed_tokens)

    def update_state_after_alloc(self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int):
        assert self.connector_scheduler is not None
        return self.connector_scheduler.update_state_after_alloc(request, blocks, num_external_tokens)

    def build_connector_meta(self, scheduler_output: SchedulerOutput) -> KVConnectorMetadata:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.build_connector_meta(scheduler_output)

    def request_finished(self, request: "Request", block_ids: list[int]) -> tuple[bool, dict[str, Any] | None]:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.request_finished(request, (block_ids,))

    def request_finished_all_groups(
        self, request: "Request", block_ids: tuple[list[int], ...]
    ) -> tuple[bool, dict[str, Any] | None]:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.request_finished(request, block_ids)

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]) -> None:
        assert self.connector_worker is not None
        self.connector_worker.register_kv_caches(kv_caches)

    def get_finished(self, finished_req_ids: set[str]) -> tuple[set[str], set[str]]:
        assert self.connector_worker is not None
        return self.connector_worker.get_finished()

    def get_block_ids_with_load_errors(self) -> set[int]:
        assert self.connector_worker is not None
        return self.connector_worker.get_block_ids_with_load_errors()

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs) -> None:
        assert self.connector_worker is not None
        assert isinstance(self._connector_metadata, MooncakeConnectorMetadata)
        self.connector_worker.start_load_kv(self._connector_metadata)

    def wait_for_layer_load(self, layer_name: str) -> None:
        pass

    def save_kv_layer(
        self, layer_name: str, kv_layer: torch.Tensor, attn_metadata: "AttentionMetadata", **kwargs
    ) -> None:
        pass

    def wait_for_save(self) -> None:
        pass

    def get_handshake_metadata(self) -> KVConnectorHandshakeMetadata | None:
        assert self.connector_worker is not None
        return self.connector_worker.xfer_handshake_metadata

    def set_xfer_handshake_metadata(self, metadata: dict[int, KVConnectorHandshakeMetadata]) -> None:
        assert self.connector_scheduler is not None
        self.connector_scheduler.set_xfer_handshake_metadata(metadata)
