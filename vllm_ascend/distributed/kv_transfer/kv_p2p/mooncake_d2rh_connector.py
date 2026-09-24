# SPDX-License-Identifier: Apache-2.0
import contextlib
import copy
import hashlib
import logging
import math
import queue
import random
import struct
import threading
import time
from collections import OrderedDict, defaultdict, deque
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any

import msgspec
import numpy as np
import torch
import zmq
from mooncake.engine import TransferEngine  # type: ignore
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.utils import BlockIds
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
from vllm.logger import logger
from vllm.utils.network_utils import get_ip, make_zmq_path, make_zmq_socket
from vllm.v1.core import kv_cache_utils
from vllm.v1.kv_cache_interface import KVCacheConfig, MambaSpec

from vllm_ascend.core.kv_cache_interface import AscendMLAAttentionSpec, AscendSlidingWindowMLASpec
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_connector import (
    GroupPull,
    GroupTransferInfo,
    KVCacheSendingThread,
    KVCacheTaskTracker,
    MooncakeAgentMetadata,
    MooncakeConnectorMetadata,
    RemotePortInfo,
    SizedDict,
    build_layer_name_to_metadata_idx,
    ensure_zmq_send,
    resolve_remote_layer_idx,
    split_if_not_byte_contiguous,
    string_to_int64_hash,
    transfer_groups_need_independent_block_ids,
    zmq_ctx,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_connector import (
    KVCacheRecvingThread as BaseKVCacheRecvingThread,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_connector import (
    MooncakeConnector as BaseMooncakeConnector,
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
from vllm_ascend.distributed.kv_transfer.utils.utils import RegisterRegions, validate_register_region_count
from vllm_ascend.utils import enable_sfa_dcp_replicated_indexer, refresh_block_size

# isort: off
if TYPE_CHECKING:
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.request import Request
# isort: on

GET_META_MSG = b"get_meta_msg"
DONE_RECVING_MSG = b"done_recving_msg"
READY_SCHEDULER = b"ready_scheduler"
STAGING_FULL = b"staging_full"
START_PULL = b"START_PULL"

# Ascend D2H registration requires 2 MiB-aligned host ranges so UBMem can use
# huge pages. torch pin_memory uses aclrtMallocHost, and rounding the allocation
# size preserves huge-page registration.
HUGEPAGE_SIZE_2M = 2 * 1024 * 1024

StagingBlockKey = tuple[int, int, int]
StagingBlockMap = dict[tuple[int, ...], int]
HostCacheKey = tuple[int, int, bytes]


def build_layer_name_to_cache_slots(
    kv_group2layeridx: dict[int, tuple[dict[str, Any], list[int]]],
) -> dict[str, list[int]]:
    """Build D2RH-only named component metadata for hybrid KV layouts."""
    result: dict[str, list[int]] = {}
    for group_spec, _ in kv_group2layeridx.values():
        for name, slots in group_spec.get("layer_cache_indices", {}).items():
            if name in result and result[name] != slots:
                raise RuntimeError(f"Conflicting KV component metadata for {name!r}")
            result[name] = slots
    return result


def resolve_group_cache_slot_pairs(
    group_spec: dict[str, Any],
    layer_indices: list[int],
    layer_idx: int,
    remote_cache_slots: dict[str, list[int]],
    local_count: int,
    remote_count: int,
) -> list[tuple[int, int]]:
    """Select D2RH cache components owned by this group using stable names."""
    local_slots = group_spec.get("layer_cache_indices")
    if local_slots is None:
        if local_count != remote_count:
            raise RuntimeError("Legacy KV component counts differ between peers")
        return [(index, index) for index in range(local_count)]
    names = group_spec.get("layer_names", [])
    if len(names) != len(layer_indices):
        raise RuntimeError("Misaligned KV component layer metadata")
    result: list[tuple[int, int]] = []
    for name, index in zip(names, layer_indices):
        if index != layer_idx:
            continue
        local = local_slots.get(name)
        remote = remote_cache_slots.get(name)
        if local is None or remote is None or len(local) != len(remote):
            raise RuntimeError(f"Missing or incompatible KV component metadata for {name!r}")
        for local_idx, remote_idx in zip(local, remote):
            if not (0 <= local_idx < local_count and 0 <= remote_idx < remote_count):
                raise RuntimeError(f"Out-of-range KV component metadata for {name!r}")
            pair = (local_idx, remote_idx)
            if pair not in result:
                result.append(pair)
    if not result:
        raise RuntimeError(f"KV group owns no components for metadata layer {layer_idx}")
    return result


def _get_non_redundant_cache_slot_pairs(
    pairs: list[tuple[int, int]],
    local_addrs: list[int],
    remote_addrs: list[int],
    block_lengths: list[int],
    local_strides: list[int],
    remote_strides: list[int],
    num_group_pulls: int,
) -> list[tuple[int, int]]:
    """Omit views fully covered by another selected view on both peers.

    Alias offsets and strides must agree for every block. Keep TP-sharded
    components unchanged because their byte offsets need not scale equally.
    """
    if num_group_pulls != 1:
        return pairs
    result = []
    for position, (local, remote) in enumerate(pairs):
        covered = False
        for other_position, (other_local, other_remote) in enumerate(pairs):
            if other_position == position:
                continue
            delta = local_addrs[local] - local_addrs[other_local]
            if (
                delta >= 0
                and delta == remote_addrs[remote] - remote_addrs[other_remote]
                and local_strides[local] == local_strides[other_local]
                and remote_strides[remote] == remote_strides[other_remote]
                and delta + block_lengths[local] <= block_lengths[other_local]
                and (block_lengths[local] < block_lengths[other_local] or other_position < position)
            ):
                covered = True
                break
        if not covered:
            result.append((local, remote))
    return result


# Offset each ZMQ endpoint by the flattened parallel rank to prevent port
# collisions between DP, TP, PP, and PCP workers:
# port = BASE + dp_rank * tp_size * pp_size * pcp_size
#        + (pp_rank + pcp_rank) * tp_size + tp_rank
D2RH_ZMQ_PORT_BASE = 38100
SCHEDULER_READY_ZMQ_PORT_BASE = 38200


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
        self.ready_request: set[str] = set()
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
            # Each rank listens on its own endpoint to avoid routing metadata
            # requests through a shared scheduler socket.
            handshake_port = self.scheduler_ready_port
            path = make_zmq_path("tcp", self.host_ip, handshake_port)
            logger.info("Starting scheduler ready listener on path: %s", path)
            with zmq_ctx(zmq.ROUTER, path) as sock:  # type: ignore
                self.run_busy_loop(sock)
        except Exception as e:
            logger.exception("Mooncake KVCacheSendingThread exception: %s", e)

    def run_busy_loop(self, sock: zmq.Socket):  # type: ignore
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
    kv_transfer_config = vllm_config.kv_transfer_config
    assert kv_transfer_config is not None
    parallel_config = vllm_config.parallel_config
    tp_size = parallel_config.tensor_parallel_size
    pp_size = parallel_config.pipeline_parallel_size
    pcp_size = parallel_config.prefill_context_parallel_size
    dp_rank = parallel_config.data_parallel_rank
    device_index = get_parallel_device_index(tp_rank, tp_size, pp_rank, pcp_rank)
    dp_offset = get_dp_port_offset(dp_rank, tp_size, pp_size, pcp_size)
    base_port = int(kv_transfer_config.get_from_extra_config("d2rh_zmq_port", D2RH_ZMQ_PORT_BASE))
    return base_port + dp_offset + device_index


def get_scheduler_ready_zmq_port(vllm_config: VllmConfig) -> int:
    kv_transfer_config = vllm_config.kv_transfer_config
    assert kv_transfer_config is not None
    parallel_config = vllm_config.parallel_config
    tp_size = parallel_config.tensor_parallel_size
    pp_size = parallel_config.pipeline_parallel_size
    pcp_size = parallel_config.prefill_context_parallel_size
    dp_rank = parallel_config.data_parallel_rank
    dp_offset = get_dp_port_offset(dp_rank, tp_size, pp_size, pcp_size)
    base_port = int(
        kv_transfer_config.get_from_extra_config("d2rh_scheduler_ready_port", SCHEDULER_READY_ZMQ_PORT_BASE)
    )
    return base_port + dp_offset


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
    reshaped_data = ori_data.reshape(prefill_pp_size, -1)
    num_groups = max(1, len(reshaped_data[0]) // num_kv_head)
    rand_group_index = rand.sample(range(num_groups), max(decode_tp_size // num_kv_head, 1))
    all_results = [
        get_remote_tp_ranks(
            reshaped_data[pp_index],
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


def _is_sliding_group_spec(group_spec: Mapping[str, Any]) -> bool:
    """Return whether transfer ids were clipped to a live sliding-window tail."""
    if "SlidingWindow" in str(group_spec.get("kv_cache_spec_type", "")):
        return True
    pending = [group_spec.get("kv_cache_spec")]
    while pending:
        value = pending.pop()
        if isinstance(value, Mapping):
            if value.get("sliding_window"):
                return True
            pending.extend(value.values())
        elif isinstance(value, (list, tuple)):
            pending.extend(value)
    return False


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
        # Valid entries are retained after a request finishes. OrderedDict
        # order is LRU (oldest first). Pending entries are never exposed as
        # hits, which keeps concurrent requests from reading half-filled host
        # blocks.
        self.cache_by_key: OrderedDict[HostCacheKey, int] = OrderedDict()
        self.cache_key_by_block: dict[int, HostCacheKey] = {}
        # Cached blocks with no active users, ordered from least to most
        # recently released. This avoids rescanning the full host cache for
        # every staging allocation when the cache is saturated.
        self.evictable_blocks: OrderedDict[int, None] = OrderedDict()
        self.pending_by_key: dict[HostCacheKey, int] = {}
        self.pending_key_by_block: dict[int, HostCacheKey] = {}
        self.pin_count: dict[int, int] = defaultdict(int)
        self.lock = threading.Lock()

    def _remove_evictable_locked(self, block_id: int) -> None:
        self.evictable_blocks.pop(block_id, None)

    def _add_evictable_locked(self, block_id: int) -> None:
        if block_id in self.cache_key_by_block and self.pin_count.get(block_id, 0) == 0:
            self.evictable_blocks[block_id] = None

    def _take_block_locked(self) -> int | None:
        if self.free_queue:
            return self.free_queue.popleft()
        while self.evictable_blocks:
            block_id, _ = self.evictable_blocks.popitem(last=False)
            cache_key = self.cache_key_by_block.get(block_id)
            if cache_key is None or self.cache_by_key.get(cache_key) != block_id:
                continue
            if self.pin_count.get(block_id, 0) != 0:
                continue
            self.cache_by_key.pop(cache_key, None)
            self.cache_key_by_block.pop(block_id, None)
            self.used_set.discard(block_id)
            return block_id
        return None

    def _release_blocks_locked(self, block_ids: set[int]) -> None:
        for block_id in block_ids:
            pins = self.pin_count.get(block_id, 0)
            if pins > 1:
                self.pin_count[block_id] = pins - 1
                continue
            self.pin_count.pop(block_id, None)
            pending_key = self.pending_key_by_block.pop(block_id, None)
            if pending_key is not None:
                self.pending_by_key.pop(pending_key, None)
            if block_id in self.cache_key_by_block:
                # Valid cached blocks stay resident but become evictable.
                self._add_evictable_locked(block_id)
                continue
            if block_id in self.used_set:
                self.used_set.remove(block_id)
                self.free_queue.append(block_id)

    def alloc_block_map(self, remote_block_ids: BlockIds) -> StagingBlockMap | None:
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

            block_map: StagingBlockMap = {}
            for key in remote_keys:
                local_block_id = self.free_queue.popleft()
                self.used_set.add(local_block_id)
                block_map[key] = local_block_id
            return block_map

    def alloc_sharded_block_map(
        self,
        remote_block_ids: BlockIds,
        group_pulls_by_port: list[list[GroupPull | dict[str, Any]]],
        block_hashes: tuple[list[bytes | None], ...] | None = None,
    ) -> tuple[StagingBlockMap, set[StagingBlockKey], dict[StagingBlockKey, HostCacheKey]] | None:
        offsets_by_group: dict[int, set[int]] = defaultdict(set)
        for group_pulls in group_pulls_by_port:
            for group_pull in group_pulls:
                group_id = int(_get_group_pull_field(group_pull, "group_id"))
                remote_tp_offset = int(_get_group_pull_field(group_pull, "remote_tp_offset"))
                offsets_by_group[group_id].add(remote_tp_offset)

        with self.lock:
            remote_keys: list[StagingBlockKey] = []
            seen_keys: set[StagingBlockKey] = set()
            block_index_by_group = [
                {block_id: idx for idx, block_id in enumerate(group_block_ids)} for group_block_ids in remote_block_ids
            ]
            for group_id, group_block_ids in enumerate(remote_block_ids):
                offsets = offsets_by_group.get(group_id) or {0}
                for remote_block_id in group_block_ids:
                    for remote_tp_offset in sorted(offsets):
                        key = (group_id, remote_block_id, remote_tp_offset)
                        if key in seen_keys:
                            continue
                        seen_keys.add(key)
                        remote_keys.append(key)

            block_map: StagingBlockMap = {}
            cache_hits: set[StagingBlockKey] = set()
            cacheable_misses: dict[StagingBlockKey, HostCacheKey] = {}
            acquired_blocks: set[int] = set()
            for key in remote_keys:
                group_id, remote_block_id, remote_tp_offset = key
                content_hash: bytes | None = None
                if block_hashes is not None and group_id < len(block_hashes):
                    group_hashes = block_hashes[group_id]
                    block_idx = block_index_by_group[group_id].get(remote_block_id, -1)
                    if 0 <= block_idx < len(group_hashes):
                        content_hash = group_hashes[block_idx]

                cache_key: HostCacheKey | None = None
                if content_hash is not None:
                    cache_key = (group_id, remote_tp_offset, bytes(content_hash))
                    local_block_id = self.cache_by_key.get(cache_key)
                    if local_block_id is not None:
                        self.cache_by_key.move_to_end(cache_key)
                        if local_block_id not in acquired_blocks:
                            self._remove_evictable_locked(local_block_id)
                            self.pin_count[local_block_id] += 1
                        block_map[key] = local_block_id
                        cache_hits.add(key)
                        acquired_blocks.add(local_block_id)
                        continue

                local_block_id = self._take_block_locked()
                if local_block_id is None:
                    self._release_blocks_locked(acquired_blocks)
                    logger.info(
                        "CPU staging does not have enough evictable blocks, need=%d acquired=%d free=%d cached=%d",
                        len(remote_keys),
                        len(acquired_blocks),
                        len(self.free_queue),
                        len(self.cache_by_key),
                    )
                    return None
                self.used_set.add(local_block_id)
                self.pin_count[local_block_id] += 1
                block_map[key] = local_block_id
                acquired_blocks.add(local_block_id)
                # Only the first concurrent loader owns a pending cache entry.
                # Other requests use a transient block until the first load is
                # committed, preventing dirty reads without blocking the queue.
                if cache_key is not None and cache_key not in self.pending_by_key:
                    self.pending_by_key[cache_key] = local_block_id
                    self.pending_key_by_block[local_block_id] = cache_key
                    cacheable_misses[key] = cache_key
            logger.debug(
                "CPU staging allocation complete: used=%s free=%s block_map=%s",
                self.used_set,
                self.free_queue,
                block_map,
            )
            return block_map, cache_hits, cacheable_misses

    def commit_block_map(
        self,
        block_map: StagingBlockMap,
        cacheable_misses: dict[StagingBlockKey, HostCacheKey],
    ) -> None:
        with self.lock:
            for staging_key, cache_key in cacheable_misses.items():
                block_id = block_map.get(staging_key)
                if block_id is None or self.pending_by_key.get(cache_key) != block_id:
                    continue
                self.pending_by_key.pop(cache_key, None)
                self.pending_key_by_block.pop(block_id, None)
                old_block_id = self.cache_by_key.pop(cache_key, None)
                if old_block_id is not None and old_block_id != block_id:
                    self._remove_evictable_locked(old_block_id)
                    self.cache_key_by_block.pop(old_block_id, None)
                    if self.pin_count.get(old_block_id, 0) == 0:
                        self.used_set.discard(old_block_id)
                        self.free_queue.append(old_block_id)
                self._remove_evictable_locked(block_id)
                self.cache_by_key[cache_key] = block_id
                self.cache_key_by_block[block_id] = cache_key
                self._add_evictable_locked(block_id)

    def cache_stats(self) -> tuple[int, int, int]:
        with self.lock:
            return len(self.cache_by_key), len(self.pending_by_key), len(self.free_queue)

    def free_block_map(self, block_map: dict[tuple[int, ...], int]) -> None:
        with self.lock:
            self._release_blocks_locked(set(block_map.values()))
            logger.debug(
                "CPU staging blocks released: used=%s free=%s block_map=%s",
                self.used_set,
                self.free_queue,
                block_map,
            )


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
        self.transfer_workers = max(
            1,
            int(vllm_config.kv_transfer_config.get_from_extra_config("d2rh_transfer_workers", 1)),
        )
        self.log_full_block_map = bool(
            vllm_config.kv_transfer_config.get_from_extra_config("d2rh_log_full_block_map", True)
        )
        self.executor = (
            ThreadPoolExecutor(
                max_workers=self.transfer_workers,
                thread_name_prefix=f"D2RH-Xfer-TP{tp_rank}",
            )
            if self.transfer_workers > 1
            else None
        )
        self.remote_sockets_lock = threading.Lock()
        self.remote_sockets: dict[str, deque[zmq.Socket]] = defaultdict(deque)  # type: ignore[name-defined]
        self.remote_metadata_lock = threading.Lock()
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
        kwargs["d2rh_enqueued_at"] = time.perf_counter()
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
                    if self.executor is None:
                        self._handle_request_logged(request_data)
                    else:
                        self.executor.submit(self._handle_request_logged, request_data)
            except Exception as e:
                logger.exception("D2RH request processing failed: %s", e)
            finally:
                self.request_queue.task_done()

    def _handle_request_logged(self, req_meta: dict[str, Any]) -> None:
        started_at = time.perf_counter()
        queue_ms = (started_at - req_meta.get("d2rh_enqueued_at", started_at)) * 1000
        try:
            self._handle_request(req_meta)
        except Exception as e:
            logger.exception("D2RH request processing failed: %s", e)
        finally:
            total_ms = (time.perf_counter() - started_at) * 1000
            logger.info(
                "D2RH_METRIC request_id=%s queue_ms=%.3f hop1_total_ms=%.3f workers=%d computed_tokens=%d",
                req_meta.get("request_id"),
                queue_ms,
                total_ms,
                self.transfer_workers,
                req_meta.get("num_computed_tokens", 0),
            )

    def run_busy_loop(self, sock: zmq.Socket) -> None:  # type: ignore[name-defined]
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
                        assert params is not None
                        remote_block_ids: BlockIds = tuple(params.get("remote_block_ids") or ())
                        raw_block_hashes = params.get("d2rh_block_hashes")
                        block_hashes = (
                            tuple(
                                [
                                    bytes.fromhex(value)
                                    if isinstance(value, str)
                                    else bytes(value)
                                    if value is not None
                                    else None
                                    for value in group
                                ]
                                for group in raw_block_hashes
                            )
                            if raw_block_hashes is not None
                            else None
                        )
                        allocation = self.cpu_kvcache_manager.alloc_sharded_block_map(
                            remote_block_ids,
                            params["group_pulls_by_port"],
                            block_hashes,
                        )
                        if allocation is None:
                            pull_ack = STAGING_FULL
                        else:
                            block_map, cache_hits, cacheable_misses = allocation
                            remote_request_id = params.get("remote_request_id")
                            if not isinstance(remote_request_id, str):
                                remote_request_id = request_id
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
                                cache_hits=cache_hits,
                                cacheable_misses=cacheable_misses,
                            )
                    except Exception as e:
                        # Release any partially created mapping to prevent CPU
                        # staging leaks on handshake/queueing failures.
                        remote_request_id = request_id
                        if params is not None:
                            candidate_request_id = params.get("remote_request_id")
                            if isinstance(candidate_request_id, str):
                                remote_request_id = candidate_request_id
                        block_map = self.remote_local_block_map.pop(remote_request_id, {})
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
            block_map = self.remote_local_block_map[remote_request_id]
            self.cpu_kvcache_manager.commit_block_map(
                block_map,
                req_meta.get("cacheable_misses", {}),
            )
            cached, pending, free = self.cpu_kvcache_manager.cache_stats()
            hits = len(req_meta.get("cache_hits", ()))
            misses = len(block_map) - hits
            logger.info(
                "D2RH_HOST_CACHE request_id=%s hits=%d misses=%d hit_rate=%.2f%% resident=%d pending=%d free=%d",
                request_id,
                hits,
                misses,
                100.0 * hits / max(1, hits + misses),
                cached,
                pending,
                free,
            )
            self.send_pull_done(request_id)
        except Exception:
            # Ensure staged CPU blocks are reclaimed if hop1 transfer fails.
            block_map = self.remote_local_block_map.pop(remote_request_id, {})
            self.remote_local_block_map.pop(request_id, None)
            if block_map:
                self.cpu_kvcache_manager.free_block_map(block_map)
            raise

    def _get_hop1_layer_pairs(
        self,
        group_spec: dict[str, Any],
        layer_indices: list[int],
        remote_layer_name_to_idx: dict[str, int],
    ) -> list[tuple[int, int]]:
        """Map D host layers to the layers actually registered by this P stage.

        The handshake is authoritative for PP ownership, including custom
        partitions, index-cache planes, and draft layers. Metadata indices are
        not necessarily identical between P and D.
        """
        prefill = self.vllm_config.kv_transfer_config.get_from_extra_config("prefill", {})
        pp_size = prefill.get("pp_size", 1)
        layer_names = group_spec.get("layer_names", [])
        if not remote_layer_name_to_idx or not layer_names:
            if pp_size > 1:
                raise RuntimeError("D2RH prefill PP requires cache layer names in the handshake metadata.")
            return [(idx, idx) for idx in dict.fromkeys(layer_indices)]
        if len(layer_names) != len(layer_indices):
            raise RuntimeError("D2RH local cache layer names and indices are misaligned.")
        pairs = []
        for layer_name, layer_idx in zip(layer_names, layer_indices):
            if pp_size > 1 and layer_name not in remote_layer_name_to_idx:
                continue
            remote_idx = resolve_remote_layer_idx(layer_idx, group_spec, layer_indices, remote_layer_name_to_idx)
            pairs.append((layer_idx, remote_idx))
        # Multiple cache names can share one metadata layer. Slot resolution
        # already collects all of that layer's components, so transfer it once.
        return list(dict.fromkeys(pairs))

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
        cache_hits: set[StagingBlockKey] = req_meta.get("cache_hits", set())
        cacheable_misses = req_meta.get("cacheable_misses", {})
        if self.log_full_block_map:
            logger.info("[D2RH Thread] block_map: %s", block_map)
        else:
            logger.debug("[D2RH Thread] block_map entries=%d", len(block_map))

        if not any(remote_block_ids):
            return

        # Validate the complete PP plan before any transfer/producer release.
        # A host-cache entry may only become valid after every stage is present.
        layer_pairs_by_port: dict[int, dict[int, list[tuple[int, int]]]] = {}
        covered_layers: dict[tuple[int, int], set[int]] = defaultdict(set)
        for handshake_port, pulls in zip(remote_handshake_ports, group_pulls_by_port):
            host, engine_id = resolve_remote_host_for_handshake_port(
                remote_port,
                handshake_port,
                remote_host,
                remote_engine_id,
                remote_multi_nodes_meta_mapping,
            )
            with self.remote_metadata_lock:
                if (
                    engine_id not in self.kv_caches_base_addr
                    or handshake_port not in self.kv_caches_base_addr[engine_id]
                ):
                    self._get_remote_metadata(host, handshake_port)
            remote_groups = self.remote_kv_group2layeridx[engine_id][handshake_port]
            remote_names = build_layer_name_to_metadata_idx(remote_groups)
            remote_slots = build_layer_name_to_cache_slots(remote_groups)
            port_pairs = layer_pairs_by_port.setdefault(handshake_port, {})
            for pull in pulls:
                group_id = _get_group_pull_field(pull, "group_id")
                offset = _get_group_pull_field(pull, "remote_tp_offset")
                spec, indices = self.kv_group2layeridx[group_id]
                pairs = self._get_hop1_layer_pairs(spec, indices, remote_names)
                for local_idx, remote_idx in pairs:
                    resolve_group_cache_slot_pairs(
                        spec,
                        indices,
                        local_idx,
                        remote_slots,
                        len(self.cpu_kv_caches_base_addr[local_idx]),
                        len(self.kv_caches_base_addr[engine_id][handshake_port][remote_idx]),
                    )
                port_pairs[group_id] = pairs
                covered_layers[group_id, offset].update(idx for idx, _ in pairs)
        for (group_id, offset), covered in covered_layers.items():
            missing = set(self.kv_group2layeridx[group_id][1]) - covered
            if remote_block_ids[group_id] and missing:
                raise RuntimeError(
                    f"Incomplete D2RH PP layer coverage: group={group_id}, "
                    f"tp_offset={offset}, missing={sorted(missing)}"
                )

        for remote_handshake_port, group_pulls in zip(remote_handshake_ports, group_pulls_by_port):
            port_host, port_engine_id = resolve_remote_host_for_handshake_port(
                remote_port,
                remote_handshake_port,
                remote_host,
                remote_engine_id,
                remote_multi_nodes_meta_mapping,
            )
            with self.remote_metadata_lock:
                if (
                    port_engine_id not in self.kv_caches_base_addr
                    or remote_handshake_port not in self.kv_caches_base_addr[port_engine_id]
                ):
                    self._get_remote_metadata(port_host, remote_handshake_port)

            remote_base_addrs = self.kv_caches_base_addr[port_engine_id][remote_handshake_port]
            remote_cache_slots = build_layer_name_to_cache_slots(
                self.remote_kv_group2layeridx[port_engine_id][remote_handshake_port]
            )
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
                layer_pairs = layer_pairs_by_port[remote_handshake_port][group_id]
                if not layer_pairs:
                    continue
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
                is_mamba_group = group_spec["kv_cache_spec_type"] == "MambaSpec"
                if not is_mamba_group and not _is_sliding_group_spec(group_spec):
                    # Apply the D-side HBM prefix offset while the list still
                    # has its original absolute block positions.  Host-cache
                    # filtering below can remove arbitrary entries, after
                    # which applying an absolute offset would over-slice the
                    # already compacted miss list.
                    remote_block_token_size = self.block_size * self.group_compress_ratios[group_id]
                    remote_start_block = req_meta.get("num_computed_tokens", 0) // remote_block_token_size
                    # HBM-resident prefixes are not copied into new Host blocks.
                    # Do not publish those unwritten misses as valid cache entries.
                    # Existing Host hits remain valid; skipped allocations stay
                    # pinned until the normal H2D completion cleanup releases them.
                    if remote_start_block:
                        for block_id in remote_group_block_ids[:remote_start_block]:
                            cacheable_misses.pop((group_id, block_id, remote_tp_offset), None)
                    remote_group_block_ids = remote_group_block_ids[remote_start_block:]
                    local_group_block_ids = local_group_block_ids[remote_start_block:]
                if cache_hits:
                    uncached_pairs = [
                        (remote_id, local_id)
                        for remote_id, local_id in zip(remote_group_block_ids, local_group_block_ids)
                        if (group_id, remote_id, remote_tp_offset) not in cache_hits
                    ]
                    remote_group_block_ids = [pair[0] for pair in uncached_pairs]
                    local_group_block_ids = [pair[1] for pair in uncached_pairs]
                if not local_group_block_ids:
                    continue

                if is_mamba_group:
                    if len(local_group_block_ids) != len(remote_group_block_ids):
                        raise RuntimeError("For MambaSpec num block should equal on P node and CPU staging.")
                    grouped_remote_block_ids = [[remote_group_block_ids[-1]]]
                    grouped_local_block_ids = [[local_group_block_ids[0]]]
                else:
                    first_local_idx, first_remote_idx = layer_pairs[0]
                    local_scale = self.cpu_block_size_scale[first_local_idx][0]
                    remote_scale = remote_block_size_scale[first_remote_idx][0]
                    kernel_local_block_ids = expand_block_ids(local_group_block_ids, local_scale)
                    kernel_remote_block_ids = expand_block_ids(remote_group_block_ids, remote_scale)
                    # Absolute HBM-prefix slicing was already applied to both
                    # remote_group_block_ids and local_group_block_ids before
                    # host-cache filtering.  Do not slice the expanded staging
                    # ids a second time. Sliding groups deliberately keep their
                    # complete live tail.
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

                for layer_idx, remote_layer_idx in layer_pairs:
                    cache_pairs = resolve_group_cache_slot_pairs(
                        group_spec,
                        layer_indices,
                        layer_idx,
                        remote_cache_slots,
                        len(self.cpu_kv_caches_base_addr[layer_idx]),
                        len(remote_base_addrs[remote_layer_idx]),
                    )
                    cache_pairs = _get_non_redundant_cache_slot_pairs(
                        cache_pairs,
                        self.cpu_kv_caches_base_addr[layer_idx],
                        remote_base_addrs[remote_layer_idx],
                        self.cpu_block_len_per_addr[layer_idx],
                        self.cpu_block_stride_per_addr[layer_idx],
                        remote_block_stride_per_addr[remote_layer_idx],
                        num_group_pulls,
                    )
                    for cache_idx, remote_cache_idx in cache_pairs:
                        src_layer_base_addr = self.cpu_kv_caches_base_addr[layer_idx][cache_idx]
                        dst_layer_base_addr = remote_base_addrs[remote_layer_idx][remote_cache_idx]
                        block_len = self.cpu_block_len_per_addr[layer_idx][cache_idx]
                        block_stride = self.cpu_block_stride_per_addr[layer_idx][cache_idx]
                        remote_block_stride = remote_block_stride_per_addr[remote_layer_idx][remote_cache_idx]
                        inner_block_len = block_len // num_group_pulls
                        transfer_remote_block_ids, transfer_local_block_ids = split_if_not_byte_contiguous(
                            grouped_remote_block_ids,
                            grouped_local_block_ids,
                            src_block_stride=remote_block_stride,
                            dst_block_stride=block_stride,
                            block_len=inner_block_len,
                        )
                        for remote_block_id, local_block_id in zip(transfer_remote_block_ids, transfer_local_block_ids):
                            src_list.append(src_layer_base_addr + local_block_id[0] * block_stride)
                            dst_list.append(dst_layer_base_addr + remote_block_id[0] * remote_block_stride)
                            length_list.append(inner_block_len * len(local_block_id))

            if src_list:
                _bt_start = time.perf_counter()
                ret = self.engine.batch_transfer_sync_read(session_id, src_list, dst_list, length_list)
                _bt_end = time.perf_counter()
                logger.info(
                    "[batch_transfer_sync_read] elapsed: %.6f s, src_list length: %d, total bytes: %d",
                    _bt_end - _bt_start,
                    len(src_list),
                    sum(length_list),
                )
                if ret < 0:
                    raise RuntimeError(f"D2RH hop1 transfer failed, ret: {ret}")
            self._send_done_recv_signal(
                remote_request_id,
                port_host,
                remote_handshake_port,
                remote_port_send_num,
            )

    def _get_remote_metadata(self, remote_host: str, remote_handshake_port: int) -> None:
        sock: zmq.Socket | None = None  # type: ignore[name-defined]
        try:
            sock = self._get_remote_socket(remote_host, remote_handshake_port)
            ensure_zmq_send(sock, self.encoder.encode((GET_META_MSG, "")), f"{remote_host}:{remote_handshake_port}")
            metadata_bytes = self._recv_from_socket(sock, f"{remote_host}:{remote_handshake_port}")
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
        sock: zmq.Socket | None = None  # type: ignore[name-defined]
        try:
            sock = self._get_remote_socket(remote_host, remote_handshake_port)
            remote_path = f"{remote_host}:{remote_handshake_port}"
            ensure_zmq_send(
                sock,
                self.encoder.encode((DONE_RECVING_MSG, request_id, remote_port_send_num or {})),
                remote_path,
            )
            resp = self._recv_from_socket(sock, remote_path)
            if resp != b"ACK":
                raise RuntimeError(f"Failed to receive ACK, resp: {resp.decode('utf-8')}")
        finally:
            if sock is not None:
                self._return_remote_socket(sock, remote_host, remote_handshake_port)

    def send_pull_done(self, request_id: str) -> None:
        sock: zmq.Socket | None = None  # type: ignore[name-defined]
        try:
            sock = self._get_remote_socket(self.host_ip, self.scheduler_ready_port)
            scheduler_path = f"{self.host_ip}:{self.scheduler_ready_port}"
            ensure_zmq_send(sock, self.encoder.encode((READY_SCHEDULER, request_id)), scheduler_path)
            resp = self._recv_from_socket(sock, scheduler_path)
            if resp != b"ACK":
                raise RuntimeError(f"Failed to receive ACK, resp: {resp.decode('utf-8')}")
        finally:
            if sock is not None:
                self._return_remote_socket(sock, self.host_ip, self.scheduler_ready_port)

    def _get_remote_socket(self, remote_host: str, remote_handshake_port: int) -> zmq.Socket:  # type: ignore[name-defined]
        remote_path = make_zmq_path("tcp", remote_host, remote_handshake_port)
        with self.remote_sockets_lock:
            if self.remote_sockets[remote_path]:
                return self.remote_sockets[remote_path].popleft()
            ctx = zmq.Context()  # type: ignore
            sock = make_zmq_socket(ctx=ctx, path=remote_path, socket_type=zmq.REQ, bind=False)  # type: ignore
            sock.setsockopt(zmq.SNDTIMEO, int(self.timeout * 1000))  # type: ignore
            return sock

    def _recv_from_socket(self, sock: zmq.Socket, path: str) -> bytes:  # type: ignore[name-defined]
        # zmq.Poller is not thread-safe. A request owns its checked-out REQ
        # socket, so a short-lived poller makes parallel hop-1 workers safe.
        poller = zmq.Poller()  # type: ignore
        poller.register(sock, zmq.POLLIN)  # type: ignore
        try:
            return ensure_zmq_recv(sock, poller, path, timeout=self.timeout)
        finally:
            poller.unregister(sock)

    def _return_remote_socket(
        self,
        sock: zmq.Socket,  # type: ignore[name-defined]
        remote_host: str,
        remote_handshake_port: int,
    ) -> None:
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
        self._h2d_remote_request_ids: dict[str, str] = {}
        self.log_full_block_map = bool(
            self.vllm_config.kv_transfer_config.get_from_extra_config("d2rh_log_full_block_map", True)
        )
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
        if self.log_full_block_map:
            logger.info("[H2D Thread] block_map: %s", block_map)
        else:
            logger.debug("[H2D Thread] block_map entries=%d", len(block_map))

        remote_block_ids: BlockIds = req_meta["remote_block_ids"]
        group_pulls: list[GroupPull] = req_meta.get("group_pulls", [])
        offset_by_group: dict[int, int] = {
            group_pull.group_id: group_pull.remote_tp_offset for group_pull in group_pulls
        }
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
        self._transfer_staged_kv_cache_all_groups(cpu_req_meta)

    def _transfer_staged_kv_cache_all_groups(self, req_meta: dict[str, Any]):
        """Handle a KV cache transfer request."""
        remote_request_id = req_meta["remote_request_id"]
        local_block_ids: BlockIds = req_meta["local_block_ids"]
        remote_block_ids: BlockIds = req_meta["remote_block_ids"]
        local_block_ids_replicate_k: BlockIds = req_meta.get("local_block_ids_replicate_k", tuple())
        remote_block_ids_replicate_k: BlockIds = req_meta.get("remote_block_ids_replicate_k", tuple())
        has_replicate_k_blocks = any(local_block_ids_replicate_k) and any(remote_block_ids_replicate_k)
        group_pulls: list[GroupPull] = req_meta["group_pulls"]
        remote_engine_id = req_meta["remote_engine_id"]
        remote_host = req_meta["remote_host"]
        remote_handshake_port = req_meta["remote_handshake_port"]
        # A full prefix hit only requires notifying the P worker.
        num_local_blocks = sum(len(group_block_ids) for group_block_ids in local_block_ids)
        if num_local_blocks == 0 and not has_replicate_k_blocks:
            return

        with self.remote_metadata_lock:
            has_remote_metadata = (
                remote_engine_id in self.kv_caches_base_addr
                and remote_handshake_port in self.kv_caches_base_addr[remote_engine_id]
            )
        if not has_remote_metadata:
            self._get_remote_metadata(remote_host, remote_handshake_port)
        with self.remote_metadata_lock:
            remote_kv_caches_base_addrs = self.kv_caches_base_addr[remote_engine_id][remote_handshake_port]
            local_kv_caches_base_addrs = self.kv_caches_base_addr[self.local_engine_id][self.local_handshake_port]
            remote_transfer_port = self.remote_te_port[remote_engine_id][remote_handshake_port]
            remote_block_stride_per_addr = self.remote_block_stride_per_addr[remote_engine_id][remote_handshake_port]
            remote_kv_group2layeridx = self.remote_kv_group2layeridx.get(remote_engine_id, {}).get(
                remote_handshake_port,
                self.kv_group2layeridx,
            )
        remote_layer_name_to_idx = build_layer_name_to_metadata_idx(remote_kv_group2layeridx)
        remote_cache_slots = build_layer_name_to_cache_slots(remote_kv_group2layeridx)
        session_id = f"{remote_host}:{remote_transfer_port}"

        req_start_time = time.perf_counter()
        src_list: list[int] = []
        dst_list: list[int] = []
        length_list: list[int] = []
        attention_group_reformat_block_ids: list[tuple[tuple[int, list[list[int]], int, list[int]], bool]] = []
        grouped_remote_k_block_ids: list[list[int]] = []
        grouped_local_k_block_ids: list[list[int]] = []
        if has_replicate_k_blocks:
            grouped_remote_k_block_ids, grouped_local_k_block_ids = base_group_concurrent_contiguous(
                remote_block_ids_replicate_k[0],
                local_block_ids_replicate_k[0],
            )

        def pp_layer_indices(layer_indices: list[int], prefill_pp_rank: int, group_spec: dict[str, Any]) -> list[int]:
            first_layer_index, end_layer_index = self.pp_layer_indices[prefill_pp_rank]
            if self.vllm_config.speculative_config is not None and prefill_pp_rank == self._prefill_pp_size - 1:
                end_layer_index += self.num_draft_layers
            is_index_cache_plane = group_spec.get("kv_cache_spec_type") == "AscendSFAIndexerCacheSpec"

            def in_partition(metadata_layer_idx: int) -> bool:
                transformer_layer = (
                    metadata_layer_idx - self.index_cache_plane_base
                    if is_index_cache_plane and metadata_layer_idx >= self.index_cache_plane_base
                    else metadata_layer_idx
                )
                return first_layer_index <= transformer_layer < end_layer_index

            return [layer_idx for layer_idx in layer_indices if in_partition(layer_idx)]

        use_transfer_group_block_ids = transfer_groups_need_independent_block_ids(
            self.kv_group2layeridx,
            self.block_size_scale,
        )

        def get_remote_layer_idx(
            local_layer_idx: int,
            group_spec: dict[str, Any],
            local_layer_indices: list[int],
        ) -> int:
            # Older peers and lightweight tests may not provide layer names in
            # their cache metadata. Identical layouts remain position-compatible.
            if not remote_layer_name_to_idx or not group_spec.get("layer_names"):
                return local_layer_idx
            return resolve_remote_layer_idx(
                local_layer_idx,
                group_spec,
                local_layer_indices,
                remote_layer_name_to_idx,
            )

        for group_pull in group_pulls:
            group_idx = group_pull.group_id
            group_spec, layer_indices = self.kv_group2layeridx[group_idx]
            kv_cache_group_id = group_spec.get("kv_cache_group_id", group_idx)
            raw_layer_indices = layer_indices
            # Preserve the original name/index alignment for slot resolution,
            # but transfer and reformat each metadata layer only once.
            layer_indices = list(dict.fromkeys(pp_layer_indices(layer_indices, group_pull.prefill_pp_rank, group_spec)))

            if not layer_indices:
                continue
            tp_num_need_pulls = group_pull.num_group_pulls
            inner_offset = group_pull.remote_tp_offset
            is_mamba_group = group_spec["kv_cache_spec_type"] == "MambaSpec"
            block_id_idx = group_idx if use_transfer_group_block_ids else kv_cache_group_id
            local_group_block_ids = local_block_ids[block_id_idx]
            remote_group_block_ids = remote_block_ids[block_id_idx]
            has_group_blocks = bool(local_group_block_ids)
            if not has_group_blocks and (is_mamba_group or not has_replicate_k_blocks):
                continue
            if not is_mamba_group:
                grouped_remote_block_ids: list[list[int]] = []
                grouped_local_block_ids: list[list[int]] = []
                if has_group_blocks:
                    is_group_transfer_end = group_pull.is_group_transfer_end
                    # Block IDs are expanded to kernel granularity and truncated
                    # in _get_kv_split_metadata.
                    kernel_remote_block_ids = remote_group_block_ids
                    kernel_local_block_ids = local_group_block_ids

                    if tp_num_need_pulls == 1:
                        grouped_remote_block_ids, grouped_local_block_ids = base_group_concurrent_contiguous(
                            kernel_remote_block_ids, kernel_local_block_ids
                        )
                    else:
                        grouped_remote_block_ids = [[block_id] for block_id in kernel_remote_block_ids]
                        grouped_local_block_ids = [[block_id] for block_id in kernel_local_block_ids]
                    attention_group_reformat_block_ids.append(
                        (
                            (group_idx, grouped_local_block_ids, tp_num_need_pulls, layer_indices),
                            is_group_transfer_end,
                        )
                    )
            else:
                # Ascend Hybrid Mamba supports "align" (prefix caching) and
                # "none" (no prefix caching), but not "all".
                if self.mamba_cache_mode == "align":
                    if len(remote_group_block_ids) != 1:
                        raise RuntimeError(
                            "Mooncake Mamba transfer requires exactly one normalized remote state block; "
                            f"request_id={remote_request_id}, group_idx={group_idx}, "
                            f"remote_block_count={len(remote_group_block_ids)}, "
                            f"local_block_count={len(local_group_block_ids)}."
                        )
                    remote_state_block_id = remote_group_block_ids[0]
                else:
                    transfer_block_idx = len(remote_group_block_ids) - self.num_speculative_tokens - 1
                    if transfer_block_idx < 0:
                        raise RuntimeError(
                            "Invalid non-aligned Mamba state block metadata: "
                            f"request_id={remote_request_id}, group_idx={group_idx}, "
                            f"remote_block_count={len(remote_group_block_ids)}, "
                            f"num_speculative_tokens={self.num_speculative_tokens}."
                        )
                    remote_state_block_id = remote_group_block_ids[transfer_block_idx]
                grouped_remote_block_ids = [[remote_state_block_id]]
                grouped_local_block_ids = [[local_group_block_ids[0]]]

            if is_mamba_group:
                for layer_idx in layer_indices:
                    remote_layer_idx = get_remote_layer_idx(
                        layer_idx,
                        group_spec,
                        raw_layer_indices,
                    )
                    start_meta_idx = len(src_list)
                    self._append_mamba_transfer_meta(
                        src_list,
                        dst_list,
                        length_list,
                        group_spec=group_spec,
                        src_layer_base_addr=local_kv_caches_base_addrs[layer_idx],
                        dst_layer_base_addr=remote_kv_caches_base_addrs[remote_layer_idx],
                        block_len=self.block_len_per_addr[layer_idx],
                        block_stride=self.block_stride_per_addr[layer_idx],
                        remote_block_stride=remote_block_stride_per_addr[remote_layer_idx],
                        remote_block_id=grouped_remote_block_ids[0][0],
                        local_block_id=grouped_local_block_ids[0][0],
                        tp_num_need_pulls=tp_num_need_pulls,
                        remote_tp_offset=inner_offset,
                    )
                    if logger.isEnabledFor(logging.DEBUG):
                        for src, dst, length in zip(
                            src_list[start_meta_idx:], dst_list[start_meta_idx:], length_list[start_meta_idx:]
                        ):
                            logger.debug(
                                "Mooncake mamba transfer meta: request_id=%s group_idx=%s layer_idx=%s "
                                "local_block_id=%s remote_block_id=%s tp_num_need_pulls=%s "
                                "remote_tp_offset=%s  session_id=%s",
                                remote_request_id,
                                group_idx,
                                layer_idx,
                                grouped_local_block_ids[0][0],
                                grouped_remote_block_ids[0][0],
                                tp_num_need_pulls,
                                inner_offset,
                                session_id,
                            )
                continue

            for layer_idx in layer_indices:
                remote_layer_idx = get_remote_layer_idx(
                    layer_idx,
                    group_spec,
                    raw_layer_indices,
                )
                cache_pairs = resolve_group_cache_slot_pairs(
                    group_spec,
                    raw_layer_indices,
                    layer_idx,
                    remote_cache_slots,
                    len(local_kv_caches_base_addrs[layer_idx]),
                    len(remote_kv_caches_base_addrs[remote_layer_idx]),
                )
                cache_pairs = _get_non_redundant_cache_slot_pairs(
                    cache_pairs,
                    local_kv_caches_base_addrs[layer_idx],
                    remote_kv_caches_base_addrs[remote_layer_idx],
                    self.block_len_per_addr[layer_idx],
                    self.block_stride_per_addr[layer_idx],
                    remote_block_stride_per_addr[remote_layer_idx],
                    tp_num_need_pulls,
                )
                for cache_idx, remote_cache_idx in cache_pairs:
                    src_layer_base_addr = local_kv_caches_base_addrs[layer_idx][cache_idx]
                    dst_layer_base_addr = remote_kv_caches_base_addrs[remote_layer_idx][remote_cache_idx]
                    block_len = self.block_len_per_addr[layer_idx][cache_idx]
                    block_stride = self.block_stride_per_addr[layer_idx][cache_idx]
                    remote_block_stride = remote_block_stride_per_addr[remote_layer_idx][remote_cache_idx]
                    inner_block_len = block_len // tp_num_need_pulls
                    is_sfa_indexer_group = group_spec["kv_cache_spec_type"] == "AscendSFAIndexerCacheSpec"
                    if is_sfa_indexer_group and has_replicate_k_blocks:
                        transfer_remote_block_ids = grouped_remote_k_block_ids
                        transfer_local_block_ids = grouped_local_k_block_ids
                    else:
                        if not has_group_blocks:
                            continue
                        transfer_remote_block_ids, transfer_local_block_ids = split_if_not_byte_contiguous(
                            grouped_remote_block_ids,
                            grouped_local_block_ids,
                            src_block_stride=remote_block_stride,
                            dst_block_stride=block_stride,
                            block_len=inner_block_len,
                        )
                    for remote_block_id, local_block_id in zip(transfer_remote_block_ids, transfer_local_block_ids):
                        src = src_layer_base_addr + local_block_id[0] * block_stride + inner_offset * inner_block_len
                        dst = dst_layer_base_addr + remote_block_id[0] * remote_block_stride
                        length = inner_block_len * len(local_block_id)
                        src_list.append(src)
                        dst_list.append(dst)
                        length_list.append(length)
                    logger.debug(
                        "Mooncake kv transfer meta: request_id=%s group_idx=%s layer_idx=%s local_block_ids=%s "
                        "remote_block_ids=%s tp_num_need_pulls=%s remote_tp_offset=%s session_id=%s",
                        remote_request_id,
                        group_idx,
                        layer_idx,
                        grouped_local_block_ids,
                        grouped_remote_block_ids,
                        tp_num_need_pulls,
                        inner_offset,
                        session_id,
                    )
        if not src_list:
            return

        logger.debug(
            "Mooncake transfer request=%s session id=%s src=%s dst=%s length=%s",
            remote_request_id,
            session_id,
            src_list,
            dst_list,
            length_list,
        )
        transfer_start_time = time.perf_counter()
        ret = self.engine.batch_transfer_sync_read(session_id, src_list, dst_list, length_list)
        if ret < 0:
            logger.error(
                "Mooncake transfer failed for request. remote_request_id=%s, ret=%d. ",
                req_meta["remote_request_id"],
                ret,
            )
            raise RuntimeError(f"Mooncake transfer failed, ret: {ret}")

        req_end_time = time.perf_counter()
        req_transfer_elapsed = (req_end_time - req_start_time) * 1000
        logger.info(
            "KV cache transfer for request %s took %.2f ms. local_ip %s local_device_id %s remote_session_id %s "
            "plan_ms=%.2f engine_ms=%.2f descriptors=%d bytes=%d",
            remote_request_id,
            req_transfer_elapsed,
            get_ip(),
            self.tp_rank,
            session_id,
            (transfer_start_time - req_start_time) * 1000,
            (req_end_time - transfer_start_time) * 1000,
            len(src_list),
            sum(length_list),
        )

        ready_attention_group_reformat_block_ids = []
        for reformat_group, is_group_transfer_end in attention_group_reformat_block_ids:
            if is_group_transfer_end:
                ready_attention_group_reformat_block_ids.append(reformat_group)
        if ready_attention_group_reformat_block_ids:
            shard_idx = int(req_meta.get("shard_idx", 0))
            self._stash_pending_reformat(
                req_meta["request_id"],
                shard_idx,
                ready_attention_group_reformat_block_ids,
            )

    def _handle_request(self, req_meta: dict[str, Any]) -> None:
        self._h2d_remote_request_ids[req_meta["request_id"]] = req_meta["remote_request_id"]
        super()._handle_request(req_meta)

    def _send_done_recv_signal(
        self,
        request_id: str,
        remote_host: str,
        remote_handshake_port: int,
        remote_port_send_num: dict[int, RemotePortInfo],
    ) -> None:
        # D2RHThread already acknowledged the P-to-Host transfer. H2D reads
        # only the staged copy, so it must not complete the P request again.
        # Preserve base cleanup for CP peers that did not participate in a
        # pull and therefore received no first-hop completion notification.
        if remote_port_send_num and remote_port_send_num[remote_handshake_port]["num"] == 0:
            super()._send_done_recv_signal(request_id, remote_host, remote_handshake_port, remote_port_send_num)

    def _mark_request_task_done(self, request_id: str, all_task_done: bool) -> bool:
        # all_task_done marks the last SUBMITTED shard, not the last completed
        # transfer. Different PP peers run concurrently and may finish out of
        # order. Keep Host blocks pinned until the base completion counter says
        # every H2D read has returned, including on a failed-transfer path.
        completed = super()._mark_request_task_done(request_id, all_task_done)
        if completed:
            remote_request_id = self._h2d_remote_request_ids.pop(request_id, request_id)
            block_map = self.remote_local_block_map.pop(remote_request_id, None)
            local_block_map = self.remote_local_block_map.pop(request_id, None)
            if block_map is None:
                block_map = local_block_map
            if block_map:
                self.cpu_kvcache_manager.free_block_map(block_map)
        return completed


class MooncakeConnectorScheduler(BaseMooncakeConnectorScheduler):
    def __init__(self, vllm_config: VllmConfig, engine_id: str, kv_cache_config: KVCacheConfig):
        # DeepSeek-V4 exposes a logical scheduler block size that differs from
        # the generic Mooncake V1 default. Keep this compatibility adjustment
        # local to D2RH instead of changing every Mooncake V1 consumer.
        refresh_block_size(vllm_config)
        super().__init__(vllm_config, engine_id, kv_cache_config)
        self.block_size = self._get_scheduler_block_size()
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
        self.remote_sockets: dict[str, deque[zmq.Socket]] = defaultdict(deque)  # type: ignore[name-defined]
        self.remote_poller = zmq.Poller()  # type: ignore
        self.timeout = 1.0
        self.remote_sockets_lock = threading.Lock()
        self.local_host = get_ip()
        self.host_cache_hash_source = str(
            vllm_config.kv_transfer_config.get_from_extra_config("d2rh_host_cache_hash_source", "prefill")
        )
        if self.host_cache_hash_source not in ("prefill", "decode"):
            raise ValueError(
                f"d2rh_host_cache_hash_source must be 'prefill' or 'decode', got {self.host_cache_hash_source!r}"
            )
        # Use the same hash granularity as Request, not the transfer page size.
        # Older runtimes without the resolver retain the token-chain fallback.
        self._decode_hash_block_size: int | None = None
        resolve_block_sizes = getattr(kv_cache_utils, "resolve_kv_cache_block_sizes", None)
        if (
            self.kv_role == "kv_consumer"
            and self.host_cache_hash_source == "decode"
            and resolve_block_sizes is not None
        ):
            _, self._decode_hash_block_size = resolve_block_sizes(kv_cache_config, vllm_config)
        if self.kv_role == "kv_consumer":
            self.all_requests: set[str] = set()
            self.listeningthread = HostListeningThread(
                self.all_requests,
                self._decode_tp_size,
                get_scheduler_ready_zmq_port(vllm_config),
            )
            self.listeningthread.start()

    def _get_scheduler_block_size(self) -> int:
        for group in self.kv_cache_config.kv_cache_groups:
            for spec in self._get_group_unique_specs(group):
                if isinstance(spec, AscendSlidingWindowMLASpec) and spec.model_version == "deepseek_v4":
                    return spec.block_size
        return self.vllm_config.cache_config.block_size

    def _get_group_transfer_info(self, group: Any) -> GroupTransferInfo:
        specs = self._get_group_unique_specs(group)
        if specs and all(
            isinstance(spec, AscendMLAAttentionSpec) and spec.model_version == "deepseek_v4" for spec in specs
        ):
            # DeepSeek-V4 specs already express block_size in logical tokens;
            # storage_block_size accounts for compression inside the page.
            # Multiplying by compress_ratio again drops most prompt pages and
            # leaves their D-side destinations with stale KV after reuse.
            block_sizes = {spec.block_size for spec in specs}
            if len(block_sizes) != 1:
                raise ValueError("D2RH compressed KV group has inconsistent logical block sizes")
            return GroupTransferInfo(
                tokens_per_block=block_sizes.pop(),
                blocks_per_window=0,
                is_state_group=False,
            )
        return super()._get_group_transfer_info(group)

    def _d2rh_prefix_fingerprint(self, request: "Request", end_token: int) -> bytes:
        """Stable-within-engine fingerprint for KV content through end_token.

        Request.block_hashes is a chained hash, so the final complete hash
        already represents every complete token block before it.  Hash the
        remaining partial-block tokens explicitly to cover the prompt tail.
        """
        hash_block_size = int(getattr(self.vllm_config.cache_config, "hash_block_size", None) or self.block_size)
        complete_hashes = min(end_token // hash_block_size, len(request.block_hashes))
        digest = hashlib.sha256()
        digest.update(b"d2rh-host-cache-v1")
        digest.update(struct.pack(">Q", end_token))
        if complete_hashes:
            digest.update(bytes(request.block_hashes[complete_hashes - 1]))
        token_ids = request.prompt_token_ids or []
        for token_id in token_ids[complete_hashes * hash_block_size : end_token]:
            digest.update(struct.pack(">q", int(token_id)))
        return digest.digest()

    def _d2rh_get_transfer_block_hashes(
        self,
        request: "Request",
        block_ids: BlockIds,
        remote_block_ids: BlockIds,
    ) -> tuple[list[str | None], ...]:
        """Build hashes aligned exactly with request_finished remote blocks."""
        prompt_len = len(request.prompt_token_ids or [])
        cp_size = max(1, self.pcp_size * self.dcp_size)
        result: list[list[str | None]] = []
        for group_id, (blocks, group_info) in enumerate(zip(block_ids, self.group_transfer_info)):
            if group_info.is_state_group:
                state_hash = self._d2rh_prefix_fingerprint(request, prompt_len)
                pairs = [
                    (block_id, hashlib.sha256(state_hash + struct.pack(">I", idx)).digest())
                    for idx, block_id in enumerate(blocks)
                ]
            else:
                tokens_per_scheduler_block = group_info.tokens_per_block * cp_size
                num_prompt_blocks = math.ceil(prompt_len / tokens_per_scheduler_block)
                pairs = [
                    (
                        block_id,
                        self._d2rh_prefix_fingerprint(
                            request,
                            min((idx + 1) * tokens_per_scheduler_block, prompt_len),
                        ),
                    )
                    for idx, block_id in enumerate(blocks[:num_prompt_blocks])
                ]
                if group_info.blocks_per_window:
                    pairs = pairs[-group_info.blocks_per_window :]
                    pairs = [pair for pair in pairs if pair[0] != 0]

            expected_ids = list(remote_block_ids[group_id])
            if [pair[0] for pair in pairs] != expected_ids:
                logger.warning(
                    "D2RH host cache hash alignment mismatch group=%d expected=%d actual=%d; disabling cache for group",
                    group_id,
                    len(expected_ids),
                    len(pairs),
                )
                result.append([None] * len(expected_ids))
            else:
                # KV transfer parameters cross the OpenAI HTTP boundary, so
                # keep them JSON serializable. D workers decode hex to bytes.
                result.append([pair[1].hex() for pair in pairs])
        return tuple(result)

    def _d2rh_get_decode_endpoint_hashes(
        self,
        request: "Request",
        remote_block_ids: BlockIds,
    ) -> tuple[list[str | None], ...] | None:
        """Reuse Request's chained hashes at only the transferred endpoints.

        Each complete hash covers the entire preceding prefix, including cache
        identity such as salt. Include partial hash-block tokens explicitly.
        The separate namespace prevents mixing these keys with legacy chains.
        """
        hash_block_size = getattr(self, "_decode_hash_block_size", None)
        block_hashes = getattr(request, "block_hashes", ())
        tokens = request.prompt_token_ids or []
        if not hash_block_size or not block_hashes or len(block_hashes) < len(tokens) // hash_block_size:
            return None
        cp_size = max(1, self.pcp_size * self.dcp_size)
        result: list[list[str | None]] = []
        for group_id, (remote_ids, info) in enumerate(zip(remote_block_ids, self.group_transfer_info)):
            count = len(remote_ids)
            if not count:
                result.append([])
                continue
            tokens_per_block = max(1, len(tokens)) if info.is_state_group else info.tokens_per_block * cp_size
            total_blocks = math.ceil(len(tokens) / tokens_per_block)
            if not info.is_state_group and count > total_blocks:
                result.append([None] * count)
                continue
            first = total_blocks - count if info.blocks_per_window and not info.is_state_group else 0
            group_hashes: list[str | None] = []
            for index in range(count):
                end = len(tokens) if info.is_state_group else min((first + index + 1) * tokens_per_block, len(tokens))
                complete = end // hash_block_size
                # Without one complete Request hash, retain the established
                # fallback rather than inventing an incomplete cache identity.
                if not complete:
                    return None
                digest = hashlib.sha256(b"d2rh-decode-host-cache-v2" + struct.pack(">IQ", group_id, end))
                digest.update(bytes(block_hashes[complete - 1]))
                for token_id in tokens[complete * hash_block_size : end]:
                    digest.update(struct.pack(">q", int(token_id)))
                if info.is_state_group:
                    digest.update(struct.pack(">I", index))
                group_hashes.append(digest.hexdigest())
            result.append(group_hashes)
        return tuple(result)

    def _d2rh_get_decode_block_hashes(
        self,
        request: "Request",
        remote_block_ids: BlockIds,
    ) -> tuple[list[str | None], ...]:
        """Hash prompt prefixes locally on D, aligned to P's remote blocks.

        This avoids returning thousands of hash strings in the P HTTP
        response. The chained digest preserves correctness: a block can only
        hit when every token through that block endpoint is identical.
        """
        endpoint_hashes = self._d2rh_get_decode_endpoint_hashes(request, remote_block_ids)
        if endpoint_hashes is not None:
            return endpoint_hashes
        token_ids = request.prompt_token_ids or []
        prompt_len = len(token_ids)
        token_bytes = np.asarray(token_ids, dtype=np.int64).tobytes()
        cp_size = max(1, self.pcp_size * self.dcp_size)
        result: list[list[str | None]] = []

        for group_id, (remote_group_ids, group_info) in enumerate(zip(remote_block_ids, self.group_transfer_info)):
            if group_info.is_state_group:
                tokens_per_block = max(1, prompt_len)
                total_blocks = 1
            else:
                tokens_per_block = group_info.tokens_per_block * cp_size
                total_blocks = math.ceil(prompt_len / tokens_per_block)

            digest = hashlib.sha256(b"d2rh-decode-host-cache-v1" + struct.pack(">I", group_id)).digest()
            prefix_hashes: list[str | None] = []
            for block_idx in range(total_blocks):
                start = block_idx * tokens_per_block
                end = min((block_idx + 1) * tokens_per_block, prompt_len)
                digest = hashlib.sha256(digest + token_bytes[start * 8 : end * 8]).digest()
                prefix_hashes.append(digest.hex())

            num_remote_blocks = len(remote_group_ids)
            group_hashes: list[str | None]
            if group_info.is_state_group:
                final_hash = prefix_hashes[-1] if prefix_hashes else None
                group_hashes = [
                    hashlib.sha256(bytes.fromhex(final_hash) + struct.pack(">I", idx)).hexdigest()
                    if final_hash is not None
                    else None
                    for idx in range(num_remote_blocks)
                ]
            elif num_remote_blocks > len(prefix_hashes):
                logger.warning(
                    "D2RH decode hash alignment mismatch group=%d remote=%d available=%d; disabling cache for group",
                    group_id,
                    num_remote_blocks,
                    len(prefix_hashes),
                )
                group_hashes = [None] * num_remote_blocks
            elif group_info.blocks_per_window:
                group_hashes = prefix_hashes[-num_remote_blocks:]
            else:
                group_hashes = prefix_hashes[:num_remote_blocks]
            result.append(group_hashes)
        return tuple(result)

    def request_finished(
        self,
        request: "Request",
        block_ids: BlockIds,
    ) -> tuple[bool, dict[str, Any] | None]:
        delay_free, params = super().request_finished(request, block_ids)
        if self.host_cache_hash_source == "prefill" and params is not None:
            remote_block_ids: BlockIds = params["remote_block_ids"]
            params["d2rh_block_hashes"] = self._d2rh_get_transfer_block_hashes(
                request,
                block_ids,
                remote_block_ids,
            )
        return delay_free, params

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
        sock: zmq.Socket | None = None  # type: ignore[name-defined]
        reusable = False
        try:
            sock = self._get_remote_socket(self.local_host, d2rh_port)
            d2rh_path = f"{self.local_host}:{d2rh_port}"
            ensure_zmq_send(sock, self.encoder.encode((START_PULL, request_id, params)), d2rh_path)
            response = ensure_zmq_recv(sock, self.remote_poller, d2rh_path, timeout=self.timeout)
            reusable = True
            return response
        finally:
            if sock is not None:
                if reusable:
                    self._return_remote_socket(sock, self.local_host, d2rh_port)
                else:
                    self._discard_remote_socket(sock)

    def _get_remote_socket(self, remote_host: str, remote_handshake_port: int) -> zmq.Socket:  # type: ignore[name-defined]
        remote_path = make_zmq_path("tcp", remote_host, remote_handshake_port)
        with self.remote_sockets_lock:
            if self.remote_sockets[remote_path]:
                return self.remote_sockets[remote_path].popleft()
            ctx = zmq.Context()  # type: ignore
            sock = make_zmq_socket(ctx=ctx, path=remote_path, socket_type=zmq.REQ, bind=False)  # type: ignore
            sock.setsockopt(zmq.SNDTIMEO, int(self.timeout * 1000))  # type: ignore
            self.remote_poller.register(sock, zmq.POLLIN)  # type: ignore
            return sock

    def _return_remote_socket(
        self,
        sock: zmq.Socket,  # type: ignore[name-defined]
        remote_host: str,
        remote_handshake_port: int,
    ) -> None:
        remote_path = make_zmq_path("tcp", remote_host, remote_handshake_port)
        with self.remote_sockets_lock:
            self.remote_sockets[remote_path].append(sock)

    def _discard_remote_socket(self, sock: zmq.Socket) -> None:  # type: ignore[name-defined]
        """Drop a REQ socket that did not receive its matching REP."""
        with contextlib.suppress(KeyError, zmq.ZMQError):  # type: ignore[attr-defined]
            self.remote_poller.unregister(sock)
        sock.close(linger=0)

    def get_num_new_matched_tokens(self, request: "Request", num_computed_tokens: int) -> tuple[int, bool]:
        params = request.kv_transfer_params
        if params is not None and params.get("do_remote_prefill"):
            # START_PULL serializes a copy of params. Publish the D-local
            # prefix hit before that message is built and sent.
            params["num_computed_tokens"] = num_computed_tokens
            if request.request_id not in self.all_requests:
                block_hashes = request.block_hashes
                first_hash = bytes(block_hashes[0]).hex()[:16] if block_hashes else "none"
                probe_idx = min(len(block_hashes) - 1, int(len(block_hashes) * 0.90))
                probe_hash = bytes(block_hashes[probe_idx]).hex()[:16] if probe_idx >= 0 else "none"
                logger.info(
                    "D2RH_HBM_QUERY request_id=%s local_hit_tokens=%d request_hashes=%d hash0=%s hash90=%s",
                    request.request_id,
                    num_computed_tokens,
                    len(block_hashes),
                    first_hash,
                    probe_hash,
                )
                got_staging_full = False
                decode_block_hashes = None
                if self.host_cache_hash_source == "decode":
                    decode_block_hashes = self._d2rh_get_decode_block_hashes(
                        request,
                        tuple(params.get("remote_block_ids") or ()),
                    )
                for decode_tp_rank in range(self._decode_tp_size):
                    d2rh_port = get_d2rh_zmq_port(self.vllm_config, decode_tp_rank)
                    pull_params = self._build_start_pull_params(request.request_id, params, decode_tp_rank)
                    if decode_block_hashes is not None:
                        pull_params["d2rh_block_hashes"] = decode_block_hashes
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
        self._is_hma_required: bool = bool(getattr(self, "_is_hma_required", False))
        self.remote_local_block_map: dict[str, dict[tuple[int, ...], int]] = {}
        self.d2rh_thread: D2RHThread | None = None

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        """Register the KV Cache data."""
        self.use_mla = self.vllm_config.model_config.is_deepseek_mla
        self.use_sparse = hasattr(self.vllm_config.model_config.hf_text_config, "index_topk")
        self.enable_sfa_dcp_replicated_indexer = enable_sfa_dcp_replicated_indexer(self.vllm_config)

        self.num_blocks = self.kv_cache_config.num_blocks
        logger.info("num_blocks: %s", self.num_blocks)
        self.kv_caches = kv_caches
        # Maps each KV cache group to its serialized group spec and physical
        # layer indices: {group_id: (group_spec, [layer_idx0, layer_idx1, ...])}.
        self.kv_group2layeridx = self._build_kv_group2layeridx()
        self._is_hma_required = self._is_hma_required or self._requires_group_aware_attention_transfer()
        layer_name_to_idx = {
            layer_name: layer_idx
            for _, (group_spec, layer_indices) in self.kv_group2layeridx.items()
            for layer_name, layer_idx in zip(group_spec["layer_names"], layer_indices)
        }
        metadata_layers = max(layer_name_to_idx.values(), default=-1) + 1
        # Per-layer registered KV cache base addresses:
        # [layer_idx][cache_idx] -> data_ptr of one cache tensor, e.g. K/V.
        self.kv_caches_base_addr: list[list[int]] = [[] for _ in range(metadata_layers)]
        # Per-layer block scaling between logical KV blocks and tensor blocks:
        # [layer_idx][cache_idx] -> cache tensor num_blocks / logical num_blocks.
        self.block_size_scale: list[list[int]] = [[] for _ in range(metadata_layers)]
        # Per-layer byte length of one tensor block:
        # [layer_idx][cache_idx] -> element_size * prod(block_shape).
        self.block_len_per_addr: list[list[int]] = [[] for _ in range(metadata_layers)]
        # Per-layer full tensor shape for each registered KV cache address:
        # [layer_idx][cache_idx] -> cache tensor shape, including num_blocks.
        self.block_shape_per_addr: list[list[int]] = [[] for _ in range(metadata_layers)]
        # Per-layer byte stride between consecutive tensor blocks:
        # [layer_idx][cache_idx] -> stride(0) * element_size.
        self.block_stride_per_addr: list[list[int]] = [[] for _ in range(metadata_layers)]

        # TODO: For DSV4 use_compress, metadata/transfer can be optimized by
        # aggregating layer views that share the same raw KVCacheTensor.
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

        register_regions = self._get_register_regions(kv_caches)
        validate_register_region_count(register_regions)
        global_te.register_buffer(register_regions.ptrs, register_regions.lengths)

        logger.debug(
            "Mooncake register kv caches metadata: kv_group2layeridx=%s, kv_caches_base_addr=%s, "
            "block_len_per_addr=%s, block_stride_per_addr=%s, block_shape_per_addr=%s, "
            "block_size_scale=%s, ptrs=%s, lengths=%s",
            self.kv_group2layeridx,
            self.kv_caches_base_addr,
            self.block_len_per_addr,
            self.block_stride_per_addr,
            self.block_shape_per_addr,
            self.block_size_scale,
            register_regions.ptrs,
            register_regions.lengths,
        )
        # Start the role-specific transfer thread after cache registration.
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
            handshake_port=self.handshake_port,
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
            self.kv_recv_thread = self._create_recv_thread(ready_event)
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

    @staticmethod
    def _tensor_span_end(tensor: torch.Tensor) -> int:
        """Return the exclusive end address of a possibly-strided tensor."""
        if tensor.numel() == 0:
            return tensor.data_ptr()
        span = tensor.element_size()
        for size, stride in zip(tensor.shape, tensor.stride()):
            span += (size - 1) * abs(stride) * tensor.element_size()
        return tensor.data_ptr() + span

    def _get_storage_merged_device_regions(self, kv_caches: dict[str, torch.Tensor]) -> RegisterRegions:
        """Register each device backing storage once, while retaining logical views."""
        regions_by_storage: OrderedDict[int, tuple[int, int]] = OrderedDict()
        logical_tensor_count = 0
        logical_total_bytes = 0
        for kv_cache_tuple in kv_caches.values():
            for cache in self._as_kv_cache_tuple(kv_cache_tuple):
                if cache.numel() == 0:
                    continue
                storage = cache.untyped_storage()
                storage_base = storage.data_ptr()
                storage_end = storage_base + storage.nbytes()
                aligned_base = (storage_base + HUGEPAGE_SIZE_2M - 1) // HUGEPAGE_SIZE_2M * HUGEPAGE_SIZE_2M
                # NPU KV allocations have padding before their first aligned
                # view. Tiny CPU tensors used by unit tests do not.
                region_start = aligned_base if aligned_base <= cache.data_ptr() else cache.data_ptr()
                tensor_end = self._tensor_span_end(cache)
                if not (storage_base <= region_start <= cache.data_ptr() and tensor_end <= storage_end):
                    raise RuntimeError(
                        "Unable to recover one device KV registration region: "
                        f"tensor=[{cache.data_ptr()}, {tensor_end}), "
                        f"storage=[{storage_base}, {storage_end})."
                    )
                previous = regions_by_storage.get(storage_base)
                if previous is None:
                    regions_by_storage[storage_base] = (region_start, tensor_end)
                else:
                    regions_by_storage[storage_base] = (min(previous[0], region_start), max(previous[1], tensor_end))
                logical_tensor_count += 1
                logical_total_bytes += cache.nbytes

        return RegisterRegions(
            ptrs=[start for start, _ in regions_by_storage.values()],
            lengths=[end - start for start, end in regions_by_storage.values()],
            logical_tensor_count=logical_tensor_count,
            logical_total_bytes=logical_total_bytes,
        )

    def _make_cpu_staging_caches(self, kv_caches: dict[str, torch.Tensor]) -> dict[str, list[torch.Tensor]]:
        """Create pinned staging, preserving aliases within each named cache.

        Different names can belong to independent cache groups and use different
        staging block IDs. Never merge their allocations, even if HBM is aliased.
        """
        cpu_caches: dict[str, list[torch.Tensor]] = {}
        cache_layout: list[tuple[str, torch.Tensor, int]] = []
        # TP/CP shards can pack subviews differently. Preserve their established
        # independent staging layout until alias-compatible sharding is defined.
        preserve_aliases = (
            self._prefill_tp_size == self.tp_size == 1
            and getattr(self, "pcp_size", 1) == 1
            and getattr(self, "dcp_size", 1) == 1
        )
        arena_size = 0
        for layer_name, kv_cache_tuple in kv_caches.items():
            cpu_caches[layer_name] = []
            caches = self._as_kv_cache_tuple(kv_cache_tuple)
            if not preserve_aliases:
                for cache in caches:
                    cache_layout.append((layer_name, cache, arena_size))
                    raw_size = cache.numel() * cache.element_size()
                    arena_size += ((raw_size + HUGEPAGE_SIZE_2M - 1) // HUGEPAGE_SIZE_2M) * HUGEPAGE_SIZE_2M
                continue
            regions: dict[int, list[tuple[int, int, int]]] = defaultdict(list)
            for index, cache in enumerate(caches):
                regions[cache.untyped_storage().data_ptr()].append(
                    (cache.data_ptr(), self._tensor_span_end(cache), index)
                )
            offsets = {}
            for intervals in regions.values():
                merged: list[tuple[int, int, list[int]]] = []
                for start, end, index in sorted(intervals):
                    if merged and start < merged[-1][1]:
                        previous_start, previous_end, indices = merged[-1]
                        merged[-1] = (previous_start, max(previous_end, end), indices + [index])
                    else:
                        merged.append((start, end, [index]))
                for start, end, indices in merged:
                    alignment = max(caches[index].element_size() for index in indices)
                    start -= start % alignment
                    for index in indices:
                        offsets[index] = arena_size + caches[index].data_ptr() - start
                    raw_size = end - start
                    aligned_size = ((raw_size + HUGEPAGE_SIZE_2M - 1) // HUGEPAGE_SIZE_2M) * HUGEPAGE_SIZE_2M
                    arena_size += aligned_size
            for index, cache in enumerate(caches):
                cache_layout.append((layer_name, cache, offsets[index]))

        self._cpu_register_ptrs = []
        self._cpu_register_lengths = []
        if arena_size == 0:
            self.cpu_caches_hold = []
            return cpu_caches

        arena_owner = torch.empty(
            arena_size + HUGEPAGE_SIZE_2M - 1,
            dtype=torch.uint8,
            device="cpu",
            pin_memory=True,
        )
        alignment_offset = (-arena_owner.data_ptr()) % HUGEPAGE_SIZE_2M
        arena = arena_owner.narrow(0, alignment_offset, arena_size)
        self.cpu_caches_hold = [arena_owner]
        self._cpu_register_ptrs = [arena.data_ptr()]
        self._cpu_register_lengths = [arena_size]

        for layer_name, cache, offset in cache_layout:
            if preserve_aliases:
                storage_offset = (alignment_offset + offset) // cache.element_size()
                view = arena.view(cache.dtype).as_strided(cache.shape, cache.stride(), storage_offset)
            else:
                raw_size = cache.numel() * cache.element_size()
                view = arena.narrow(0, offset, raw_size).view(cache.dtype).view(cache.shape)
            cpu_caches[layer_name].append(view)
        return cpu_caches

    def _get_register_regions(self, kv_caches: dict[str, torch.Tensor]) -> RegisterRegions:
        register_regions = self._get_storage_merged_device_regions(kv_caches)
        device_region_count = len(register_regions.ptrs)
        # V1 has already flattened the HBM tensors. Record which slots belong
        # to each named cache, including distinct components of one PP layer.
        next_slot: dict[int, int] = defaultdict(int)
        layer_cache_indices: dict[str, list[int]] = {}
        layer_name_to_idx = build_layer_name_to_metadata_idx(self.kv_group2layeridx)
        for name, caches in kv_caches.items():
            layer_idx = layer_name_to_idx[name]
            start = next_slot[layer_idx]
            end = start + len(self._as_kv_cache_tuple(caches))
            layer_cache_indices[name] = list(range(start, end))
            next_slot[layer_idx] = end
        for group_spec, _ in self.kv_group2layeridx.values():
            group_spec["layer_cache_indices"] = {name: layer_cache_indices[name] for name in group_spec["layer_names"]}

        if self.kv_role == "kv_consumer":
            self.cpu_kvcache_manager = D2RHCPUCacheManager(self.num_blocks)
            logger.info("D2RH host content cache capacity_blocks=%d", self.num_blocks)
            cpu_caches = self._make_cpu_staging_caches(kv_caches)
            metadata_layers = len(self.kv_caches_base_addr)
            self.cpu_kv_caches_base_addr: list[list[int]] = [[] for _ in range(metadata_layers)]
            self.cpu_block_len_per_addr: list[list[int]] = [[] for _ in range(metadata_layers)]
            self.cpu_block_stride_per_addr: list[list[int]] = [[] for _ in range(metadata_layers)]
            self.cpu_block_size_scale: list[list[int]] = [[] for _ in range(metadata_layers)]
            for name, caches in cpu_caches.items():
                layer_idx = layer_name_to_idx[name]
                for cache in caches:
                    self.cpu_kv_caches_base_addr[layer_idx].append(cache.data_ptr())
                    self.cpu_block_len_per_addr[layer_idx].append(cache.element_size() * math.prod(cache.shape[1:]))
                    self.cpu_block_stride_per_addr[layer_idx].append(cache.stride(0) * cache.element_size())
                    self.cpu_block_size_scale[layer_idx].append(cache.shape[0] // self.num_blocks)
            # Register the owning 2M-aligned buffers, not just their tensor views.
            register_regions.ptrs.extend(self._cpu_register_ptrs)
            register_regions.lengths.extend(self._cpu_register_lengths)
        logger.info(
            "D2RH register regions: device=%d host=%d total=%d registered_bytes=%d",
            device_region_count,
            len(register_regions.ptrs) - device_region_count,
            len(register_regions.ptrs),
            register_regions.registered_bytes,
        )
        return register_regions

    def _create_recv_thread(self, ready_event: threading.Event) -> KVCacheRecvingThread:
        self.d2rh_thread = D2RHThread(
            cpu_kv_caches_base_addr=self.cpu_kv_caches_base_addr,
            cpu_block_len_per_addr=self.cpu_block_len_per_addr,
            cpu_block_stride_per_addr=self.cpu_block_stride_per_addr,
            cpu_block_size_scale=self.cpu_block_size_scale,
            cpu_kvcache_manager=self.cpu_kvcache_manager,
            remote_local_block_map=self.remote_local_block_map,
            kv_group2layeridx=self.kv_group2layeridx,
            engine=self.engine,
            vllm_config=self.vllm_config,
            d2rh_handshake_port=get_d2rh_zmq_port(self.vllm_config, self.tp_rank, self.pp_rank, self.pcp_rank),
            scheduler_ready_port=get_scheduler_ready_zmq_port(self.vllm_config),
            tp_rank=self.tp_rank,
        )
        self.d2rh_thread.start()
        return KVCacheRecvingThread(
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
            cpu_kv_caches_base_addr=self.cpu_kv_caches_base_addr,
            cpu_block_len_per_addr=self.cpu_block_len_per_addr,
            cpu_block_stride_per_addr=self.cpu_block_stride_per_addr,
            cpu_block_size_scale=self.cpu_block_size_scale,
            cpu_kvcache_manager=self.cpu_kvcache_manager,
            remote_local_block_map=self.remote_local_block_map,
            cpu_te_rpc_port=self.te_rpc_port,
        )


class MooncakeConnector(BaseMooncakeConnector):
    """V1 connector interface with D Host staging scheduler/worker extensions."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: KVCacheConfig | None = None,
    ):
        assert vllm_config.kv_transfer_config is not None
        self._kv_transfer_config = vllm_config.kv_transfer_config
        self.engine_id = self._kv_transfer_config.engine_id
        self._connector_metadata = MooncakeConnectorMetadata()
        self.connector_scheduler: MooncakeConnectorScheduler | None
        self.connector_worker: MooncakeConnectorWorker | None

        if role == KVConnectorRole.SCHEDULER:
            assert kv_cache_config is not None
            self.connector_scheduler = MooncakeConnectorScheduler(
                vllm_config,
                str(self.engine_id),
                kv_cache_config,
            )
            self.connector_worker = None
        elif role == KVConnectorRole.WORKER:
            assert kv_cache_config is not None
            self.connector_scheduler = None
            self.connector_worker = MooncakeConnectorWorker(
                vllm_config,
                str(self.engine_id),
                kv_cache_config,
            )


# Preserve the configured connector name for external plugin loading.
MooncakeD2RHConnector = MooncakeConnector
