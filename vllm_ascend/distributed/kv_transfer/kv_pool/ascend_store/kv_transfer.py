from __future__ import annotations

import ctypes
import queue
import threading
import time
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import torch
from vllm.distributed.kv_events import BlockStored
from vllm.logger import logger
from vllm.v1.core.kv_cache_utils import maybe_convert_block_hash

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.attention_fence import AttentionComputeStartGate
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.base import Backend
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.memcache_backend import (
    MemcacheBackend,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import (
    ChunkedTokenDatabase,
    KVCacheByteRange,
    ReqMeta,
    get_block_hashes,
)

_TRANSFER_SHUTDOWN_POLL_S = 0.1


_MLA_READ_DEDUP_TARGET_BYTES = 64 * 1024 * 1024
_MLA_READ_DEDUP_MIN_BYTES = 8 * 1024 * 1024


def _should_use_mla_read_dedup(total_bytes: int, tp_size: int) -> bool:
    """Avoid fixed collective/copy overhead for small replicated KV loads."""
    threshold = max(_MLA_READ_DEDUP_MIN_BYTES, _MLA_READ_DEDUP_TARGET_BYTES // max(tp_size, 1))
    return total_bytes >= threshold


def _kv_shard_indices(num_keys: int, world_size: int, tp_rank: int) -> list[int]:
    base, remainder = divmod(num_keys, world_size)
    start = tp_rank * base + min(tp_rank, remainder)
    end = start + base + (1 if tp_rank < remainder else 0)
    return list(range(start, end))


def _kv_shard_lengths(sizes: list[list[int]], world_size: int) -> list[int]:
    return [
        sum(sum(sizes[index]) for index in _kv_shard_indices(len(sizes), world_size, owner))
        for owner in range(world_size)
    ]


def _block_ids_are_contiguous(block_ids: list[int], indices: list[int]) -> bool:
    return len(indices) <= 1 or all(
        block_ids[indices[index + 1]] == block_ids[indices[index]] + 1 for index in range(len(indices) - 1)
    )


def _kv_byte_range(view: torch.Tensor | KVCacheByteRange) -> KVCacheByteRange:
    if isinstance(view, KVCacheByteRange):
        return view
    if not isinstance(view, torch.Tensor) or view.dtype != torch.int8 or not view.is_contiguous():
        raise RuntimeError("KV read deduplication requires contiguous int8 cache views")
    return KVCacheByteRange(view.untyped_storage(), view.storage_offset(), view.numel(), view.device)


def _same_kv_storage(first: KVCacheByteRange, second: KVCacheByteRange) -> bool:
    return first.device == second.device and (
        first.storage is second.storage or first.storage.data_ptr() == second.storage.data_ptr()
    )


def _coalesced_kv_views(
    views: Sequence[Sequence[torch.Tensor | KVCacheByteRange]], indices: list[int], *, assume_contiguous: bool = False
) -> list[torch.Tensor]:
    """Merge layer-major storage spans before materializing exchange tensors."""
    max_views = max((len(views[index]) for index in indices), default=0)
    if assume_contiguous and indices:
        first_key, last_key = indices[0], indices[-1]
        view_count = len(views[first_key])
        if view_count and all(len(views[index]) == view_count for index in indices):
            fast_ranges = []
            for view_index in range(view_count):
                first = _kv_byte_range(views[first_key][view_index])
                last = _kv_byte_range(views[last_key][view_index])
                total_bytes = first.size * len(indices)
                if (
                    first.size == 0
                    or first.size != last.size
                    or not _same_kv_storage(first, last)
                    or last.offset + last.size != first.offset + total_bytes
                ):
                    break
                # Block IDs are a hint: verify interior storage and partial ranges too.
                if any(
                    span.size != first.size
                    or span.offset != first.offset + position * first.size
                    or not _same_kv_storage(first, span)
                    for position, key_index in enumerate(indices)
                    for span in (_kv_byte_range(views[key_index][view_index]),)
                ):
                    break
                fast_ranges.append(KVCacheByteRange(first.storage, first.offset, total_bytes, first.device))
            else:
                return [span.as_tensor() for span in fast_ranges]

    coalesced = []
    for view_index in range(max_views):
        current = None
        for key_index in indices:
            if view_index >= len(views[key_index]):
                continue
            span = _kv_byte_range(views[key_index][view_index])
            if span.size == 0:
                continue
            if current is not None and _same_kv_storage(current, span) and current.offset + current.size == span.offset:
                current = KVCacheByteRange(current.storage, current.offset, current.size + span.size, current.device)
                continue
            if current is not None:
                coalesced.append(current.as_tensor())
            current = span
        if current is not None:
            coalesced.append(current.as_tensor())
    return coalesced


def _exchange_kv_shards(
    tp_group: Any,
    tp_rank: int,
    local_views: list[torch.Tensor],
    owner_views: dict[int, list[torch.Tensor]],
) -> None:
    exchange = getattr(tp_group, "exchange_kv_shards", None)
    if callable(exchange):
        exchange(tp_rank, local_views, owner_views)
        return

    ops = []
    for owner, recv_views in owner_views.items():
        if owner == tp_rank:
            continue
        global_rank = tp_group.ranks[owner]
        for view in local_views:
            ops.append(torch.distributed.P2POp(torch.distributed.isend, view, global_rank, group=tp_group.device_group))
        for view in recv_views:
            ops.append(torch.distributed.P2POp(torch.distributed.irecv, view, global_rank, group=tp_group.device_group))
    handles = torch.distributed.batch_isend_irecv(ops)
    for handle in handles:
        handle.wait()


_KV_STATUS_BITS_PER_BYTE = 8


def _all_reduce_kv_status(tp_group: Any, status: torch.Tensor) -> torch.Tensor:
    all_reduce = getattr(tp_group, "all_reduce_kv_status", None)
    if callable(all_reduce):
        return all_reduce(status)
    reduced = status.clone()
    # Each key belongs to one rank, so bitmap bits are disjoint across ranks.
    # SUM is equivalent to bitwise OR and cannot overflow a uint8 status byte.
    torch.distributed.all_reduce(reduced, group=tp_group.cpu_group)
    return reduced


def _all_gather_kv_shards(
    tp_group: Any,
    tp_rank: int,
    local_views: list[torch.Tensor],
    owner_views: dict[int, list[torch.Tensor]],
    shard_lengths: list[int],
    local_bytes: int,
    status_values: list[int],
    device: torch.device,
) -> list[int]:
    wire_bytes = max(shard_lengths, default=0)
    status_bytes = len(status_values)
    packet_bytes = wire_bytes + status_bytes
    payload = torch.empty(packet_bytes, dtype=torch.int8, device=device)

    destinations = []
    cursor = 0
    for view in local_views:
        end = cursor + view.numel()
        destinations.append(payload[cursor:end])
        cursor = end
    if cursor != local_bytes:
        raise RuntimeError(f"KV shard packing mismatch: expected={local_bytes} actual={cursor}")
    if destinations:
        torch._foreach_copy_(destinations, local_views)
    if local_bytes < wire_bytes:
        payload[local_bytes:wire_bytes].zero_()

    status = torch.tensor(status_values, dtype=torch.uint8, device=device)
    payload[wire_bytes:].copy_(status.view(torch.int8))
    gathered = tp_group.all_gather(payload, dim=0).flatten()
    expected_numel = packet_bytes * tp_group.world_size
    if gathered.numel() != expected_numel:
        raise RuntimeError(f"KV all-gather size mismatch: expected={expected_numel} actual={gathered.numel()}")

    destinations = []
    sources = []
    for owner, recv_views in owner_views.items():
        shard_base = owner * packet_bytes
        cursor = 0
        for view in recv_views:
            end = cursor + view.numel()
            destinations.append(view)
            sources.append(gathered[shard_base + cursor : shard_base + end])
            cursor = end
        if cursor != shard_lengths[owner]:
            raise RuntimeError(
                f"KV all-gather unpack mismatch for rank {owner}: expected={shard_lengths[owner]} actual={cursor}"
            )
    if destinations:
        torch._foreach_copy_(destinations, sources)

    # Compact the status tails locally so all owners share one device-to-host copy.
    owner_status = gathered.reshape(tp_group.world_size, packet_bytes)[:, wire_bytes:].contiguous().cpu()
    status_array = owner_status.numpy().view(np.uint8)
    return np.bitwise_or.reduce(status_array, axis=0).tolist()


_P2P_EXCHANGE_VIEW_LIMIT = 4
_P2P_EXCHANGE_MIN_VIEW_BYTES = 8 * 1024 * 1024
_P2P_EXCHANGE_LARGE_VIEW_LIMIT = 64


def _should_use_p2p_kv_exchange(world_size: int, total_views: int, total_bytes: int) -> bool:
    """Prefer direct cache writes for few views or large TP=2 spans."""
    if total_views <= _P2P_EXCHANGE_VIEW_LIMIT:
        return True
    # With one peer, large contiguous spans amortize per-view P2P overhead
    # while avoiding the all-gather pack and copyback of the full KV payload.
    # Bound the operation count: many large spans still incur costly P2P launches.
    return (
        world_size == 2
        and total_views <= _P2P_EXCHANGE_LARGE_VIEW_LIMIT
        and total_bytes >= total_views * _P2P_EXCHANGE_MIN_VIEW_BYTES
    )


def sharded_get_all_gather(
    m_store: Backend,
    tp_group: Any,
    tp_rank: int,
    keys: list[str],
    addrs: list[list[int]],
    sizes: list[list[int]],
    block_ids: list[int],
    views: Sequence[Sequence[torch.Tensor | KVCacheByteRange]],
) -> tuple[bool, set[int], int]:
    """Load a TP-disjoint KV shard and all-gather it into every local cache."""
    if not (len(keys) == len(addrs) == len(sizes) == len(block_ids) == len(views)):
        raise ValueError("KV shard metadata lengths do not match")
    if not keys:
        return False, set(), 0

    device = None
    for key_views in views:
        if key_views:
            device = key_views[0].device
            break
    if device is None:
        raise RuntimeError("KV read deduplication requires tensor views for all gathered ranges")

    world_size = tp_group.world_size
    local_indices = _kv_shard_indices(len(keys), world_size, tp_rank)
    local_keys = [keys[index] for index in local_indices]
    local_addrs = [addrs[index] for index in local_indices]
    local_sizes = [sizes[index] for index in local_indices]
    local_block_ids = [block_ids[index] for index in local_indices]
    shard_lengths = _kv_shard_lengths(sizes, world_size)
    local_bytes = shard_lengths[tp_rank]
    wire_bytes = max(shard_lengths, default=0)
    status_byte_count = (len(keys) + _KV_STATUS_BITS_PER_BYTE - 1) // _KV_STATUS_BITS_PER_BYTE

    ret = None
    local_missing: set[int] = set()
    local_failed = False
    if local_keys:
        try:
            ret = m_store.get(local_keys, local_addrs, local_sizes)
        except Exception:
            logger.exception("KV read deduplication shard get failed")
            ret = None
        if ret is None or len(ret) != len(local_keys):
            local_failed = isinstance(m_store, MemcacheBackend)
            local_missing.update(local_block_ids)
        else:
            local_missing.update(record_failed_blocks(local_block_ids, list(ret)))
            if isinstance(m_store, MemcacheBackend) and local_missing:
                local_failed = True

    missing_bytes = [0] * status_byte_count
    failure_bytes = [0] * status_byte_count
    for index in local_indices:
        if block_ids[index] in local_missing:
            byte_index, bit_index = divmod(index, _KV_STATUS_BITS_PER_BYTE)
            missing_bytes[byte_index] |= 1 << bit_index
    if local_failed:
        for index in local_indices:
            byte_index, bit_index = divmod(index, _KV_STATUS_BITS_PER_BYTE)
            missing_bytes[byte_index] |= 1 << bit_index
            failure_bytes[byte_index] |= 1 << bit_index

    if wire_bytes:
        # Keep read ownership unchanged, but use the same physical block order
        # for packing and receiving so reversed/reused allocations still coalesce.
        local_view_indices = sorted(local_indices, key=block_ids.__getitem__)
        local_views = _coalesced_kv_views(
            views,
            local_view_indices,
            assume_contiguous=_block_ids_are_contiguous(block_ids, local_view_indices),
        )
        actual_local_bytes = sum(view.numel() for view in local_views)
        if actual_local_bytes != local_bytes:
            raise RuntimeError(f"KV shard packing mismatch: expected={local_bytes} actual={actual_local_bytes}")
        owner_views: dict[int, list[torch.Tensor]] = {}
        for owner in range(world_size):
            if owner == tp_rank:
                continue
            owner_indices = sorted(_kv_shard_indices(len(keys), world_size, owner), key=block_ids.__getitem__)
            owner_view_list = _coalesced_kv_views(
                views,
                owner_indices,
                assume_contiguous=_block_ids_are_contiguous(block_ids, owner_indices),
            )
            owner_bytes = sum(view.numel() for view in owner_view_list)
            if owner_bytes != shard_lengths[owner]:
                raise RuntimeError(
                    f"KV P2P shard size mismatch for rank {owner}: expected={shard_lengths[owner]} actual={owner_bytes}"
                )
            owner_views[owner] = owner_view_list
        status_values = missing_bytes + failure_bytes
        total_views = len(local_views) + sum(len(owner_view_list) for owner_view_list in owner_views.values())
        if _should_use_p2p_kv_exchange(world_size, total_views, sum(shard_lengths)):
            status = torch.tensor(status_values, dtype=torch.uint8, device="cpu")
            status_values = _all_reduce_kv_status(tp_group, status).tolist()
            _exchange_kv_shards(tp_group, tp_rank, local_views, owner_views)
        else:
            status_values = _all_gather_kv_shards(
                tp_group,
                tp_rank,
                local_views,
                owner_views,
                shard_lengths,
                local_bytes,
                status_values,
                device,
            )
    else:
        status = torch.tensor(missing_bytes + failure_bytes, dtype=torch.uint8, device="cpu")
        status_values = _all_reduce_kv_status(tp_group, status).tolist()
    aggregated_missing = status_values[:status_byte_count]
    aggregated_failure = status_values[status_byte_count:]

    request_failed = any(aggregated_failure)
    missing_block_ids: set[int] = set()
    for byte_index, byte in enumerate(aggregated_missing):
        for bit_index in range(_KV_STATUS_BITS_PER_BYTE):
            index = byte_index * _KV_STATUS_BITS_PER_BYTE + bit_index
            if index >= len(keys):
                break
            if byte & (1 << bit_index):
                missing_block_ids.add(block_ids[index])
    return request_failed, missing_block_ids, len(local_keys)


class KVTransferThread(threading.Thread):
    _invalid_block_ids: set[int]
    _invalid_block_ids_lock: threading.Lock

    def __init__(
        self,
        m_store: Backend,
        token_database: ChunkedTokenDatabase,
        block_size: int | list[int],
        tp_rank: int,
        tp_size: int = 1,
        dcp_size: int = 1,
        ready_event: threading.Event | None = None,
        name: str = "KVTransferThread",
    ):
        super().__init__(daemon=True, name=name)
        self.m_store = m_store
        self.ready_event = ready_event or threading.Event()
        self.block_size = block_size
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self.dcp_size = dcp_size
        self.token_database = token_database
        self.num_addrs_per_block = len(token_database.group_block_len[0])
        self.done_task_lock = threading.Lock()
        # Serialize request submission with the transition to a fatal state.
        # Otherwise a producer can enqueue immediately after the failed
        # consumer has drained the queue, leaving work with no live consumer.
        self._request_submit_lock = threading.Lock()
        self._stopping = threading.Event()
        self.request_queue: queue.Queue[Any] = queue.Queue()
        self.finished_requests: set[str] = set()
        self.failed_requests: set[str] = set()
        self.kv_event_lock = threading.Lock()
        self.kv_events: list[BlockStored] = []
        self._fatal_error: BaseException | None = None
        self.save_disabled = threading.Event()
        self.load_disabled = threading.Event()

    def _mark_failed_requests(self, requests: list[ReqMeta]) -> None:
        logger.error("Memcache transfer failed for requests=%s", [request.req_id for request in requests])
        for request in requests:
            request.can_save = False
        with self.done_task_lock:
            self.failed_requests.update(request.req_id for request in requests if len(request.block_ids_by_group) > 1)
        self.save_disabled.set()
        with self._invalid_block_ids_lock:
            self._invalid_block_ids.update(
                block_id
                for request in requests
                if len(request.block_ids_by_group) == 1
                for group in request.block_ids_by_group
                for block_id in group
                if block_id > 0
            )

    def get_and_clear_failed_requests(self) -> set[str]:
        with self.done_task_lock:
            failed = self.failed_requests.copy()
            self.failed_requests.clear()
        return failed

    def _get_block_size(self, kv_cache_group_id: int = 0) -> int:
        if isinstance(self.block_size, list):
            if kv_cache_group_id >= len(self.block_size):
                return self.block_size[0]
            return self.block_size[kv_cache_group_id]
        return self.block_size

    def add_request(self, request: Any) -> None:
        with self._request_submit_lock:
            if self._stopping.is_set():
                return
            if self._fatal_error is not None:
                logger.error("Reject KV transfer submission to failed thread %s: %s", self.name, self._fatal_error)
                return
            self.request_queue.put(request)

    def stop(self) -> None:
        with self._request_submit_lock:
            if not self._stopping.is_set():
                self._stopping.set()
                if self.is_alive():
                    self.request_queue.put(None)

    def _wait_for_event(
        self,
        event: threading.Event | AttentionComputeStartGate,
        failure_checker: Callable[[], None] | None = None,
    ) -> bool:
        while not self._stopping.is_set():
            if event.wait(timeout=_TRANSFER_SHUTDOWN_POLL_S):
                return not self._stopping.is_set()
            if failure_checker is not None:
                failure_checker()
        return False

    def get_and_clear_finished_requests(
        self,
        req_ids: set[str] | None = None,
    ) -> set[str]:
        """
        Get and clear the requests that have been completed.
        Returns:
            A set of request IDs that have been completed.
        """
        with self.done_task_lock:
            if req_ids is None:
                finished_requests = self.finished_requests.copy()
                self.finished_requests.clear()
            else:
                finished_requests = self.finished_requests & req_ids
                self.finished_requests -= finished_requests
        return finished_requests

    def discard_finished_requests(self, req_ids: set[str]) -> None:
        with self.done_task_lock:
            self.finished_requests -= req_ids

    def raise_if_failed(self) -> None:
        if self._fatal_error is not None:
            raise RuntimeError(f"{self.name} failed during asynchronous transfer") from self._fatal_error

    def set_finished_request(self, req_id):
        with self.done_task_lock:
            self.finished_requests.add(req_id)

    @staticmethod
    def _split_transfer_packets(
        gvas: np.ndarray,
        addrs: np.ndarray,
        sizes: np.ndarray,
        max_transfer_bytes: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if max_transfer_bytes <= 0:
            return gvas, addrs, sizes

        split_counts: np.ndarray = (sizes + max_transfer_bytes - 1) // max_transfer_bytes
        total_splits = int(split_counts.sum())
        if total_splits == sizes.shape[0]:
            return gvas, addrs, sizes

        split_indices: np.ndarray = np.arange(int(split_counts.max()), dtype=np.int64)
        split_mask = split_indices[:, None] < split_counts[None, :]
        entry_indices = np.broadcast_to(
            np.arange(sizes.shape[0], dtype=np.int64),
            split_mask.shape,
        )[split_mask]
        transfer_offsets = np.broadcast_to(
            split_indices[:, None] * max_transfer_bytes,
            split_mask.shape,
        )[split_mask]

        split_gvas = gvas[entry_indices] + transfer_offsets
        split_addrs = addrs[entry_indices] + transfer_offsets
        split_sizes = np.minimum(
            max_transfer_bytes,
            sizes[entry_indices] - transfer_offsets,
        )
        return split_gvas, split_addrs, split_sizes

    def _batch_copy_with_limits(
        self,
        gvas: np.ndarray,
        addrs: np.ndarray,
        sizes: np.ndarray,
        direction: int,
        max_transfer_blocks: int,
        max_transfer_bytes: int,
    ) -> int:
        if len(gvas) == 0:
            return 0

        # direction: 0/SMEMB_COPY_L2G = save (write), 1/SMEMB_COPY_G2L = load (read)
        dir_name = "save(L2G)" if direction == 0 else "load(G2L)" if direction == 1 else f"dir{direction}"
        logger.debug(
            "[KVPOOL] batch_copy %s gvas=%d total_bytes=%d",
            dir_name,
            len(gvas),
            int(sizes.sum()) if len(sizes) else 0,
        )

        max_transfer_addrs = 0
        if max_transfer_blocks > 0:
            max_transfer_addrs = max_transfer_blocks * self.num_addrs_per_block
        if max_transfer_addrs <= 0:
            max_transfer_addrs = len(gvas)

        assert self.m_store.store is not None
        for start in range(0, len(gvas), max_transfer_addrs):
            end = start + max_transfer_addrs
            split_gvas, split_addrs, split_sizes = self._split_transfer_packets(
                gvas[start:end],
                addrs[start:end],
                sizes[start:end],
                max_transfer_bytes,
            )
            logger.debug(
                "[KVPOOL] batch_copy %s split_gvas=%s split_sizes=%s",
                dir_name,
                split_gvas.tolist(),
                split_sizes.tolist(),
            )
            res = self.m_store.store.batch_copy(
                split_gvas.tolist(),
                split_addrs.tolist(),
                split_sizes.tolist(),
                direction,
            )
            if res != 0:
                logger.error("[KVPOOL] batch_copy %s FAILED res=%d", dir_name, res)
                return res
        return 0

    @staticmethod
    def _range_transfer_batches(
        keys: list[str],
        all_buffers: list[list[int]],
        all_sizes: list[list[int]],
        all_offsets: list[list[int]],
        max_transfer_blocks: int,
        max_transfer_bytes: int,
    ) -> list[tuple[list[str], list[list[int]], list[list[int]], list[list[int]]]]:
        """Split backend range calls by key count and per-segment byte size."""
        row_count = len(keys)
        if not (len(all_buffers) == len(all_sizes) == len(all_offsets) == row_count):
            raise ValueError("Layerwise range metadata must contain one buffer/size/offset row per key")

        # Default configuration (both limits unset) means one batch holding every
        # key and no per-segment splitting, so every row passes through as-is
        # unless it carries an empty or negative segment. Detect that case and
        # hand back the caller's own lists: this runs once per key, per layer,
        # per group, and rebuilding them element-wise was the receive thread's
        # largest single block of exposed time.
        if max_transfer_blocks <= 0 and max_transfer_bytes <= 0:
            for buffers, sizes, offsets in zip(all_buffers, all_sizes, all_offsets, strict=True):
                if not (len(buffers) == len(sizes) == len(offsets)):
                    raise ValueError("Layerwise range rows must align buffers, sizes, and offsets")
                if sizes and min(sizes) >= 0 and 0 not in sizes:
                    continue
                break
            else:
                return [(keys, all_buffers, all_sizes, all_offsets)]

        normalized_buffers: list[list[int]] = []
        normalized_sizes: list[list[int]] = []
        normalized_offsets: list[list[int]] = []
        for buffers, sizes, offsets in zip(all_buffers, all_sizes, all_offsets, strict=True):
            if not (len(buffers) == len(sizes) == len(offsets)):
                raise ValueError("Layerwise range rows must align buffers, sizes, and offsets")
            split_buffers: list[int] = []
            split_sizes: list[int] = []
            split_offsets: list[int] = []
            for buffer, size, offset in zip(buffers, sizes, offsets, strict=True):
                if size < 0:
                    raise ValueError(f"Layerwise range size must be non-negative, got {size}")
                if size == 0:
                    continue
                packet_size = max_transfer_bytes if max_transfer_bytes > 0 else size
                for packet_offset in range(0, size, packet_size):
                    split_buffers.append(buffer + packet_offset)
                    split_sizes.append(min(packet_size, size - packet_offset))
                    split_offsets.append(offset + packet_offset)
            normalized_buffers.append(split_buffers)
            normalized_sizes.append(split_sizes)
            normalized_offsets.append(split_offsets)

        rows_per_batch = max_transfer_blocks if max_transfer_blocks > 0 else max(1, row_count)
        return [
            (
                keys[start:end],
                normalized_buffers[start:end],
                normalized_sizes[start:end],
                normalized_offsets[start:end],
            )
            for start in range(0, row_count, rows_per_batch)
            for end in [min(start + rows_per_batch, row_count)]
        ]

    def _set_os_thread_name(self) -> None:
        try:
            libc = ctypes.CDLL("libc.so.6")
            # Linux task comm is limited to 15 visible bytes plus NUL.
            libc.prctl(15, self.name[:15].encode(), 0, 0, 0)
        except Exception:
            pass

    def run(self):
        """Run the thread to handle KV cache transfer requests."""
        self._set_os_thread_name()
        self.m_store.set_device()
        self.ready_event.set()
        while True:
            request_data = None
            try:
                request_data = self.request_queue.get()
                if request_data is None or self._stopping.is_set():
                    self.request_queue.task_done()
                    if request_data is None:
                        return
                    continue
                self._handle_request(request_data)
            except Exception as e:
                with self._request_submit_lock:
                    self._fatal_error = e
                    try:
                        self._handle_request_exception(request_data)
                    except Exception:
                        logger.exception("Failed to clean up KV transfer request after an unexpected dispatch error")
                logger.error(
                    "Error in KVCacheTransferThread(%s). type=%s, error=%s. Check thread state and request processing.",
                    self.name,
                    type(e).__name__,
                    e,
                )
                return

    def _handle_request(self, req_meta: Any):
        pass

    def _handle_request_exception(self, request_data: Any):
        """Allow subclasses to complete queue/request bookkeeping on errors."""
        pass

    def lookup(
        self,
        keys: list[str],
    ) -> list[bool]:
        """
        Check the existence of all keys from the cache engine.
        :return: A bool list where True means the key exists in store.
        """
        try:
            res = self.m_store.exists(keys)  # type: ignore[assignment]
            exists_list = [False] * len(keys)
            for index, value in enumerate(res):  # type: ignore[arg-type]
                exists_list[index] = value == 1
            return exists_list
        except Exception as e:
            logger.error(
                "Remote connection failed in lookup. type=%s, error=%s. Check network and remote store.",
                type(e).__name__,
                e,
            )
            return [False] * len(keys)

    def _get_missing_indices(self, keys: list[str], require_exists_check: bool = False) -> list[int]:
        """Filter existing keys unless the backend can do so during put.

        Callers that need the exact newly stored key set, such as KV event
        publishers, can force connector-side filtering.
        """
        if not require_exists_check and not self.m_store.requires_exists_before_put:
            return list(range(len(keys)))
        exists_states = self.lookup(keys)
        return [index for index, exists in enumerate(exists_states) if not exists]

    def update_kv_event(self, event: list[BlockStored]):
        with self.kv_event_lock:
            self.kv_events.extend(event)

    def get_kv_events(self) -> list[BlockStored]:
        with self.kv_event_lock:
            events = self.kv_events.copy()
            self.kv_events.clear()
        return events

    @staticmethod
    def _skip_null_blocks(req_meta: ReqMeta, group_id: int, cache_role: str = "kv") -> bool:
        if cache_role != "kv":
            return False
        skip_flags = req_meta.skip_null_blocks_by_group
        return group_id < len(skip_flags) and skip_flags[group_id] if skip_flags else False

    def _prepare_value(
        self,
        start: int,
        end: int,
        block_ids: list[int],
        kv_cache_group_id: int = 0,
        cache_role: str = "kv",
        block_id: int | None = None,
    ):
        try:
            return self.token_database.prepare_value(
                start,
                end,
                block_ids,
                kv_cache_group_id=kv_cache_group_id,
                cache_role=cache_role,
                block_id=block_id,
            )
        except TypeError:
            return self.token_database.prepare_value(start, end, block_ids)

    def _decode_adaptor_prefill_pp(
        self,
        keys: list[str],
        addrs: list[list[int]],
        sizes: list[list[int]],
        kv_cache_group_id: int = 0,
        cache_role: str = "kv",
    ):
        try:
            return self.token_database.decode_adaptor_prefill_pp(
                keys,
                addrs,
                sizes,
                kv_cache_group_id=kv_cache_group_id,
                cache_role=cache_role,
            )
        except TypeError:
            return self.token_database.decode_adaptor_prefill_pp(keys, addrs, sizes)


class KVCacheStoreBatch:
    """FIFO fence for a batch of asynchronous KV store requests."""

    def __init__(self) -> None:
        self.done = threading.Event()


class KVCacheStoreSendingThread(KVTransferThread):
    def __init__(
        self,
        m_store: Backend,
        token_database: ChunkedTokenDatabase,
        block_size: int | list[int],
        tp_rank: int,
        tp_size: int = 1,
        pcp_rank: int = 0,
        pcp_size: int = 1,
        dcp_size: int = 1,
        put_step: int = 1,
        kv_role: str = "kv_producer",
        ready_event: threading.Event | None = None,
        group_uses_align_state: list[bool] | None = None,
        enable_kv_event: bool = False,
        worker: Any = None,
        sharded_mamba_group_ids: set[int] | None = None,
    ):
        super().__init__(
            m_store, token_database, block_size, tp_rank, tp_size, dcp_size, ready_event, name="KVCacheSendingThread"
        )
        self.pcp_rank = pcp_rank
        self.pcp_size = pcp_size
        self.put_step = put_step
        self.kv_role = kv_role
        # Request ids can be reused after preemption, so track each queued
        # store generation independently.
        self.stored_requests: dict[str, set[int]] = {}
        self._next_local_store_job_id = -1
        self.group_uses_align_state = group_uses_align_state or []
        self.sharded_mamba_group_ids = sharded_mamba_group_ids or set()
        self.enable_kv_event = enable_kv_event
        self.completed_events_lock = threading.Lock()
        self.completed_events: dict[int, int] = {}
        self.worker = worker
        # Request-local durable high-water marks. They are advanced only after
        # the backend store succeeds, so a failed range is retried on the next
        # Decode boundary instead of being lost when scheduler metadata moves
        # forward. Keeping this on the worker also avoids rescanning the full
        # prefix on every successful Decode save.
        self._saved_offsets: dict[str, int] = {}
        # Token suffix retained only after a failed store. This lets a retry
        # publish correct KV events without copying the full request history on
        # the normal Decode path.
        self._retry_token_ids: dict[str, tuple[int, list[int]]] = {}

    def lookup_store_keys(self, keys: list[str], kv_cache_group_id: int = 0) -> list[bool] | None:
        """Return existence in the logical Decode-key domain.

        A Decode consumer may expand one logical block into one key per
        Prefill PP rank. The block is reusable only when every expanded key is
        present; checking only ``@pp_rank:0`` would turn a partial PUT into a
        false hit and permanently skip the missing partitions on retry.
        Return None after logging malformed results; callers must fail the job.
        """
        partitions = getattr(self.token_database, "partitions", None)
        if self.kv_role != "kv_consumer" or not keys or partitions is None or len(partitions) <= 1:
            return [bool(value) for value in self.lookup(keys)]

        empty_buffers: list[list[int]] = [[] for _ in keys]
        expanded_keys, _, _ = self._decode_adaptor_prefill_pp(
            keys,
            empty_buffers,
            empty_buffers,
            kv_cache_group_id=kv_cache_group_id,
        )
        if len(expanded_keys) % len(keys) != 0:
            logger.error(
                "Prefill-PP key adaptor expanded %s logical keys to an invalid count %s", len(keys), len(expanded_keys)
            )
            return None
        partitions_per_key = len(expanded_keys) // len(keys)
        if partitions_per_key == 0:
            logger.error("Prefill-PP key adaptor returned no store keys")
            return None
        expanded_exists = self.lookup(expanded_keys)
        if len(expanded_exists) != len(expanded_keys):
            logger.error(
                "KV store returned %s lookup results for %s Prefill-PP keys", len(expanded_exists), len(expanded_keys)
            )
            return None
        return [
            all(expanded_exists[start : start + partitions_per_key])
            for start in range(0, len(expanded_exists), partitions_per_key)
        ]

    def add_stored_request(self, req_meta: ReqMeta) -> None:
        with self.done_task_lock:
            if req_meta.store_job_id is None:
                req_meta.store_job_id = self._next_local_store_job_id
                self._next_local_store_job_id -= 1
            self.finished_requests.discard(req_meta.req_id)
            self.stored_requests.setdefault(req_meta.req_id, set()).add(req_meta.store_job_id)

    def add_save_batch(self, requests: list[ReqMeta]) -> KVCacheStoreBatch:
        """Queue requests followed by a fence that completes after the batch."""
        save_batch = KVCacheStoreBatch()
        # The lock makes registration, queue publication, and the transition
        # to a fatal state mutually exclusive. A failed consumer therefore
        # either drains this complete batch or rejects it before registration.
        with self._request_submit_lock:
            if self._stopping.is_set():
                save_batch.done.set()
                return save_batch
            if self._fatal_error is not None:
                logger.error("Reject KV save batch to failed thread %s: %s", self.name, self._fatal_error)
                # No source KV is read, but completion must release Scheduler pins.
                # Reject without registering jobs, advancing offsets, or emitting BlockStored.
                with self.completed_events_lock:
                    for request in requests:
                        if request.event_id is not None:
                            self.completed_events[request.event_id] = self.completed_events.get(request.event_id, 0) + 1
                save_batch.done.set()
                return save_batch
            for request in requests:
                self.add_stored_request(request)
            for request in requests:
                self.request_queue.put(request)
            self.request_queue.put(save_batch)
        return save_batch

    def is_live_store_job(self, req_meta: ReqMeta) -> bool:
        with self.done_task_lock:
            return req_meta.store_job_id in self.stored_requests.get(req_meta.req_id, ())

    def is_stored_request(self, req_id: str) -> bool:
        with self.done_task_lock:
            return bool(self.stored_requests.get(req_id))

    def get_stored_request_count(self, req_id: str) -> int | None:
        with self.done_task_lock:
            jobs = self.stored_requests.get(req_id)
            return None if jobs is None else len(jobs)

    def get_stored_requests_snapshot(self) -> dict[str, int]:
        with self.done_task_lock:
            return {req_id: len(jobs) for req_id, jobs in self.stored_requests.items()}

    def finish_store_job(self, req_meta: ReqMeta) -> int | None:
        with self.done_task_lock:
            jobs = self.stored_requests.get(req_meta.req_id)
            if jobs is None or req_meta.store_job_id not in jobs:
                return None
            jobs.remove(req_meta.store_job_id)
            remaining = len(jobs)
            if remaining == 0:
                del self.stored_requests[req_meta.req_id]
            return remaining

    def delete_finished_stored_request(self, req_id: str):
        with self.done_task_lock:
            self.stored_requests.pop(req_id, None)

    def get_saved_offset(self, req_id: str) -> int:
        with self.done_task_lock:
            return self._saved_offsets.get(req_id, 0)

    def record_saved_offset(self, req_meta: ReqMeta, token_len: int) -> None:
        with self.done_task_lock:
            if req_meta.store_job_id in self.stored_requests.get(req_meta.req_id, ()):
                self._saved_offsets[req_meta.req_id] = max(self._saved_offsets.get(req_meta.req_id, 0), token_len)

    def reset_saved_request(self, req_id: str) -> None:
        with self.done_task_lock:
            self._saved_offsets.pop(req_id, None)
            self._retry_token_ids.pop(req_id, None)

    def get_event_token_ids(self, req_meta: ReqMeta, start: int, end: int) -> list[int] | None:
        if req_meta.token_ids is None:
            return None
        token_ids_start = req_meta.save_start_token
        token_ids = req_meta.token_ids
        with self.done_task_lock:
            if req_meta.store_job_id not in self.stored_requests.get(req_meta.req_id, ()):
                return []
            retry = self._retry_token_ids.get(req_meta.req_id)
            if retry is not None:
                retry_start, retry_ids = retry
                if retry_start + len(retry_ids) == token_ids_start:
                    token_ids_start = retry_start
                    token_ids = retry_ids + token_ids
        token_ids_end = token_ids_start + len(token_ids)
        if start < token_ids_start or end > token_ids_end:
            return []
        return token_ids[start - token_ids_start : end - token_ids_start]

    def _remember_retry_token_ids(self, req_meta: ReqMeta) -> None:
        if not self.enable_kv_event or req_meta.token_ids is None:
            return
        token_ids_start = req_meta.save_start_token
        token_ids = req_meta.token_ids.copy()
        with self.done_task_lock:
            if req_meta.store_job_id not in self.stored_requests.get(req_meta.req_id, ()):
                return
            retry = self._retry_token_ids.get(req_meta.req_id)
            if retry is not None:
                retry_start, retry_ids = retry
                if retry_start + len(retry_ids) == token_ids_start:
                    token_ids_start = retry_start
                    token_ids = retry_ids + token_ids
            self._retry_token_ids[req_meta.req_id] = (token_ids_start, token_ids)

    def _clear_retry_token_ids(self, req_meta: ReqMeta) -> None:
        with self.done_task_lock:
            if req_meta.store_job_id in self.stored_requests.get(req_meta.req_id, ()):
                self._retry_token_ids.pop(req_meta.req_id, None)

    def get_completed_events(self):
        with self.completed_events_lock:
            if not self.completed_events:
                return None
            completed_events = self.completed_events.copy()
            self.completed_events.clear()
        return completed_events

    def _handle_request_exception(self, request_data: Any):
        def finish_failed_request(failed_request: Any) -> None:
            if isinstance(failed_request, ReqMeta):
                remaining = self.finish_store_job(failed_request)
                if remaining == 0:
                    self.set_finished_request(failed_request.req_id)
                if remaining is not None and failed_request.event_id is not None:
                    with self.completed_events_lock:
                        self.completed_events[failed_request.event_id] = (
                            self.completed_events.get(failed_request.event_id, 0) + 1
                        )
            elif isinstance(failed_request, KVCacheStoreBatch):
                # Complete the batch fence even after a fatal dispatch error.
                # Request fences observe _fatal_error before allowing block reuse.
                failed_request.done.set()

        # run() holds _request_submit_lock, and this thread is the only consumer.
        # unfinished_tasks counts queued items plus the current item if unacknowledged.
        # Equal counts mean the current item was already acknowledged; do not repeat it.
        queued_request_count = self.request_queue.qsize()
        current_needs_ack = self.request_queue.unfinished_tasks > queued_request_count
        if request_data is not None:
            finish_failed_request(request_data)
            if current_needs_ack:
                self.request_queue.task_done()
        # Acknowledge each dequeued item exactly once, including batch fences.
        for _ in range(queued_request_count):
            try:
                queued_request = self.request_queue.get_nowait()
            except queue.Empty:
                break
            finish_failed_request(queued_request)
            self.request_queue.task_done()

    def _handle_request(self, req_meta: ReqMeta | KVCacheStoreBatch):
        if isinstance(req_meta, KVCacheStoreBatch):
            req_meta.done.set()
            self.request_queue.task_done()
            return

        if self.worker is not None and getattr(self.worker, "tp_mismatch", False):
            req_id = req_meta.req_id
            tracked_request = self.is_live_store_job(req_meta)
            try:
                succeeded = self.worker._store_kv_tp_mismatch(req_meta)
                if tracked_request:
                    if succeeded is True:
                        self.record_saved_offset(req_meta, req_meta.token_len_chunk)
                        self._clear_retry_token_ids(req_meta)
                    else:
                        self._remember_retry_token_ids(req_meta)
            except Exception:
                self._remember_retry_token_ids(req_meta)
                logger.exception("Failed to store KV cache for TP-mismatch request %s", req_id)
            finally:
                remaining = self.finish_store_job(req_meta) if tracked_request else None
                if tracked_request and remaining == 0:
                    self.set_finished_request(req_id)
                if req_meta.event_id is not None:
                    with self.completed_events_lock:
                        self.completed_events[req_meta.event_id] = 1
                self.request_queue.task_done()
            return

        req_id = req_meta.req_id
        tracked_request = False
        try:
            tracked_request = self.is_live_store_job(req_meta)
            if not tracked_request:
                return
            # Only explicit success may advance the saved offset; failures return False.
            succeeded = self._handle_stored_request(req_meta)
            if succeeded is True:
                self.record_saved_offset(req_meta, req_meta.token_len_chunk)
                self._clear_retry_token_ids(req_meta)
            else:
                self._remember_retry_token_ids(req_meta)
        except Exception:
            self._remember_retry_token_ids(req_meta)
            logger.exception("Failed to store KV cache for request %s", req_id)
        finally:
            remaining = self.finish_store_job(req_meta) if tracked_request else None
            if tracked_request and remaining == 0:
                self.set_finished_request(req_id)
            if req_meta.event_id is not None:
                with self.completed_events_lock:
                    self.completed_events[req_meta.event_id] = 1
            self.request_queue.task_done()

    def _handle_stored_request(self, req_meta: ReqMeta) -> bool:
        """Store missing KV chunks; log validation/PUT failures and return False."""
        if req_meta.can_save is False:
            return False
        token_len = req_meta.token_len_chunk
        req_id = req_meta.req_id
        current_event = req_meta.current_event
        try:
            store_masks = self.token_database.store_mask(token_len, req_meta.num_prompt_tokens)
        except AssertionError as exc:
            logger.debug("Skip AscendStore store mask for unaligned request %s: %s", req_id, exc)
            store_masks = None
        load_spec = req_meta.load_spec
        skip_start = load_spec.vllm_cached_tokens if load_spec is not None else 0
        skip_end = (
            (
                load_spec.kvpool_store_skip_tokens
                if load_spec.kvpool_store_skip_tokens is not None
                else load_spec.kvpool_cached_tokens
            )
            if load_spec is not None
            else 0
        )
        save_start_token = self.get_saved_offset(req_id)

        def should_skip(start: int, end: int, group_save_start: int) -> bool:
            # This is the worker's durable high-water mark and intentionally
            # may lag ReqMeta.save_start_token. In Decode-only mode the first
            # job therefore checks the complete prefix and can restore missing
            # ancestors of the new Decode block. After a successful store it
            # advances, avoiding a full-prefix lookup on later boundaries.
            if end <= group_save_start:
                return True
            return skip_end > skip_start and start >= skip_start and end <= skip_end

        for group_id in req_meta.kv_cache_group_ids or [0]:
            group_block_size = self._get_block_size(group_id)

            group_store_mask = (
                store_masks[group_id] if store_masks is not None and group_id < len(store_masks) else None
            )
            if group_store_mask is not None and not group_store_mask:
                continue

            starts: list[int] = []
            ends: list[int] = []
            keys: list[str] = []
            block_hashes = []
            key_block_ids: list[int] = []
            block_ids = req_meta.block_ids_by_group[group_id]
            skip_null_blocks = self._skip_null_blocks(req_meta, group_id)
            align_state_group = group_id < len(self.group_uses_align_state) and self.group_uses_align_state[group_id]
            # Align-state slots below the saved extent can become valid later.
            save_start = 0 if align_state_group else save_start_token

            def chunk_filter(
                start: int,
                group_block_size=group_block_size,
                save_start=save_start,
            ) -> bool:
                return not should_skip(start, start + group_block_size, save_start)

            tp_replicas = self.put_step if self.dcp_size <= 1 and not align_state_group else 1
            # PCP=2, TP=4, KV heads=2, tp_replicas=2:
            # PCP  TP (KV 0 / KV 1)  shard_rank  filtered candidates
            #  0        0 / 2            0      0, 4, ...
            #  0        1 / 3            1      1, 5, ...
            #  1        0 / 2            2      2, 6, ...
            #  1        1 / 3            3      3, 7, ...
            if group_id in self.sharded_mamba_group_ids:
                # Each TP/PCP rank saves its own head shard for every candidate checkpoint.
                shard_rank, shard_size = 0, 1
            else:
                shard_rank = self.pcp_rank * tp_replicas + self.tp_rank % tp_replicas
                shard_size = self.pcp_size * tp_replicas
            iterator = self.token_database.process_token_key_strings_with_block_ids(
                token_len,
                req_meta.block_hashes,
                block_ids,
                kv_cache_group_id=group_id,
                skip_null_blocks=skip_null_blocks,
                chunk_filter=chunk_filter,
                shard_rank=shard_rank,
                shard_size=shard_size,
                chunk_mask=group_store_mask,
            )
            for start, end, key, block_hash, block_id in iterator:
                # SWA may have freed an older block before this asynchronous
                # save runs. Its slot is then the null block (id 0), whose
                # address must never be published under the old key.
                if block_id <= 0:
                    continue
                starts.append(start)
                ends.append(end)
                keys.append(key)
                if self.enable_kv_event:
                    block_hashes.append(block_hash)
                key_block_ids.append(block_id)

            if not keys:
                continue
            partitions = getattr(self.token_database, "partitions", None)
            if self.kv_role == "kv_consumer" and partitions is not None and len(partitions) > 1:
                # Decode consumers split one logical block across Prefill PP
                # ranks. A partial prior PUT must be retried, so require every
                # partition key to exist before skipping the logical block.
                exists_states = self.lookup_store_keys(keys, kv_cache_group_id=group_id)
                if exists_states is None:
                    return False
                missing_indices = [index for index, exists in enumerate(exists_states) if not exists]
            else:
                missing_indices = self._get_missing_indices(keys, require_exists_check=self.enable_kv_event)
            if not missing_indices:
                continue
            starts = [starts[index] for index in missing_indices]
            ends = [ends[index] for index in missing_indices]
            keys = [keys[index] for index in missing_indices]
            if self.enable_kv_event:
                block_hashes = [block_hashes[index] for index in missing_indices]
            key_block_ids = [key_block_ids[index] for index in missing_indices]

            logger.debug(
                "Storing KV cache for %d out of %d blocks for request %s in group %d",
                len(keys),
                token_len // group_block_size,
                req_id,
                group_id,
            )
            addrs = []
            sizes = []
            stored_events: list[BlockStored] = []
            all_hashes = []
            if self.enable_kv_event:
                group_block_hashes = get_block_hashes(
                    req_meta.block_hashes,
                    group_block_size,
                    getattr(self.token_database, "hash_block_size", group_block_size),
                )
                all_hashes = [maybe_convert_block_hash(bh) for bh in group_block_hashes]
            logger.debug(
                "KV pool put request=%s group=%d token_len=%d keys=%d sample_keys=%s",
                req_id,
                group_id,
                token_len,
                len(keys),
                keys[:3],
            )
            for index, start in enumerate(starts):
                addr, size, _ = self._prepare_value(
                    start,
                    ends[index],
                    block_ids,
                    kv_cache_group_id=group_id,
                    block_id=key_block_ids[index],
                )
                addrs.append(addr)
                sizes.append(size)
                if self.enable_kv_event:
                    token_ids = self.get_event_token_ids(req_meta, start, ends[index])
                    block_size = (
                        req_meta.original_block_size[group_id]
                        if isinstance(req_meta.original_block_size, list)
                        else req_meta.original_block_size
                    )
                    if block_size is not None:
                        block_idx = start // group_block_size
                        if block_idx >= len(all_hashes):
                            continue
                        current_hash = all_hashes[block_idx]
                        parent_hash = all_hashes[block_idx - 1] if block_idx > 0 else None
                        stored_event = BlockStored(
                            block_hashes=[current_hash],
                            parent_block_hash=parent_hash,
                            token_ids=token_ids,
                            block_size=block_size,
                            lora_id=None,
                            medium="cpu",
                            lora_name=None,
                        )
                        stored_events.append(stored_event)
                        logger.debug("Added kv cache event '%s' to kv cache events queue", stored_event)

            logical_key_count = len(keys)
            if self.kv_role == "kv_consumer":
                keys, addrs, sizes = self._decode_adaptor_prefill_pp(
                    keys,
                    addrs,
                    sizes,
                    kv_cache_group_id=group_id,
                )
            if not keys or len(keys) % logical_key_count != 0:
                logger.error(
                    "Invalid PP key expansion for request %s: logical=%d physical=%d",
                    req_id,
                    logical_key_count,
                    len(keys),
                )
                return False
            if current_event is not None:
                current_event.synchronize()
            put_result = self.m_store.put(keys, addrs, sizes)
            if isinstance(put_result, list):
                if len(put_result) != len(keys):
                    logger.error("KV store backend returned %s results for %s keys", len(put_result), len(keys))
                    return False
                if not all(put_result):
                    if len(put_result) % logical_key_count != 0:
                        logger.error(
                            "KV store returned %s physical results for %s logical keys",
                            len(put_result),
                            logical_key_count,
                        )
                        return False
                    physical_keys_per_logical = len(put_result) // logical_key_count
                    logical_success = [
                        all(put_result[start : start + physical_keys_per_logical])
                        for start in range(0, len(put_result), physical_keys_per_logical)
                    ]
                    if self.enable_kv_event and len(stored_events) == len(logical_success):
                        self.update_kv_event(
                            [
                                event
                                for event, succeeded in zip(stored_events, logical_success, strict=True)
                                if succeeded
                            ]
                        )
                    logger.error("KV store backend partially failed to put request %s", req_id)
                    return False
            elif put_result is False:
                logger.error("KV store backend failed to put request %s", req_id)
                return False
            if self.enable_kv_event and stored_events:
                self.update_kv_event(stored_events)
        return True


class KVCacheStoreRecvingThread(KVTransferThread):
    def __init__(
        self,
        m_store: Backend,
        token_database: ChunkedTokenDatabase,
        block_size: int | list[int],
        tp_rank: int,
        tp_size: int = 1,
        dcp_size: int = 1,
        ready_event: threading.Event | None = None,
        invalid_block_ids: set[int] | None = None,
        invalid_block_ids_lock: threading.Lock | None = None,
        worker: Any = None,
        record_operation: Callable[[str, float, int], None] | None = None,
        read_dedup: bool = False,
    ):
        super().__init__(
            m_store,
            token_database,
            block_size,
            tp_rank,
            tp_size,
            dcp_size,
            ready_event,
            name="KVCacheStoreRecvingThread",
        )
        self._invalid_block_ids = invalid_block_ids if invalid_block_ids is not None else set()
        self._invalid_block_ids_lock = invalid_block_ids_lock or threading.Lock()
        self.worker = worker
        self._record_operation_cb = record_operation
        self.read_dedup = read_dedup

    def _handle_request(self, req_meta: ReqMeta):
        try:
            load_spec = req_meta.load_spec
            req_id = req_meta.req_id
            if load_spec is None:
                logger.error("KV pool async recv request %s has no load spec; skip load.", req_id)
                self.set_finished_request(req_id)
                return

            token_len = load_spec.token_len
            if self.worker is not None and getattr(self.worker, "tp_mismatch", False):
                group_block_size = self._get_block_size(0)
                mask_num = load_spec.vllm_cached_tokens // group_block_size * group_block_size
                loaded = self.worker._load_kv_tp_mismatch(
                    req_meta.block_hashes,
                    req_meta.block_ids_by_group[0],
                    token_len,
                    mask_num,
                )
                if loaded is False:
                    self._mark_failed_requests([req_meta])
                self.set_finished_request(req_id)
                return

            addr_list = []
            size_list = []
            key_list = []
            block_id_list: list[int] = []
            view_list: list[list[KVCacheByteRange]] = []
            group_ids = req_meta.kv_cache_group_ids or [0]
            load_masks = self.token_database.load_mask(req_meta.block_hashes, token_len)
            for group_id in group_ids:
                block_ids = req_meta.block_ids_by_group[group_id]
                group_block_size = self._get_block_size(group_id)
                mask_num = load_spec.vllm_cached_tokens // group_block_size * group_block_size

                def chunk_filter(start: int, group_id=group_id) -> bool:
                    return self.token_database.mask_allows_chunk(load_masks, group_id, start)

                token_iter = self.token_database.process_token_key_strings_with_block_ids(
                    token_len,
                    req_meta.block_hashes,
                    block_ids,
                    mask_num,
                    kv_cache_group_id=group_id,
                    skip_null_blocks=self._skip_null_blocks(req_meta, group_id),
                    chunk_filter=chunk_filter,
                )
                for start, end, key, _block_hash, block_id in token_iter:
                    if self.read_dedup:
                        addr, size, views, block_id = self.token_database.prepare_value_with_ranges(
                            start,
                            end,
                            block_ids,
                            kv_cache_group_id=group_id,
                            block_id=block_id,
                        )
                        view_list.append(views)
                    else:
                        addr, size, block_id = self._prepare_value(
                            start,
                            end,
                            block_ids,
                            kv_cache_group_id=group_id,
                            block_id=block_id,
                        )
                    key_list.append(key)
                    addr_list.append(addr)
                    size_list.append(size)
                    block_id_list.append(block_id)
            if not key_list:
                self.set_finished_request(req_id)
                return
            if self.read_dedup:
                if self.worker is None or not hasattr(self.worker, "_load_kv_read_dedup"):
                    raise RuntimeError("KV read deduplication requires the KVPool worker")
                load_get_start = time.perf_counter() if self._record_operation_cb is not None else 0.0
                read_result = self.worker._load_kv_read_dedup(
                    key_list,
                    addr_list,
                    size_list,
                    block_id_list,
                    view_list,
                )
                if read_result is not None:
                    request_failed, missing_block_ids, local_key_count = read_result
                    if self._record_operation_cb is not None:
                        self._record_operation_cb(
                            "load_get",
                            time.perf_counter() - load_get_start,
                            local_key_count,
                        )
                    if missing_block_ids:
                        with self._invalid_block_ids_lock:
                            self._invalid_block_ids.update(missing_block_ids)
                    if request_failed:
                        self._mark_failed_requests([req_meta])
                    self.set_finished_request(req_id)
                    return
                logger.debug("KV read deduplication unavailable; using legacy full read")
            key_list_c = key_list[self.tp_rank % len(key_list) :] + key_list[: self.tp_rank % len(key_list)]
            addr_list_c = addr_list[self.tp_rank % len(addr_list) :] + addr_list[: self.tp_rank % len(addr_list)]
            size_list_c = size_list[self.tp_rank % len(size_list) :] + size_list[: self.tp_rank % len(size_list)]
            block_id_list_c = (
                block_id_list[self.tp_rank % len(block_id_list) :] + block_id_list[: self.tp_rank % len(block_id_list)]
            )
            logger.debug(
                "KV pool async recv calls backend get request=%s token_len=%d groups=%s keys=%d sample_keys=%s",
                req_id,
                token_len,
                req_meta.kv_cache_group_ids or [0],
                len(key_list_c),
                key_list_c[:3],
            )
            load_get_start = time.perf_counter() if self._record_operation_cb is not None else 0.0
            ret = self.m_store.get(key_list_c, addr_list_c, size_list_c)
            if self._record_operation_cb is not None:
                self._record_operation_cb(
                    "load_get",
                    time.perf_counter() - load_get_start,
                    len(key_list_c),
                )
            if isinstance(self.m_store, MemcacheBackend) and (
                ret is None or len(ret) != len(key_list_c) or any(r != 0 for r in ret)
            ):
                self._mark_failed_requests([req_meta])
                self.set_finished_request(req_id)
                return
            if ret is not None and any(r != 0 for r in ret):
                missing_block_ids = record_failed_blocks(
                    block_id_list_c,
                    ret,
                )
                if len(req_meta.block_ids_by_group) == 1:
                    with self._invalid_block_ids_lock:
                        self._invalid_block_ids.update(missing_block_ids)
                elif missing_block_ids:
                    logger.error(
                        "KV load failed for hybrid request %s. "
                        "Skip invalid-block fallback to avoid scheduler crash. "
                        "failed_blocks=%s",
                        req_id,
                        missing_block_ids,
                    )
            elif ret is None:
                missing_block_ids = record_failed_blocks(
                    block_id_list_c,
                    [1] * len(block_id_list_c),
                )
                if len(req_meta.block_ids_by_group) == 1:
                    with self._invalid_block_ids_lock:
                        self._invalid_block_ids.update(missing_block_ids)
                elif missing_block_ids:
                    logger.error(
                        "KV load failed for hybrid request %s. "
                        "Skip invalid-block fallback to avoid scheduler crash. "
                        "failed_blocks=%s",
                        req_id,
                        missing_block_ids,
                    )
            logger.debug(
                "KV pool async recv backend get returned request=%s token_len=%d groups=%s keys=%d",
                req_id,
                token_len,
                req_meta.kv_cache_group_ids or [0],
                len(key_list_c),
            )
            self.set_finished_request(req_id)
        finally:
            self.request_queue.task_done()


def record_failed_blocks(
    block_ids: list[int],
    ret_codes: list[int],
) -> set[int]:
    failed_blocks: set[int] = set()
    for block_id, code in zip(block_ids, ret_codes):
        if code != 0:
            failed_blocks.add(block_id)
    if failed_blocks:
        logger.error(
            "Failed to load blocks. failed_count=%d, failed_blocks=%s. Check block availability and memory state.",
            len(failed_blocks),
            failed_blocks,
        )
    return failed_blocks
