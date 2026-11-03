# SPDX-License-Identifier: Apache-2.0
"""Mooncake HBM reads with bounded staging into the shared SFA host pool."""

from __future__ import annotations

import threading
from bisect import bisect_right
from collections.abc import Callable, Iterable
from typing import Any

BACKEND_MOONCAKE = "mooncake"
STAGING_ALIGNMENT_BYTES = 2 * 1024 * 1024
DEFAULT_STAGING_BYTES = 16 * 1024 * 1024


class _MemoryRanges:
    def __init__(self, regions: Iterable[tuple[int, int]]):
        merged: list[tuple[int, int]] = []
        for base, size in sorted(regions):
            if base <= 0 or size <= 0:
                raise ValueError("SFA transfer memory regions must have positive bases and sizes")
            end = base + size
            if merged and base <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
            else:
                merged.append((base, end))
        self._regions = merged
        self._starts = [base for base, _ in merged]

    def contains(self, base: int, size: int) -> bool:
        index = bisect_right(self._starts, base) - 1
        return index >= 0 and base + size <= self._regions[index][1]


class MooncakePullBackend:
    """Keep host GVAs out of Mooncake's Ascend memory registration API.

    One backend belongs to one consumer worker. The lock protects its staging
    window if more than one read path calls it. ``land`` must finish the local
    HBM-to-host copy before returning: only then may the window be reused and
    the pull thread send READ_DONE to P.
    """

    def __init__(
        self,
        engine: Any,
        *,
        host_regions: Iterable[tuple[int, int]],
        device_regions: Iterable[tuple[int, int]],
        staging_ptr: int,
        staging_bytes: int,
        land: Callable[[int, int, int], None],
    ):
        if staging_ptr <= 0 or staging_bytes <= 0:
            raise ValueError("Mooncake staging pointer and capacity must be positive")
        self._engine = engine
        self._host_regions = _MemoryRanges(host_regions)
        self._device_regions = _MemoryRanges(device_regions)
        self._staging_ptr = staging_ptr
        self._staging_bytes = staging_bytes
        self._land = land
        self._lock = threading.Lock()

    def get_rpc_port(self) -> int:
        return self._engine.get_rpc_port()

    def batch_transfer_sync_read(
        self,
        session_id: str,
        local_buffers: list[int],
        peer_buffers: list[int],
        length_list: list[int],
    ) -> int:
        if not len(local_buffers) == len(peer_buffers) == len(length_list):
            raise ValueError("Mooncake pull descriptor counts must match")
        # Validate the entire plan before performing any network or local copy.
        plan = []
        for local, peer, size in zip(local_buffers, peer_buffers, length_list):
            local, peer, size = int(local), int(peer), int(size)
            if size < 0 or local <= 0 or peer <= 0:
                raise ValueError("Mooncake pull descriptors contain an invalid address or length")
            if size == 0:
                continue
            host = self._host_regions.contains(local, size)
            if not host and not self._device_regions.contains(local, size):
                raise ValueError(f"Mooncake pull destination is outside registered SFA regions: {local:#x}+{size}")
            plan.append((local, peer, size, host))

        with self._lock:
            index = 0
            while index < len(plan):
                local, peer, size, host = plan[index]
                if not host:
                    end = index + 1
                    while end < len(plan) and not plan[end][3]:
                        end += 1
                    direct = plan[index:end]
                    result = self._engine.batch_transfer_sync_read(
                        session_id,
                        [item[0] for item in direct],
                        [item[1] for item in direct],
                        [item[2] for item in direct],
                    )
                    if result != 0:
                        return result
                    index = end
                    continue
                for offset in range(0, size, self._staging_bytes):
                    length = min(size - offset, self._staging_bytes)
                    result = self._engine.batch_transfer_sync_read(
                        session_id, [self._staging_ptr], [peer + offset], [length]
                    )
                    if result != 0:
                        return result
                    self._land(local + offset, self._staging_ptr, length)
                index += 1
        return 0


def get_device_cache_regions(kv_caches: dict[str, Any]):
    # The transport is optional. Loading its descriptor planner does not import
    # torch, Mooncake, or initialize a device.
    import torch

    from vllm_ascend.distributed.kv_transfer.utils.utils import collect_storage_merged_register_regions

    def tensors(value):
        if isinstance(value, torch.Tensor):
            if value.device.type != "cpu" and value.numel() > 0:
                yield value
        elif isinstance(value, (tuple, list)):
            for item in value:
                yield from tensors(item)

    device_caches = {name: tuple(tensors(value)) for name, value in kv_caches.items()}
    return collect_storage_merged_register_regions(device_caches)


def build_mooncake_consumer_backend(
    engine: Any,
    *,
    manager: Any,
    kv_caches: dict[str, Any],
    num_blocks: int,
    staging_bytes: int = DEFAULT_STAGING_BYTES,
) -> MooncakePullBackend:
    import torch

    from vllm_ascend.distributed.kv_transfer.utils.mooncake_transfer_engine import global_te
    from vllm_ascend.distributed.kv_transfer.utils.utils import validate_register_region_count

    if isinstance(staging_bytes, bool) or not isinstance(staging_bytes, int) or staging_bytes <= 0:
        raise ValueError("mooncake_staging_bytes must be a positive integer")
    if staging_bytes % STAGING_ALIGNMENT_BYTES:
        raise ValueError(f"mooncake_staging_bytes must be a multiple of {STAGING_ALIGNMENT_BYTES}")
    if num_blocks <= 0:
        raise ValueError("Mooncake SFA pull requires a positive host block capacity")
    if not len(manager.gvas_k_bases) == len(manager.gvas_v_bases) == len(manager.cpu_block_lens):
        raise ValueError("Mooncake SFA host pool bases and block sizes do not match")
    host_regions = []
    for k_base, v_base, (k_stride, v_stride) in zip(manager.gvas_k_bases, manager.gvas_v_bases, manager.cpu_block_lens):
        host_regions.extend([(int(k_base), num_blocks * int(k_stride)), (int(v_base), num_blocks * int(v_stride))])
    _MemoryRanges(host_regions)

    regions = get_device_cache_regions(kv_caches)
    validate_register_region_count(regions)
    global_te.register_buffer(regions.ptrs, regions.lengths)
    # Keep the aligned view, stream and descriptors alive in the callback owned
    # by this worker. Registered memory is valid for the worker-process lifetime.
    device = torch.device("npu", torch.npu.current_device())
    with torch.npu.device(device):
        storage = torch.empty(staging_bytes + STAGING_ALIGNMENT_BYTES, dtype=torch.int8, device=device)
        offset = (-storage.data_ptr()) % STAGING_ALIGNMENT_BYTES
        staging = storage[offset : offset + staging_bytes]
        global_te.register_buffer([staging.data_ptr()], [staging_bytes])
        stream = torch.npu.Stream()
        sources = torch.empty(1, dtype=torch.int64, device=device)
        destinations = torch.empty_like(sources)
        lengths = torch.empty_like(sources)
        count = torch.ones(1, dtype=torch.int32, device=device)
        stream.wait_stream(torch.npu.current_stream())

    def land(destination: int, source: int, length: int) -> None:
        with torch.npu.device(device), torch.npu.stream(stream):
            # Retain the registered allocation through its final view.
            assert source == staging.data_ptr()
            sources.fill_(source)
            destinations.fill_(destination)
            lengths.fill_(length)
            # The manager owns the SHARED host GVA mapping on every TP rank.
            # Reuse its local-copy API; the cross-host transport stays Mooncake.
            manager.copy_sfa_kv(sources, destinations, lengths, count)
            stream.synchronize()

    return MooncakePullBackend(
        engine,
        host_regions=host_regions,
        device_regions=zip(regions.ptrs, regions.lengths),
        staging_ptr=staging.data_ptr(),
        staging_bytes=staging_bytes,
        land=land,
    )
