# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Transport staging for owner-sharded PCP KV caches.

An owner slab maps the complete logical KV address range but owns only
``1 / pcp_size`` of its physical pages.  A host transport engine (HIXL /
Mooncake) cannot register that VMM range on every platform: the range
contains peer-mapped pages, and ``HcommMemReg`` rejects it on A2.

The connector therefore keeps the complete-block key contract and stages
each block through a locally allocated HBM buffer that the transport engine
*can* register::

    owner VA (local + peer pages) --D2D copy--> staging HBM --> transport

Every PCP rank still saves complete logical blocks, so the decode side
receives a complete KV cache.  No block id, block table, or key changes.
"""

from __future__ import annotations

import threading
from collections.abc import Iterable, Sequence

import torch
from vllm.logger import logger

# One staged transfer is at most a chunk of blocks.  A 20k prompt at
# block_size=128 is ~35 MiB for DSV4.1; 128 MiB covers several chunks while
# staying inside the ``gpu-memory-utilization`` slack used by owner staging.
OWNER_TRANSFER_STAGING_BYTES = 128 << 20
_STAGING_CANDIDATES = (128 << 20, 64 << 20, 32 << 20, 16 << 20)


def owner_backed_tensor(kv_caches: dict) -> torch.Tensor | None:
    """Return one tensor backed by an owner slab, or ``None`` when replicated."""
    from vllm_ascend.attention.context_parallel.pcp_owner_vmm import (
        owner_slab_for_tensor,
    )

    for cache in kv_caches.values():
        tensors = cache if isinstance(cache, (tuple, list)) else (cache,)
        for tensor in tensors:
            if tensor is not None and owner_slab_for_tensor(tensor) is not None:
                return tensor
    return None


def _device_copy(src_addrs: Sequence[int], dst_addrs: Sequence[int], sizes: Sequence[int]) -> None:
    """Copy device ranges on the current NPU stream and wait for completion."""
    if not src_addrs:
        return
    copy_op = getattr(torch.ops, "_C_ascend", None)
    if copy_op is None or not hasattr(copy_op, "swap_blocks_batch"):
        raise RuntimeError(
            "owner-sharded KV transfer requires the vllm-ascend swap_blocks_batch op; "
            "rebuild the extension or disable additional_config.pcp_config.kv_sharing=owner"
        )
    src = torch.tensor(src_addrs, dtype=torch.int64)
    dst = torch.tensor(dst_addrs, dtype=torch.int64)
    length = torch.tensor(sizes, dtype=torch.int64)
    # 2 = device-to-device. The op submits on the current NPU stream.
    copy_op.swap_blocks_batch(src, dst, length, 2)
    torch.npu.current_stream().synchronize()


class OwnerTransferStager:
    """Stage complete owner-backed pages through a transport-registered buffer."""

    def __init__(
        self,
        device: torch.device,
        *,
        capacity_bytes: int = OWNER_TRANSFER_STAGING_BYTES,
        buffer: torch.Tensor | None = None,
    ) -> None:
        self.device = device
        self.buffer = buffer
        if self.buffer is None:
            self.buffer = self._allocate(capacity_bytes)
        if self.buffer.dtype != torch.int8 or not self.buffer.is_contiguous():
            raise ValueError("Owner transfer staging buffer must be a contiguous int8 tensor")
        self.ptr = int(self.buffer.data_ptr())
        self.capacity = int(self.buffer.numel()) * self.buffer.element_size()
        if self.capacity <= 0:
            raise ValueError("Owner transfer staging buffer must be non-empty")
        self._registered = False
        # The worker starts separate sending and receiving threads; both may
        # need the single staging allocation.
        self._lock = threading.Lock()

    def _allocate(self, requested: int) -> torch.Tensor:
        torch.npu.set_device(self.device)
        candidates = [requested, *(size for size in _STAGING_CANDIDATES if size < requested)]
        last_error: BaseException | None = None
        for nbytes in candidates:
            try:
                return torch.empty(nbytes, dtype=torch.int8, device=self.device)
            except RuntimeError as error:  # OOM: try the next smaller buffer
                last_error = error
                logger.warning_once(
                    "Owner transfer staging could not allocate %.1f MiB: %s",
                    nbytes / (1 << 20),
                    error,
                )
        raise RuntimeError("owner-sharded KV transfer cannot allocate a staging buffer") from last_error

    def register(self, register_fn) -> None:
        """Register the staging buffer with the transport engine exactly once."""
        if self._registered:
            return
        register_fn([self.ptr], [self.capacity])
        self._registered = True

    @staticmethod
    def _validate(addrs: Iterable[Sequence[int]], sizes: Iterable[Sequence[int]]) -> None:
        addrs = list(addrs)
        sizes = list(sizes)
        if len(addrs) != len(sizes):
            raise ValueError("Owner transfer addresses and sizes must have the same key count")
        for key_addrs, key_sizes in zip(addrs, sizes, strict=True):
            if len(key_addrs) != len(key_sizes):
                raise ValueError("Owner transfer key address/size lists must align")

    def _batches(self, addrs, sizes):
        """Yield ``(indices, staged_addrs, src_flat, dst_flat, size_flat)``."""
        self._validate(addrs, sizes)
        indices: list[int] = []
        staged_by_index: dict[int, list[int]] = {}
        used = 0

        def emit():
            if not indices:
                return
            src_flat: list[int] = []
            dst_flat: list[int] = []
            size_flat: list[int] = []
            batch_staged: list[list[int]] = []
            for index in indices:
                key_addrs = addrs[index]
                key_sizes = sizes[index]
                key_staged = staged_by_index[index]
                batch_staged.append(key_staged)
                for source, staged_addr, size in zip(key_addrs, key_staged, key_sizes, strict=True):
                    src_flat.append(int(source))
                    dst_flat.append(int(staged_addr))
                    size_flat.append(int(size))
            yield list(indices), batch_staged, src_flat, dst_flat, size_flat

        for index, key_sizes in enumerate(sizes):
            key_bytes = sum(int(size) for size in key_sizes)
            if key_bytes > self.capacity:
                raise ValueError(
                    f"One owner KV key needs {key_bytes} bytes but the transport staging buffer "
                    f"holds {self.capacity}; disable additional_config.pcp_config.owner_capacity "
                    "or lower the per-request save chunk."
                )
            if indices and used + key_bytes > self.capacity:
                yield from emit()
                indices = []
                staged_by_index = {}
                used = 0
            key_staged = []
            cursor = used
            for size in key_sizes:
                key_staged.append(self.ptr + cursor)
                cursor += int(size)
            staged_by_index[index] = key_staged
            indices.append(index)
            used += key_bytes
        yield from emit()

    def put(self, backend, keys: list[str], addrs, sizes) -> list[bool]:
        """Stage source pages and PUT complete keys; returns one result per key."""
        with self._lock:
            return self._put_locked(backend, keys, addrs, sizes)

    def _ensure_device(self) -> None:
        if self.device.type == "npu":
            # The transfer threads are created after the worker sets its
            # device; re-assert it before issuing an NPU copy.
            torch.npu.set_device(self.device)

    def _put_locked(self, backend, keys: list[str], addrs, sizes) -> list[bool]:
        self._ensure_device()
        results: list[bool] = []
        for indices, staged, src_flat, dst_flat, size_flat in self._batches(addrs, sizes):
            _device_copy(src_flat, dst_flat, size_flat)
            batch_keys = [keys[index] for index in indices]
            batch_sizes = [sizes[index] for index in indices]
            ret = backend.put(batch_keys, staged, batch_sizes)
            if isinstance(ret, list):
                if len(ret) != len(batch_keys):
                    raise RuntimeError(f"KV store returned {len(ret)} results for {len(batch_keys)} staged owner keys")
                results.extend(bool(value) for value in ret)
            else:
                results.extend([ret is not False] * len(batch_keys))
        return results

    def get(self, backend, keys: list[str], addrs, sizes):
        """GET complete keys into staging, then copy to owner pages."""
        with self._lock:
            return self._get_locked(backend, keys, addrs, sizes)

    def _get_locked(self, backend, keys: list[str], addrs, sizes):
        self._ensure_device()
        results: list[int] = []
        for indices, staged, _src_flat, _dst_flat, _size_flat in self._batches(addrs, sizes):
            batch_keys = [keys[index] for index in indices]
            batch_sizes = [sizes[index] for index in indices]
            ret = backend.get(batch_keys, staged, batch_sizes)
            if ret is None:
                return None
            if len(ret) != len(batch_keys):
                raise RuntimeError(f"KV store returned {len(ret)} results for {len(batch_keys)} staged owner keys")
            staged_flat = [addr for key_staged in staged for addr in key_staged]
            owner_flat = [int(addr) for index in indices for addr in addrs[index]]
            size_flat = [int(size) for index in indices for size in sizes[index]]
            _device_copy(staged_flat, owner_flat, size_flat)
            results.extend(ret)
        return results
