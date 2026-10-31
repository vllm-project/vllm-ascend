# SPDX-License-Identifier: Apache-2.0
"""MemFabric BM storage; consumers retain metadata and NPU pointers only."""

from __future__ import annotations

import ctypes
import threading
import weakref
from typing import Any, ClassVar

import torch

GIB = 1 << 30
CHUNK_ROWS = 1 << 22
DRAM_MAP_HOST_VA = 1 << 9  # SMEM_BM_FLAG_DRAM_MAP_HOST_VA


class MemfabricRuntime:
    """Share BM's process-wide session between this worker's Engram layers.

    All references must name the same generation and device. TransferEngine
    uses a separate API; another subsystem must not initialize the BM singleton.
    """

    _lock = threading.RLock()
    _active: ClassVar[MemfabricRuntime | None] = None
    mf: Any
    bm: Any
    spec: tuple
    device: torch.device
    references: int
    objects: set[int]
    rank: int

    @classmethod
    def acquire(cls, generation, store_url, world_size, device, *, owner):
        device = torch.device(device)
        if device.type != "npu" or device.index is None:
            raise ValueError("An explicit NPU device index is required")
        spec = (str(generation), store_url, world_size, device, owner)
        with cls._lock:
            if cls._active is not None:
                if cls._active.spec != spec:
                    raise RuntimeError("A worker may attach to only one MemFabric Engram generation/device")
                cls._active.references += 1
                return cls._active
            import memfabric_hybrid as mf
            from memfabric_hybrid import bm

            runtime = cls()
            runtime.mf, runtime.bm, runtime.spec = mf, bm, spec
            runtime.device, runtime.references = device, 1
            runtime.objects = set()
            runtime.check(mf.initialize(), "initialize")
            config = bm.BmConfig()
            config.dynamic_world_size = True
            config.start_store = owner
            with torch.npu.device(device):
                runtime.check(bm.initialize(store_url, world_size, device.index, config), "BM initialize")
            runtime.rank = bm.bm_rank_id()
            cls._active = runtime
            return runtime

    def check(self, result, operation):
        if result:
            raise RuntimeError(f"Engram MemFabric {operation}: {result}: {self.mf.get_last_err_msg()}")

    def close(self):
        with self._lock:
            if self.references > 1:
                self.references -= 1
                return
            if self.objects:
                raise RuntimeError("Close Engram mappings before their MemFabric session")
            with torch.npu.device(self.device):
                self.bm.uninitialize()
            self.references = 0
            type(self)._active = None
            # mf.uninitialize() is not refcounted and would also disable other
            # MemFabric users, including an independently owned TransferEngine.


class MappedHostBuffer:
    """One BM object: only the owner contributes physical DRAM.

    Consumers join with zero local DRAM/HBM and resolve their own NPU address
    once at attachment. No copy API is used on lookup or publication paths.
    """

    def __init__(self, runtime, object_id, rows, width, dtype, *, owner, owner_rank=None):
        if rows <= 0 or width <= 0 or dtype not in (torch.int8, torch.float32):
            raise ValueError("Shared Engram requires positive INT8/FP32 dimensions")
        if not 0 <= object_id < 64 or object_id in runtime.objects:
            raise ValueError("Duplicate or invalid MemFabric Engram object id")
        self.runtime, self.object_id = runtime, object_id
        self.rows, self.width, self.dtype = rows, width, dtype
        self.owner, self.device = owner, runtime.device
        self.row_bytes = width * torch.empty((), dtype=dtype).element_size()
        self.size = ((rows * self.row_bytes + GIB - 1) // GIB) * GIB
        self.closed = False
        self._views: weakref.WeakSet[HostTableSlice] = weakref.WeakSet()
        self.handle: Any = None
        joined = False
        with torch.npu.device(self.device):
            try:
                self.handle = runtime.bm.create2(
                    object_id,
                    self.size if owner else 0,
                    self.size,
                    data_op_type=runtime.bm.BmDataOpType.SDMA,
                    flags=DRAM_MAP_HOST_VA,
                )
                runtime.check(self.handle.join(), "join")
                joined = True
                rank = runtime.rank if owner else owner_rank
                self.gva = self.handle.peer_rank_ptr(rank, runtime.bm.BmMemType.HOST)
                self.device_pointer(0)
                if owner:
                    self.host_ptr = self.handle.gva_to_va(self.gva, runtime.bm.BmMemType.LOCAL_HOST)
                    if not self.host_ptr:
                        raise RuntimeError("MemFabric did not map owner DRAM for CPU publication")
                elif self.handle.local_mem_size(runtime.bm.BmMemType.HOST) != 0:
                    raise RuntimeError("An Engram consumer must not allocate a local table replica")
            except BaseException:
                if self.handle is not None:
                    if joined:
                        self.handle.leave()
                    self.handle.destroy()
                raise
        runtime.objects.add(object_id)

    def device_pointer(self, offset, size=1):
        pointer = self.handle.gva_to_va(self.gva + offset, self.runtime.bm.BmMemType.LOCAL_DEVICE)
        if not pointer:
            raise RuntimeError("MemFabric DRAM is not directly NPU-addressable; use a reachable A3 supernode")
        if size > 1:
            last = self.handle.gva_to_va(self.gva + offset + size - 1, self.runtime.bm.BmMemType.LOCAL_DEVICE)
            if last != pointer + size - 1:
                raise RuntimeError("MemFabric NPU range is not contiguous within an Engram lookup chunk")
        return pointer

    def publish(self, start, source):
        if self.closed or not self.owner:
            raise RuntimeError("Only a live owner may publish table rows")
        if source.device.type != "cpu" or source.dtype != self.dtype:
            raise ValueError("Publication requires a matching CPU dtype")
        if source.ndim != 2 or source.shape[1] != self.width or start < 0 or start + len(source) > self.rows:
            raise ValueError("Publication is outside the table shape")
        source = source.contiguous()
        offset = start * self.row_bytes
        remaining = source.numel() * source.element_size()
        consumed = 0
        # A large BM GVA can have independently mapped physical slices.
        while remaining:
            count = min(remaining, GIB - offset % GIB)
            host = self.handle.gva_to_va(self.gva + offset, self.runtime.bm.BmMemType.LOCAL_HOST)
            if not host:
                raise RuntimeError("MemFabric owner publication range is not CPU-addressable")
            ctypes.memmove(host, source.data_ptr() + consumed, count)
            offset += count
            consumed += count
            remaining -= count

    def view(self, start=0, end=None):
        end = self.rows if end is None else end
        if self.closed or not 0 <= start < end <= self.rows:
            raise ValueError("Invalid or closed shared Engram slice")
        view = HostTableSlice(self, start, end)
        self._views.add(view)
        return view

    def close(self):
        if self.closed:
            return
        if self._views:
            raise RuntimeError("Release all Engram slices before closing their mapping")
        with torch.npu.device(self.device):
            torch.npu.synchronize()
            self.runtime.check(self.handle.leave(), "leave")
            self.handle.destroy()
        self.runtime.objects.remove(self.object_id)
        self.closed = True


class HostTableSlice:
    """Existing lookup ABI: metadata-only tensor plus stable NPU chunk pointers."""

    def __init__(self, mapping, start, end):
        self._mapping = mapping
        self.tensor = torch.empty((end - start, mapping.width), dtype=mapping.dtype, device="meta")
        self.ptrs = torch.tensor(
            [
                mapping.device_pointer(row * mapping.row_bytes, min(CHUNK_ROWS, end - row) * mapping.row_bytes)
                for row in range(start, end, CHUNK_ROWS)
            ],
            dtype=torch.int64,
            device=mapping.device,
        )

    def close(self):
        if self._mapping is None:
            return
        with torch.npu.device(self._mapping.device):
            torch.npu.synchronize()
        self._mapping._views.discard(self)
        self.ptrs = self.tensor = None
        self._mapping = None
