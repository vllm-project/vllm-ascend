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
# The transport engines (HIXL / Mooncake) register page-aligned memory.  A
# plain ``torch.empty`` returned by the caching allocator is not necessarily
# 2 MiB aligned, and Mooncake rejects an unaligned registration range.  The
# staging allocation is over-allocated and sliced so its base is aligned.
OWNER_TRANSFER_STAGING_ALIGN = 2 << 20

# Direct P2P connectors do not push blocks into a store.  The decode worker
# pulls them from prefill memory, so the prefill worker has to stage the
# requested complete logical blocks into its registered buffer first and then
# tell the decode worker which staging slots to read.  These are the control
# messages that rendezvous the two sides; they are only used when the prefill
# worker advertised ``owner_staging`` in its handshake metadata.
OWNER_STAGE_MSG = b"owner_stage_msg"
OWNER_STAGE_ACK = b"owner_stage_ack"
OWNER_STAGE_FAIL = b"owner_stage_fail"

# Bound the per-copy request so a long prompt does not build one enormous
# Python-side address list before submitting it to the NPU.
_STAGE_COPY_BATCH_RANGES = 8192


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


def _allocate_staging_tensor(device: torch.device, requested: int) -> torch.Tensor:
    """Allocate an int8 staging allocation, falling back to smaller sizes."""
    if device.type == "npu":
        torch.npu.set_device(device)
    candidates = [requested, *(size for size in _STAGING_CANDIDATES if size < requested)]
    align = OWNER_TRANSFER_STAGING_ALIGN
    last_error: BaseException | None = None
    for nbytes in candidates:
        try:
            raw = torch.empty(nbytes + align, dtype=torch.int8, device=device)
            # The slice keeps the raw allocation alive and is still contiguous.
            offset = (-raw.data_ptr()) % align
            return raw[offset : offset + nbytes]
        except RuntimeError as error:  # OOM: try the next smaller buffer
            last_error = error
            logger.warning_once(
                "Owner transfer staging could not allocate %.1f MiB: %s",
                nbytes / (1 << 20),
                error,
            )
    raise RuntimeError("owner-sharded KV transfer cannot allocate a staging buffer") from last_error


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
        return _allocate_staging_tensor(self.device, requested)

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


class OwnerBlockStageBuffer:
    """Field-major staging buffer for complete logical KV blocks.

    Direct P2P transfer engines read from a *registered* address on the
    producer.  An owner slab maps the complete logical range, but only
    ``1 / pcp_size`` of its 2 MiB chunks are physically local, so the range
    cannot be registered (A2 HIXL rejects it).  This buffer holds copies of
    complete logical blocks in registered HBM.

    The layout is field-major: every ``(layer, cache)`` field gets a
    contiguous region of ``slot_count * block_bytes`` bytes, so consecutive
    slots of the same field are adjacent.  That lets the decode side reuse its
    existing "consecutive blocks become one transfer" optimisation once block
    ids are rewritten to slot ids.

    A slot holds one complete logical block: every field of the block is
    copied into the same slot.  A block that crosses owner chunk boundaries is
    copied as one range per field, so the transport always sees the whole
    block.
    """

    def __init__(
        self,
        device: torch.device,
        fields: Sequence[tuple[object, int]],
        *,
        capacity_bytes: int = OWNER_TRANSFER_STAGING_BYTES,
        buffer: torch.Tensor | None = None,
    ) -> None:
        if not fields:
            raise ValueError("Owner block staging needs at least one KV cache field")
        self.device = device
        self._field_keys = [key for key, _ in fields]
        if len(set(self._field_keys)) != len(self._field_keys):
            raise ValueError("Owner block staging field keys must be unique")
        self._field_bytes: dict[object, int] = {}
        for key, block_bytes in fields:
            if not isinstance(block_bytes, int) or block_bytes <= 0:
                raise ValueError(f"Owner block staging field {key!r} has no positive block size")
            self._field_bytes[key] = block_bytes
        self.slot_bytes = sum(self._field_bytes.values())

        self.buffer = buffer
        if self.buffer is None:
            self.buffer = self._allocate(capacity_bytes)
        if self.buffer.dtype != torch.int8 or not self.buffer.is_contiguous():
            raise ValueError("Owner block staging buffer must be a contiguous int8 tensor")
        self.ptr = int(self.buffer.data_ptr())
        capacity = int(self.buffer.numel()) * self.buffer.element_size()
        self.slot_count = capacity // self.slot_bytes
        if self.slot_count <= 0:
            raise ValueError(
                f"Owner block staging needs {self.slot_bytes} bytes per complete block but the "
                f"buffer holds {capacity}; increase additional_config.pcp_config.owner_capacity "
                "headroom or lower the per-request save chunk"
            )
        self.capacity = self.slot_count * self.slot_bytes

        self._region_ptr: dict[object, int] = {}
        offset = 0
        for key in self._field_keys:
            self._region_ptr[key] = self.ptr + self.slot_count * offset
            offset += self._field_bytes[key]
        self._used = bytearray(self.slot_count)
        self._lock = threading.Lock()

    def _allocate(self, requested: int) -> torch.Tensor:
        return _allocate_staging_tensor(self.device, requested)

    def field_ptr(self, key: object) -> int:
        """Registered address of one field's first slot."""
        return self._region_ptr[key]

    def field_block_bytes(self, key: object) -> int:
        """Bytes one complete block occupies in one field's region."""
        return self._field_bytes[key]

    def slot_address(self, key: object, slot: int) -> int:
        return self._region_ptr[key] + int(slot) * self._field_bytes[key]

    def allocate_slots(self, count: int) -> int | None:
        """Reserve ``count`` consecutive slots, or return ``None`` if full."""
        if count <= 0:
            return 0
        with self._lock:
            run_start = -1
            run = 0
            for index, used in enumerate(self._used):
                if used:
                    run_start = -1
                    run = 0
                    continue
                if run == 0:
                    run_start = index
                run += 1
                if run == count:
                    for slot in range(run_start, run_start + count):
                        self._used[slot] = 1
                    return run_start
            return None

    def free_slots(self, first_slot: int, count: int) -> None:
        if count <= 0:
            return
        with self._lock:
            end = min(first_slot + count, self.slot_count)
            for slot in range(max(0, first_slot), end):
                self._used[slot] = 0


def owner_stage_keys(group_blocks: Iterable[tuple[int, Iterable[int]]]) -> list[tuple[int, int]]:
    """Flatten ``(group, block ids)`` into the ordered staging key sequence.

    Both the prefill and decode side run this over the same list so that a
    position in the returned sequence is the slot offset.  Duplicate
    ``(group, block)`` pairs (for example an SFA replicate-K list that repeats
    a tail block) are staged once and share one slot.
    """
    keys: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for group, block_ids in group_blocks:
        group_id = int(group)
        for block_id in block_ids:
            key = (group_id, int(block_id))
            if key in seen:
                continue
            seen.add(key)
            keys.append(key)
    return keys


def owner_stage_slot_map(first_slot: int, keys: Sequence[tuple[int, int]]) -> dict[tuple[int, int], int]:
    """Map each staged ``(group, block)`` key to its absolute slot id."""
    return {key: int(first_slot) + position for position, key in enumerate(keys)}


class OwnerP2PStageController:
    """Stage complete logical blocks for a direct P2P transfer engine.

    ``group_fields`` maps a KV cache group id to the ``(layer, cache)`` field
    keys that belong to it.  ``field_sources`` maps each field key to its
    ``(logical_base_address, block_stride)`` on the owner slab.
    """

    def __init__(
        self,
        stager: OwnerBlockStageBuffer,
        group_fields: dict[int, Sequence[object]],
        field_sources: dict[object, tuple[int, int]],
    ) -> None:
        if not group_fields:
            raise ValueError("Owner P2P staging needs at least one KV cache group")
        self.stager = stager
        self.group_fields = {int(group): list(keys) for group, keys in group_fields.items()}
        self.field_sources = {key: (int(base), int(stride)) for key, (base, stride) in field_sources.items()}
        for group, keys in self.group_fields.items():
            if not keys:
                raise ValueError(f"Owner P2P staging group {group} has no KV cache fields")
            for key in keys:
                if key not in self.field_sources:
                    raise ValueError(f"Owner P2P staging group {group} references unknown field {key!r}")
        self._prepared: dict[str, tuple[int, int]] = {}
        self._lock = threading.Lock()

    def _ensure_device(self) -> None:
        if self.stager.device.type == "npu":
            torch.npu.set_device(self.stager.device)

    def _copy_blocks(self, first_slot: int, keys: Sequence[tuple[int, int]]) -> None:
        src: list[int] = []
        dst: list[int] = []
        sizes: list[int] = []
        for position, (group, block_id) in enumerate(keys):
            slot = first_slot + position
            for key in self.group_fields[group]:
                base, stride = self.field_sources[key]
                src.append(base + block_id * stride)
                dst.append(self.stager.slot_address(key, slot))
                sizes.append(self.stager.field_block_bytes(key))
        self._ensure_device()
        for start in range(0, len(src), _STAGE_COPY_BATCH_RANGES):
            _device_copy(
                src[start : start + _STAGE_COPY_BATCH_RANGES],
                dst[start : start + _STAGE_COPY_BATCH_RANGES],
                sizes[start : start + _STAGE_COPY_BATCH_RANGES],
            )

    def stage_request(
        self,
        request_id: str,
        group_blocks: Sequence[tuple[int, Iterable[int]]],
    ) -> tuple[int, int]:
        """Stage one request's complete blocks; returns ``(first_slot, count)``."""
        keys = owner_stage_keys(group_blocks)
        if not keys:
            return 0, 0
        unknown = sorted({group for group, _ in keys if group not in self.group_fields})
        if unknown:
            raise ValueError(f"Owner P2P staging received unknown KV cache groups: {unknown}")
        if any(block_id < 0 for _, block_id in keys):
            raise ValueError("Owner P2P staging received a negative block id")
        with self._lock:
            previous = self._prepared.get(request_id)
            if previous is not None:
                if previous[1] != len(keys):
                    raise RuntimeError(
                        f"Owner P2P staging was prepared twice for request {request_id} with different block counts"
                    )
                return previous
            first_slot = self.stager.allocate_slots(len(keys))
            if first_slot is None:
                raise RuntimeError(
                    f"Owner P2P staging buffer is full: request {request_id} needs {len(keys)} complete "
                    f"blocks and the buffer holds {self.stager.slot_count}"
                )
            try:
                self._copy_blocks(first_slot, keys)
            except BaseException:
                self.stager.free_slots(first_slot, len(keys))
                raise
            self._prepared[request_id] = (first_slot, len(keys))
            return first_slot, len(keys)

    def release_request(self, request_id: str) -> None:
        with self._lock:
            entry = self._prepared.pop(request_id, None)
            if entry is not None:
                self.stager.free_slots(*entry)
