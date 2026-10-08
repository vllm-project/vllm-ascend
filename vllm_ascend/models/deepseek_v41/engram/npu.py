# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NPU-side Engram storage, routing and lookup.

Tables are BF16 or INT8 with group-32 FP32 scales, sharded into contiguous hash
head buckets.  With ``EngramConfig.cpu_offload`` the shard stays in host memory:
``aclrtHostRegisterV2`` pins it and ``aclrtHostGetDevicePointer`` publishes the
address the NPU gather kernel reads, so an offloaded table needs neither an H2D
copy nor a host-side gather.
"""

import ctypes
import mmap
import os
from functools import cache
from multiprocessing import shared_memory
from unittest.mock import patch

import torch
import torch.distributed as dist
from vllm.logger import logger
from vllm.triton_utils import tl, triton

SCALE_GROUP = 32
# A 384M row table overflows the 32 bit offset arithmetic a single Triton tile
# can express, so the device address of every group of rows is published
# separately.
CHUNK_ROWS = 1 << 22
MAX_SINGLE_REGISTRATION_BYTES = 32 * 1024**3
ACL_HOST_REG_MAPPED = 0x2
ACL_HOST_REG_PINNED = 0x10000000


def engram_cpu_offload(vllm_config) -> bool:
    """Whether the Engram table is offloaded to host memory (UVA lookup).

    ``--engram-config`` turns on host offload. Without it, the tables stay on
    the device, exactly like upstream.
    """

    engram_config = getattr(vllm_config, "engram_config", None)
    return bool(engram_config is not None and engram_config.cpu_offload)


def quantize_engram_rows(rows):
    """Group32 symmetric INT8 with FP32 power-of-two scales and ties-to-even."""
    grouped = rows.float().unflatten(-1, (-1, SCALE_GROUP))
    maximum = grouped.abs().amax(-1, keepdim=True)
    scale = torch.where(maximum == 0, torch.ones_like(maximum), maximum / 127)
    # NPU exp2 can return one ULP below an exact power of two, changing
    # ties-to-even codes. ldexp constructs the binary scale exactly.
    exponent = torch.ceil(torch.log2(scale))
    scale = torch.where(torch.isfinite(exponent), torch.ldexp(torch.ones_like(scale), exponent.int()), scale)
    codes = torch.round(grouped / scale).clamp(-127, 127).to(torch.int8).flatten(-2)
    return codes, scale.squeeze(-1)


def dequantize_engram_rows(codes, scale):
    # Keep one FP32 work buffer: in-place scaling avoids the extra FP32 result
    # allocation created by the broadcast multiply expression.
    decoded = codes.float().unflatten(-1, (-1, SCALE_GROUP))
    decoded.mul_(scale.unsqueeze(-1))
    return decoded.flatten(-2).bfloat16()


@triton.jit
def _engram_int8_gather_dequant_kernel(
    weight_ptr,
    scale_ptr,
    ids_ptr,
    output_ptr,
    rows,
    vocab_start,
    vocab_end,
    ids_stride_t,
    WIDTH: tl.constexpr,
    GROUP: tl.constexpr,
    HEAD_START: tl.constexpr,
    LOCAL_HEADS: tl.constexpr,
    PAD_HEADS: tl.constexpr,
    QUANTIZED: tl.constexpr,
):
    row = tl.program_id(0)
    if row >= rows:
        return
    offsets = tl.arange(0, WIDTH)
    # Row `row` is (token, local head); ids is [tokens, n_hash_cols] and this
    # shard owns heads [HEAD_START, HEAD_START + LOCAL_HEADS). Local heads are
    # written contiguously and the rest of PAD_HEADS stays untouched, so a
    # narrower shard never writes into the padding other ranks read.
    token = row // LOCAL_HEADS
    local = row % LOCAL_HEADS
    source_row = tl.load(ids_ptr + token * ids_stride_t + HEAD_START + local).to(tl.int64)
    # Same last line of defence as the host-uva kernel, and the same contract
    # as upstream's lookup: ids are global and only this shard's vocab range is
    # owned; anything else reads row 0 and is masked back to zero.
    owned = (source_row >= vocab_start) & (source_row < vocab_end)
    local_row = tl.where(owned, source_row - vocab_start, 0)
    codes = tl.load(weight_ptr + local_row * WIDTH + offsets).to(tl.float32)
    if QUANTIZED:
        scales = tl.load(scale_ptr + local_row * (WIDTH // GROUP) + offsets // GROUP)
        codes = codes * scales
    result = codes.to(tl.bfloat16)
    tl.store(
        output_ptr + (token * PAD_HEADS + local) * WIDTH + offsets,
        tl.where(owned, result, tl.zeros_like(result)),
    )


def gather_dequantize_engram_int8(
    weight: torch.Tensor,
    scales: torch.Tensor | None,
    ids: torch.Tensor,
    width: int,
    *,
    head_start: int = 0,
    local_heads: int = 1,
    pad_heads: int | None = None,
    output: torch.Tensor | None = None,
    vocab_start: int = 0,
    vocab_end: int | None = None,
) -> torch.Tensor:
    """Gather device rows, dequantizing INT8 when scales are supplied.

    Returns ``[tokens * pad_heads, width]``; the head path views it as
    ``[tokens, pad_heads, width]``.
    """

    # Importing the ops package initializes the active Triton backend, so keep
    # it out of CPU-only routing and test workers.
    from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton

    pad_heads = local_heads if pad_heads is None else pad_heads
    vocab_end = weight.shape[0] if vocab_end is None else vocab_end
    tokens = ids.shape[0]
    rows = tokens * local_heads
    if output is None:
        output = torch.empty((tokens * pad_heads, width), dtype=torch.bfloat16, device=weight.device)
    if rows == 0:
        return output
    init_device_properties_triton()
    _engram_int8_gather_dequant_kernel[(rows,)](
        weight,
        scales,
        ids,
        output,
        rows,
        vocab_start,
        vocab_end,
        ids.stride(0),
        WIDTH=width,
        GROUP=SCALE_GROUP,
        HEAD_START=head_start,
        LOCAL_HEADS=local_heads,
        PAD_HEADS=pad_heads,
        QUANTIZED=scales is not None,
        num_warps=4,
    )
    return output


@cache
def _host_library() -> ctypes.CDLL:
    """The CANN runtime entry points that publish host memory to the device."""

    lib = ctypes.CDLL("libascendcl.so")
    lib.aclrtMallocHost.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t, ctypes.c_uint32]
    lib.aclrtMallocHost.restype = ctypes.c_int
    lib.aclrtFreeHost.argtypes = [ctypes.c_void_p]
    lib.aclrtFreeHost.restype = ctypes.c_int
    lib.aclrtHostRegisterV2.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint32]
    lib.aclrtHostRegisterV2.restype = ctypes.c_int
    lib.aclrtHostGetDevicePointer.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint32]
    lib.aclrtHostGetDevicePointer.restype = ctypes.c_int
    lib.aclrtHostUnregister.argtypes = [ctypes.c_void_p]
    lib.aclrtHostUnregister.restype = ctypes.c_int
    return lib


class HostUvaBuffer:
    """Host memory the device gathers from directly."""

    def __init__(self, shape, dtype, device):
        self.lib = _host_library()
        rows = int(shape[0])
        row_elements = int(torch.Size(shape[1:]).numel())
        self.row_bytes = row_elements * torch.empty((), dtype=dtype).element_size()
        size = rows * self.row_bytes
        self.pointer = ctypes.c_void_p()
        rc = self.lib.aclrtMallocHost(ctypes.byref(self.pointer), size, 0)
        if rc:
            raise RuntimeError(f"aclrtMallocHost failed: rc={rc} size={size}")
        self.buffer = (ctypes.c_char * size).from_address(self.pointer.value)
        self.tensor = torch.frombuffer(self.buffer, dtype=dtype).reshape(shape)
        try:
            rc = self.lib.aclrtHostRegisterV2(self.pointer, size, ACL_HOST_REG_MAPPED | ACL_HOST_REG_PINNED)
            if rc:
                raise RuntimeError(f"aclrtHostRegisterV2 failed: rc={rc} size={size}")
            address = ctypes.c_void_p()
            rc = self.lib.aclrtHostGetDevicePointer(self.pointer, ctypes.byref(address), 0)
            if rc:
                raise RuntimeError(f"aclrtHostGetDevicePointer failed: rc={rc}")
            self.ptrs = torch.tensor(
                [address.value + start * self.row_bytes for start in range(0, rows, CHUNK_ROWS)],
                dtype=torch.int64,
                device=device,
            )
        except Exception:
            # A half-built buffer must not leave the host range registered: the
            # caller only sees the exception, so nothing else can release it.
            self.lib.aclrtHostUnregister(self.pointer)
            self.lib.aclrtFreeHost(self.pointer)
            self.tensor = None
            self.buffer = None
            self.pointer = ctypes.c_void_p()
            raise

    def close(self):
        """Unregister and release the host range; safe to call more than once.

        The NPU gathers out of this range through the device address the
        registration published, so the mapping must not go away while work that
        reads it is still in flight.
        """
        if self.pointer is None or not self.pointer.value:
            return
        if self.ptrs is not None and self.ptrs.device.type == "npu":
            torch.npu.synchronize()
        rc = self.lib.aclrtHostUnregister(self.pointer)
        if rc:
            raise RuntimeError(f"aclrtHostUnregister failed: rc={rc}")
        self.tensor = None
        self.buffer = None
        self.ptrs = None
        rc = self.lib.aclrtFreeHost(self.pointer)
        if rc:
            raise RuntimeError(f"aclrtFreeHost failed: rc={rc}")
        self.pointer = ctypes.c_void_p()


@triton.jit
def _engram_host_uva_gather_dequant_kernel(
    codes_ptrs,
    scales_ptrs,
    ids,
    output,
    rows,
    vocab_start,
    vocab_end,
    ids_stride_t,
    CHUNK: tl.constexpr,
    WIDTH: tl.constexpr,
    GROUP: tl.constexpr,
    HEAD_START: tl.constexpr,
    LOCAL_HEADS: tl.constexpr,
    PAD_HEADS: tl.constexpr,
    QUANTIZED: tl.constexpr,
):
    row = tl.program_id(0)
    if row < rows:
        token = row // LOCAL_HEADS
        head_local = row % LOCAL_HEADS
        index = tl.load(ids + token * ids_stride_t + HEAD_START + head_local).to(tl.int64)
        # The kernel owns the last line of defence: an id outside this shard
        # must not become an address the pointer table is indexed with, whoever
        # computed it.
        owned = (index >= vocab_start) & (index < vocab_end)
        local_row = tl.where(owned, index - vocab_start, 0)
        chunk = local_row // CHUNK
        local = local_row % CHUNK
        if QUANTIZED:
            codes = tl.load(codes_ptrs + chunk).to(tl.pointer_type(tl.int8))
        else:
            codes = tl.load(codes_ptrs + chunk).to(tl.pointer_type(tl.bfloat16))
        col = tl.arange(0, WIDTH)
        value = tl.load(codes + local * WIDTH + col).to(tl.float32)
        if QUANTIZED:
            scales = tl.load(scales_ptrs + chunk).to(tl.pointer_type(tl.float32))
            scale = tl.load(scales + local * (WIDTH // GROUP) + col // GROUP)
            value = value * scale
        result = value.to(tl.bfloat16)
        tl.store(
            output + (token * PAD_HEADS + head_local) * WIDTH + col,
            tl.where(owned, result, tl.zeros_like(result)),
        )


def gather_dequantize_host_uva(
    codes: HostUvaBuffer,
    scales: HostUvaBuffer | None,
    ids: torch.Tensor,
    *,
    head_start: int = 0,
    local_heads: int = 1,
    pad_heads: int | None = None,
    output: torch.Tensor | None = None,
    vocab_start: int = 0,
    vocab_end: int | None = None,
) -> torch.Tensor:
    """Gather registered host rows, preserving BF16 when scales are absent.

    Returns ``[tokens * pad_heads, width]``; the head path views it as
    ``[tokens, pad_heads, width]``.
    """

    # A partial close can unregister later chunks before an earlier chunk
    # fails. Retaining the backing for cleanup retry does not make those
    # device addresses usable again. Reject before native imports or access.
    if getattr(codes, "_closing", False) or getattr(scales, "_closing", False):
        raise RuntimeError("Engram host UVA lookup cannot use a closing or closed shared buffer")

    from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton

    pad_heads = local_heads if pad_heads is None else pad_heads
    width = codes.tensor.shape[-1]
    vocab_end = codes.tensor.shape[0] if vocab_end is None else vocab_end
    tokens = ids.shape[0]
    rows = tokens * local_heads
    if output is None:
        output = torch.empty((tokens * pad_heads, width), dtype=torch.bfloat16, device=ids.device)
    if rows == 0:
        return output
    init_device_properties_triton()
    _engram_host_uva_gather_dequant_kernel[(rows,)](
        codes.ptrs,
        scales.ptrs if scales is not None else None,
        ids.to(torch.int64),
        output,
        rows,
        vocab_start,
        vocab_end,
        ids.stride(0),
        CHUNK=CHUNK_ROWS,
        WIDTH=width,
        GROUP=SCALE_GROUP,
        HEAD_START=head_start,
        LOCAL_HEADS=local_heads,
        PAD_HEADS=pad_heads,
        QUANTIZED=scales is not None,
        num_warps=4,
    )
    return output


class EngramBufferInitializationError(RuntimeError):
    """Keep a partially initialized mapping owned when releasing it fails.

    The caller must retain ``buffer`` until an explicit ``close()`` succeeds.
    Otherwise SharedMemory's destructor could unmap a range still registered
    with CANN while the failed constructor is unwinding.
    """

    def __init__(self, buffer: "SharedUvaBuffer", message: str) -> None:
        super().__init__(message)
        self.buffer = buffer


class _MemfdSharedMemory:
    """One internal-shmem backing with validated peer attachment and THP advice.

    The creator keeps its FD open until every peer has attached and throughout
    its lifetime. Existing peer mappings own references to the same file. This
    uses ordinary MAP_SHARED mappings, without fixed addresses or host policy
    changes. The API intentionally matches SharedMemory's storage operations.
    """

    def __init__(self, size: int, name=None):
        if type(size) is not int or size <= 0:
            raise ValueError("Engram memfd size must be a positive integer")
        self.size = size
        self.name: tuple[str, int, int, int, int, int] | None = None
        self.buf = None
        self._mapping = None
        self._fd = None
        creator = name is None
        try:
            if creator:
                self._fd = os.memfd_create("vllm-engram-shared", os.MFD_CLOEXEC)
                os.ftruncate(self._fd, size)
                info = os.fstat(self._fd)
                self.name = ("memfd", os.getpid(), self._fd, info.st_dev, info.st_ino, size)
            else:
                if (
                    type(name) is not tuple
                    or len(name) != 6
                    or type(name[0]) is not str
                    or name[0] != "memfd"
                    or any(type(value) is not int for value in name[1:])
                    or name[1] <= 0
                    or name[2] < 0
                    or name[3] < 0
                    or name[4] <= 0
                    or name[5] != size
                ):
                    raise ValueError("Invalid Engram memfd descriptor or size")
                self.name = name
                _, pid, descriptor, device, inode, expected_size = name
                self._fd = os.open(f"/proc/{pid}/fd/{descriptor}", os.O_RDWR)
                info = os.fstat(self._fd)
                if (info.st_dev, info.st_ino, info.st_size) != (device, inode, expected_size):
                    raise ValueError("Engram memfd attachment identity changed")
            self._mapping = mmap.mmap(self._fd, size, flags=mmap.MAP_SHARED, prot=mmap.PROT_READ | mmap.PROT_WRITE)
            self.buf = memoryview(self._mapping)
            # A failed or unavailable advice is a collective initialization
            # failure, never an implicit switch to a different storage path.
            self._mapping.madvise(mmap.MADV_HUGEPAGE)
            if creator:
                page_size = int(os.sysconf("SC_PAGE_SIZE"))
                pages = self.buf[::page_size]
                try:
                    pages[:] = bytes(len(pages))
                finally:
                    pages.release()
            address = ctypes.addressof(ctypes.c_char.from_buffer(self.buf))
            logger.info(
                "Engram memfd shared backing mapped: size=%d, host_va_mod_2m=%d, creator=%s; "
                "MADV_HUGEPAGE accepted (actual huge-page coverage is not guaranteed)",
                size,
                address % (2 * 1024**2),
                creator,
            )
        except Exception as exc:
            try:
                self.close()
            except Exception as cleanup:
                raise EngramBufferInitializationError(
                    self, f"Engram memfd initialization failed: {exc}; release failed: {cleanup}"
                ) from exc
            raise

    def unlink(self) -> None:
        """memfd has no POSIX name to unlink; FD and mapping ownership suffice."""

    def close(self) -> None:
        if self.buf is not None:
            self.buf.release()
            self.buf = None
        if self._mapping is not None:
            try:
                self._mapping.close()
            except Exception:
                # Retain a usable mapping owner for another cleanup attempt.
                self.buf = memoryview(self._mapping)
                raise
            self._mapping = None
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None


class SharedUvaBuffer:
    """One host range mapped and registered by every rank of a local group.

    Large ranges use one memfd with huge-page advice and creator pre-touch before
    peer attachment. Small TP slices retain POSIX SharedMemory. Every rank maps
    the full owned range with CANN; large ranges are registered by lookup chunk,
    while small ranges keep one registration. Independent registrations need
    not have contiguous device addresses. Only POSIX backing has a name to unlink.
    """

    def __init__(self, shape, dtype, device, group):
        self.lib = _host_library()
        self.group = group
        rows = int(shape[0])
        row_elements = int(torch.Size(shape[1:]).numel())
        self.row_bytes = row_elements * torch.empty((), dtype=dtype).element_size()
        size = rows * self.row_bytes
        self.shm = None
        self.tensor = None
        self.ptrs = None
        self.pointer = ctypes.c_void_p()
        self._registered_host_ptrs: list[ctypes.c_void_p] = []
        self._closing = False

        cpu_group = group.cpu_group
        leader = dist.get_global_rank(cpu_group, 0)
        payload: list[str | tuple[str, int, int, int, int, int] | None] = [None]
        if group.rank_in_group == 0:
            try:
                self.shm = (
                    _MemfdSharedMemory(size)
                    if size > MAX_SINGLE_REGISTRATION_BYTES
                    else shared_memory.SharedMemory(create=True, size=size)
                )
                payload = [self.shm.name]
            except Exception as exc:  # noqa: BLE001 - reported to the group
                if isinstance(exc, EngramBufferInitializationError):
                    self.shm = exc.buffer
                cleanup_error = None
                if self.shm is not None:
                    try:
                        self.shm.unlink()
                        self.shm.close()
                        self.shm = None
                    except Exception as cleanup:
                        cleanup_error = f"; release failed: {cleanup}"
                payload = [f"ERROR: {type(exc).__name__}: {exc}{cleanup_error or ''}"]
        try:
            dist.broadcast_object_list(payload, src=leader, group=cpu_group)
        except Exception as exc:
            self._release_after_collective_failure(f"Engram shared backing broadcast failed: {exc}")
        name = payload[0]
        if name is None or (isinstance(name, str) and name.startswith("ERROR:")):
            if self.shm is not None:
                raise EngramBufferInitializationError(self, f"Engram shared backing creation failed: {name}")
            raise RuntimeError(f"Engram shared backing creation failed: {name}")

        error = None
        try:
            if self.shm is None:
                if size > MAX_SINGLE_REGISTRATION_BYTES:
                    self.shm = _MemfdSharedMemory(size, name)
                else:
                    # Python 3.12 tracks attachments as owners. Match vLLM's
                    # SharedMemory attach path so only the creator unlinks it.
                    with patch("multiprocessing.resource_tracker.register", lambda *args, **kwargs: None):
                        self.shm = shared_memory.SharedMemory(name=name)
            assert self.shm.size >= size
            address_of_mapping = ctypes.c_void_p(ctypes.addressof(ctypes.c_char.from_buffer(self.shm.buf)))
            registration_rows = CHUNK_ROWS if size > MAX_SINGLE_REGISTRATION_BYTES else rows
            device_ptrs = []
            for start in range(0, rows, registration_rows):
                stop = min(start + registration_rows, rows)
                offset = start * self.row_bytes
                registration_size = (stop - start) * self.row_bytes
                host_pointer = ctypes.c_void_p(address_of_mapping.value + offset)
                rc = self.lib.aclrtHostRegisterV2(
                    host_pointer, registration_size, ACL_HOST_REG_MAPPED | ACL_HOST_REG_PINNED
                )
                if rc:
                    raise RuntimeError(
                        f"aclrtHostRegisterV2 failed: rc={rc} size={registration_size} "
                        f"chunk={start // CHUNK_ROWS} offset={offset}"
                    )
                # Record ownership before obtaining the device pointer: a failed
                # pointer query must still unregister this successfully pinned
                # range. Keep the original pointer field for small-buffer callers.
                self._registered_host_ptrs.append(host_pointer)
                self.pointer = self._registered_host_ptrs[0]
                address = ctypes.c_void_p()
                rc = self.lib.aclrtHostGetDevicePointer(host_pointer, ctypes.byref(address), 0)
                if rc:
                    raise RuntimeError(
                        f"aclrtHostGetDevicePointer failed: rc={rc} "
                        f"chunk={start // CHUNK_ROWS} offset={offset} size={registration_size}"
                    )
                device_ptrs.extend(
                    address.value + (chunk_start - start) * self.row_bytes
                    for chunk_start in range(start, stop, CHUNK_ROWS)
                )
            self.tensor = torch.frombuffer(self.shm.buf, dtype=dtype, count=rows * row_elements).reshape(shape)
            self.ptrs = torch.tensor(device_ptrs, dtype=torch.int64, device=device)
        except Exception as exc:  # noqa: BLE001 - aggregated below
            if isinstance(exc, EngramBufferInitializationError):
                self.shm = exc.buffer
            error = f"{type(exc).__name__}: {exc}"

        errors: list[str | None] = [None] * group.world_size
        try:
            dist.all_gather_object(errors, error, group=cpu_group)
            # Fence every mapping (or every failure) before the leader unlinks.
            dist.barrier(group=cpu_group)
        except Exception as exc:
            self._release_after_collective_failure(f"Engram shared-memory initialization collective failed: {exc}")
        failures = "; ".join(f"rank {rank}: {failure}" for rank, failure in enumerate(errors) if failure is not None)
        if failures:
            if group.rank_in_group == 0:
                # Drop the name before close() drops the SharedMemory owner;
                # existing mappings remain valid until their explicit close.
                self._unlink()
            try:
                self.close()
            except Exception as exc:  # a registered mapping must stay owned
                raise EngramBufferInitializationError(
                    self,
                    f"Engram shared-memory initialization failed: {failures}; release failed: {exc}",
                ) from exc
            raise RuntimeError(f"Engram shared-memory initialization failed: {failures}")
        if group.rank_in_group == 0:
            self._unlink()

    def _release_after_collective_failure(self, message: str) -> None:
        if self.group.rank_in_group == 0:
            self._unlink()
        try:
            self.close()
        except Exception as exc:
            raise EngramBufferInitializationError(self, f"{message}; release failed: {exc}") from exc
        raise RuntimeError(message)

    def _unlink(self) -> None:
        try:
            shm = self.shm
            if shm is not None:
                shm.unlink()
        except OSError as exc:
            logger.warning("Engram shared backing unlink failed: %s", exc)

    def close(self) -> None:
        """Unregister and close the mapping; shared memory is never aclrtFreeHost.

        Safe to call more than once. Registrations are undone in reverse order;
        successful unregisters are removed immediately and never repeated. A
        failed unregister retains the remaining ranges and shared backing for a
        retry. The mapping cannot be closed while any range remains registered.
        Once cleanup starts the buffer is no longer available for lookup, even
        if a later unregister fails after earlier chunks have been released.
        """
        self._closing = True
        if self._registered_host_ptrs:
            if self.ptrs is not None and self.ptrs.device.type == "npu":
                torch.npu.synchronize()
            while self._registered_host_ptrs:
                host_pointer = self._registered_host_ptrs[-1]
                rc = self.lib.aclrtHostUnregister(host_pointer)
                if rc:
                    raise RuntimeError(
                        f"aclrtHostUnregister failed for shared Engram: rc={rc} "
                        f"pointer={host_pointer.value:#x} remaining={len(self._registered_host_ptrs)}"
                    )
                self._registered_host_ptrs.pop()
        self.pointer = ctypes.c_void_p()
        self.tensor = None
        self.ptrs = None
        if self.shm is not None:
            self.shm.close()
            self.shm = None
