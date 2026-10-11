# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NPU-side Engram storage, routing and lookup.

Tables are BF16, INT8/FP32 or native MXFP8/E8M0, sharded into contiguous hash
head buckets.  With ``EngramConfig.cpu_offload`` the shard stays in host memory:
``aclrtHostRegisterV2`` pins it and ``aclrtHostGetDevicePointer`` publishes the
address the NPU gather kernel reads, so an offloaded table needs neither an H2D
copy nor a host-side gather.
"""

import ctypes
from functools import cache
from multiprocessing import shared_memory
from unittest.mock import patch

import torch
import torch.distributed as dist
from vllm.logger import logger

from vllm_ascend.device.device_op import DeviceOperator

SCALE_GROUP = 32
# A 384M row table overflows the 32 bit offset arithmetic a single Triton tile
# can express, so the device address of every group of rows is published
# separately.
CHUNK_ROWS = 1 << 22


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
    """Gather device rows, dequantizing INT8 or MXFP8 when scales are supplied.

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
    from vllm_ascend.ops.triton.engram_lookup import _engram_int8_gather_dequant_kernel

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
        MXFP8=weight.dtype == torch.float8_e4m3fn,
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
            rc = self.lib.aclrtHostRegisterV2(self.pointer, size, DeviceOperator.host_register_flags())
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
    # The kernel widens loaded IDs to int64; no extra cast/copy is needed here.
    from vllm_ascend.ops.triton.engram_lookup import _engram_host_uva_gather_dequant_kernel

    _engram_host_uva_gather_dequant_kernel[(rows,)](
        codes.ptrs,
        scales.ptrs if scales is not None else None,
        ids,
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
        MXFP8=codes.tensor.dtype == torch.float8_e4m3fn,
        num_warps=4,
    )
    return output


class SharedUvaBuffer:
    """One host range mapped and registered by every rank of a local group.

    The leader creates the SharedMemory segment; every rank registers its own
    mapping with CANN. The leader unlinks it after all ranks have attached.
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

        cpu_group = group.cpu_group
        leader = dist.get_global_rank(cpu_group, 0)
        payload: list[str | None] = [None]
        if group.rank_in_group == 0:
            try:
                self.shm = shared_memory.SharedMemory(create=True, size=size)
                payload = [self.shm.name]
            except Exception as exc:  # noqa: BLE001 - reported to the group
                if self.shm is not None:
                    self.shm.unlink()
                    self.shm.close()
                payload = [f"ERROR: {type(exc).__name__}: {exc}"]
        dist.broadcast_object_list(payload, src=leader, group=cpu_group)
        name = payload[0]
        if name is None or name.startswith("ERROR:"):
            raise RuntimeError(f"Engram shared backing creation failed: {name}")

        error = None
        try:
            if self.shm is None:
                # Python 3.12 tracks attachments as owners. Match vLLM's
                # SharedMemory attach path so only the creator unlinks it.
                with patch("multiprocessing.resource_tracker.register", lambda *args, **kwargs: None):
                    self.shm = shared_memory.SharedMemory(name=name)
            assert self.shm.size >= size
            address_of_mapping = ctypes.c_void_p(ctypes.addressof(ctypes.c_char.from_buffer(self.shm.buf)))
            rc = self.lib.aclrtHostRegisterV2(address_of_mapping, size, DeviceOperator.host_register_flags())
            if rc:
                raise RuntimeError(f"aclrtHostRegisterV2 failed: rc={rc} size={size}")
            # Registration succeeded: from here on the mapping owes exactly one
            # aclrtHostUnregister, and close() must keep the owner until it
            # returns success.  ``pointer`` is therefore set only
            # after the registration it has to undo exists.
            self.pointer = address_of_mapping
            address = ctypes.c_void_p()
            rc = self.lib.aclrtHostGetDevicePointer(self.pointer, ctypes.byref(address), 0)
            if rc:
                raise RuntimeError(f"aclrtHostGetDevicePointer failed: rc={rc}")
            self.tensor = torch.frombuffer(self.shm.buf, dtype=dtype, count=rows * row_elements).reshape(shape)
            self.ptrs = torch.tensor(
                [address.value + start * self.row_bytes for start in range(0, rows, CHUNK_ROWS)],
                dtype=torch.int64,
                device=device,
            )
        except Exception as exc:  # noqa: BLE001 - aggregated below
            error = f"{type(exc).__name__}: {exc}"

        errors: list[str | None] = [None] * group.world_size
        dist.all_gather_object(errors, error, group=cpu_group)
        failures = "; ".join(f"rank {rank}: {failure}" for rank, failure in enumerate(errors) if failure is not None)
        # Fence every mapping (or every failure) before the leader unlinks.
        dist.barrier(group=cpu_group)
        if failures:
            try:
                self.close()
            except RuntimeError as exc:  # a rank that did register still owes it
                failures = f"{failures}; release failed: {exc}"
            if group.rank_in_group == 0:
                self._unlink()
            raise RuntimeError(f"Engram shared-memory initialization failed: {failures}")
        if group.rank_in_group == 0:
            self._unlink()

    def _unlink(self) -> None:
        try:
            shm = self.shm
            if shm is not None:
                shm.unlink()
        except OSError as exc:
            logger.warning("Engram shared backing unlink failed: %s", exc)

    def close(self) -> None:
        """Unregister and close the mapping; shared memory is never aclrtFreeHost.

        Safe to call more than once.  A failed unregister raises and keeps the
        pointer, the CPU views and the device address table, so the caller can
        retry: the backing stays owned until CANN has really let go of it, the
        same contract as the private ``HostUvaBuffer``.  The
        mapping is only closed after the unregister succeeded -- or when
        there is nothing registered to begin with, e.g. after a failed
        ``aclrtHostRegisterV2``.
        """
        if self.pointer is not None and self.pointer.value:
            if self.ptrs is not None and self.ptrs.device.type == "npu":
                torch.npu.synchronize()
            rc = self.lib.aclrtHostUnregister(self.pointer)
            if rc:
                raise RuntimeError(f"aclrtHostUnregister failed for shared Engram: rc={rc}")
        self.pointer = ctypes.c_void_p()
        self.tensor = None
        self.ptrs = None
        if self.shm is not None:
            self.shm.close()
            self.shm = None


class EngramUrmaCubeLookup:
    """Experimental 950DT AICPU UDMA gather for direct MXFP8 Cube WKV.

    Each lookup owns its host registration and device queue. Calls must be
    serialized on the Engram stream, including captured graph replays. Close
    after discarding graphs and before closing the source host mapping.
    """

    PAGE_BYTES = 4096
    LAUNCHER_CLASS = "EngramUrmaAicpu"

    def __init__(self):
        from pathlib import Path

        import vllm_ascend.vllm_ascend_C  # noqa: F401

        self.device_index = torch.npu.current_device()
        directory = Path(__file__).resolve().parents[3] / "engram_aicpu"
        launcher_class = getattr(torch.classes._C_ascend, self.LAUNCHER_CLASS)
        self.gather_launcher = launcher_class(str(directory / "libengram_aicpu.json"))
        self.gather_function = self.gather_launcher.function_handle()
        self.host_library = ctypes.CDLL(str(directory / "libengram_urma_host.so"))
        self.host_library.EngramReadRegister.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.c_uint,
        ]
        self.host_library.EngramReadRegister.restype = ctypes.c_void_p
        self.host_library.EngramReadRelease.argtypes = [ctypes.c_void_p]
        self.host_library.EngramReadRelease.restype = None
        self.host_state = None
        self.source = None
        self.metadata = None
        self.state_slot = torch.zeros(1, dtype=torch.int64, device=f"npu:{self.device_index}")
        self.last_call = None
        self.staging_buffers = {}
        # Spread host registrations across the two physical host chips.
        # AICPU namespaces name their local chip c0. The kernel selects between
        # its d0e3/d1e3 endpoints on the first successful transfer.
        self.host_chip = self.device_index // 4
        self.chip = self.die = 0

    def _bind_source(self, codes):
        if self.source is not None:
            if self.source is not codes:
                raise ValueError("A URMA lookup instance must retain the same source")
            return
        if torch.npu.is_current_stream_capturing():
            raise RuntimeError("Warm up URMA registration before graph capture")
        metadata = ctypes.create_string_buffer(128)
        status = (ctypes.c_uint32 * 16)()
        state = self.host_library.EngramReadRegister(
            codes.tensor.data_ptr(),
            codes.tensor.numel() * codes.tensor.element_size(),
            ctypes.addressof(metadata),
            status,
            self.host_chip,
        )
        if not state:
            raise RuntimeError(f"Engram URMA host registration failed: {list(status)}")
        try:
            self.metadata = torch.frombuffer(metadata, dtype=torch.uint8).clone().to(f"npu:{self.device_index}")
        except Exception:
            self.host_library.EngramReadRelease(state)
            raise
        self.host_state = state
        self.source = codes

    def _stage_codes(self, codes, ids, rows, width, head_start, local_heads, vocab_start, vocab_end):
        self._bind_source(codes)
        key = (rows, width)
        staged = self.staging_buffers.get(key)
        if staged is None:
            storage = torch.empty(rows * width + 2 * self.PAGE_BYTES, dtype=torch.uint8, device=ids.device)
            offset = (-storage.data_ptr()) % self.PAGE_BYTES
            staged = storage[offset : offset + rows * width].view(rows, width)
            self.staging_buffers[key] = staged
        torch.ops._C_ascend.engram_urma_gather(
            self.gather_function,
            self.metadata,
            self.state_slot,
            ids,
            staged,
            head_start,
            local_heads,
            vocab_start,
            vocab_end,
            self.chip,
            self.die,
        )
        self.last_call = (ids, staged, head_start, local_heads, vocab_start, vocab_end)
        return staged

    def close(self):
        if self.host_state is None:
            return
        with torch.npu.device(self.device_index):
            torch.npu.synchronize()
            if self.last_call is not None:
                self._close_device()
                torch.npu.synchronize()
                if self.state_slot.cpu().item() != 0:
                    raise RuntimeError("Engram URMA device resources did not close")
            self.host_library.EngramReadRelease(self.host_state)
            self.host_state = None
            self.source = self.metadata = self.last_call = None
            self.staging_buffers.clear()

    def _close_device(self):
        ids, staged, head_start, local_heads, vocab_start, vocab_end = self.last_call
        torch.ops._C_ascend.engram_urma_gather(
            self.gather_function,
            self.metadata,
            self.state_slot,
            ids,
            staged,
            head_start,
            local_heads,
            vocab_start,
            vocab_end,
            self.chip,
            self.die,
            True,
        )

    def lookup_codes(
        self,
        codes,
        scales,
        ids,
        *,
        head_start=0,
        local_heads=1,
        vocab_start=0,
        vocab_end=None,
        output_codes,
        output_scales,
    ):
        if codes.tensor.dtype != torch.float8_e4m3fn:
            raise ValueError("Direct Cube Engram lookup requires E4M3 codes")
        if scales.device.type != "npu" or scales.dtype != torch.uint8:
            raise ValueError("Direct Cube Engram lookup requires HBM uint8 E8M0 scales")
        if output_codes.dtype != torch.uint8 or output_scales.dtype != torch.uint8:
            raise ValueError("Direct Cube output buffers must retain raw FP8/E8M0 bytes")
        if ids.ndim != 2 or ids.dtype not in (torch.int32, torch.int64) or ids.stride(1) != 1:
            raise ValueError("Cube IDs require int32/int64 with contiguous columns")
        if head_start < 0 or local_heads <= 0 or head_start + local_heads > ids.shape[1]:
            raise ValueError("Invalid Engram head range")
        if ids.device.index != self.device_index or scales.device != ids.device:
            raise ValueError("Cube launcher and tensors must use the same NPU")
        width = codes.tensor.shape[-1]
        vocab_end = codes.tensor.shape[0] if vocab_end is None else vocab_end
        if vocab_start > vocab_end or vocab_end - vocab_start > codes.tensor.shape[0]:
            raise ValueError("Invalid Cube vocabulary range")
        if scales.shape != (codes.tensor.shape[0], width // SCALE_GROUP) or not scales.is_contiguous():
            raise ValueError("Invalid Cube scale shape")
        if (
            output_codes.shape != (ids.shape[0] * local_heads, width)
            or output_scales.shape != (ids.shape[0], local_heads * (width // SCALE_GROUP))
            or not output_codes.is_contiguous()
            or not output_scales.is_contiguous()
            or output_codes.device != ids.device
            or output_scales.device != ids.device
        ):
            raise ValueError("Invalid Cube output shape or device")
        tokens = ids.shape[0]
        rows = tokens * local_heads
        if not rows:
            return output_codes, output_scales
        staged = self._stage_codes(
            codes, ids, rows, codes.tensor.shape[-1], head_start, local_heads, vocab_start, vocab_end
        )
        output_codes[:rows].copy_(staged)
        selected = ids[:, head_start : head_start + local_heads].reshape(-1).to(torch.int64)
        owned = (selected >= vocab_start) & (selected < vocab_end)
        local = torch.where(owned, selected - vocab_start, torch.zeros_like(selected))
        gathered = scales.index_select(0, local)
        gathered.view(torch.int8).masked_fill_(~owned[:, None], 0)
        output_scales[:tokens].copy_(gathered.view(tokens, -1))
        return output_codes, output_scales


def compressed_engram_views(buffer, heads, width):
    """Fixed-address contiguous codes/scales stored in two packed planes."""
    capacity = buffer.shape[0]
    code_width = heads * width
    flat = buffer.view(-1)
    return (
        flat[: capacity * code_width].view(capacity, code_width),
        flat[capacity * code_width :].view(capacity, code_width // SCALE_GROUP),
    )
