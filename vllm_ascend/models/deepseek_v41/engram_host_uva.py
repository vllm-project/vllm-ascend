# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Host-resident Engram shards registered for device reads (UVA).

The INT8 rows stay in host memory.  ``aclrtHostRegisterV2`` pins them and
``aclrtHostGetDevicePointer`` publishes the device address the NPU gather
kernel uses, so an offloaded shard is read where the rows already are: no H2D
copy, no host gather, and the routing around the shard is unchanged.

The address table is chunked because a 384M row table overflows the 32 bit
offset arithmetic a single Triton tile can express.
"""

import ctypes

import torch
from vllm.triton_utils import tl, triton

ACL_HOST_REG_MAPPED = 0x2
ACL_HOST_REG_PINNED = 0x10000000
CHUNK_ROWS = 1 << 22


def eng_cpu_offload(vllm_config) -> bool:
    """Whether the Engram table is offloaded to host memory (UVA lookup)."""

    return bool(vllm_config.engram_config.cpu_offload)


def engram_enabled(text_config) -> bool:
    """Whether the checkpoint declares Engram n-gram layers."""

    return bool(getattr(text_config, "engram_layer_ids", None))


def _host_library() -> ctypes.CDLL:
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
        size = rows * row_elements * torch.empty((), dtype=dtype).element_size()
        self.pointer = ctypes.c_void_p()
        rc = self.lib.aclrtMallocHost(ctypes.byref(self.pointer), size, 0)
        if rc:
            raise RuntimeError(f"aclrtMallocHost failed: rc={rc} size={size}")
        self.buffer = (ctypes.c_char * size).from_address(self.pointer.value)
        self.tensor = torch.frombuffer(self.buffer, dtype=dtype).reshape(shape)
        rc = self.lib.aclrtHostRegisterV2(self.pointer, size, ACL_HOST_REG_MAPPED | ACL_HOST_REG_PINNED)
        if rc:
            raise RuntimeError(f"aclrtHostRegisterV2 failed: rc={rc} size={size}")
        address = ctypes.c_void_p()
        rc = self.lib.aclrtHostGetDevicePointer(self.pointer, ctypes.byref(address), 0)
        if rc:
            raise RuntimeError(f"aclrtHostGetDevicePointer failed: rc={rc}")
        row_bytes = row_elements * self.tensor.element_size()
        self.ptrs = torch.tensor(
            [address.value + start * row_bytes for start in range(0, rows, CHUNK_ROWS)],
            dtype=torch.int64,
            device=device,
        )

    def close(self):
        self.lib.aclrtHostUnregister(self.pointer)
        self.tensor = None
        self.buffer = None
        self.lib.aclrtFreeHost(self.pointer)
        self.pointer = None


@triton.jit
def _engram_host_uva_kernel(
    codes_ptrs,
    scales_ptrs,
    ids,
    output,
    rows,
    CHUNK: tl.constexpr,
    WIDTH: tl.constexpr,
):
    row = tl.program_id(0)
    if row < rows:
        index = tl.load(ids + row).to(tl.int64)
        chunk = index // CHUNK
        local = index % CHUNK
        codes = tl.load(codes_ptrs + chunk).to(tl.pointer_type(tl.int8))
        scales = tl.load(scales_ptrs + chunk).to(tl.pointer_type(tl.float32))
        col = tl.arange(0, WIDTH)
        value = tl.load(codes + local * WIDTH + col).to(tl.float32)
        scale = tl.load(scales + local * (WIDTH // 32) + col // 32)
        tl.store(output + row * WIDTH + col, (value * scale).to(tl.bfloat16))


def gather_dequantize_host_uva(codes: HostUvaBuffer, scales: HostUvaBuffer, ids: torch.Tensor) -> torch.Tensor:
    """Gather rows ``ids`` from a registered host table and dequantize on device."""

    rows = ids.numel()
    output = torch.empty((rows, 256), dtype=torch.bfloat16, device=ids.device)
    if rows == 0:
        return output
    _engram_host_uva_kernel[(rows,)](
        codes.ptrs,
        scales.ptrs,
        ids.reshape(-1).to(torch.int64),
        output,
        rows,
        CHUNK=CHUNK_ROWS,
        WIDTH=256,
        num_warps=4,
    )
    return output
