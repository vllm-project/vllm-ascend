# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HiSparse gather-compact single-op accuracy tests on the A2/A3 arch22 kernel."""

import ctypes
from functools import partial

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.hisparse.cache_ops import HiSparseNPUOps
from vllm_ascend.hisparse.host_pool import _ACL_HAC_AIV, allocate_mapped_host
from vllm_ascend.utils import AscendDeviceType, get_ascend_device_type

_ACL_ERROR_RT_FEATURE_NOT_SUPPORT = 207000


@pytest.fixture(autouse=True)
def _require_arch22():
    if not torch.npu.is_available():
        pytest.skip("requires an Ascend NPU")
    if get_ascend_device_type() not in (AscendDeviceType.A2, AscendDeviceType.A3):
        pytest.skip("requires the arch22 kernel")
    if not hasattr(torch.ops._C_ascend, "hisparse_resolve_residency"):
        import vllm_ascend.vllm_ascend_C  # type: ignore[import-not-found, import-untyped]  # noqa: F401
    assert hasattr(torch.ops._C_ascend, "hisparse_resolve_residency"), "rebuild the arch22 extension"


@pytest.fixture
def mapped_host_allocator(_require_arch22):
    """Prefer native mapping; require working MemFabric when it is unsupported."""
    device = torch.npu.current_device()
    acl = ctypes.CDLL("libascendcl.so")
    query = acl.aclrtHostMemMapCapabilities
    query.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
    query.restype = ctypes.c_int
    capability = ctypes.c_int()
    status = query(device, _ACL_HAC_AIV, ctypes.byref(capability))
    if status == _ACL_ERROR_RT_FEATURE_NOT_SUPPORT or (status == 0 and capability.value == 0):
        # Select the existing backend explicitly for these tests. Allocation
        # failures must propagate, including a missing/unusable MemFabric SDK.
        return partial(allocate_mapped_host, memfabric=True)
    if status != 0:
        raise RuntimeError(f"AIV Host mapping capability query failed unexpectedly: status={status}")
    return allocate_mapped_host


@pytest.mark.parametrize("rows,topk,row_bytes", [(1, 17, 656), (4, 2048, 656), (16, 2048, 1152)])
@pytest.mark.parametrize("capture", [False, True])
def test_hisparse_gather_compact_refill_tiles_current_counts_and_preserves_page_gaps(
    rows, topk, row_bytes, capture, mapped_host_allocator
):
    block = 128
    blocks = (rows * topk + block - 1) // block + 1
    host = mapped_host_allocator((blocks * block, row_bytes), torch.uint8)
    mapping = host._hisparse_registration
    host.copy_((torch.arange(host.shape[0], dtype=torch.int32) % 251).to(torch.uint8)[:, None].expand_as(host))
    backing = torch.full((blocks, 2, block, row_bytes), 203, dtype=torch.uint8, device="npu")
    hot = backing[:, 1]
    ops = HiSparseNPUOps(mapping)
    original_source = torch.arange(rows * topk, dtype=torch.int32).view(rows, topk)
    original_destination = original_source + block
    original_source[:, 4] = -1
    original_source[:, 6] = host.shape[0] + 7
    original_destination[:, 2] = -1
    original_destination[:, 5] = blocks * block + 3
    source, destination = original_source.to("npu"), original_destination.to("npu")
    counts = torch.zeros(rows, dtype=torch.int32, device="npu")

    def forward():
        ops.hisparse_gather_compact(host, hot, source, destination, counts)

    graph = None
    try:
        forward()
        torch.npu.synchronize()
        if capture:
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph):
                forward()
        for count in (0, 1, 7, 8, 9, 15, 16, 17, topk, 0):
            backing.fill_(203)
            current = torch.full((rows,), count, dtype=torch.int32)
            if rows > 1:
                current[1] = -3
            if rows > 2 and count == topk:
                current[2] = topk + 3
            counts.copy_(current)
            source.copy_(original_source if count else torch.full_like(original_source, -1))
            destination.copy_(original_destination if count else torch.full_like(original_destination, -1))
            if graph is None:
                forward()
            else:
                graph.replay()
            torch.npu.synchronize()
            expected = torch.full(backing.shape, 203, dtype=torch.uint8)
            for row in range(rows):
                length = min(topk, max(0, int(current[row])))
                src = original_source[row, :length].long()
                dst = original_destination[row, :length].long()
                valid = (src >= 0) & (dst >= 0) & (dst < blocks * block)
                payload = torch.zeros((length, row_bytes), dtype=torch.uint8)
                from_host = valid & (src < host.shape[0])
                payload[from_host] = host[src[from_host]]
                expected[dst[valid] // block, 1, dst[valid] % block] = payload[valid]
            torch.testing.assert_close(backing.cpu(), expected, rtol=0, atol=0)
    finally:
        torch.npu.synchronize()
        if graph is not None:
            graph.reset()
        mapping.close()
