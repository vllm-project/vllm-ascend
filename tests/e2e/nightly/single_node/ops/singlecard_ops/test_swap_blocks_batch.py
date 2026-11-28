# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""swap-blocks-batch single-op accuracy tests on the A2/A3 arch22 kernel."""

import ctypes
from functools import partial

import pytest
import torch
import torch_npu  # noqa: F401

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


@pytest.mark.parametrize("count", [1, 2047])
def test_swap_blocks_batch_dma_to_mapped_host(count, mapped_host_allocator):
    """The existing D2H ABI writes the same allocation that AIV reads."""
    if not hasattr(torch.ops._C_ascend, "swap_blocks_batch"):
        import vllm_ascend.vllm_ascend_C  # type: ignore[import-not-found, import-untyped]  # noqa: F401
    host = mapped_host_allocator((count, 576), torch.bfloat16)
    source = (torch.arange(count, device="npu") % 97).to(torch.bfloat16)[:, None].expand(count, 576).contiguous()
    row_bytes = 576 * source.element_size()
    src = torch.arange(count, dtype=torch.int64) * row_bytes + source.data_ptr()
    dst = torch.arange(count, dtype=torch.int64) * row_bytes + host.data_ptr()
    sizes = torch.full((count,), row_bytes, dtype=torch.int64)
    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    try:
        with torch.npu.stream(stream):
            torch.ops._C_ascend.swap_blocks_batch(src, dst, sizes, 1)
        stream.synchronize()
        torch.testing.assert_close(host, source.cpu())
    finally:
        torch.npu.synchronize()
        host._hisparse_registration.close()
