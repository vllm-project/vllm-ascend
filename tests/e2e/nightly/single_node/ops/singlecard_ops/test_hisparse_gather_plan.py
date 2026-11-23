# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HiSparse gather-plan single-op accuracy tests on the A2/A3 arch22 kernel."""

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


def test_hisparse_gather_plan_respects_request_state_padding(mapped_host_allocator):
    host = mapped_host_allocator((32, 16), torch.float32)
    mapping = host._hisparse_registration
    host.copy_(torch.arange(host.numel(), dtype=torch.float32).view_as(host))
    hot = torch.full((8, 4, 16), -17.0, device="npu")
    ops = HiSparseNPUOps(mapping)
    source = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8], [8, 7, 6, 5, 4, 3, 2, 1]], device="npu", dtype=torch.int32)
    destination = source + 4
    mask = torch.ones_like(source, dtype=torch.int32)
    state = torch.tensor([0, -1], device="npu", dtype=torch.int32)
    attention = torch.empty_like(source)
    try:
        ops.hisparse_gather_plan(host, hot, source, destination, mask, state, attention, 8)
        torch.npu.synchronize()
        cache = hot.cpu().view(32, 16)
        torch.testing.assert_close(cache[5:13], host[1:9])
        torch.testing.assert_close(attention[1].cpu(), torch.full((8,), -1, dtype=torch.int32))
    finally:
        torch.npu.synchronize()
        mapping.close()
