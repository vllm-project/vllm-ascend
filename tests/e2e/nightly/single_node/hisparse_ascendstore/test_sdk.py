# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Opt-in registration test for MemCache SDK 1.3.0 and a dedicated host_tcp pool.

Start a dedicated MemCache metadata service and set MMC_LOCAL_CONFIG_PATH as
for the serving instance. Pass --hisparse-sdk to run on NPU 0.
"""

from importlib.metadata import version

import pytest
import torch
import torch_npu  # noqa: F401
from vllm.config import ParallelConfig

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.memcache_backend import (
    MemcacheBackend,
    require_hisparse_host_tcp,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.buffer import (
    DEVICE_BUFFER_ALIGNMENT,
    BufferDescriptor,
    MemoryLocation,
)


def test_real_memcache_registration_rollback_and_retry(pytestconfig):
    if not pytestconfig.getoption("--hisparse-sdk"):
        pytest.skip("Pass --hisparse-sdk with a dedicated MemCache host_tcp service")
    assert version("memcache_hybrid") == "1.3.0"
    assert torch.npu.is_available()
    require_hisparse_host_tcp()
    torch.npu.set_device(0)
    size = DEVICE_BUFFER_ALIGNMENT
    host_tensor = torch.empty(size, dtype=torch.uint8, pin_memory=True)
    host = BufferDescriptor(address=host_tensor.data_ptr(), size_bytes=size, memory_location=MemoryLocation.HOST)
    allocations = [torch.empty(3 * size, dtype=torch.uint8, device="npu:0") for _ in range(2)]
    candidate, conflict = [
        BufferDescriptor(address=t.data_ptr() + (-t.data_ptr() % size), size_bytes=size, alignment_bytes=size)
        for t in allocations
    ]
    backend = MemcacheBackend(ParallelConfig(), device_id=0)
    try:
        sdk = backend.store
        assert sdk is not None
        backend.register_buffer([host])
        # A pre-existing SDK registration with different geometry forces the
        # final entry to fail after the candidate has actually been registered.
        assert sdk.register_buffer(conflict.address, 2 * size) == 0
        with pytest.raises(RuntimeError, match="MemCache buffer registration failed"):
            backend.register_buffer([host, candidate, conflict])
        # The earlier Host registration survives. Re-registering the candidate
        # with a different size succeeds only if the failed batch rolled it back.
        assert sdk.register_buffer(host.address, size // 2) == -1
        assert sdk.unregister_buffer(conflict.address, 2 * size) == 0
        resized = BufferDescriptor(address=candidate.address, size_bytes=2 * size, alignment_bytes=size)
        backend.register_buffer([host, resized, conflict])
        assert sdk.register_buffer(candidate.address, 2 * size) == 0
        backend.unregister_buffer([host.address, resized.address, conflict.address], [size, 2 * size, size])
    finally:
        backend.close()
